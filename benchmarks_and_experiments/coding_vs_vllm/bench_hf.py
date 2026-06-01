"""HF-dataset reuse benchmark: kvboost CacheBlend vs vLLM prefix caching.

Small sample (default 10) of RAG-style prompts where passages recur across
requests in shuffled order (see hf_workload.py). Replayed **sequentially**
per backend so cross-request KV reuse accumulates — the whole point of the
comparison.

What it shows:
  * vLLM prefix caching reuses only an exact shared leading prefix, so when
    the same passage appears at a different position its KV is recomputed.
  * kvboost CacheBlend reuses each cached chunk wherever it lands, repairing
    only the seams — so recurring passages stay cheap regardless of order.

The observable effect is TTFT: on later samples that re-encounter passages,
CacheBlend should keep TTFT lower/flatter than prefix caching. We also pull
each backend's native cache telemetry (vLLM: usage.cached_tokens; kvboost:
/v1/stats kv reuse) for a direct hit-rate read where available.

Usage
-----
    python bench_hf.py \\
        --kvboost-url http://localhost:9000 --kvboost-model Qwen/Qwen2.5-3B-Instruct \\
        --vllm-url    http://localhost:8001 --vllm-model    Qwen/Qwen2.5-3B-Instruct \\
        --dataset squad --n 10
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from typing import Dict, List

import httpx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import hf_workload as hw
from bench_coding import Outcome, run_prompt, COMPLETED, _wait_healthy


async def replay_sequential(
    base_url: str, model: str, backend: str,
    samples: List["hw.RagSample"], timeout_s: float,
) -> List[Outcome]:
    """One request at a time so request N can reuse request <N's cache."""
    results: List[Outcome] = []
    async with httpx.AsyncClient() as client:
        for s in samples:
            r = await run_prompt(client, base_url, model, backend, s, timeout_s)
            results.append(r)
            hit = (f"cached={r.cached_tokens}/{r.prompt_tokens}"
                   if r.cached_tokens else "")
            tag = "ok " if r.status == COMPLETED else r.status
            print(f"  [{backend}] {tag:<10} {s.name:<7} "
                  f"prompt={r.prompt_tokens or s.target_tokens:>5} "
                  f"ttft={r.ttft_ms:7.0f}ms out={r.out_tokens:4d} {hit}",
                  flush=True)
    return results


def fetch_kvboost_reuse(url: str) -> Dict:
    """kvboost reports KV reuse via /v1/stats (cache hit-rate), not usage."""
    try:
        s = httpx.get(f"{url}/v1/stats", timeout=5.0).json()
        cache = s.get("cache", {})
        return {
            "hits": cache.get("cache_hits"),
            "approximate_hits": cache.get("approximate_hits"),
            "misses": cache.get("cache_misses"),
            "hit_rate": cache.get("hit_rate"),
        }
    except Exception:
        return {}


def report(by_backend: Dict[str, List[Outcome]], kv_stats: Dict[str, Dict]) -> None:
    print("\n" + "=" * 78)
    print("HF RAG REUSE — kvboost CacheBlend vs vLLM prefix caching (sequential)")
    print("=" * 78)
    hdr = (f"{'backend':<10} {'ok':>3} {'ttft_mean':>10} {'ttft_p50':>9} "
           f"{'ttft_last':>10} {'decTok/s':>9} {'cacheHit':>16}")
    print(hdr)
    print("-" * len(hdr))
    for b, outs in by_backend.items():
        ok = [o for o in outs if o.status == COMPLETED]
        ttfts = [o.ttft_ms for o in ok if o.ttft_ms]
        decs = [o.decode_tok_s for o in ok if o.decode_tok_s]
        mean_ttft = statistics.mean(ttfts) if ttfts else 0.0
        p50 = sorted(ttfts)[len(ttfts) // 2] if ttfts else 0.0
        last = ttfts[-1] if ttfts else 0.0
        dec = statistics.mean(decs) if decs else 0.0
        # cache hit: prefer usage.cached_tokens (vLLM); else /v1/stats (kvboost)
        tot_cached = sum(o.cached_tokens for o in ok)
        tot_prompt = sum(o.prompt_tokens for o in ok)
        if tot_cached:
            hit = f"{tot_cached}/{tot_prompt} tok"
        elif kv_stats.get(b, {}).get("hit_rate") is not None:
            hit = f"{kv_stats[b]['hit_rate']:.0%} (stats)"
        else:
            hit = "n/a"
        print(f"{b:<10} {len(ok):>3} {mean_ttft:>10.0f} {p50:>9.0f} "
              f"{last:>10.0f} {dec:>9.1f} {hit:>16}")
    print()
    print("TTFT trace per sample (ms) — watch reuse warm up:")
    for b, outs in by_backend.items():
        trace = " ".join(f"{o.ttft_ms:5.0f}" if o.status == COMPLETED else "  err"
                         for o in outs)
        print(f"  {b:<10} {trace}")
    print()
    print("Read: flatter/lower TTFT across the sequence = more effective reuse. "
          "CacheBlend should stay low on samples whose passages recurred "
          "(any order); prefix caching only when the leading prefix matched.")


def main() -> int:
    ap = argparse.ArgumentParser(description="HF RAG reuse: kvboost vs vLLM")
    ap.add_argument("--kvboost-url", default=None)
    ap.add_argument("--vllm-url", default=None)
    ap.add_argument("--kvboost-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset", default="squad")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--passages-per", type=int, default=4)
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true",
                    help="skip the dataset download, use built-in passages")
    args = ap.parse_args()

    targets = []
    if args.kvboost_url:
        targets.append(("kvboost", args.kvboost_url, args.kvboost_model))
    if args.vllm_url:
        targets.append(("vllm", args.vllm_url, args.vllm_model))
    if not targets:
        print("ERROR: give at least one of --kvboost-url / --vllm-url", file=sys.stderr)
        return 2

    samples = hw.load_rag_samples(
        dataset=args.dataset, n=args.n, passages_per=args.passages_per,
        pool_size=args.pool_size, seed=args.seed, max_tokens=args.max_tokens,
        synthetic=args.synthetic,
    )
    print(f"Loaded {len(samples)} RAG samples "
          f"(pool={args.pool_size} passages, {args.passages_per}/prompt, shuffled).")
    print("Replaying the SAME sample sequence on each backend.\n")

    by_backend: Dict[str, List[Outcome]] = {}
    kv_stats: Dict[str, Dict] = {}
    for name, url, model in targets:
        if not _wait_healthy(url):
            print(f"ERROR: {name} at {url} not healthy", file=sys.stderr)
            return 1
        print(f"── {name} ──")
        by_backend[name] = asyncio.run(
            replay_sequential(url, model, name, samples, args.timeout_s))
        if name == "kvboost":
            kv_stats[name] = fetch_kvboost_reuse(url)

    report(by_backend, kv_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
