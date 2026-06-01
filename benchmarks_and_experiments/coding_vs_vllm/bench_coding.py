"""kvboost vs vLLM on a REAL coding dataset — TTFT, throughput, OOM recovery.

Reports the two kvboost features plus full throughput:

  1. FASTER TTFT (KV reuse)  — a coding-agent reuse workload: a shared real
     repo-context prefix + varying real tasks, replayed SEQUENTIALLY so the
     prefix KV is reused across requests. We report TTFT and watch it drop as
     reuse warms (kvboost chunk-reuse + CacheBlend vs vLLM prefix caching).

  2. OOM RECOVERY            — ramp real-code context length and record, per
     backend, whether each request COMPLETED / was REJECTED gracefully (4xx) /
     hard-failed (OOM 5xx / connection-drop crash / timeout). kvboost adapts
     (chunked prefill, per-request kv-bits, clean 413); vLLM OOMs past budget.

  THROUGHPUT (both axes)     — for every completed request:
     * INPUT/prefill tok/s  = prompt_tokens / TTFT   (context ingestion rate;
       this is where KV reuse pays off — reused chunks aren't re-prefilled)
     * DECODE tok/s         = (out_tokens-1) / (last_tok - first_tok)
     * SYSTEM tok/s         = total output tokens / wall

No synthetic data: prompts are built from a real HF dataset (default
openai_humaneval). Needs ``pip install datasets``.

Usage
-----
    python bench_coding.py \\
        --kvboost-url http://localhost:9000 --kvboost-model Qwen/Qwen2.5-3B-Instruct \\
        --vllm-url    http://localhost:8001 --vllm-model    Qwen/Qwen2.5-3B-Instruct \\
        --dataset openai_humaneval --mode both
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import coding_workload as cw


# ── Outcome classification ────────────────────────────────────────────────────

COMPLETED = "COMPLETED"
REJECTED = "REJECTED_GRACEFUL"
OOM_FAIL = "OOM_FAIL"
CONN_DROP = "CONN_DROP"
TIMEOUT = "TIMEOUT"
SERVER_ERR = "SERVER_ERROR"

_OOM_MARKERS = ("out of memory", "cuda oom", "outofmemory", "cuda error",
                "no memory", "kv cache", "out of available memory")


@dataclass
class Outcome:
    backend: str
    label: str
    target_tokens: int
    status: str
    http_code: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    out_tokens: int = 0
    ttft_ms: float = 0.0
    wall_ms: float = 0.0
    first_tok_time: float = 0.0
    last_tok_time: float = 0.0
    detail: str = ""

    @property
    def decode_tok_s(self) -> float:
        if self.out_tokens > 1 and self.last_tok_time > self.first_tok_time:
            return (self.out_tokens - 1) / (self.last_tok_time - self.first_tok_time)
        return 0.0

    @property
    def input_tok_s(self) -> float:
        """Prefill/context-ingestion rate: prompt tokens per second of prefill.
        TTFT ≈ prefill time (first token emitted right after prefill). Reuse
        makes this soar because reused chunks aren't re-prefilled."""
        if self.prompt_tokens > 0 and self.ttft_ms > 0:
            return self.prompt_tokens / (self.ttft_ms / 1000.0)
        return 0.0


def _classify_http(code: int, body_text: str) -> Tuple[str, str]:
    low = body_text.lower()
    if code == 200:
        return COMPLETED, ""
    if code in (400, 413, 422):
        return REJECTED, body_text[:200]
    if code >= 500:
        if any(m in low for m in _OOM_MARKERS):
            return OOM_FAIL, body_text[:200]
        return SERVER_ERR, body_text[:200]
    return SERVER_ERR, f"HTTP {code}: {body_text[:160]}"


# ── Streaming request ──────────────────────────────────────────────────────────


async def run_prompt(
    client: httpx.AsyncClient, base_url: str, model: str, backend: str,
    prompt, timeout_s: float,
) -> Outcome:
    o = Outcome(backend=backend, label=prompt.name,
                target_tokens=prompt.target_tokens, status=SERVER_ERR)
    body = prompt.to_body(model, stream=True)
    t0 = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=body, timeout=timeout_s,
        ) as resp:
            o.http_code = resp.status_code
            if resp.status_code != 200:
                raw = (await resp.aread()).decode("utf-8", "replace")
                o.status, o.detail = _classify_http(resp.status_code, raw)
                o.wall_ms = (time.perf_counter() - t0) * 1000
                return o
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "error" in data:
                    err = data["error"]
                    o.status, o.detail = _classify_http(
                        err.get("code", 500) or 500, json.dumps(err))
                    o.wall_ms = (time.perf_counter() - t0) * 1000
                    return o
                usage = data.get("usage")
                if usage:
                    o.prompt_tokens = usage.get("prompt_tokens", o.prompt_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    o.cached_tokens = details.get("cached_tokens", o.cached_tokens)
                ch = (data.get("choices") or [{}])
                if ch:
                    delta = (ch[0].get("delta") or {}).get("content")
                    if delta:
                        now = time.perf_counter()
                        if o.out_tokens == 0:
                            o.first_tok_time = now
                            o.ttft_ms = (now - t0) * 1000
                        o.out_tokens += 1
                        o.last_tok_time = now
            o.wall_ms = (time.perf_counter() - t0) * 1000
            o.status = COMPLETED
            return o
    except httpx.TimeoutException as e:
        o.status, o.detail = TIMEOUT, repr(e)[:160]
    except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as e:
        o.status, o.detail = CONN_DROP, repr(e)[:160]
    except Exception as e:
        o.status, o.detail = SERVER_ERR, repr(e)[:160]
    o.wall_ms = (time.perf_counter() - t0) * 1000
    return o


# ── Reuse / TTFT + throughput mode (sequential) ────────────────────────────────


async def bench_reuse(
    base_url: str, model: str, backend: str, prompts, timeout_s: float,
) -> List[Outcome]:
    """Sequential replay so request N reuses request <N's cached prefix KV."""
    results: List[Outcome] = []
    async with httpx.AsyncClient() as client:
        for p in prompts:
            r = await run_prompt(client, base_url, model, backend, p, timeout_s)
            results.append(r)
            tag = "ok " if r.status == COMPLETED else r.status
            print(f"  [{backend}] {tag:<10} {r.label:<8} "
                  f"prompt={r.prompt_tokens or '?':>6} ttft={r.ttft_ms:7.0f}ms "
                  f"in={r.input_tok_s:7.0f}t/s dec={r.decode_tok_s:5.1f}t/s "
                  f"out={r.out_tokens}", flush=True)
    return results


# ── OOM ramp (sequential, recovery-aware) ──────────────────────────────────────


async def bench_oom(
    base_url: str, model: str, backend: str, prompts, timeout_s: float,
) -> List[Outcome]:
    results: List[Outcome] = []
    async with httpx.AsyncClient() as client:
        for p in prompts:
            r = await run_prompt(client, base_url, model, backend, p, timeout_s)
            results.append(r)
            mark = {COMPLETED: "✓ completed", REJECTED: "▲ rejected(graceful)",
                    OOM_FAIL: "✗ OOM", CONN_DROP: "✗ conn-drop(crash?)",
                    TIMEOUT: "✗ timeout", SERVER_ERR: "✗ error"}.get(r.status, r.status)
            print(f"  [{backend}] ~{p.target_tokens:>6} tok  {mark:<22} "
                  f"prompt={r.prompt_tokens or '?':>6} ttft={r.ttft_ms:7.0f}ms "
                  f"{('+'+r.detail) if r.detail else ''}", flush=True)
            await asyncio.sleep(1.0)   # let the server GC between heavy prompts
    return results


# ── Reporting ──────────────────────────────────────────────────────────────────


def _agg(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[max(0, int(q * len(s)) - 1)]


def report_reuse(by_backend: Dict[str, List[Outcome]]) -> None:
    print("\n" + "=" * 84)
    print("FASTER TTFT + THROUGHPUT — coding-agent reuse (sequential, shared repo context)")
    print("=" * 84)
    hdr = (f"{'backend':<10} {'ok':>3} {'ttft1st':>8} {'ttftP50':>8} {'ttftLast':>9} "
           f"{'inTok/s':>8} {'decTok/s':>9} {'sysTok/s':>9}")
    print(hdr)
    print("-" * len(hdr))
    for b, outs in by_backend.items():
        ok = [o for o in outs if o.status == COMPLETED]
        ttfts = [o.ttft_ms for o in ok if o.ttft_ms]
        intps = [o.input_tok_s for o in ok if o.input_tok_s]
        dectps = [o.decode_tok_s for o in ok if o.decode_tok_s]
        tot_out = sum(o.out_tokens for o in ok)
        wall = sum(o.wall_ms for o in ok) / 1000.0
        sys_tps = tot_out / wall if wall > 0 else 0.0
        first = ttfts[0] if ttfts else 0.0
        last = ttfts[-1] if ttfts else 0.0
        print(f"{b:<10} {len(ok):>3} {first:>8.0f} {_agg(ttfts,0.5):>8.0f} "
              f"{last:>9.0f} {statistics.mean(intps) if intps else 0:>8.0f} "
              f"{statistics.mean(dectps) if dectps else 0:>9.1f} {sys_tps:>9.1f}")
    print()
    print("TTFT trace per request (ms) — watch reuse warm up after the 1st:")
    for b, outs in by_backend.items():
        trace = " ".join(f"{o.ttft_ms:6.0f}" if o.status == COMPLETED else "   err"
                         for o in outs)
        print(f"  {b:<10}{trace}")
    print("\nRead: ttftLast ≪ ttft1st = reuse working; higher inTok/s = faster "
          "context ingestion. vLLM usually leads decTok/s (continuous batching) "
          "— the kvboost story here is TTFT + input throughput on reused context.")


def report_oom(by_backend: Dict[str, List[Outcome]]) -> None:
    print("\n" + "=" * 84)
    print("OOM RECOVERY — real-code context ramp")
    print("=" * 84)
    backends = list(by_backend.keys())
    sizes = sorted({o.target_tokens for outs in by_backend.values() for o in outs})
    short = {COMPLETED: "✓ ok", REJECTED: "▲ reject", OOM_FAIL: "✗ OOM",
             CONN_DROP: "✗ crash", TIMEOUT: "✗ t/o", SERVER_ERR: "✗ err"}
    hdr = f"{'ctx~tok':>8} " + " ".join(f"{b:>18}" for b in backends)
    print(hdr); print("-" * len(hdr))
    for sz in sizes:
        cells = []
        for b in backends:
            o = next((x for x in by_backend[b] if x.target_tokens == sz), None)
            if o is None:
                cells.append(f"{'-':>18}")
            else:
                s = short.get(o.status, o.status)
                extra = f" {o.ttft_ms/1000:.0f}s" if o.status == COMPLETED else ""
                cells.append(f"{s + extra:>18}")
        print(f"{sz:>8} " + " ".join(cells))
    print()
    for b in backends:
        outs = by_backend[b]
        completed = [o.target_tokens for o in outs if o.status == COMPLETED]
        crashed = [o.target_tokens for o in outs if o.status in (OOM_FAIL, CONN_DROP)]
        print(f"  {b}: largest COMPLETED ≈{max(completed) if completed else 0} tok; "
              + (f"first hard-fail (OOM/crash) ≈{min(crashed)} tok"
                 if crashed else "no hard OOM/crash observed"))


# ── Main ─────────────────────────────────────────────────────────────────────


def _wait_healthy(url: str, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for path in ("/health", "/v1/models"):
            try:
                if httpx.get(f"{url}{path}", timeout=3.0).status_code < 500:
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="kvboost vs vLLM — real coding benchmark")
    ap.add_argument("--kvboost-url", default=None)
    ap.add_argument("--vllm-url", default=None)
    ap.add_argument("--kvboost-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset", default="openai_humaneval")
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--mode", choices=["ttft", "oom", "both"], default="both")
    ap.add_argument("--n", type=int, default=10, help="reuse-workload request count")
    ap.add_argument("--n-files", type=int, default=6,
                    help="real files in the shared repo-context prefix")
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[2000, 8000, 16000, 32000, 64000, 96000],
                    help="OOM-ramp target context sizes (approx tokens)")
    ap.add_argument("--corpus-size", type=int, default=40,
                    help="real code units to pull from the dataset")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    args = ap.parse_args()

    targets = []
    if args.kvboost_url:
        targets.append(("kvboost", args.kvboost_url, args.kvboost_model))
    if args.vllm_url:
        targets.append(("vllm", args.vllm_url, args.vllm_model))
    if not targets:
        print("ERROR: give at least one of --kvboost-url / --vllm-url", file=sys.stderr)
        return 2

    # Load the real corpus ONCE; both backends see identical prompts.
    corpus = cw.load_corpus(args.dataset, split=args.dataset_split,
                            n_units=args.corpus_size)
    print(f"Loaded {len(corpus)} real code units from '{args.dataset}'.")

    for name, url, _ in targets:
        if not _wait_healthy(url):
            print(f"ERROR: {name} at {url} not healthy", file=sys.stderr)
            return 1

    reuse_res: Dict[str, List[Outcome]] = {}
    oom_res: Dict[str, List[Outcome]] = {}

    if args.mode in ("ttft", "both"):
        prompts = cw.reuse_prompts(corpus, n=args.n, n_files=args.n_files)
        print(f"\nReuse workload: {len(prompts)} requests, shared {args.n_files}-file "
              f"repo prefix (~{prompts[0].target_tokens} tok), sequential.")
        for name, url, model in targets:
            print(f"── {name} ──")
            reuse_res[name] = asyncio.run(
                bench_reuse(url, model, name, prompts, args.timeout_s))

    if args.mode in ("oom", "both"):
        prompts = cw.oom_prompts(corpus, contexts=args.contexts)
        print(f"\nOOM ramp: contexts {args.contexts} (real code concatenated).")
        for name, url, model in targets:
            print(f"── {name} ──")
            oom_res[name] = asyncio.run(
                bench_oom(url, model, name, prompts, args.timeout_s))

    if reuse_res:
        report_reuse(reuse_res)
    if oom_res:
        report_oom(oom_res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
