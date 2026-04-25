#!/usr/bin/env python3
"""
Latency Benchmark: KVBoost vs vLLM (prefix-caching) vs Baseline

Measures Time-To-First-Token (TTFT) and end-to-end latency.
For vLLM: enable_prefix_caching=True enables KV cache reuse across requests.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DATASET_REPO = "JetBrains-Research/lca-bug-localization"
DATASET_CONFIG = "py"

_SAMPLE_CACHE: Dict[tuple, List[Dict]] = {}


@dataclass
class LatencyResult:
    """Latency measurement for one inference"""
    sample_id: str
    backend: str
    model: str
    context_length: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float          # Time to first token
    total_latency_ms: float  # End-to-end latency
    tokens_per_second: float

    # For vLLM with prefix caching
    cache_hit: bool = False
    cache_reuse_ratio: float = 0.0


def _load_longbench_samples(
    n_samples: int,
    max_context_tokens: int,
) -> List[Dict]:
    cache_key = (n_samples, max_context_tokens)
    if cache_key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[cache_key]

    from datasets import load_dataset
    import json as _json, random as _random

    try:
        ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split="train")
    except Exception as e:
        log.error("Could not load %s: %s", DATASET_REPO, e)
        return []

    samples = []
    for item in ds:
        if len(samples) >= n_samples:
            break
        diff = item.get("diff", "")
        if not diff or len(diff) < 10:
            continue
        issue_title = item.get("issue_title", "").strip()
        issue_body = (item.get("issue_body", "") or "").strip()
        question = issue_title + ("\n" + issue_body[:1000] if issue_body else "")
        if not question:
            continue
        changed_files = item.get("changed_files", [])
        if isinstance(changed_files, str):
            try:
                changed_files = _json.loads(changed_files)
            except Exception:
                changed_files = [changed_files]
        if not changed_files:
            continue
        correct_file = changed_files[0]
        approx_tokens = int(len((diff + question).split()) * 0.75)
        if approx_tokens > max_context_tokens:
            continue
        options = list(dict.fromkeys(changed_files))
        if len(options) < 2:
            continue
        _random.seed(42)
        choices = [correct_file] + [f for f in options if f != correct_file][:3]
        _random.shuffle(choices)
        correct_idx = choices.index(correct_file)
        letter = chr(ord("A") + correct_idx)
        samples.append({
            "task": "bug_localization",
            "context": diff,
            "input": question,
            "answers": [letter, correct_file],
            "approx_tokens": approx_tokens,
            "choices": choices,
            "correct_idx": correct_idx,
        })

    log.info("Loaded %d samples from %s", len(samples), DATASET_REPO)
    _SAMPLE_CACHE[cache_key] = samples
    return samples


def _format_prompt(context: str, question: str, choices: Optional[List[str]] = None) -> str:
    header = (
        "You are a code reviewer. Read the following git diff carefully.\n\n"
        f"Diff:\n{context}\n\nIssue: {question}\n\n"
    )
    if choices:
        opts = "\n".join(f"  {chr(ord('A')+i)}) {c}" for i, c in enumerate(choices))
        return header + f"Which file contains the bug? Choose one:\n{opts}\n\nAnswer (letter only):"
    return header + "Which file contains the bug?\n\nAnswer:"




def _measure_kvboost(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
    recompute_strategy: str = "selective",
    chunk_boundary_window: int = 0,
    overlap_k: int = 0,
    sink_tokens: int = 0,
) -> List[LatencyResult]:
    import torch
    from kvboost import KVBoost, GenerationMode

    engine = KVBoost.from_pretrained(
        model_name=model,
        max_cache_bytes=4_000_000_000,
        chunk_size=128,
        recompute_overlap=16,
        recompute_strategy=recompute_strategy,
        chunk_boundary_window=chunk_boundary_window,
        overlap_k=overlap_k,
        sink_tokens=sink_tokens,
    )
    tokenizer = engine.tokenizer

    results = []
    for i, sample in enumerate(samples):
        prompt = _format_prompt(sample["context"], sample["input"], sample.get("choices"))
        prompt_tokens = len(tokenizer.encode(prompt))

        # Warm GPU/caches on first sample to avoid cold-start skew
        if i == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        result = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ttft_ms = result.ttft_ms
        total_ms = result.total_ms
        completion_tokens = result.generated_tokens
        tps = result.tokens_per_sec

        results.append(LatencyResult(
            sample_id=f"{sample['task']}_{i:04d}",
            backend="kvboost",
            model=model,
            context_length=sample["approx_tokens"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
            cache_hit=False,
            cache_reuse_ratio=result.kv_reuse_ratio,
        ))
        log.debug("kvboost sample %d: ttft=%.1fms total=%.1fms reuse=%.1f%%",
                  i, ttft_ms, total_ms, result.kv_reuse_ratio * 100)

    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _measure_vllm_prefixcache(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
    enable_prefix_caching: bool = True,
) -> List[LatencyResult]:
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, enable_prefix_caching=enable_prefix_caching)
    params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    results = []
    for i, sample in enumerate(samples):
        prompt = _format_prompt(sample["context"], sample["input"], sample.get("choices"))
        prompt_tokens = len(tokenizer.encode(prompt))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = llm.generate([prompt], params)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        out = outputs[0]
        completion_tokens = len(out.outputs[0].token_ids)

        # vLLM doesn't expose per-request TTFT directly; approximate from generation metrics
        # If prefix was cached, TTFT ≈ time to decode first new token (fast)
        # We use the finish_reason and timing metadata when available
        ttft_ms = getattr(out, "ttft_ms", None)
        if ttft_ms is None:
            # Fallback: estimate TTFT as total / completion_tokens (uniform decoding)
            ttft_ms = total_ms / max(completion_tokens, 1)

        tps = completion_tokens / (total_ms / 1000) if total_ms > 0 else 0.0

        # Detect cache hit from vLLM RequestOutput metadata
        num_cached = getattr(out, "num_cached_tokens", None)
        cache_hit = num_cached is not None and num_cached > 0
        cache_reuse_ratio = (num_cached / prompt_tokens) if (cache_hit and prompt_tokens > 0) else 0.0

        results.append(LatencyResult(
            sample_id=f"{sample['task']}_{i:04d}",
            backend="vllm_prefixcache",
            model=model,
            context_length=sample["approx_tokens"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
            cache_hit=cache_hit,
            cache_reuse_ratio=cache_reuse_ratio,
        ))
        log.debug("vllm sample %d: total=%.1fms tps=%.1f cache_hit=%s",
                  i, total_ms, tps, cache_hit)

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _measure_baseline(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
) -> List[LatencyResult]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    hf_model.eval()

    results = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            prompt = _format_prompt(sample["context"], sample["input"], sample.get("choices"))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_tokens = inputs["input_ids"].shape[1]

            if device == "cuda":
                torch.cuda.synchronize()

            # TTFT: time the first forward pass / first token generation
            t_start = time.perf_counter()

            # Generate first token only
            first_out = hf_model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t_start) * 1000

            # Full generation for total latency
            t_gen = time.perf_counter()
            gen_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t_gen) * 1000 + ttft_ms

            completion_tokens = gen_ids.shape[1] - prompt_tokens
            tps = completion_tokens / (total_ms / 1000) if total_ms > 0 else 0.0

            results.append(LatencyResult(
                sample_id=f"{sample['task']}_{i:04d}",
                backend="baseline",
                model=model,
                context_length=sample["approx_tokens"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                total_latency_ms=total_ms,
                tokens_per_second=tps,
            ))
            log.debug("baseline sample %d: ttft=%.1fms total=%.1fms", i, ttft_ms, total_ms)

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def benchmark_latency(
    backend: str,
    model: str,
    n_samples: int = 50,
    dataset_path: Optional[Path] = None,
    max_context_tokens: int = 8192,
    vllm_prefix_caching: bool = True,
    kvboost_recompute_strategy: str = "selective",
    kvboost_chunk_boundary_window: int = 0,
    kvboost_overlap_k: int = 0,
    kvboost_sink_tokens: int = 0,
) -> List[LatencyResult]:
    """
    Run latency benchmark for a specific backend.

    Args:
        backend: 'kvboost', 'vllm_prefixcache', or 'baseline'
        model: Model name
        n_samples: Number of test samples
        dataset_path: Unused; kept for API compatibility
        max_context_tokens: Max context length
        vllm_prefix_caching: For vLLM backend, enable prefix caching
        kvboost_recompute_strategy: KVBoost recompute strategy ('selective', 'cacheblend', 'none')
        kvboost_chunk_boundary_window: Adaptive boundary splitting window
        kvboost_overlap_k: Overlapping chunk encoding tokens
        kvboost_sink_tokens: Attention sink prefix tokens

    Returns:
        List of LatencyResult objects
    """
    log.info(
        "Starting latency benchmark for %s with %s (vllm_prefix_caching=%s)",
        backend,
        model,
        vllm_prefix_caching if backend == "vllm_prefixcache" else "N/A",
    )

    samples = _load_longbench_samples(n_samples, max_context_tokens)
    if not samples:
        log.warning("No LongBench samples loaded — check dataset availability")
        return []

    if backend == "kvboost":
        results = _measure_kvboost(
            samples, model,
            recompute_strategy=kvboost_recompute_strategy,
            chunk_boundary_window=kvboost_chunk_boundary_window,
            overlap_k=kvboost_overlap_k,
            sink_tokens=kvboost_sink_tokens,
        )
    elif backend == "vllm_prefixcache":
        results = _measure_vllm_prefixcache(samples, model, enable_prefix_caching=vllm_prefix_caching)
    elif backend == "baseline":
        results = _measure_baseline(samples, model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    log.info(
        "Processed %d latency samples for %s (ttft_mean=%.1fms, tps_mean=%.1f)",
        len(results),
        backend,
        np.mean([r.ttft_ms for r in results]) if results else 0.0,
        np.mean([r.tokens_per_second for r in results]) if results else 0.0,
    )
    return results


def aggregate_latency_results(
    results: Dict[str, List[LatencyResult]]
) -> Dict[str, Dict]:
    """Aggregate latency results by backend"""
    aggregated = {}

    for backend, backend_results in results.items():
        if not backend_results:
            continue

        ttfts = [r.ttft_ms for r in backend_results]
        total_lats = [r.total_latency_ms for r in backend_results]
        tps = [r.tokens_per_second for r in backend_results]

        cache_hits = sum(1 for r in backend_results if r.cache_hit)
        reuse_ratios = [r.cache_reuse_ratio for r in backend_results if r.cache_reuse_ratio > 0]
        avg_cache_reuse = float(np.mean(reuse_ratios)) if reuse_ratios else 0.0

        aggregated[backend] = {
            "n_samples": len(backend_results),
            "ttft_ms": {
                "mean": float(np.mean(ttfts)),
                "median": float(np.median(ttfts)),
                "p25": float(np.percentile(ttfts, 25)),
                "p75": float(np.percentile(ttfts, 75)),
                "p95": float(np.percentile(ttfts, 95)),
                "min": float(np.min(ttfts)),
                "max": float(np.max(ttfts)),
            },
            "total_latency_ms": {
                "mean": float(np.mean(total_lats)),
                "median": float(np.median(total_lats)),
                "p95": float(np.percentile(total_lats, 95)),
            },
            "tokens_per_second": {
                "mean": float(np.mean(tps)),
                "median": float(np.median(tps)),
            },
            "cache_hit_rate": cache_hits / len(backend_results) if backend == "vllm_prefixcache" else 0.0,
            "avg_cache_reuse_ratio": avg_cache_reuse if backend == "vllm_prefixcache" else 0.0,
            "samples": [asdict(r) for r in backend_results],
        }

    return aggregated


def print_latency_table(aggregated: Dict[str, Dict]):
    """Print latency comparison table"""
    print("\n" + "="*110)
    print("  LATENCY BENCHMARK RESULTS (Time-To-First-Token & Throughput)")
    print("="*110)
    print(
        f"  {'Backend':<20} {'TTFT Mean':>12} {'TTFT P95':>12} {'Total Lat':>12} "
        f"{'Throughput':>12} {'vs Baseline':>12}"
    )
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    baseline_ttft = None
    for backend in ['kvboost', 'vllm_prefixcache', 'baseline']:
        if backend not in aggregated:
            continue

        agg = aggregated[backend]
        ttft_mean = agg["ttft_ms"]["mean"]
        ttft_p95 = agg["ttft_ms"]["p95"]
        total_lat = agg["total_latency_ms"]["mean"]
        tps = agg["tokens_per_second"]["mean"]

        if backend == 'baseline':
            baseline_ttft = ttft_mean

        speedup = baseline_ttft / ttft_mean if baseline_ttft and ttft_mean > 0 else 1.0
        speedup_str = f"{speedup:.2f}x" if backend != 'baseline' else "<<<BASELINE"

        print(
            f"  {backend:<20} {ttft_mean:>11.1f}ms {ttft_p95:>11.1f}ms "
            f"{total_lat:>11.1f}ms {tps:>11.1f}tok/s {speedup_str:>12}"
        )

    print("="*110)

    if 'vllm_prefixcache' in aggregated:
        agg = aggregated['vllm_prefixcache']
        cache_hit_rate = agg.get("cache_hit_rate", 0)
        cache_reuse = agg.get("avg_cache_reuse_ratio", 0)
        print(f"  vLLM Prefix Caching:  Cache Hit Rate={cache_hit_rate:.1%}  Avg Reuse={cache_reuse:.1%}")

    print("="*110 + "\n")


def save_latency_results(
    aggregated: Dict[str, Dict],
    model: str,
    output_path: Optional[Path] = None
) -> Path:
    """Save latency results to JSON"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"latency_benchmark_{model.replace('/', '_')}_{timestamp}.json"

    output_path.parent.mkdir(exist_ok=True, parents=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "latency",
        "model": model,
        "results": aggregated,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    log.info("Latency results saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Latency Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-vllm-prefix-caching", action="store_true")

    args = parser.parse_args()

    results = {}
    for backend in ['kvboost', 'vllm_prefixcache', 'baseline']:
        vllm_pc = not args.no_vllm_prefix_caching if backend == 'vllm_prefixcache' else False
        results[backend] = benchmark_latency(backend, args.model, args.n_samples, vllm_prefix_caching=vllm_pc)

    aggregated = aggregate_latency_results(results)
    print_latency_table(aggregated)
    save_latency_results(aggregated, args.model, args.output)
