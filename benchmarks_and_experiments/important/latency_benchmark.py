#!/usr/bin/env python3
"""
Latency Benchmark: KVBoost vs vLLM (prefix-caching) vs Baseline

Measures Time-To-First-Token (TTFT) and end-to-end latency.
For vLLM: enable_prefix_caching=True enables KV cache reuse across requests.
"""

import gc
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DATASET_REPO = "JetBrains-Research/lca-bug-localization"
DATASET_CONFIG = "py"

_SAMPLE_CACHE: Dict[tuple, List[Dict]] = {}

LETTERS = ["A", "B", "C", "D"]


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
    model_name: str = "Qwen/Qwen2.5-3B",
) -> List[Dict]:
    cache_key = (n_samples, max_context_tokens, model_name)
    if cache_key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[cache_key]

    from datasets import load_dataset
    from transformers import AutoTokenizer

    log.info("Loading %s ...", DATASET_REPO)
    try:
        ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split="train")
    except Exception as e:
        log.error("Could not load %s: %s", DATASET_REPO, e)
        return []

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    raw_samples = []
    all_files: set = set()
    skipped_long = 0
    skipped_nodiff = 0

    for row in ds:
        diff = row.get("diff", "")
        if not diff or len(diff.strip()) < 10:
            skipped_nodiff += 1
            continue
        ctx_tokens = len(tok.encode(diff))
        if ctx_tokens > max_context_tokens:
            skipped_long += 1
            continue
        changed_files_raw = row.get("changed_files", "")
        if isinstance(changed_files_raw, list):
            changed_files = [str(f).strip().strip("'\"`[]") for f in changed_files_raw if str(f).strip()]
        elif isinstance(changed_files_raw, str):
            import ast
            try:
                parsed = ast.literal_eval(changed_files_raw)
                changed_files = [str(f).strip() for f in (parsed if isinstance(parsed, list) else [parsed]) if str(f).strip()]
            except Exception:
                changed_files = [f.strip().strip("'\"`[]") for f in changed_files_raw.replace(",", "\n").split("\n") if f.strip().strip("'\"`[]")]
        else:
            changed_files = []
        if not changed_files:
            skipped_nodiff += 1
            continue
        all_files.update(changed_files)
        issue_title = (row.get("issue_title", "") or "").strip()
        issue_body = (row.get("issue_body", "") or "").strip()
        question = issue_title
        if issue_body:
            body_preview = issue_body[:1000] + ("..." if len(issue_body) > 1000 else "")
            question = f"{issue_title}\n{body_preview}"
        raw_samples.append({
            "diff": diff,
            "correct_file": changed_files[0],
            "all_changed_files": changed_files,
            "question": question,
            "ctx_tokens": ctx_tokens,
            "repo": f"{row.get('repo_owner', '')}/{row.get('repo_name', '')}",
            "changed_files_count": row.get("changed_files_count", len(changed_files)),
        })

    log.info("After filtering: %d eligible, %d too long, %d no diff/files",
             len(raw_samples), skipped_long, skipped_nodiff)

    if not raw_samples:
        log.error("No samples fit within %d tokens!", max_context_tokens)
        return []

    all_files_list = sorted(all_files)
    rng = np.random.RandomState(42)

    def _bucket(t: int) -> str:
        if t < 512: return "0-512"
        elif t < 1024: return "512-1K"
        elif t < 2048: return "1K-2K"
        elif t < 4096: return "2K-4K"
        else: return "4K+"

    bucket_pools: Dict[str, list] = defaultdict(list)
    for s in raw_samples:
        bucket_pools[_bucket(s["ctx_tokens"])].append(s)
    for bk in bucket_pools:
        rng.shuffle(bucket_pools[bk])

    n_buckets = len(bucket_pools)
    per_bucket = n_samples // n_buckets if n_buckets else n_samples
    selected = []
    overflow = []
    for bk in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+"]:
        pool = bucket_pools.get(bk, [])
        take = min(per_bucket, len(pool))
        selected.extend(pool[:take])
        overflow.extend(pool[take:])
    remaining = n_samples - len(selected)
    if remaining > 0 and overflow:
        rng.shuffle(overflow)
        selected.extend(overflow[:remaining])

    samples = []
    for idx, raw in enumerate(selected):
        correct_file = raw["correct_file"]
        distractor_pool = [f for f in all_files_list if f not in raw["all_changed_files"]]
        rng.shuffle(distractor_pool)
        while len(distractor_pool) < 6:
            distractor_pool.append(f"src/fake_module_{len(distractor_pool)}.py")

        base_info = {
            "task": "bug_localization",
            "context": raw["diff"],
            "ctx_tokens": raw["ctx_tokens"],
            "approx_tokens": raw["ctx_tokens"],
        }

        d1 = distractor_pool[:3]
        choices_1 = d1 + [correct_file]
        rng.shuffle(choices_1)
        q1_pos = choices_1.index(correct_file)
        samples.append({
            **base_info,
            "id": f"bug_{idx}_q1",
            "input": f"Bug report: {raw['question']}\n\nWhich file was modified to fix this bug?",
            "choices": choices_1,
            "answers": [LETTERS[q1_pos], correct_file],
            "correct_idx": q1_pos,
            "pair_group": idx,
        })

        d2 = distractor_pool[3:6]
        choices_2 = d2 + [correct_file]
        rng.shuffle(choices_2)
        for _ in range(20):
            if choices_2.index(correct_file) != q1_pos:
                break
            rng.shuffle(choices_2)
        else:
            pos = choices_2.index(correct_file)
            target = (q1_pos + 1) % 4
            choices_2[pos], choices_2[target] = choices_2[target], choices_2[pos]
        q2_pos = choices_2.index(correct_file)
        samples.append({
            **base_info,
            "id": f"bug_{idx}_q2",
            "input": f"Issue: {raw['question']}\n\nIdentify the file that contains the fix for this issue.",
            "choices": choices_2,
            "answers": [LETTERS[q2_pos], correct_file],
            "correct_idx": q2_pos,
            "pair_group": idx,
        })

    log.info("Loaded %d paired samples from %s", len(samples), DATASET_REPO)
    _SAMPLE_CACHE[cache_key] = samples
    return samples


def _format_prompt_prefix(context: str) -> str:
    return (
        "Read the following code diff and answer the question.\n\n"
        f"--- BEGIN DIFF ---\n{context}\n--- END DIFF ---\n\n"
    )


def _format_prompt_suffix(question: str, choices: List[str]) -> str:
    parts = [f"{question}\n"]
    for i, choice in enumerate(choices):
        parts.append(f"{LETTERS[i]}. {choice}")
    parts.append("\nAnswer with just the letter (A, B, C, or D):")
    return "\n".join(parts)


def _format_prompt(context: str, question: str, choices: Optional[List[str]] = None) -> str:
    if choices:
        return _format_prompt_prefix(context) + _format_prompt_suffix(question, choices)
    return (
        "Read the following code diff and answer the question.\n\n"
        f"--- BEGIN DIFF ---\n{context}\n--- END DIFF ---\n\n"
        f"{question}\n\nAnswer:"
    )




_CHECKPOINT_INTERVAL = 10


def _atomic_checkpoint(path: Path, data: object) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def _measure_kvboost(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
    max_cache_bytes: int = 4_000_000_000,
    recency_window_chunks: int = 8,
    recompute_strategy: str = "selective",
    chunk_boundary_window: int = 0,
    overlap_k: int = 0,
    sink_tokens: int = 0,
    checkpoint: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> List[LatencyResult]:
    import torch
    from kvboost import KVBoost, GenerationMode

    engine = KVBoost.from_pretrained(
        model_name=model,
        max_cache_bytes=max_cache_bytes,
        recency_window_chunks=recency_window_chunks,
        chunk_size=128,
        recompute_overlap=16,
        recompute_strategy=recompute_strategy,
        chunk_boundary_window=chunk_boundary_window,
        overlap_k=overlap_k,
        sink_tokens=sink_tokens,
    )
    tokenizer = engine.tokenizer
    n = len(samples)

    ttft_running = []
    results = []
    for i, sample in enumerate(samples):
        is_q1 = sample.get("id", "").endswith("_q1")
        if is_q1:
            engine.reset_cache()

        prefix = _format_prompt_prefix(sample["context"])
        suffix = _format_prompt_suffix(sample["input"], sample["choices"])
        prompt = prefix + suffix
        prompt_tokens = len(tokenizer.encode(prompt))
        cacheable_prefix_len = len(tokenizer.encode(prefix, add_special_tokens=True))

        if i == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()

        result = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
            cacheable_prefix_len=cacheable_prefix_len,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ttft_ms = result.ttft_ms
        total_ms = result.total_ms
        completion_tokens = result.generated_tokens
        tps = result.tokens_per_sec
        ttft_running.append(ttft_ms)
        query_type = "COLD" if is_q1 else "WARM"

        r = LatencyResult(
            sample_id=sample.get("id", f"{sample['task']}_{i:04d}"),
            backend="kvboost",
            model=model,
            context_length=sample["approx_tokens"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
            cache_hit=not is_q1,
            cache_reuse_ratio=result.kv_reuse_ratio,
        )
        results.append(r)
        log.info("[kvboost latency %d/%d] %s  ttft=%.1fms  total=%.1fms  tps=%.1f  reuse=%.1f%%  ctx=%d tok  (avg_ttft=%.1fms)",
                 i + 1, n, query_type, ttft_ms, total_ms, tps, result.kv_reuse_ratio * 100,
                 sample["approx_tokens"], np.mean(ttft_running))
        if checkpoint and checkpoint_path and (i + 1) % _CHECKPOINT_INTERVAL == 0:
            _atomic_checkpoint(checkpoint_path, {
                "n_done": i + 1, "n_total": n,
                "avg_ttft_ms": round(float(np.mean(ttft_running)), 2),
                "samples": [{
                    **asdict(r2),
                    "pair_group": samples[j].get("pair_group"),
                    "query_type": "COLD" if samples[j].get("id", "").endswith("_q1") else "WARM",
                    "repo": samples[j].get("repo", ""),
                    "choices": samples[j].get("choices", []),
                    "question": samples[j].get("input", ""),
                } for j, r2 in enumerate(results)],
            })
            log.debug("  checkpoint saved (%d/%d)", i + 1, n)

    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _measure_vllm_prefixcache(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
    max_context_tokens: int = 8192,
    enable_prefix_caching: bool = True,
    gpu_memory_utilization: float = 0.95,
    enforce_eager: bool = True,
    max_num_seqs: int = 1,
    checkpoint: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> List[LatencyResult]:
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log.info("[vllm_prefixcache] GPU free before LLM init: %.2f GiB", torch.cuda.mem_get_info()[0] / 1e9)

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, enable_prefix_caching=enable_prefix_caching,
              max_model_len=max_context_tokens + 512,
              gpu_memory_utilization=gpu_memory_utilization,
              enforce_eager=enforce_eager,
              max_num_seqs=max_num_seqs)
    params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    n = len(samples)
    ttft_running = []

    results = []
    for i, sample in enumerate(samples):
        prompt = _format_prompt(sample["context"], sample["input"], sample.get("choices", []))
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
        ttft_ms = getattr(out, "ttft_ms", None) or total_ms / max(completion_tokens, 1)
        tps = completion_tokens / (total_ms / 1000) if total_ms > 0 else 0.0
        num_cached = getattr(out, "num_cached_tokens", None)
        cache_hit = num_cached is not None and num_cached > 0
        cache_reuse_ratio = (num_cached / prompt_tokens) if (cache_hit and prompt_tokens > 0) else 0.0
        ttft_running.append(ttft_ms)

        r = LatencyResult(
            sample_id=sample.get("id", f"{sample['task']}_{i:04d}"),
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
        )
        results.append(r)
        log.info("[vllm_prefixcache latency %d/%d] ttft=%.1fms  total=%.1fms  tps=%.1f  cache_hit=%s  ctx=%d tok  (avg_ttft=%.1fms)",
                 i + 1, n, ttft_ms, total_ms, tps, "✓" if cache_hit else "✗",
                 sample["approx_tokens"], np.mean(ttft_running))
        if checkpoint and checkpoint_path and (i + 1) % _CHECKPOINT_INTERVAL == 0:
            _atomic_checkpoint(checkpoint_path, {
                "n_done": i + 1, "n_total": n,
                "avg_ttft_ms": round(float(np.mean(ttft_running)), 2),
                "samples": [{
                    **asdict(r2),
                    "pair_group": samples[j].get("pair_group"),
                    "query_type": "COLD" if samples[j].get("id", "").endswith("_q1") else "WARM",
                    "repo": samples[j].get("repo", ""),
                    "choices": samples[j].get("choices", []),
                    "question": samples[j].get("input", ""),
                } for j, r2 in enumerate(results)],
            })
            log.debug("  checkpoint saved (%d/%d)", i + 1, n)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return results


def _measure_baseline(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
    checkpoint: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> List[LatencyResult]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
    ).to(device)
    hf_model.eval()
    n = len(samples)
    ttft_running = []

    results = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            prompt = _format_prompt(sample["context"], sample["input"], sample.get("choices", []))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_tokens = inputs["input_ids"].shape[1]

            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            hf_model.generate(**inputs, max_new_tokens=1, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
            if device == "cuda":
                torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t_start) * 1000

            t_gen = time.perf_counter()
            gen_ids = hf_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
            if device == "cuda":
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t_gen) * 1000 + ttft_ms

            completion_tokens = gen_ids.shape[1] - prompt_tokens
            tps = completion_tokens / (total_ms / 1000) if total_ms > 0 else 0.0
            ttft_running.append(ttft_ms)

            r = LatencyResult(
                sample_id=sample.get("id", f"{sample['task']}_{i:04d}"),
                backend="baseline",
                model=model,
                context_length=sample["approx_tokens"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                total_latency_ms=total_ms,
                tokens_per_second=tps,
            )
            results.append(r)
            log.info("[baseline latency %d/%d] ttft=%.1fms  total=%.1fms  tps=%.1f  ctx=%d tok  (avg_ttft=%.1fms)",
                     i + 1, n, ttft_ms, total_ms, tps, sample["approx_tokens"], np.mean(ttft_running))
            if checkpoint and checkpoint_path and (i + 1) % _CHECKPOINT_INTERVAL == 0:
                _atomic_checkpoint(checkpoint_path, {
                    "n_done": i + 1, "n_total": n,
                    "avg_ttft_ms": round(float(np.mean(ttft_running)), 2),
                    "samples": [{
                        **asdict(r2),
                        "pair_group": samples[j].get("pair_group"),
                        "query_type": "COLD" if samples[j].get("id", "").endswith("_q1") else "WARM",
                        "repo": samples[j].get("repo", ""),
                        "choices": samples[j].get("choices", []),
                        "question": samples[j].get("input", ""),
                    } for j, r2 in enumerate(results)],
                })
                log.debug("  checkpoint saved (%d/%d)", i + 1, n)

    del hf_model
    gc.collect()
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
    checkpoint: bool = True,
    kvboost_max_cache_bytes: int = 4_000_000_000,
    kvboost_recency_window_chunks: int = 8,
    kvboost_recompute_strategy: str = "selective",
    kvboost_chunk_boundary_window: int = 0,
    kvboost_overlap_k: int = 0,
    kvboost_sink_tokens: int = 0,
    vllm_gpu_memory_utilization: float = 0.95,
    vllm_enforce_eager: bool = True,
    vllm_max_num_seqs: int = 1,
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
        checkpoint: Enable mid-run checkpointing
        kvboost_max_cache_bytes: KVBoost max KV cache size in bytes
        kvboost_recency_window_chunks: KVBoost recency window (chunks pinned from eviction)
        kvboost_recompute_strategy: KVBoost recompute strategy ('selective', 'cacheblend', 'none')
        kvboost_chunk_boundary_window: Adaptive boundary splitting window
        kvboost_overlap_k: Overlapping chunk encoding tokens
        kvboost_sink_tokens: Attention sink prefix tokens

    Returns:
        List of LatencyResult objects
    """
    log.info("Starting latency benchmark: backend=%s  model=%s  n=%d  max_ctx=%d  vllm_pc=%s",
             backend, model, n_samples, max_context_tokens,
             vllm_prefix_caching if backend == "vllm_prefixcache" else "N/A")

    samples = _load_longbench_samples(n_samples, max_context_tokens, model_name=model)
    if not samples:
        log.warning("No LongBench samples loaded — check dataset availability")
        return []

    ckpt_path = RESULTS_DIR / f"latency_ckpt_{backend}_{model.replace('/', '_')}.json" if checkpoint else None

    if backend == "kvboost":
        results = _measure_kvboost(
            samples, model,
            max_cache_bytes=kvboost_max_cache_bytes,
            recency_window_chunks=kvboost_recency_window_chunks,
            recompute_strategy=kvboost_recompute_strategy,
            chunk_boundary_window=kvboost_chunk_boundary_window,
            overlap_k=kvboost_overlap_k,
            sink_tokens=kvboost_sink_tokens,
            checkpoint=checkpoint,
            checkpoint_path=ckpt_path,
        )
    elif backend == "vllm_prefixcache":
        results = _measure_vllm_prefixcache(samples, model,
                                             max_context_tokens=max_context_tokens,
                                             enable_prefix_caching=vllm_prefix_caching,
                                             gpu_memory_utilization=vllm_gpu_memory_utilization,
                                             enforce_eager=vllm_enforce_eager,
                                             max_num_seqs=vllm_max_num_seqs,
                                             checkpoint=checkpoint, checkpoint_path=ckpt_path)
    elif backend == "baseline":
        results = _measure_baseline(samples, model,
                                    checkpoint=checkpoint, checkpoint_path=ckpt_path)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    log.info("DONE latency %s: %d samples  avg_ttft=%.1fms  avg_tps=%.1f",
             backend, len(results),
             np.mean([r.ttft_ms for r in results]) if results else 0.0,
             np.mean([r.tokens_per_second for r in results]) if results else 0.0)
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
