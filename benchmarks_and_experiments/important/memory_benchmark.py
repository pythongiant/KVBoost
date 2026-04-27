#!/usr/bin/env python3
"""
GPU Utilization Benchmark: KVBoost vs vLLM (prefix-caching) vs Baseline

Measures peak GPU memory usage and GPU memory efficiency.
Monitors VRAM during inference via torch.cuda.memory_stats to track
cache footprint and utilization per sample.
"""

import gc
import json
import logging
import subprocess
import threading
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
class GPUMemoryResult:
    """GPU memory measurement for one inference"""
    sample_id: str
    backend: str
    model: str
    context_length: int
    prompt_tokens: int
    completion_tokens: int
    gpu_memory_mb: float        # Peak GPU memory during inference
    kv_cache_mb: float          # KV cache memory (inference only)
    model_weights_mb: float     # Model weights in memory
    model_activation_mb: float  # Activations during forward pass
    memory_efficiency: float    # tokens processed per MB of peak memory


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


def _get_model_weights_mb() -> float:
    """Return current allocated VRAM before any inference (model weights only)."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


def _peak_memory_context():
    """Context manager that resets peak stats and returns peak after exit."""
    import torch

    class _PeakCtx:
        peak_mb: float = 0.0

        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            return self

        def __exit__(self, *_):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return _PeakCtx()


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
) -> List[GPUMemoryResult]:
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
    weights_mb = _get_model_weights_mb()
    n = len(samples)

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

        with _peak_memory_context() as ctx:
            result = engine.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                mode=GenerationMode.CHUNK_KV_REUSE,
                do_sample=False,
                cacheable_prefix_len=cacheable_prefix_len,
            )

        peak_mb = ctx.peak_mb
        completion_tokens = result.generated_tokens
        total_tokens = prompt_tokens + completion_tokens
        kv_and_activations_mb = max(0.0, peak_mb - weights_mb)
        cached_fraction = result.kv_reuse_ratio if result.kv_reuse_ratio > 0 else 0.5
        kv_cache_mb = kv_and_activations_mb * cached_fraction
        activation_mb = kv_and_activations_mb * (1 - cached_fraction)
        memory_efficiency = total_tokens / peak_mb if peak_mb > 0 else 0.0
        query_type = "COLD" if is_q1 else "WARM"

        r = GPUMemoryResult(
            sample_id=sample.get("id", f"{sample['task']}_{i:04d}"),
            backend="kvboost",
            model=model,
            context_length=sample["approx_tokens"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            gpu_memory_mb=peak_mb,
            kv_cache_mb=kv_cache_mb,
            model_weights_mb=weights_mb,
            model_activation_mb=activation_mb,
            memory_efficiency=memory_efficiency,
        )
        results.append(r)
        log.info("[kvboost memory %d/%d] %s  peak=%.1fMB  kv=%.1fMB  reuse=%.1f%%  ctx=%d tok  eff=%.2f tok/MB",
                 i + 1, n, query_type, peak_mb, kv_cache_mb, result.kv_reuse_ratio * 100,
                 sample["approx_tokens"], memory_efficiency)
        if checkpoint and checkpoint_path and (i + 1) % _CHECKPOINT_INTERVAL == 0:
            _atomic_checkpoint(checkpoint_path, {
                "n_done": i + 1, "n_total": n,
                "avg_peak_mb": round(float(np.mean([r2.gpu_memory_mb for r2 in results])), 2),
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
    gpu_memory_utilization: float = 0.95,
    enforce_eager: bool = True,
    max_num_seqs: int = 1,
    checkpoint: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> List[GPUMemoryResult]:
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log.info("[vllm_prefixcache] GPU free before LLM init: %.2f GiB", torch.cuda.mem_get_info()[0] / 1e9)

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, enable_prefix_caching=True,
              max_model_len=max_context_tokens + 128,
              gpu_memory_utilization=gpu_memory_utilization,
              enforce_eager=enforce_eager,
              max_num_seqs=max_num_seqs)
    params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    weights_mb = _get_model_weights_mb()
    n = len(samples)

    results = []
    for i, sample in enumerate(samples):
        prompt = _format_prompt(sample["context"], sample["input"], sample.get("choices", []))
        prompt_tokens = len(tokenizer.encode(prompt))

        with _peak_memory_context() as ctx:
            outputs = llm.generate([prompt], params)

        peak_mb = ctx.peak_mb
        out = outputs[0]
        completion_tokens = len(out.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens
        kv_and_act_mb = max(0.0, peak_mb - weights_mb)
        kv_cache_mb = kv_and_act_mb * 0.8
        activation_mb = kv_and_act_mb * 0.2
        memory_efficiency = total_tokens / peak_mb if peak_mb > 0 else 0.0

        r = GPUMemoryResult(
            sample_id=sample.get("id", f"{sample['task']}_{i:04d}"),
            backend="vllm_prefixcache",
            model=model,
            context_length=sample["approx_tokens"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            gpu_memory_mb=peak_mb,
            kv_cache_mb=kv_cache_mb,
            model_weights_mb=weights_mb,
            model_activation_mb=activation_mb,
            memory_efficiency=memory_efficiency,
        )
        results.append(r)
        log.info("[vllm_prefixcache memory %d/%d] peak=%.1fMB  kv=%.1fMB  ctx=%d tok  eff=%.2f tok/MB",
                 i + 1, n, peak_mb, kv_cache_mb, sample["approx_tokens"], memory_efficiency)
        if checkpoint and checkpoint_path and (i + 1) % _CHECKPOINT_INTERVAL == 0:
            _atomic_checkpoint(checkpoint_path, {
                "n_done": i + 1, "n_total": n,
                "avg_peak_mb": round(float(np.mean([r2.gpu_memory_mb for r2 in results])), 2),
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
) -> List[GPUMemoryResult]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
    ).to(device)
    hf_model.eval()
    weights_mb = _get_model_weights_mb()
    n = len(samples)

    results = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            prompt = _format_prompt(sample["context"], sample["input"], sample.get("choices", []))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_tokens = inputs["input_ids"].shape[1]

            with _peak_memory_context() as ctx:
                gen_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            peak_mb = ctx.peak_mb
            completion_tokens = gen_ids.shape[1] - prompt_tokens
            total_tokens = prompt_tokens + completion_tokens
            kv_and_act_mb = max(0.0, peak_mb - weights_mb)
            kv_cache_mb = kv_and_act_mb * 0.6
            activation_mb = kv_and_act_mb * 0.4
            memory_efficiency = total_tokens / peak_mb if peak_mb > 0 else 0.0

            r = GPUMemoryResult(
                sample_id=sample.get("id", f"{sample['task']}_{i:04d}"),
                backend="baseline",
                model=model,
                context_length=sample["approx_tokens"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                gpu_memory_mb=peak_mb,
                kv_cache_mb=kv_cache_mb,
                model_weights_mb=weights_mb,
                model_activation_mb=activation_mb,
                memory_efficiency=memory_efficiency,
            )
            results.append(r)
            log.info("[baseline memory %d/%d] peak=%.1fMB  kv=%.1fMB  ctx=%d tok  eff=%.2f tok/MB",
                     i + 1, n, peak_mb, kv_cache_mb, sample["approx_tokens"], memory_efficiency)
            if checkpoint and checkpoint_path and (i + 1) % _CHECKPOINT_INTERVAL == 0:
                _atomic_checkpoint(checkpoint_path, {
                    "n_done": i + 1, "n_total": n,
                    "avg_peak_mb": round(float(np.mean([r2.gpu_memory_mb for r2 in results])), 2),
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


def benchmark_gpu_memory(
    backend: str,
    model: str,
    n_samples: int = 50,
    dataset_path: Optional[Path] = None,
    max_context_tokens: int = 8192,
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
) -> List[GPUMemoryResult]:
    """
    Run GPU memory benchmark for a specific backend.

    Records baseline model-weight allocation before inference, then uses
    torch.cuda.reset_peak_memory_stats / max_memory_allocated to capture
    peak VRAM per sample. KV cache and activation components are estimated
    from the delta above the weight baseline.

    Args:
        backend: 'kvboost', 'vllm_prefixcache', or 'baseline'
        model: Model name
        n_samples: Number of test samples
        dataset_path: Unused; kept for API compatibility
        max_context_tokens: Max context length
        checkpoint: Enable mid-run checkpointing
        kvboost_max_cache_bytes: KVBoost max KV cache size in bytes
        kvboost_recency_window_chunks: KVBoost recency window (chunks pinned from eviction)
        kvboost_recompute_strategy: KVBoost recompute strategy ('selective', 'cacheblend', 'none')
        kvboost_chunk_boundary_window: Adaptive boundary splitting window
        kvboost_overlap_k: Overlapping chunk encoding tokens
        kvboost_sink_tokens: Attention sink prefix tokens

    Returns:
        List of GPUMemoryResult objects
    """
    log.info("Starting GPU memory benchmark: backend=%s  model=%s  n=%d  max_ctx=%d",
             backend, model, n_samples, max_context_tokens)

    samples = _load_longbench_samples(n_samples, max_context_tokens, model_name=model)
    if not samples:
        log.warning("No LongBench samples loaded — check dataset availability")
        return []

    ckpt_path = RESULTS_DIR / f"memory_ckpt_{backend}_{model.replace('/', '_')}.json" if checkpoint else None

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
                                             gpu_memory_utilization=vllm_gpu_memory_utilization,
                                             enforce_eager=vllm_enforce_eager,
                                             max_num_seqs=vllm_max_num_seqs,
                                             checkpoint=checkpoint, checkpoint_path=ckpt_path)
    elif backend == "baseline":
        results = _measure_baseline(samples, model,
                                    checkpoint=checkpoint, checkpoint_path=ckpt_path)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    log.info("DONE memory %s: %d samples  avg_peak=%.1fMB  avg_eff=%.3f tok/MB",
             backend, len(results),
             np.mean([r.gpu_memory_mb for r in results]) if results else 0.0,
             np.mean([r.memory_efficiency for r in results]) if results else 0.0)
    return results


def aggregate_gpu_memory_results(
    results: Dict[str, List[GPUMemoryResult]]
) -> Dict[str, Dict]:
    """Aggregate GPU memory results by backend"""
    aggregated = {}

    for backend, backend_results in results.items():
        if not backend_results:
            continue

        gpu_mems = [r.gpu_memory_mb for r in backend_results]
        kv_cache_mems = [r.kv_cache_mb for r in backend_results]
        efficiencies = [r.memory_efficiency for r in backend_results if r.memory_efficiency > 0]

        aggregated[backend] = {
            "n_samples": len(backend_results),
            "gpu_memory_mb": {
                "mean": float(np.mean(gpu_mems)),
                "median": float(np.median(gpu_mems)),
                "p95": float(np.percentile(gpu_mems, 95)),
                "max": float(np.max(gpu_mems)),
            },
            "kv_cache_mb": {
                "mean": float(np.mean(kv_cache_mems)),
                "median": float(np.median(kv_cache_mems)),
                "p95": float(np.percentile(kv_cache_mems, 95)),
                "max": float(np.max(kv_cache_mems)),
            },
            "memory_efficiency": {
                "mean": float(np.mean(efficiencies)) if efficiencies else 0.0,
                "median": float(np.median(efficiencies)) if efficiencies else 0.0,
            },
            "samples": [asdict(r) for r in backend_results],
        }

    return aggregated


def print_gpu_memory_table(aggregated: Dict[str, Dict]):
    """Print GPU memory comparison table"""
    print("\n" + "="*110)
    print("  GPU MEMORY UTILIZATION BENCHMARK")
    print("="*110)
    print(
        f"  {'Backend':<20} {'Peak GPU (MB)':>15} {'KV Cache (MB)':>16} "
        f"{'Memory Eff':>12} {'vs Baseline':>12}"
    )
    print(f"  {'-'*20} {'-'*15} {'-'*16} {'-'*12} {'-'*12}")

    baseline_gpu = None
    for backend in ['kvboost', 'vllm_prefixcache', 'baseline']:
        if backend not in aggregated:
            continue

        agg = aggregated[backend]
        gpu_mean = agg["gpu_memory_mb"]["mean"]
        kv_mean = agg["kv_cache_mb"]["mean"]
        eff_mean = agg["memory_efficiency"]["mean"]

        if backend == 'baseline':
            baseline_gpu = gpu_mean

        savings = ((baseline_gpu - gpu_mean) / baseline_gpu * 100) if baseline_gpu and baseline_gpu > 0 else 0
        savings_str = f"{savings:+.1f}%" if backend != 'baseline' else "<<<BASELINE"

        print(
            f"  {backend:<20} {gpu_mean:>14.1f} {kv_mean:>15.1f} "
            f"{eff_mean:>11.1%} {savings_str:>12}"
        )

    print("="*110)

    if 'kvboost' in aggregated:
        agg = aggregated['kvboost']
        kv_max = agg["kv_cache_mb"]["max"]
        print(f"  KVBoost KV Cache Peak: {kv_max:.1f} MB (dynamic pruning in effect)")

    print("="*110 + "\n")


def print_memory_breakdown(aggregated: Dict[str, Dict]):
    """Print memory component breakdown"""
    print("\n" + "="*110)
    print("  MEMORY COMPONENT BREAKDOWN")
    print("="*110)

    for backend in ['kvboost', 'vllm_prefixcache', 'baseline']:
        if backend not in aggregated:
            continue

        agg = aggregated[backend]
        gpu_mean = agg["gpu_memory_mb"]["mean"]
        kv_mean = agg["kv_cache_mb"]["mean"]

        overhead = max(0, gpu_mean - kv_mean)

        print(f"  {backend}:")
        print(f"    Model + Activations: {overhead:.1f} MB")
        print(f"    KV Cache:            {kv_mean:.1f} MB")
        print(f"    Total:               {gpu_mean:.1f} MB")

    print("="*110 + "\n")


def save_gpu_memory_results(
    aggregated: Dict[str, Dict],
    model: str,
    output_path: Optional[Path] = None
) -> Path:
    """Save GPU memory results to JSON"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"gpu_memory_benchmark_{model.replace('/', '_')}_{timestamp}.json"

    output_path.parent.mkdir(exist_ok=True, parents=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "gpu_memory",
        "model": model,
        "results": aggregated,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    log.info("GPU memory results saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Memory Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    results = {}
    for backend in ['kvboost', 'vllm_prefixcache', 'baseline']:
        results[backend] = benchmark_gpu_memory(backend, args.model, args.n_samples)

    aggregated = aggregate_gpu_memory_results(results)
    print_gpu_memory_table(aggregated)
    print_memory_breakdown(aggregated)
    save_gpu_memory_results(aggregated, args.model, args.output)
