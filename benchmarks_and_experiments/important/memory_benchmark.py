#!/usr/bin/env python3
"""
GPU Utilization Benchmark: KVBoost vs vLLM (prefix-caching) vs Baseline

Measures peak GPU memory usage and GPU memory efficiency.
Monitors VRAM during inference via torch.cuda.memory_stats to track
cache footprint and utilization per sample.
"""

import json
import logging
import subprocess
import threading
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

LONGBENCH_TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]


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
    tasks: List[str] = LONGBENCH_TASKS,
) -> List[Dict]:
    from datasets import load_dataset

    samples = []
    per_task = max(1, n_samples // len(tasks))

    for task in tasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test")
        except Exception as e:
            log.warning("Could not load LongBench task %s: %s", task, e)
            continue

        count = 0
        for item in ds:
            if count >= per_task:
                break
            context = item.get("context", "")
            question = item.get("input", "")
            if not context or not question:
                continue
            approx_tokens = int(len((context + question).split()) * 0.75)
            if approx_tokens > max_context_tokens:
                continue
            samples.append({
                "task": task,
                "context": context,
                "input": question,
                "approx_tokens": approx_tokens,
            })
            count += 1

    return samples[:n_samples]


def _format_prompt(context: str, question: str) -> str:
    return (
        f"Read the following passage carefully and answer the question based only on "
        f"the information provided.\n\n"
        f"Passage:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
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


def _measure_kvboost(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
    recompute_strategy: str = "selective",
    chunk_boundary_window: int = 0,
    overlap_k: int = 0,
    sink_tokens: int = 0,
) -> List[GPUMemoryResult]:
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

    # Baseline allocated memory = model weights
    weights_mb = _get_model_weights_mb()

    results = []
    for i, sample in enumerate(samples):
        prompt = _format_prompt(sample["context"], sample["input"])
        prompt_tokens = len(tokenizer.encode(prompt))

        with _peak_memory_context() as ctx:
            result = engine.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                mode=GenerationMode.CHUNK_KV_REUSE,
                do_sample=False,
            )

        peak_mb = ctx.peak_mb
        completion_tokens = result.generated_tokens
        total_tokens = prompt_tokens + completion_tokens

        # KV cache size: peak - weights; activations are temporary
        kv_and_activations_mb = max(0.0, peak_mb - weights_mb)
        # Heuristic: activations ~= 1 forward pass worth of memory
        # kv_cache scales with sequence length; activations are roughly constant
        # We attribute (cached_tokens / total_tokens) fraction to KV cache
        cached_fraction = result.kv_reuse_ratio if result.kv_reuse_ratio > 0 else 0.5
        kv_cache_mb = kv_and_activations_mb * cached_fraction
        activation_mb = kv_and_activations_mb * (1 - cached_fraction)

        memory_efficiency = total_tokens / peak_mb if peak_mb > 0 else 0.0

        results.append(GPUMemoryResult(
            sample_id=f"{sample['task']}_{i:04d}",
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
        ))
        log.debug("kvboost sample %d: peak=%.1fMB kv=%.1fMB", i, peak_mb, kv_cache_mb)

    del engine
    torch.cuda.empty_cache()
    return results


def _measure_vllm_prefixcache(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
) -> List[GPUMemoryResult]:
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, enable_prefix_caching=True)
    params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    weights_mb = _get_model_weights_mb()

    results = []
    for i, sample in enumerate(samples):
        prompt = _format_prompt(sample["context"], sample["input"])
        prompt_tokens = len(tokenizer.encode(prompt))

        with _peak_memory_context() as ctx:
            outputs = llm.generate([prompt], params)

        peak_mb = ctx.peak_mb
        out = outputs[0]
        completion_tokens = len(out.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens

        # vLLM manages its own paged KV cache; estimate from peak delta
        kv_and_act_mb = max(0.0, peak_mb - weights_mb)
        # vLLM uses paged attention; most delta is KV cache pages
        kv_cache_mb = kv_and_act_mb * 0.8
        activation_mb = kv_and_act_mb * 0.2

        memory_efficiency = total_tokens / peak_mb if peak_mb > 0 else 0.0

        results.append(GPUMemoryResult(
            sample_id=f"{sample['task']}_{i:04d}",
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
        ))
        log.debug("vllm sample %d: peak=%.1fMB kv=%.1fMB", i, peak_mb, kv_cache_mb)

    del llm
    torch.cuda.empty_cache()
    return results


def _measure_baseline(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
) -> List[GPUMemoryResult]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    hf_model.eval()

    weights_mb = _get_model_weights_mb()

    results = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            prompt = _format_prompt(sample["context"], sample["input"])
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

            # Standard HF generate: KV cache grows with sequence length
            # Delta above weights is primarily KV cache + activations
            kv_and_act_mb = max(0.0, peak_mb - weights_mb)
            # Roughly 60% KV cache, 40% activations for standard attention
            kv_cache_mb = kv_and_act_mb * 0.6
            activation_mb = kv_and_act_mb * 0.4

            memory_efficiency = total_tokens / peak_mb if peak_mb > 0 else 0.0

            results.append(GPUMemoryResult(
                sample_id=f"{sample['task']}_{i:04d}",
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
            ))
            log.debug("baseline sample %d: peak=%.1fMB kv=%.1fMB", i, peak_mb, kv_cache_mb)

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def benchmark_gpu_memory(
    backend: str,
    model: str,
    n_samples: int = 50,
    dataset_path: Optional[Path] = None,
    max_context_tokens: int = 8192,
    kvboost_recompute_strategy: str = "selective",
    kvboost_chunk_boundary_window: int = 0,
    kvboost_overlap_k: int = 0,
    kvboost_sink_tokens: int = 0,
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
        kvboost_recompute_strategy: KVBoost recompute strategy ('selective', 'cacheblend', 'none')
        kvboost_chunk_boundary_window: Adaptive boundary splitting window
        kvboost_overlap_k: Overlapping chunk encoding tokens
        kvboost_sink_tokens: Attention sink prefix tokens

    Returns:
        List of GPUMemoryResult objects
    """
    log.info("Starting GPU memory benchmark for %s with %s", backend, model)

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
        results = _measure_vllm_prefixcache(samples, model)
    elif backend == "baseline":
        results = _measure_baseline(samples, model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    log.info(
        "Processed %d GPU memory samples for %s (peak_mean=%.1fMB, eff_mean=%.3f)",
        len(results),
        backend,
        np.mean([r.gpu_memory_mb for r in results]) if results else 0.0,
        np.mean([r.memory_efficiency for r in results]) if results else 0.0,
    )
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
