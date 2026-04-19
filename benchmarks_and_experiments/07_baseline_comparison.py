#!/usr/bin/env python3
"""
Experiment 7: Comparison Against Existing Systems
===================================================
Compares our chunk-level KV reuse against:

  1. HuggingFace native (no caching) — already our baseline
  2. vLLM automatic prefix caching (if available)
  3. Our own PREFIX_CACHE mode (ablation)
  4. Feature comparison table (qualitative)

For vLLM/SGLang, we measure via their APIs if installed.
If not installed, we provide the benchmark framework and
a qualitative comparison table.

Usage:
  python 07_baseline_comparison.py
  python 07_baseline_comparison.py --include-vllm --vllm-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

from utils import (
    get_logger, load_engine, save_results, compute_stats,
    SYSTEM_PROMPT, RAG_DOCUMENT, QUERIES, RAG_QUERIES,
    GenerationMode,
)

log = get_logger("07_baseline_comparison")


def check_vllm_available() -> bool:
    try:
        import vllm
        return True
    except ImportError:
        return False


def check_sglang_available() -> bool:
    try:
        import sglang
        return True
    except ImportError:
        return False


def benchmark_our_system(model_name: str, chunk_size: int,
                         runs: int, max_new_tokens: int) -> Dict:
    """Benchmark all three of our modes."""
    engine = load_engine(model_name=model_name, chunk_size=chunk_size)

    workloads = {
        "system_prompt": (SYSTEM_PROMPT, QUERIES[:3]),
        "rag_document": (RAG_DOCUMENT, RAG_QUERIES[:3]),
    }

    results = {}
    modes = [GenerationMode.BASELINE, GenerationMode.PREFIX_CACHE, GenerationMode.CHUNK_KV_REUSE]

    for wname, (prefix, queries) in workloads.items():
        engine.warm_chunks(prefix, position_offset=0)
        results[wname] = {}

        for mode in modes:
            ttfts, totals = [], []
            outputs = []

            for run_i in range(runs):
                for query in queries:
                    prompt = prefix + "\n\n" + query
                    r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                        mode=mode, do_sample=False)
                    if run_i > 0 or runs == 1:
                        ttfts.append(r.ttft_ms)
                        totals.append(r.total_ms)
                    if run_i == 0:
                        outputs.append(r.output_text[:80])

            results[wname][mode.value] = {
                "ttft_ms": compute_stats(ttfts),
                "total_ms": compute_stats(totals),
                "n_samples": len(ttfts),
                "sample_outputs": outputs,
            }

    return results


def benchmark_vllm(model_name: str, queries: List[str],
                   prefix: str, max_new_tokens: int, runs: int) -> Optional[Dict]:
    """Benchmark vLLM's automatic prefix caching."""
    if not check_vllm_available():
        log.info("vLLM not installed, skipping vLLM benchmark")
        log.info("  Install: pip install vllm")
        return None

    from vllm import LLM, SamplingParams

    log.info("Loading vLLM with model=%s", model_name)
    llm = LLM(
        model=model_name,
        enable_prefix_caching=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
    )

    ttfts = []
    for run_i in range(runs):
        for query in queries:
            prompt = prefix + "\n\n" + query
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            elapsed = (time.perf_counter() - t0) * 1000
            # vLLM doesn't directly expose TTFT, use total time as proxy
            if run_i > 0 or runs == 1:
                ttfts.append(elapsed)

    return {
        "ttft_ms": compute_stats(ttfts),
        "note": "vLLM total generation time (TTFT not directly exposed)",
        "n_samples": len(ttfts),
        "prefix_caching": True,
    }


def benchmark_vllm_no_prefix_cache(model_name: str, queries: List[str],
                                    prefix: str, max_new_tokens: int,
                                    runs: int) -> Optional[Dict]:
    """Benchmark vLLM without prefix caching for comparison."""
    if not check_vllm_available():
        return None

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        enable_prefix_caching=False,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    ttfts = []
    for run_i in range(runs):
        for query in queries:
            prompt = prefix + "\n\n" + query
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            elapsed = (time.perf_counter() - t0) * 1000
            if run_i > 0 or runs == 1:
                ttfts.append(elapsed)

    return {
        "ttft_ms": compute_stats(ttfts),
        "note": "vLLM without prefix caching",
        "n_samples": len(ttfts),
        "prefix_caching": False,
    }


def qualitative_comparison() -> Dict:
    """
    Feature comparison table between systems.
    This is always generated regardless of whether external systems are installed.
    """
    return {
        "features": [
            {
                "feature": "Prefix caching (exact shared prefix)",
                "our_system": True,
                "vllm": True,
                "sglang_radix": True,
                "notes": "All systems support basic prefix caching",
            },
            {
                "feature": "Non-prefix chunk reuse",
                "our_system": True,
                "vllm": False,
                "sglang_radix": "partial",
                "notes": "Our system can reuse arbitrary chunks, not just leading prefix. "
                         "SGLang RadixAttention supports subtree matching.",
            },
            {
                "feature": "Selective boundary recompute",
                "our_system": True,
                "vllm": False,
                "sglang_radix": False,
                "notes": "Unique to our approach. Fixes KV staleness at chunk seams.",
            },
            {
                "feature": "Content-addressable cache keys",
                "our_system": True,
                "vllm": "hash-based",
                "sglang_radix": "trie-based",
                "notes": "We use SHA256 of token bytes. vLLM uses block hashing. "
                         "SGLang uses radix trie on token sequences.",
            },
            {
                "feature": "Disk-backed cold storage",
                "our_system": True,
                "vllm": False,
                "sglang_radix": False,
                "notes": "Our two-tier storage allows larger effective cache sizes.",
            },
            {
                "feature": "Semantic chunking",
                "our_system": True,
                "vllm": False,
                "sglang_radix": False,
                "notes": "Our system supports paragraph-boundary chunking for better reuse.",
            },
            {
                "feature": "Continuous batching",
                "our_system": False,
                "vllm": True,
                "sglang_radix": True,
                "notes": "Production serving systems support in-flight batching.",
            },
            {
                "feature": "PagedAttention",
                "our_system": False,
                "vllm": True,
                "sglang_radix": True,
                "notes": "Memory-efficient KV storage via paging. Orthogonal to our approach.",
            },
            {
                "feature": "Multi-GPU / tensor parallel",
                "our_system": False,
                "vllm": True,
                "sglang_radix": True,
                "notes": "Our system is single-device. Could be extended.",
            },
            {
                "feature": "Quantized KV cache",
                "our_system": False,
                "vllm": True,
                "sglang_radix": False,
                "notes": "FP8/INT8 KV cache reduces memory 2x. Complementary to our approach.",
            },
        ],
        "summary": (
            "Our system's key differentiator is chunk-level reuse with selective "
            "boundary recomputation, enabling cache hits for non-prefix shared content "
            "(e.g., RAG documents that appear at different positions). This is complementary "
            "to, not competitive with, production serving optimizations like PagedAttention "
            "and continuous batching. The ideal production system would combine chunk-level "
            "KV reuse with PagedAttention for memory efficiency."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 7: Baseline comparison"
    )
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--include-vllm", action="store_true",
                        help="Include vLLM benchmarks (requires vllm installed + GPU)")
    parser.add_argument("--output", type=str, default="07_baseline_comparison.json")
    args = parser.parse_args()

    all_results = {}

    # Our system
    log.info("=" * 60)
    log.info("Benchmarking our system")
    log.info("=" * 60)
    all_results["our_system"] = benchmark_our_system(
        args.model, args.chunk_size, args.runs, args.max_new_tokens
    )

    # vLLM (optional)
    if args.include_vllm:
        log.info("=" * 60)
        log.info("Benchmarking vLLM")
        log.info("=" * 60)
        vllm_with = benchmark_vllm(
            args.model, QUERIES[:3], SYSTEM_PROMPT, args.max_new_tokens, args.runs
        )
        vllm_without = benchmark_vllm_no_prefix_cache(
            args.model, QUERIES[:3], SYSTEM_PROMPT, args.max_new_tokens, args.runs
        )
        all_results["vllm"] = {
            "with_prefix_caching": vllm_with,
            "without_prefix_caching": vllm_without,
        }
    else:
        all_results["vllm"] = {
            "status": "skipped (pass --include-vllm to enable)",
            "install": "pip install vllm (requires CUDA GPU)",
        }

    # Qualitative comparison (always included)
    all_results["qualitative_comparison"] = qualitative_comparison()

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 80)

    our = all_results["our_system"]
    for wname, modes in our.items():
        print(f"\n--- {wname} ---")
        print(f"{'Mode':<20} {'TTFT(ms)':<15} {'Speedup':<10}")
        print("-" * 45)
        baseline_ttft = modes.get("baseline", {}).get("ttft_ms", {}).get("mean", 1)
        for mode in ["baseline", "prefix_cache", "chunk_kv_reuse"]:
            if mode in modes:
                ttft = modes[mode]["ttft_ms"].get("mean", 0)
                speedup = baseline_ttft / max(ttft, 0.01)
                tag = f"{speedup:.1f}x" if mode != "baseline" else "-"
                print(f"{mode:<20} {ttft:<15.1f} {tag:<10}")

    print("\n--- Feature Comparison ---")
    for feat in all_results["qualitative_comparison"]["features"]:
        ours = "Y" if feat["our_system"] is True else ("N" if feat["our_system"] is False else feat["our_system"])
        vllm = "Y" if feat["vllm"] is True else ("N" if feat["vllm"] is False else feat["vllm"])
        sgl = "Y" if feat["sglang_radix"] is True else ("N" if feat["sglang_radix"] is False else feat["sglang_radix"])
        print(f"  {feat['feature']:<40} Ours={ours:<6} vLLM={vllm:<10} SGLang={sgl}")

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
