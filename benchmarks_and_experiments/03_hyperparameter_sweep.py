#!/usr/bin/env python3
"""
Experiment 3: Hyperparameter Sweep
====================================
Systematically varies chunk_size and recompute_overlap to find the
Pareto frontier of latency vs. quality vs. memory.

Sweeps:
  - chunk_size:         32, 64, 128, 256, 512
  - recompute_overlap:  0, 4, 8, 16, 32

For each (chunk_size, overlap) pair, measures:
  - TTFT (latency)
  - KV memory usage
  - Cache hit rate
  - Output match rate vs. baseline (quality proxy)

Usage:
  python 03_hyperparameter_sweep.py
  python 03_hyperparameter_sweep.py --chunk-sizes 64,128,256 --overlaps 0,8,16
"""

from __future__ import annotations

import argparse
import gc

import torch

from utils import (
    get_logger, save_results, compute_stats,
    SYSTEM_PROMPT, RAG_DOCUMENT, QUERIES, RAG_QUERIES,
    GenerationMode,
)

from kvboost import KVBoost as InferenceEngine

log = get_logger("03_hyperparameter_sweep")

DEFAULT_CHUNK_SIZES = [32, 64, 128, 256, 512]
DEFAULT_OVERLAPS = [0, 4, 8, 16, 32]

WORKLOADS = {
    "system_prompt": (SYSTEM_PROMPT, QUERIES[:3]),
    "rag_document": (RAG_DOCUMENT, RAG_QUERIES[:3]),
}


def get_baseline_outputs(engine, prefix: str, queries: list, max_new_tokens: int) -> list:
    """Get baseline outputs for quality comparison."""
    outputs = []
    for query in queries:
        prompt = prefix + "\n\n" + query
        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=GenerationMode.BASELINE, do_sample=False)
        outputs.append(r.output_text)
    return outputs


def run_config(model_name: str, chunk_size: int, overlap: int,
               runs: int, max_new_tokens: int) -> dict:
    """Run one (chunk_size, overlap) configuration."""
    log.info("  chunk_size=%d, overlap=%d", chunk_size, overlap)

    # Overlap can't exceed chunk_size
    effective_overlap = min(overlap, chunk_size - 1)

    try:
        engine = InferenceEngine.from_pretrained(
            model_name=model_name,
            chunk_size=chunk_size,
            recompute_overlap=effective_overlap,
            max_chunks=256,
        )
    except Exception as e:
        return {"error": str(e)}

    config_results = {
        "chunk_size": chunk_size,
        "recompute_overlap": effective_overlap,
        "workloads": {},
    }

    for wname, (prefix, queries) in WORKLOADS.items():
        # Get baseline outputs for comparison
        baseline_outputs = get_baseline_outputs(engine, prefix, queries, max_new_tokens)

        # Warm cache
        engine.warm_chunks(prefix, position_offset=0)

        ttfts, match_rates = [], []
        for run_i in range(runs):
            for qi, query in enumerate(queries):
                prompt = prefix + "\n\n" + query
                r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                    mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)
                if run_i > 0 or runs == 1:
                    ttfts.append(r.ttft_ms)
                    match_rates.append(1.0 if r.output_text == baseline_outputs[qi] else 0.0)

        stats = engine.cache_stats()
        config_results["workloads"][wname] = {
            "ttft_ms": compute_stats(ttfts),
            "output_match_rate": round(sum(match_rates) / max(len(match_rates), 1), 3),
            "cache_hit_rate": stats["hit_rate"],
            "hot_memory_mb": stats["hot_memory_mb"],
            "hot_chunks": stats["hot_chunks"],
            "n_samples": len(ttfts),
        }

    # Memory per chunk
    if engine.cache_manager._hot:
        first_chunk = next(iter(engine.cache_manager._hot.values()))
        config_results["memory_per_chunk_mb"] = round(first_chunk.memory_bytes() / 1e6, 2)
    else:
        config_results["memory_per_chunk_mb"] = 0

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return config_results


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Hyperparameter sweep")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-sizes", type=str,
                        default=",".join(str(x) for x in DEFAULT_CHUNK_SIZES))
    parser.add_argument("--overlaps", type=str,
                        default=",".join(str(x) for x in DEFAULT_OVERLAPS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=str, default="03_hyperparameter_sweep.json")
    args = parser.parse_args()

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    overlaps = [int(x) for x in args.overlaps.split(",")]

    all_results = []

    for cs in chunk_sizes:
        for ov in overlaps:
            if ov >= cs:
                log.info("  Skipping overlap=%d >= chunk_size=%d", ov, cs)
                continue
            result = run_config(args.model, cs, ov, args.runs, args.max_new_tokens)
            all_results.append(result)

    # Print Pareto table
    print("\n" + "=" * 100)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 100)
    print(f"{'ChunkSz':<10} {'Overlap':<10} {'TTFT(ms)':<12} {'Match%':<10} "
          f"{'HitRate':<10} {'MemMB':<10} {'MB/chunk':<10}")
    print("-" * 100)

    for r in all_results:
        if "error" in r:
            print(f"{r.get('chunk_size', '?'):<10} {r.get('recompute_overlap', '?'):<10} ERROR")
            continue
        # Average across workloads
        ttfts = []
        matches = []
        for wdata in r["workloads"].values():
            ttfts.append(wdata["ttft_ms"].get("mean", 0))
            matches.append(wdata["output_match_rate"])
        avg_ttft = sum(ttfts) / max(len(ttfts), 1)
        avg_match = sum(matches) / max(len(matches), 1)
        first_w = list(r["workloads"].values())[0] if r["workloads"] else {}

        print(
            f"{r['chunk_size']:<10} {r['recompute_overlap']:<10} "
            f"{avg_ttft:<12.1f} {avg_match * 100:<10.1f} "
            f"{first_w.get('cache_hit_rate', 0):<10.3f} "
            f"{first_w.get('hot_memory_mb', 0):<10.1f} "
            f"{r.get('memory_per_chunk_mb', 0):<10.2f}"
        )

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
