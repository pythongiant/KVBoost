#!/usr/bin/env python3
"""
Experiment 6: Memory Analysis
===============================
Measures the memory cost of KV caching and analyzes the
memory-vs-speedup trade-off:

  1. Memory per chunk at different chunk sizes
  2. Total memory as max_chunks varies
  3. Break-even analysis: when does caching save more than it costs?
  4. Disk tier latency vs. recomputation cost

Usage:
  python 06_memory_analysis.py
  python 06_memory_analysis.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import gc
import tempfile
import time
from typing import Dict

import torch

from utils import (
    get_logger, save_results, compute_stats,
    SYSTEM_PROMPT, RAG_DOCUMENT, QUERIES, RAG_QUERIES,
    GenerationMode,
)

from kvboost import KVBoost as InferenceEngine

log = get_logger("06_memory_analysis")


def measure_memory_per_chunk(model_name: str, chunk_sizes: list) -> Dict:
    """Measure KV memory consumption per chunk at different sizes."""
    log.info("=== Memory per chunk ===")
    results = []

    for cs in chunk_sizes:
        engine = InferenceEngine.from_pretrained(
            model_name=model_name, chunk_size=cs, max_chunks=8,
        )
        engine.warm_chunks(SYSTEM_PROMPT, position_offset=0)

        stats = engine.cache_stats()
        n_chunks = stats["hot_chunks"]
        total_mb = stats["hot_memory_mb"]
        per_chunk_mb = total_mb / max(n_chunks, 1)

        # Get model config for theoretical calculation
        cfg = engine.model.config
        n_layers = cfg.num_hidden_layers
        n_heads = cfg.num_attention_heads
        head_dim = cfg.hidden_size // n_heads
        # Theoretical: n_layers * 2 * batch * heads * seq * head_dim * bytes
        theoretical_bytes = n_layers * 2 * 1 * n_heads * cs * head_dim * 2  # float16
        theoretical_mb = theoretical_bytes / 1e6

        results.append({
            "chunk_size": cs,
            "actual_mb_per_chunk": round(per_chunk_mb, 3),
            "theoretical_mb_per_chunk": round(theoretical_mb, 3),
            "total_hot_mb": round(total_mb, 3),
            "hot_chunks": n_chunks,
            "model_layers": n_layers,
            "model_heads": n_heads,
            "head_dim": head_dim,
        })
        log.info("  chunk_size=%d: %.2fMB/chunk (theoretical %.2fMB), %d chunks",
                 cs, per_chunk_mb, theoretical_mb, n_chunks)

        del engine
        gc.collect()

    return results


def measure_total_memory_scaling(model_name: str, chunk_size: int,
                                 max_chunks_values: list) -> Dict:
    """Measure total memory as cache capacity (max_chunks) increases."""
    log.info("=== Total memory scaling ===")

    # Build a long text to fill the cache
    long_text = (SYSTEM_PROMPT + "\n\n" + RAG_DOCUMENT + "\n\n") * 5
    results = []

    for max_chunks in max_chunks_values:
        engine = InferenceEngine.from_pretrained(
            model_name=model_name, chunk_size=chunk_size,
            max_chunks=max_chunks,
        )
        engine.warm_chunks(long_text, position_offset=0)

        stats = engine.cache_stats()
        results.append({
            "max_chunks": max_chunks,
            "actual_hot_chunks": stats["hot_chunks"],
            "hot_memory_mb": stats["hot_memory_mb"],
        })
        log.info("  max_chunks=%d: %d stored, %.1fMB",
                 max_chunks, stats["hot_chunks"], stats["hot_memory_mb"])

        del engine
        gc.collect()

    return results


def measure_break_even(model_name: str, chunk_size: int) -> Dict:
    """
    Find the break-even point: how many cache hits justify the memory cost.
    Compare cumulative time saved vs. memory occupied.
    """
    log.info("=== Break-even analysis ===")

    engine = InferenceEngine.from_pretrained(
        model_name=model_name, chunk_size=chunk_size, max_chunks=256,
    )

    prefix = SYSTEM_PROMPT
    queries = QUERIES

    # Measure baseline prefill cost
    baseline_ttfts = []
    for query in queries:
        prompt = prefix + "\n\n" + query
        r = engine.generate(prompt, max_new_tokens=16,
                            mode=GenerationMode.BASELINE, do_sample=False)
        baseline_ttfts.append(r.ttft_ms)
    avg_baseline_ttft = sum(baseline_ttfts) / len(baseline_ttfts)

    # Warm cache
    engine.warm_chunks(prefix, position_offset=0)

    # Measure cached prefill cost
    cached_ttfts = []
    for query in queries:
        prompt = prefix + "\n\n" + query
        r = engine.generate(prompt, max_new_tokens=16,
                            mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)
        cached_ttfts.append(r.ttft_ms)
    avg_cached_ttft = sum(cached_ttfts) / len(cached_ttfts)

    # Time saved per request
    time_saved_per_request = avg_baseline_ttft - avg_cached_ttft

    # Memory cost
    stats = engine.cache_stats()
    cache_memory_mb = stats["hot_memory_mb"]

    # Initial cache warmup cost (time to compute and store all chunks)
    t0 = time.perf_counter()
    engine2 = InferenceEngine.from_pretrained(
        model_name=model_name, chunk_size=chunk_size, max_chunks=256,
    )
    engine2.warm_chunks(prefix, position_offset=0)
    warmup_ms = (time.perf_counter() - t0) * 1000

    # Break-even: requests needed to recoup warmup cost
    break_even_requests = warmup_ms / max(time_saved_per_request, 0.01)

    result = {
        "avg_baseline_ttft_ms": round(avg_baseline_ttft, 2),
        "avg_cached_ttft_ms": round(avg_cached_ttft, 2),
        "time_saved_per_request_ms": round(time_saved_per_request, 2),
        "cache_memory_mb": cache_memory_mb,
        "warmup_time_ms": round(warmup_ms, 2),
        "break_even_requests": round(break_even_requests, 1),
        "speedup_factor": round(avg_baseline_ttft / max(avg_cached_ttft, 0.01), 2),
    }

    log.info("  Baseline TTFT: %.1fms, Cached TTFT: %.1fms", avg_baseline_ttft, avg_cached_ttft)
    log.info("  Time saved/request: %.1fms", time_saved_per_request)
    log.info("  Cache memory: %.1fMB", cache_memory_mb)
    log.info("  Warmup time: %.1fms", warmup_ms)
    log.info("  Break-even after: %.0f requests", break_even_requests)

    del engine, engine2
    gc.collect()

    return result


def measure_disk_tier(model_name: str, chunk_size: int) -> Dict:
    """
    Compare cold-storage (disk) latency vs. recomputation.
    """
    log.info("=== Disk tier analysis ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = InferenceEngine.from_pretrained(
            model_name=model_name, chunk_size=chunk_size,
            max_chunks=4,  # Small to force eviction
            disk_cache_dir=tmpdir,
        )

        # Fill cache beyond capacity to trigger evictions to disk
        long_text = (SYSTEM_PROMPT + "\n\n" + RAG_DOCUMENT + "\n\n") * 3
        engine.warm_chunks(long_text, position_offset=0)

        stats = engine.cache_stats()

        # Measure disk promotion time
        # Force a cache miss by querying an evicted chunk
        token_ids = engine._encode(SYSTEM_PROMPT)
        chunks = engine.chunk_registry.split(token_ids)

        promotion_times = []
        recompute_times = []

        for start, end, slice_ids in chunks[:2]:
            from kvboost.models import chunk_id_from_tokens
            cid = chunk_id_from_tokens(slice_ids)

            # If it's in hot cache, skip
            if cid in engine.cache_manager._hot:
                continue

            # Measure disk promotion
            disk_path = Path(tmpdir) / f"{cid}.pt"
            if disk_path.exists():
                t0 = time.perf_counter()
                chunk = engine.cache_manager.get(cid)
                promotion_time = (time.perf_counter() - t0) * 1000
                promotion_times.append(promotion_time)

                # Measure recomputation cost
                t0 = time.perf_counter()
                _ = engine._encode_to_kv(slice_ids, position_offset=start)
                recompute_time = (time.perf_counter() - t0) * 1000
                recompute_times.append(recompute_time)

        result = {
            "max_chunks": 4,
            "hot_chunks": stats["hot_chunks"],
            "disk_promotion_ms": compute_stats(promotion_times) if promotion_times else {},
            "recomputation_ms": compute_stats(recompute_times) if recompute_times else {},
            "disk_faster_than_recompute": (
                (sum(promotion_times) / max(len(promotion_times), 1))
                < (sum(recompute_times) / max(len(recompute_times), 1))
            ) if promotion_times and recompute_times else None,
        }

        if promotion_times and recompute_times:
            log.info("  Disk promotion: %.2fms avg", sum(promotion_times) / len(promotion_times))
            log.info("  Recomputation: %.2fms avg", sum(recompute_times) / len(recompute_times))
        else:
            log.info("  No evicted chunks found on disk for comparison")

        del engine
        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Memory analysis")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=str, default="06_memory_analysis.json")
    args = parser.parse_args()

    all_results = {}

    # 1. Memory per chunk
    all_results["memory_per_chunk"] = measure_memory_per_chunk(
        args.model, [32, 64, 128, 256, 512]
    )

    # 2. Total memory scaling
    all_results["total_memory_scaling"] = measure_total_memory_scaling(
        args.model, args.chunk_size, [8, 16, 32, 64, 128, 256]
    )

    # 3. Break-even analysis
    all_results["break_even"] = measure_break_even(args.model, args.chunk_size)

    # 4. Disk tier analysis
    all_results["disk_tier"] = measure_disk_tier(args.model, args.chunk_size)

    # Print summary
    print("\n" + "=" * 70)
    print("MEMORY ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n--- Memory Per Chunk ---")
    print(f"{'ChunkSz':<10} {'Actual MB':<12} {'Theoretical MB':<16}")
    for r in all_results["memory_per_chunk"]:
        print(f"{r['chunk_size']:<10} {r['actual_mb_per_chunk']:<12.3f} "
              f"{r['theoretical_mb_per_chunk']:<16.3f}")

    print("\n--- Total Memory Scaling ---")
    print(f"{'MaxChunks':<12} {'Stored':<10} {'Memory MB':<12}")
    for r in all_results["total_memory_scaling"]:
        print(f"{r['max_chunks']:<12} {r['actual_hot_chunks']:<10} "
              f"{r['hot_memory_mb']:<12.1f}")

    be = all_results["break_even"]
    print(f"\n--- Break-Even ---")
    print(f"  Speedup: {be['speedup_factor']}x")
    print(f"  Time saved/request: {be['time_saved_per_request_ms']:.1f}ms")
    print(f"  Cache memory: {be['cache_memory_mb']:.1f}MB")
    print(f"  Break-even after: {be['break_even_requests']:.0f} requests")

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
