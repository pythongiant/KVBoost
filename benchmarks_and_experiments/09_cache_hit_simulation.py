#!/usr/bin/env python3
"""
Experiment 9: Cache Hit Rate Under Realistic Traffic
======================================================
Simulates a request stream and measures cache behavior over time:

  1. Hit rate over time as cache warms up (cold-start curve)
  2. Effect of max_chunks on steady-state hit rate
  3. Working set size distribution
  4. LRU eviction effectiveness under different traffic patterns

Traffic patterns:
  - Uniform: all queries equally likely
  - Zipfian: few queries very popular (realistic)
  - Temporal: query popularity shifts over time (concept drift)

Usage:
  python 09_cache_hit_simulation.py
  python 09_cache_hit_simulation.py --n-requests 500 --max-chunks 64
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from utils import (
    get_logger, load_engine, save_results, compute_stats,
    SYSTEM_PROMPT,
    GenerationMode,
)

from kvboost import KVBoost as InferenceEngine
from kvboost.models import chunk_id_from_tokens

log = get_logger("09_cache_hit_simulation")

# Expanded query pool for realistic traffic simulation
QUERY_POOL = [
    # Popular queries (would be frequent under Zipfian)
    "What is the time complexity of quicksort?",
    "Explain how hash tables work.",
    "What is the CAP theorem?",
    "How does TCP/IP work?",
    "What is a REST API?",
    # Medium popularity
    "Explain the observer design pattern.",
    "What is dependency injection?",
    "How does garbage collection work?",
    "What are microservices?",
    "Explain how HTTPS works.",
    "What is a load balancer?",
    "How do database indexes work?",
    "What is eventual consistency?",
    "Explain WebSockets vs HTTP.",
    "What is container orchestration?",
    # Rare queries
    "Explain the Raft consensus algorithm in detail.",
    "How do bloom filters work and when to use them?",
    "What is the difference between ACID and BASE?",
    "Explain CRDTs and their use in distributed systems.",
    "How does speculative execution work in CPUs?",
    "What is NUMA-aware programming?",
    "Explain the internals of B+ tree indexes.",
    "How does Kubernetes scheduling work?",
    "What is the actor model in concurrent programming?",
    "Explain how JIT compilation works.",
]

# Multiple system prompts to simulate different "applications"
SYSTEM_PROMPTS = [
    SYSTEM_PROMPT,
    (
        "You are a helpful coding assistant. Answer programming questions concisely "
        "with working code examples. Use Python unless the user specifies another language."
    ),
    (
        "You are a database expert. Help users with SQL queries, schema design, "
        "and database optimization. Always consider performance implications."
    ),
]


def zipfian_sample(n_items: int, alpha: float = 1.2) -> int:
    """Sample from a Zipfian distribution over [0, n_items)."""
    weights = [1.0 / (i + 1) ** alpha for i in range(n_items)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(range(n_items), weights=probs, k=1)[0]


def generate_traffic(pattern: str, n_requests: int,
                     n_queries: int, n_prompts: int) -> List[Tuple[int, int]]:
    """
    Generate a stream of (system_prompt_idx, query_idx) pairs.
    """
    requests = []

    if pattern == "uniform":
        for _ in range(n_requests):
            sp = random.randint(0, n_prompts - 1)
            q = random.randint(0, n_queries - 1)
            requests.append((sp, q))

    elif pattern == "zipfian":
        for _ in range(n_requests):
            sp = zipfian_sample(n_prompts, alpha=1.5)
            q = zipfian_sample(n_queries, alpha=1.2)
            requests.append((sp, q))

    elif pattern == "temporal":
        # First half: mostly prompt 0 + popular queries
        # Second half: shift to prompt 1 + different queries
        half = n_requests // 2
        for i in range(n_requests):
            if i < half:
                sp = 0 if random.random() < 0.8 else random.randint(0, n_prompts - 1)
                q = zipfian_sample(min(n_queries, 10), alpha=1.5)
            else:
                sp = 1 if random.random() < 0.8 else random.randint(0, n_prompts - 1)
                q = zipfian_sample(n_queries, alpha=1.0)
                # Shift query distribution
                q = (q + n_queries // 2) % n_queries
            requests.append((sp, q))

    return requests


def simulate_cache(engine: InferenceEngine, traffic: List[Tuple[int, int]],
                   system_prompts: List[str], queries: List[str],
                   max_new_tokens: int, window_size: int = 50) -> Dict:
    """
    Process the traffic stream and track cache metrics over time.
    """
    # Reset cache stats
    engine.cache_manager.hits = 0
    engine.cache_manager.misses = 0

    # Metrics over time
    cumulative_hits = []
    cumulative_misses = []
    window_hit_rates = []
    ttfts = []
    unique_chunks_seen = set()

    recent_hits = []
    recent_misses = []

    for i, (sp_idx, q_idx) in enumerate(traffic):
        sp = system_prompts[sp_idx % len(system_prompts)]
        query = queries[q_idx % len(queries)]
        prompt = sp + "\n\n" + query

        # Track chunks for this request
        token_ids = engine._encode(prompt)
        for _, _, slice_ids in engine.chunk_registry.split(token_ids):
            cid = chunk_id_from_tokens(slice_ids)
            unique_chunks_seen.add(cid)

        hits_before = engine.cache_manager.hits
        misses_before = engine.cache_manager.misses

        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)

        hits_delta = engine.cache_manager.hits - hits_before
        misses_delta = engine.cache_manager.misses - misses_before

        cumulative_hits.append(engine.cache_manager.hits)
        cumulative_misses.append(engine.cache_manager.misses)
        ttfts.append(r.ttft_ms)

        recent_hits.append(hits_delta)
        recent_misses.append(misses_delta)

        # Sliding window hit rate
        if len(recent_hits) > window_size:
            recent_hits.pop(0)
            recent_misses.pop(0)

        window_total = sum(recent_hits) + sum(recent_misses)
        window_hr = sum(recent_hits) / max(window_total, 1)
        window_hit_rates.append(round(window_hr, 3))

        if (i + 1) % 100 == 0:
            log.info("  Request %d/%d: window_hit_rate=%.3f, unique_chunks=%d, TTFT=%.1fms",
                     i + 1, len(traffic), window_hr, len(unique_chunks_seen), r.ttft_ms)

    # Summary stats
    total = engine.cache_manager.hits + engine.cache_manager.misses
    return {
        "total_requests": len(traffic),
        "total_cache_hits": engine.cache_manager.hits,
        "total_cache_misses": engine.cache_manager.misses,
        "overall_hit_rate": round(engine.cache_manager.hits / max(total, 1), 3),
        "unique_chunks_seen": len(unique_chunks_seen),
        "hot_chunks_at_end": engine.cache_stats()["hot_chunks"],
        "ttft_ms": compute_stats(ttfts),
        # Time series (sampled every 10 requests)
        "hit_rate_over_time": window_hit_rates[::10],
        "cumulative_hits": cumulative_hits[::10],
        "ttft_over_time": [round(t, 1) for t in ttfts[::10]],
        # Cold start metrics
        "cold_start_ttft_first_10": compute_stats(ttfts[:10]),
        "warm_ttft_last_10": compute_stats(ttfts[-10:]),
    }


def experiment_warmup_curve(engine: InferenceEngine,
                            n_requests: int, max_new_tokens: int) -> Dict:
    """Measure how quickly hit rate improves from cold start."""
    log.info("=== Warmup curve (cold start) ===")
    traffic = generate_traffic("zipfian", n_requests,
                               len(QUERY_POOL), len(SYSTEM_PROMPTS))
    return simulate_cache(engine, traffic, SYSTEM_PROMPTS, QUERY_POOL,
                          max_new_tokens)


def experiment_max_chunks_sweep(model_name: str, chunk_size: int,
                                n_requests: int, max_new_tokens: int,
                                max_chunks_values: list) -> List[Dict]:
    """Vary max_chunks and measure steady-state hit rate."""
    log.info("=== Max chunks sweep ===")
    results = []

    # Use same traffic for fair comparison
    traffic = generate_traffic("zipfian", n_requests,
                               len(QUERY_POOL), len(SYSTEM_PROMPTS))

    for mc in max_chunks_values:
        log.info("  max_chunks=%d", mc)
        engine = InferenceEngine.from_pretrained(
            model_name=model_name, chunk_size=chunk_size, max_chunks=mc,
        )
        result = simulate_cache(engine, traffic, SYSTEM_PROMPTS, QUERY_POOL,
                                max_new_tokens, window_size=50)
        result["max_chunks"] = mc
        results.append(result)

        del engine

    return results


def experiment_traffic_patterns(engine: InferenceEngine,
                                n_requests: int, max_new_tokens: int) -> Dict:
    """Compare cache behavior under different traffic patterns."""
    log.info("=== Traffic pattern comparison ===")
    results = {}

    for pattern in ["uniform", "zipfian", "temporal"]:
        log.info("  Pattern: %s", pattern)
        # Reset engine cache
        engine.cache_manager._hot.clear()
        engine.cache_manager.hits = 0
        engine.cache_manager.misses = 0

        traffic = generate_traffic(pattern, n_requests,
                                   len(QUERY_POOL), len(SYSTEM_PROMPTS))
        result = simulate_cache(engine, traffic, SYSTEM_PROMPTS, QUERY_POOL,
                                max_new_tokens)
        result["pattern"] = pattern
        results[pattern] = result

    return results


def analyze_working_set(engine: InferenceEngine, n_requests: int) -> Dict:
    """Analyze the working set size distribution."""
    log.info("=== Working set analysis ===")

    chunk_frequency = Counter()
    traffic = generate_traffic("zipfian", n_requests,
                               len(QUERY_POOL), len(SYSTEM_PROMPTS))

    for sp_idx, q_idx in traffic:
        sp = SYSTEM_PROMPTS[sp_idx % len(SYSTEM_PROMPTS)]
        query = QUERY_POOL[q_idx % len(QUERY_POOL)]
        prompt = sp + "\n\n" + query
        token_ids = engine._encode(prompt)
        for _, _, slice_ids in engine.chunk_registry.split(token_ids):
            cid = chunk_id_from_tokens(slice_ids)
            chunk_frequency[cid] += 1

    total_chunks = len(chunk_frequency)
    total_accesses = sum(chunk_frequency.values())

    # Distribution analysis
    top_10_pct = sorted(chunk_frequency.values(), reverse=True)[:max(total_chunks // 10, 1)]
    top_10_pct_accesses = sum(top_10_pct)

    return {
        "total_unique_chunks": total_chunks,
        "total_accesses": total_accesses,
        "top_10pct_chunk_count": len(top_10_pct),
        "top_10pct_access_share": round(top_10_pct_accesses / max(total_accesses, 1), 3),
        "max_frequency": max(chunk_frequency.values()) if chunk_frequency else 0,
        "min_frequency": min(chunk_frequency.values()) if chunk_frequency else 0,
        "median_frequency": sorted(chunk_frequency.values())[total_chunks // 2] if chunk_frequency else 0,
        "frequency_histogram": dict(Counter(
            min(f, 20) for f in chunk_frequency.values()  # bucket at 20+
        )),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 9: Cache hit rate simulation"
    )
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--n-requests", type=int, default=200,
                        help="Number of simulated requests")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="Keep short for simulation speed")
    parser.add_argument("--output", type=str, default="09_cache_hit_simulation.json")
    args = parser.parse_args()

    random.seed(42)

    all_results = {}

    # 1. Warmup curve
    engine = load_engine(model_name=args.model, chunk_size=args.chunk_size,
                         max_chunks=args.max_chunks)
    all_results["warmup_curve"] = experiment_warmup_curve(
        engine, args.n_requests, args.max_new_tokens
    )

    # 2. Max chunks sweep
    all_results["max_chunks_sweep"] = experiment_max_chunks_sweep(
        args.model, args.chunk_size, min(args.n_requests, 100),
        args.max_new_tokens,
        max_chunks_values=[8, 16, 32, 64, 128],
    )

    # 3. Traffic patterns
    engine2 = load_engine(model_name=args.model, chunk_size=args.chunk_size,
                          max_chunks=args.max_chunks)
    all_results["traffic_patterns"] = experiment_traffic_patterns(
        engine2, args.n_requests, args.max_new_tokens
    )

    # 4. Working set analysis
    all_results["working_set"] = analyze_working_set(engine2, args.n_requests)

    # Print summary
    print("\n" + "=" * 70)
    print("CACHE HIT SIMULATION SUMMARY")
    print("=" * 70)

    wc = all_results["warmup_curve"]
    print(f"\n--- Warmup Curve ---")
    print(f"  Total requests: {wc['total_requests']}")
    print(f"  Overall hit rate: {wc['overall_hit_rate']:.3f}")
    print(f"  Cold start TTFT (first 10): {wc['cold_start_ttft_first_10'].get('mean', 0):.1f}ms")
    print(f"  Warm TTFT (last 10): {wc['warm_ttft_last_10'].get('mean', 0):.1f}ms")
    print(f"  Unique chunks seen: {wc['unique_chunks_seen']}")

    print(f"\n--- Max Chunks Sweep ---")
    print(f"{'MaxChunks':<12} {'HitRate':<10} {'TTFT(ms)':<12}")
    for r in all_results["max_chunks_sweep"]:
        print(f"{r['max_chunks']:<12} {r['overall_hit_rate']:<10.3f} "
              f"{r['ttft_ms'].get('mean', 0):<12.1f}")

    print(f"\n--- Traffic Patterns ---")
    print(f"{'Pattern':<12} {'HitRate':<10} {'TTFT(ms)':<12} {'UniqueChunks':<14}")
    for pattern, r in all_results["traffic_patterns"].items():
        print(f"{pattern:<12} {r['overall_hit_rate']:<10.3f} "
              f"{r['ttft_ms'].get('mean', 0):<12.1f} {r['unique_chunks_seen']:<14}")

    ws = all_results["working_set"]
    print(f"\n--- Working Set ---")
    print(f"  Unique chunks: {ws['total_unique_chunks']}")
    print(f"  Top 10% chunks handle {ws['top_10pct_access_share'] * 100:.1f}% of accesses")

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
