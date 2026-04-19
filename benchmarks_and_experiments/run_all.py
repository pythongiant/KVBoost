#!/usr/bin/env python3
"""
Master runner for all benchmarks and experiments.

Runs experiments sequentially (or a subset) and collects all results
into the results/ directory.

Usage:
  python run_all.py                          # Run all experiments
  python run_all.py --experiments 1,2,4      # Run specific experiments
  python run_all.py --quick                  # Quick mode (fewer runs)
  python run_all.py --model Qwen/Qwen2-0.5B # Use a different model
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")

EXPERIMENTS = {
    1: ("01_scale_models", "Scale across model sizes"),
    2: ("02_latency_breakdown", "Latency breakdown by pipeline stage"),
    3: ("03_hyperparameter_sweep", "Chunk size and recompute overlap sweep"),
    4: ("04_output_quality", "Output quality validation"),
    5: ("05_realistic_workloads", "Long-context RAG, multi-turn, server sim"),
    6: ("06_memory_analysis", "Memory cost and break-even analysis"),
    7: ("07_baseline_comparison", "Comparison against existing systems"),
    8: ("08_chunking_strategies", "FIXED vs SEMANTIC vs DOCUMENT chunking"),
    9: ("09_cache_hit_simulation", "Cache hit rate under traffic simulation"),
    10: ("10_statistical_rigor", "Statistical rigor with significance tests"),
}


def run_experiment(exp_num: int, module_name: str, extra_args: list) -> bool:
    """Run a single experiment by importing and calling its main()."""
    log.info("=" * 70)
    log.info("EXPERIMENT %d: %s", exp_num, EXPERIMENTS[exp_num][1])
    log.info("=" * 70)

    t0 = time.perf_counter()

    try:
        # Save and replace sys.argv for the experiment's argparse
        saved_argv = sys.argv
        sys.argv = [module_name + ".py"] + extra_args
        module = importlib.import_module(module_name)
        # Reload in case it was imported before with different args
        importlib.reload(module)
        module.main()
        sys.argv = saved_argv

        elapsed = time.perf_counter() - t0
        log.info("Experiment %d completed in %.1fs", exp_num, elapsed)
        return True

    except Exception as e:
        log.error("Experiment %d FAILED: %s", exp_num, e, exc_info=True)
        sys.argv = saved_argv
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all KV-cache benchmarks and experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiments:
  1  Scale across model sizes
  2  Latency breakdown by pipeline stage
  3  Chunk size and recompute overlap sweep
  4  Output quality validation (perplexity, KL divergence)
  5  Realistic workloads (long-context RAG, multi-turn, server sim)
  6  Memory cost and break-even analysis
  7  Comparison against existing systems (vLLM, SGLang)
  8  Chunking strategy comparison (FIXED, SEMANTIC, DOCUMENT)
  9  Cache hit rate under simulated traffic
  10 Statistical rigor with significance tests

Examples:
  python run_all.py                           # all experiments
  python run_all.py --experiments 2,4,10      # specific experiments
  python run_all.py --quick                   # reduced runs for speed
  python run_all.py --model Qwen/Qwen2-0.5B  # different model
""",
    )
    parser.add_argument(
        "--experiments", type=str, default="all",
        help="Comma-separated experiment numbers (e.g., '1,2,4') or 'all'",
    )
    parser.add_argument(
        "--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model to use for experiments",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=128,
        help="Default chunk size",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer runs per experiment",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32,
        help="Max tokens to generate per query",
    )
    args = parser.parse_args()

    # Determine which experiments to run
    if args.experiments == "all":
        exp_nums = sorted(EXPERIMENTS.keys())
    else:
        exp_nums = [int(x.strip()) for x in args.experiments.split(",")]

    # Experiments that accept --runs
    accepts_runs = {1, 2, 3, 5, 7, 8, 10}

    # Build common args
    common_args = ["--model", args.model, "--chunk-size", str(args.chunk_size)]
    common_args.extend(["--max-new-tokens", "16" if args.quick else str(args.max_new_tokens)])

    # Run experiments
    total_start = time.perf_counter()
    results = {}

    for exp_num in exp_nums:
        if exp_num not in EXPERIMENTS:
            log.warning("Unknown experiment number: %d, skipping", exp_num)
            continue

        module_name = EXPERIMENTS[exp_num][0]

        # Build experiment-specific args
        exp_args = list(common_args)

        # Add --runs only for experiments that accept it
        if args.quick and exp_num in accepts_runs:
            exp_args.extend(["--runs", "2"])

        # Quick mode overrides
        if args.quick:
            if exp_num == 1:
                exp_args.extend(["--models", args.model])
            elif exp_num == 3:
                exp_args.extend(["--chunk-sizes", "64,128", "--overlaps", "0,16"])
            elif exp_num == 9:
                exp_args.extend(["--n-requests", "50"])
            elif exp_num == 10:
                exp_args.extend(["--warmup-runs", "1"])

        success = run_experiment(exp_num, module_name, exp_args)
        results[exp_num] = {
            "name": EXPERIMENTS[exp_num][1],
            "success": success,
        }

    total_elapsed = time.perf_counter() - total_start

    # Final summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"\nResults saved to: benchmarks_and_experiments/results/\n")

    for exp_num, info in sorted(results.items()):
        status = "PASS" if info["success"] else "FAIL"
        print(f"  [{status}] Experiment {exp_num}: {info['name']}")

    n_pass = sum(1 for r in results.values() if r["success"])
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} experiments passed")


if __name__ == "__main__":
    main()
