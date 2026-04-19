#!/usr/bin/env python3
"""
Experiment 10: Statistical Rigor
==================================
Wraps any experiment configuration with proper statistical methodology:

  1. 10+ runs per configuration with warmup discarding
  2. Cold-start vs. warm-cache separation
  3. Confidence intervals and significance tests
  4. Outlier detection and reporting

Usage:
  python 10_statistical_rigor.py
  python 10_statistical_rigor.py --runs 20 --warmup-runs 3
"""

from __future__ import annotations

import argparse
import math
import statistics
from typing import Dict, List, Tuple

from utils import (
    get_logger, load_engine, save_results,
    SYSTEM_PROMPT, RAG_DOCUMENT, QUERIES, RAG_QUERIES,
    GenerationMode,
)

log = get_logger("10_statistical_rigor")


def confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval assuming approximately normal distribution."""
    n = len(values)
    if n < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)

    mean = statistics.mean(values)
    stderr = statistics.stdev(values) / math.sqrt(n)

    # z-scores for common confidence levels
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    margin = z * stderr
    return (round(mean - margin, 3), round(mean + margin, 3))


def welch_t_test(group_a: List[float], group_b: List[float]) -> Dict:
    """
    Welch's t-test for unequal variances.
    Tests whether the means of two groups are significantly different.
    Returns t-statistic, approximate degrees of freedom, and p-value estimate.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return {"error": "Need at least 2 samples per group"}

    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)

    # Welch's t-statistic
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-10:
        return {"t_statistic": 0.0, "significant": False,
                "note": "Variance too small (identical results)"}

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
    df = num / max(denom, 1e-10)

    # Approximate p-value using normal distribution (good for df > 30)
    # For smaller df, this is approximate
    p_approx = 2 * (1 - _normal_cdf(abs(t_stat)))

    return {
        "mean_a": round(mean_a, 3),
        "mean_b": round(mean_b, 3),
        "difference": round(mean_a - mean_b, 3),
        "t_statistic": round(t_stat, 3),
        "degrees_of_freedom": round(df, 1),
        "p_value_approx": round(p_approx, 6),
        "significant_at_005": p_approx < 0.05,
        "significant_at_001": p_approx < 0.01,
    }


def _normal_cdf(x: float) -> float:
    """Approximate CDF of standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def detect_outliers(values: List[float], method: str = "iqr") -> Dict:
    """Detect outliers using IQR method."""
    if len(values) < 4:
        return {"outliers": [], "clean_values": values}

    sorted_v = sorted(values)
    n = len(sorted_v)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[3 * n // 4]
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = [v for v in values if v < lower or v > upper]
    clean = [v for v in values if lower <= v <= upper]

    return {
        "n_total": len(values),
        "n_outliers": len(outliers),
        "outlier_values": [round(v, 3) for v in outliers],
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "bounds": (round(lower, 3), round(upper, 3)),
        "clean_values": clean,
    }


def comprehensive_stats(values: List[float], label: str = "",
                        confidence: float = 0.95) -> Dict:
    """Compute comprehensive statistics for a set of measurements."""
    if not values:
        return {}

    n = len(values)
    sorted_v = sorted(values)

    result = {
        "label": label,
        "n": n,
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }

    if n > 1:
        result["stdev"] = round(statistics.stdev(values), 3)
        result["stderr"] = round(statistics.stdev(values) / math.sqrt(n), 3)
        ci = confidence_interval(values, confidence)
        result[f"ci_{int(confidence * 100)}"] = ci
        result["cv"] = round(statistics.stdev(values) / max(abs(statistics.mean(values)), 1e-10), 3)

    if n >= 10:
        result["p5"] = round(sorted_v[int(n * 0.05)], 3)
        result["p95"] = round(sorted_v[int(n * 0.95)], 3)

    if n >= 20:
        result["p1"] = round(sorted_v[int(n * 0.01)], 3)
        result["p99"] = round(sorted_v[int(n * 0.99)], 3)

    outlier_info = detect_outliers(values)
    result["outliers"] = outlier_info["n_outliers"]

    return result


def run_rigorous_benchmark(model_name: str, chunk_size: int,
                           total_runs: int, warmup_runs: int,
                           max_new_tokens: int) -> Dict:
    """
    Run the core benchmark with full statistical rigor.
    """
    engine = load_engine(model_name=model_name, chunk_size=chunk_size)

    workloads = {
        "system_prompt": (SYSTEM_PROMPT, QUERIES),
        "rag_document": (RAG_DOCUMENT, RAG_QUERIES),
    }

    modes = [GenerationMode.BASELINE, GenerationMode.CHUNK_KV_REUSE]
    all_results = {}

    for wname, (prefix, queries) in workloads.items():
        log.info("=" * 50)
        log.info("Workload: %s (%d runs + %d warmup)", wname, total_runs, warmup_runs)
        log.info("=" * 50)

        engine.warm_chunks(prefix, position_offset=0)
        workload_results = {}

        for mode in modes:
            log.info("  Mode: %s", mode.value)

            # Collect all measurements
            all_ttfts: List[float] = []
            all_totals: List[float] = []
            all_tps: List[float] = []
            cold_ttfts: List[float] = []

            for run_i in range(warmup_runs + total_runs):
                for query in queries:
                    prompt = prefix + "\n\n" + query
                    r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                        mode=mode, do_sample=False)

                    if run_i < warmup_runs:
                        cold_ttfts.append(r.ttft_ms)
                    else:
                        all_ttfts.append(r.ttft_ms)
                        all_totals.append(r.total_ms)
                        all_tps.append(r.tokens_per_sec)

            # Comprehensive stats
            ttft_stats = comprehensive_stats(all_ttfts, f"{mode.value}_ttft")
            total_stats = comprehensive_stats(all_totals, f"{mode.value}_total")
            tps_stats = comprehensive_stats(all_tps, f"{mode.value}_tps")

            # Outlier analysis
            ttft_outliers = detect_outliers(all_ttfts)

            workload_results[mode.value] = {
                "ttft_ms": ttft_stats,
                "total_ms": total_stats,
                "tokens_per_sec": tps_stats,
                "cold_start_ttft_ms": comprehensive_stats(cold_ttfts, "cold_start"),
                "outlier_analysis": {
                    "n_outliers": ttft_outliers["n_outliers"],
                    "outlier_values": ttft_outliers["outlier_values"],
                    "bounds": ttft_outliers["bounds"],
                },
                "raw_ttfts": [round(t, 2) for t in all_ttfts],  # For reproducibility
            }

            log.info("    TTFT: mean=%.1f ± %.1fms (n=%d, %d outliers)",
                     ttft_stats.get("mean", 0),
                     ttft_stats.get("stdev", 0),
                     ttft_stats.get("n", 0),
                     ttft_outliers["n_outliers"])

        # Significance test: baseline vs chunk_kv
        baseline_ttfts = workload_results["baseline"]["raw_ttfts"]
        cached_ttfts = workload_results["chunk_kv_reuse"]["raw_ttfts"]

        significance = welch_t_test(baseline_ttfts, cached_ttfts)
        workload_results["significance_test"] = significance

        log.info("  Significance test: t=%.2f, p=%.6f, significant=%s",
                 significance.get("t_statistic", 0),
                 significance.get("p_value_approx", 1),
                 significance.get("significant_at_005", False))

        # Effect size (Cohen's d)
        if len(baseline_ttfts) > 1 and len(cached_ttfts) > 1:
            pooled_std = math.sqrt(
                (statistics.variance(baseline_ttfts) + statistics.variance(cached_ttfts)) / 2
            )
            cohens_d = (statistics.mean(baseline_ttfts) - statistics.mean(cached_ttfts)) / max(pooled_std, 1e-10)
            workload_results["effect_size_cohens_d"] = round(cohens_d, 3)

            # Interpretation
            if abs(cohens_d) < 0.2:
                effect_label = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_label = "small"
            elif abs(cohens_d) < 0.8:
                effect_label = "medium"
            else:
                effect_label = "large"
            workload_results["effect_size_label"] = effect_label

        all_results[wname] = workload_results

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Statistical rigor"
    )
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=10,
                        help="Measurement runs per query (after warmup)")
    parser.add_argument("--warmup-runs", type=int, default=3,
                        help="Warmup runs to discard")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=str, default="10_statistical_rigor.json")
    args = parser.parse_args()

    results = run_rigorous_benchmark(
        args.model, args.chunk_size,
        args.runs, args.warmup_runs,
        args.max_new_tokens,
    )

    # Print summary
    print("\n" + "=" * 90)
    print("STATISTICAL RIGOR SUMMARY")
    print("=" * 90)

    for wname, modes in results.items():
        print(f"\n--- {wname} ---")
        print(f"{'Metric':<25} {'Baseline':<30} {'ChunkKV':<30}")
        print("-" * 85)

        b = modes.get("baseline", {}).get("ttft_ms", {})
        c = modes.get("chunk_kv_reuse", {}).get("ttft_ms", {})

        print(f"{'TTFT mean (ms)':<25} "
              f"{b.get('mean', 0):.1f} ± {b.get('stdev', 0):.1f}{'':<15} "
              f"{c.get('mean', 0):.1f} ± {c.get('stdev', 0):.1f}")
        print(f"{'TTFT median (ms)':<25} {b.get('median', 0):<30.1f} {c.get('median', 0):<30.1f}")
        ci_b = b.get("ci_95", (0, 0))
        ci_c = c.get("ci_95", (0, 0))
        print(f"{'95% CI':<25} [{ci_b[0]:.1f}, {ci_b[1]:.1f}]{'':<18} "
              f"[{ci_c[0]:.1f}, {ci_c[1]:.1f}]")
        print(f"{'CV':<25} {b.get('cv', 0):<30.3f} {c.get('cv', 0):<30.3f}")
        print(f"{'Outliers':<25} "
              f"{modes.get('baseline', {}).get('outlier_analysis', {}).get('n_outliers', 0):<30} "
              f"{modes.get('chunk_kv_reuse', {}).get('outlier_analysis', {}).get('n_outliers', 0):<30}")

        sig = modes.get("significance_test", {})
        print(f"\n  Welch's t-test: t={sig.get('t_statistic', 0):.2f}, "
              f"p={sig.get('p_value_approx', 1):.6f}")
        print(f"  Significant at p<0.05: {sig.get('significant_at_005', False)}")
        print(f"  Significant at p<0.01: {sig.get('significant_at_001', False)}")
        print(f"  Effect size (Cohen's d): {modes.get('effect_size_cohens_d', 'N/A')} "
              f"({modes.get('effect_size_label', 'N/A')})")

        # Cold vs warm
        cold_b = modes.get("baseline", {}).get("cold_start_ttft_ms", {})
        cold_c = modes.get("chunk_kv_reuse", {}).get("cold_start_ttft_ms", {})
        print(f"\n  Cold start TTFT: baseline={cold_b.get('mean', 0):.1f}ms, "
              f"chunk_kv={cold_c.get('mean', 0):.1f}ms")

    path = save_results(results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
