#!/usr/bin/env python3
"""
Experiment runner: executes each backend in an isolated subprocess, then
loads all result checkpoints and prints a unified comparison table.

Usage:
    python run_experiment.py [same flags as run_benchmarks.py]

Each backend runs as a separate `python run_benchmarks.py --backends <b>` call
so they get a clean CUDA context. Results are collected from the checkpoint
files written by each run, then merged and saved as a single experiment JSON.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

BENCHMARK_SCRIPT = HERE / "run_benchmarks.py"

BACKENDS = ["kvboost", "vllm_prefixcache", "baseline"]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_backend(
    backend: str,
    common_args: List[str],
    kvboost_args: List[str],
    vllm_args: List[str],
    timeout: int = 7200,
) -> bool:
    cmd = [sys.executable, str(BENCHMARK_SCRIPT), "--backends", backend] + common_args
    if backend == "kvboost":
        cmd += kvboost_args
    if backend == "vllm_prefixcache":
        cmd += vllm_args

    log.info("=" * 80)
    log.info("STARTING backend=%s", backend)
    log.info("CMD: %s", " ".join(cmd))
    log.info("=" * 80)

    t0 = time.monotonic()
    result = subprocess.run(cmd, timeout=timeout)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        log.error("backend=%s FAILED (exit %d) after %.1fs", backend, result.returncode, elapsed)
        return False
    log.info("backend=%s finished in %.1fs", backend, elapsed)
    return True


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_latest_checkpoint(prefix: str, model: str) -> Optional[Dict]:
    model_slug = model.replace("/", "_")
    ckpt = RESULTS_DIR / f"{prefix}_ckpt_{model_slug}.json"
    if ckpt.exists():
        with open(ckpt) as f:
            return json.load(f)

    # Fall back to the most recent timestamped file matching the prefix
    candidates = sorted(
        RESULTS_DIR.glob(f"{prefix}_benchmark_{model_slug}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        with open(candidates[0]) as f:
            data = json.load(f)
        return data.get("results", data)
    return None


def _load_backend_checkpoints(backends: List[str], model: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for backend in backends:
        for prefix in ("accuracy", "latency", "memory"):
            model_slug = model.replace("/", "_")
            ckpt_path = RESULTS_DIR / f"{prefix}_ckpt_{backend}_{model_slug}.json"
            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    data = json.load(f)
                out.setdefault(backend, {})[prefix] = data
    return out


# ---------------------------------------------------------------------------
# Per-sample stats
# ---------------------------------------------------------------------------

def _accuracy_stats(samples: List[Dict]) -> Dict:
    if not samples:
        return {}
    cold = [s for s in samples if s.get("query_type") == "COLD"]
    warm = [s for s in samples if s.get("query_type") == "WARM"]

    def _acc(lst):
        return sum(s["correct"] for s in lst) / len(lst) if lst else None

    reuse_vals = [s["kv_reuse_pct"] for s in samples if s.get("kv_reuse_pct") is not None]
    warm_reuse = [s["kv_reuse_pct"] for s in warm if s.get("kv_reuse_pct") is not None]

    return {
        "n_total": len(samples),
        "n_cold": len(cold),
        "n_warm": len(warm),
        "accuracy_overall": _acc(samples),
        "accuracy_cold": _acc(cold),
        "accuracy_warm": _acc(warm),
        "avg_kv_reuse_pct": sum(reuse_vals) / len(reuse_vals) if reuse_vals else None,
        "avg_kv_reuse_pct_warm": sum(warm_reuse) / len(warm_reuse) if warm_reuse else None,
    }


def _latency_stats(samples: List[Dict]) -> Dict:
    if not samples:
        return {}
    import statistics
    cold = [s for s in samples if s.get("query_type") == "COLD"]
    warm = [s for s in samples if s.get("query_type") == "WARM"]

    def _ttft(lst):
        vals = [s["ttft_ms"] for s in lst if s.get("ttft_ms") is not None]
        if not vals:
            return {}
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p25": vals_sorted[max(0, int(n * 0.25) - 1)],
            "p75": vals_sorted[min(n - 1, int(n * 0.75))],
            "p95": vals_sorted[min(n - 1, int(n * 0.95))],
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
        }

    tps_vals = [s["tokens_per_second"] for s in samples if s.get("tokens_per_second")]
    reuse = [s.get("cache_reuse_ratio", 0) for s in warm]

    return {
        "n_total": len(samples),
        "n_cold": len(cold),
        "n_warm": len(warm),
        "ttft_ms_overall": _ttft(samples),
        "ttft_ms_cold": _ttft(cold),
        "ttft_ms_warm": _ttft(warm),
        "avg_tokens_per_second": statistics.mean(tps_vals) if tps_vals else None,
        "avg_cache_reuse_ratio_warm": sum(reuse) / len(reuse) if reuse else None,
    }


def _memory_stats(samples: List[Dict]) -> Dict:
    if not samples:
        return {}
    import statistics
    cold = [s for s in samples if s.get("query_type") == "COLD"]
    warm = [s for s in samples if s.get("query_type") == "WARM"]

    def _peak(lst):
        vals = [s["gpu_memory_mb"] for s in lst if s.get("gpu_memory_mb") is not None]
        if not vals:
            return {}
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "max": max(vals),
            "p95": sorted(vals)[min(len(vals) - 1, int(len(vals) * 0.95))],
        }

    eff_vals = [s["memory_efficiency"] for s in samples if s.get("memory_efficiency")]
    kv_vals = [s["kv_cache_mb"] for s in samples if s.get("kv_cache_mb") is not None]

    return {
        "n_total": len(samples),
        "n_cold": len(cold),
        "n_warm": len(warm),
        "peak_gpu_mb_overall": _peak(samples),
        "peak_gpu_mb_cold": _peak(cold),
        "peak_gpu_mb_warm": _peak(warm),
        "avg_kv_cache_mb": statistics.mean(kv_vals) if kv_vals else None,
        "avg_memory_efficiency_tok_per_mb": statistics.mean(eff_vals) if eff_vals else None,
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".1f", suffix=""):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}{suffix}"


def _print_accuracy_table(by_backend: Dict[str, Dict], backends: List[str]):
    print("\n" + "=" * 100)
    print("  ACCURACY")
    print("=" * 100)
    hdr = f"  {'Backend':<22} {'N':>6}  {'Overall':>8}  {'COLD':>8}  {'WARM':>8}  {'KV Reuse (warm)':>16}"
    print(hdr)
    print("  " + "-" * 96)

    baseline_overall = None
    for b in backends:
        s = by_backend.get(b, {}).get("accuracy_stats", {})
        if not s:
            print(f"  {b:<22}  (no data)")
            continue
        overall = s.get("accuracy_overall")
        cold = s.get("accuracy_cold")
        warm = s.get("accuracy_warm")
        reuse = s.get("avg_kv_reuse_pct_warm")
        n = s.get("n_total", 0)

        if b == "baseline":
            baseline_overall = overall
        delta = ""
        if b != "baseline" and baseline_overall is not None and overall is not None:
            delta = f"  ({overall - baseline_overall:+.1%} vs baseline)"

        reuse_str = _fmt(reuse, ".1f", "%") if reuse is not None else "N/A"
        print(
            f"  {b:<22} {n:>6}  {_fmt(overall, '.1%'):>8}  "
            f"{_fmt(cold, '.1%'):>8}  {_fmt(warm, '.1%'):>8}  {reuse_str:>16}{delta}"
        )
    print("=" * 100)


def _print_latency_table(by_backend: Dict[str, Dict], backends: List[str]):
    print("\n" + "=" * 115)
    print("  LATENCY  (Time-To-First-Token)")
    print("=" * 115)
    hdr = (
        f"  {'Backend':<22} {'N':>6}  "
        f"{'TTFT mean':>10}  {'TTFT p95':>10}  "
        f"{'COLD mean':>10}  {'WARM mean':>10}  "
        f"{'tok/s':>8}  {'Speedup':>8}"
    )
    print(hdr)
    print("  " + "-" * 111)

    baseline_ttft = None
    for b in backends:
        s = by_backend.get(b, {}).get("latency_stats", {})
        if not s:
            print(f"  {b:<22}  (no data)")
            continue
        overall = s.get("ttft_ms_overall", {})
        cold = s.get("ttft_ms_cold", {})
        warm = s.get("ttft_ms_warm", {})
        tps = s.get("avg_tokens_per_second")
        n = s.get("n_total", 0)

        mean = overall.get("mean")
        if b == "baseline":
            baseline_ttft = mean
        speedup = ""
        if b != "baseline" and baseline_ttft and mean:
            speedup = f"{baseline_ttft / mean:.2f}x"
        elif b == "baseline":
            speedup = "1.00x"

        print(
            f"  {b:<22} {n:>6}  "
            f"{_fmt(mean, '.1f', 'ms'):>10}  {_fmt(overall.get('p95'), '.1f', 'ms'):>10}  "
            f"{_fmt(cold.get('mean'), '.1f', 'ms'):>10}  {_fmt(warm.get('mean'), '.1f', 'ms'):>10}  "
            f"{_fmt(tps, '.1f'):>8}  {speedup:>8}"
        )
    print("=" * 115)


def _print_memory_table(by_backend: Dict[str, Dict], backends: List[str]):
    print("\n" + "=" * 105)
    print("  GPU MEMORY  (peak during inference)")
    print("=" * 105)
    hdr = (
        f"  {'Backend':<22} {'N':>6}  "
        f"{'Peak mean':>10}  {'Peak p95':>10}  "
        f"{'COLD mean':>10}  {'WARM mean':>10}  "
        f"{'KV cache':>10}  {'Savings':>8}"
    )
    print(hdr)
    print("  " + "-" * 101)

    baseline_peak = None
    for b in backends:
        s = by_backend.get(b, {}).get("memory_stats", {})
        if not s:
            print(f"  {b:<22}  (no data)")
            continue
        overall = s.get("peak_gpu_mb_overall", {})
        cold = s.get("peak_gpu_mb_cold", {})
        warm = s.get("peak_gpu_mb_warm", {})
        kv = s.get("avg_kv_cache_mb")
        n = s.get("n_total", 0)

        mean = overall.get("mean")
        if b == "baseline":
            baseline_peak = mean
        savings = ""
        if b != "baseline" and baseline_peak and mean:
            savings = f"{(baseline_peak - mean) / baseline_peak:+.1%}"
        elif b == "baseline":
            savings = "baseline"

        print(
            f"  {b:<22} {n:>6}  "
            f"{_fmt(mean, '.1f', 'MB'):>10}  {_fmt(overall.get('p95'), '.1f', 'MB'):>10}  "
            f"{_fmt(cold.get('mean'), '.1f', 'MB'):>10}  {_fmt(warm.get('mean'), '.1f', 'MB'):>10}  "
            f"{_fmt(kv, '.1f', 'MB'):>10}  {savings:>8}"
        )
    print("=" * 105)


def _print_validity_checks(by_backend: Dict[str, Dict], backends: List[str]):
    print("\n" + "=" * 80)
    print("  VALIDITY CHECKS")
    print("=" * 80)
    # Cold accuracy should match across backends (same inputs, greedy decode)
    cold_accs = {}
    for b in backends:
        s = by_backend.get(b, {}).get("accuracy_stats", {})
        v = s.get("accuracy_cold")
        if v is not None:
            cold_accs[b] = v
    if cold_accs:
        vals = list(cold_accs.values())
        spread = max(vals) - min(vals)
        flag = "⚠ DIVERGED (>2pp)" if spread > 0.02 else "✓ consistent"
        print(f"  Cold accuracy spread: {spread:.1%}  {flag}")
        for b, v in cold_accs.items():
            print(f"    {b:<22} cold_acc={v:.1%}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run each backend in isolation then compare results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-context-tokens", type=int, default=6000)
    parser.add_argument(
        "--backends", nargs="+", default=BACKENDS,
        choices=BACKENDS,
        help="Which backends to run (default: all three)"
    )
    parser.add_argument("--no-checkpoint", action="store_true")
    # KVBoost
    parser.add_argument("--max-cache-bytes", type=float, default=1.5e9)
    parser.add_argument("--recency-window-chunks", type=int, default=8)
    parser.add_argument("--recompute-strategy", default="cacheblend",
                        choices=["selective", "cacheblend", "none"])
    parser.add_argument("--chunk-boundary-window", type=int, default=16)
    parser.add_argument("--overlap-k", type=int, default=16)
    parser.add_argument("--sink-tokens", type=int, default=32)
    # vLLM
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--vllm-no-enforce-eager", action="store_true")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=1)
    # Runner
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Max seconds per backend (default: 7200)")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip subprocess runs; just load existing checkpoints and print tables")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_")

    # --- build arg lists for the subprocess ---
    common_args = [
        "--model", args.model,
        "--n-samples", str(args.n_samples),
        "--max-context-tokens", str(args.max_context_tokens),
        "--output-dir", str(args.output_dir),
    ]
    if args.no_checkpoint:
        common_args.append("--no-checkpoint")

    kvboost_args = [
        "--max-cache-bytes", str(args.max_cache_bytes),
        "--recency-window-chunks", str(args.recency_window_chunks),
        "--recompute-strategy", args.recompute_strategy,
        "--chunk-boundary-window", str(args.chunk_boundary_window),
        "--overlap-k", str(args.overlap_k),
        "--sink-tokens", str(args.sink_tokens),
    ]

    vllm_args = [
        "--vllm-gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
        "--vllm-max-num-seqs", str(args.vllm_max_num_seqs),
    ]
    if args.vllm_no_enforce_eager:
        vllm_args.append("--vllm-no-enforce-eager")

    # --- run each backend ---
    run_status: Dict[str, bool] = {}
    if not args.skip_run:
        for backend in args.backends:
            ok = _run_backend(backend, common_args, kvboost_args, vllm_args, timeout=args.timeout)
            run_status[backend] = ok
    else:
        log.info("--skip-run: skipping subprocess execution")
        run_status = {b: True for b in args.backends}

    # --- load checkpoints ---
    log.info("Loading checkpoints...")
    raw_ckpts = _load_backend_checkpoints(args.backends, args.model)

    # --- compute stats and assemble full report ---
    by_backend: Dict[str, Dict] = {}
    for backend in args.backends:
        bdata = raw_ckpts.get(backend, {})
        entry: Dict[str, Any] = {
            "run_ok": run_status.get(backend, False),
        }
        for bench_name, stat_fn in [
            ("accuracy", _accuracy_stats),
            ("latency", _latency_stats),
            ("memory", _memory_stats),
        ]:
            ckpt = bdata.get(bench_name, {})
            samples = ckpt.get("samples", [])
            entry[f"{bench_name}_n_done"] = ckpt.get("n_done", len(samples))
            entry[f"{bench_name}_n_total"] = ckpt.get("n_total", None)
            entry[f"{bench_name}_samples"] = samples
            entry[f"{bench_name}_stats"] = stat_fn(samples)
        by_backend[backend] = entry

    # --- print tables ---
    print(f"\n{'=' * 100}")
    print(f"  EXPERIMENT RESULTS")
    print(f"  Model:    {args.model}")
    print(f"  Samples:  {args.n_samples}  (max_ctx={args.max_context_tokens})")
    print(f"  Backends: {', '.join(args.backends)}")
    print(f"  KVBoost:  cache={args.max_cache_bytes/1e9:.2f}GB  recency={args.recency_window_chunks}  "
          f"strategy={args.recompute_strategy}  boundary={args.chunk_boundary_window}  "
          f"overlap={args.overlap_k}  sink={args.sink_tokens}")
    print(f"  vLLM:     gpu_mem={args.vllm_gpu_memory_utilization}  "
          f"enforce_eager={not args.vllm_no_enforce_eager}  max_num_seqs={args.vllm_max_num_seqs}")
    print(f"  Run at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 100}")

    _print_accuracy_table(by_backend, args.backends)
    _print_latency_table(by_backend, args.backends)
    _print_memory_table(by_backend, args.backends)
    _print_validity_checks(by_backend, args.backends)

    # --- save unified JSON ---
    report = {
        "experiment_timestamp": timestamp,
        "model": args.model,
        "n_samples": args.n_samples,
        "max_context_tokens": args.max_context_tokens,
        "backends": args.backends,
        "config": {
            "kvboost": {
                "max_cache_bytes": args.max_cache_bytes,
                "recency_window_chunks": args.recency_window_chunks,
                "recompute_strategy": args.recompute_strategy,
                "chunk_boundary_window": args.chunk_boundary_window,
                "overlap_k": args.overlap_k,
                "sink_tokens": args.sink_tokens,
            },
            "vllm": {
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "enforce_eager": not args.vllm_no_enforce_eager,
                "max_num_seqs": args.vllm_max_num_seqs,
            },
        },
        "run_status": run_status,
        "results": by_backend,
    }

    out_path = args.output_dir / f"experiment_{model_slug}_{timestamp}.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
