#!/usr/bin/env python3
"""
Deep-dive analyzer for a sharegpt_3way result JSON.

What it surfaces (beyond `compare.py`, which is for side-by-side bar charts):
  * Hardware + software fingerprint from RunMetadata.
  * Headline latency/throughput metrics + percentile distributions.
  * Per-turn-index breakdown (where cross-turn cache reuse should show).
  * Stop-reason distribution (eos / max_tokens / error).
  * Error breakdown — type counts + sample messages.
  * Speculative phase breakdown (KVBoost): per-turn draft/verify/rollback ms,
    acceptance rate, avg committed/round, share of phase time.
  * System telemetry: peak VRAM and host RSS distribution.
  * Backend-specific telemetry shape — keys present in `backend_telemetry`
    so you know what's available for downstream analysis.
  * Optional per-turn CSV export with every field (one row per turn).

Usage:
    python analyze_results.py results/kvboost.json
    python analyze_results.py results/kvboost.json results/vllm.json
    python analyze_results.py results/*.json --csv per_turn.csv
    python analyze_results.py results/kvboost.json --brief
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Loading ─────────────────────────────────────────────────────────────

def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Formatting helpers ──────────────────────────────────────────────────

def _fmt(value: Any, kind: str = "auto") -> str:
    if value is None:
        return "—"
    if kind == "ms":
        return f"{value:.1f} ms"
    if kind == "tps":
        return f"{value:.2f} tok/s"
    if kind == "ratio":
        return f"{value:.1%}"
    if kind == "mb":
        return f"{value:.0f} MB"
    if kind == "int":
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _section(title: str, ch: str = "=") -> None:
    print()
    print(ch * 88)
    print(f"  {title}")
    print(ch * 88)


def _pct(xs: List[float], q: float) -> Optional[float]:
    return float(np.percentile(xs, q)) if xs else None


def _all_turns(data: dict, only_ok: bool = False) -> List[dict]:
    out: List[dict] = []
    for conv in data.get("results", []) or []:
        for turn in conv.get("turns", []) or []:
            if only_ok and turn.get("error"):
                continue
            out.append(turn)
    return out


# ── Sections ────────────────────────────────────────────────────────────

def print_fingerprint(data: dict, path: Path) -> None:
    backend = data.get("backend", "?")
    rm = data.get("run_metadata") or {}
    _section(f"{backend.upper()}  —  {path.name}")
    print(f"  host       : {rm.get('hostname', '?')}  ({rm.get('platform', '?')})")
    print(f"  cpu        : {rm.get('cpu_model') or rm.get('platform', '?')}  ({rm.get('cpu_count', '?')} cores)")
    print(f"  ram        : {(rm.get('ram_total_mb') or 0) / 1024:.1f} GiB")
    print(f"  gpu        : {rm.get('gpu_name', '?')}  ({(rm.get('gpu_mem_total_mb') or 0):.0f} MB)  "
          f"cc={rm.get('gpu_compute_capability', '?')}")
    print(f"  cuda       : {rm.get('cuda_version', '?')}   driver: {rm.get('driver_version', '?')}")
    print(f"  python     : {rm.get('python_version', '?')}   torch: {rm.get('torch_version', '?')}   "
          f"transformers: {rm.get('transformers_version', '?')}")
    backend_ver = (rm.get('kvboost_version') or rm.get('vllm_version')
                   or rm.get('llama_cpp_version') or "?")
    print(f"  backend ver: {backend_ver}")
    git = rm.get('git_sha', '') or ''
    print(f"  git        : {git[:10]}  branch={rm.get('git_branch', '?')}  dirty={rm.get('git_dirty', False)}")
    print(f"  started    : {rm.get('start_iso', '?')}")
    print(f"  finished   : {rm.get('end_iso', '?')}")
    print(f"  wall       : {data.get('wall_s', 0):.1f} s")
    print()
    cfg = data.get("config") or {}
    print(f"  target model : {data.get('model', '?')}")
    print(f"  draft model  : {data.get('draft_model', '?')}")
    print(f"  config       : n_samples={cfg.get('n_samples')}  "
          f"turns={cfg.get('min_turns')}-{cfg.get('max_turns')}  "
          f"max_ctx_tok={cfg.get('max_context_tokens')}  "
          f"max_new_tok={cfg.get('max_new_tokens')}  gamma={cfg.get('gamma')}")


def print_headline(data: dict) -> None:
    backend = data.get("backend", "?")
    m = data.get("metrics") or {}
    ov = m.get("overall") or {}
    _section(f"{backend.upper()} — headline metrics", ch="─")
    rows = [
        ("conversations",        m.get("n_conversations", 0)),
        ("turns",                m.get("n_turns_total", 0)),
        ("errors",               m.get("n_turns_errored", 0)),
        ("", ""),
        ("TTFT p50",             _fmt(ov.get("ttft_p50_ms"), "ms")),
        ("TTFT p90",             _fmt(ov.get("ttft_p90_ms"), "ms")),
        ("TTFT p99",             _fmt(ov.get("ttft_p99_ms"), "ms")),
        ("TTFT mean",            _fmt(ov.get("ttft_mean_ms"), "ms")),
        ("ITL p50",              _fmt(ov.get("itl_p50_ms"), "ms")),
        ("ITL p90",              _fmt(ov.get("itl_p90_ms"), "ms")),
        ("decode tok/s mean",    _fmt(ov.get("decode_tps_mean"), "tps")),
        ("decode tok/s p50",     _fmt(ov.get("decode_tps_p50"), "tps")),
        ("output tok/s (wall)",  _fmt(ov.get("output_token_throughput"), "tps")),
        ("req throughput",       _fmt(ov.get("request_throughput_rps")) + " rps"),
        ("", ""),
        ("cache hit ratio mean", _fmt(ov.get("avg_cache_hit_ratio"), "ratio")),
        ("avg cached tokens",    _fmt(ov.get("avg_cached_tokens"), "int") if ov.get("avg_cached_tokens") else "—"),
        ("avg prompt tokens",    _fmt(ov.get("avg_prompt_tokens"), "int") if ov.get("avg_prompt_tokens") else "—"),
        ("avg output tokens",    _fmt(ov.get("avg_output_tokens"), "int") if ov.get("avg_output_tokens") else "—"),
        ("spec acceptance rate", _fmt(ov.get("spec_acceptance_rate"), "ratio")),
        ("", ""),
        ("peak VRAM max",        _fmt(ov.get("peak_vram_mb_max"), "mb")),
        ("peak VRAM mean",       _fmt(ov.get("peak_vram_mb_mean"), "mb")),
        ("host RSS max",         _fmt(ov.get("host_rss_mb_max"), "mb")),
        ("host RSS mean",        _fmt(ov.get("host_rss_mb_mean"), "mb")),
    ]
    for k, v in rows:
        print(f"  {k:<24} {v}")


def print_distributions(data: dict) -> None:
    ok = _all_turns(data, only_ok=True)
    if not ok:
        return
    _section(f"{data.get('backend','?').upper()} — distributions (ok turns only, n={len(ok)})", ch="─")
    metrics = [
        ("ttft_ms",      "TTFT (ms)",        ""),
        ("total_ms",     "total_ms",         ""),
        ("decode_ms",    "decode_ms",        ""),
        ("itl_ms",       "ITL (ms/tok)",     ""),
        ("decode_tps",   "decode_tps",       ""),
        ("output_tokens", "output_tokens",   ""),
        ("prompt_tokens", "prompt_tokens",   ""),
        ("cached_tokens", "cached_tokens",   ""),
    ]
    print(f"  {'metric':<18}{'min':>10}{'p25':>10}{'p50':>10}{'p75':>10}{'p90':>10}{'p99':>10}{'max':>10}{'mean':>10}")
    print(f"  {'-'*18}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*10}")
    for key, label, _ in metrics:
        xs = [float(t.get(key) or 0.0) for t in ok if t.get(key) is not None]
        if not xs:
            continue
        print(f"  {label:<18}"
              f"{min(xs):>10.2f}"
              f"{_pct(xs, 25) or 0:>10.2f}"
              f"{_pct(xs, 50) or 0:>10.2f}"
              f"{_pct(xs, 75) or 0:>10.2f}"
              f"{_pct(xs, 90) or 0:>10.2f}"
              f"{_pct(xs, 99) or 0:>10.2f}"
              f"{max(xs):>10.2f}"
              f"{float(np.mean(xs)):>10.2f}")


def print_per_turn(data: dict) -> None:
    m = data.get("metrics") or {}
    by_turn = m.get("by_turn") or {}
    if not by_turn:
        return
    _section(f"{data.get('backend','?').upper()} — per-turn breakdown (cache-reuse signature)", ch="─")
    print(f"  {'turn':<6}{'n':>6}{'ttft p50':>12}{'ttft p90':>12}{'itl p50':>11}"
          f"{'decode tps':>13}{'cache hit':>12}{'hist tok':>11}{'out tok':>10}")
    print(f"  {'-'*6}{'-'*6}{'-'*12}{'-'*12}{'-'*11}{'-'*13}{'-'*12}{'-'*11}{'-'*10}")
    for k in sorted(by_turn.keys(), key=lambda x: int(x)):
        d = by_turn[k]
        print(f"  {k:<6}{d.get('n', 0):>6}"
              f"{_fmt(d.get('ttft_p50'), 'ms'):>12}"
              f"{_fmt(d.get('ttft_p90'), 'ms'):>12}"
              f"{_fmt(d.get('itl_p50'), 'ms'):>11}"
              f"{(d.get('decode_tps_mean') or 0):>12.1f} "
              f"{(d.get('cache_hit_ratio_mean') or 0):>11.1%}"
              f"{(d.get('avg_history_tokens') or 0):>10.0f} "
              f"{(d.get('avg_output_tokens') or 0):>9.0f}")


def print_stop_reasons(data: dict) -> None:
    m = data.get("metrics") or {}
    reasons = m.get("stop_reasons") or {}
    if not reasons:
        return
    _section(f"{data.get('backend','?').upper()} — stop reasons", ch="─")
    total = sum(reasons.values())
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        print(f"  {reason or 'unknown':<20}{count:>8} ({pct:5.1f}%)")


def print_errors(data: dict, sample: int = 5) -> None:
    backend = data.get("backend", "?")
    errs: List[Tuple[str, int, str]] = []
    for conv in data.get("results", []) or []:
        for t in conv.get("turns", []) or []:
            if t.get("error"):
                errs.append((conv.get("conv_id", "?"), int(t.get("turn_idx", -1)), str(t["error"])))
    if not errs:
        return
    _section(f"{backend.upper()} — errors ({len(errs)})", ch="─")
    types = Counter(e.split(":")[0] for _, _, e in errs)
    print("  by type:")
    for typ, count in types.most_common():
        print(f"    {typ:<32}{count:>6}")
    print()
    print(f"  sample (first {min(sample, len(errs))}):")
    for conv_id, turn_idx, e in errs[:sample]:
        short = e if len(e) <= 140 else e[:140] + "…"
        print(f"    [{conv_id}/{turn_idx}] {short}")


def print_spec_phases(data: dict) -> None:
    if data.get("backend") != "kvboost":
        return
    phase_rows = []
    for conv in data.get("results", []) or []:
        for t in conv.get("turns", []) or []:
            spec = ((t.get("backend_telemetry") or {}).get("spec") or {})
            if spec:
                phase_rows.append(spec)
    if not phase_rows:
        return
    _section("KVBOOST — speculative phase breakdown (per-turn aggregates)", ch="─")
    acc_rates = [r["acceptance_rate"] for r in phase_rows if r.get("acceptance_rate") is not None]
    committed = [r["avg_committed_per_round"] for r in phase_rows if r.get("avg_committed_per_round") is not None]
    draft_ms    = [float(r.get("draft_time_ms", 0)) for r in phase_rows]
    verify_ms   = [float(r.get("verify_time_ms", 0)) for r in phase_rows]
    rollback_ms = [float(r.get("rollback_time_ms", 0)) for r in phase_rows]
    rounds      = [int(r.get("rounds", 0)) for r in phase_rows]
    accepted    = [int(r.get("accepted", 0)) for r in phase_rows]
    proposed    = [int(r.get("proposed", 0)) for r in phase_rows]

    print(f"  turns with spec activity: {len(phase_rows)}")
    if acc_rates:
        print(f"  acceptance rate (mean/p50/p90): "
              f"{np.mean(acc_rates):.3f} / {np.median(acc_rates):.3f} / {np.percentile(acc_rates, 90):.3f}")
    if committed:
        print(f"  avg committed/round (mean/p50): {np.mean(committed):.3f} / {np.median(committed):.3f}")
    if rounds:
        print(f"  rounds/turn (mean/max):         {np.mean(rounds):.1f} / {max(rounds)}")
    if sum(proposed) > 0:
        print(f"  proposed total: {sum(proposed)}  accepted total: {sum(accepted)}  "
              f"global acceptance: {sum(accepted)/sum(proposed):.3f}")

    total_phase = sum(draft_ms) + sum(verify_ms) + sum(rollback_ms)
    print()
    print(f"  Per-turn means:  draft={np.mean(draft_ms):.1f} ms  "
          f"verify={np.mean(verify_ms):.1f} ms  rollback={np.mean(rollback_ms):.2f} ms")
    if total_phase > 0:
        print(f"  Share of phase time across run:  "
              f"draft={100*sum(draft_ms)/total_phase:.1f}%  "
              f"verify={100*sum(verify_ms)/total_phase:.1f}%  "
              f"rollback={100*sum(rollback_ms)/total_phase:.2f}%")


def print_system_telemetry(data: dict) -> None:
    turns = _all_turns(data, only_ok=True)
    if not turns:
        return
    vram_peak = [t.get("vram_mb_peak") for t in turns if t.get("vram_mb_peak") is not None]
    vram_before = [t.get("vram_mb_before") for t in turns if t.get("vram_mb_before") is not None]
    rss = [t.get("host_rss_mb") for t in turns if t.get("host_rss_mb") is not None]
    if not (vram_peak or vram_before or rss):
        return
    _section(f"{data.get('backend','?').upper()} — system telemetry (per-turn)", ch="─")
    if vram_peak:
        print(f"  VRAM peak    : min={min(vram_peak):.0f}  p50={_pct(vram_peak,50):.0f}  "
              f"p90={_pct(vram_peak,90):.0f}  p99={_pct(vram_peak,99):.0f}  max={max(vram_peak):.0f} MB")
    if vram_before:
        print(f"  VRAM resident: min={min(vram_before):.0f}  p50={_pct(vram_before,50):.0f}  "
              f"p90={_pct(vram_before,90):.0f}  max={max(vram_before):.0f} MB")
    if rss:
        print(f"  Host RSS     : min={min(rss):.0f}  p50={_pct(rss,50):.0f}  "
              f"p90={_pct(rss,90):.0f}  max={max(rss):.0f} MB")


def print_backend_telemetry_keys(data: dict) -> None:
    """List what keys are present in `backend_telemetry` so you know what's
    available for further analysis. Doesn't print the values themselves —
    those vary per backend and would clutter the output."""
    turns = _all_turns(data, only_ok=True)
    if not turns:
        return
    key_paths: Counter = Counter()

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}.")
        else:
            key_paths[prefix.rstrip(".")] += 1

    for t in turns:
        bt = t.get("backend_telemetry") or {}
        if bt:
            walk(bt)
    if not key_paths:
        return
    _section(f"{data.get('backend','?').upper()} — backend telemetry keys (count out of {len(turns)} turns)", ch="─")
    for path, count in sorted(key_paths.items()):
        print(f"  {path:<48} {count:>6}")


# ── Multi-file comparison (lighter than compare.py — text only) ─────────

def print_comparison(datasets: List[Tuple[dict, Path]]) -> None:
    if len(datasets) < 2:
        return
    _section("COMPARISON", ch="=")
    backends = [d.get("backend", p.stem) for d, p in datasets]
    cols = "  {:<28}" + "{:>16}" * len(datasets)
    print(cols.format("metric", *backends))
    print("  " + "-" * 28 + ("-" * 16) * len(datasets))

    def get(d, path):
        cur = d
        for k in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(k)
            else:
                return None
        return cur

    rows = [
        ("n_conversations",     "metrics.n_conversations",            "int"),
        ("n_turns",             "metrics.n_turns_total",              "int"),
        ("n_errors",            "metrics.n_turns_errored",            "int"),
        ("wall_s",              "wall_s",                             "float"),
        ("TTFT p50 (ms)",       "metrics.overall.ttft_p50_ms",        "float"),
        ("TTFT p90 (ms)",       "metrics.overall.ttft_p90_ms",        "float"),
        ("TTFT p99 (ms)",       "metrics.overall.ttft_p99_ms",        "float"),
        ("ITL p50 (ms/tok)",    "metrics.overall.itl_p50_ms",         "float"),
        ("decode tok/s mean",   "metrics.overall.decode_tps_mean",    "float"),
        ("output tok/s (wall)", "metrics.overall.output_token_throughput", "float"),
        ("cache hit ratio",     "metrics.overall.avg_cache_hit_ratio", "ratio"),
        ("spec acceptance",     "metrics.overall.spec_acceptance_rate", "ratio"),
        ("peak VRAM max (MB)",  "metrics.overall.peak_vram_mb_max",   "float"),
    ]
    for label, path, kind in rows:
        vals = [get(d, path) for d, _ in datasets]
        cells = []
        for v in vals:
            if v is None:
                cells.append("—")
            elif kind == "int":
                cells.append(str(int(v)))
            elif kind == "ratio":
                cells.append(f"{v:.1%}")
            else:
                cells.append(f"{v:.2f}")
        print(cols.format(label, *cells))


# ── Per-turn CSV export ─────────────────────────────────────────────────

def write_csv(datasets: List[Tuple[dict, Path]], csv_path: Path) -> None:
    fields = [
        "backend", "conv_id", "turn_idx", "n_turns_total",
        "turn_start_iso",
        "history_tokens", "prompt_tokens", "cached_tokens", "cache_hit_ratio",
        "ttft_ms", "total_ms", "decode_ms",
        "output_tokens", "itl_ms", "decode_tps",
        "spec_accepted", "spec_proposed", "spec_rounds",
        "stop_reason", "error",
        "vram_mb_before", "vram_mb_after", "vram_mb_peak", "host_rss_mb",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        total = 0
        for d, _ in datasets:
            backend = d.get("backend", "?")
            for conv in d.get("results", []) or []:
                for t in conv.get("turns", []) or []:
                    row = {"backend": backend, "conv_id": conv.get("conv_id")}
                    for fld in fields:
                        if fld in row:
                            continue
                        row[fld] = t.get(fld)
                    w.writerow(row)
                    total += 1
    print(f"\nWrote per-turn CSV: {csv_path}  ({total} rows)")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="Result JSON file(s).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Write per-turn data across all input files to this CSV.")
    p.add_argument("--brief", action="store_true",
                   help="Skip per-file deep-dive; show only multi-file comparison.")
    p.add_argument("--no-distributions", action="store_true",
                   help="Skip the percentile distribution table.")
    p.add_argument("--no-telemetry-keys", action="store_true",
                   help="Skip the backend_telemetry key inventory.")
    args = p.parse_args()

    datasets: List[Tuple[dict, Path]] = []
    for path in args.paths:
        if not path.exists():
            print(f"WARN: {path} does not exist, skipping")
            continue
        datasets.append((load(path), path))

    if not datasets:
        print("No valid result files loaded.")
        return 1

    if not args.brief:
        for data, path in datasets:
            print_fingerprint(data, path)
            print_headline(data)
            if not args.no_distributions:
                print_distributions(data)
            print_per_turn(data)
            print_stop_reasons(data)
            print_errors(data)
            print_spec_phases(data)
            print_system_telemetry(data)
            if not args.no_telemetry_keys:
                print_backend_telemetry_keys(data)

    print_comparison(datasets)

    if args.csv:
        write_csv(datasets, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
