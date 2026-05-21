#!/usr/bin/env python3
"""
3-way comparison plot + table for the ShareGPT benchmark.

Reads results/kvboost.json, results/vllm.json, results/llamacpp.json
(any subset is fine — backends with no JSON are skipped) and emits:
  * results/3way_summary.png   — bar charts + per-turn TTFT chart
  * stdout                     — side-by-side table

Usage:
  python compare.py
  python compare.py --inputs results/kvboost.json results/vllm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

RESULTS_DIR = Path(__file__).resolve().parent / "results"

BACKEND_LABELS = {
    "kvboost":  "KVBoost\n(cacheblend + spec)",
    "vllm":     "vLLM\n(prefix-cache + spec)",
    "llamacpp": "llama.cpp\n(prefix-cache + spec)",
}
BACKEND_COLORS = {
    "kvboost":  "#2ecc71",
    "vllm":     "#3498db",
    "llamacpp": "#e67e22",
}


def load_payloads(inputs: List[Path]) -> Dict[str, dict]:
    out = {}
    for p in inputs:
        if not p.exists():
            print(f"  (skip) {p} not found")
            continue
        with open(p) as f:
            data = json.load(f)
        out[data["backend"]] = data
    return out


def print_table(payloads: Dict[str, dict]) -> None:
    backends = list(payloads.keys())
    if not backends:
        print("No payloads to compare.")
        return

    metric_keys = [
        ("ttft_p50_ms",            "TTFT p50 (ms)",          "{:.1f}"),
        ("ttft_p90_ms",            "TTFT p90 (ms)",          "{:.1f}"),
        ("ttft_p99_ms",            "TTFT p99 (ms)",          "{:.1f}"),
        ("itl_p50_ms",             "ITL p50 (ms/tok)",       "{:.2f}"),
        ("itl_p90_ms",             "ITL p90 (ms/tok)",       "{:.2f}"),
        ("decode_tps_mean",        "Decode tok/s (mean)",    "{:.2f}"),
        ("avg_cache_hit_ratio",    "Cache hit ratio",        "{:.1%}"),
        ("avg_cached_tokens",      "Cached tokens (avg)",    "{:.0f}"),
        ("avg_prompt_tokens",      "Prompt tokens (avg)",    "{:.0f}"),
        ("avg_output_tokens",      "Output tokens (avg)",    "{:.0f}"),
        ("request_throughput_rps", "Request throughput rps", "{:.3f}"),
        ("output_token_throughput","Output tok/s (wall)",    "{:.2f}"),
        ("spec_acceptance_rate",   "Spec acceptance rate",   "{:.1%}"),
    ]

    header = f"  {'Metric':<26}" + "".join(f"{BACKEND_LABELS.get(b, b).splitlines()[0]:>20}" for b in backends)
    print(f"\n{'=' * len(header)}")
    print("  ShareGPT 3-way comparison (500 samples)")
    print("=" * len(header))
    print(header)
    print(f"  {'-' * 26}" + "".join(f"{'-' * 19:>20}" for _ in backends))
    for key, label, fmt in metric_keys:
        row = f"  {label:<26}"
        for b in backends:
            v = payloads[b]["metrics"]["overall"].get(key)
            row += f"{(fmt.format(v) if v is not None else '—'):>20}"
        print(row)
    print(f"  {'-' * 26}" + "".join(f"{'-' * 19:>20}" for _ in backends))
    row = f"  {'Wall time (s)':<26}"
    for b in backends:
        row += f"{payloads[b].get('wall_s', 0):>20.1f}"
    print(row)
    print("=" * len(header) + "\n")


def plot_comparison(payloads: Dict[str, dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    backends = list(payloads.keys())
    labels = [BACKEND_LABELS.get(b, b) for b in backends]
    colors = [BACKEND_COLORS.get(b, "#888") for b in backends]

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "ShareGPT 3-way benchmark — KVBoost vs vLLM vs llama.cpp\n"
        "Qwen2.5-7B-Instruct target + Qwen2.5-1.5B-Instruct draft (γ=5)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30)

    def _bar(ax, key, title, fmt):
        vals = [payloads[b]["metrics"]["overall"].get(key) or 0 for b in backends]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    fmt.format(v), ha="center", va="bottom", fontsize=9)

    _bar(fig.add_subplot(gs[0, 0]), "ttft_p50_ms", "TTFT p50 (ms) — lower is better", "{:.0f}")
    _bar(fig.add_subplot(gs[0, 1]), "ttft_p90_ms", "TTFT p90 (ms) — lower is better", "{:.0f}")
    _bar(fig.add_subplot(gs[0, 2]), "decode_tps_mean", "Decode tok/s — higher is better", "{:.1f}")
    _bar(fig.add_subplot(gs[1, 0]), "itl_p50_ms", "ITL p50 (ms/tok) — lower is better", "{:.2f}")
    _bar(fig.add_subplot(gs[1, 1]), "output_token_throughput",
         "Output tok/s wall — higher is better", "{:.1f}")

    # Per-turn TTFT lines — the money chart for prefix-reuse comparison.
    ax = fig.add_subplot(gs[1, 2])
    for b in backends:
        bt = payloads[b]["metrics"].get("by_turn", {})
        if not bt:
            continue
        keys = sorted(bt.keys(), key=lambda x: int(x))
        xs = [int(k) + 1 for k in keys]
        ys = [bt[k]["ttft_p50"] for k in keys]
        ax.plot(xs, ys, "o-", label=BACKEND_LABELS.get(b, b).replace("\n", " "),
                color=BACKEND_COLORS.get(b, "#888"), lw=2)
    ax.set_xlabel("Turn number")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("TTFT vs turn number (money chart)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=[
        str(RESULTS_DIR / "kvboost.json"),
        str(RESULTS_DIR / "vllm.json"),
        str(RESULTS_DIR / "llamacpp.json"),
    ])
    parser.add_argument("--plot", default=str(RESULTS_DIR / "3way_summary.png"))
    args = parser.parse_args()

    payloads = load_payloads([Path(p) for p in args.inputs])
    if not payloads:
        raise SystemExit("No result JSONs found. Run the backend scripts first.")

    print_table(payloads)
    plot_comparison(payloads, Path(args.plot))


if __name__ == "__main__":
    main()
