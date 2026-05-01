#!/usr/bin/env python3
"""
Generate benchmark plots from a saved vllm_sharegpt_replay.json result file.

Usage:
  python plot_results.py                          # uses results/vllm_sharegpt_replay.json
  python plot_results.py path/to/results.json
  python plot_results.py results.json --out-dir /tmp/plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def plot(metrics: dict, model: str, output_dir: Path):
    by_turn      = metrics["by_turn"]
    turn_indices = sorted(by_turn.keys(), key=int)
    turn_labels  = [int(i) + 1 for i in turn_indices]

    avg_ttfts  = [by_turn[i]["avg_ttft_ms"]          for i in turn_indices]
    p50_ttfts  = [by_turn[i]["p50_ttft_ms"]           for i in turn_indices]
    p90_ttfts  = [by_turn[i]["p90_ttft_ms"]           for i in turn_indices]
    hit_ratios = [by_turn[i]["avg_cache_hit_ratio"]   for i in turn_indices]
    cached_tok = [by_turn[i]["avg_cached_tokens"]     for i in turn_indices]
    hist_tok   = [by_turn[i]["avg_history_tokens"]    for i in turn_indices]
    ns         = [by_turn[i]["n"]                     for i in turn_indices]

    ov      = metrics["overall"]
    n_conv  = metrics["n_conversations"]
    n_turns = metrics["n_turns_total"]

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"vLLM Prefix Cache — ShareGPT Replay  |  {model}\n"
        f"{n_conv} conversations · {n_turns} turns",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0 ──────────────────────────────────────────────────────

    # 1. TTFT vs turn (mean + p50/p90 band)
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(turn_labels, avg_ttfts, "o-",  lw=2,   color="#3498db", label="Avg TTFT")
    ax.plot(turn_labels, p50_ttfts, "s--", lw=1.2, color="#2ecc71", label="p50 TTFT")
    ax.plot(turn_labels, p90_ttfts, "^--", lw=1.2, color="#e74c3c", label="p90 TTFT")
    ax.fill_between(turn_labels, p50_ttfts, p90_ttfts, alpha=0.10, color="#3498db")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("TTFT (ms)", fontsize=10)
    ax.set_title("TTFT vs Turn Number  (vLLM prefix cache, ShareGPT)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turn_labels)

    # 2. Summary stats table
    ax_s = fig.add_subplot(gs[0, 2])
    ax_s.axis("off")

    def _fmt(val, fmt):
        return fmt.format(val) if val is not None else "—"

    rows = [
        ("TTFT p50 (ms)",       _fmt(ov.get("ttft_p50"),             "{:.1f}")),
        ("TTFT p90 (ms)",       _fmt(ov.get("ttft_p90"),             "{:.1f}")),
        ("TTFT p99 (ms)",       _fmt(ov.get("ttft_p99"),             "{:.1f}")),
        ("Cold TTFT p50 (ms)",  _fmt(ov.get("cold_ttft_p50"),        "{:.1f}")),
        ("Warm TTFT p50 (ms)",  _fmt(ov.get("warm_ttft_p50"),        "{:.1f}")),
        ("Warm/Cold speedup",   _fmt(ov.get("warm_vs_cold_speedup"), "{:.2f}×")),
        ("Avg cache hit ratio", _fmt(ov.get("avg_cache_hit_ratio"),  "{:.1%}")),
        ("Avg cached tokens",   _fmt(ov.get("avg_cached_tokens"),    "{:.0f}")),
        ("Avg prompt tokens",   _fmt(ov.get("avg_prompt_tokens"),    "{:.0f}")),
        ("Throughput (rps)",    _fmt(ov.get("throughput_rps"),       "{}")),
    ]
    table = ax_s.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 1:
            cell.set_facecolor("#eaf4fb")
        else:
            cell.set_facecolor("#fdfefe")
    ax_s.set_title("Overall Summary", fontsize=11, pad=8)

    # ── Row 1 ──────────────────────────────────────────────────────

    # 3. Prefix cache hit ratio per turn
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(turn_labels, [r * 100 for r in hit_ratios],
            "D-", lw=2, color="#9b59b6", markersize=7)
    ax.fill_between(turn_labels, [r * 100 for r in hit_ratios],
                    alpha=0.15, color="#9b59b6")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("Prefix Cache Hit (%)", fontsize=10)
    ax.set_title("Prefix Cache Hit Rate by Turn", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turn_labels)

    # 4. History tokens vs cached tokens
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(turn_labels, hist_tok,   "o-",  lw=2, color="#e67e22", label="Avg history tokens")
    ax.plot(turn_labels, cached_tok, "s--", lw=2, color="#27ae60", label="Avg cached tokens")
    ax.fill_between(turn_labels, cached_tok, hist_tok,
                    alpha=0.10, color="#e74c3c", label="Recomputed tokens")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("Tokens", fontsize=10)
    ax.set_title("History vs Cached Token Growth", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turn_labels)

    # 5. Sample count per turn
    ax = fig.add_subplot(gs[1, 2])
    bars = ax.bar(turn_labels, ns, color="#95a5a6", alpha=0.75, width=0.6)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ns) * 0.01,
                str(n), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("Conversations (n)", fontsize=10)
    ax.set_title("Conversations Reaching Each Turn", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(turn_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "vllm_sharegpt_ttft_vs_turn.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", nargs="?",
                        default=str(RESULTS_DIR / "vllm_sharegpt_replay.json"))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    model = data.get("model", "unknown")
    metrics = data["metrics"]
    plot(metrics, model, Path(args.out_dir))


if __name__ == "__main__":
    main()
