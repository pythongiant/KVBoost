#!/usr/bin/env python3
"""
Generate benchmark plots from a saved sharegpt_replay.json result file.

Usage:
  python plot_results.py                          # uses results/sharegpt_replay.json
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
    by_turn = metrics["by_turn"]
    turn_indices = sorted(by_turn.keys(), key=int)
    turns_labels = [int(i) + 1 for i in turn_indices]

    baseline_ttfts  = [by_turn[i]["avg_baseline_ttft_ms"] for i in turn_indices]
    kvboost_ttfts   = [by_turn[i]["avg_kvboost_ttft_ms"]  for i in turn_indices]
    reuse_ratios    = [by_turn[i]["avg_reuse_ratio"]       for i in turn_indices]
    speedups        = [by_turn[i]["speedup"]               for i in turn_indices]
    history_tokens  = [by_turn[i]["avg_history_tokens"]    for i in turn_indices]
    sample_counts   = [by_turn[i]["n"]                     for i in turn_indices]

    overall = metrics["overall"]
    n_conv  = metrics["n_conversations"]
    n_turns = metrics["n_turns_total"]

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"KVBoost ShareGPT Replay — {model}\n"
        f"{n_conv} conversations · {n_turns} turns",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0 ──────────────────────────────────────────────────────

    # 1. TTFT vs turn number (main money chart)
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(turns_labels, baseline_ttfts, "o-", lw=2,
            label="Baseline (no cache)", color="#e74c3c")
    ax.plot(turns_labels, kvboost_ttfts,  "s-", lw=2,
            label="KVBoost (cached)",    color="#2ecc71")
    ax.fill_between(turns_labels, kvboost_ttfts, baseline_ttfts,
                    alpha=0.12, color="#2ecc71", label="Savings region")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("Avg TTFT (ms)", fontsize=10)
    ax.set_title("TTFT vs Turn Number  (ShareGPT real conversations)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turns_labels)

    # 2. Summary stats box
    ax_s = fig.add_subplot(gs[0, 2])
    ax_s.axis("off")
    lines = [
        ("Speedup (mean)",      f"{overall['speedup_mean']:.2f}×"),
        ("Speedup (p50)",       f"{overall['speedup_p50']:.2f}×"),
        ("Speedup (p90)",       f"{overall['speedup_p90']:.2f}×"),
        ("Baseline TTFT p50",   f"{overall['baseline_ttft_p50']:.1f} ms"),
        ("KVBoost TTFT p50",    f"{overall['kvboost_ttft_p50']:.1f} ms"),
        ("Baseline TTFT p90",   f"{overall['baseline_ttft_p90']:.1f} ms"),
        ("KVBoost TTFT p90",    f"{overall['kvboost_ttft_p90']:.1f} ms"),
        ("Avg KV reuse ratio",  f"{overall['avg_kv_reuse_ratio']:.1%}"),
        ("Baseline throughput", f"{overall['baseline_throughput_rps']} rps"),
        ("KVBoost throughput",  f"{overall['kvboost_throughput_rps']} rps"),
    ]
    col_labels = ["Metric", "Value"]
    table = ax_s.table(
        cellText=lines,
        colLabels=col_labels,
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
            cell.set_facecolor("#eafaf1")
        else:
            cell.set_facecolor("#fdfefe")
    ax_s.set_title("Overall Summary", fontsize=11, pad=8)

    # ── Row 1 ──────────────────────────────────────────────────────

    # 3. Speedup per turn (bar)
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(turns_labels, speedups, color="#3498db", alpha=0.85, width=0.6)
    ax.axhline(y=1.0, color="#e74c3c", linestyle="--", lw=1.2, label="No speedup (1×)")
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{sp:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("TTFT Speedup (×)", fontsize=10)
    ax.set_title("KVBoost Speedup per Turn", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(turns_labels)

    # 4. KV reuse ratio per turn
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(turns_labels, [r * 100 for r in reuse_ratios],
            "D-", lw=2, color="#9b59b6", markersize=7)
    ax.fill_between(turns_labels, [r * 100 for r in reuse_ratios],
                    alpha=0.15, color="#9b59b6")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("KV Reuse Ratio (%)", fontsize=10)
    ax.set_title("Cache Hit Rate by Turn", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turns_labels)

    # 5. Avg history tokens + sample count
    ax = fig.add_subplot(gs[1, 2])
    color_tok = "#e67e22"
    ax.plot(turns_labels, history_tokens, "^-", lw=2,
            color=color_tok, markersize=7, label="Avg history tokens")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("Avg History Tokens", fontsize=10, color=color_tok)
    ax.tick_params(axis="y", labelcolor=color_tok)
    ax2 = ax.twinx()
    ax2.bar(turns_labels, sample_counts, alpha=0.25, color="#95a5a6", width=0.5,
            label="Conversations at turn")
    ax2.set_ylabel("Conversations (n)", fontsize=9, color="#7f8c8d")
    ax2.tick_params(axis="y", labelcolor="#7f8c8d")
    ax.set_title("Context Growth & Sample Size", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turns_labels)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sharegpt_ttft_vs_turn.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", nargs="?",
                        default=str(RESULTS_DIR / "sharegpt_replay.json"))
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
