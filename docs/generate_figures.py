#!/usr/bin/env python3
"""Generate README figures from experiment result JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks_and_experiments" / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
COLORS = {"baseline": "#4c72b0", "prefix_cache": "#dd8452", "chunk_kv": "#55a868"}


# ── Figure 1: TTFT across modes (Experiment 1) ───────────────────────────
def fig_ttft_modes():
    with open(RESULTS / "01_scale_models.json") as f:
        data = json.load(f)[0]["workloads"]

    workloads = ["system_prompt_reuse", "rag_doc_reuse"]
    labels = ["System Prompt", "RAG Document"]
    modes = ["baseline", "prefix_cache", "chunk_kv_reuse"]
    mode_labels = ["Baseline", "Prefix Cache", "Chunk KV Reuse"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(workloads))
    width = 0.25

    for i, (mode, ml) in enumerate(zip(modes, mode_labels)):
        vals = [data[w][mode]["ttft_ms"]["mean"] for w in workloads]
        errs = [data[w][mode]["ttft_ms"]["stdev"] for w in workloads]
        color = list(COLORS.values())[i]
        bars = ax.bar(x + i * width, vals, width, label=ml, color=color,
                      yerr=errs, capsize=4, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token by Generation Mode")
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "ttft_modes.png", dpi=150)
    plt.close(fig)
    print("  -> ttft_modes.png")


# ── Figure 2: Latency breakdown (Experiment 2) ───────────────────────────
def fig_latency_breakdown():
    with open(RESULTS / "02_latency_breakdown.json") as f:
        data = json.load(f)["rag_doc_reuse"]

    stages = ["cache_lookup_ms", "selective_recompute_ms",
              "cpu_to_gpu_transfer_ms", "live_prefill_ms", "decode_ms"]
    stage_labels = ["Cache Lookup", "Selective Recompute",
                    "CPU→GPU Transfer", "Live Prefill", "Decode"]
    colors = ["#8dd3c7", "#fb8072", "#fdb462", "#80b1d3", "#b3de69"]

    modes = ["baseline", "chunk_kv_reuse"]
    mode_labels = ["Baseline", "Chunk KV Reuse"]

    fig, ax = plt.subplots(figsize=(6, 5))

    for mi, (mode, ml) in enumerate(zip(modes, mode_labels)):
        bottom = 0
        for si, (stage, sl) in enumerate(zip(stages, stage_labels)):
            val = data[mode][stage]["mean"]
            ax.barh(mi, val, left=bottom, color=colors[si],
                    label=sl if mi == 0 else None, edgecolor="white", linewidth=0.5)
            if val > 40:
                ax.text(bottom + val / 2, mi, f"{val:.0f}",
                        ha="center", va="center", fontsize=8)
            bottom += val

    ax.set_yticks([0, 1])
    ax.set_yticklabels(mode_labels)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Latency Breakdown — RAG Workload")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "latency_breakdown.png", dpi=150)
    plt.close(fig)
    print("  -> latency_breakdown.png")


# ── Figure 3: Multi-turn conversation (Experiment 5) ─────────────────────
def fig_multi_turn():
    with open(RESULTS / "05_realistic_workloads.json") as f:
        data = json.load(f)["multi_turn"]

    turns_b = [d for d in data if d["mode"] == "baseline"]
    turns_c = [d for d in data if d["mode"] == "chunk_kv_reuse"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    t = [d["turn"] for d in turns_b]
    ax1.plot(t, [d["ttft_ms"] for d in turns_b], "o-",
             color=COLORS["baseline"], label="Baseline TTFT", linewidth=2)
    ax1.plot(t, [d["ttft_ms"] for d in turns_c], "s-",
             color=COLORS["chunk_kv"], label="Chunk KV TTFT", linewidth=2)
    ax1.set_xlabel("Conversation Turn")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Multi-Turn Conversation: TTFT & Cache Reuse")

    ax2 = ax1.twinx()
    ax2.fill_between(t, [d["kv_reuse_ratio"] * 100 for d in turns_c],
                     alpha=0.15, color=COLORS["chunk_kv"])
    ax2.plot(t, [d["kv_reuse_ratio"] * 100 for d in turns_c], "--",
             color=COLORS["chunk_kv"], alpha=0.5, label="Reuse %")
    ax2.set_ylabel("KV Reuse (%)")
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "multi_turn.png", dpi=150)
    plt.close(fig)
    print("  -> multi_turn.png")


# ── Figure 4: Cache warmup & traffic patterns (Experiment 9) ─────────────
def fig_cache_warmup():
    with open(RESULTS / "09_cache_hit_simulation.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: warmup curve
    curve = data["warmup_curve"]
    buckets = np.array([10, 20, 30, 40, 50])
    rates = [r * 100 for r in curve["hit_rate_over_time"]]
    ax1.plot(buckets, rates, "o-", color=COLORS["chunk_kv"], linewidth=2, markersize=7)
    ax1.set_xlabel("Requests Processed")
    ax1.set_ylabel("Cache Hit Rate (%)")
    ax1.set_title("Cache Warmup Curve (Zipfian Traffic)")
    ax1.set_ylim(0, 70)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    for x, y in zip(buckets, rates):
        ax1.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    # Right: traffic patterns
    patterns = data["traffic_patterns"]
    names = ["uniform", "zipfian", "temporal"]
    labels = ["Uniform", "Zipfian", "Temporal"]
    hit_rates = [patterns[n]["overall_hit_rate"] * 100 for n in names]
    ttfts = [patterns[n]["ttft_ms"]["mean"] for n in names]

    bar_colors = ["#4c72b0", "#dd8452", "#55a868"]
    bars = ax2.bar(labels, hit_rates, color=bar_colors, edgecolor="white", width=0.5)
    ax2.set_ylabel("Cache Hit Rate (%)")
    ax2.set_title("Hit Rate by Traffic Pattern")
    ax2.set_ylim(0, 80)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

    for bar, rate, ttft in zip(bars, hit_rates, ttfts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{rate:.0f}%\n({ttft:.0f}ms)", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "cache_warmup.png", dpi=150)
    plt.close(fig)
    print("  -> cache_warmup.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_ttft_modes()
    fig_latency_breakdown()
    fig_multi_turn()
    fig_cache_warmup()
    print("Done.")
