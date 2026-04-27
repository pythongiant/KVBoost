"""
Generate benchmark figures from experiment JSON.
Usage: python plot_benchmarks.py [path/to/experiment.json]
Saves figures to docs/figures/.
"""

import json
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "kvboost":          "#4C72B0",
    "vllm_prefixcache": "#DD8452",
    "baseline":         "#55A868",
    "cold":             "#5A9ECC",
    "warm":             "#E87041",
    "neutral":          "#888888",
}
LABELS = {
    "kvboost":          "KVBoost",
    "vllm_prefixcache": "vLLM (prefix cache)",
    "baseline":         "Baseline (HF)",
}

STYLE = dict(dpi=150, facecolor="white")


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_experiment(path=None):
    if path:
        return json.load(open(path))
    files = sorted(RESULTS_DIR.glob("experiment_*.json"))
    if not files:
        raise FileNotFoundError("No experiment JSON found in results/")
    return json.load(open(files[-1]))


def _save(fig, name):
    p = FIGURES_DIR / name
    fig.savefig(p, **STYLE, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.relative_to(REPO_ROOT)}")
    return p


# ── Figure 1: COLD vs WARM TTFT bar chart ─────────────────────────────────────

def fig_cold_warm_ttft(data):
    backends = ["baseline", "vllm_prefixcache", "kvboost"]
    cold_ms = [data["results"][b]["latency_stats"]["ttft_ms_cold"]["mean"]  for b in backends]
    warm_ms = [data["results"][b]["latency_stats"]["ttft_ms_warm"]["mean"]  for b in backends]

    x = np.arange(len(backends))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_cold = ax.bar(x - w/2, cold_ms, w, label="COLD (no cache)", color=COLORS["cold"],   edgecolor="white", linewidth=0.5)
    bars_warm = ax.bar(x + w/2, warm_ms, w, label="WARM (cached)",   color=COLORS["warm"],   edgecolor="white", linewidth=0.5)

    # value labels
    for bar in list(bars_cold) + list(bars_warm):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 8, f"{h:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[b] for b in backends], fontsize=11)
    ax.set_ylabel("Time to First Token (ms)", fontsize=11)
    ax.set_title("COLD vs WARM Time to First Token\n(lower is better)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(cold_ms) * 1.20)
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # speedup annotations for warm bars
    baseline_warm = warm_ms[0]
    for i, (b, wm) in enumerate(zip(backends[1:], warm_ms[1:]), 1):
        speedup = baseline_warm / wm
        ax.annotate(f"{speedup:.1f}×\nvs baseline",
                    xy=(x[i] + w/2, wm),
                    xytext=(x[i] + w/2 + 0.05, wm + 80),
                    fontsize=8, color="#333333",
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))

    fig.tight_layout()
    return _save(fig, "cold_warm_ttft.png")


# ── Figure 2: TTFT distribution (CDF) ─────────────────────────────────────────

def fig_ttft_cdf(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, qtype in zip(axes, ["COLD", "WARM"]):
        key = "ttft_ms_cold" if qtype == "COLD" else "ttft_ms_warm"
        for b in ["baseline", "vllm_prefixcache", "kvboost"]:
            samples = data["results"][b]["latency_samples"]
            vals = sorted(s["ttft_ms"] for s in samples if s["query_type"] == qtype)
            n = len(vals)
            cdf = np.arange(1, n + 1) / n
            ax.plot(vals, cdf * 100, label=LABELS[b], color=COLORS[b], linewidth=1.8)

        ax.set_xlabel("TTFT (ms)", fontsize=10)
        ax.set_ylabel("Percentile", fontsize=10)
        ax.set_title(f"{qtype} queries — TTFT CDF", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 101)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Cumulative Distribution of TTFT by Query Type", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "ttft_cdf.png")


# ── Figure 3: TTFT by context-length bucket ────────────────────────────────────

def fig_ttft_by_bucket(data):
    BUCKETS = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 99999)]
    BUCKET_LABELS = ["0–512", "512–1K", "1K–2K", "2K–4K", "4K+"]
    backends = ["baseline", "vllm_prefixcache", "kvboost"]

    def mean_ttft(samples, lo, hi, qtype):
        vals = [s["ttft_ms"] for s in samples
                if lo <= s["context_length"] < hi and s["query_type"] == qtype]
        return np.mean(vals) if vals else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, qtype in zip(axes, ["COLD", "WARM"]):
        x = np.arange(len(BUCKETS))
        width = 0.25
        offsets = np.linspace(-width, width, len(backends))

        for i, b in enumerate(backends):
            samples = data["results"][b]["latency_samples"]
            means = [mean_ttft(samples, lo, hi, qtype) for lo, hi in BUCKETS]
            valid_x = [xi + offsets[i] for xi, m in zip(x, means) if m is not None]
            valid_m = [m for m in means if m is not None]
            valid_labels = [BUCKET_LABELS[j] for j, m in enumerate(means) if m is not None]
            bars = ax.bar(valid_x, valid_m, width * 0.85, label=LABELS[b],
                          color=COLORS[b], edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_LABELS, fontsize=10)
        ax.set_xlabel("Context length (tokens)", fontsize=10)
        ax.set_ylabel("Mean TTFT (ms)", fontsize=10)
        ax.set_title(f"{qtype} — TTFT by context length", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("TTFT by Context-Length Bucket", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "ttft_by_bucket.png")


# ── Figure 4: KV reuse distribution ───────────────────────────────────────────

def fig_kv_reuse(data):
    samples = data["results"]["kvboost"]["latency_samples"]
    warm_reuse = [s["cache_reuse_ratio"] * 100
                  for s in samples if s["query_type"] == "WARM"]

    BUCKET_EDGES = [0, 20, 40, 60, 80, 100]
    counts, _ = np.histogram(warm_reuse, bins=BUCKET_EDGES)
    pcts = counts / len(warm_reuse) * 100
    labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, pcts, color=COLORS["kvboost"], edgecolor="white", linewidth=0.5, zorder=3)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    avg = np.mean(warm_reuse)
    ax.annotate(f"Mean reuse: {avg:.1f}%",
                xy=(3.5, max(pcts) * 0.85), fontsize=11, color="#333333",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF3FA", edgecolor="#4C72B0", linewidth=1))

    ax.set_xlabel("KV Cache Reuse (%)", fontsize=11)
    ax.set_ylabel("Share of warm queries (%)", fontsize=11)
    ax.set_title("KV Cache Reuse Distribution\n(KVBoost, warm queries)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "kv_reuse_distribution.png")


# ── Figure 5: Speedup summary ──────────────────────────────────────────────────

def fig_speedup_summary(data):
    baseline_mean  = data["results"]["baseline"]["latency_stats"]["ttft_ms_overall"]["mean"]
    baseline_cold  = data["results"]["baseline"]["latency_stats"]["ttft_ms_cold"]["mean"]
    baseline_warm  = data["results"]["baseline"]["latency_stats"]["ttft_ms_warm"]["mean"]

    rows = []
    for b in ["kvboost", "vllm_prefixcache"]:
        ls = data["results"][b]["latency_stats"]
        rows.append((
            LABELS[b],
            baseline_mean  / ls["ttft_ms_overall"]["mean"],
            baseline_cold  / ls["ttft_ms_cold"]["mean"],
            baseline_warm  / ls["ttft_ms_warm"]["mean"],
        ))

    categories = ["Overall", "COLD", "WARM"]
    x = np.arange(len(categories))
    w = 0.35
    offsets = [-w/2, w/2]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (label, ov, co, wa), off, b in zip(rows, offsets, ["kvboost", "vllm_prefixcache"]):
        vals = [ov, co, wa]
        bars = ax.bar(x + off, vals, w, label=label, color=COLORS[b], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{v:.1f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Speedup vs Baseline", fontsize=11)
    ax.set_title("TTFT Speedup vs HuggingFace Baseline\n(higher is better)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(baseline_warm / data["results"]["kvboost"]["latency_stats"]["ttft_ms_warm"]["mean"],
                       baseline_warm / data["results"]["vllm_prefixcache"]["latency_stats"]["ttft_ms_warm"]["mean"]) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "speedup_summary.png")


# ── Figure 6: Accuracy vs reuse rate (scatter) ────────────────────────────────

def fig_accuracy_vs_reuse(data):
    """Show per-pair accuracy grouped by reuse bucket for KVBoost warm queries."""
    acc_samples = data["results"]["kvboost"]["accuracy_samples"]
    lat_samples = data["results"]["kvboost"]["latency_samples"]

    # build reuse map: pair_group -> reuse_ratio
    reuse_map = {}
    for s in lat_samples:
        if s["query_type"] == "WARM":
            reuse_map[s["pair_group"]] = s["cache_reuse_ratio"] * 100

    # bucket and measure accuracy
    BUCKET_EDGES = [0, 20, 40, 60, 80, 100.001]
    BUCKET_LABELS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

    correct_by_bucket = [[] for _ in BUCKET_LABELS]
    for s in acc_samples:
        if s["query_type"] == "WARM" and s["pair_group"] in reuse_map:
            r = reuse_map[s["pair_group"]]
            for i, (lo, hi) in enumerate(zip(BUCKET_EDGES, BUCKET_EDGES[1:])):
                if lo <= r < hi:
                    correct_by_bucket[i].append(s["correct"])
                    break

    accs   = [100 * np.mean(c) if c else None for c in correct_by_bucket]
    counts = [len(c) for c in correct_by_bucket]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    valid = [(l, a, c) for l, a, c in zip(BUCKET_LABELS, accs, counts) if a is not None]
    labels_v, accs_v, counts_v = zip(*valid) if valid else ([], [], [])

    bars = ax.bar(labels_v, accs_v, color=COLORS["kvboost"], edgecolor="white", linewidth=0.5, zorder=3)
    for bar, acc, cnt in zip(bars, accs_v, counts_v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{acc:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=9)

    ax.axhline(data["results"]["baseline"]["accuracy_stats"]["accuracy_warm"] * 100,
               color=COLORS["baseline"], linestyle="--", linewidth=1.5, label="Baseline warm acc.")
    ax.axhline(data["results"]["kvboost"]["accuracy_stats"]["accuracy_cold"] * 100,
               color=COLORS["kvboost"], linestyle=":", linewidth=1.5, label="KVBoost cold acc.")

    ax.set_xlabel("KV Cache Reuse (%)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("KVBoost Accuracy by Reuse Level\n(warm queries)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(90, 102)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "accuracy_vs_reuse.png")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = _load_experiment(path)
    model = data.get("model", "")
    n = data.get("n_samples", "?")
    print(f"Loaded experiment: model={model}, n_samples={n}")
    print(f"Saving figures to {FIGURES_DIR.relative_to(REPO_ROOT)}/")

    fig_cold_warm_ttft(data)
    fig_ttft_cdf(data)
    fig_ttft_by_bucket(data)
    fig_kv_reuse(data)
    fig_speedup_summary(data)
    fig_accuracy_vs_reuse(data)

    print("Done.")


if __name__ == "__main__":
    main()
