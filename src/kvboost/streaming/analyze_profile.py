"""Aggregate a ``KVBOOST_PROFILE`` JSONL trace into a steady-state breakdown.

Reads the JSONL emitted by :class:`StreamingProfiler.flush` and prints a
markdown table that maps directly onto the Phase-3 decision criteria in
the kernel-fusion plan:

- Per-projection forward time (q/k/v/o, gate/up/down) — drives the
  "gate+up concat" and "QKV concat" rows.
- DMA wait per layer (``scheduler.before_layer`` sum) — drives the
  "triple-buffer / async dequant" row.
- Hook-rebind cost — drives the "refactor rebind" row.
- Total per-token cost as the denominator for all percentages.

The first iteration is dropped from the steady-state aggregate because
the pipeline-prime cost of the first forward is dominated by initial
host→device DMAs, not by the per-token kernels we're trying to optimize.

Usage::

    python -m kvboost.streaming.analyze_profile /tmp/kvboost_trace.jsonl

    # or programmatically:
    from kvboost.streaming.analyze_profile import summarize
    print(summarize("/tmp/kvboost_trace.jsonl"))
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class _RegionStats:
    name: str
    sub_path: Optional[str]
    sum_ms: float = 0.0
    count: int = 0

    @property
    def mean_ms(self) -> float:
        return self.sum_ms / self.count if self.count else 0.0


def _iter_records(path: str) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _aggregate(records: Iterable[dict], drop_first_iteration: bool) -> tuple[
    dict[str, _RegionStats],            # region totals keyed by display name
    dict[int, list[dict]],              # per-iteration records (in order)
]:
    """Bucket records into per-region stats and per-iteration lists.

    qlinear.forward is split by sub_path; everything else is keyed by
    region name alone. Sums are computed over per-iteration totals
    (so we mean across tokens, not across raw kernel invocations).
    """
    by_iter: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_iter[rec["iteration"]].append(rec)

    iterations = sorted(by_iter.keys())
    if drop_first_iteration and len(iterations) > 1:
        iterations = iterations[1:]

    # Per-iteration sum per region key, then mean across iterations.
    per_iter_totals: dict[str, list[float]] = defaultdict(list)
    for it in iterations:
        per_key_sum: dict[str, float] = defaultdict(float)
        for rec in by_iter[it]:
            if rec["dt_ms"] is None:
                continue
            key = _display_key(rec)
            per_key_sum[key] += rec["dt_ms"]
        for key, total_ms in per_key_sum.items():
            per_iter_totals[key].append(total_ms)

    stats: dict[str, _RegionStats] = {}
    for key, samples in per_iter_totals.items():
        sub_path = key.split("::", 1)[1] if "::" in key else None
        s = _RegionStats(name=key, sub_path=sub_path)
        s.sum_ms = sum(samples)
        s.count = len(samples)
        stats[key] = s

    return stats, by_iter


def _display_key(rec: dict) -> str:
    name = rec["name"]
    sub_path = rec.get("sub_path")
    if name == "qlinear.forward" and sub_path:
        # Just the trailing projection name (e.g. "self_attn.q_proj" → "q_proj").
        leaf = sub_path.rsplit(".", 1)[-1]
        return f"qlinear.forward::{leaf}"
    return name


def summarize(path: str, *, drop_first_iteration: bool = True) -> str:
    """Return a markdown summary of the trace at ``path``."""
    stats, by_iter = _aggregate(_iter_records(path), drop_first_iteration)

    if not stats:
        return f"No records in {path}.\n"

    total = stats.get("model.forward.total")
    total_mean = total.mean_ms if total else 0.0
    iterations = sorted(by_iter.keys())
    if drop_first_iteration and len(iterations) > 1:
        steady_iters = iterations[1:]
    else:
        steady_iters = iterations

    lines: list[str] = []
    lines.append(f"# Streaming profile: {path}")
    lines.append("")
    lines.append(
        f"- Iterations captured: **{len(iterations)}** "
        f"(steady-state mean over {len(steady_iters)})"
    )
    if total_mean:
        lines.append(f"- Mean per-token total: **{total_mean:7.2f} ms** "
                     f"→ **{1000.0 / total_mean:5.2f} tok/s**")
    lines.append("")

    def _pct(ms: float) -> str:
        if total_mean <= 0:
            return "    —"
        return f"{100.0 * ms / total_mean:5.1f}%"

    # Group qlinear projections together; sort by mean desc within a group.
    qlinear_rows: list[tuple[str, float]] = []
    other_rows: list[tuple[str, float]] = []
    for key, s in stats.items():
        if key.startswith("qlinear.forward::"):
            qlinear_rows.append((key.split("::", 1)[1], s.mean_ms))
        else:
            other_rows.append((key, s.mean_ms))

    qlinear_rows.sort(key=lambda x: x[1], reverse=True)
    other_rows.sort(key=lambda x: x[1], reverse=True)
    qlinear_total_ms = sum(ms for _, ms in qlinear_rows)

    lines.append("| Region | Mean ms / token | % of total |")
    lines.append("|---|---:|---:|")
    for key, ms in other_rows:
        lines.append(f"| `{key}` | {ms:7.2f} | {_pct(ms)} |")
    if qlinear_rows:
        lines.append(
            f"| `qlinear.forward` *(sum)* | {qlinear_total_ms:7.2f} | "
            f"{_pct(qlinear_total_ms)} |"
        )
        for sub, ms in qlinear_rows:
            lines.append(f"|  └ `{sub}` | {ms:7.2f} | {_pct(ms)} |")
    lines.append("")

    lines.append("## Decision hints")
    hints = _decision_hints(stats, qlinear_total_ms, total_mean)
    if not hints:
        lines.append("- No fusion threshold tripped — profile harder "
                     "(longer prompt, batched, or look at flash_attn).")
    else:
        lines.extend(f"- {h}" for h in hints)
    lines.append("")

    return "\n".join(lines)


def _decision_hints(
    stats: dict[str, _RegionStats],
    qlinear_total_ms: float,
    total_mean: float,
) -> list[str]:
    """Mechanically apply the Phase-3 decision table from the plan."""
    if total_mean <= 0:
        return []

    by_sub: dict[str, float] = {}
    for key, s in stats.items():
        if key.startswith("qlinear.forward::"):
            by_sub[key.split("::", 1)[1]] = s.mean_ms

    hints: list[str] = []
    pct = lambda x: 100.0 * x / total_mean  # noqa: E731

    gate_up = by_sub.get("gate_proj", 0) + by_sub.get("up_proj", 0)
    if pct(gate_up) > 25:
        hints.append(
            f"Gate+Up concat: gate+up = {pct(gate_up):.1f}% > 25% — "
            "one fused matmul replaces two."
        )

    qkv = by_sub.get("q_proj", 0) + by_sub.get("k_proj", 0) + by_sub.get("v_proj", 0)
    if pct(qkv) > 20:
        hints.append(
            f"QKV concat: q+k+v = {pct(qkv):.1f}% > 20% — "
            "fuse the three projections after gate+up is done."
        )

    dma = stats.get("scheduler.before_layer")
    dma_ms = dma.mean_ms if dma else 0.0
    if pct(dma_ms) > 50:
        hints.append(
            f"Async dequant / triple-buffer slots: DMA wait = {pct(dma_ms):.1f}% > 50% — "
            "compute is starved waiting on PCIe; kernel fusion alone won't help."
        )

    down = by_sub.get("down_proj", 0)
    if pct(down) > 15:
        hints.append(
            f"Fused MLP: down_proj alone = {pct(down):.1f}% > 15% — "
            "candidate for residual-in-place fusion (after gate+up lands)."
        )

    rebind = stats.get("hook.rebind")
    rebind_ms = rebind.mean_ms if rebind else 0.0
    if pct(rebind_ms) > 5:
        hints.append(
            f"Refactor rebind: hook.rebind = {pct(rebind_ms):.1f}% > 5% — "
            "Python overhead per layer; batch the rebinds."
        )

    return hints


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate a KVBOOST_PROFILE JSONL trace.")
    p.add_argument("path", help="JSONL trace file written by StreamingProfiler.flush()")
    p.add_argument(
        "--include-first",
        action="store_true",
        help="Include iteration 1 (TTFT) in the steady-state aggregate.",
    )
    args = p.parse_args(argv)
    sys.stdout.write(summarize(args.path, drop_first_iteration=not args.include_first))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
