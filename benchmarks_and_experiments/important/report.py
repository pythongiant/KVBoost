#!/usr/bin/env python3
"""
Report generator: reads an experiment JSON and produces a self-contained HTML report.

Usage:
    python report.py results/experiment_Qwen_Qwen2.5-3B_20260427_045300.json
    python report.py results/experiment_*.json   # latest auto-selected
"""

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _pct(v, digits=1):
    if v is None:
        return "N/A"
    return f"{v * 100:.{digits}f}%"


def _ms(v, digits=1):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f} ms"


def _mb(v, digits=1):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f} MB"


def _num(v, digits=2):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _delta_badge(val, ref, lower_is_better=False, pct=False):
    """Return HTML badge: green if better than ref, red if worse."""
    if val is None or ref is None or ref == 0:
        return ""
    ratio = (val - ref) / abs(ref)
    better = (ratio < 0) if lower_is_better else (ratio > 0)
    color = "#22c55e" if better else "#ef4444"
    sign = "+" if ratio > 0 else ""
    if pct:
        label = f"{sign}{ratio * 100:.1f}pp"
    else:
        label = f"{sign}{ratio * 100:.1f}%"
    return f'<span style="color:{color};font-size:0.82em;margin-left:4px">{label}</span>'


def _speedup_badge(val, ref):
    """Return colored speedup vs baseline."""
    if val is None or ref is None or val == 0:
        return ""
    sp = ref / val
    color = "#22c55e" if sp > 1.05 else ("#ef4444" if sp < 0.95 else "#6b7280")
    return f'<span style="color:{color}">{sp:.2f}×</span>'


def _bucket(tokens: int) -> str:
    if tokens < 512:   return "0–512"
    if tokens < 1024:  return "512–1K"
    if tokens < 2048:  return "1K–2K"
    if tokens < 4096:  return "2K–4K"
    return "4K+"

BUCKET_ORDER = ["0–512", "512–1K", "1K–2K", "2K–4K", "4K+"]


def _bucket_stats(samples: List[Dict], value_fn, filter_fn=None) -> Dict[str, Dict]:
    buckets: Dict[str, List] = {b: [] for b in BUCKET_ORDER}
    for s in samples:
        if filter_fn and not filter_fn(s):
            continue
        ctx = s.get("context_tokens") or s.get("context_length") or 0
        val = value_fn(s)
        if val is not None:
            buckets[_bucket(ctx)].append(val)
    out = {}
    for b, vals in buckets.items():
        if vals:
            out[b] = {"mean": statistics.mean(vals), "n": len(vals)}
    return out


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

BACKEND_COLORS = {
    "kvboost":        {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e40af", "chart": "#3b82f6"},
    "vllm_prefixcache": {"bg": "#fce7f3", "border": "#ec4899", "text": "#9d174d", "chart": "#ec4899"},
    "baseline":       {"bg": "#f3f4f6", "border": "#9ca3af", "text": "#374151", "chart": "#9ca3af"},
}

BACKEND_LABELS = {
    "kvboost": "KVBoost",
    "vllm_prefixcache": "vLLM (prefix cache)",
    "baseline": "Baseline (HF)",
}


def _backend_pill(b: str) -> str:
    c = BACKEND_COLORS.get(b, {"bg": "#f3f4f6", "border": "#9ca3af", "text": "#374151"})
    label = BACKEND_LABELS.get(b, b)
    return (
        f'<span style="background:{c["bg"]};border:1px solid {c["border"]};'
        f'color:{c["text"]};padding:2px 8px;border-radius:9999px;font-size:0.8em;'
        f'font-weight:600;white-space:nowrap">{label}</span>'
    )


def _section(title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p style="color:#6b7280;margin:4px 0 16px;font-size:0.9em">{subtitle}</p>' if subtitle else ""
    return f"""
<section style="margin-bottom:40px">
  <h2 style="font-size:1.25rem;font-weight:700;color:#111827;border-bottom:2px solid #e5e7eb;
             padding-bottom:8px;margin-bottom:4px">{title}</h2>
  {sub}
  {body}
</section>"""


def _table(headers: List[str], rows: List[List[str]], highlight_col: int = None) -> str:
    th_style = (
        'style="padding:10px 14px;text-align:left;font-size:0.78em;font-weight:700;'
        'text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;'
        'background:#f9fafb;border-bottom:2px solid #e5e7eb"'
    )
    ths = "".join(f"<th {th_style}>{h}</th>" for h in headers)
    tr_parts = []
    for ri, row in enumerate(rows):
        tds = []
        for ci, cell in enumerate(row):
            bg = "#fffbeb" if (highlight_col is not None and ci == highlight_col) else ("" if ri % 2 else "#fafafa")
            td_style = f'style="padding:9px 14px;border-bottom:1px solid #f3f4f6;background:{bg};vertical-align:middle"'
            tds.append(f"<td {td_style}>{cell}</td>")
        tr_parts.append(f"<tr>{''.join(tds)}</tr>")
    return (
        '<div style="overflow-x:auto">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
        f"<thead><tr>{ths}</tr></thead>"
        f"<tbody>{''.join(tr_parts)}</tbody>"
        "</table></div>"
    )


def _kv_grid(items: List[tuple]) -> str:
    cards = []
    for label, value, sub in items:
        cards.append(
            f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
            f'padding:14px 18px;min-width:140px">'
            f'<div style="font-size:0.75em;color:#6b7280;text-transform:uppercase;'
            f'letter-spacing:0.05em;margin-bottom:4px">{label}</div>'
            f'<div style="font-size:1.4em;font-weight:700;color:#111827">{value}</div>'
            f'<div style="font-size:0.8em;color:#6b7280;margin-top:2px">{sub}</div>'
            f'</div>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px">'
        + "".join(cards) + "</div>"
    )


def _bar_chart_svg(data: Dict[str, float], unit: str = "", width: int = 480,
                   lower_is_better: bool = False, ref_key: str = "baseline") -> str:
    """Horizontal bar chart as inline SVG."""
    if not data:
        return ""
    max_val = max(data.values())
    if max_val == 0:
        return ""
    bar_h, gap, pad_l, pad_r, pad_t, pad_b = 28, 8, 130, 60, 10, 10
    n = len(data)
    height = n * (bar_h + gap) + pad_t + pad_b
    ref = data.get(ref_key)
    lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:inherit;overflow:visible">'
    ]
    chart_w = width - pad_l - pad_r
    for i, (key, val) in enumerate(data.items()):
        y = pad_t + i * (bar_h + gap)
        bar_w = int(val / max_val * chart_w)
        color = BACKEND_COLORS.get(key, {}).get("chart", "#9ca3af")
        label = BACKEND_LABELS.get(key, key)
        # label
        lines.append(
            f'<text x="{pad_l - 6}" y="{y + bar_h // 2 + 4}" '
            f'text-anchor="end" font-size="12" fill="#374151">{label}</text>'
        )
        # bar
        lines.append(
            f'<rect x="{pad_l}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'fill="{color}" rx="4"/>'
        )
        # value label
        disp = f"{val:.1f}{unit}"
        if ref and ref != 0 and key != ref_key:
            ratio = ref / val if lower_is_better else val / ref
            sign = "+" if ratio > 1 else ""
            disp += f"  ({sign}{(ratio - 1) * 100:.0f}%)"
        lines.append(
            f'<text x="{pad_l + bar_w + 5}" y="{y + bar_h // 2 + 4}" '
            f'font-size="11" fill="#374151">{disp}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _spark_bucket_svg(bucket_data_by_backend: Dict[str, Dict[str, Dict]],
                      unit: str = "", lower_is_better: bool = False) -> str:
    """Grouped bar chart by context-length bucket."""
    buckets = [b for b in BUCKET_ORDER if any(b in bd for bd in bucket_data_by_backend.values())]
    if not buckets:
        return ""
    backends = list(bucket_data_by_backend.keys())
    n_b = len(buckets)
    n_be = len(backends)
    bar_w = 24
    gap_inner = 3
    gap_outer = 18
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 30
    all_vals = [
        v["mean"]
        for bd in bucket_data_by_backend.values()
        for v in bd.values()
    ]
    max_val = max(all_vals) if all_vals else 1
    chart_h = 160
    total_w = pad_l + n_b * (n_be * (bar_w + gap_inner) + gap_outer) + pad_r

    lines = [
        f'<svg width="{total_w}" height="{chart_h + pad_t + pad_b}" '
        f'xmlns="http://www.w3.org/2000/svg" style="font-family:inherit;overflow:visible">'
    ]
    # y-axis gridlines
    for tick in [0.25, 0.5, 0.75, 1.0]:
        y = pad_t + chart_h - int(tick * chart_h)
        lines.append(
            f'<line x1="{pad_l}" y1="{y}" x2="{total_w - pad_r}" y2="{y}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        val_label = f"{max_val * tick:.0f}"
        lines.append(
            f'<text x="{pad_l - 4}" y="{y + 4}" text-anchor="end" '
            f'font-size="10" fill="#9ca3af">{val_label}</text>'
        )

    for bi, bucket in enumerate(buckets):
        group_x = pad_l + bi * (n_be * (bar_w + gap_inner) + gap_outer)
        for bei, backend in enumerate(backends):
            bd = bucket_data_by_backend.get(backend, {})
            if bucket not in bd:
                continue
            val = bd[bucket]["mean"]
            bar_h = int(val / max_val * chart_h)
            x = group_x + bei * (bar_w + gap_inner)
            y = pad_t + chart_h - bar_h
            color = BACKEND_COLORS.get(backend, {}).get("chart", "#9ca3af")
            lines.append(
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
                f'fill="{color}" rx="3" opacity="0.85"/>'
            )
        # bucket label
        label_x = group_x + (n_be * (bar_w + gap_inner)) // 2
        lines.append(
            f'<text x="{label_x}" y="{pad_t + chart_h + 14}" '
            f'text-anchor="middle" font-size="10" fill="#6b7280">{bucket}</text>'
        )

    # legend
    lx = pad_l
    for backend in backends:
        color = BACKEND_COLORS.get(backend, {}).get("chart", "#9ca3af")
        label = BACKEND_LABELS.get(backend, backend)
        lines.append(f'<rect x="{lx}" y="{pad_t + chart_h + 22}" width="10" height="10" fill="{color}" rx="2"/>')
        lines.append(
            f'<text x="{lx + 13}" y="{pad_t + chart_h + 31}" font-size="10" fill="#374151">{label}</text>'
        )
        lx += len(label) * 6.5 + 24

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_overview(data: Dict, backends: List[str]) -> str:
    results = data["results"]
    cfg = data.get("config", {})
    kv = cfg.get("kvboost", {})
    vl = cfg.get("vllm", {})

    run_badges = " ".join(
        f'{_backend_pill(b)} <span style="color:{"#22c55e" if results[b].get("run_ok") else "#ef4444"};'
        f'font-size:0.8em">{"✓ ok" if results[b].get("run_ok") else "✗ failed"}</span>'
        for b in backends
    )

    items = [
        ("Model", data["model"].split("/")[-1], data["model"]),
        ("Samples", str(data["n_samples"]), f"per backend × {len(backends)} backends"),
        ("Max context", f"{data['max_context_tokens']:,} tok", "token filter"),
        ("Experiment", data.get("experiment_timestamp", ""), ""),
    ]
    grid = _kv_grid(items)

    cfg_rows = []
    if kv:
        cfg_rows.append(["KVBoost cache", f"{kv.get('max_cache_bytes', 0)/1e9:.1f} GB"])
        cfg_rows.append(["KVBoost recency window", f"{kv.get('recency_window_chunks')} chunks"])
        cfg_rows.append(["KVBoost recompute strategy", kv.get("recompute_strategy", "")])
        cfg_rows.append(["KVBoost boundary window", str(kv.get("chunk_boundary_window", 0))])
        cfg_rows.append(["KVBoost overlap-k", str(kv.get("overlap_k", 0))])
        cfg_rows.append(["KVBoost sink tokens", str(kv.get("sink_tokens", 0))])
    if vl:
        cfg_rows.append(["vLLM gpu_memory_utilization", str(vl.get("gpu_memory_utilization"))])
        cfg_rows.append(["vLLM enforce_eager", str(vl.get("enforce_eager"))])
        cfg_rows.append(["vLLM max_num_seqs", str(vl.get("max_num_seqs"))])

    cfg_table = _table(["Parameter", "Value"], cfg_rows) if cfg_rows else ""

    body = f"""
{grid}
<p style="margin-bottom:8px"><strong>Backends:</strong> {run_badges}</p>
<details style="margin-top:12px">
  <summary style="cursor:pointer;color:#3b82f6;font-size:0.88em">Show experiment config</summary>
  <div style="margin-top:10px">{cfg_table}</div>
</details>"""
    return _section("Experiment Overview", body)


def _build_accuracy(data: Dict, backends: List[str]) -> str:
    results = data["results"]
    baseline_cold = results.get("baseline", {}).get("accuracy_stats", {}).get("accuracy_cold")
    baseline_overall = results.get("baseline", {}).get("accuracy_stats", {}).get("accuracy_overall")

    rows = []
    chart_data_overall = {}
    chart_data_cold = {}
    chart_data_warm = {}

    for b in backends:
        s = results[b].get("accuracy_stats", {})
        if not s:
            rows.append([_backend_pill(b), "—", "—", "—", "—", "—"])
            continue
        n = s.get("n_total", 0)
        overall = s.get("accuracy_overall")
        cold = s.get("accuracy_cold")
        warm = s.get("accuracy_warm")
        reuse = s.get("avg_kv_reuse_pct_warm")

        chart_data_overall[b] = (overall or 0) * 100
        chart_data_cold[b] = (cold or 0) * 100
        chart_data_warm[b] = (warm or 0) * 100

        delta_cold = _delta_badge(cold, baseline_cold, pct=True)
        delta_overall = _delta_badge(overall, baseline_overall, pct=True)
        reuse_str = f"{reuse:.1f}%" if reuse is not None else "—"

        rows.append([
            _backend_pill(b),
            str(n),
            f"{_pct(overall)}{delta_overall}",
            f"{_pct(cold)}{delta_cold}",
            _pct(warm),
            reuse_str,
        ])

    tbl = _table(
        ["Backend", "N", "Overall", "COLD (fresh cache)", "WARM (cached)", "KV Reuse % (warm)"],
        rows,
    )

    # Validity check
    cold_vals = {
        b: results[b]["accuracy_stats"]["accuracy_cold"]
        for b in backends
        if results[b].get("accuracy_stats", {}).get("accuracy_cold") is not None
    }
    if len(cold_vals) > 1:
        spread = max(cold_vals.values()) - min(cold_vals.values())
        flag_color = "#ef4444" if spread > 0.02 else "#22c55e"
        flag_text = "diverged — inputs may differ" if spread > 0.02 else "consistent (good)"
        validity = (
            f'<p style="font-size:0.85em;margin-top:12px">'
            f'Cold-accuracy spread across backends: '
            f'<strong style="color:{flag_color}">{spread*100:.1f}pp — {flag_text}</strong>'
            f'<span style="color:#6b7280"> (should be ≈0 if inputs are identical)</span></p>'
        )
    else:
        validity = ""

    # per-bucket breakdown
    bucket_by_backend = {}
    for b in backends:
        acc_samples = results[b].get("accuracy_samples", [])
        if acc_samples:
            bucket_by_backend[b] = _bucket_stats(
                acc_samples,
                value_fn=lambda s: 1.0 if s["correct"] else 0.0,
                filter_fn=lambda s: s.get("query_type") == "COLD",
            )

    bucket_chart = ""
    if len(bucket_by_backend) > 0:
        bucket_chart = (
            '<p style="font-size:0.85em;color:#6b7280;margin-top:16px;margin-bottom:4px">'
            'Cold accuracy by context-length bucket</p>'
            + _spark_bucket_svg(bucket_by_backend, unit="%")
        )

    bar = _bar_chart_svg(chart_data_cold, unit="%", ref_key="baseline")
    body = f"""
{tbl}
{validity}
<div style="display:flex;flex-wrap:wrap;gap:32px;margin-top:20px;align-items:flex-start">
  <div>
    <p style="font-size:0.82em;color:#6b7280;margin-bottom:6px">Cold accuracy vs baseline</p>
    {bar}
  </div>
  <div>{bucket_chart}</div>
</div>"""
    return _section(
        "Accuracy",
        body,
        subtitle="Exact-match on bug-localization (4-choice MC). COLD = fresh cache, WARM = q2 after q1 cached the diff prefix."
    )


def _build_latency(data: Dict, backends: List[str]) -> str:
    results = data["results"]
    baseline_mean = results.get("baseline", {}).get("latency_stats", {}).get("ttft_ms_overall", {}).get("mean")

    rows = []
    chart_cold = {}
    chart_warm = {}

    for b in backends:
        s = results[b].get("latency_stats", {})
        if not s:
            rows.append([_backend_pill(b), "—", "—", "—", "—", "—", "—"])
            continue
        n = s.get("n_total", 0)
        overall = s.get("ttft_ms_overall", {})
        cold = s.get("ttft_ms_cold", {})
        warm = s.get("ttft_ms_warm", {})
        tps = s.get("avg_tokens_per_second")
        cache_reuse = s.get("avg_cache_reuse_ratio_warm")

        cold_mean = cold.get("mean")
        warm_mean = warm.get("mean")
        if cold_mean: chart_cold[b] = cold_mean
        if warm_mean: chart_warm[b] = warm_mean

        speedup = _speedup_badge(overall.get("mean"), baseline_mean)

        rows.append([
            _backend_pill(b),
            str(n),
            f"{_ms(overall.get('mean'))} {speedup}",
            f"{_ms(overall.get('p95'))}",
            f"{_ms(cold_mean)}",
            f"{_ms(warm_mean)}",
            _num(tps) + " tok/s" if tps else "—",
        ])

    tbl = _table(
        ["Backend", "N", "TTFT mean", "TTFT p95", "COLD mean", "WARM mean", "Throughput"],
        rows,
    )

    bar_cold = _bar_chart_svg(chart_cold, unit=" ms", lower_is_better=True, ref_key="baseline")
    bar_warm = _bar_chart_svg(chart_warm, unit=" ms", lower_is_better=True, ref_key="baseline")

    # per-bucket TTFT
    bucket_cold_by_backend = {}
    bucket_warm_by_backend = {}
    for b in backends:
        lat_samples = results[b].get("latency_samples", [])
        if lat_samples:
            bucket_cold_by_backend[b] = _bucket_stats(
                lat_samples,
                value_fn=lambda s: s.get("ttft_ms"),
                filter_fn=lambda s: s.get("query_type") == "COLD",
            )
            bucket_warm_by_backend[b] = _bucket_stats(
                lat_samples,
                value_fn=lambda s: s.get("ttft_ms"),
                filter_fn=lambda s: s.get("query_type") == "WARM",
            )

    bucket_cold_chart = (
        '<p style="font-size:0.85em;color:#6b7280;margin-top:16px;margin-bottom:4px">'
        'COLD TTFT by context-length bucket (ms)</p>'
        + _spark_bucket_svg(bucket_cold_by_backend, unit=" ms", lower_is_better=True)
    ) if bucket_cold_by_backend else ""

    bucket_warm_chart = (
        '<p style="font-size:0.85em;color:#6b7280;margin-top:16px;margin-bottom:4px">'
        'WARM TTFT by context-length bucket (ms)</p>'
        + _spark_bucket_svg(bucket_warm_by_backend, unit=" ms", lower_is_better=True)
    ) if bucket_warm_by_backend else ""

    body = f"""
{tbl}
<div style="display:flex;flex-wrap:wrap;gap:32px;margin-top:20px;align-items:flex-start">
  <div>
    <p style="font-size:0.82em;color:#6b7280;margin-bottom:6px">COLD TTFT (lower = faster)</p>
    {bar_cold}
  </div>
  <div>
    <p style="font-size:0.82em;color:#6b7280;margin-bottom:6px">WARM TTFT (lower = faster)</p>
    {bar_warm}
  </div>
</div>
<div style="display:flex;flex-wrap:wrap;gap:32px;margin-top:8px;align-items:flex-start">
  <div>{bucket_cold_chart}</div>
  <div>{bucket_warm_chart}</div>
</div>"""
    return _section(
        "Latency — Time to First Token",
        body,
        subtitle="COLD = q1 (no cached KVs). WARM = q2 (diff prefix already cached from q1). "
                 "Speedup shown relative to Baseline."
    )


def _build_memory(data: Dict, backends: List[str]) -> str:
    results = data["results"]
    baseline_peak = results.get("baseline", {}).get("memory_stats", {}).get("peak_gpu_mb_overall", {}).get("mean")

    rows = []
    chart_overall = {}

    for b in backends:
        s = results[b].get("memory_stats", {})
        if not s:
            rows.append([_backend_pill(b), "—", "—", "—", "—", "—", "—"])
            continue
        n = s.get("n_total", 0)
        overall = s.get("peak_gpu_mb_overall", {})
        cold = s.get("peak_gpu_mb_cold", {})
        warm = s.get("peak_gpu_mb_warm", {})
        kv = s.get("avg_kv_cache_mb")
        eff = s.get("avg_memory_efficiency_tok_per_mb")

        mean = overall.get("mean")
        if mean: chart_overall[b] = mean

        savings = _delta_badge(mean, baseline_peak, lower_is_better=True)

        rows.append([
            _backend_pill(b),
            str(n),
            f"{_mb(mean)}{savings}",
            _mb(overall.get("p95")),
            _mb(cold.get("mean")),
            _mb(warm.get("mean")),
            f"{_mb(kv)}  /  {_num(eff, 3)} tok/MB" if kv is not None else "—",
        ])

    tbl = _table(
        ["Backend", "N", "Peak mean", "Peak p95", "COLD mean", "WARM mean", "KV cache avg / efficiency"],
        rows,
    )

    bar = _bar_chart_svg(chart_overall, unit=" MB", lower_is_better=True, ref_key="baseline")

    # per-bucket peak
    bucket_by_backend = {}
    for b in backends:
        mem_samples = results[b].get("memory_samples", [])
        if mem_samples:
            bucket_by_backend[b] = _bucket_stats(
                mem_samples,
                value_fn=lambda s: s.get("gpu_memory_mb"),
            )

    bucket_chart = (
        '<p style="font-size:0.85em;color:#6b7280;margin-top:16px;margin-bottom:4px">'
        'Peak GPU memory by context-length bucket (MB)</p>'
        + _spark_bucket_svg(bucket_by_backend, unit=" MB", lower_is_better=True)
    ) if bucket_by_backend else ""

    body = f"""
{tbl}
<div style="display:flex;flex-wrap:wrap;gap:32px;margin-top:20px;align-items:flex-start">
  <div>
    <p style="font-size:0.82em;color:#6b7280;margin-bottom:6px">Peak GPU memory — overall mean (lower = better)</p>
    {bar}
  </div>
  <div>{bucket_chart}</div>
</div>"""
    return _section(
        "GPU Memory",
        body,
        subtitle="Peak torch.cuda.max_memory_allocated() per inference. "
                 "KV cache MB is estimated; efficiency = (prompt + completion tokens) / peak MB."
    )


def _build_kv_reuse(data: Dict, backends: List[str]) -> str:
    """KVBoost-specific deep-dive: reuse % distribution."""
    results = data["results"]
    kvb = results.get("kvboost", {})
    acc_samples = kvb.get("accuracy_samples", [])
    warm = [s for s in acc_samples if s.get("query_type") == "WARM" and s.get("kv_reuse_pct", 0) > 0]
    if not warm:
        return ""

    reuse_vals = [s["kv_reuse_pct"] for s in warm]
    # histogram buckets 0-20, 20-40, 40-60, 60-80, 80-100
    hist = [0] * 5
    for v in reuse_vals:
        idx = min(int(v / 20), 4)
        hist[idx] += 1
    total = len(reuse_vals)

    bar_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    max_h = max(hist) if hist else 1
    svg_w, svg_h = 400, 140
    bar_w = 56
    gap = 12
    pad_l, pad_t = 40, 10
    color = BACKEND_COLORS["kvboost"]["chart"]

    lines = [
        f'<svg width="{svg_w}" height="{svg_h + 30}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:inherit;overflow:visible">'
    ]
    for i, (cnt, label) in enumerate(zip(hist, bar_labels)):
        x = pad_l + i * (bar_w + gap)
        bh = int(cnt / max_h * svg_h) if max_h else 0
        y = pad_t + svg_h - bh
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="{color}" rx="4" opacity="0.8"/>')
        pct = cnt / total * 100
        if bh > 14:
            lines.append(
                f'<text x="{x + bar_w // 2}" y="{y + 14}" text-anchor="middle" '
                f'font-size="11" fill="white" font-weight="600">{pct:.0f}%</text>'
            )
        lines.append(
            f'<text x="{x + bar_w // 2}" y="{pad_t + svg_h + 18}" text-anchor="middle" '
            f'font-size="10" fill="#6b7280">{label}</text>'
        )
    lines.append("</svg>")
    hist_svg = "\n".join(lines)

    s = kvb.get("latency_stats", {})
    cold_mean = (s.get("ttft_ms_cold") or {}).get("mean")
    warm_mean = (s.get("ttft_ms_warm") or {}).get("mean")
    speedup_val = (cold_mean / warm_mean) if (cold_mean and warm_mean and warm_mean > 0) else None
    speedup_str = f"{speedup_val:.2f}×" if speedup_val else "N/A"
    avg_reuse = sum(reuse_vals) / len(reuse_vals)

    items = [
        ("Warm samples", str(len(warm)), "with reuse > 0"),
        ("Avg KV reuse", f"{avg_reuse:.1f}%", "of diff prefix tokens"),
        ("COLD→WARM speedup", speedup_str, "TTFT ratio"),
    ]

    body = f"""
{_kv_grid(items)}
<p style="font-size:0.85em;color:#6b7280;margin-bottom:6px">KV reuse % distribution (warm queries)</p>
{hist_svg}
<p style="font-size:0.82em;color:#6b7280;margin-top:8px">
  Higher reuse = more of the diff prefix was fetched from cache rather than recomputed.
  A left-heavy distribution means short diffs where the chunk boundary lands late;
  right-heavy means most of the context is being cached effectively.
</p>"""
    return _section("KVBoost — Cache Reuse Deep Dive", body,
                    subtitle="Only WARM queries (q2) with kv_reuse_pct > 0.")


def _build_cold_warm_comparison(data: Dict, backends: List[str]) -> str:
    """Table: for each backend, how much does WARM improve over COLD?"""
    results = data["results"]
    rows = []
    for b in backends:
        ls = results[b].get("latency_stats", {})
        acc_s = results[b].get("accuracy_stats", {})
        cold_ttft = (ls.get("ttft_ms_cold") or {}).get("mean")
        warm_ttft = (ls.get("ttft_ms_warm") or {}).get("mean")
        cold_acc = acc_s.get("accuracy_cold")
        warm_acc = acc_s.get("accuracy_warm")

        if cold_ttft and warm_ttft and cold_ttft > 0:
            speedup = cold_ttft / warm_ttft
            sp_color = "#22c55e" if speedup > 1.05 else "#6b7280"
            sp_str = f'<span style="color:{sp_color};font-weight:600">{speedup:.2f}×</span>'
        else:
            sp_str = "—"

        acc_delta = ""
        if cold_acc is not None and warm_acc is not None:
            d = (warm_acc - cold_acc) * 100
            color = "#22c55e" if d >= 0 else "#ef4444"
            acc_delta = f'<span style="color:{color};font-size:0.85em">{d:+.1f}pp</span>'

        rows.append([
            _backend_pill(b),
            _ms(cold_ttft),
            _ms(warm_ttft),
            sp_str,
            _pct(cold_acc),
            f"{_pct(warm_acc)} {acc_delta}",
        ])

    tbl = _table(
        ["Backend", "COLD TTFT", "WARM TTFT", "Speedup (COLD/WARM)", "COLD acc", "WARM acc"],
        rows,
    )
    body = tbl
    return _section(
        "Cold vs Warm Cache Comparison",
        body,
        subtitle="Direct comparison of first (cold) vs second (warm) query within each pair. "
                 "Speedup > 1× means caching helps latency. Accuracy should stay stable."
    )


def _build_failure_analysis(data: Dict, backends: List[str]) -> str:
    """Show sample errors where backends disagree."""
    results = data["results"]
    # Find pair_groups where kvboost got it wrong but baseline got it right, or vice versa
    kvb_map = {}
    base_map = {}
    for s in results.get("kvboost", {}).get("accuracy_samples", []):
        kvb_map[s["id"]] = s
    for s in results.get("baseline", {}).get("accuracy_samples", []):
        base_map[s["id"]] = s

    disagreements = []
    for sid, ks in kvb_map.items():
        bs = base_map.get(sid)
        if bs and ks["correct"] != bs["correct"]:
            disagreements.append({
                "id": sid,
                "query_type": ks.get("query_type", ""),
                "ctx_tokens": ks.get("context_tokens", 0),
                "gold": ks.get("gold", ""),
                "kvboost": ks.get("predicted", ""),
                "kvboost_correct": ks.get("correct"),
                "baseline": bs.get("predicted", ""),
                "baseline_correct": bs.get("correct"),
            })

    if not disagreements:
        note = '<p style="color:#6b7280;font-size:0.9em">No disagreements between KVBoost and Baseline — outputs are identical.</p>'
        return _section("Failure Analysis", note,
                        subtitle="Samples where KVBoost and Baseline gave different correctness.")

    rows = []
    for d in sorted(disagreements, key=lambda x: x["ctx_tokens"])[:40]:
        kv_c = "✓" if d["kvboost_correct"] else "✗"
        ba_c = "✓" if d["baseline_correct"] else "✗"
        kv_color = "#22c55e" if d["kvboost_correct"] else "#ef4444"
        ba_color = "#22c55e" if d["baseline_correct"] else "#ef4444"
        rows.append([
            d["id"],
            d["query_type"],
            str(d["ctx_tokens"]),
            d["gold"],
            f'<span style="color:{kv_color}">{kv_c} {d["kvboost"]}</span>',
            f'<span style="color:{ba_color}">{ba_c} {d["baseline"]}</span>',
        ])

    tbl = _table(
        ["Sample ID", "Type", "Ctx tokens", "Gold", "KVBoost pred", "Baseline pred"],
        rows,
    )
    n = len(disagreements)
    n_kvb_wrong = sum(1 for d in disagreements if not d["kvboost_correct"])
    n_base_wrong = sum(1 for d in disagreements if not d["baseline_correct"])

    summary = _kv_grid([
        ("Total disagreements", str(n), "out of all pairs"),
        ("KVBoost wrong", str(n_kvb_wrong), "baseline was right"),
        ("Baseline wrong", str(n_base_wrong), "kvboost was right"),
    ])

    showing = f'<p style="font-size:0.82em;color:#6b7280;margin-bottom:8px">Showing first 40 of {n}</p>' if n > 40 else ""
    body = summary + showing + tbl
    return _section(
        "Failure Analysis",
        body,
        subtitle="Samples where KVBoost and Baseline gave different correctness. "
                 "Net positive = KVBoost fixed cases baseline missed; net negative = cache hurt quality."
    )


# ---------------------------------------------------------------------------
# Full HTML assembly
# ---------------------------------------------------------------------------

def build_html(data: Dict) -> str:
    backends = [b for b in data.get("backends", []) if data["results"].get(b, {}).get("run_ok", False)]
    all_backends = data.get("backends", [])

    sections = "".join([
        _build_overview(data, all_backends),
        _build_accuracy(data, all_backends),
        _build_cold_warm_comparison(data, all_backends),
        _build_latency(data, all_backends),
        _build_memory(data, all_backends),
        _build_kv_reuse(data, all_backends),
        _build_failure_analysis(data, all_backends),
    ])

    model_short = data["model"].split("/")[-1]
    ts = data.get("experiment_timestamp", "")
    title = f"KVBoost Benchmark — {model_short} — {ts}"

    # Legend
    legend_pills = " ".join(_backend_pill(b) for b in all_backends)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f8fafc;
      color: #1f2937;
      margin: 0;
      padding: 0;
    }}
    .wrapper {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    h1 {{ font-size: 1.6rem; font-weight: 800; color: #111827; margin-bottom: 4px; }}
    details summary {{ user-select: none; }}
    details[open] summary {{ margin-bottom: 8px; }}
    @media print {{
      details {{ display: block; }}
      details summary {{ display: none; }}
    }}
  </style>
</head>
<body>
<div class="wrapper">
  <h1>{title}</h1>
  <p style="color:#6b7280;font-size:0.88em;margin-bottom:24px">
    Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp; {legend_pills}
  </p>
  {sections}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from experiment JSON")
    parser.add_argument(
        "json_file",
        nargs="?",
        help="Path to experiment JSON. If omitted, uses the latest in results/"
    )
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path")
    args = parser.parse_args()

    if args.json_file:
        json_path = Path(args.json_file)
    else:
        candidates = sorted(RESULTS_DIR.glob("experiment_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No experiment JSON found in results/. Run run_experiment.py first.")
            raise SystemExit(1)
        json_path = candidates[0]
        print(f"Using latest: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    html = build_html(data)

    out_path = args.out or json_path.with_suffix(".html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
