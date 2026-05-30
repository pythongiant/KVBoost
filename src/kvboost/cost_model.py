"""Probed hardware + model coefficients, consumed by ``oom_planner``.

The OOM planner needs concrete numbers about THIS hardware + THIS
model to predict request memory usage:
  - total VRAM, per-layer bytes, num layers, hidden dim, num heads
  - PCIe host-to-device bandwidth (for cost of any future host offload)
  - HBM bandwidth (for the decode-step latency roofline)
  - step_latency_ms (measured directly, with roofline as fallback)

This module probes those values once at server startup and returns a
:class:`CostCoefficients` instance. All probes are bounded (<3 s
total) and defensive — if any probe fails the field falls back to a
conservative default. The planner is responsible for the actual cost
arithmetic (see ``oom_planner.estimate_peak_mb``); this module just
supplies the inputs.

Cost framing follows three converging lines of work:

* **LLM-in-a-Flash** (Alizadeh et al., arxiv:2312.11514) — frames
  inference cost as ``T = compute_time + data_loading_time`` and
  optimizes ``read_chunk_size`` against flash/PCIe bandwidth.
* **Sarathi-Serve chunked prefill** (ACM SIGOPS 2025) — models per-step
  prefill cost as a function of chunk size and accumulated KV length;
  halving chunk size roughly doubles the per-prompt step count.
* **PreScope** (arxiv:2509.23638) — explicitly argues that
  resource-constrained inference needs a *global cost model* computed
  once and reused, not re-derived per decision.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# ── Defaults used when probes fail or device is non-CUDA ──
# Conservative: chosen so the affected knob looks expensive and is
# de-prioritized unless nothing else can help. They are NOT meant to be
# "average GPU" numbers — they're meant to fail safe.
_DEFAULT_PCIE_GBPS = 4.0           # PCIe Gen3 x16 ≈ 16 GB/s peak; assume 1/4 of that
_DEFAULT_HBM_GBPS = 200.0          # mid-tier datacenter card
_DEFAULT_STEP_LATENCY_MS = 50.0    # decode step on an 8B-class model
_DEFAULT_PER_LAYER_MB = 150.0      # 8B AWQ ÷ 36 layers ≈ 150 MB
_DEFAULT_NUM_LAYERS = 32


@dataclass
class CostCoefficients:
    """Hardware + model coefficients probed once at server startup.

    Units are explicit in the field names to avoid the classic
    "is this MB or MiB" footgun. Bandwidth fields are GiB/s (binary),
    memory fields are MiB (binary), times are ms.

    This dataclass is pure data — the planner does the cost arithmetic
    so that the formulas live next to the policy that uses them.
    Keeping ``step_latency_ms`` etc. here (rather than re-deriving in
    the planner) means a single source of truth for hardware probes.
    """

    # ── Memory shape ──
    total_vram_mb: float            # total device memory, MiB
    per_layer_mb: float             # avg parameter bytes per decoder layer, MiB
    num_layers: int                 # decoder layer count

    # ── Speeds ──
    pcie_h2d_gibps: float           # host→device transfer rate, GiB/s
    hbm_bandwidth_gibps: float      # device memory bandwidth, GiB/s
    step_latency_ms: float          # one decode forward pass, ms
                                    # (measured directly when possible;
                                    # falls back to model_bytes / HBM_bw)
    step_latency_source: str = "roofline"  # "measured" | "roofline" | "default"

    # ── Activation floor ──
    # Peak transient activation memory of a SINGLE forward pass,
    # independent of prompt length. This is the working set across all
    # decoder layers (intermediate hidden states, attention buffers,
    # MLP projections) that the allocator high-water-mark captures even
    # for a 1-token decode. The planner adds this as a constant floor;
    # without it, short-prompt peak predictions undershoot by the full
    # model activation working set (the +30000% residuals we observed).
    # Measured during the decode microbenchmark; 0.0 when unmeasured
    # (planner then falls back to a per-layer heuristic).
    baseline_activation_mb: float = 0.0

    def summary(self) -> str:
        return (
            f"VRAM={self.total_vram_mb / 1024:.1f} GiB, "
            f"layers={self.num_layers} × {self.per_layer_mb:.0f} MiB, "
            f"PCIe H→D={self.pcie_h2d_gibps:.1f} GiB/s, "
            f"HBM={self.hbm_bandwidth_gibps:.0f} GiB/s, "
            f"step_latency={self.step_latency_ms:.1f} ms ({self.step_latency_source}), "
            f"act_floor={self.baseline_activation_mb:.0f} MiB"
        )


# ── Probes ────────────────────────────────────────────────────────────────────


def _probe_pcie_h2d(device, *, size_mb: int = 64, repeats: int = 3) -> float:
    """Measure host→device transfer rate in GiB/s.

    Uses a pinned-host fp16 buffer to avoid the "first transfer is slow"
    pageable-memory penalty. Repeats ``repeats`` times and takes the
    best run (warmup amortization).
    """
    import torch

    nbytes = size_mb * (1024 ** 2)
    n_elems = nbytes // 2  # fp16
    try:
        cpu_t = torch.empty(n_elems, dtype=torch.float16, pin_memory=True)
    except Exception:
        cpu_t = torch.empty(n_elems, dtype=torch.float16)

    # Warmup
    _ = cpu_t.to(device, non_blocking=True)
    torch.cuda.synchronize(device)

    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = cpu_t.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        best = min(best, time.perf_counter() - t0)

    # GiB/s
    return (nbytes / best) / (1024.0 ** 3)


def _probe_hbm_bandwidth(device, *, size_mb: int = 256, repeats: int = 5) -> float:
    """Measure on-device memory bandwidth in GiB/s via D2D copy."""
    import torch

    nbytes = size_mb * (1024 ** 2)
    n_elems = nbytes // 2
    src = torch.empty(n_elems, dtype=torch.float16, device=device)
    dst = torch.empty_like(src)

    # Warmup
    dst.copy_(src)
    torch.cuda.synchronize(device)

    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize(device)
        best = min(best, time.perf_counter() - t0)

    # D2D involves reading src + writing dst → 2 × nbytes of traffic.
    return (2 * nbytes / best) / (1024.0 ** 3)


def _probe_decode_step(
    engine, *, warmup: int = 2, repeats: int = 5,
) -> "tuple[float, float]":
    """Probe a real decode forward for ``(step_latency_ms, activation_mb)``.

    Two measurements from the same microbenchmark:

      - **step_latency_ms**: median single-token decode time. Captures
        every overhead the roofline misses (kernel launch, attention
        scratch, speculative-draft forwards, streaming DMA).

      - **activation_mb**: peak transient activation of one forward,
        measured as ``max_memory_allocated − allocated_before``. This
        is the working set across all decoder layers that exists for
        ANY forward regardless of prompt length. The planner uses it
        as a floor so short-prompt peak predictions don't undershoot
        by the full model activation working set. Returns 0.0 on
        non-CUDA devices (no allocator stats).

    Bounded by ``warmup + repeats`` forwards. Wrapped in try/except by
    the caller — any failure falls back to the roofline latency
    estimate and a 0.0 activation floor.
    """
    import torch
    device = engine.device
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("engine.tokenizer not available")
    is_cuda = str(device).startswith("cuda")

    # Build a minimal valid input. Some tokenizers reject empty strings.
    seed = tokenizer("Hi", return_tensors="pt").input_ids.to(device)
    bos_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else (tokenizer.eos_token_id or 0)
    )
    next_tok = torch.tensor([[bos_id]], device=device, dtype=torch.long)

    model = engine.model
    activation_mb = 0.0
    with torch.no_grad():
        # Prime KV cache with the seed prompt.
        out = model(seed, use_cache=True)
        past = out.past_key_values

        # Warmup decodes (excluded from timing).
        for _ in range(warmup):
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values

        if is_cuda:
            torch.cuda.synchronize(device)
            # Measure activation high-water-mark of one isolated forward:
            # the delta between peak-during-forward and the resident set
            # before it. Excludes weights + KV (both already resident),
            # leaving the pure transient activation working set.
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            activation_mb = max(0.0, (peak - allocated_before) / (1024.0 ** 2))

        # Timed decodes.
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if is_cuda:
                torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - t0)

    samples.sort()
    median = samples[len(samples) // 2]
    return median * 1000.0, activation_mb


def _detect_num_layers(config: Any) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return _DEFAULT_NUM_LAYERS


def _model_bytes(model) -> int:
    """Sum bytes of all parameters and registered buffers.

    Captures AWQ packed tensors (int32 qweights, fp16 scales, int32
    qzeros) as well as plain fp16/bf16 weights. For streaming models,
    streamed-out parameters live on ``meta`` and report element_size
    of their declared dtype, which is what we want — we're costing
    the *logical* model size, not what's currently resident.
    """
    total = 0
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        try:
            total += p.numel() * p.element_size()
        except Exception:
            pass
    for b in model.buffers():
        if id(b) in seen:
            continue
        seen.add(id(b))
        try:
            total += b.numel() * b.element_size()
        except Exception:
            pass
    # AWQ packed tensors on StreamingQLinear are plain attributes, not
    # parameters/buffers — they wouldn't be counted by the loops above.
    # Walk submodules once and pick them up.
    for mod in model.modules():
        for attr in ("qweight", "qzeros", "scales"):
            t = getattr(mod, attr, None)
            if t is not None and hasattr(t, "numel") and hasattr(t, "element_size"):
                if id(t) in seen:
                    continue
                seen.add(id(t))
                try:
                    total += t.numel() * t.element_size()
                except Exception:
                    pass
    return total


def probe_cost_coefficients(
    engine,
    *,
    skip_bandwidth_probes: bool = False,
    skip_step_latency_probe: bool = False,
) -> CostCoefficients:
    """Probe GPU + model and return populated coefficients.

    Safe on any device — falls back to defaults for fields that can't
    be measured (e.g. PCIe on CPU). Total probe wall time is bounded
    at ~3 s (transfer benchmarks + a few decode steps).
    """
    import torch

    device = getattr(engine, "device", None) or torch.device("cpu")
    is_cuda = str(device).startswith("cuda")

    # ── VRAM ──
    if is_cuda:
        idx = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_vram_mb = props.total_memory / (1024.0 ** 2)
    else:
        total_vram_mb = 0.0

    # ── Model shape ──
    try:
        config = engine.model.config
        num_layers = _detect_num_layers(config)
    except Exception:
        num_layers = _DEFAULT_NUM_LAYERS

    try:
        model_bytes = _model_bytes(engine.model)
        per_layer_mb = (model_bytes / num_layers) / (1024.0 ** 2)
    except Exception:
        per_layer_mb = _DEFAULT_PER_LAYER_MB

    # ── Speeds ──
    pcie_h2d_gibps = _DEFAULT_PCIE_GBPS
    hbm_gibps = _DEFAULT_HBM_GBPS
    if is_cuda and not skip_bandwidth_probes:
        try:
            pcie_h2d_gibps = _probe_pcie_h2d(device)
        except Exception as e:
            log.warning("PCIe H→D probe failed (%s); using default %.1f GiB/s",
                        e, _DEFAULT_PCIE_GBPS)
        try:
            hbm_gibps = _probe_hbm_bandwidth(device)
        except Exception as e:
            log.warning("HBM bandwidth probe failed (%s); using default %.0f GiB/s",
                        e, _DEFAULT_HBM_GBPS)

    # ── Step latency: prefer measured, fall back to roofline ──
    # A real decode forward captures kernel-launch latency, attention
    # scratch allocation, optional speculative-draft cost, streaming-
    # layer DMA — every overhead source the roofline misses. The
    # roofline (``model_bytes / HBM_bw``) is the theoretical lower
    # bound; we keep it as fallback so non-CUDA paths still produce a
    # sensible number.
    step_latency_ms = _DEFAULT_STEP_LATENCY_MS
    step_latency_source = "default"
    baseline_activation_mb = 0.0
    if is_cuda and not skip_step_latency_probe:
        try:
            step_latency_ms, baseline_activation_mb = _probe_decode_step(engine)
            step_latency_source = "measured"
        except Exception as e:
            log.warning(
                "Decode step probe failed (%s); falling back to roofline estimate.",
                e,
            )
    if step_latency_source != "measured":
        try:
            model_gib = model_bytes / (1024.0 ** 3)
            roofline_ms = (model_gib / hbm_gibps) * 1000.0
            step_latency_ms = max(1.0, roofline_ms)
            step_latency_source = "roofline"
        except Exception:
            pass  # keep the default

    cc = CostCoefficients(
        total_vram_mb=total_vram_mb,
        per_layer_mb=per_layer_mb,
        num_layers=num_layers,
        pcie_h2d_gibps=pcie_h2d_gibps,
        hbm_bandwidth_gibps=hbm_gibps,
        step_latency_ms=step_latency_ms,
        step_latency_source=step_latency_source,
        baseline_activation_mb=baseline_activation_mb,
    )
    log.info("Probed cost coefficients: %s", cc.summary())
    return cc
