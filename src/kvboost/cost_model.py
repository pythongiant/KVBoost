"""Auto-calibrated cost coefficients for OOM recovery decisions.

The OOM recovery scoring framework picks knobs by maximizing
``freed_bytes / wall_time_cost_seconds``. The numerator is easy — every
knob has a closed-form for how many bytes it frees. The denominator
(cost in seconds) depends on hardware (VRAM, PCIe bandwidth, GPU
memory bandwidth) and model (param count, layer count). Hardcoding
these numbers would make recovery picks wrong on every GPU that isn't
the one they were tuned on.

This module probes the GPU and the loaded model once at server startup
and produces a :class:`CostCoefficients` instance that the recovery
loop consults. All probes are bounded (<2 s total) and defensive — if
any probe fails the field falls back to a conservative default that
makes the corresponding knob look expensive (and so avoided unless
nothing else can help).

Cost framing follows three converging lines of work:

* **LLM-in-a-Flash** (Alizadeh et al., arxiv:2312.11514) — frames
  inference cost as ``T = compute_time + data_loading_time`` and
  optimizes ``read_chunk_size`` against flash/PCIe bandwidth.
* **Sarathi-Serve chunked prefill** (ACM SIGOPS 2025) — models per-step
  prefill cost as a function of chunk size and accumulated KV length;
  halving chunk size roughly doubles the per-prompt step count.
* **PreScope** (arxiv:2509.23638) — explicitly argues that
  resource-constrained MoE/streaming inference needs a *global cost
  model* computed once and reused, not re-derived per decision.

The signature: ``score(knob) = freed_bytes(knob) / cost_seconds(knob)``.
Knob with the highest finite score wins; ``-inf`` means the knob
can't help at all (e.g. cache is empty, can't evict).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

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
    """Per-knob cost inputs, all probed from the live model and GPU.

    Units are explicit in the field names to avoid the classic
    "is this MB or MiB" footgun. All bandwidth fields are GiB/s
    (binary), all memory fields are MiB (binary), all times are
    seconds or milliseconds as suffixed.
    """

    # ── Memory shape ──
    total_vram_mb: float            # total device memory, MiB
    per_layer_mb: float             # avg parameter bytes per decoder layer, MiB
    num_layers: int                 # decoder layer count

    # ── Speeds ──
    pcie_h2d_gibps: float           # host→device transfer rate, GiB/s
    hbm_bandwidth_gibps: float      # device memory bandwidth, GiB/s
    step_latency_ms: float          # one forward pass at small batch, ms

    # ── Workload assumptions (operator overrides; reasonable defaults) ──
    expected_decode_tokens: int = 1024
    expected_cache_hit_rate: float = 0.3

    # ── Cost functions per knob ─────────────────────────────────────
    #
    # Each returns expected wall-time penalty in SECONDS applied to
    # this request and (where relevant) the future stream of requests.

    def cost_lower_cache(self, evict_mb: float) -> float:
        """Cost of clearing ``evict_mb`` of cached KV state.

        Penalty hits future requests that would have hit the evicted
        chunks. Model: each cached MiB stores roughly ``MiB ÷ 72 KB =
        14 tokens`` of context at int8 KV (or 7 at fp16), each of
        which costs one prefill step on a future hit. Probability of
        hit is ``expected_cache_hit_rate``.
        """
        if evict_mb <= 0:
            return float("inf")
        # int8 KV ≈ 72 KB/token; fp16 ≈ 144 KB/token. Use 100 KB as
        # a tier-agnostic average. Tweak via field if needed.
        tokens_per_mib = 1024.0 / 100.0     # ~10 tokens per MiB
        tokens_lost = evict_mb * tokens_per_mib
        prefill_seconds = tokens_lost * (self.step_latency_ms / 1000.0)
        return prefill_seconds * self.expected_cache_hit_rate

    def cost_lower_prefill_chunk(
        self,
        old_chunk: int,
        new_chunk: int,
        prompt_tokens: Optional[int] = None,
    ) -> float:
        """Cost of shrinking prefill chunk from ``old_chunk`` to ``new_chunk``.

        Per Sarathi-Serve framing: prompt is processed in
        ``ceil(prompt_tokens / chunk)`` forward passes, each costing
        ~``step_latency_ms``. Halving the chunk roughly doubles step
        count. Penalty hits THIS request only (single-shot).
        """
        if new_chunk >= old_chunk or new_chunk <= 0:
            return float("inf")
        # Default: assume a moderately long prompt if not told. The
        # cost ratio (new/old) is what matters; absolute prompt size
        # only changes the seconds scale.
        if prompt_tokens is None:
            prompt_tokens = 4096
        old_eff = max(old_chunk, 1) if old_chunk > 0 else prompt_tokens
        old_steps = max(1, prompt_tokens / old_eff)
        new_steps = prompt_tokens / new_chunk
        extra_steps = max(0.0, new_steps - old_steps)
        return extra_steps * (self.step_latency_ms / 1000.0)

    def cost_lower_streaming(self, delta_resident_layers: int) -> float:
        """Cost of marking ``delta_resident_layers`` additional layers as
        streamed.

        Per LLM-in-a-Flash framing: each streamed layer pays one
        host→device DMA per forward pass. Across all decode tokens
        of an in-flight request this multiplies. Penalty hits THIS
        request's decode AND every subsequent request until the
        knob is restored.
        """
        if delta_resident_layers <= 0:
            return float("inf")
        bytes_per_layer = self.per_layer_mb * (1024.0 ** 2)
        per_step_dma_seconds = (
            bytes_per_layer * delta_resident_layers
            / (self.pcie_h2d_gibps * (1024.0 ** 3))
        )
        return per_step_dma_seconds * self.expected_decode_tokens

    def summary(self) -> str:
        return (
            f"VRAM={self.total_vram_mb / 1024:.1f} GiB, "
            f"layers={self.num_layers} × {self.per_layer_mb:.0f} MiB, "
            f"PCIe H→D={self.pcie_h2d_gibps:.1f} GiB/s, "
            f"HBM={self.hbm_bandwidth_gibps:.0f} GiB/s, "
            f"step_latency={self.step_latency_ms:.1f} ms"
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
    workload_decode_tokens: int = 1024,
    workload_cache_hit_rate: float = 0.3,
    skip_bandwidth_probes: bool = False,
) -> CostCoefficients:
    """Probe GPU + model and return populated coefficients.

    Safe on any device — falls back to defaults for fields that can't
    be measured (e.g. PCIe on CPU). Total probe wall time is bounded
    at ~2 s (three short transfer benchmarks).
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

    # ── Step latency: derive from HBM bandwidth + model size ──
    # A decode step is memory-bandwidth-bound on dense LLMs: it must
    # stream every weight through SMs once. ``step ≈ model_bytes / HBM_bw``.
    # This is the same lower bound used in Sarathi-Serve and Anyscale's
    # roofline analyses. The actual step is somewhat higher (overhead),
    # but for ratio-based cost decisions we only need consistency, not
    # absolute accuracy.
    try:
        model_gib = model_bytes / (1024.0 ** 3)
        step_latency_ms = (model_gib / hbm_gibps) * 1000.0
        # Floor at 1 ms — sub-millisecond is suspicious and breaks
        # downstream cost arithmetic.
        step_latency_ms = max(1.0, step_latency_ms)
    except Exception:
        step_latency_ms = _DEFAULT_STEP_LATENCY_MS

    cc = CostCoefficients(
        total_vram_mb=total_vram_mb,
        per_layer_mb=per_layer_mb,
        num_layers=num_layers,
        pcie_h2d_gibps=pcie_h2d_gibps,
        hbm_bandwidth_gibps=hbm_gibps,
        step_latency_ms=step_latency_ms,
        expected_decode_tokens=workload_decode_tokens,
        expected_cache_hit_rate=workload_cache_hit_rate,
    )
    log.info("Probed cost coefficients: %s", cc.summary())
    return cc
