"""Proactive OOM avoidance via per-request planning.

The previous design caught CUDA OOMs mid-request and reactively shrank
knobs until something fit (or didn't). It worked but had two flaws:

1. **Slow failure**: a too-big prompt could spend minutes cascading
   through knob shrinks before the GPU finally collapsed.
2. **Silent harm**: every shrink mutated global engine state and didn't
   always restore it, so a failed request could degrade subsequent
   requests without anyone noticing.

The new design plans BEFORE dispatch. Given the request's prompt size
and the live GPU memory snapshot, it picks a ``(chunk_size, kv_bits)``
configuration that is *predicted* to fit, applies it for the duration
of that request only, and restores afterward. If no configuration
fits, the request is rejected up-front (HTTP 413) or — under
``--auto-truncate`` — truncated to the largest prefix that does fit.

What it does NOT do:
  - Touch layer streaming (residency is a server-wide knob — flipping
    it per request would yank weights in/out of VRAM and shred decode
    latency for everyone).
  - Evict the global KV cache (only the per-request KV bits and chunk
    size are adjusted; the cache stays exactly as it was).
  - Catch unexpected OOMs. The thin safety net for those is a separate
    concern; the planner's promise is "if I commit to a plan, I'm
    saying it will fit, with 15% margin."

Memory model
------------
For a request with N prompt tokens, configured chunk size K and KV
bits B, peak transient memory is::

    peak_mb = kv_total + activation + attention_scratch

where::

    kv_total          = (N + max_new) × bytes_per_token_kv(B) × layers
    activation        = K × hidden_dim × 4_bytes  (per-layer, but
                        not all layers materialize simultaneously —
                        we cost the worst case: one layer's worth)
    attention_scratch = the LITERAL HBM the active attention backend
                        allocates for one prefill forward (see
                        ``_attention_scratch_bytes``)

``attention_scratch`` is backend-specific, not a single worst-case
constant. Only ``eager`` materializes the full ``K × N × n_heads`` fp32
score matrix — the genuinely O(KN) term. Every flash/tiled backend
kvboost actually runs (``flash_attention_2`` / ``sage`` / ``triton_flash``
/ ``flashinfer`` / PyTorch ``sdpa`` on the fp16-causal-no-mask prefill
path) keeps the scores in SRAM and allocates only per-call Q/K/V working
buffers — linear in sequence length, not the K×N product. Modelling the
math-backend worst case for those over-predicted peak by the chunk/seqlen
ratio, so the planner now costs each backend's real allocation.

These coefficients are derived from the same probe ``cost_model.py``
runs at startup. We still add a 15% safety margin on top of the
prediction to absorb allocator fragmentation, scratchpads, and CUDA's
reserved-but-unallocated blocks, and the calibration tracker validates
the per-backend model against measured peaks request by request.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── Calibration tracker ───────────────────────────────────────────────────────


@dataclass
class _Residual:
    """One observation: predicted vs actual peak memory."""
    chunk_size: int
    kv_bits: int
    prompt_tokens: int
    predicted_mb: float
    actual_mb: float

    @property
    def error_frac(self) -> float:
        """Relative residual: (actual − predicted) / predicted.

        Positive means we underestimated (risky; planner thought it
        would use less than it actually did). Negative means we
        overestimated (safe but wasteful)."""
        if self.predicted_mb <= 0:
            return 0.0
        return (self.actual_mb - self.predicted_mb) / self.predicted_mb


class CalibrationTracker:
    """Rolling window of prediction residuals + suggested safety margin.

    Operator-visible stats answer two questions:

      1. "Is my safety margin enough?" — look at the p95 residual.
         If p95 > current margin, you have request-killing surprises.

      2. "Where is the planner systematically wrong?" — slice
         residuals by ``(chunk_size, kv_bits)`` cohort. A bias in one
         cohort points at a missing term in the memory model.

    Bounded ring buffer (``maxlen=window``); old samples evict.
    """

    def __init__(self, window: int = 256):
        self.window = window
        self.residuals: Deque[_Residual] = deque(maxlen=window)

    def record(
        self, *,
        chunk_size: int, kv_bits: int, prompt_tokens: int,
        predicted_mb: float, actual_mb: float,
    ) -> None:
        self.residuals.append(_Residual(
            chunk_size=chunk_size, kv_bits=kv_bits,
            prompt_tokens=prompt_tokens,
            predicted_mb=predicted_mb, actual_mb=actual_mb,
        ))

    def suggested_margin(self, default: float = 0.15) -> float:
        """Suggested safety margin = p95 of recent residuals, floored
        at 5% and capped at 50%. Returns ``default`` until we have at
        least 16 samples (avoid reacting to startup noise)."""
        if len(self.residuals) < 16:
            return default
        errs = sorted(r.error_frac for r in self.residuals)
        # p95 — index ceil(0.95 × n) − 1
        idx = max(0, int(0.95 * len(errs)) - 1)
        p95 = errs[idx]
        # Floor + cap protect against degenerate measurements
        return max(0.05, min(0.50, p95 if p95 > 0 else default))

    def stats(self) -> dict:
        """Aggregate stats for ``/v1/stats`` or operator inspection."""
        if not self.residuals:
            return {"n_samples": 0}
        errs = [r.error_frac for r in self.residuals]
        return {
            "n_samples": len(self.residuals),
            "window": self.window,
            "residual_median": statistics.median(errs),
            "residual_p95": sorted(errs)[max(0, int(0.95 * len(errs)) - 1)],
            "residual_max": max(errs),
            "residual_min": min(errs),
            "suggested_margin": self.suggested_margin(),
            "cohorts": self._cohort_stats(),
        }

    def _cohort_stats(self) -> dict:
        """Per-(chunk, kv_bits) median error — surfaces systematic bias."""
        buckets: dict = {}
        for r in self.residuals:
            key = f"chunk={r.chunk_size},kv={r.kv_bits}"
            buckets.setdefault(key, []).append(r.error_frac)
        return {
            k: {"n": len(v), "median_err": statistics.median(v)}
            for k, v in buckets.items()
        }


# ── GPU memory introspection ──────────────────────────────────────────────────


def gpu_mem_snapshot(device: Any = None) -> dict:
    """Live snapshot of GPU memory state, all in MiB.

    Captured fields:
      - ``free_mb``      : memory the CUDA driver reports as free.
      - ``allocated_mb`` : bytes currently held by live PyTorch tensors.
      - ``reserved_mb``  : bytes PyTorch's caching allocator has claimed
                           (allocated + reserved-but-unallocated).
      - ``peak_mb``      : peak allocator usage since last reset
                           (call ``reset_peak_memory_stats()`` to zero it).
      - ``total_mb``     : device total memory.

    The ``reserved − allocated`` gap is the allocator's cache — useful for
    spotting fragmentation (high cache, low free). Returns an empty dict on
    CPU / MPS / probe failure so callers can safely ``snapshot.get(...)``.
    """
    try:
        import torch
        if device is None:
            if not torch.cuda.is_available():
                return {}
            idx = torch.cuda.current_device()
        else:
            dev = torch.device(device)
            if dev.type != "cuda":
                return {}
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        return {
            "free_mb": free_bytes / (1024.0 ** 2),
            "total_mb": total_bytes / (1024.0 ** 2),
            "allocated_mb": torch.cuda.memory_allocated(idx) / (1024.0 ** 2),
            "reserved_mb": torch.cuda.memory_reserved(idx) / (1024.0 ** 2),
            "peak_mb": torch.cuda.max_memory_allocated(idx) / (1024.0 ** 2),
        }
    except Exception:
        return {}


def format_snapshot(snap: dict) -> str:
    """Compact one-line representation for log messages."""
    if not snap:
        return "<no GPU>"
    frag = snap.get("reserved_mb", 0) - snap.get("allocated_mb", 0)
    return (
        f"free={snap.get('free_mb', 0):.0f}MiB "
        f"alloc={snap.get('allocated_mb', 0):.0f}MiB "
        f"resv={snap.get('reserved_mb', 0):.0f}MiB "
        f"frag={frag:.0f}MiB "
        f"peak={snap.get('peak_mb', 0):.0f}MiB"
    )


def reset_peak_mem_stats(device: Any = None) -> None:
    """Reset the per-device peak-allocated counter. Call before a request to
    measure peak usage on a per-request basis."""
    try:
        import torch
        if device is None:
            if not torch.cuda.is_available():
                return
            idx = torch.cuda.current_device()
        else:
            dev = torch.device(device)
            if dev.type != "cuda":
                return
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(idx)
    except Exception:
        pass


# Configurations tried in order of preference (cheapest first).
# Each entry is (chunk_size, kv_bits). The planner walks this list and
# picks the first configuration whose predicted peak fits.
#
# Ordering rationale:
#   - Larger chunks are faster (fewer forward passes per prompt) but
#     have higher per-step activation peaks. Prefer largest viable.
#   - int16 KV (no quantization) is highest quality but heaviest.
#     int8 is essentially free-quality. int4 has measurable but small
#     accuracy impact. Walk down only as forced.
_PLAN_LADDER: List[Tuple[int, int]] = [
    (1024, 16), (1024, 8),
    (512, 16),  (512, 8),
    (256, 8),
    (128, 8),   (128, 4),
    (64, 4),
    (32, 4),
]

# Margin we keep between predicted peak and free VRAM. Covers allocator
# fragmentation, scratch tensors, CUDA reserved-but-unallocated blocks,
# and approximation error in our memory model. 15% is empirically the
# point where prediction misses become rare on H100/L4/4090-class cards.
_SAFETY_MARGIN_FRAC = 0.15


@dataclass
class RequestPlan:
    """A concrete per-request configuration the engine should adopt.

    Constructed by ``OOMPlanner.plan()``. Applied via the
    ``OOMPlanner.apply()`` context manager which mutates engine
    attributes for the request's duration and restores them after.
    """
    chunk_size: int            # prefill_chunk_size to use during this request
    kv_bits: int               # 16 | 8 | 4
    prompt_tokens: int         # how many prompt tokens to actually process
                               # (may be < the request's original prompt
                               # length if auto-truncate kicked in)
    estimated_peak_mb: float   # predicted peak VRAM during prefill
    free_vram_mb: float        # what was free when we planned
    truncated_from: Optional[int] = None  # original prompt length if truncated

    def __str__(self) -> str:
        base = (
            f"chunk={self.chunk_size}, kv_bits={self.kv_bits}, "
            f"prompt_tokens={self.prompt_tokens}, "
            f"peak={self.estimated_peak_mb:.0f}/{self.free_vram_mb:.0f} MiB"
        )
        if self.truncated_from is not None:
            base += f" (truncated from {self.truncated_from})"
        return base


class RequestTooLargeError(Exception):
    """Raised when no configuration on the plan ladder fits.

    HTTP layer should translate this to 413 Payload Too Large with a
    body that names ``prompt_tokens``, the predicted peak, and the
    available VRAM so the caller knows how much they need to trim.
    """
    def __init__(
        self,
        prompt_tokens: int,
        peak_mb: float,
        free_mb: float,
        suggested_max_tokens: Optional[int] = None,
    ):
        self.prompt_tokens = prompt_tokens
        self.peak_mb = peak_mb
        self.free_mb = free_mb
        self.suggested_max_tokens = suggested_max_tokens
        msg = (
            f"prompt of {prompt_tokens} tokens cannot fit on this GPU at "
            f"any planner configuration (predicted peak {peak_mb:.0f} MiB "
            f"vs {free_mb:.0f} MiB free)"
        )
        if suggested_max_tokens is not None:
            msg += f"; reduce to ~{suggested_max_tokens} tokens or use a smaller model"
        super().__init__(msg)


class OOMPlanner:
    """Plans a per-request ``(chunk_size, kv_bits)`` configuration.

    Lifecycle:
        1. Constructed once at server start with ``engine`` and the
           probed ``cost_coefficients``.
        2. ``planner.plan(prompt_tokens, max_new_tokens)`` returns a
           ``RequestPlan`` or raises ``RequestTooLargeError``.
        3. ``with planner.apply(plan): engine.generate(...)`` adopts
           the plan, runs the request, and restores engine state.
    """

    def __init__(
        self,
        engine,
        cost_coefficients,
        *,
        auto_truncate: bool = False,
        safety_margin_frac: float = _SAFETY_MARGIN_FRAC,
        calibration_window: int = 256,
    ):
        self.engine = engine
        self.cc = cost_coefficients
        self.auto_truncate = auto_truncate
        self.safety_margin_frac = safety_margin_frac
        self.calibration = CalibrationTracker(window=calibration_window)

        # Cached model-shape derivatives — these don't change request to
        # request, so we compute them once. ``num_layers`` and
        # ``per_layer_mb`` come from the cost-model probe.
        cfg = getattr(engine.model, "config", None)
        self.hidden_dim = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or 4096
        )
        self.num_layers = cost_coefficients.num_layers
        self.num_heads = (
            getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "n_head", None)
            or 32
        )
        self.num_kv_heads = (
            getattr(cfg, "num_key_value_heads", None) or self.num_heads
        )
        self.head_dim = self.hidden_dim // max(self.num_heads, 1)

        # Which attention backend prefill actually runs through. This
        # decides whether peak scratch is the materialized O(K·N) fp32 score
        # matrix (eager) or the tiled/linear footprint of a flash kernel.
        # See ``_attention_scratch_bytes``.
        self.attn_impl = self._detect_prefill_attn_impl()

    # Backends whose prefill forward NEVER materializes the score matrix —
    # the K×N scores live in on-chip SRAM tiles, so the only HBM transients
    # are per-call Q/K/V working copies (linear in seq len). PyTorch SDPA
    # belongs here on kvboost's prefill path: fp16, causal,
    # ``attention_mask is None`` ⇒ it dispatches to the flash / mem-efficient
    # kernel, never the math backend. ``eager`` (and anything unrecognized)
    # is treated as materializing — the conservative side.
    _TILED_ATTN_IMPLS = frozenset({
        "sdpa", "flash_attention_2", "flash_attention_3",
        "sage", "triton_flash", "flashinfer", "kvboost_cuda",
    })

    def _detect_prefill_attn_impl(self) -> str:
        """Resolved attention impl HF committed to at load, lower-cased.

        Read from ``model.config._attn_implementation`` — where transformers
        records the backend it actually wired up, including our registered
        ``sage`` / ``triton_flash`` keys. Returns ``"unknown"`` when the
        attribute is absent so the caller falls back to the conservative
        (materializing) scratch model. Note: when sage/triton_flash
        self-disable at runtime they fall back to SDPA, which is still a
        tiled kernel on this path — so the classification stays correct
        without us having to watch their per-call runtime state.
        """
        cfg = getattr(self.engine.model, "config", None)
        impl = getattr(cfg, "_attn_implementation", None)
        return impl.lower() if isinstance(impl, str) else "unknown"

    # ── Memory estimation ────────────────────────────────────────────

    def _bytes_per_token_kv(self, kv_bits: int) -> float:
        """Per-token KV cache size across all layers in BYTES.

        ``2`` factor covers K and V tensors. ``kv_bits/8`` converts
        precision to bytes. GQA models have fewer KV heads than
        attention heads, which we account for via ``num_kv_heads``.
        """
        per_head = self.head_dim * (kv_bits / 8.0)
        per_layer = 2 * self.num_kv_heads * per_head
        return per_layer * self.num_layers

    def estimate_peak_mb(
        self,
        prompt_tokens: int,
        chunk_size: int,
        kv_bits: int,
        max_new_tokens: int = 0,
    ) -> float:
        """Predicted peak *transient* VRAM for this request, in MiB.

        "Transient" = memory allocated on top of what's already resident
        (model weights + any existing KV). The fitting check compares
        this against *free* VRAM, which already excludes the resident
        set — so the two are consistent.

        Structure: ``peak = kv_total + max(floor, chunk_act + scratch)``.
          - **KV total** (additive): every prompt + generated token gets a
            KV entry at ``kv_bits`` precision. Persists for the whole
            request and is live at the same time as the attention scratch,
            so it adds on top.
          - **Activation floor**: the per-forward working set across all
            decoder layers that exists regardless of prompt length —
            measured by the cost-model probe (``baseline_activation_mb``).
            Falls back to a per-layer heuristic when unmeasured.
          - **Chunk activation + attention scratch**: scale with the
            *effective* chunk (= ``min(chunk_size, prompt_tokens)`` — a
            single prefill forward never has more queries than the prompt
            length; using raw chunk_size here over-counted short prompts
            by chunk/prompt, the dominant source of the -55% residual).

        We take ``max(floor, chunk_act + scratch)`` rather than summing:
        the floor is the minimal-forward working set, which already
        contains a small chunk's activation. For a large chunk the scaled
        terms dominate; for a tiny prompt the floor dominates. Summing
        double-counted the small-chunk activation.
        """
        total_tokens = prompt_tokens + max_new_tokens

        kv_bytes = total_tokens * self._bytes_per_token_kv(kv_bits)

        # Per-forward activation floor (constant in prompt length).
        floor_bytes = self._activation_floor_mb() * (1024.0 ** 2)

        # A single prefill forward processes at most ``prompt_tokens``
        # queries — never more than the prompt, regardless of chunk_size.
        effective_chunk = min(chunk_size, max(prompt_tokens, 1))

        # One chunk's hidden states, fp16 (2 bytes).
        chunk_activation_bytes = effective_chunk * self.hidden_dim * 2

        # Literal attention-scratch allocation for the active backend — the
        # materialized O(K·N) score matrix only for eager; the tiled/linear
        # working set for the flash family. See _attention_scratch_bytes.
        scratch_bytes = self._attention_scratch_bytes(
            effective_chunk, prompt_tokens
        )

        attention_transient = max(
            floor_bytes, chunk_activation_bytes + scratch_bytes
        )
        peak = kv_bytes + attention_transient
        return peak / (1024.0 ** 2)

    def _attention_scratch_bytes(
        self, effective_chunk: int, prompt_tokens: int
    ) -> float:
        """Literal peak attention-scratch HBM allocation, in bytes, for the
        prefill backend in ``self.attn_impl``.

        This used to be modelled as the SDPA *math* backend worst case
        (``effective_chunk × prompt_tokens × n_heads`` fp32 scores ×2). That
        allocation is real only for ``eager``. Every flash/tiled backend
        kvboost runs keeps the scores in SRAM and never allocates the K×N
        matrix in HBM, so for those we cost the *literal* transients the
        kernel materializes — all linear in sequence length.
        """
        hd = self.head_dim
        q_elems = effective_chunk * self.num_heads * hd        # this chunk's Q
        kv_elems = prompt_tokens * self.num_kv_heads * hd       # full K (or V)

        if self.attn_impl not in self._TILED_ATTN_IMPLS:
            # eager (or unrecognized/forced-math): the full fp32 score matrix
            # + softmax temporary. The genuinely O(K·N) term, and the reason
            # chunking exists for this backend.
            return effective_chunk * prompt_tokens * self.num_heads * 4 * 2

        if self.attn_impl == "sage":
            # INT8 SageAttention upcasts to fp32 before quantising
            # (kernels/sage_attn._quant_per_token does ``q.float()``;
            # _smooth_and_quant_k does ``k.float()`` then ``k - delta``).
            # Per element: Q → fp32(4) + int8(1) + fp16 out(2);
            # K → fp32(4) + (k−delta) fp32(4) + int8(1); V → fp16(2).
            # Per-token scales (one fp32 per token-head) are negligible.
            return q_elems * (4 + 1 + 2) + kv_elems * (4 + 4 + 1) + kv_elems * 2

        # fp16 flash family (flash_attention_2 / triton_flash / flashinfer /
        # sdpa): contiguous Q/K/V working copies (2 bytes) + fp16 output + a
        # small fp32 logsumexp (one per query-head). Budget ~2 fp16 copies of
        # K/V to cover ``.contiguous()`` plus the kernel's internal staging.
        return (q_elems + 2 * kv_elems) * 2 + effective_chunk * self.num_heads * 4

    def _activation_floor_mb(self) -> float:
        """Per-forward activation working-set floor in MiB.

        Prefers the measured ``baseline_activation_mb`` from the cost-
        model probe. When unmeasured (0.0 — non-CUDA, or probe failed),
        falls back to a crude per-layer heuristic: a few layers' worth
        of hidden-state buffers. The heuristic is deliberately modest;
        on CPU the fitting check sees infinite free VRAM anyway, so the
        floor only matters on a real GPU where the probe succeeds.
        """
        if self.cc is not None and getattr(self.cc, "baseline_activation_mb", 0.0) > 0:
            return float(self.cc.baseline_activation_mb)
        # Heuristic: residual stream + MLP intermediate (~4× hidden) for
        # a handful of layers held in flight. fp16 (2 bytes).
        approx_bytes = 4 * self.hidden_dim * self.num_layers * 2
        return approx_bytes / (1024.0 ** 2)

    def _free_vram_mb(self) -> float:
        """Live snapshot of free VRAM on the engine's device."""
        try:
            import torch
            device = self.engine.device
            if not str(device).startswith("cuda"):
                return float("inf")
            idx = torch.device(device).index
            if idx is None:
                idx = torch.cuda.current_device()
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            return free_bytes / (1024.0 ** 2)
        except Exception as e:
            log.warning("mem_get_info failed (%s); assuming infinite VRAM", e)
            return float("inf")

    # ── Planning ─────────────────────────────────────────────────────

    def plan(self, prompt_tokens: int, max_new_tokens: int = 0) -> RequestPlan:
        """Pick a configuration that fits within the safety margin.

        Tries ``_PLAN_LADDER`` in order (cheapest first), returns the
        first config whose predicted peak ≤ ``free_vram × (1 - margin)``.
        If nothing fits and ``auto_truncate=False``, raises
        ``RequestTooLargeError``. If ``auto_truncate=True``, binary
        searches for the longest prefix that fits with the most
        aggressive config.
        """
        free_mb = self._free_vram_mb()
        budget_mb = free_mb * (1.0 - self.safety_margin_frac)

        for chunk_size, kv_bits in _PLAN_LADDER:
            peak = self.estimate_peak_mb(
                prompt_tokens, chunk_size, kv_bits, max_new_tokens
            )
            if peak <= budget_mb:
                log.debug(
                    "Plan committed: chunk=%d kv_bits=%d peak=%.0f MiB ≤ "
                    "budget=%.0f MiB (free=%.0f MiB)",
                    chunk_size, kv_bits, peak, budget_mb, free_mb,
                )
                return RequestPlan(
                    chunk_size=chunk_size,
                    kv_bits=kv_bits,
                    prompt_tokens=prompt_tokens,
                    estimated_peak_mb=peak,
                    free_vram_mb=free_mb,
                )

        # Nothing on the ladder fits at this prompt size.
        most_aggressive = _PLAN_LADDER[-1]
        worst_peak = self.estimate_peak_mb(
            prompt_tokens, most_aggressive[0], most_aggressive[1], max_new_tokens
        )

        if not self.auto_truncate:
            # Binary search for a *suggested* max prompt size to put in the
            # 413 error message — not used, just informational.
            suggested = self._find_max_fitting_prompt(
                budget_mb, max_new_tokens, most_aggressive,
            )
            raise RequestTooLargeError(
                prompt_tokens=prompt_tokens,
                peak_mb=worst_peak,
                free_mb=free_mb,
                suggested_max_tokens=suggested,
            )

        # Auto-truncate path: find the largest prefix that fits at the
        # most aggressive config, return a plan with truncated prompt.
        max_fitting = self._find_max_fitting_prompt(
            budget_mb, max_new_tokens, most_aggressive,
        )
        if max_fitting is None or max_fitting <= 0:
            raise RequestTooLargeError(
                prompt_tokens=prompt_tokens,
                peak_mb=worst_peak,
                free_mb=free_mb,
            )
        truncated_peak = self.estimate_peak_mb(
            max_fitting, most_aggressive[0], most_aggressive[1], max_new_tokens
        )
        log.warning(
            "Auto-truncating prompt from %d to %d tokens to fit (peak %.0f "
            "MiB ≤ budget %.0f MiB)",
            prompt_tokens, max_fitting, truncated_peak, budget_mb,
        )
        return RequestPlan(
            chunk_size=most_aggressive[0],
            kv_bits=most_aggressive[1],
            prompt_tokens=max_fitting,
            estimated_peak_mb=truncated_peak,
            free_vram_mb=free_mb,
            truncated_from=prompt_tokens,
        )

    def _find_max_fitting_prompt(
        self,
        budget_mb: float,
        max_new_tokens: int,
        config: Tuple[int, int],
    ) -> Optional[int]:
        """Binary search the largest ``prompt_tokens`` that fits in budget
        at the given (chunk, kv_bits) config. Returns None if even 1
        token doesn't fit (model + overhead is already over budget)."""
        chunk_size, kv_bits = config
        lo, hi = 1, 1_000_000   # 1M tokens upper bound; will narrow fast
        if self.estimate_peak_mb(1, chunk_size, kv_bits, max_new_tokens) > budget_mb:
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            peak = self.estimate_peak_mb(mid, chunk_size, kv_bits, max_new_tokens)
            if peak <= budget_mb:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ── Pre/post telemetry hooks ─────────────────────────────────────

    def log_pre_request(self, plan: RequestPlan) -> None:
        """Log mem snapshot before dispatch and reset the peak counter so
        the post-request snapshot reflects only this request's usage.

        Call this immediately before passing the plan into
        ``engine.generate(prefill_chunk_size=..., kv_cache_bits=...)``.
        """
        device = getattr(self.engine, "device", None)
        reset_peak_mem_stats(device)
        pre = gpu_mem_snapshot(device)
        # Stash the resident set so log_post_request can subtract it from
        # the absolute peak to recover the TRANSIENT actual — the same
        # quantity estimate_peak_mb predicts. Single-worker engine ⇒ one
        # in-flight request ⇒ a plain instance attr is safe.
        self._allocated_before_mb = pre.get("allocated_mb", 0.0)
        log.info("Plan committed: %s | mem-pre: %s", plan, format_snapshot(pre))

    def log_post_request(self, plan: RequestPlan, prompt_tokens: int) -> None:
        """Log mem snapshot after the request returns and record the
        prediction residual in the calibration tracker.

        ``prompt_tokens`` is needed so the calibration tracker can index
        residuals by request shape (helps separate "small prompts always
        overshoot" from "long prompts always undershoot").

        The residual compares predicted-transient against ACTUAL-transient
        (= absolute peak − resident-before-request), not against the raw
        ``max_memory_allocated``. The raw peak includes resident model
        weights, which estimate_peak_mb deliberately excludes; comparing
        the two directly produced meaningless +200%/+30000% residuals.
        """
        device = getattr(self.engine, "device", None)
        post = gpu_mem_snapshot(device)
        abs_peak = post.get("peak_mb", 0.0)
        allocated_before = getattr(self, "_allocated_before_mb", 0.0)
        # Transient = peak high-water-mark minus what was resident at
        # reset time. Clamp at 0 in case of measurement skew.
        actual_transient = max(0.0, abs_peak - allocated_before)
        predicted = plan.estimated_peak_mb
        if actual_transient > 0 and predicted > 0:
            error_pct = (actual_transient - predicted) / predicted * 100.0
            log.info(
                "Request done | mem-post: %s | "
                "predicted=%.0fMiB actual=%.0fMiB (transient, abs_peak=%.0f) "
                "err=%+.1f%%",
                format_snapshot(post), predicted, actual_transient,
                abs_peak, error_pct,
            )
            self.calibration.record(
                chunk_size=plan.chunk_size,
                kv_bits=plan.kv_bits,
                prompt_tokens=prompt_tokens,
                predicted_mb=predicted,
                actual_mb=actual_transient,
            )
        else:
            log.info("Request done | mem-post: %s", format_snapshot(post))

    # ── Telemetry ────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Stats payload for ``/v1/stats``. Per-request history is logged
        at INFO; this returns aggregate calibration stats."""
        return {
            "auto_truncate": self.auto_truncate,
            "safety_margin_frac": self.safety_margin_frac,
            "ladder": [
                {"chunk_size": c, "kv_bits": b} for c, b in _PLAN_LADDER
            ],
            "model_shape": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            },
            "attn_impl": self.attn_impl,
            "attn_scratch_model": (
                "materialized" if self.attn_impl not in self._TILED_ATTN_IMPLS
                else "sage_int8" if self.attn_impl == "sage"
                else "tiled_fp16"
            ),
            "free_vram_mb_now": self._free_vram_mb(),
            "calibration": self.calibration.stats(),
        }
