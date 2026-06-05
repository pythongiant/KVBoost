"""Marlin int4 GEMM wrapper.

Marlin lives in ``awq_ext`` / ``vllm._C`` / ``autoawq_kernels`` depending on
the install. The exact symbol name has shifted between releases, so we probe
a small set of common entry points at import time and pick whichever resolves.

If none are present, :func:`marlin_awq_available` returns ``False`` and the
higher-level ``awq_linear`` falls back to ExLlamaV2 or pure torch.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


# (module-path, attr) candidates, in order of preference. Each function is
# expected to follow the signature
#   fn(x, qweight, scales, qzeros, group_size, ...) -> out
# but the exact extra args differ between kernels, so we adapt below.
_GEMM_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("awq_ext", "gemm_forward_cuda"),
    ("awq_ext", "awq_gemm"),
    ("autoawq_kernels", "awq_gemm"),
    ("vllm._C.ops", "awq_marlin_gemm"),
)

_REPACK_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("awq_ext", "marlin_repack_from_awq"),
    ("autoawq_kernels", "awq_marlin_repack"),
    ("vllm._C.ops", "awq_marlin_repack"),
)


def _try_resolve(candidates: tuple[tuple[str, str], ...]) -> Optional[Callable[..., Any]]:
    for module_name, attr in candidates:
        try:
            mod = __import__(module_name, fromlist=[attr])
        except Exception:
            continue
        fn = getattr(mod, attr, None)
        if fn is not None:
            logger.debug("resolved %s.%s for marlin path", module_name, attr)
            return fn
    return None


def _resolve_gemm(
    candidates: tuple[tuple[str, str], ...]
) -> tuple[Optional[Callable[..., Any]], bool]:
    """Resolve the GEMM fn AND whether it consumes the Marlin-repacked layout.

    Only vLLM's ``awq_marlin_gemm`` reads the repacked layout; the autoawq
    kernels (``gemm_forward_cuda`` / ``awq_gemm``) want the ORIGINAL AWQ packing,
    so repacking the weights under them yields garbage. We track this so the
    loader's repack step can no-op for the raw-layout kernels.
    """
    for module_name, attr in candidates:
        try:
            mod = __import__(module_name, fromlist=[attr])
        except Exception:
            continue
        fn = getattr(mod, attr, None)
        if fn is not None:
            needs_repack = attr == "awq_marlin_gemm"
            logger.debug(
                "resolved %s.%s for AWQ GEMM (marlin_layout=%s)",
                module_name, attr, needs_repack,
            )
            return fn, needs_repack
    return None, False


_GEMM_FN, _GEMM_NEEDS_REPACK = _resolve_gemm(_GEMM_CANDIDATES)
_REPACK_FN: Optional[Callable[..., Any]] = _try_resolve(_REPACK_CANDIDATES)


# ── Probe the kernel's call signature once at load time ──────────────────────
# AWQ int4 GEMM kernels DISAGREE on the call layout, so we can't assume one:
#   * autoawq  awq_ext.gemm_forward_cuda : (x, qw, SCALES, ZEROS, split_k_iters)
#   * vLLM     awq_gemm                  : (x, qw, ZEROS, SCALES, split_k_iters)
#   * vLLM     awq_marlin_gemm           : (x, qw, ZEROS, SCALES, group_size)
# They differ in BOTH the scales/zeros order AND the trailing int. We probe each
# combination with tiny tensors and cache the first that runs. Because scales is
# fp16 and zeros is int32, the WRONG scales/zeros order hits a kernel dtype check
# and raises — so try/except reliably discriminates the order.
# (An earlier version hard-coded only the vLLM (zeros, scales) order, which
# silently disabled autoawq's awq_ext on every box that had it -> the slow torch
# dequant fallback. That was the bug.)

_SPLIT_K_ITERS = 8  # autoawq's default; tunes K-dim reduction parallelism

# (label, scales_first, last_kind), most-preferred first. ``scales_first``
# selects the autoawq (True) vs vLLM (False) order of the scales/zeros pair.
_GEMM_SIG_CANDIDATES = (
    ("autoawq (x,qw,scales,zeros,split_k)",    True,  "split_k"),
    ("vllm    (x,qw,zeros,scales,split_k)",    False, "split_k"),
    ("vllm    (x,qw,zeros,scales,group_size)", False, "group_size"),
    ("autoawq (x,qw,scales,zeros,group_size)", True,  "group_size"),
)


def _make_gemm_caller(scales_first: bool, last_kind: str) -> Callable[..., Any]:
    """Wrap _GEMM_FN so callers invoke it canonically as
    ``caller(x_2d, qweight, qzeros, scales, group_size)`` regardless of the
    kernel's native scales/zeros order or trailing-int convention."""
    def _call(x_2d, qw, qz, sc, group_size):  # noqa: ANN001
        last = _SPLIT_K_ITERS if last_kind == "split_k" else group_size
        if scales_first:
            return _GEMM_FN(x_2d, qw, sc, qz, last)
        return _GEMM_FN(x_2d, qw, qz, sc, last)
    return _call


def _probe_gemm_signature() -> Optional[Callable[..., Any]]:
    """Return a caller invoking the resolved GEMM fn with the correct arg order,
    or None if no kernel is available or every known signature fails the probe.
    """
    if _GEMM_FN is None:
        return None

    try:
        import torch as _torch
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            return None

        # Minimal valid AWQ shapes: K=256 (2 groups of 128), N=256 (=32 × pack 8).
        group_size_probe = 128
        in_f, out_f = 256, 32
        n_groups = in_f // group_size_probe
        x_p      = _torch.zeros(1, in_f,            dtype=_torch.float16, device=device)
        qw_p     = _torch.zeros(in_f, out_f,        dtype=_torch.int32,   device=device)
        scales_p = _torch.ones(n_groups, out_f * 8, dtype=_torch.float16, device=device)
        qzeros_p = _torch.zeros(n_groups, out_f,    dtype=_torch.int32,   device=device)

        for label, scales_first, last_kind in _GEMM_SIG_CANDIDATES:
            caller = _make_gemm_caller(scales_first, last_kind)
            try:
                out = caller(x_p, qw_p, qzeros_p, scales_p, group_size_probe)
            except (RuntimeError, TypeError):
                continue
            # Guard against a silently-accepted wrong layout: the output must be
            # (M, out_features) and finite.
            if out.shape[0] != 1 or out.shape[-1] != out_f * 8 \
                    or not _torch.isfinite(out).all():
                continue
            logger.info("marlin/awq GEMM: using %s signature", label)
            return caller

        logger.warning(
            "marlin/awq GEMM fn %r: no known signature worked during probe "
            "(tried autoawq + vLLM orders x split_k/group_size) — disabling "
            "kernel. AWQ will use ExLlamaV2 or the torch dequant fallback.",
            _GEMM_FN,
        )
        return None

    except Exception as exc:
        logger.warning(
            "marlin/awq GEMM probe failed unexpectedly: %s. "
            "Falling back to ExLlamaV2 / torch.", exc,
        )
        return None


# Resolved once at module import. None means "no usable AWQ GEMM kernel found."
_GEMM_CALLER: Optional[Callable[..., Any]] = _probe_gemm_signature()


def marlin_awq_available() -> bool:
    """Whether a Marlin-style int4 GEMM kernel is importable on this system."""
    return _GEMM_CALLER is not None and torch.cuda.is_available()


def marlin_awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    group_size: int,
) -> torch.Tensor:
    """Invoke the resolved AWQ GEMM kernel (autoawq or vLLM Marlin variant).

    The kernel signature is probed once at import time (_GEMM_CALLER).
    All known kernels use the layout (x, qweight, qzeros, scales, last_arg);
    qzeros comes BEFORE scales. The old code had them swapped, causing a
    RuntimeError on every call that silently fell through to the slow torch
    dequant fallback.
    """
    if _GEMM_CALLER is None:
        raise RuntimeError("no Marlin/AWQ GEMM kernel available")

    pack = 8
    out_features = qweight.shape[1] * pack
    out_shape = x.shape[:-1] + (out_features,)
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()

    out = _GEMM_CALLER(x_2d, qweight, qzeros, scales, group_size)
    out = out.reshape(out_shape)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def awq_marlin_repack(
    qweight: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """Repack an AWQ ``qweight`` into Marlin's layout, if a repack kernel
    is available. Falls back to returning the input contiguous if not.
    """
    # Repack ONLY when the resolved GEMM actually consumes the Marlin layout
    # (vLLM awq_marlin_gemm). The autoawq raw-layout kernels must keep the
    # ORIGINAL AWQ packing, or the GEMM reads garbage.
    if _REPACK_FN is None or not _GEMM_NEEDS_REPACK:
        return qweight.contiguous()
    try:
        return _REPACK_FN(qweight, in_features, out_features, num_bits)
    except TypeError:
        return _REPACK_FN(qweight, num_bits=num_bits)


__all__ = [
    "marlin_awq_available",
    "marlin_awq_linear",
    "awq_marlin_repack",
]
