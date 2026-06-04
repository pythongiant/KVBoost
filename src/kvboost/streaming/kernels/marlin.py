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


_GEMM_FN: Optional[Callable[..., Any]] = _try_resolve(_GEMM_CANDIDATES)
_REPACK_FN: Optional[Callable[..., Any]] = _try_resolve(_REPACK_CANDIDATES)


# ── Probe the kernel's call signature once at load time ──────────────────────
# All known AWQ GEMM kernels use (x, qweight, qzeros, scales, last_arg) where
# last_arg is either split_k_iters (autoawq style) or group_size (vLLM Marlin
# style). We probe with tiny tensors here and cache the working call so the
# hot forward path never pays try/except overhead.

_SPLIT_K_ITERS = 8  # autoawq's default; tunes K-dim parallelism


def _probe_gemm_signature() -> Optional[Callable[..., Any]]:
    """Return a zero-argument callable that calls the resolved GEMM fn with the
    correct arg order, or None if no kernel is available or the probe fails.

    All known kernels share the layout:
        fn(x_2d, qweight, qzeros, scales, last_arg)
    where last_arg is split_k_iters (int) or group_size (int).
    The old code had scales/qzeros SWAPPED on the first try, causing a
    RuntimeError on every forward that silently fell through to the slow
    torch dequant path.
    """
    if _GEMM_FN is None:
        return None

    try:
        import torch as _torch
        # Minimal tensors: in=128 (one group), out=16 (× pack=8 → 128 packed).
        group_size_probe = 128
        in_f, out_f = group_size_probe, 16
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            return None

        x_p       = _torch.zeros(1, in_f,   dtype=_torch.float16, device=device)
        qw_p      = _torch.zeros(in_f, out_f, dtype=_torch.int32,   device=device)
        scales_p  = _torch.ones(1, out_f * 8, dtype=_torch.float16, device=device)
        qzeros_p  = _torch.zeros(1, out_f,    dtype=_torch.int32,   device=device)

        # Try split_k_iters style (autoawq / awq_ext).
        try:
            _GEMM_FN(x_p, qw_p, qzeros_p, scales_p, _SPLIT_K_ITERS)
            logger.debug("marlin/awq GEMM: using split_k_iters signature")

            def _call(x_2d, qw, qz, sc, *_):  # noqa: ANN001
                return _GEMM_FN(x_2d, qw, qz, sc, _SPLIT_K_ITERS)

            return _call
        except (RuntimeError, TypeError):
            pass

        # Try group_size style (vLLM awq_marlin_gemm).
        try:
            _GEMM_FN(x_p, qw_p, qzeros_p, scales_p, group_size_probe)
            logger.debug("marlin/awq GEMM: using group_size signature")

            def _call(x_2d, qw, qz, sc, group_size):  # noqa: ANN001
                return _GEMM_FN(x_2d, qw, qz, sc, group_size)

            return _call
        except (RuntimeError, TypeError):
            pass

        logger.warning(
            "marlin/awq GEMM fn %r: neither split_k_iters nor group_size "
            "signature worked during probe — disabling kernel. AWQ will use "
            "ExLlamaV2 or the torch dequant fallback.",
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
    if _REPACK_FN is None:
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
