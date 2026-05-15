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


def marlin_awq_available() -> bool:
    """Whether a Marlin-style int4 GEMM kernel is importable on this system."""
    return _GEMM_FN is not None and torch.cuda.is_available()


def marlin_awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    group_size: int,
) -> torch.Tensor:
    """Invoke the resolved Marlin/AWQ GEMM kernel.

    The first kernel signature we try is the historical AutoAWQ one:
    ``gemm_forward_cuda(x, qweight, scales, qzeros, group_size)``. If that
    fails (e.g. vLLM's awq_marlin_gemm uses a different argument order),
    we let the caller decide whether to fall back.
    """
    if _GEMM_FN is None:
        raise RuntimeError("no Marlin/AWQ GEMM kernel available")

    try:
        out = _GEMM_FN(x, qweight, scales, qzeros, group_size)
    except TypeError:
        # Try the vLLM-style signature.
        out = _GEMM_FN(x, qweight, qzeros, scales, group_size)

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
