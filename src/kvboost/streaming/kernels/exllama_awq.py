"""ExLlamaV2 AWQ kernel fallback.

ExLlamaV2 exposes a 4-bit GEMM that accepts shapes Marlin sometimes rejects
(e.g. odd ``out_features`` or unusual ``group_size``). We probe for the
``exllamav2_kernels`` extension and adapt to whichever entry point is exported.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


_GEMM_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("exllamav2_kernels", "gemm_half_q_half"),
    ("exllamav2_ext", "gemm_half_q_half"),
)


def _try_resolve() -> Optional[Callable[..., Any]]:
    for module_name, attr in _GEMM_CANDIDATES:
        try:
            mod = __import__(module_name, fromlist=[attr])
        except Exception:
            continue
        fn = getattr(mod, attr, None)
        if fn is not None:
            logger.debug("resolved %s.%s for exllamav2 path", module_name, attr)
            return fn
    return None


_GEMM_FN: Optional[Callable[..., Any]] = _try_resolve()


def exllama_awq_available() -> bool:
    return _GEMM_FN is not None and torch.cuda.is_available()


def exllama_awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    group_size: int,
) -> torch.Tensor:
    if _GEMM_FN is None:
        raise RuntimeError("no ExLlamaV2 AWQ kernel available")

    # ExLlamaV2's signature varies by version; pass positional and let
    # callers catch TypeError to fall through to torch reference.
    out = _GEMM_FN(x, qweight, scales, qzeros, group_size)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


__all__ = [
    "exllama_awq_available",
    "exllama_awq_linear",
]
