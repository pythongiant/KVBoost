"""Proven external/JIT inference kernels kvboost routes to at runtime.

* FlashInfer decode-attention (``flashinfer_attn``) — ``--attn-impl flashinfer``.
* SageAttention INT8 prefill + FP16 Triton flash (``sage_attn``) —
  ``--attn-impl sage`` / ``--attn-impl triton_flash``.

Each kernel is gated on availability + a one-time numerical self-check, and
falls back to PyTorch SDPA so a missing or misbehaving kernel never corrupts
output. ``resolve_attn_impl`` registers the requested backend with HuggingFace
(if its dependency is present) before model load, else downgrades to ``sdpa``.
"""
import logging

from .flashinfer_attn import flashinfer_available, install_flashinfer_attention
from .sage_attn import (
    install_sage_attention,
    sage_attention_forward,
    sage_available,
    triton_available,
    triton_flash_attention_forward,
)

_log = logging.getLogger("kvboost.kernels")


def resolve_attn_impl(requested: str) -> str:
    """Map a requested attn-impl to one HF can actually load.

    Registers the backend with HuggingFace if its dependency is importable,
    otherwise falls back to ``"sdpa"`` with a warning. ``"auto"`` and stock
    impls (``"sdpa"``, ``"eager"``, ``"flash_attention_2"``) pass through.
    """
    if requested == "flashinfer":
        if install_flashinfer_attention():
            return "flashinfer"
        _log.warning("attn-impl 'flashinfer' requested but unavailable; using sdpa.")
        return "sdpa"

    if requested in ("sage", "triton_flash"):
        if install_sage_attention():
            return requested
        _log.warning(
            "attn-impl '%s' requested but Triton is unavailable; using sdpa.",
            requested,
        )
        return "sdpa"

    return requested


__all__ = [
    "flashinfer_available",
    "install_flashinfer_attention",
    "install_sage_attention",
    "sage_attention_forward",
    "sage_available",
    "triton_available",
    "triton_flash_attention_forward",
    "resolve_attn_impl",
]
