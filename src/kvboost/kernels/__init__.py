"""Proven external inference kernels kvboost routes to at runtime.

Currently: FlashInfer decode-attention (see ``flashinfer_attn``). Each kernel
is gated on availability + a numerical self-check, and falls back to PyTorch
SDPA so a missing or misbehaving kernel never corrupts output.
"""
from .flashinfer_attn import (
    flashinfer_available,
    install_flashinfer_attention,
    resolve_attn_impl,
)

__all__ = [
    "flashinfer_available",
    "install_flashinfer_attention",
    "resolve_attn_impl",
]
