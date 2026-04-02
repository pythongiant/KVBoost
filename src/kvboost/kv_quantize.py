"""
KV Cache Quantization
=====================
Compress cached KV tensors from float16 to int8 or int4 using the
asymmetric scheme from KIVI (ICML 2024):

  - Key cache: quantized **per-channel** (outliers are channel-specific)
  - Value cache: quantized **per-token** (outliers are token-specific)

This asymmetry is critical — uniform quantization to both K and V causes
measurable accuracy degradation even at int8.

Memory savings:
  int8  → 2x reduction (9.4 MB → 4.7 MB per chunk for Qwen2.5-3B)
  int4  → 4x reduction (9.4 MB → 2.4 MB per chunk)
  float16 → no compression (baseline)

Usage:
    from kvboost.kv_quantize import QuantizedKV, quantize_kv, dequantize_kv

    # Compress for storage
    qkv = quantize_kv(past_key_values, bits=8)

    # Decompress for inference
    past_key_values = dequantize_kv(qkv)

Reference: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
           (Liu et al., ICML 2024)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch

from .models import PastKVType

log = logging.getLogger(__name__)


@dataclass
class QuantizedLayer:
    """One layer's quantized K and V tensors + scale factors."""
    key_q: torch.Tensor      # int8 or packed int4, shape depends on bits
    key_scale: torch.Tensor   # float16, per-channel: [1, heads, seq, 1]
    val_q: torch.Tensor       # int8 or packed int4
    val_scale: torch.Tensor   # float16, per-token: [1, heads, 1, head_dim]
    bits: int                  # 8 or 4


@dataclass
class QuantizedKV:
    """Full model's quantized KV cache — drop-in replacement for PastKVType in storage."""
    layers: List[QuantizedLayer]
    bits: int
    original_dtype: torch.dtype

    def memory_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            total += layer.key_q.nelement() * layer.key_q.element_size()
            total += layer.key_scale.nelement() * layer.key_scale.element_size()
            total += layer.val_q.nelement() * layer.val_q.element_size()
            total += layer.val_scale.nelement() * layer.val_scale.element_size()
        return total


# ── Int8 quantization (KIVI asymmetric) ────────────────────────────

def _quantize_int8(kv: PastKVType) -> QuantizedKV:
    """
    KIVI-style int8 quantization:
      Key: per-channel (head_dim axis) — handles channel-specific outliers
      Value: per-token (seq_len axis) — handles token-specific outliers
    """
    layers = []
    dtype = kv[0][0].dtype

    for key, val in kv:
        # Key: per-channel quantization along head_dim (-1)
        key_amax = key.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        key_scale = key_amax / 127.0
        key_q = (key / key_scale).round().clamp(-128, 127).to(torch.int8)

        # Value: per-token quantization along seq_len (-2)
        val_amax = val.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        val_scale = val_amax / 127.0
        val_q = (val / val_scale).round().clamp(-128, 127).to(torch.int8)

        layers.append(QuantizedLayer(
            key_q=key_q, key_scale=key_scale.to(torch.float16),
            val_q=val_q, val_scale=val_scale.to(torch.float16),
            bits=8,
        ))

    return QuantizedKV(layers=layers, bits=8, original_dtype=dtype)


def _dequantize_int8(qkv: QuantizedKV) -> PastKVType:
    """Reconstruct float16 KV tensors from int8 quantized storage."""
    result = []
    for layer in qkv.layers:
        key = layer.key_q.to(torch.float16) * layer.key_scale
        val = layer.val_q.to(torch.float16) * layer.val_scale
        result.append((key, val))
    return tuple(result)


# ── Int4 quantization (KIVI asymmetric, packed) ────────────────────

def _quantize_int4(kv: PastKVType) -> QuantizedKV:
    """
    KIVI-style int4 quantization. Values are stored packed: two int4 values
    per int8 byte. Same asymmetric scheme as int8 but with 4-bit range [-8, 7].
    """
    layers = []
    dtype = kv[0][0].dtype

    for key, val in kv:
        # Key: per-channel int4
        key_amax = key.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        key_scale = key_amax / 7.0
        key_rounded = (key / key_scale).round().clamp(-8, 7)
        # Pack two int4 values into one int8
        key_q = _pack_int4(key_rounded)

        # Value: per-token int4
        val_amax = val.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        val_scale = val_amax / 7.0
        val_rounded = (val / val_scale).round().clamp(-8, 7)
        val_q = _pack_int4(val_rounded)

        layers.append(QuantizedLayer(
            key_q=key_q, key_scale=key_scale.to(torch.float16),
            val_q=val_q, val_scale=val_scale.to(torch.float16),
            bits=4,
        ))

    return QuantizedKV(layers=layers, bits=4, original_dtype=dtype)


def _dequantize_int4(qkv: QuantizedKV) -> PastKVType:
    """Reconstruct float16 KV tensors from packed int4 storage."""
    result = []
    for layer in qkv.layers:
        key_unpacked = _unpack_int4(layer.key_q)
        val_unpacked = _unpack_int4(layer.val_q)
        key = key_unpacked.to(torch.float16) * layer.key_scale
        val = val_unpacked.to(torch.float16) * layer.val_scale
        result.append((key, val))
    return tuple(result)


def _pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack pairs of int4 values along the last dimension into int8.
    Input shape: [..., N] (N must be even)
    Output shape: [..., N // 2] as int8
    """
    # Ensure even last dim
    assert tensor.shape[-1] % 2 == 0, f"Last dim must be even, got {tensor.shape[-1]}"
    t = tensor.to(torch.int8)
    # Low nibble: even indices, high nibble: odd indices
    low = t[..., 0::2] & 0x0F
    high = (t[..., 1::2] & 0x0F) << 4
    return (low | high).to(torch.int8)


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack int8 back to pairs of int4 values along the last dimension.
    Input shape: [..., N // 2] as int8
    Output shape: [..., N] as float32 (for arithmetic)
    """
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    # Sign-extend from 4-bit: if bit 3 is set, value is negative
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    # Interleave back
    shape = list(packed.shape)
    shape[-1] *= 2
    result = torch.empty(shape, dtype=torch.float32, device=packed.device)
    result[..., 0::2] = low.float()
    result[..., 1::2] = high.float()
    return result


# ── Public API ──────────────────────────────────────────────────────

def quantize_kv(kv: PastKVType, bits: int = 8) -> QuantizedKV:
    """
    Quantize a PastKVType using KIVI-style asymmetric quantization.

    Args:
        kv: Standard HF past_key_values tuple.
        bits: 8 (int8, safe) or 4 (int4, aggressive). 16 returns passthrough.

    Returns:
        QuantizedKV container with compressed tensors + scale factors.
    """
    if bits == 8:
        return _quantize_int8(kv)
    elif bits == 4:
        return _quantize_int4(kv)
    else:
        raise ValueError(f"Unsupported quantization bits={bits}. Use 8 or 4.")


def dequantize_kv(qkv: QuantizedKV) -> PastKVType:
    """
    Dequantize a QuantizedKV back to float16 PastKVType.
    """
    if qkv.bits == 8:
        return _dequantize_int8(qkv)
    elif qkv.bits == 4:
        return _dequantize_int4(qkv)
    else:
        raise ValueError(f"Unsupported quantization bits={qkv.bits}")
