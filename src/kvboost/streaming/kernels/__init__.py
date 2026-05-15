"""AWQ kernel wrappers for streamed linear layers.

Public entry points:

- :func:`awq_linear` — choose Marlin/ExLlamaV2/torch fallback automatically.
- :func:`awq_dequantize_reference` — pure-torch reference for parity tests.

The wrappers accept the standard AWQ tensor layout produced by AutoAWQ:

- ``qweight``: ``(in_features, out_features // pack)`` int32, packed 4-bit
- ``scales``:  ``(in_features // group_size, out_features)`` fp16
- ``qzeros``:  ``(in_features // group_size, out_features // pack)`` int32
- ``bias``:    ``(out_features,)`` fp16 or None

``pack`` is 8 for 4-bit AWQ. ``group_size`` is read from ``quantize_config.json``
and passed in explicitly.
"""

from __future__ import annotations

from typing import Optional

import torch


# AutoAWQ packs 8 nibbles per int32 in this column order:
#   col_idx = [0, 4, 1, 5, 2, 6, 3, 7]
# See AutoAWQ awq.utils.utils.unpack_awq. We pre-compute the shift table
# per (device, dtype) so the dequant kernel doesn't recreate it each call.
_AWQ_BIT_ORDER: tuple[int, ...] = (0, 4, 1, 5, 2, 6, 3, 7)
_SHIFT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _shift_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _SHIFT_CACHE.get(key)
    if cached is not None:
        return cached
    order = torch.tensor(_AWQ_BIT_ORDER, device=device, dtype=dtype)
    shifts = (order * 4).view(1, 1, len(_AWQ_BIT_ORDER))
    _SHIFT_CACHE[key] = shifts
    return shifts

from .exllama_awq import exllama_awq_linear, exllama_awq_available
from .marlin import (
    marlin_awq_linear,
    marlin_awq_available,
    awq_marlin_repack,
)


def awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    group_size: int,
    prefer: str = "auto",
) -> torch.Tensor:
    """Run an AWQ-quantized linear with the best available kernel.

    ``prefer`` may be ``"auto"``, ``"marlin"``, ``"exllama_v2"``, or ``"torch"``.
    ``"torch"`` is the pure-Python dequant path — slow, but device-portable and
    used by parity tests.
    """
    if prefer == "torch":
        return _torch_awq_linear(x, qweight, scales, qzeros, bias, group_size)

    if prefer in ("auto", "marlin") and marlin_awq_available():
        try:
            return marlin_awq_linear(
                x, qweight, scales, qzeros, bias, group_size=group_size
            )
        except (RuntimeError, NotImplementedError):
            if prefer == "marlin":
                raise

    if prefer in ("auto", "exllama_v2") and exllama_awq_available():
        try:
            return exllama_awq_linear(
                x, qweight, scales, qzeros, bias, group_size=group_size
            )
        except (RuntimeError, NotImplementedError):
            if prefer == "exllama_v2":
                raise

    return _torch_awq_linear(x, qweight, scales, qzeros, bias, group_size)


def awq_dequantize_reference(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize AWQ ``qweight`` into a dense ``(in, out)`` matrix.

    Reference implementation used by tests. Matches AutoAWQ's bit ordering.
    """
    if qweight.dtype != torch.int32:
        raise ValueError(f"qweight must be int32, got {qweight.dtype}")
    if qzeros.dtype != torch.int32:
        raise ValueError(f"qzeros must be int32, got {qzeros.dtype}")

    pack = 8  # 32 / 4
    in_features = qweight.shape[0]
    out_features = qweight.shape[1] * pack

    shifts = _shift_table(qweight.device, qweight.dtype)

    qw = qweight.unsqueeze(-1)  # (in, out//pack, 1)
    unpacked = (qw >> shifts) & 0xF  # (in, out//pack, pack)
    unpacked = unpacked.reshape(in_features, out_features)

    qz = qzeros.unsqueeze(-1)
    unpacked_zeros = (qz >> shifts) & 0xF
    unpacked_zeros = unpacked_zeros.reshape(qzeros.shape[0], out_features)

    # Broadcast scales / zeros across the group dimension.
    group_idx = torch.arange(in_features, device=qweight.device) // group_size
    scales_full = scales[group_idx]                     # (in, out) fp16
    zeros_full = unpacked_zeros[group_idx].to(scales.dtype)

    return (unpacked.to(scales.dtype) - zeros_full) * scales_full


def _torch_awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: Optional[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    weight = awq_dequantize_reference(qweight, scales, qzeros, group_size)
    out = x.to(weight.dtype) @ weight
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


__all__ = [
    "awq_linear",
    "awq_dequantize_reference",
    "awq_marlin_repack",
    "marlin_awq_available",
    "exllama_awq_available",
]
