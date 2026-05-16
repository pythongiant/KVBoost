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
    *,
    chunk_groups: int = 16,
) -> torch.Tensor:
    """Fused dequant + matmul, chunked along input groups.

    Never materializes the full dense ``(in_features, out_features)`` weight
    matrix — instead processes ``chunk_groups`` groups (= ``chunk_groups *
    group_size`` rows) at a time and accumulates into the output. Peak
    additional GPU memory per call is roughly
    ``chunk_groups * group_size * out_features * 2 bytes`` (e.g. for
    ``chunk_groups=16, group_size=128, out_features=5120`` that's ~20 MB,
    vs ~280 MB for the dense materialization).

    Mathematically equivalent (within fp16 rounding) to
    ``x @ awq_dequantize_reference(...) + bias``. Used by
    :class:`StreamingQLinear` when the torch fallback is selected, which
    is the common case for larger models on small GPUs where the dense
    materialization would OOM.
    """
    pack = 8
    in_features = qweight.shape[0]
    out_features = qweight.shape[1] * pack
    num_groups = scales.shape[0]

    shifts = _shift_table(qweight.device, qweight.dtype)

    out_shape = x.shape[:-1] + (out_features,)
    out = torch.zeros(out_shape, dtype=scales.dtype, device=x.device)

    x_compute = x.to(scales.dtype) if x.dtype != scales.dtype else x

    for g_start in range(0, num_groups, chunk_groups):
        g_end = min(g_start + chunk_groups, num_groups)
        row_start = g_start * group_size
        row_end = min(g_end * group_size, in_features)

        # Dequant only the rows in this group chunk.
        qw_chunk = qweight[row_start:row_end].unsqueeze(-1)
        unpacked = ((qw_chunk >> shifts) & 0xF).to(scales.dtype)
        unpacked = unpacked.reshape(row_end - row_start, out_features)

        qz_chunk = qzeros[g_start:g_end].unsqueeze(-1)
        zeros_grp = ((qz_chunk >> shifts) & 0xF).to(scales.dtype)
        zeros_grp = zeros_grp.reshape(g_end - g_start, out_features)

        scales_grp = scales[g_start:g_end]  # (g_chunk, out_features)

        # Broadcast scales/zeros across the group_size rows inside each group.
        # repeat_interleave is cheap because the group_size is small.
        rows_in_chunk = row_end - row_start
        # Use indexing instead of repeat_interleave to keep memory low.
        local_group_idx = (
            torch.arange(rows_in_chunk, device=qweight.device) // group_size
        )
        scales_full = scales_grp[local_group_idx]
        zeros_full = zeros_grp[local_group_idx]

        # In-place fuse: weight_chunk = (unpacked - zeros) * scales
        unpacked.sub_(zeros_full).mul_(scales_full)
        del zeros_full, scales_full

        # Accumulate matmul into out.
        out.add_(x_compute[..., row_start:row_end] @ unpacked)
        del unpacked

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
