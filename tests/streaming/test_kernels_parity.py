"""Parity tests for the AWQ kernel wrappers.

The torch-reference path always exists; on CUDA we additionally check that
Marlin / ExLlamaV2 paths agree with it when their kernels are importable.
"""

from __future__ import annotations

import pytest
import torch

from kvboost.streaming.kernels import (
    awq_dequantize_reference,
    awq_linear,
    marlin_awq_available,
    exllama_awq_available,
)


def test_dequant_reference_shape(small_awq_layer):
    weight = awq_dequantize_reference(
        small_awq_layer["qweight"],
        small_awq_layer["scales"],
        small_awq_layer["qzeros"],
        small_awq_layer["group_size"],
    )
    assert weight.shape == (
        small_awq_layer["in_features"],
        small_awq_layer["out_features"],
    )
    assert weight.dtype == small_awq_layer["scales"].dtype


def test_torch_linear_fused_matches_reference_dequant_matmul(small_awq_layer):
    """The fused chunked path (``_torch_awq_linear``) must produce the same
    result as the reference dense-dequant-then-matmul, within fp16 noise.
    """
    from kvboost.streaming.kernels import _torch_awq_linear

    x = torch.randn(4, small_awq_layer["in_features"], dtype=torch.float16)

    fused = _torch_awq_linear(
        x,
        small_awq_layer["qweight"],
        small_awq_layer["scales"],
        small_awq_layer["qzeros"],
        bias=None,
        group_size=small_awq_layer["group_size"],
    )

    weight = awq_dequantize_reference(
        small_awq_layer["qweight"],
        small_awq_layer["scales"],
        small_awq_layer["qzeros"],
        small_awq_layer["group_size"],
    )
    reference = x @ weight

    # Strict: same algebra, same fp16 — only difference is op ordering
    # across chunks. Allow tiny accumulation drift.
    assert fused.shape == reference.shape
    rel = (fused - reference).abs() / reference.abs().clamp(min=1.0)
    assert rel.max() < 1e-2


def test_torch_linear_fused_handles_various_chunk_sizes(small_awq_layer):
    """Chunk size is a memory/perf knob; output must be invariant."""
    from kvboost.streaming.kernels import _torch_awq_linear

    x = torch.randn(2, small_awq_layer["in_features"], dtype=torch.float16)
    outputs = []
    for cg in (1, 2, 4, 16):
        out = _torch_awq_linear(
            x,
            small_awq_layer["qweight"],
            small_awq_layer["scales"],
            small_awq_layer["qzeros"],
            bias=None,
            group_size=small_awq_layer["group_size"],
            chunk_groups=cg,
        )
        outputs.append(out)

    # Chunk size should not change the result meaningfully. Allow tiny
    # fp16 noise from non-associative accumulation across chunk boundaries.
    for o in outputs[1:]:
        rel = (o - outputs[0]).abs() / outputs[0].abs().clamp(min=1.0)
        assert rel.max() < 5e-3, f"chunk-size sensitivity: rel max {rel.max().item()}"


def test_torch_linear_matches_manual_matmul(small_awq_layer):
    x = torch.randn(4, small_awq_layer["in_features"], dtype=torch.float16)
    out = awq_linear(
        x,
        small_awq_layer["qweight"],
        small_awq_layer["scales"],
        small_awq_layer["qzeros"],
        bias=None,
        group_size=small_awq_layer["group_size"],
        prefer="torch",
    )
    weight = awq_dequantize_reference(
        small_awq_layer["qweight"],
        small_awq_layer["scales"],
        small_awq_layer["qzeros"],
        small_awq_layer["group_size"],
    )
    expected = x @ weight
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3)


def test_torch_linear_bias_added(small_awq_layer):
    x = torch.randn(2, small_awq_layer["in_features"], dtype=torch.float16)
    bias = torch.randn(small_awq_layer["out_features"], dtype=torch.float16)
    weight = awq_dequantize_reference(
        small_awq_layer["qweight"],
        small_awq_layer["scales"],
        small_awq_layer["qzeros"],
        small_awq_layer["group_size"],
    )
    with_bias = awq_linear(
        x, small_awq_layer["qweight"], small_awq_layer["scales"],
        small_awq_layer["qzeros"], bias=bias,
        group_size=small_awq_layer["group_size"], prefer="torch",
    )
    expected = x @ weight + bias
    # Relative tolerance: AWQ products can swing to large magnitudes in fp16.
    rel = (with_bias - expected).abs() / expected.abs().clamp(min=1.0)
    assert rel.max() < 1e-2


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.skipif(not marlin_awq_available(), reason="Marlin kernel unavailable")
def test_marlin_matches_reference(small_awq_layer):
    layer = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in small_awq_layer.items()}
    x = torch.randn(2, layer["in_features"], dtype=torch.float16, device="cuda")

    ref = awq_linear(
        x, layer["qweight"], layer["scales"], layer["qzeros"], bias=None,
        group_size=layer["group_size"], prefer="torch",
    )
    marlin = awq_linear(
        x, layer["qweight"], layer["scales"], layer["qzeros"], bias=None,
        group_size=layer["group_size"], prefer="marlin",
    )
    assert torch.allclose(marlin, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.skipif(not exllama_awq_available(), reason="ExLlamaV2 kernel unavailable")
def test_exllama_matches_reference(small_awq_layer):
    layer = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in small_awq_layer.items()}
    x = torch.randn(2, layer["in_features"], dtype=torch.float16, device="cuda")

    ref = awq_linear(
        x, layer["qweight"], layer["scales"], layer["qzeros"], bias=None,
        group_size=layer["group_size"], prefer="torch",
    )
    exl = awq_linear(
        x, layer["qweight"], layer["scales"], layer["qzeros"], bias=None,
        group_size=layer["group_size"], prefer="exllama_v2",
    )
    assert torch.allclose(exl, ref, atol=5e-3, rtol=5e-3)
