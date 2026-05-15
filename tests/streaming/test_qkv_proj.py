"""Tests for the slot-pointer-rebind contract of StreamingQLinear (M2)."""

from __future__ import annotations

import pytest
import torch

from kvboost.streaming.qkv_proj import StreamingQLinear


def test_unbound_forward_raises(small_awq_layer):
    layer = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
    )
    with pytest.raises(RuntimeError):
        layer(torch.zeros(1, small_awq_layer["in_features"], dtype=torch.float16))


def test_rebind_then_forward(small_awq_layer):
    layer = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
    )
    layer.rebind(
        qweight=small_awq_layer["qweight"],
        scales=small_awq_layer["scales"],
        qzeros=small_awq_layer["qzeros"],
    )
    assert layer.is_bound

    x = torch.randn(3, small_awq_layer["in_features"], dtype=torch.float16)
    out = layer(x)
    assert out.shape == (3, small_awq_layer["out_features"])


def test_rebind_swap_does_not_change_module_identity(small_awq_layer):
    """Marlin's launch-config cache keys on module identity + shape; the
    pointer-rebind must not produce a new submodule object."""
    layer = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
    )
    before = id(layer)
    layer.rebind(
        qweight=small_awq_layer["qweight"],
        scales=small_awq_layer["scales"],
        qzeros=small_awq_layer["qzeros"],
    )
    layer.rebind(
        qweight=small_awq_layer["qweight"].clone(),
        scales=small_awq_layer["scales"].clone(),
        qzeros=small_awq_layer["qzeros"].clone(),
    )
    assert id(layer) == before


def test_cache_dense_matches_streaming(small_awq_layer):
    """Cached-dense rebind must produce numerically identical output to the
    streaming (per-forward dequant) path. Both go through the same
    ``awq_dequantize_reference`` — the only difference is *when* it runs.
    """
    streaming = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
        cache_dense=False,
    )
    cached = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
        cache_dense=True,
    )
    for m in (streaming, cached):
        m.rebind(
            qweight=small_awq_layer["qweight"],
            scales=small_awq_layer["scales"],
            qzeros=small_awq_layer["qzeros"],
        )

    x = torch.randn(2, small_awq_layer["in_features"], dtype=torch.float16)
    out_stream = streaming(x)
    out_cached = cached(x)
    assert torch.equal(out_stream, out_cached)


def test_cache_dense_drops_packed_tensors(small_awq_layer):
    """After a cache_dense rebind, the packed tensors should be freed so a
    later memory check can reason about the dense-only footprint.
    """
    cached = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
        cache_dense=True,
    )
    cached.rebind(
        qweight=small_awq_layer["qweight"],
        scales=small_awq_layer["scales"],
        qzeros=small_awq_layer["qzeros"],
    )
    assert cached._qweight is None
    assert cached._scales is None
    assert cached._qzeros is None
    assert cached._dense_weight is not None
    assert cached._dense_weight.shape == (
        small_awq_layer["in_features"],
        small_awq_layer["out_features"],
    )


def test_cache_dense_preserves_module_identity(small_awq_layer):
    """Rebinding a cache_dense layer twice must not produce a new module."""
    cached = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
        cache_dense=True,
    )
    before = id(cached)
    cached.rebind(
        qweight=small_awq_layer["qweight"],
        scales=small_awq_layer["scales"],
        qzeros=small_awq_layer["qzeros"],
    )
    cached.rebind(
        qweight=small_awq_layer["qweight"].clone(),
        scales=small_awq_layer["scales"].clone(),
        qzeros=small_awq_layer["qzeros"].clone(),
    )
    assert id(cached) == before


def test_cache_dense_with_bias(small_awq_layer):
    bias = torch.randn(small_awq_layer["out_features"], dtype=torch.float16)
    cached = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
        cache_dense=True,
    )
    cached.rebind(
        qweight=small_awq_layer["qweight"],
        scales=small_awq_layer["scales"],
        qzeros=small_awq_layer["qzeros"],
        bias=bias,
    )
    x = torch.randn(3, small_awq_layer["in_features"], dtype=torch.float16)
    out = cached(x)
    assert out.shape == (3, small_awq_layer["out_features"])


def test_rebind_validates_shape(small_awq_layer):
    layer = StreamingQLinear(
        in_features=small_awq_layer["in_features"],
        out_features=small_awq_layer["out_features"],
        group_size=small_awq_layer["group_size"],
        prefer="torch",
    )
    with pytest.raises(ValueError):
        layer.rebind(
            qweight=torch.zeros(1, 1, dtype=torch.int32),
            scales=small_awq_layer["scales"],
            qzeros=small_awq_layer["qzeros"],
        )
