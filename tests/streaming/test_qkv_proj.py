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
