"""Shared fixtures for the streaming-backend test suite."""

from __future__ import annotations

import pytest
import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture
def cuda_only():
    if not cuda_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def small_awq_layer():
    """Tiny synthetic AWQ layer suitable for kernel parity tests.

    Dimensions are chosen to satisfy AutoAWQ's group-size and packing
    constraints: ``in_features`` divisible by ``group_size``, ``out_features``
    divisible by ``pack=8``.
    """
    torch.manual_seed(0)
    in_features = 128
    out_features = 64
    group_size = 32
    pack = 8

    qweight = torch.randint(
        low=0,
        high=2**31 - 1,
        size=(in_features, out_features // pack),
        dtype=torch.int32,
    )
    scales = torch.randn(
        in_features // group_size,
        out_features,
        dtype=torch.float16,
    ) * 0.01
    qzeros = torch.randint(
        low=0,
        high=2**31 - 1,
        size=(in_features // group_size, out_features // pack),
        dtype=torch.int32,
    )
    return {
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "qweight": qweight,
        "scales": scales,
        "qzeros": qzeros,
    }
