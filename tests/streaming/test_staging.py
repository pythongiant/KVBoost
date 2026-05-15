"""Tests for the staging-slot layout (CPU emulation).

The ``StagingArena`` itself is CUDA-only; here we exercise just
``SlotLayout.from_layer_specs`` which is pure-Python and platform-independent.
The CUDA arena is exercised in test_scheduler_smoke.py under a CUDA skip.
"""

from __future__ import annotations

import pytest
import torch

from kvboost.streaming.awq_loader import LayerSpec, TensorSpec
from kvboost.streaming.staging import SlotLayout, align_up


def _make_layer(idx: int, resident: bool = False) -> LayerSpec:
    tensors = {
        "qweight": TensorSpec(
            name="qweight", path="/tmp/fake",  # type: ignore[arg-type]
            shape=(64, 8), dtype=torch.int32,
            layer_idx=idx, is_quantized=True, nbytes=64 * 8 * 4,
        ),
        "scales": TensorSpec(
            name="scales", path="/tmp/fake",  # type: ignore[arg-type]
            shape=(8, 64), dtype=torch.float16,
            layer_idx=idx, is_quantized=False, nbytes=8 * 64 * 2,
        ),
    }
    return LayerSpec(layer_idx=idx, tensors=tensors, resident=resident)


def test_align_up_basic():
    assert align_up(0, 16) == 0
    assert align_up(1, 16) == 16
    assert align_up(15, 16) == 16
    assert align_up(16, 16) == 16
    assert align_up(17, 16) == 32


def test_align_up_rejects_bad_alignment():
    with pytest.raises(ValueError):
        align_up(10, 0)


def test_slot_layout_picks_max_required_bytes():
    layers = [_make_layer(0), _make_layer(1), _make_layer(2, resident=True)]
    layout = SlotLayout.from_layer_specs(layers, alignment=16)
    assert layout.slot_bytes >= 64 * 8 * 4 + 8 * 64 * 2
    assert "qweight" in layout.placements
    assert "scales" in layout.placements
    # Offsets must be aligned.
    for placement in layout.placements.values():
        assert placement.offset % layout.alignment == 0


def test_slot_layout_rejects_schema_drift():
    a = _make_layer(0)
    b = _make_layer(1)
    # Mutate layer b so it has a tensor 'a' doesn't.
    b.tensors["extra"] = TensorSpec(
        name="extra", path="/tmp/fake",  # type: ignore[arg-type]
        shape=(4,), dtype=torch.float16, layer_idx=1, nbytes=8,
    )
    with pytest.raises(ValueError):
        SlotLayout.from_layer_specs([a, b])


def test_slot_layout_empty_when_all_resident():
    layers = [_make_layer(0, resident=True), _make_layer(1, resident=True)]
    layout = SlotLayout.from_layer_specs(layers, streamed_only=True)
    assert layout.slot_bytes == 0
