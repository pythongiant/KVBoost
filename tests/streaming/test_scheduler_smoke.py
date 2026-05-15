"""CUDA smoke tests for StreamingScheduler.

Covers M2 (single-stream prefetch sanity) and M3 (dual-buffer event flow).
Each test is skipped automatically when CUDA is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from kvboost.streaming.awq_loader import LayerSpec, TensorSpec


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _make_layer(idx: int, resident: bool, shape=(32, 4)) -> LayerSpec:
    return LayerSpec(
        layer_idx=idx,
        resident=resident,
        tensors={
            "qweight": TensorSpec(
                name="qweight", path="/tmp/fake",  # type: ignore[arg-type]
                shape=shape, dtype=torch.int32,
                layer_idx=idx, is_quantized=True,
                nbytes=shape[0] * shape[1] * 4,
            ),
        },
    )


@cuda
def test_scheduler_runs_resident_only():
    from kvboost.streaming.scheduler import StreamingScheduler

    layers = [_make_layer(i, resident=True) for i in range(3)]

    def prefetch(_idx: int) -> dict[str, torch.Tensor]:
        raise AssertionError("resident layer must not prefetch")

    def run_layer(idx, hidden, _past, _views, _slot, _plan) -> torch.Tensor:
        return hidden + idx

    sched = StreamingScheduler(
        layer_specs=layers,
        prefetch_source_fn=prefetch,
        run_layer_fn=run_layer,
        device=torch.device("cuda"),
    )
    hidden = torch.zeros(4, device="cuda")
    past = [None] * 3
    out = sched.forward(hidden, past)
    assert out.tolist() == [0 + 1 + 2] * 4


@cuda
def test_scheduler_streams_and_recycles_slot():
    from kvboost.streaming.scheduler import StreamingScheduler

    layers = [_make_layer(i, resident=False) for i in range(4)]
    sources: list[int] = []

    def prefetch(idx: int) -> dict[str, torch.Tensor]:
        sources.append(idx)
        return {"qweight": torch.zeros(32, 4, dtype=torch.int32)}

    def run_layer(idx, hidden, _past, views, slot_id, _plan) -> torch.Tensor:
        assert views is not None
        assert slot_id in (0, 1)
        assert views["qweight"].shape == (32, 4)
        return hidden + 1.0

    sched = StreamingScheduler(
        layer_specs=layers,
        prefetch_source_fn=prefetch,
        run_layer_fn=run_layer,
        device=torch.device("cuda"),
    )
    hidden = torch.zeros(2, device="cuda")
    past = [None] * len(layers)
    out = sched.forward(hidden, past)
    assert torch.allclose(out, torch.full_like(out, 4.0))
    # All four layers must have been prefetched exactly once.
    assert sorted(sources) == [0, 1, 2, 3]


@cuda
def test_scheduler_mixes_resident_and_streamed():
    from kvboost.streaming.scheduler import StreamingScheduler

    plan = [True, False, False, True]  # resident pattern
    layers = [_make_layer(i, resident=plan[i]) for i in range(4)]

    def prefetch(_idx: int) -> dict[str, torch.Tensor]:
        return {"qweight": torch.zeros(32, 4, dtype=torch.int32)}

    seen_streamed: list[int] = []

    def run_layer(idx, hidden, _past, views, slot_id, _plan) -> torch.Tensor:
        if plan[idx]:
            assert views is None and slot_id is None
        else:
            seen_streamed.append(idx)
            assert views is not None
        return hidden

    sched = StreamingScheduler(
        layer_specs=layers,
        prefetch_source_fn=prefetch,
        run_layer_fn=run_layer,
        device=torch.device("cuda"),
    )
    sched.forward(torch.zeros(2, device="cuda"), [None] * 4)
    assert seen_streamed == [1, 2]
