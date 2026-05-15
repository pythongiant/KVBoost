"""Rust StreamingSchedulerBackend bookkeeping parity tests."""

from __future__ import annotations

import pytest

try:
    from kvboost_native import StreamingSchedulerBackend  # type: ignore
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(
    not HAS_RUST,
    reason="kvboost_native extension not built (run `maturin develop` in crates/kvboost_native).",
)


def test_assign_slot_round_robin():
    sched = StreamingSchedulerBackend(num_slots=2)
    a = sched.assign_slot(0)
    b = sched.assign_slot(1)
    c = sched.assign_slot(2)
    assert a.slot_id == 0
    assert b.slot_id == 1
    assert c.slot_id == 0
    # The third assignment evicts the layer in slot 0 (layer_idx=0).
    assert c.evicted_layer == 0


def test_get_slot_for_layer_tracks_assignments():
    sched = StreamingSchedulerBackend(num_slots=2)
    sched.assign_slot(5)
    assert sched.get_slot_for_layer(5) == 1  # 5 % 2 == 1
    assert sched.get_slot_for_layer(99) is None


def test_record_events_roundtrip():
    sched = StreamingSchedulerBackend(num_slots=2)
    sched.assign_slot(0)
    sched.record_transfer_event(0, 42)
    sched.record_compute_event(0, 99)
    assert sched.get_transfer_event(0) == 42
    assert sched.get_compute_event(0) == 99


def test_reset_clears_state():
    sched = StreamingSchedulerBackend(num_slots=2)
    sched.assign_slot(0)
    sched.assign_slot(1)
    sched.reset()
    assert sched.get_slot_for_layer(0) is None
    assert sched.get_slot_for_layer(1) is None


def test_allocate_event_monotonic():
    sched = StreamingSchedulerBackend(num_slots=2)
    ids = [sched.allocate_event() for _ in range(4)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 4


def test_invalid_slot_id_raises():
    sched = StreamingSchedulerBackend(num_slots=2)
    with pytest.raises(ValueError):
        sched.record_transfer_event(5, 1)
