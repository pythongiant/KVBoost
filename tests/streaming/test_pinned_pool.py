"""Tests for the pinned-host memory pool."""

from __future__ import annotations

import pytest
import torch

from kvboost.streaming.pinned_pool import PinnedMemoryPool


def _cpu_tensor(nbytes: int) -> torch.Tensor:
    return torch.zeros(nbytes // 4, dtype=torch.int32)


def test_register_and_get_roundtrip():
    pool = PinnedMemoryPool()
    t = _cpu_tensor(1024)
    h = pool.register(t)
    assert pool.get(h).data_ptr() == t.data_ptr()
    assert h in pool
    assert pool.total_bytes == 1024


def test_release_frees_bytes():
    pool = PinnedMemoryPool()
    h = pool.register(_cpu_tensor(2048))
    pool.release(h)
    assert pool.total_bytes == 0
    assert h not in pool


def test_budget_blocks_overflow():
    pool = PinnedMemoryPool(max_bytes=2048)
    pool.register(_cpu_tensor(1024))
    with pytest.raises(RuntimeError):
        pool.register(_cpu_tensor(2048))


def test_lru_eviction_evicts_oldest():
    fills: list[int] = []

    def rehydrate(h: int) -> torch.Tensor:
        fills.append(h)
        return _cpu_tensor(512)

    pool = PinnedMemoryPool(max_bytes=1024, lru_enabled=True, rehydrate_fn=rehydrate)
    h1 = pool.register(_cpu_tensor(512))
    h2 = pool.register(_cpu_tensor(512))
    # Touch h1 so h2 becomes oldest.
    pool.get(h1)

    h3 = pool.register(_cpu_tensor(512))
    assert h3 in pool
    assert h2 not in pool  # evicted
    assert h1 in pool

    # Accessing h2 again should trigger the rehydrate callback.
    pool.get(h2)
    assert fills == [h2]


def test_single_allocation_over_budget_raises():
    pool = PinnedMemoryPool(max_bytes=512, lru_enabled=False)
    with pytest.raises(RuntimeError):
        pool.register(_cpu_tensor(1024))
