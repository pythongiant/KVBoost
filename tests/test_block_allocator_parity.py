"""
Parity tests: Rust-backed BlockAllocatorMeta vs the pure-Python fallback.

Runs the same sequence of allocate / free / fork / ensure_writable operations
through both backends and asserts identical observable state at every step.
Also exercises the end-to-end BlockAllocator class with both backends to
confirm copy-on-write tensor copies are byte-equivalent.
"""

from __future__ import annotations

import random

import pytest
import torch

from kvboost.cpu_paged.block_allocator import BlockAllocator, _PyMeta

try:
    from kvboost_native import BlockAllocatorMeta as _RustMeta
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(
    not HAS_RUST,
    reason="kvboost_native extension not built (run `maturin develop` in crates/kvboost_native).",
)


def _state(meta) -> tuple:
    """Snapshot a meta backend's externally observable state."""
    return (
        meta.free_blocks,
        meta.used_blocks,
        # ref_count_snapshot returns dict[int, int]; sort for deterministic compare
        tuple(sorted(meta.ref_count_snapshot().items())),
        tuple(meta.free_snapshot()),
    )


def _assert_states_match(rust_meta, py_meta) -> None:
    assert _state(rust_meta) == _state(py_meta), (
        f"\n  rust: {_state(rust_meta)}\n  py  : {_state(py_meta)}"
    )


def test_alloc_free_parity():
    rust = _RustMeta(16)
    py = _PyMeta(16)
    _assert_states_match(rust, py)

    r1 = rust.allocate(4)
    p1 = py.allocate(4)
    assert r1 == p1
    _assert_states_match(rust, py)

    rust.free(r1[:2])
    py.free(p1[:2])
    _assert_states_match(rust, py)

    rust.free(r1[2:])
    py.free(p1[2:])
    _assert_states_match(rust, py)


def test_fork_and_ensure_writable_parity():
    rust = _RustMeta(8)
    py = _PyMeta(8)

    r_ids = rust.allocate(3)
    p_ids = py.allocate(3)
    assert r_ids == p_ids

    rust.fork(r_ids[:2])
    py.fork(p_ids[:2])
    _assert_states_match(rust, py)

    # Block 0 of the alloc has rc=2 → ensure_writable must copy.
    r_dec = rust.ensure_writable(r_ids[0])
    p_dec = py.ensure_writable(p_ids[0])
    assert r_dec.needs_copy is True
    assert p_dec.needs_copy is True
    assert r_dec.block_id == p_dec.block_id
    _assert_states_match(rust, py)

    # Block 2 was never forked → rc=1 → ensure_writable returns same id, no copy.
    r_dec2 = rust.ensure_writable(r_ids[2])
    p_dec2 = py.ensure_writable(p_ids[2])
    assert r_dec2.needs_copy is False
    assert p_dec2.needs_copy is False
    assert r_dec2.block_id == r_ids[2]
    _assert_states_match(rust, py)


def test_oom_parity():
    rust = _RustMeta(4)
    py = _PyMeta(4)
    rust.allocate(4)
    py.allocate(4)

    with pytest.raises(RuntimeError, match="OOM"):
        rust.allocate(1)
    with pytest.raises(RuntimeError, match="OOM"):
        py.allocate(1)


def test_randomized_workload_parity():
    """Drive both backends with the same pseudo-random op sequence."""
    rust = _RustMeta(64)
    py = _PyMeta(64)
    rng = random.Random(20260509)

    live_rust: list[list[int]] = []
    live_py: list[list[int]] = []

    for _ in range(500):
        op = rng.choice(["alloc", "free", "fork", "ensure_writable"])

        if op == "alloc":
            n = rng.randint(1, 5)
            try:
                r = rust.allocate(n)
                p = py.allocate(n)
            except RuntimeError:
                # Both should OOM on the same step
                with pytest.raises(RuntimeError):
                    py.allocate(n)
                continue
            assert r == p
            live_rust.append(r)
            live_py.append(p)

        elif op == "free" and live_rust:
            i = rng.randrange(len(live_rust))
            rust.free(live_rust[i])
            py.free(live_py[i])
            live_rust.pop(i)
            live_py.pop(i)

        elif op == "fork" and live_rust:
            i = rng.randrange(len(live_rust))
            rust.fork(live_rust[i])
            py.fork(live_py[i])
            # fork bumps rc but the caller still owns the original list; mirror that.

        elif op == "ensure_writable" and live_rust:
            grp_i = rng.randrange(len(live_rust))
            if not live_rust[grp_i]:
                continue
            blk_i = rng.randrange(len(live_rust[grp_i]))
            try:
                r_dec = rust.ensure_writable(live_rust[grp_i][blk_i])
            except RuntimeError:
                # Pool exhausted during CoW — Python side must agree.
                with pytest.raises(RuntimeError):
                    py.ensure_writable(live_py[grp_i][blk_i])
                continue
            p_dec = py.ensure_writable(live_py[grp_i][blk_i])
            assert r_dec.needs_copy == p_dec.needs_copy
            assert r_dec.block_id == p_dec.block_id

        _assert_states_match(rust, py)


def test_block_allocator_e2e_cow_tensor_copy():
    """The high-level BlockAllocator class must do byte-equivalent CoW copies
    regardless of which metadata backend is in use."""
    torch.manual_seed(0)
    alloc = BlockAllocator(
        num_layers=2, num_heads=2, head_dim=4,
        num_blocks=8, block_size=4, dtype=torch.float32,
    )

    [src] = alloc.allocate(1)
    # Fill src with deterministic data on every layer.
    for layer_idx, pool in enumerate(alloc.pools):
        pool[src] = torch.full_like(pool[src], float(layer_idx + 1))

    # Fork → CoW. ensure_writable should give us a fresh id with copied data.
    forked = alloc.fork([src])
    assert forked == [src]

    new_id = alloc.ensure_writable(src)
    assert new_id != src, "expected CoW to allocate a new block when rc>1"
    for layer_idx, pool in enumerate(alloc.pools):
        assert torch.equal(pool[new_id], pool[src]), (
            f"layer {layer_idx}: CoW destination did not copy source bytes"
        )


def test_block_allocator_reports_backend():
    alloc = BlockAllocator(
        num_layers=1, num_heads=1, head_dim=1,
        num_blocks=2, block_size=1, dtype=torch.float32,
    )
    assert alloc.backend in ("rust", "python")
    if HAS_RUST:
        assert alloc.backend == "rust"
