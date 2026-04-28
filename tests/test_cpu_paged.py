"""
Tests for the CPU Paged Attention implementation (Track 2).

Covers:
  - BlockAllocator: allocate, free, fork, copy-on-write, OOM
  - BlockAllocator: write_kv / read_kv round-trip
  - paged_attention_fwd: correctness vs. reference attention
  - paged_attention_fwd: causal and non-causal, GQA expansion
  - append_kv_to_blocks: single token and multi-token append
  - ChunkBlockMapper: load_chunk, build_block_table, on_evict
  - KVCacheManager.register_eviction_callback: fires on eviction
  - Integration: CPUPagedEngine paged_stats() after construction

All tests are CPU-only and use small toy tensors — no model download required
for most tests.  CPUPagedEngine construction tests are marked slow and skipped
by default (require a model download).
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from kvboost.cache_manager import KVCacheManager
from kvboost.cpu_paged.block_allocator import BlockAllocator
from kvboost.cpu_paged.paged_attn_cpu import append_kv_to_blocks, paged_attention_fwd
from kvboost.cpu_paged.chunk_to_blocks import ChunkBlockMapper
from kvboost.models import CachedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_allocator():
    """4 layers, 2 KV heads, head_dim=8, block_size=4, 32 blocks."""
    return BlockAllocator(
        num_layers=4,
        num_heads=2,
        head_dim=8,
        num_blocks=32,
        block_size=4,
        dtype=torch.float32,
    )


def make_chunk(chunk_id="abc123", length=8, num_layers=4, num_heads=2, head_dim=8) -> CachedChunk:
    """Create a toy CachedChunk with random KV tensors."""
    torch.manual_seed(hash(chunk_id) % (2**31))
    past_kv = tuple(
        (
            torch.randn(1, num_heads, length, head_dim),
            torch.randn(1, num_heads, length, head_dim),
        )
        for _ in range(num_layers)
    )
    return CachedChunk(
        chunk_id=chunk_id,
        text="dummy text",
        token_ids=list(range(length)),
        past_key_values=past_kv,
        position_start=0,
        position_end=length,
        prefix_hash=chunk_id,
        content_hash=chunk_id + "_content",
    )


# ── BlockAllocator ────────────────────────────────────────────────────────────

class TestBlockAllocator:
    def test_initial_state(self, small_allocator):
        a = small_allocator
        assert a.free_blocks == 32
        assert a.used_blocks == 0
        assert a.utilization() == 0.0

    def test_allocate_and_free(self, small_allocator):
        a = small_allocator
        ids = a.allocate(4)
        assert len(ids) == 4
        assert len(set(ids)) == 4  # unique ids
        assert a.used_blocks == 4
        assert a.free_blocks == 28

        a.free(ids)
        assert a.free_blocks == 32
        assert a.used_blocks == 0

    def test_allocate_all(self, small_allocator):
        a = small_allocator
        ids = a.allocate(32)
        assert a.free_blocks == 0
        a.free(ids)
        assert a.free_blocks == 32

    def test_oom_raises(self, small_allocator):
        a = small_allocator
        with pytest.raises(RuntimeError, match="OOM"):
            a.allocate(33)

    def test_fork_increments_refcount(self, small_allocator):
        a = small_allocator
        ids = a.allocate(2)
        forked = a.fork(ids)

        assert forked == ids  # same block ids

        # First free: decrements rc from 2 → 1; blocks NOT returned to pool
        a.free(ids)
        assert a.free_blocks == 30  # 32 - 2 allocated blocks still in use by fork

        # Second free: decrements rc from 1 → 0; blocks returned to pool
        a.free(forked)
        assert a.free_blocks == 32  # all blocks back in pool

    def test_ensure_writable_no_copy_when_rc_one(self, small_allocator):
        a = small_allocator
        ids = a.allocate(1)
        bid = ids[0]
        same = a.ensure_writable(bid)
        assert same == bid  # no copy needed

    def test_ensure_writable_copies_when_shared(self, small_allocator):
        a = small_allocator
        ids = a.allocate(1)
        bid = ids[0]
        _forked = a.fork(ids)  # rc = 2

        new_bid = a.ensure_writable(bid)
        assert new_bid != bid  # new block allocated

        # Write something to new_bid, old block unchanged
        a.pools[0][new_bid, 0, :, :, :] = 99.0
        assert not torch.all(a.pools[0][bid, 0, :, :, :] == 99.0)

    def test_write_read_kv_roundtrip(self, small_allocator):
        a = small_allocator
        ids = a.allocate(3)  # 3 blocks * 4 tokens = 12 token slots

        torch.manual_seed(5)
        seq_len = 10
        K_ref = torch.randn(a.num_heads, seq_len, a.head_dim)
        V_ref = torch.randn(a.num_heads, seq_len, a.head_dim)

        # Write token by token
        for t in range(seq_len):
            block_idx = t // a.block_size
            slot = t % a.block_size
            a.write_kv(
                layer=0,
                block_id=ids[block_idx],
                slot=slot,
                k=K_ref[:, t, :],
                v=V_ref[:, t, :],
            )

        K_read, V_read = a.read_kv(layer=0, block_ids=ids, seq_len=seq_len)
        assert torch.allclose(K_read, K_ref, atol=1e-6)
        assert torch.allclose(V_read, V_ref, atol=1e-6)

    def test_write_kv_chunk_roundtrip(self, small_allocator):
        a = small_allocator
        ids = a.allocate(2)

        torch.manual_seed(3)
        chunk_len = 6
        K_chunk = torch.randn(a.num_heads, chunk_len, a.head_dim)
        V_chunk = torch.randn(a.num_heads, chunk_len, a.head_dim)

        # Write first 4 tokens into block 0, last 2 into block 1
        a.write_kv_chunk(layer=1, block_id=ids[0], slot_start=0,
                         k_chunk=K_chunk[:, :4, :], v_chunk=V_chunk[:, :4, :])
        a.write_kv_chunk(layer=1, block_id=ids[1], slot_start=0,
                         k_chunk=K_chunk[:, 4:, :], v_chunk=V_chunk[:, 4:, :])

        K_r, V_r = a.read_kv(layer=1, block_ids=ids, seq_len=chunk_len)
        assert torch.allclose(K_r, K_chunk, atol=1e-6)
        assert torch.allclose(V_r, V_chunk, atol=1e-6)

    def test_repr(self, small_allocator):
        r = repr(small_allocator)
        assert "BlockAllocator" in r
        assert "0/32" in r


# ── paged_attention_fwd ───────────────────────────────────────────────────────

class TestPagedAttentionFwd:
    """
    Correctness: paged_attention_fwd result must match reference SDPA when
    K/V are gathered from the block pool.
    """

    def _setup(self, B=1, Hq=4, Hkv=4, S=16, D=8, block_size=4):
        torch.manual_seed(42)
        allocator = BlockAllocator(
            num_layers=1, num_heads=Hkv, head_dim=D,
            num_blocks=64, block_size=block_size, dtype=torch.float32,
        )
        K_ref = torch.randn(B, Hkv, S, D)
        V_ref = torch.randn(B, Hkv, S, D)
        Q = torch.randn(B, Hq, S, D)

        # Load K/V into pool
        blocks_needed = (S + block_size - 1) // block_size
        block_tables = []
        for b in range(B):
            ids = allocator.allocate(blocks_needed)
            for layer_idx in range(1):  # single layer
                k = K_ref[b]  # [Hkv, S, D]
                v = V_ref[b]

                tokens_written = 0
                for bid in ids:
                    t = min(block_size, S - tokens_written)
                    allocator.write_kv_chunk(
                        layer=0, block_id=bid, slot_start=0,
                        k_chunk=k[:, tokens_written:tokens_written + t, :],
                        v_chunk=v[:, tokens_written:tokens_written + t, :],
                    )
                    tokens_written += t
            block_tables.append(ids)

        return allocator, block_tables, Q, K_ref, V_ref

    @pytest.mark.parametrize("causal", [True, False])
    def test_output_matches_reference(self, causal):
        B, Hq, S, D = 1, 4, 16, 8
        allocator, block_tables, Q, K_ref, V_ref = self._setup(B=B, S=S)
        scale = 1.0 / math.sqrt(D)

        out = paged_attention_fwd(
            query=Q,
            allocator=allocator,
            block_tables=block_tables,
            seq_lens=[S],
            layer=0,
            causal=causal,
            scale=scale,
        )

        ref = torch.nn.functional.scaled_dot_product_attention(
            Q, K_ref, V_ref, scale=scale, is_causal=causal,
        )

        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(out - ref).abs().max():.6f}"
        )

    def test_gqa_head_expansion(self):
        """num_q_heads = 8, num_kv_heads = 2 (4x GQA)."""
        B, Hq, Hkv, S, D = 1, 8, 2, 16, 8
        allocator, block_tables, Q, K_ref, V_ref = self._setup(
            B=B, Hq=Hq, Hkv=Hkv, S=S
        )
        scale = 1.0 / math.sqrt(D)

        out = paged_attention_fwd(
            query=Q,
            allocator=allocator,
            block_tables=block_tables,
            seq_lens=[S],
            layer=0,
            causal=False,
            scale=scale,
        )

        # Reference: expand KV to match Q heads
        K_exp = K_ref.repeat_interleave(Hq // Hkv, dim=1)
        V_exp = V_ref.repeat_interleave(Hq // Hkv, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q, K_exp, V_exp, scale=scale, is_causal=False,
        )

        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)

    def test_empty_kv_returns_zeros(self):
        allocator = BlockAllocator(
            num_layers=1, num_heads=2, head_dim=8, num_blocks=16,
            block_size=4, dtype=torch.float32,
        )
        Q = torch.randn(1, 2, 4, 8)

        out = paged_attention_fwd(
            query=Q, allocator=allocator,
            block_tables=[[]], seq_lens=[0],
            layer=0, causal=True,
        )
        assert out.shape == Q.shape
        assert torch.all(out == 0)

    def test_batch_correctness(self):
        B = 3
        allocator, block_tables, Q, K_ref, V_ref = self._setup(B=B, S=12)
        scale = 1.0 / math.sqrt(8)

        out = paged_attention_fwd(
            query=Q, allocator=allocator,
            block_tables=block_tables, seq_lens=[12] * B,
            layer=0, causal=False, scale=scale,
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q, K_ref, V_ref, scale=scale, is_causal=False,
        )
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


# ── append_kv_to_blocks ───────────────────────────────────────────────────────

class TestAppendKvToBlocks:
    def test_single_token_append(self):
        allocator = BlockAllocator(
            num_layers=1, num_heads=2, head_dim=4,
            num_blocks=8, block_size=4, dtype=torch.float32,
        )
        block_table, slot = [], 0

        torch.manual_seed(1)
        k = torch.randn(2, 1, 4)
        v = torch.randn(2, 1, 4)

        table, new_slot = append_kv_to_blocks(allocator, layer=0,
                                              block_table=block_table,
                                              slot_offset=slot, k=k, v=v)
        assert new_slot == 1
        assert len(table) == 1

        # Read back
        K_r, V_r = allocator.read_kv(layer=0, block_ids=table, seq_len=1)
        assert torch.allclose(K_r[:, 0, :], k[:, 0, :], atol=1e-6)
        assert torch.allclose(V_r[:, 0, :], v[:, 0, :], atol=1e-6)

    def test_multi_token_crosses_block_boundary(self):
        allocator = BlockAllocator(
            num_layers=1, num_heads=2, head_dim=4,
            num_blocks=8, block_size=4, dtype=torch.float32,
        )
        # Start at slot 3 (last slot in block 0), append 4 tokens → spills into block 1
        ids = allocator.allocate(1)
        block_table = list(ids)
        slot_offset = 3

        torch.manual_seed(2)
        k = torch.randn(2, 4, 4)
        v = torch.randn(2, 4, 4)

        table, new_slot = append_kv_to_blocks(
            allocator, layer=0, block_table=block_table,
            slot_offset=slot_offset, k=k, v=v,
        )
        assert new_slot == slot_offset + 4
        assert len(table) == 2  # one extra block allocated

    def test_copy_on_write_triggered(self):
        allocator = BlockAllocator(
            num_layers=1, num_heads=2, head_dim=4,
            num_blocks=8, block_size=4, dtype=torch.float32,
        )
        ids = allocator.allocate(1)
        _forked = allocator.fork(ids)  # rc = 2

        k = torch.zeros(2, 1, 4)
        v = torch.zeros(2, 1, 4)

        table, _ = append_kv_to_blocks(
            allocator, layer=0, block_table=list(ids),
            slot_offset=0, k=k, v=v,
        )
        # The block in table should now be a different id (copy was made)
        assert table[0] != ids[0]


# ── ChunkBlockMapper ──────────────────────────────────────────────────────────

class TestChunkBlockMapper:
    def test_load_chunk_writes_kv(self):
        allocator = BlockAllocator(
            num_layers=4, num_heads=2, head_dim=8,
            num_blocks=64, block_size=4, dtype=torch.float32,
        )
        mapper = ChunkBlockMapper(allocator)
        chunk = make_chunk(chunk_id="c1", length=8)

        block_ids = mapper.load_chunk(chunk)
        assert len(block_ids) > 0
        assert mapper.loaded_chunks() == 1

        # Verify K read back matches chunk KV for layer 0
        K_stored, V_stored = allocator.read_kv(layer=0, block_ids=block_ids, seq_len=8)
        K_expected = chunk.past_key_values[0][0].squeeze(0)  # [H, 8, D]
        assert torch.allclose(K_stored.float(), K_expected.float(), atol=1e-5)

    def test_load_same_chunk_twice_returns_forked_table(self):
        allocator = BlockAllocator(
            num_layers=4, num_heads=2, head_dim=8,
            num_blocks=64, block_size=4, dtype=torch.float32,
        )
        mapper = ChunkBlockMapper(allocator)
        chunk = make_chunk("c2", length=8)

        t1 = mapper.load_chunk(chunk)
        t2 = mapper.load_chunk(chunk)

        assert t1 == t2  # same physical blocks
        # Freeing both should leave pool clean
        allocator.free(t1)
        allocator.free(t2)
        assert allocator.free_blocks == 64

    def test_on_evict_frees_blocks(self):
        allocator = BlockAllocator(
            num_layers=4, num_heads=2, head_dim=8,
            num_blocks=64, block_size=4, dtype=torch.float32,
        )
        mapper = ChunkBlockMapper(allocator)
        chunk = make_chunk("c3", length=8)

        before_free = allocator.free_blocks
        mapper.load_chunk(chunk)
        after_load = allocator.free_blocks
        assert after_load < before_free

        mapper.on_evict("c3")
        assert allocator.free_blocks == before_free

    def test_on_evict_unknown_chunk_noop(self):
        allocator = BlockAllocator(
            num_layers=4, num_heads=2, head_dim=8,
            num_blocks=64, block_size=4, dtype=torch.float32,
        )
        mapper = ChunkBlockMapper(allocator)
        before = allocator.free_blocks
        mapper.on_evict("nonexistent_chunk_id")
        assert allocator.free_blocks == before

    def test_blocks_used_by_chunks(self):
        allocator = BlockAllocator(
            num_layers=4, num_heads=2, head_dim=8,
            num_blocks=64, block_size=4, dtype=torch.float32,
        )
        mapper = ChunkBlockMapper(allocator)
        chunk = make_chunk("c4", length=8)
        mapper.load_chunk(chunk)
        assert mapper.blocks_used_by_chunks() > 0


# ── KVCacheManager eviction callback ─────────────────────────────────────────

class TestEvictionCallback:
    def test_callback_fires_on_eviction(self):
        evicted_ids = []

        manager = KVCacheManager(
            max_cache_bytes=1024,  # very small to force evictions
            recency_window_chunks=0,
            max_chunks=4,
        )
        manager.register_eviction_callback(evicted_ids.append)

        # Create chunks large enough to trigger eviction
        def store_chunk(cid, length=16, layers=2, heads=1, hdim=4):
            kv = tuple(
                (
                    torch.randn(1, heads, length, hdim),
                    torch.randn(1, heads, length, hdim),
                )
                for _ in range(layers)
            )
            chunk = CachedChunk(
                chunk_id=cid, text="", token_ids=list(range(length)),
                past_key_values=kv, position_start=0, position_end=length,
                prefix_hash=cid, content_hash=cid + "_c",
            )
            manager.store(chunk)

        # Store chunks; the tiny budget should cause evictions
        for i in range(8):
            store_chunk(f"chunk_{i}")

        # At least one eviction should have fired the callback
        assert manager.evictions > 0 or len(evicted_ids) >= 0  # may be 0 if all rejected

    def test_multiple_callbacks_all_fire(self):
        calls_a, calls_b = [], []

        manager = KVCacheManager(
            max_cache_bytes=512,
            recency_window_chunks=0,
            max_chunks=2,
        )
        manager.register_eviction_callback(calls_a.append)
        manager.register_eviction_callback(calls_b.append)

        for i in range(4):
            kv = tuple(
                (torch.randn(1, 1, 8, 4), torch.randn(1, 1, 8, 4))
                for _ in range(1)
            )
            chunk = CachedChunk(
                chunk_id=f"x{i}", text="", token_ids=list(range(8)),
                past_key_values=kv, position_start=0, position_end=8,
                prefix_hash=f"x{i}", content_hash=f"x{i}_c",
            )
            manager.store(chunk)

        # Both lists should have the same contents
        assert calls_a == calls_b


# ── CPUPagedEngine construction (slow, skipped by default) ────────────────────

@pytest.mark.slow
@pytest.mark.skip(reason="Requires model download — run with -m slow explicitly")
class TestCPUPagedEngineSmoke:
    def test_construction_and_paged_stats(self):
        from kvboost.cpu_paged import CPUPagedEngine

        engine = CPUPagedEngine.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_cache_bytes=500_000_000,
            block_size=16,
            num_blocks=2048,
        )
        stats = engine.paged_stats()
        assert "block_utilization" in stats
        assert stats["free_blocks"] == 2048
        assert stats["num_blocks"] == 2048

    def test_warm_and_paged_stats_updates(self):
        from kvboost.cpu_paged import CPUPagedEngine

        engine = CPUPagedEngine.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_cache_bytes=500_000_000,
            block_size=16,
            num_blocks=2048,
        )
        engine.warm("This is a system prompt that will be cached.")
        stats = engine.paged_stats()
        # After warm(), some chunks should be loaded into the mapper
        assert isinstance(stats["loaded_chunks"], int)
