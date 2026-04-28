"""
Bridge between KVBoost CachedChunks and the BlockAllocator physical block pool.

When KVBoost's cache_manager returns a CachedChunk its past_key_values are
stored as standard HuggingFace tensors:

    past_key_values[layer] = (K, V)
    K, V shape: [1, num_heads, chunk_len, head_dim]

ChunkBlockMapper writes these tensors into the physical block pool and maintains
a mapping chunk_id → block_ids so blocks can be freed when a chunk is evicted.

Prefix sharing
--------------
If two assembled prompts share a chunk (same prefix_hash), their block_tables
point to the same physical blocks via BlockAllocator.fork().  Any write during
the decode phase triggers copy-on-write via BlockAllocator.ensure_writable().

Integration with KVCacheManager
--------------------------------
Register `mapper.on_evict` as an eviction callback on KVCacheManager so blocks
are freed whenever a chunk leaves the hot tier.  This keeps the block pool and
the KV cache in sync without a GC pass.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import torch

from ..models import AssembledPrompt, CachedChunk, PastKVType
from .block_allocator import BlockAllocator

log = logging.getLogger(__name__)


class ChunkBlockMapper:
    """
    Manages the lifecycle of physical blocks corresponding to CachedChunks.

    Parameters
    ----------
    allocator : BlockAllocator to write into / free from
    """

    def __init__(self, allocator: BlockAllocator) -> None:
        self.allocator = allocator
        # chunk_id (prefix_hash) → list of block_ids (per-layer identical table)
        self._chunk_blocks: Dict[str, List[int]] = {}
        self._lock = threading.Lock()

    # ── Load a chunk into the block pool ─────────────────────────────────────

    def load_chunk(self, chunk: CachedChunk) -> List[int]:
        """
        Write *chunk*'s K/V tensors into physical blocks and return the
        block_table (same table for all layers — blocks are indexed uniformly).

        If the chunk is already loaded, return the existing block_table
        (increments ref count via fork).
        """
        with self._lock:
            if chunk.chunk_id in self._chunk_blocks:
                existing = self._chunk_blocks[chunk.chunk_id]
                # Fork: share blocks, increment ref counts
                return self.allocator.fork(existing)

        chunk_len = chunk.length
        bs = self.allocator.block_size
        num_blocks_needed = (chunk_len + bs - 1) // bs

        block_ids = self.allocator.allocate(num_blocks_needed)

        # Write each layer's K/V into the pool
        for layer_idx, (K, V) in enumerate(chunk.past_key_values):
            # K, V: [1, num_heads, chunk_len, head_dim]
            K = K.squeeze(0).to(self.allocator.dtype)  # [H, chunk_len, D]
            V = V.squeeze(0).to(self.allocator.dtype)

            tokens_written = 0
            for blk_pos, block_id in enumerate(block_ids):
                slot_start = 0  # always write from slot 0 within a fresh block
                tokens_this_block = min(bs, chunk_len - tokens_written)
                k_slice = K[:, tokens_written:tokens_written + tokens_this_block, :]
                v_slice = V[:, tokens_written:tokens_written + tokens_this_block, :]
                self.allocator.write_kv_chunk(
                    layer=layer_idx,
                    block_id=block_id,
                    slot_start=slot_start,
                    k_chunk=k_slice,
                    v_chunk=v_slice,
                )
                tokens_written += tokens_this_block

        with self._lock:
            self._chunk_blocks[chunk.chunk_id] = block_ids

        return list(block_ids)

    # ── Build a block table from an AssembledPrompt ───────────────────────────

    def build_block_table(
        self, assembled: AssembledPrompt, cached_chunks: List[CachedChunk]
    ) -> Tuple[List[int], int]:
        """
        Given the list of CachedChunks that make up *assembled.cached_past_kv*,
        load each into the pool (or reuse if already loaded) and concatenate
        their block_tables into one sequence-level table.

        Returns
        -------
        (block_table, cached_length)
            block_table   : physical block_ids covering [0, cached_length)
            cached_length : number of tokens in the block table
        """
        full_block_table: List[int] = []
        for chunk in cached_chunks:
            chunk_blocks = self.load_chunk(chunk)
            full_block_table.extend(chunk_blocks)
        return full_block_table, assembled.cached_length

    # ── Eviction callback ─────────────────────────────────────────────────────

    def on_evict(self, chunk_id: str) -> None:
        """
        Called by KVCacheManager when a chunk is evicted.
        Frees the associated physical blocks back to the allocator.
        """
        with self._lock:
            block_ids = self._chunk_blocks.pop(chunk_id, None)
        if block_ids is not None:
            self.allocator.free(block_ids)
            log.debug("ChunkBlockMapper: freed %d blocks for chunk %s",
                      len(block_ids), chunk_id[:8])

    # ── Utility ───────────────────────────────────────────────────────────────

    def loaded_chunks(self) -> int:
        return len(self._chunk_blocks)

    def blocks_used_by_chunks(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._chunk_blocks.values())
