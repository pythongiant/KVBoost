"""
Physical block allocator for CPU paged attention.

Layout
------
The entire KV pool for *one layer* lives in a single pre-allocated tensor:

    kv_pool : [num_blocks, 2, num_heads, block_size, head_dim]
               ^           ^  ^          ^            ^
               block_id   K/V heads     tokens/block  features

A sequence's KV is described by a block_table (list of physical block_ids).
Logical token i lives in:
    block_id  = block_table[i // block_size]
    slot      = i % block_size

All layers share the same block_table for a sequence; each layer has its own
pool tensor of identical shape.

Copy-on-write prefix sharing
-----------------------------
When two sequences share a KVBoost prefix chunk (same prefix_hash) they point
to the same physical blocks for that prefix.  Before any write to a block we
check ref_count; if > 1 we copy the block to a fresh allocation first.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

import torch


class BlockAllocator:
    """
    Manages a fixed pool of physical KV blocks across all layers.

    Parameters
    ----------
    num_layers  : number of transformer layers
    num_heads   : number of KV heads (may differ from Q heads for GQA)
    head_dim    : dimension per head
    num_blocks  : total blocks in the pool
    block_size  : tokens per block (default 16)
    dtype       : storage dtype (default torch.float16)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_blocks: int = 4096,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype

        # Physical KV pool: one tensor per layer
        # Shape: [num_blocks, 2, num_heads, block_size, head_dim]
        self.pools: List[torch.Tensor] = [
            torch.zeros(
                num_blocks, 2, num_heads, block_size, head_dim,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        # Free list: stack of available block ids
        self._free: List[int] = list(range(num_blocks))

        # Reference counts: how many sequences point to each block
        self._ref_count: Dict[int, int] = {}

        self._lock = threading.Lock()

    # ── Allocation ────────────────────────────────────────────────────────────

    def allocate(self, n: int) -> List[int]:
        """Allocate *n* fresh physical blocks. Raises RuntimeError if OOM."""
        with self._lock:
            if len(self._free) < n:
                raise RuntimeError(
                    f"BlockAllocator OOM: requested {n} blocks but only "
                    f"{len(self._free)} free (pool size {self.num_blocks})."
                )
            block_ids = [self._free.pop() for _ in range(n)]
            for bid in block_ids:
                self._ref_count[bid] = 1
            return block_ids

    def free(self, block_ids: List[int]) -> None:
        """Decrement ref counts and return blocks to the free list when rc==0."""
        with self._lock:
            for bid in block_ids:
                rc = self._ref_count.get(bid, 0)
                if rc <= 1:
                    self._ref_count.pop(bid, None)
                    self._free.append(bid)
                else:
                    self._ref_count[bid] = rc - 1

    def fork(self, block_ids: List[int]) -> List[int]:
        """
        Copy-on-write fork: increment ref counts for shared blocks.
        The caller gets a new block_table that logically points to the same
        data.  Any write must call ensure_writable() first.
        """
        with self._lock:
            for bid in block_ids:
                self._ref_count[bid] = self._ref_count.get(bid, 1) + 1
        return list(block_ids)  # same ids, higher ref count

    def ensure_writable(self, block_id: int) -> int:
        """
        If block_id has ref_count > 1, allocate a fresh block, copy the data,
        decrement the original's ref count and return the new block id.
        Otherwise return block_id unchanged.
        """
        with self._lock:
            rc = self._ref_count.get(block_id, 1)
            if rc <= 1:
                return block_id
            # Need to copy
            if not self._free:
                raise RuntimeError("BlockAllocator OOM during copy-on-write.")
            new_id = self._free.pop()
            self._ref_count[new_id] = 1
            self._ref_count[block_id] = rc - 1

        # Copy all layers (outside lock for performance)
        for pool in self.pools:
            pool[new_id].copy_(pool[block_id])

        return new_id

    # ── Block I/O helpers ─────────────────────────────────────────────────────

    def write_kv(
        self,
        layer: int,
        block_id: int,
        slot: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Write a single token's K and V into (layer, block_id, slot).

        k, v: [num_heads, head_dim]
        """
        self.pools[layer][block_id, 0, :, slot, :] = k
        self.pools[layer][block_id, 1, :, slot, :] = v

    def write_kv_chunk(
        self,
        layer: int,
        block_id: int,
        slot_start: int,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
    ) -> None:
        """
        Write multiple consecutive tokens starting at slot_start.

        k_chunk, v_chunk: [num_heads, chunk_len, head_dim]
        """
        chunk_len = k_chunk.size(1)
        self.pools[layer][block_id, 0, :, slot_start:slot_start + chunk_len, :] = k_chunk
        self.pools[layer][block_id, 1, :, slot_start:slot_start + chunk_len, :] = v_chunk

    def read_kv(
        self,
        layer: int,
        block_ids: List[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K and V tensors for *seq_len* tokens described by block_ids.

        Returns:
            K: [num_heads, seq_len, head_dim]
            V: [num_heads, seq_len, head_dim]
        """
        pool = self.pools[layer]
        bs = self.block_size
        num_blocks = len(block_ids)

        # Gather all blocks then slice to seq_len
        # Shape after stack: [num_blocks, 2, num_heads, block_size, head_dim]
        gathered = pool[block_ids]  # index_select via advanced indexing

        # Reshape to [2, num_heads, num_blocks*block_size, head_dim]
        k_full = gathered[:, 0].permute(1, 0, 2, 3).reshape(
            self.num_heads, num_blocks * bs, self.head_dim
        )
        v_full = gathered[:, 1].permute(1, 0, 2, 3).reshape(
            self.num_heads, num_blocks * bs, self.head_dim
        )
        return k_full[:, :seq_len, :], v_full[:, :seq_len, :]

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def utilization(self) -> float:
        return self.used_blocks / self.num_blocks

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, block_size={self.block_size}, "
            f"blocks={self.used_blocks}/{self.num_blocks}, "
            f"util={self.utilization():.1%})"
        )
