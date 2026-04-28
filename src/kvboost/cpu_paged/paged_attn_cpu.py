"""
CPU paged attention forward pass.

Implements attention where K and V are stored in a block pool managed by
BlockAllocator rather than in contiguous tensors.  Optimised for CPU via
index_select + batched matmul — no custom C extension required.

Two modes
---------
Prefill (query_len > 1):
    Computes full self-attention over [query_len, kv_len] using gathered K/V.
    Used when processing uncached or live tokens.

Decode (query_len == 1):
    Computes attention for a single new token over the full KV history.
    This is the tight inner loop during autoregressive generation.

All shapes follow HuggingFace convention:
    Q, K, V, O : [batch, num_heads, seq, head_dim]
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .block_allocator import BlockAllocator


def paged_attention_fwd(
    query: torch.Tensor,
    allocator: BlockAllocator,
    block_tables: List[List[int]],
    seq_lens: List[int],
    layer: int,
    causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Paged attention forward for one transformer layer.

    Parameters
    ----------
    query       : [batch, num_q_heads, query_len, head_dim]
    allocator   : BlockAllocator holding the physical KV pool
    block_tables: list of block_id lists, one per batch element
    seq_lens    : KV sequence length for each batch element (before this query)
    layer       : which layer's pool to read from
    causal      : apply causal mask during prefill
    scale       : attention scale; defaults to 1/sqrt(head_dim)

    Returns
    -------
    output : [batch, num_q_heads, query_len, head_dim]
    """
    B, Hq, Sq, D = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # For GQA: num_kv_heads may differ from num_q_heads
    Hkv = allocator.num_heads
    assert Hq % Hkv == 0, f"num_q_heads ({Hq}) must be divisible by num_kv_heads ({Hkv})"
    groups = Hq // Hkv  # heads per KV group

    outputs = []

    for b in range(B):
        kv_len = seq_lens[b]
        table = block_tables[b]

        if kv_len == 0:
            outputs.append(torch.zeros(Hq, Sq, D, dtype=query.dtype))
            continue

        # Gather K and V from the block pool for this sequence
        # K, V: [Hkv, kv_len, D]
        K, V = allocator.read_kv(layer, table, kv_len)
        K = K.to(query.dtype)
        V = V.to(query.dtype)

        # Expand KV heads to match Q heads (GQA broadcast)
        if groups > 1:
            # [Hkv, kv_len, D] → [Hq, kv_len, D]
            K = K.repeat_interleave(groups, dim=0)
            V = V.repeat_interleave(groups, dim=0)

        # Q for this batch element: [Hq, Sq, D]
        q = query[b]  # [Hq, Sq, D]

        # Scaled dot-product: [Hq, Sq, kv_len]
        # Use F.scaled_dot_product_attention — handles causal mask + memory-efficient path
        # q: [1, Hq, Sq, D]  K/V: [1, Hq, kv_len, D]
        q_4d = q.unsqueeze(0)
        K_4d = K.unsqueeze(0)
        V_4d = V.unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q_4d, K_4d, V_4d,
            scale=scale,
            is_causal=(causal and Sq > 1),  # causal only applies during prefill
        )  # [1, Hq, Sq, D]

        outputs.append(out.squeeze(0))  # [Hq, Sq, D]

    return torch.stack(outputs, dim=0)  # [B, Hq, Sq, D]


def append_kv_to_blocks(
    allocator: BlockAllocator,
    layer: int,
    block_table: List[int],
    slot_offset: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[List[int], int]:
    """
    Append K/V for one or more new tokens into the block pool.

    Allocates new blocks as needed.  Returns the (possibly extended)
    block_table and the new slot_offset.

    Parameters
    ----------
    allocator    : BlockAllocator
    layer        : layer index
    block_table  : current block_table (mutated in place conceptually)
    slot_offset  : absolute token index of the first token to append
    k            : [num_kv_heads, num_tokens, head_dim]
    v            : [num_kv_heads, num_tokens, head_dim]

    Returns
    -------
    (updated_block_table, new_slot_offset)
    """
    block_table = list(block_table)
    bs = allocator.block_size
    num_tokens = k.size(1)

    for t in range(num_tokens):
        slot = slot_offset + t
        block_idx = slot // bs
        slot_in_block = slot % bs

        # Allocate a new block if needed
        if block_idx >= len(block_table):
            new_blocks = allocator.allocate(1)
            block_table.extend(new_blocks)

        block_id = block_table[block_idx]
        # Copy-on-write if shared
        block_id = allocator.ensure_writable(block_id)
        block_table[block_idx] = block_id

        allocator.write_kv(
            layer=layer,
            block_id=block_id,
            slot=slot_in_block,
            k=k[:, t, :],
            v=v[:, t, :],
        )

    return block_table, slot_offset + num_tokens
