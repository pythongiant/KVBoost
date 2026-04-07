"""
Batch Inference
===============
Processes multiple prompts sharing a common prefix in a single batched
forward pass. The shared prefix KV is loaded from cache once, then
broadcast (zero-copy via expand()) across the batch for the suffix prefill.

Key operations:
  - find_common_chunk_prefix: detect shared prefix across prompts
  - broadcast_kv: expand [1, H, S, D] → [B, H, S, D] without copying
  - group_by_prefix: auto-cluster prompts by shared prefix hash
  - batched_decode: decode loop for multiple sequences in parallel
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .models import PastKVType, content_hash_from_tokens

log = logging.getLogger(__name__)


def find_common_chunk_prefix(
    all_token_ids: List[List[int]], chunk_size: int
) -> int:
    """
    Returns length of the longest chunk-aligned prefix shared by all prompts.
    Stops at the first chunk where any prompt diverges.
    """
    if not all_token_ids or len(all_token_ids) < 2:
        return len(all_token_ids[0]) if all_token_ids else 0

    min_len = min(len(ids) for ids in all_token_ids)
    common = 0

    for i in range(0, min_len, chunk_size):
        chunk_end = i + chunk_size
        if chunk_end > min_len:
            break
        reference = all_token_ids[0][i:chunk_end]
        if all(ids[i:chunk_end] == reference for ids in all_token_ids[1:]):
            common = chunk_end
        else:
            break

    return common


def broadcast_kv(kv: PastKVType, batch_size: int) -> PastKVType:
    """
    Expand cached KV from [1, heads, seq, dim] to [batch, heads, seq, dim].
    Uses expand() — zero-copy, shares underlying storage.
    """
    return tuple(
        (k.expand(batch_size, -1, -1, -1),
         v.expand(batch_size, -1, -1, -1))
        for k, v in kv
    )


def pad_and_mask(
    suffix_ids_list: List[List[int]], pad_token_id: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Pad suffix token lists to equal length. Returns (padded_ids, attention_masks).
    Masks are 1 for real tokens, 0 for padding.
    """
    max_len = max(len(s) for s in suffix_ids_list)
    padded = []
    masks = []
    for ids in suffix_ids_list:
        pad_len = max_len - len(ids)
        padded.append(ids + [pad_token_id] * pad_len)
        masks.append([1] * len(ids) + [0] * pad_len)
    return padded, masks


def group_by_prefix(
    prompts: List[str],
    token_ids_list: List[List[int]],
    chunk_size: int,
    n_prefix_chunks: int = 3,
) -> Dict[str, List[int]]:
    """
    Group prompt indices by the hash of their first N chunks.
    Prompts sharing the same prefix chunks are batched together.
    Returns: prefix_key → [prompt_indices].
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, ids in enumerate(token_ids_list):
        # Hash the first N chunk-aligned blocks
        end = min(len(ids), n_prefix_chunks * chunk_size)
        prefix = ids[:end]
        key = content_hash_from_tokens(prefix) if prefix else "empty"
        groups[key].append(i)
    return dict(groups)


def batched_decode(
    model,
    past_kv: PastKVType,
    first_tokens: torch.Tensor,
    start_pos: int,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    do_sample: bool = False,
    device: str = "cpu",
) -> Tuple[List[List[int]], PastKVType]:
    """
    Batched autoregressive decode loop.

    Args:
        model: HuggingFace CausalLM.
        past_kv: KV cache from prefill, shape [B, H, S, D].
        first_tokens: [B] tensor of first generated tokens.
        start_pos: position ID for the next token.
        max_new_tokens: max tokens to generate per sequence.
        eos_token_id: stop token.
        temperature: sampling temperature.
        do_sample: greedy vs sampling.
        device: target device.

    Returns:
        (generated_ids, final_kv) where generated_ids[i] is the token
        list for batch element i.
    """
    from .engine import InferenceEngine  # avoid circular import

    batch_size = first_tokens.shape[0]
    generated = [[t.item()] for t in first_tokens]
    finished = [False] * batch_size
    cur_ids = first_tokens.unsqueeze(1)  # [B, 1]
    cur_pos = start_pos

    for _ in range(max_new_tokens - 1):
        if all(finished):
            break

        pos_ids = torch.full(
            (batch_size, 1), cur_pos, dtype=torch.long, device=device
        )

        with torch.no_grad():
            out = model(
                input_ids=cur_ids,
                past_key_values=InferenceEngine._as_cache(past_kv),
                position_ids=pos_ids,
                use_cache=True,
            )

        past_kv = InferenceEngine._normalize_past_kv(out.past_key_values)
        logits = out.logits[:, -1, :]  # [B, vocab]

        # Sample or argmax per batch element
        next_tokens = []
        for b in range(batch_size):
            if finished[b]:
                next_tokens.append(eos_token_id)
            else:
                tok = InferenceEngine._sample(logits[b:b+1], temperature, do_sample)
                generated[b].append(tok)
                if tok == eos_token_id:
                    finished[b] = True
                next_tokens.append(tok)

        cur_ids = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
        cur_pos += 1

    return generated, past_kv
