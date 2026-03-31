"""
PromptAssembler
===============
Given a full prompt (token_ids) and the KVCacheManager, assembles:

  cached_past_kv    — merged KV tensors for the reused prefix/chunks
  live_token_ids    — tokens still needing a forward pass
  live_position_ids — absolute positions for each live token
  chunk_boundaries  — seam positions for SelectiveRecompute

Two assembly modes
------------------
PREFIX_ONLY  : only the leading contiguous cached prefix is reused
               (safe, correct, no seam issues)
CHUNK_REUSE  : any matching chunk, anywhere in the prompt, is reused
               (faster, seams exist → feed to SelectiveRecompute)
"""

from __future__ import annotations

import enum
import logging
from typing import List, Optional, Tuple

import torch

from .models import AssembledPrompt, PastKVType
from .cache_manager import KVCacheManager
from .chunk_registry import ChunkRegistry

log = logging.getLogger(__name__)


class AssemblyMode(str, enum.Enum):
    PREFIX_ONLY = "prefix_only"
    CHUNK_REUSE = "chunk_reuse"


class PromptAssembler:
    def __init__(
        self,
        cache_manager: KVCacheManager,
        chunk_registry: ChunkRegistry,
        mode: AssemblyMode = AssemblyMode.CHUNK_REUSE,
    ):
        self.cache = cache_manager
        self.registry = chunk_registry
        self.mode = mode

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def assemble(self, token_ids: List[int]) -> AssembledPrompt:
        if self.mode == AssemblyMode.PREFIX_ONLY:
            return self._prefix_only(token_ids)
        return self._chunk_reuse(token_ids)

    # ------------------------------------------------------------------
    # Mode: prefix only (exact leading match)
    # ------------------------------------------------------------------

    def _prefix_only(self, token_ids: List[int]) -> AssembledPrompt:
        merged_kv, covered = self.cache.build_prefix_kv(
            token_ids, self.registry.chunk_size
        )
        live_ids = token_ids[covered:]
        live_pos = list(range(covered, len(token_ids)))
        hit_ratio = covered / max(len(token_ids), 1)

        # Boundaries: one boundary at position 0 → covered (if any cached)
        boundaries: List[Tuple[int, int]] = []
        if covered > 0:
            boundaries = [(0, covered)]

        return AssembledPrompt(
            full_token_ids=token_ids,
            cached_past_kv=merged_kv,
            cached_length=covered,
            live_token_ids=live_ids,
            live_position_ids=live_pos,
            chunk_boundaries=boundaries,
            cache_hit_ratio=hit_ratio,
        )

    # ------------------------------------------------------------------
    # Mode: chunk reuse (any matching chunk)
    # ------------------------------------------------------------------

    def _chunk_reuse(self, token_ids: List[int]) -> AssembledPrompt:
        chunk_size = self.registry.chunk_size
        matching = self.cache.find_matching_chunks(token_ids, chunk_size)

        if not matching:
            return AssembledPrompt(
                full_token_ids=token_ids,
                cached_past_kv=None,
                cached_length=0,
                live_token_ids=token_ids,
                live_position_ids=list(range(len(token_ids))),
                chunk_boundaries=[],
                cache_hit_ratio=0.0,
            )

        # Build coverage map: which token positions are covered by a cache hit
        # We assemble chunks in order, skip overlapping regions
        covered_end = 0
        kv_parts: List[PastKVType] = []
        boundaries: List[Tuple[int, int]] = []
        total_cached = 0

        # Sort by start position
        matching.sort(key=lambda x: x[0])

        # The strategy: collect contiguous or gap-free cached regions
        # from the start of the prompt (prefix style, but across registered chunks)
        for start_pos, chunk in matching:
            end_pos = start_pos + chunk.length

            if start_pos != covered_end:
                # There's a gap — stop extending cache prefix
                # (gaps would require re-encoding the gap tokens into
                # the attention stream, which breaks prefix semantics)
                break

            # Move chunk KV to inference device
            kv = KVCacheManager._move_kv(chunk.past_key_values, self.cache.device)
            kv_parts.append(kv)
            boundaries.append((start_pos, end_pos))
            covered_end = end_pos
            total_cached += chunk.length

        if not kv_parts:
            return AssembledPrompt(
                full_token_ids=token_ids,
                cached_past_kv=None,
                cached_length=0,
                live_token_ids=token_ids,
                live_position_ids=list(range(len(token_ids))),
                chunk_boundaries=[],
                cache_hit_ratio=0.0,
            )

        merged_kv = KVCacheManager.merge_kv_list(kv_parts)
        live_ids = token_ids[covered_end:]
        live_pos = list(range(covered_end, len(token_ids)))
        hit_ratio = total_cached / max(len(token_ids), 1)

        log.debug(
            "Assembled: %d cached tokens, %d live tokens (%.0f%% hit)",
            total_cached, len(live_ids), hit_ratio * 100,
        )

        return AssembledPrompt(
            full_token_ids=token_ids,
            cached_past_kv=merged_kv,
            cached_length=covered_end,
            live_token_ids=live_ids,
            live_position_ids=live_pos,
            chunk_boundaries=boundaries,
            cache_hit_ratio=hit_ratio,
        )

    # ------------------------------------------------------------------
    # Utility: build position_ids tensor for live tokens
    # ------------------------------------------------------------------

    @staticmethod
    def make_position_ids(
        live_positions: List[int], device: str
    ) -> torch.Tensor:
        return torch.tensor(live_positions, dtype=torch.long, device=device).unsqueeze(0)

    @staticmethod
    def make_attention_mask(
        cached_len: int, live_len: int, device: str
    ) -> torch.Tensor:
        """
        Causal attention mask for a prompt with cached prefix.
        Shape: [1, live_len, cached_len + live_len]
        All live tokens attend to full cached prefix + their causal prefix.
        """
        total = cached_len + live_len
        mask = torch.ones(1, live_len, total, dtype=torch.bool, device=device)
        # Apply causal mask for the live-to-live portion
        for i in range(live_len):
            mask[0, i, cached_len + i + 1 :] = False
        return mask
