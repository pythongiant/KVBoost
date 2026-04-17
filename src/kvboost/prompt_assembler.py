"""
PromptAssembler
===============
Given a full prompt (token_ids) and the KVCacheManager, assembles:

  cached_past_kv    — merged KV tensors for the reused prefix/chunks
  live_token_ids    — tokens still needing a forward pass
  live_position_ids — absolute positions for each live token
  chunk_boundaries  — seam positions for SelectiveRecompute
  has_approximate   — True if any chunk was an approximate (content-only) match

Two assembly modes
------------------
PREFIX_ONLY  : only the leading contiguous cached prefix is reused
               (safe, correct, no seam issues)
CHUNK_REUSE  : any matching chunk, anywhere in the prompt, is reused
               (faster, seams exist → feed to recompute strategy)
"""

from __future__ import annotations

import enum
import logging
from typing import List, Optional, Tuple

import torch

from .models import AssembledPrompt, PastKVType
from .cache_manager import KVCacheManager, ChunkMatch
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

    def assemble(
        self,
        token_ids: List[int],
        chunk_splits: Optional[List[Tuple[int, int, List[int]]]] = None,
    ) -> AssembledPrompt:
        if self.mode == AssemblyMode.PREFIX_ONLY:
            return self._prefix_only(token_ids, chunk_splits=chunk_splits)
        return self._chunk_reuse(token_ids, chunk_splits=chunk_splits)

    # ------------------------------------------------------------------
    # Mode: prefix only (exact leading match)
    # ------------------------------------------------------------------

    def _prefix_only(
        self,
        token_ids: List[int],
        chunk_splits: Optional[List[Tuple[int, int, List[int]]]] = None,
    ) -> AssembledPrompt:
        merged_kv, covered = self.cache.build_prefix_kv(
            token_ids, self.registry.chunk_size, chunk_splits=chunk_splits,
        )
        live_ids = token_ids[covered:]
        live_pos = list(range(covered, len(token_ids)))
        hit_ratio = covered / max(len(token_ids), 1)

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
            has_approximate=False,  # prefix mode only uses exact matches
        )

    # ------------------------------------------------------------------
    # Mode: chunk reuse (any matching chunk, two-tier keying)
    # ------------------------------------------------------------------

    def _chunk_reuse(
        self,
        token_ids: List[int],
        chunk_splits: Optional[List[Tuple[int, int, List[int]]]] = None,
    ) -> AssembledPrompt:
        chunk_size = self.registry.chunk_size
        matching = self.cache.find_matching_chunks(
            token_ids, chunk_size, chunk_splits=chunk_splits,
        )

        if not matching:
            return AssembledPrompt(
                full_token_ids=token_ids,
                cached_past_kv=None,
                cached_length=0,
                live_token_ids=token_ids,
                live_position_ids=list(range(len(token_ids))),
                chunk_boundaries=[],
                cache_hit_ratio=0.0,
                has_approximate=False,
            )

        covered_end = 0
        kv_parts: List[PastKVType] = []
        boundaries: List[Tuple[int, int]] = []
        total_cached = 0
        any_approximate = False

        matching.sort(key=lambda x: x[0])

        for start_pos, chunk_match in matching:
            chunk = chunk_match.chunk
            end_pos = start_pos + chunk.length

            if start_pos != covered_end:
                break

            kv = KVCacheManager._move_kv(chunk.past_key_values, self.cache.device)
            kv_parts.append(kv)
            boundaries.append((start_pos, end_pos))
            covered_end = end_pos
            total_cached += chunk.length

            if chunk_match.approximate:
                any_approximate = True

        if not kv_parts:
            return AssembledPrompt(
                full_token_ids=token_ids,
                cached_past_kv=None,
                cached_length=0,
                live_token_ids=token_ids,
                live_position_ids=list(range(len(token_ids))),
                chunk_boundaries=[],
                cache_hit_ratio=0.0,
                has_approximate=False,
            )

        merged_kv = KVCacheManager.merge_kv_list(kv_parts)
        live_ids = token_ids[covered_end:]
        live_pos = list(range(covered_end, len(token_ids)))
        hit_ratio = total_cached / max(len(token_ids), 1)

        log.debug(
            "Assembled: %d cached tokens, %d live tokens (%.0f%% hit, approximate=%s)",
            total_cached, len(live_ids), hit_ratio * 100, any_approximate,
        )

        return AssembledPrompt(
            full_token_ids=token_ids,
            cached_past_kv=merged_kv,
            cached_length=covered_end,
            live_token_ids=live_ids,
            live_position_ids=live_pos,
            chunk_boundaries=boundaries,
            cache_hit_ratio=hit_ratio,
            has_approximate=any_approximate,
        )

    # ------------------------------------------------------------------
    # Utility
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
        total = cached_len + live_len
        mask = torch.ones(1, live_len, total, dtype=torch.bool, device=device)
        for i in range(live_len):
            mask[0, i, cached_len + i + 1 :] = False
        return mask
