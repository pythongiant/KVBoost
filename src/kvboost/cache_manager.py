"""
KVCacheManager
==============
Two-tier storage:
  Tier 1 (hot)  : in-process dict, tensors on CPU RAM
  Tier 2 (cold) : mmap'd files on disk via torch.save / torch.load
                  (optional, disabled by default)

Eviction policy: LRU by access_count + age.
"""

from __future__ import annotations

import os
import time
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .models import CachedChunk, PastKVType, chunk_id_from_tokens

log = logging.getLogger(__name__)


class KVCacheManager:
    def __init__(
        self,
        max_chunks: int = 64,
        disk_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.max_chunks = max_chunks
        self.device = device
        self._hot: OrderedDict[str, CachedChunk] = OrderedDict()

        # Optional disk tier
        self._disk_dir: Optional[Path] = None
        if disk_dir:
            self._disk_dir = Path(disk_dir)
            self._disk_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, chunk: CachedChunk) -> None:
        """Store a chunk. Evicts LRU entry if over capacity."""
        if chunk.chunk_id in self._hot:
            # Refresh order
            self._hot.move_to_end(chunk.chunk_id)
            return

        # Move KV tensors to storage device
        chunk.past_key_values = self._move_kv(chunk.past_key_values, self.device)

        if len(self._hot) >= self.max_chunks:
            self._evict_lru()

        self._hot[chunk.chunk_id] = chunk
        log.debug("Stored chunk %s (%.2fMB)", chunk.chunk_id[:8], chunk.memory_bytes() / 1e6)

    def get(self, chunk_id: str) -> Optional[CachedChunk]:
        """Retrieve a chunk by id, promotes to hot if found on disk."""
        if chunk_id in self._hot:
            self.hits += 1
            chunk = self._hot[chunk_id]
            chunk.touch()
            self._hot.move_to_end(chunk_id)
            return chunk

        # Try disk tier
        if self._disk_dir:
            path = self._disk_dir / f"{chunk_id}.pt"
            if path.exists():
                self.hits += 1
                chunk = self._load_from_disk(path)
                self.store(chunk)  # promote to hot
                return chunk

        self.misses += 1
        return None

    def get_or_none(self, token_ids: List[int]) -> Optional[CachedChunk]:
        """Look up by token sequence (hashes internally)."""
        cid = chunk_id_from_tokens(token_ids)
        return self.get(cid)

    def build_prefix_kv(
        self, token_ids: List[int], chunk_size: int
    ) -> Tuple[Optional[PastKVType], int]:
        """
        Greedily assemble the longest cached prefix from non-overlapping
        fixed-size chunks of `token_ids`.

        Returns:
            merged_kv  : concatenated past_key_values (or None)
            covered_len: how many tokens are covered
        """
        chunks: List[CachedChunk] = []
        pos = 0

        while pos + chunk_size <= len(token_ids):
            slice_ids = token_ids[pos : pos + chunk_size]
            chunk = self.get_or_none(slice_ids)
            if chunk is None:
                break  # stop at first miss to keep prefix contiguous
            chunks.append(chunk)
            pos += chunk_size

        if not chunks:
            return None, 0

        merged = self.merge_kv_list([c.past_key_values for c in chunks])
        return merged, pos

    def find_matching_chunks(
        self, token_ids: List[int], chunk_size: int
    ) -> List[Tuple[int, CachedChunk]]:
        """
        Non-greedy scan: finds ALL cached chunks anywhere in token_ids.
        Returns list of (start_pos, chunk) pairs in order.
        Used by PromptAssembler for non-prefix reuse.
        """
        results = []
        step = chunk_size
        for start in range(0, len(token_ids) - chunk_size + 1, step):
            slice_ids = token_ids[start : start + chunk_size]
            chunk = self.get_or_none(slice_ids)
            if chunk is not None:
                results.append((start, chunk))
        return results

    def invalidate(self, chunk_id: str) -> None:
        self._hot.pop(chunk_id, None)
        if self._disk_dir:
            path = self._disk_dir / f"{chunk_id}.pt"
            path.unlink(missing_ok=True)

    def stats(self) -> Dict:
        total = self.hits + self.misses
        hot_mb = sum(c.memory_bytes() for c in self._hot.values()) / 1e6
        return {
            "hot_chunks": len(self._hot),
            "hot_memory_mb": round(hot_mb, 2),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": round(self.hits / max(total, 1), 3),
        }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def merge_kv_list(kv_list: List[PastKVType]) -> PastKVType:
        """
        Concatenate a list of past_key_values along the sequence dimension.
        All elements must have the same num_layers / heads / head_dim.
        """
        if not kv_list:
            raise ValueError("kv_list is empty")
        if len(kv_list) == 1:
            return kv_list[0]

        num_layers = len(kv_list[0])
        merged = []
        for layer_idx in range(num_layers):
            keys = torch.cat([kv[layer_idx][0] for kv in kv_list], dim=2)
            vals = torch.cat([kv[layer_idx][1] for kv in kv_list], dim=2)
            merged.append((keys, vals))
        return tuple(merged)

    @staticmethod
    def slice_kv(
        kv: PastKVType, start: int, end: int
    ) -> PastKVType:
        """Slice KV cache along sequence dimension."""
        return tuple(
            (layer[0][:, :, start:end, :], layer[1][:, :, start:end, :])
            for layer in kv
        )

    @staticmethod
    def kv_seq_len(kv: PastKVType) -> int:
        return kv[0][0].shape[2]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        if not self._hot:
            return
        # Pop least-recently-used (first item in OrderedDict)
        lru_id, lru_chunk = next(iter(self._hot.items()))
        log.debug("Evicting LRU chunk %s", lru_id[:8])

        if self._disk_dir:
            self._save_to_disk(lru_chunk)

        del self._hot[lru_id]

    def _save_to_disk(self, chunk: CachedChunk) -> None:
        path = self._disk_dir / f"{chunk.chunk_id}.pt"
        payload = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "token_ids": chunk.token_ids,
            "past_key_values": chunk.past_key_values,
            "position_start": chunk.position_start,
            "position_end": chunk.position_end,
            "created_at": chunk.created_at,
            "access_count": chunk.access_count,
        }
        torch.save(payload, path)

    def _load_from_disk(self, path: Path) -> CachedChunk:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        return CachedChunk(**payload)

    @staticmethod
    def _move_kv(kv: PastKVType, device: str) -> PastKVType:
        return tuple((layer[0].to(device), layer[1].to(device)) for layer in kv)
