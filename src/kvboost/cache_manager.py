"""
KVCacheManager
==============
Two-tier storage with two-tier keying:

  Storage tiers:
    Tier 1 (hot)  : in-process dict, tensors on CPU RAM
    Tier 2 (cold) : mmap'd files on disk via torch.save / torch.load

  Key tiers:
    prefix_hash   : hash(parent_hash, tokens) — exact match, positionally correct
    content_hash  : hash(tokens) — approximate match, needs full recompute

  Lookup order:
    1. Try prefix_hash (exact) → use directly
    2. Try content_hash (approximate) → use but flag for full recompute

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

from .models import CachedChunk, PastKVType, content_hash_from_tokens, chained_hash

log = logging.getLogger(__name__)


class ChunkMatch:
    """Result of a cache lookup — tracks whether the match is exact or approximate."""
    __slots__ = ("chunk", "approximate")

    def __init__(self, chunk: CachedChunk, approximate: bool = False):
        self.chunk = chunk
        self.approximate = approximate


class KVCacheManager:
    def __init__(
        self,
        max_chunks: int = 64,
        disk_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.max_chunks = max_chunks
        self.device = device

        # Primary store keyed by prefix_hash (exact match)
        self._hot: OrderedDict[str, CachedChunk] = OrderedDict()

        # Secondary index: content_hash → prefix_hash for approximate lookup
        self._content_index: Dict[str, str] = {}

        # Frequency counter: chunk_id → number of generate() calls it appeared in.
        # Chunks that appear across many requests (system prompts) get high counts
        # and are protected from eviction. One-off document chunks stay at 1.
        self._frequency: Dict[str, int] = {}

        # Optional disk tier
        self._disk_dir: Optional[Path] = None
        if disk_dir:
            self._disk_dir = Path(disk_dir)
            self._disk_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.hits = 0
        self.misses = 0
        self.approximate_hits = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, chunk: CachedChunk) -> None:
        """Store a chunk. Evicts lowest-frequency entry if over capacity."""
        key = chunk.prefix_hash or chunk.chunk_id

        if key in self._hot:
            self._hot.move_to_end(key)
            self._frequency[key] = self._frequency.get(key, 0) + 1
            return

        # Move KV tensors to storage device
        chunk.past_key_values = self._move_kv(chunk.past_key_values, self.device)

        if len(self._hot) >= self.max_chunks:
            self._evict_lfu()

        self._hot[key] = chunk
        self._frequency[key] = 1

        # Index by content_hash for approximate lookup
        if chunk.content_hash:
            self._content_index[chunk.content_hash] = key

        log.debug("Stored chunk %s (%.2fMB)", key[:8], chunk.memory_bytes() / 1e6)

    def get(self, chunk_id: str) -> Optional[CachedChunk]:
        """Retrieve a chunk by prefix_hash (exact match only)."""
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
                self.store(chunk)
                return chunk

        self.misses += 1
        return None

    def get_by_content(self, content_hash: str) -> Optional[ChunkMatch]:
        """
        Look up by content_hash (approximate match).
        Returns ChunkMatch with approximate=True if found via content index.
        """
        # Check if any stored chunk has this content hash
        if content_hash in self._content_index:
            prefix_key = self._content_index[content_hash]
            if prefix_key in self._hot:
                self.approximate_hits += 1
                chunk = self._hot[prefix_key]
                chunk.touch()
                self._hot.move_to_end(prefix_key)
                return ChunkMatch(chunk=chunk, approximate=True)

        return None

    def lookup(self, token_ids: List[int], parent_hash: Optional[str] = None) -> Optional[ChunkMatch]:
        """
        Two-tier lookup:
        1. Try prefix-chained hash (exact) — correct position + context
        2. Fall back to content hash (approximate) — flagged for full recompute

        Returns ChunkMatch or None.
        """
        # Tier 1: exact match by prefix chain
        p_hash = chained_hash(token_ids, parent_hash)
        chunk = self.get(p_hash)
        if chunk is not None:
            return ChunkMatch(chunk=chunk, approximate=False)

        # Tier 2: approximate match by content
        c_hash = content_hash_from_tokens(token_ids)
        return self.get_by_content(c_hash)

    def get_or_none(self, token_ids: List[int]) -> Optional[CachedChunk]:
        """Legacy API: look up by content hash only (backwards compat)."""
        c_hash = content_hash_from_tokens(token_ids)
        if c_hash in self._content_index:
            prefix_key = self._content_index[c_hash]
            if prefix_key in self._hot:
                self.hits += 1
                chunk = self._hot[prefix_key]
                chunk.touch()
                self._hot.move_to_end(prefix_key)
                return chunk
        # Direct lookup by content hash (for chunks stored with old keying)
        if c_hash in self._hot:
            self.hits += 1
            chunk = self._hot[c_hash]
            chunk.touch()
            self._hot.move_to_end(c_hash)
            return chunk
        self.misses += 1
        return None

    def build_prefix_kv(
        self, token_ids: List[int], chunk_size: int
    ) -> Tuple[Optional[PastKVType], int]:
        """
        Greedily assemble the longest cached prefix using chained hashes.
        Only exact matches are used (no approximate fallback for prefix mode).
        """
        chunks: List[CachedChunk] = []
        pos = 0
        parent_hash = None

        while pos + chunk_size <= len(token_ids):
            slice_ids = token_ids[pos : pos + chunk_size]
            p_hash = chained_hash(slice_ids, parent_hash)
            chunk = self.get(p_hash)
            if chunk is None:
                break
            chunks.append(chunk)
            parent_hash = p_hash
            pos += chunk_size

        if not chunks:
            return None, 0

        merged = self.merge_kv_list([c.past_key_values for c in chunks])
        return merged, pos

    def find_matching_chunks(
        self, token_ids: List[int], chunk_size: int
    ) -> List[Tuple[int, ChunkMatch]]:
        """
        Scan for all matching chunks using two-tier lookup.
        Returns list of (start_pos, ChunkMatch) pairs in order.
        Each ChunkMatch carries approximate=True/False.
        """
        results = []
        parent_hash = None

        for start in range(0, len(token_ids) - chunk_size + 1, chunk_size):
            slice_ids = token_ids[start : start + chunk_size]
            match = self.lookup(slice_ids, parent_hash)
            if match is not None:
                results.append((start, match))
                # Chain the hash for the next chunk (use the exact prefix hash)
                parent_hash = chained_hash(slice_ids, parent_hash)
            else:
                # Chain is broken — subsequent chunks can't be exact matches
                parent_hash = None

        return results

    def invalidate(self, chunk_id: str) -> None:
        chunk = self._hot.pop(chunk_id, None)
        if chunk and chunk.content_hash in self._content_index:
            if self._content_index[chunk.content_hash] == chunk_id:
                del self._content_index[chunk.content_hash]
        if self._disk_dir:
            path = self._disk_dir / f"{chunk_id}.pt"
            path.unlink(missing_ok=True)

    def stats(self) -> Dict:
        total = self.hits + self.misses + self.approximate_hits
        hot_mb = sum(c.memory_bytes() for c in self._hot.values()) / 1e6
        return {
            "hot_chunks": len(self._hot),
            "hot_memory_mb": round(hot_mb, 2),
            "cache_hits": self.hits,
            "approximate_hits": self.approximate_hits,
            "cache_misses": self.misses,
            "hit_rate": round((self.hits + self.approximate_hits) / max(total, 1), 3),
            "exact_hit_rate": round(self.hits / max(total, 1), 3),
        }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def merge_kv_list(kv_list: List[PastKVType]) -> PastKVType:
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
    def slice_kv(kv: PastKVType, start: int, end: int) -> PastKVType:
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

    def _evict_lfu(self) -> None:
        """
        Frequency-aware eviction: evict the chunk with the lowest frequency
        count. Chunks that appear across many generate() calls (system prompts)
        are protected; one-off document chunks are evicted first.

        Tie-breaking: among chunks with equal frequency, evict the LRU one
        (first in OrderedDict insertion order).
        """
        if not self._hot:
            return

        # Find the entry with the lowest frequency
        min_freq = float("inf")
        victim_id = None
        for cid in self._hot:
            freq = self._frequency.get(cid, 0)
            if freq < min_freq:
                min_freq = freq
                victim_id = cid

        if victim_id is None:
            return

        victim = self._hot[victim_id]
        log.debug("Evicting chunk %s (freq=%d)", victim_id[:8], min_freq)

        # Clean up content index
        if victim.content_hash in self._content_index:
            if self._content_index[victim.content_hash] == victim_id:
                del self._content_index[victim.content_hash]
        if self._disk_dir:
            self._save_to_disk(victim)
        del self._hot[victim_id]
        self._frequency.pop(victim_id, None)

    def _save_to_disk(self, chunk: CachedChunk) -> None:
        key = chunk.prefix_hash or chunk.chunk_id
        path = self._disk_dir / f"{key}.pt"
        payload = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "token_ids": chunk.token_ids,
            "past_key_values": chunk.past_key_values,
            "position_start": chunk.position_start,
            "position_end": chunk.position_end,
            "prefix_hash": chunk.prefix_hash,
            "content_hash": chunk.content_hash,
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
