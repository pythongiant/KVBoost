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

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

from .models import CachedChunk, PastKVType, content_hash_from_tokens, chained_hash
from .kv_quantize import QuantizedKV, quantize_kv, dequantize_kv
from .disk_tier import DiskTier
from .compat import default_device

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
        device: Optional[str] = None,
        kv_cache_bits: int = 16,
    ):
        self.max_chunks = max_chunks
        self.device = device if device is not None else default_device()
        self.kv_cache_bits = kv_cache_bits  # 16 = no quantization, 8 = int8, 4 = int4

        # Primary store keyed by prefix_hash (exact match)
        self._hot: OrderedDict[str, CachedChunk] = OrderedDict()

        # Quantized storage: chunk_id → QuantizedKV (when bits < 16)
        self._quantized: Dict[str, QuantizedKV] = {}

        # Secondary index: content_hash → prefix_hash for approximate lookup
        self._content_index: Dict[str, str] = {}

        # Frequency counter: chunk_id → number of generate() calls it appeared in.
        # Chunks that appear across many requests (system prompts) get high counts
        # and are protected from eviction. One-off document chunks stay at 1.
        self._frequency: Dict[str, int] = {}

        # Optional disk tier (flat mmap block pool)
        self._disk: Optional[DiskTier] = None
        if disk_dir:
            self._disk = DiskTier(
                cache_dir=disk_dir,
                max_chunks=max_chunks * 2,  # cold tier can hold more than hot
            )

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

        # Quantize for compressed storage (if enabled)
        if self.kv_cache_bits < 16:
            qkv = quantize_kv(chunk.past_key_values, bits=self.kv_cache_bits)
            self._quantized[key] = qkv
            # Release the full-precision tensors — they'll be dequantized on get()
            chunk.past_key_values = ()  # empty sentinel
            log.debug(
                "Stored chunk %s quantized int%d (%.2fMB → %.2fMB)",
                key[:8], self.kv_cache_bits,
                qkv.memory_bytes() / 1e6 * (16 / self.kv_cache_bits),
                qkv.memory_bytes() / 1e6,
            )

        if len(self._hot) >= self.max_chunks:
            self._evict_lfu()

        self._hot[key] = chunk
        self._frequency[key] = 1

        # Index by content_hash for approximate lookup
        if chunk.content_hash:
            self._content_index[chunk.content_hash] = key

        if self.kv_cache_bits >= 16:
            log.debug("Stored chunk %s (%.2fMB)", key[:8], chunk.memory_bytes() / 1e6)

    def get(self, chunk_id: str) -> Optional[CachedChunk]:
        """Retrieve a chunk by prefix_hash (exact match only). Dequantizes if needed."""
        if chunk_id in self._hot:
            self.hits += 1
            chunk = self._hot[chunk_id]
            chunk.touch()
            self._hot.move_to_end(chunk_id)

            # Dequantize on load if stored quantized
            if chunk_id in self._quantized:
                chunk = self._dequantize_chunk(chunk, chunk_id)

            return chunk

        # Try disk tier
        if self._disk and self._disk.contains(chunk_id):
            self.hits += 1
            chunk = self._disk.read(chunk_id, device=self.device)
            if chunk is not None:
                self.store(chunk)  # promote to hot
                return self.get(chunk_id)  # re-enter to handle quantization

        self.misses += 1
        return None

    def _dequantize_chunk(self, chunk: CachedChunk, key: str) -> CachedChunk:
        """Reconstruct full-precision KV tensors from quantized storage."""
        qkv = self._quantized[key]
        # Return a copy with dequantized tensors (don't modify stored chunk)
        import copy
        out = copy.copy(chunk)
        out.past_key_values = dequantize_kv(qkv)
        return out

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
                # Dequantize if needed
                if prefix_key in self._quantized:
                    chunk = self._dequantize_chunk(chunk, prefix_key)
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
        self._quantized.pop(chunk_id, None)
        self._frequency.pop(chunk_id, None)
        if self._disk:
            self._disk.remove(chunk_id)

    def clear(self) -> None:
        """
        Fully clear the cache: reset all internal storage and stat counters.
        
        Clears:
          - _hot: in-memory hot cache
          - _quantized: quantized KV storage
          - _content_index: content hash → prefix hash index
          - _frequency: access frequency tracking
          - Stats counters: hits, misses, approximate_hits
        
        Does NOT clear disk tier (cold storage is preserved for recovery).
        
        Use this between benchmark iterations to ensure clean cache state.
        """
        self._hot.clear()
        self._quantized.clear()
        self._content_index.clear()
        self._frequency.clear()
        self.hits = 0
        self.misses = 0
        self.approximate_hits = 0

    def stats(self) -> Dict:
        total = self.hits + self.misses + self.approximate_hits
        hot_mb = sum(c.memory_bytes() for c in self._hot.values()) / 1e6
        result = {
            "hot_chunks": len(self._hot),
            "hot_memory_mb": round(hot_mb, 2),
            "cache_hits": self.hits,
            "approximate_hits": self.approximate_hits,
            "cache_misses": self.misses,
            "hit_rate": round((self.hits + self.approximate_hits) / max(total, 1), 3),
            "exact_hit_rate": round(self.hits / max(total, 1), 3),
        }
        if self._disk:
            result.update(self._disk.stats())
        return result

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

        # Demote to disk tier if available
        if self._disk:
            # Dequantize first if stored quantized
            if victim_id in self._quantized:
                victim = self._dequantize_chunk(victim, victim_id)
            self._disk.write(victim)

        del self._hot[victim_id]
        self._frequency.pop(victim_id, None)
        self._quantized.pop(victim_id, None)

    @staticmethod
    def _move_kv(kv: PastKVType, device: str) -> PastKVType:
        return tuple((layer[0].to(device), layer[1].to(device)) for layer in kv)
