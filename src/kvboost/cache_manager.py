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

Eviction policy: hard byte budget with recency window + importance-based eviction.

  - max_cache_bytes is required and enforced on every store().
  - The most-recently-stored `recency_window_chunks` are pinned (never evicted).
  - When budget is exceeded, chunks outside the window are evicted lowest
    importance first (ties → LRU). Pruned = evicted: a dropped chunk is
    gone entirely and becomes a cache miss on next lookup (recomputed on demand).
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
        *,
        max_cache_bytes: int,
        recency_window_chunks: int = 8,
        max_chunks: int = 64,
        disk_dir: Optional[str] = None,
        device: Optional[str] = None,
        kv_cache_bits: int = 16,
    ):
        if max_cache_bytes is None or max_cache_bytes <= 0:
            raise ValueError(
                "max_cache_bytes is required and must be > 0 "
                "(strictly memory-bounded KV cache)."
            )
        if recency_window_chunks < 0:
            raise ValueError("recency_window_chunks must be >= 0.")
        self.max_cache_bytes = int(max_cache_bytes)
        self.recency_window_chunks = int(recency_window_chunks)
        self.max_chunks = max_chunks
        self.device = device if device is not None else default_device()
        self.kv_cache_bits = kv_cache_bits  # 16 = no quantization, 8 = int8, 4 = int4

        # Primary store keyed by prefix_hash (exact match). Insertion order
        # is also recency order — the last N entries form the pinned window.
        self._hot: OrderedDict[str, CachedChunk] = OrderedDict()

        # Quantized storage: chunk_id → QuantizedKV (when bits < 16)
        self._quantized: Dict[str, QuantizedKV] = {}

        # Secondary index: content_hash → prefix_hash for approximate lookup
        self._content_index: Dict[str, str] = {}

        # Frequency counter: chunk_id → number of generate() calls it appeared in.
        # Used as a secondary tiebreaker when importance is equal.
        self._frequency: Dict[str, int] = {}

        # Byte-size bookkeeping: tracked at store/evict time so we never
        # have to walk every chunk on the hot path.
        self._bytes_per_chunk: Dict[str, int] = {}
        self._bytes_used: int = 0

        # Optional disk tier (flat mmap block pool)
        self._disk: Optional[DiskTier] = None
        if disk_dir:
            self._disk = DiskTier(
                cache_dir=disk_dir,
                max_chunks=max_chunks * 2,  # cold tier can hold more than hot
            )

        # Eviction callbacks: called with chunk_id when a chunk is evicted
        self._eviction_callbacks: list = []

        # Stats
        self.hits = 0
        self.misses = 0
        self.approximate_hits = 0
        self.evictions = 0
        self.budget_rejections = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_eviction_callback(self, callback) -> None:
        """Register a callable(chunk_id: str) invoked whenever a chunk is evicted."""
        self._eviction_callbacks.append(callback)

    def store(self, chunk: CachedChunk) -> None:
        """
        Store a chunk under the strict byte budget.

        On insert:
          1. Move KV tensors to storage device (and optionally quantize).
          2. Compute the chunk's on-device byte size.
          3. Evict older chunks (outside the recency window) — lowest
             importance first — until the new chunk fits in the budget.
          4. If the new chunk still doesn't fit after evicting everything
             outside the window, it is rejected (pruned = evicted).
        """
        key = chunk.prefix_hash or chunk.chunk_id

        if key in self._hot:
            # Re-hit on an already-cached chunk: bump recency + frequency.
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

        incoming_bytes = self._chunk_bytes(chunk, key)

        # Reject chunks that can't possibly fit even with an empty cache.
        if incoming_bytes > self.max_cache_bytes:
            log.debug(
                "Rejecting chunk %s: %.2fMB exceeds total budget %.2fMB",
                key[:8], incoming_bytes / 1e6, self.max_cache_bytes / 1e6,
            )
            self.budget_rejections += 1
            # Drop any quantized artifact we created above.
            self._quantized.pop(key, None)
            return

        # Evict older chunks (outside recency window) until we fit.
        self._evict_until_fits(incoming_bytes)

        # If the recency window alone still doesn't leave room, reject.
        if self._bytes_used + incoming_bytes > self.max_cache_bytes:
            log.debug(
                "Rejecting chunk %s: recency window (%d chunks, %.2fMB) "
                "leaves no room for %.2fMB",
                key[:8], self.recency_window_chunks,
                self._bytes_used / 1e6, incoming_bytes / 1e6,
            )
            self.budget_rejections += 1
            self._quantized.pop(key, None)
            return

        self._hot[key] = chunk
        self._frequency[key] = 1
        self._bytes_per_chunk[key] = incoming_bytes
        self._bytes_used += incoming_bytes

        # Index by content_hash for approximate lookup
        if chunk.content_hash:
            self._content_index[chunk.content_hash] = key

        if self.kv_cache_bits >= 16:
            log.debug(
                "Stored chunk %s (%.2fMB, cache=%.2f/%.2fMB, %d chunks)",
                key[:8], incoming_bytes / 1e6,
                self._bytes_used / 1e6, self.max_cache_bytes / 1e6,
                len(self._hot),
            )

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
        self,
        token_ids: List[int],
        chunk_size: int,
        chunk_splits: Optional[List[Tuple[int, int, List[int]]]] = None,
    ) -> Tuple[Optional[PastKVType], int]:
        """
        Greedily assemble the longest cached prefix using chained hashes.
        Only exact matches are used (no approximate fallback for prefix mode).

        If chunk_splits is provided (from adaptive/variable splitting),
        uses those boundaries instead of fixed chunk_size stride.
        """
        chunks: List[CachedChunk] = []
        parent_hash = None

        if chunk_splits is not None:
            for start, end, slice_ids in chunk_splits:
                p_hash = chained_hash(slice_ids, parent_hash)
                chunk = self.get(p_hash)
                if chunk is None:
                    break
                chunks.append(chunk)
                parent_hash = p_hash
        else:
            pos = 0
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

        covered = sum(c.length for c in chunks)
        merged = self.merge_kv_list([c.past_key_values for c in chunks])
        return merged, covered

    def find_matching_chunks(
        self,
        token_ids: List[int],
        chunk_size: int,
        chunk_splits: Optional[List[Tuple[int, int, List[int]]]] = None,
    ) -> List[Tuple[int, ChunkMatch]]:
        """
        Scan for all matching chunks using two-tier lookup.
        Returns list of (start_pos, ChunkMatch) pairs in order.
        Each ChunkMatch carries approximate=True/False.

        If chunk_splits is provided (from adaptive/variable splitting),
        uses those boundaries instead of fixed chunk_size stride.
        """
        results = []
        parent_hash = None

        if chunk_splits is not None:
            for start, end, slice_ids in chunk_splits:
                match = self.lookup(slice_ids, parent_hash)
                if match is not None:
                    results.append((start, match))
                    parent_hash = chained_hash(slice_ids, parent_hash)
                else:
                    parent_hash = None
        else:
            for start in range(0, len(token_ids) - chunk_size + 1, chunk_size):
                slice_ids = token_ids[start : start + chunk_size]
                match = self.lookup(slice_ids, parent_hash)
                if match is not None:
                    results.append((start, match))
                    parent_hash = chained_hash(slice_ids, parent_hash)
                else:
                    parent_hash = None

        return results

    def invalidate(self, chunk_id: str) -> None:
        chunk = self._hot.pop(chunk_id, None)
        if chunk and chunk.content_hash in self._content_index:
            if self._content_index[chunk.content_hash] == chunk_id:
                del self._content_index[chunk.content_hash]
        self._quantized.pop(chunk_id, None)
        self._frequency.pop(chunk_id, None)
        self._bytes_used -= self._bytes_per_chunk.pop(chunk_id, 0)
        if self._bytes_used < 0:
            self._bytes_used = 0
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
        self._bytes_per_chunk.clear()
        self._bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.approximate_hits = 0
        self.evictions = 0
        self.budget_rejections = 0

    def stats(self) -> Dict:
        total = self.hits + self.misses + self.approximate_hits
        result = {
            "hot_chunks": len(self._hot),
            "hot_memory_mb": round(self._bytes_used / 1e6, 2),
            "budget_mb": round(self.max_cache_bytes / 1e6, 2),
            "budget_utilization": round(self._bytes_used / max(self.max_cache_bytes, 1), 3),
            "recency_window_chunks": self.recency_window_chunks,
            "cache_hits": self.hits,
            "approximate_hits": self.approximate_hits,
            "cache_misses": self.misses,
            "evictions": self.evictions,
            "budget_rejections": self.budget_rejections,
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

    def _chunk_bytes(self, chunk: CachedChunk, key: str) -> int:
        """
        Size in bytes of a chunk as stored. Uses the quantized representation
        if present, otherwise the raw KV tensors.
        """
        if key in self._quantized:
            return int(self._quantized[key].memory_bytes())
        return int(chunk.memory_bytes())

    def _pinned_keys(self) -> set:
        """
        The most-recently-stored `recency_window_chunks` keys. These are
        NEVER evicted: they represent the model's hard sliding window of
        "guaranteed to survive" context. Returned as a set for O(1) lookup.
        """
        if self.recency_window_chunks <= 0 or not self._hot:
            return set()
        # _hot is an OrderedDict in insertion order; last N keys = window.
        all_keys = list(self._hot.keys())
        return set(all_keys[-self.recency_window_chunks:])

    def _evict_until_fits(self, incoming_bytes: int) -> None:
        """
        Evict older chunks (outside the recency window) until the incoming
        chunk would fit in the budget. Victim selection:
          1. Lowest importance first.
          2. Tie → lowest frequency.
          3. Tie → LRU (earliest in insertion order).

        Stops when either the budget has room or no evictable chunks remain.
        """
        target = self.max_cache_bytes - incoming_bytes
        if self._bytes_used <= target:
            return

        pinned = self._pinned_keys()
        candidates = [k for k in self._hot.keys() if k not in pinned]
        if not candidates:
            return

        # Sort candidates: lowest importance, then frequency, then LRU order.
        # Insertion order in self._hot gives us LRU → so enumerate for index.
        order_index = {k: i for i, k in enumerate(self._hot.keys())}
        candidates.sort(
            key=lambda k: (
                float(self._hot[k].importance),
                self._frequency.get(k, 0),
                order_index[k],
            )
        )

        for victim_id in candidates:
            if self._bytes_used <= target:
                break
            self._evict_one(victim_id)

    def _evict_one(self, victim_id: str) -> None:
        """Drop a single chunk. Pruned = evicted: cache miss next time."""
        victim = self._hot.get(victim_id)
        if victim is None:
            return

        log.debug(
            "Evicting chunk %s (importance=%.3f, freq=%d, %.2fMB)",
            victim_id[:8],
            float(victim.importance),
            self._frequency.get(victim_id, 0),
            self._bytes_per_chunk.get(victim_id, 0) / 1e6,
        )

        if victim.content_hash in self._content_index:
            if self._content_index[victim.content_hash] == victim_id:
                del self._content_index[victim.content_hash]

        # Demote to disk tier if available (preserves exact-match recovery).
        if self._disk:
            if victim_id in self._quantized:
                victim = self._dequantize_chunk(victim, victim_id)
            self._disk.write(victim)

        del self._hot[victim_id]
        self._frequency.pop(victim_id, None)
        self._quantized.pop(victim_id, None)
        self._bytes_used -= self._bytes_per_chunk.pop(victim_id, 0)
        if self._bytes_used < 0:
            self._bytes_used = 0
        self.evictions += 1

        for cb in self._eviction_callbacks:
            try:
                cb(victim_id)
            except Exception:
                log.exception("Error in eviction callback for chunk %s", victim_id[:8])

    @staticmethod
    def _move_kv(kv: PastKVType, device: str) -> PastKVType:
        return tuple((layer[0].to(device), layer[1].to(device)) for layer in kv)
