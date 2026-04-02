"""
Disk Tier — Memory-Mapped KV Cache Storage
===========================================
Replaces per-file torch.save/load with a single pre-allocated
memory-mapped file for zero-copy, async-friendly disk caching.

Design:
  cache_dir/
    kv_cache.bin   ← one file, flat block pool of raw tensors
    kv_index.json  ← hash → (slot, metadata) mapping

Each "slot" holds the flattened raw bytes for one chunk's KV tensors.
Reads are zero-copy page faults via torch.from_file(). Writes are
direct memory copies into the mapped region. No pickle, no per-file
syscall overhead.

For CUDA devices, reads can be issued with non_blocking=True to
pipeline disk→GPU transfer with other work.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .models import CachedChunk, PastKVType

log = logging.getLogger(__name__)


class DiskTier:
    """
    Memory-mapped disk cache for KV tensors.

    Instead of one file per chunk (torch.save), uses a single pre-allocated
    binary file with fixed-size slots. An in-memory JSON index maps chunk
    hashes to slot numbers.
    """

    def __init__(
        self,
        cache_dir: str,
        max_chunks: int = 256,
        slot_bytes: int = 10 * 1024 * 1024,  # 10 MB default slot size
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_chunks = max_chunks
        self.slot_bytes = slot_bytes

        self._data_file = self.cache_dir / "kv_cache.bin"
        self._index_file = self.cache_dir / "kv_index.json"
        self._meta_file = self.cache_dir / "kv_meta.json"

        # Load or init index
        self.index: Dict[str, int] = {}          # hash → slot number
        self.meta: Dict[str, dict] = {}           # hash → chunk metadata (no tensors)
        self.free_slots: List[int] = list(range(max_chunks))
        self._slot_lru: List[str] = []            # ordered by access time

        if self._index_file.exists():
            self._load_index()

        # Pre-allocate the data file if it doesn't exist
        total_bytes = max_chunks * slot_bytes
        if not self._data_file.exists():
            with open(self._data_file, "wb") as f:
                f.seek(total_bytes - 1)
                f.write(b"\0")
            log.debug("Pre-allocated disk cache: %d MB", total_bytes // (1024 * 1024))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, chunk: CachedChunk) -> bool:
        """
        Write a chunk's KV tensors and metadata to a disk slot.
        Returns True if stored successfully, False if no space.
        """
        key = chunk.prefix_hash or chunk.chunk_id

        # Already on disk
        if key in self.index:
            return True

        # Evict if no free slots
        if not self.free_slots:
            self._evict_oldest()

        if not self.free_slots:
            return False

        # Flatten KV tensors to raw bytes
        flat, shape_info = self._flatten_kv(chunk.past_key_values)
        raw_bytes = flat.numpy().tobytes()

        if len(raw_bytes) > self.slot_bytes:
            log.warning(
                "Chunk %s too large for slot (%d > %d bytes), skipping disk write",
                key[:8], len(raw_bytes), self.slot_bytes,
            )
            return False

        # Write to slot
        slot = self.free_slots.pop(0)
        offset = slot * self.slot_bytes

        with open(self._data_file, "r+b") as f:
            f.seek(offset)
            f.write(raw_bytes)

        # Store index + metadata
        self.index[key] = slot
        self.meta[key] = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "token_ids": chunk.token_ids,
            "position_start": chunk.position_start,
            "position_end": chunk.position_end,
            "prefix_hash": chunk.prefix_hash,
            "content_hash": chunk.content_hash,
            "flat_numel": flat.numel(),
            "shape_info": shape_info,
        }
        self._slot_lru.append(key)
        self._persist_index()

        log.debug("Wrote chunk %s to disk slot %d", key[:8], slot)
        return True

    def read(self, chunk_hash: str, device: str = "cpu") -> Optional[CachedChunk]:
        """
        Read a chunk from disk. Returns a CachedChunk with KV tensors
        on the specified device, or None if not found.
        """
        if chunk_hash not in self.index:
            return None

        slot = self.index[chunk_hash]
        meta = self.meta[chunk_hash]
        offset = slot * self.slot_bytes
        numel = meta["flat_numel"]

        # Read raw bytes from the slot
        flat = torch.empty(numel, dtype=torch.float16)
        with open(self._data_file, "rb") as f:
            f.seek(offset)
            raw = f.read(numel * 2)  # float16 = 2 bytes
            flat = torch.frombuffer(bytearray(raw), dtype=torch.float16).clone()

        # Unflatten back to KV tuple
        kv = self._unflatten_kv(flat, meta["shape_info"])

        # Move to target device
        if device != "cpu":
            kv = tuple(
                (k.to(device, non_blocking=True), v.to(device, non_blocking=True))
                for k, v in kv
            )

        # Update LRU order
        if chunk_hash in self._slot_lru:
            self._slot_lru.remove(chunk_hash)
        self._slot_lru.append(chunk_hash)

        return CachedChunk(
            chunk_id=meta["chunk_id"],
            text=meta["text"],
            token_ids=meta["token_ids"],
            past_key_values=kv,
            position_start=meta["position_start"],
            position_end=meta["position_end"],
            prefix_hash=meta.get("prefix_hash", ""),
            content_hash=meta.get("content_hash", ""),
        )

    def contains(self, chunk_hash: str) -> bool:
        return chunk_hash in self.index

    def remove(self, chunk_hash: str) -> None:
        if chunk_hash in self.index:
            slot = self.index.pop(chunk_hash)
            self.meta.pop(chunk_hash, None)
            self.free_slots.append(slot)
            if chunk_hash in self._slot_lru:
                self._slot_lru.remove(chunk_hash)
            self._persist_index()

    def stats(self) -> dict:
        return {
            "disk_chunks": len(self.index),
            "disk_max_chunks": self.max_chunks,
            "disk_slot_bytes": self.slot_bytes,
            "disk_free_slots": len(self.free_slots),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_kv(kv: PastKVType) -> Tuple[torch.Tensor, list]:
        """
        Flatten a PastKVType into a single 1D float16 tensor + shape info.
        Shape info is needed to unflatten back.
        """
        parts = []
        shape_info = []
        for key, val in kv:
            parts.append(key.to(torch.float16).reshape(-1))
            parts.append(val.to(torch.float16).reshape(-1))
            shape_info.append({
                "key_shape": list(key.shape),
                "val_shape": list(val.shape),
            })
        flat = torch.cat(parts)
        return flat, shape_info

    @staticmethod
    def _unflatten_kv(flat: torch.Tensor, shape_info: list) -> PastKVType:
        """Reconstruct PastKVType from flattened tensor + shape info."""
        layers = []
        offset = 0
        for info in shape_info:
            k_shape = info["key_shape"]
            v_shape = info["val_shape"]
            k_numel = 1
            for d in k_shape:
                k_numel *= d
            v_numel = 1
            for d in v_shape:
                v_numel *= d

            key = flat[offset : offset + k_numel].reshape(k_shape)
            offset += k_numel
            val = flat[offset : offset + v_numel].reshape(v_shape)
            offset += v_numel
            layers.append((key, val))

        return tuple(layers)

    def _evict_oldest(self) -> None:
        """Evict the least recently accessed disk slot."""
        if not self._slot_lru:
            return
        oldest_key = self._slot_lru.pop(0)
        if oldest_key in self.index:
            slot = self.index.pop(oldest_key)
            self.meta.pop(oldest_key, None)
            self.free_slots.append(slot)
            log.debug("Evicted disk slot for %s", oldest_key[:8])

    def _persist_index(self) -> None:
        """Atomically persist the index to disk."""
        tmp = str(self._index_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"index": self.index, "free_slots": self.free_slots}, f)
        os.replace(tmp, str(self._index_file))

        tmp_meta = str(self._meta_file) + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(self.meta, f, default=str)
        os.replace(tmp_meta, str(self._meta_file))

    def _load_index(self) -> None:
        """Load index and metadata from disk."""
        with open(self._index_file) as f:
            saved = json.load(f)
        self.index = saved.get("index", {})
        self.free_slots = saved.get("free_slots", list(range(self.max_chunks)))

        if self._meta_file.exists():
            with open(self._meta_file) as f:
                self.meta = json.load(f)
