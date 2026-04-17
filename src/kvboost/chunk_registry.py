"""
ChunkRegistry
=============
Splits text into cacheable chunks and owns the chunk_size policy.

Chunking strategies
-------------------
FIXED      : fixed token-count windows (default, predictable)
SEMANTIC   : split on paragraph / sentence boundaries (better reuse)
DOCUMENT   : treat entire document as one chunk
"""

from __future__ import annotations

import enum
import logging
import re
from typing import List, Optional, Set, Tuple

from .models import chunk_id_from_tokens

log = logging.getLogger(__name__)


class ChunkStrategy(str, enum.Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    DOCUMENT = "document"


class ChunkRegistry:
    """
    Converts (text, token_ids) into a list of (start, end, sub_token_ids) triples
    according to the configured strategy.

    The registry itself holds no KV state — that lives in KVCacheManager.
    """

    def __init__(
        self,
        chunk_size: int = 128,
        strategy: ChunkStrategy = ChunkStrategy.FIXED,
        min_chunk_tokens: int = 32,
        boundary_window: int = 0,
    ):
        self.chunk_size = chunk_size
        self.strategy = strategy
        self.min_chunk_tokens = min_chunk_tokens
        # Clamp: window can't be larger than half the chunk size
        self.boundary_window = min(boundary_window, chunk_size // 2)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def split(
        self,
        token_ids: List[int],
        text: str = "",
        boundary_tokens: Optional[Set[int]] = None,
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Split token_ids into cacheable slices.

        Returns list of (start, end, slice_token_ids).
        end is exclusive.

        If boundary_window > 0 and boundary_tokens is provided,
        split points are nudged to sentence/clause boundaries.
        """
        use_adaptive = (
            self.boundary_window > 0
            and boundary_tokens
            and self.strategy in (ChunkStrategy.FIXED, ChunkStrategy.SEMANTIC)
        )

        if self.strategy == ChunkStrategy.FIXED:
            if use_adaptive:
                return self._adaptive_split(token_ids, boundary_tokens)
            return self._fixed_split(token_ids)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            return self._semantic_split(token_ids, text, boundary_tokens)
        elif self.strategy == ChunkStrategy.DOCUMENT:
            return [(0, len(token_ids), token_ids)]
        raise ValueError(f"Unknown strategy {self.strategy}")

    def chunk_ids_for(self, token_ids: List[int]) -> List[str]:
        """Return the chunk_ids (hashes) for each chunk of this token sequence."""
        return [
            chunk_id_from_tokens(slice_ids)
            for _, _, slice_ids in self.split(token_ids)
        ]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _fixed_split(
        self, token_ids: List[int]
    ) -> List[Tuple[int, int, List[int]]]:
        chunks = []
        pos = 0
        n = len(token_ids)
        while pos < n:
            end = min(pos + self.chunk_size, n)
            slice_ids = token_ids[pos:end]
            if len(slice_ids) >= self.min_chunk_tokens:
                chunks.append((pos, end, slice_ids))
            pos = end
        return chunks

    def _adaptive_split(
        self, token_ids: List[int], boundary_tokens: Set[int]
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Like _fixed_split, but nudges each split point to the nearest
        sentence/clause boundary within ±boundary_window of the target.

        Prefers the boundary token closest to the target split point.
        Falls back to the fixed position if no boundary is found.
        """
        chunks = []
        pos = 0
        n = len(token_ids)
        w = self.boundary_window

        while pos < n:
            target = pos + self.chunk_size
            if target >= n:
                # Last chunk — take whatever remains
                slice_ids = token_ids[pos:]
                if len(slice_ids) >= self.min_chunk_tokens:
                    chunks.append((pos, n, slice_ids))
                break

            # Search [target - w, target + w] for a boundary token
            lo = max(pos + self.min_chunk_tokens, target - w)
            hi = min(n, target + w)

            best = target  # fallback: exact fixed position
            best_dist = w + 1

            for i in range(lo, hi):
                if token_ids[i] in boundary_tokens:
                    # Split *after* the boundary token (i+1)
                    dist = abs((i + 1) - target)
                    if dist < best_dist:
                        best_dist = dist
                        best = i + 1

            end = min(best, n)
            slice_ids = token_ids[pos:end]
            if len(slice_ids) >= self.min_chunk_tokens:
                chunks.append((pos, end, slice_ids))
            pos = end

        return chunks

    def _semantic_split(
        self,
        token_ids: List[int],
        text: str,
        boundary_tokens: Optional[Set[int]] = None,
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Use paragraph / double-newline boundaries as split points,
        then fall back to fixed/adaptive chunking if segments are too large.
        """
        if not text:
            if self.boundary_window > 0 and boundary_tokens:
                return self._adaptive_split(token_ids, boundary_tokens)
            return self._fixed_split(token_ids)

        # Split text on double newlines → get character offsets
        para_splits = [m.end() for m in re.finditer(r"\n\n+", text)]
        n_chars = max(len(text), 1)
        n_tokens = len(token_ids)

        # Map char offsets to approximate token offsets
        token_splits = sorted(
            set(
                int(cs / n_chars * n_tokens)
                for cs in para_splits
                if 0 < int(cs / n_chars * n_tokens) < n_tokens
            )
        )

        # Choose sub-split strategy for oversized segments
        use_adaptive = self.boundary_window > 0 and boundary_tokens

        def _subsplit(sub: List[int]) -> List[Tuple[int, int, List[int]]]:
            if use_adaptive:
                return self._adaptive_split(sub, boundary_tokens)
            return self._fixed_split(sub)

        # Merge with chunk_size constraint
        boundaries = [0] + token_splits + [n_tokens]
        chunks = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            sub = token_ids[start:end]
            # If sub-segment is too large, further split
            if len(sub) > self.chunk_size:
                for s, e, sl in _subsplit(sub):
                    chunks.append((start + s, start + e, sl))
            elif len(sub) >= self.min_chunk_tokens:
                chunks.append((start, end, sub))
        return chunks
