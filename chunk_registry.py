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
from typing import List, Tuple

from models import chunk_id_from_tokens

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
    ):
        self.chunk_size = chunk_size
        self.strategy = strategy
        self.min_chunk_tokens = min_chunk_tokens

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def split(
        self, token_ids: List[int], text: str = ""
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Split token_ids into cacheable slices.

        Returns list of (start, end, slice_token_ids).
        end is exclusive.
        """
        if self.strategy == ChunkStrategy.FIXED:
            return self._fixed_split(token_ids)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            return self._semantic_split(token_ids, text)
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

    def _semantic_split(
        self, token_ids: List[int], text: str
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Use paragraph / double-newline boundaries as split points,
        then fall back to fixed chunking if segments are too large.
        Works purely on token positions by finding sentence-boundary
        tokens (very approximate without a proper aligner).
        For a prototype, we use fixed chunking at paragraph-size
        aligned boundaries.
        """
        # Without a true token→char aligner we approximate:
        # find newline tokens in the sequence by guessing they appear
        # roughly proportional to char positions.  Then enforce chunk_size
        # as the max.
        if not text:
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

        # Merge with chunk_size constraint
        boundaries = [0] + token_splits + [n_tokens]
        chunks = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            sub = token_ids[start:end]
            # If sub-segment is too large, further split fixed
            if len(sub) > self.chunk_size:
                for s, e, sl in self._fixed_split(sub):
                    chunks.append((start + s, start + e, sl))
            elif len(sub) >= self.min_chunk_tokens:
                chunks.append((start, end, sub))
        return chunks
