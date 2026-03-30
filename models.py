"""
Core data structures for chunk-level KV cache reuse.
KV tensors live in HuggingFace past_key_values format:
  tuple[num_layers] of (key, value)
  key/value shape: [batch=1, num_heads, seq_len, head_dim]
"""

from __future__ import annotations
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# HF past_key_values type alias
PastKVType = Tuple[Tuple["torch.Tensor", "torch.Tensor"], ...]


@dataclass
class CachedChunk:
    """
    A single cached chunk: a slice of tokenized text + its KV tensors.
    position_start / position_end are the absolute token positions at
    which this chunk was originally encoded. They drive position_ids
    on reuse so RoPE offsets stay consistent.
    """
    chunk_id: str               # SHA256 of token_ids bytes
    text: str                   # original text (debugging / display)
    token_ids: List[int]        # tokenized form
    past_key_values: PastKVType # extracted KV tensors (on CPU by default)
    position_start: int         # absolute position of first token
    position_end: int           # absolute position of last token + 1
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    recomputed: bool = False    # True if boundary recompute was applied

    @property
    def length(self) -> int:
        return self.position_end - self.position_start

    def touch(self) -> None:
        self.access_count += 1

    def memory_bytes(self) -> int:
        total = 0
        for layer in self.past_key_values:
            total += layer[0].nelement() * layer[0].element_size()
            total += layer[1].nelement() * layer[1].element_size()
        return total

    def __repr__(self) -> str:
        mb = self.memory_bytes() / 1e6
        return (
            f"CachedChunk(id={self.chunk_id[:8]}, "
            f"pos=[{self.position_start},{self.position_end}), "
            f"len={self.length}, mem={mb:.2f}MB, hits={self.access_count})"
        )


@dataclass
class AssembledPrompt:
    """
    Result of stitching cached chunks + live (uncached) tail tokens.

    cached_past_kv   : merged KV tensors for all cached tokens
    cached_length    : number of tokens covered by cached_past_kv
    live_token_ids   : tokens that still need a fresh forward pass
    live_position_ids: absolute positions for each live token
    chunk_boundaries : list of (start, end) for each reused chunk
                       (used by SelectiveRecompute to find seam positions)
    cache_hit_ratio  : fraction of total tokens served from cache
    """
    full_token_ids: List[int]
    cached_past_kv: Optional[PastKVType]
    cached_length: int
    live_token_ids: List[int]
    live_position_ids: List[int]
    chunk_boundaries: List[Tuple[int, int]]
    cache_hit_ratio: float

    @property
    def total_length(self) -> int:
        return len(self.full_token_ids)


def chunk_id_from_tokens(token_ids: List[int]) -> str:
    """Deterministic, canonical hash for a token sequence."""
    raw = b"".join(t.to_bytes(4, "little") for t in token_ids)
    return hashlib.sha256(raw).hexdigest()