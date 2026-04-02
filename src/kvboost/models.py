"""
Core data structures for chunk-level KV cache reuse.
KV tensors live in HuggingFace past_key_values format:
  tuple[num_layers] of (key, value)
  key/value shape: [batch=1, num_heads, seq_len, head_dim]

Cache Key Design
----------------
Two-tier keying (inspired by vLLM v1 block hashes):

  prefix_hash  : hash(parent_hash, token_ids) — unique to the full prefix
                 chain. Same tokens at different positions or with different
                 preceding context get different keys. Used for exact match.

  content_hash : hash(token_ids) — content-only, position-independent.
                 Used for approximate reuse: same tokens in a different
                 context can still be a cache hit, but the chunk is flagged
                 for mandatory full recompute (not just boundary repair).

This resolves the RoPE position collision bug (same tokens at different
positions reusing KV encoded at wrong positions) and the cross-chunk
attention contamination bug (KV vectors encoding the wrong preceding context).
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
    chunk_id: str               # primary key (prefix_hash for new, content_hash for legacy)
    text: str                   # original text (debugging / display)
    token_ids: List[int]        # tokenized form
    past_key_values: PastKVType # extracted KV tensors (on CPU by default)
    position_start: int         # absolute position of first token
    position_end: int           # absolute position of last token + 1
    prefix_hash: str = ""       # hash(parent_hash, tokens) — positional + contextual
    content_hash: str = ""      # hash(tokens) — content-only, position-independent
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

    cached_past_kv    : merged KV tensors for all cached tokens
    cached_length     : number of tokens covered by cached_past_kv
    live_token_ids    : tokens that still need a fresh forward pass
    live_position_ids : absolute positions for each live token
    chunk_boundaries  : list of (start, end) for each reused chunk
                        (used by SelectiveRecompute to find seam positions)
    cache_hit_ratio   : fraction of total tokens served from cache
    has_approximate   : True if any chunk was matched by content_hash
                        (not prefix_hash) — signals that full recompute
                        is needed, not just boundary repair
    """
    full_token_ids: List[int]
    cached_past_kv: Optional[PastKVType]
    cached_length: int
    live_token_ids: List[int]
    live_position_ids: List[int]
    chunk_boundaries: List[Tuple[int, int]]
    cache_hit_ratio: float
    has_approximate: bool = False

    @property
    def total_length(self) -> int:
        return len(self.full_token_ids)


@dataclass
class WarmResult:
    """Diagnostic returned by engine.warm() to help catch alignment issues."""
    chunks_stored: int
    token_count: int
    chunk_size: int
    chunk_boundary_aligned: bool
    partial_tail_tokens: int
    alignment_warning: Optional[str] = None

    def __repr__(self) -> str:
        aligned = "aligned" if self.chunk_boundary_aligned else f"partial tail={self.partial_tail_tokens}"
        return f"WarmResult(stored={self.chunks_stored}, tokens={self.token_count}, {aligned})"


# ── Hashing helpers ────────────────────────────────────────────────

def content_hash_from_tokens(token_ids: List[int]) -> str:
    """Content-only hash. Same tokens always produce the same key."""
    raw = b"".join(t.to_bytes(4, "little") for t in token_ids)
    return hashlib.sha256(raw).hexdigest()


def chained_hash(token_ids: List[int], parent_hash: Optional[str] = None) -> str:
    """
    Prefix-chained hash (vLLM-style).
    key = SHA256(parent_hash || token_bytes)

    Same tokens with different parent hashes produce different keys,
    so the same text at different positions in different conversations
    correctly gets separate cache entries.
    """
    parent = (parent_hash or "root").encode("utf-8")
    tokens = b"".join(t.to_bytes(4, "little") for t in token_ids)
    return hashlib.sha256(parent + tokens).hexdigest()


# Backwards compatibility alias
chunk_id_from_tokens = content_hash_from_tokens
