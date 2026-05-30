"""``DraftTree`` data structure + attention-mask construction.

The tree is a flat struct-of-arrays in BFS order. Node 0 is the root
(the synthetic ``last_committed`` token that the target has already
seen but the verifier still needs as the prefix context). Real drafted
tokens live at indices ``[1, N)``.

The attention mask is the critical correctness piece: each query
position must see only its own ancestor chain in the tree plus the
unconditional past_kv prefix. Get this wrong and you get plausible-
looking-but-mathematically-incorrect logits — the bit-exact greedy
parity test in ``tests/speculative/tree/`` is the only line of defense.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

log = logging.getLogger(__name__)


@dataclass
class DraftTree:
    """Flat struct-of-arrays representation.

    All arrays are 1-D, length ``N``. Node 0 is the root (= the
    ``last_committed`` token); its ``parent[0] == -1``. Children appear
    in BFS order so a prefix of the arrays corresponds to depth ``≤ d``.

    ``token_ids`` stays on CPU until handed to the verifier (which
    promotes it to GPU); ``parent`` / ``depth`` stay on CPU because the
    acceptance walk and mask construction are O(N) on tiny N.
    """

    token_ids: List[int] = field(default_factory=list)
    parent: List[int] = field(default_factory=list)
    depth: List[int] = field(default_factory=list)
    prob: List[float] = field(default_factory=list)
    path_logprob: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Convenience: every tree carries a synthetic root at index 0
        # holding the boundary token. Constructors should call
        # ``add_root(last_committed_id)`` immediately after init.
        pass

    # ── Mutators (build-time only) ───────────────────────────────────

    def add_root(self, token_id: int) -> int:
        """Set node 0 = root holding ``last_committed`` token id. Returns 0."""
        if self.token_ids:
            raise RuntimeError("add_root must be the first node added")
        self.token_ids.append(int(token_id))
        self.parent.append(-1)
        self.depth.append(0)
        self.prob.append(1.0)
        self.path_logprob.append(0.0)
        return 0

    def add_child(
        self, parent_idx: int, token_id: int, prob: float,
    ) -> int:
        """Append a child of ``parent_idx`` and return its new index."""
        if parent_idx < 0 or parent_idx >= len(self.token_ids):
            raise ValueError(
                f"parent_idx {parent_idx} out of range [0, "
                f"{len(self.token_ids)})"
            )
        p = max(prob, 1e-12)
        new_idx = len(self.token_ids)
        self.token_ids.append(int(token_id))
        self.parent.append(int(parent_idx))
        self.depth.append(self.depth[parent_idx] + 1)
        self.prob.append(float(prob))
        # Cumulative log-prob lets us prune by path likelihood without
        # walking the parent chain on every prune call.
        self.path_logprob.append(
            self.path_logprob[parent_idx]
            + (float(torch.log(torch.tensor(p))) if p > 0 else -1e9)
        )
        return new_idx

    # ── Accessors ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.token_ids)

    @property
    def max_depth(self) -> int:
        return max(self.depth) if self.depth else 0

    def children_of(self, idx: int) -> List[int]:
        """Walk children — O(N), only used in acceptance walk and tests."""
        return [i for i, p in enumerate(self.parent) if p == idx]

    def ancestors_of(self, idx: int) -> List[int]:
        """Return ancestor chain from root to ``idx`` (inclusive)."""
        chain: List[int] = []
        cur = idx
        while cur != -1:
            chain.append(cur)
            cur = self.parent[cur]
        return list(reversed(chain))

    def path_to(self, idx: int) -> List[int]:
        """Token ids along the path from root (exclusive) to ``idx``
        (inclusive). Convenience for committing the accepted path."""
        return [self.token_ids[i] for i in self.ancestors_of(idx)[1:]]


# ── Attention mask construction ───────────────────────────────────────────────


def build_tree_attention_mask(
    tree: DraftTree,
    *,
    committed_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the 4-D additive attention mask the target forward needs.

    Shape: ``(1, 1, N, committed_length + N)``. Values: ``0.0`` where
    attention is allowed, ``-inf`` where masked.

    Each query position ``i`` (0..N-1):
      - sees every past_kv column unconditionally (the prefix has
        already been verified and committed).
      - sees only its own ancestor chain among the N tree positions
        (including itself).

    The mask is constructed in pure Python over the tree (N ≤ ~32),
    then materialized as a single tensor. The cost is microseconds; no
    need for fancy vectorization.
    """
    N = tree.n_nodes
    if N == 0:
        raise ValueError("cannot build mask for empty tree")
    if committed_length < 0:
        raise ValueError(f"committed_length must be >= 0, got {committed_length}")

    # Precompute ancestor sets (each set includes the node itself).
    ancestors: List[List[int]] = [tree.ancestors_of(i) for i in range(N)]

    # Initialize to -inf, then punch holes where attention is allowed.
    # We use float here; the target forward will dtype-cast as needed.
    neg_inf = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    mask = torch.full(
        (N, committed_length + N), neg_inf, dtype=dtype, device=device,
    )

    # Prefix is fully visible to every query.
    if committed_length > 0:
        mask[:, :committed_length] = 0.0

    # Tree region: query i sees ancestor chain of i.
    for i in range(N):
        for a in ancestors[i]:
            mask[i, committed_length + a] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, N, committed + N)


def build_tree_position_ids(
    tree: DraftTree,
    *,
    committed_length: int,
    device: torch.device,
) -> torch.Tensor:
    """``position_ids[i] = committed_length + depth[i]``, shape ``(1, N)``.

    Each path looks causal in its own coordinate frame — RoPE sees
    each ancestor chain as a normal sequence. This is the standard
    Medusa/SpecInfer trick; no novelty here.
    """
    N = tree.n_nodes
    depths = torch.tensor(tree.depth, dtype=torch.long, device=device)
    return (depths + committed_length).unsqueeze(0)


def flatten_tree_input_ids(
    tree: DraftTree, *, device: torch.device,
) -> torch.Tensor:
    """Flat ``(1, N)`` input tensor for the verifier forward."""
    return torch.tensor(
        tree.token_ids, dtype=torch.long, device=device,
    ).unsqueeze(0)
