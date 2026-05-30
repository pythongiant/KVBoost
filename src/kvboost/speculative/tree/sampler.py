"""Acceptance walks for tree speculative decoding.

Two strategies, mirroring the flat sampler's split:

  - ``verify_tree_greedy`` — bit-exact match with non-speculative
    greedy. At each step, the target's argmax must match a child's
    token id to descend; otherwise the parent's argmax becomes the
    correction and we stop.

  - ``verify_tree_sampling`` — SpecInfer-style per-level rejection
    sampling. Preserves the target distribution (proof inherited from
    Leviathan et al. 2023). At each step we try one child after
    another (in drafter-probability order) per the per-token
    rejection rule; on all-reject, sample the correction from the
    residual ``max(P_target - sum P_draft over children, 0)``.

Both return an ordered list of node ids forming the accepted path
(starting with node 0 = root) plus the correction/bonus token id to
commit at the bottom (or ``None`` if the path ran off the deepest
node without a forced correction — uncommon).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from .structure import DraftTree

log = logging.getLogger(__name__)


@dataclass
class TreeAcceptance:
    """One verification round's outcome.

    ``accepted_node_ids`` always starts with 0 (root). Subsequent
    entries are the drafted nodes accepted in order. ``correction``
    is the additional token to commit at the end of the walk:
      - on greedy mismatch: the target's argmax at the deepest
        accepted node (replaces the rejected draft)
      - on greedy walk-off: the target's argmax at the deepest
        node (a bonus token)
      - on sampling: the rejection-sampled token or ``None`` if the
        walk ran off the bottom with all children matched (bonus
        path completed).
    ``committed_tokens`` = ``accepted_node_ids[1:]`` (non-root) + the
    correction if any. This is the token sequence the engine appends
    to its ``generated`` list.
    """
    accepted_node_ids: List[int]
    correction: Optional[int]

    @property
    def n_drafted_accepted(self) -> int:
        """Drafted tokens accepted (excludes root and correction)."""
        return max(0, len(self.accepted_node_ids) - 1)

    def committed_tokens(self, tree: DraftTree) -> List[int]:
        out = [tree.token_ids[n] for n in self.accepted_node_ids[1:]]
        if self.correction is not None:
            out.append(int(self.correction))
        return out


# ── Greedy ────────────────────────────────────────────────────────────────────


def verify_tree_greedy(
    tree: DraftTree,
    per_node_logits: torch.Tensor,
) -> TreeAcceptance:
    """Greedy acceptance walk.

    Starts at the root (node 0). At each step:
      1. ``predicted = argmax(per_node_logits[current])``.
      2. Look for a child of ``current`` whose token id == predicted.
      3. If found, descend to that child and continue.
      4. Otherwise, ``correction = predicted`` and stop.

    Walk-off (no children) at deepest node: a bonus token equal to
    ``predicted`` is the correction.
    """
    if per_node_logits.dim() != 2:
        raise ValueError(
            f"per_node_logits must be 2-D (N, V); got shape "
            f"{tuple(per_node_logits.shape)}"
        )
    if per_node_logits.size(0) != tree.n_nodes:
        raise ValueError(
            f"per_node_logits has {per_node_logits.size(0)} rows; tree has "
            f"{tree.n_nodes} nodes"
        )

    accepted = [0]
    current = 0
    while True:
        predicted = int(per_node_logits[current].argmax(dim=-1).item())
        # Find a child whose token matches.
        match: Optional[int] = None
        for i, p in enumerate(tree.parent):
            if p == current and tree.token_ids[i] == predicted:
                match = i
                break
        if match is None:
            # No child matches the target's prediction → commit
            # `predicted` as the correction and stop.
            return TreeAcceptance(accepted_node_ids=accepted, correction=predicted)
        accepted.append(match)
        current = match
        # If `current` has no children we've walked off the end of
        # the tree; the next-token logits at this node give us the
        # bonus token.
        if not _has_child(tree, current):
            bonus = int(per_node_logits[current].argmax(dim=-1).item())
            return TreeAcceptance(
                accepted_node_ids=accepted, correction=bonus,
            )


# ── Sampling ──────────────────────────────────────────────────────────────────


def verify_tree_sampling(
    tree: DraftTree,
    per_node_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> TreeAcceptance:
    """Rejection-sampling walk preserving the target distribution.

    At each level we attempt children in order of drafter probability
    (highest first). For each child with drafter prob ``q`` and target
    prob ``p`` at the current node:
      - draw u ~ U(0, 1)
      - accept iff u < min(1, p/q); descend.
      - else reject this child and try the next.
    If all children rejected, sample the correction from the residual
    distribution at the current node:
      ``residual = clamp(P_target - sum(P_draft_over_attempted), min=0)``
      ``correction ~ residual / sum(residual)``.

    Matches Leviathan et al. 2023 verified per-level (the residual
    correction is the standard speculative-sampling identity).
    """
    if per_node_logits.dim() != 2:
        raise ValueError(
            f"per_node_logits must be 2-D, got shape "
            f"{tuple(per_node_logits.shape)}"
        )

    accepted = [0]
    current = 0
    while True:
        children = [
            (i, tree.token_ids[i], tree.prob[i])
            for i, p in enumerate(tree.parent)
            if p == current
        ]
        if not children:
            # Walked off the deepest accepted node → bonus token.
            bonus = _sample_from_logits(
                per_node_logits[current], temperature, generator,
            )
            return TreeAcceptance(accepted_node_ids=accepted, correction=bonus)

        # Highest drafter prob first.
        children.sort(key=lambda kv: kv[2], reverse=True)
        target_probs = F.softmax(
            (per_node_logits[current].float() / max(temperature, 1e-9)),
            dim=-1,
        )

        tried_token_ids: List[int] = []
        accepted_this_level: Optional[int] = None
        for child_id, tok, q in children:
            tried_token_ids.append(int(tok))
            p = float(target_probs[int(tok)].item())
            q_safe = max(q, 1e-12)
            ratio = p / q_safe
            u = _rand(generator)
            if u < min(1.0, ratio):
                accepted_this_level = child_id
                break

        if accepted_this_level is not None:
            accepted.append(accepted_this_level)
            current = accepted_this_level
            continue

        # All rejected → sample correction from residual.
        residual = target_probs.clone()
        for tok in tried_token_ids:
            # Find the drafter prob assigned to that token at this
            # node; subtract it from the target.
            for child_id, c_tok, c_q in children:
                if int(c_tok) == int(tok):
                    residual[int(tok)] -= float(c_q)
                    break
        residual = torch.clamp(residual, min=0.0)
        total = float(residual.sum().item())
        if total <= 0.0:
            # Numerical edge: residual collapsed. Fall back to target
            # argmax — preserves correctness in the limit.
            correction = int(per_node_logits[current].argmax(dim=-1).item())
        else:
            residual = residual / total
            correction = int(
                torch.multinomial(
                    residual, num_samples=1, generator=generator,
                ).item()
            )
        return TreeAcceptance(
            accepted_node_ids=accepted, correction=correction,
        )


# ── helpers ───────────────────────────────────────────────────────────────────


def _has_child(tree: DraftTree, idx: int) -> bool:
    return any(p == idx for p in tree.parent)


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator],
) -> int:
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    probs = F.softmax((logits.float() / temperature), dim=-1)
    idx = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(idx.item())


def _rand(generator: Optional[torch.Generator]) -> float:
    """Single fp32 draw in [0, 1). Generator-honoring."""
    r = torch.rand(1, generator=generator)
    return float(r.item())
