"""Tree drafter: wraps the existing ``DraftModel`` to produce a
``DraftTree`` by branching on per-step drafter confidence.

The existing ``DraftModel.draft(last_token, k)`` is autoregressive over
a single sequence; it shares its ``_past_kv`` across draft calls. We
need multiple branches with diverging KV state.

Strategy: do **one drafter forward per node** (preserves the existing
drafter's behavior exactly; no architecture changes). For each node
we feed the parent's token + parent's KV, take the top-B logits,
spawn B children. Per-node KV is stashed in a ``fork_registry`` so
the eventual ``commit_path`` can promote the accepted-path's deepest
KV back to ``DraftModel._past_kv``.

The branching factor B at each node is *variable* — chosen by a small
confidence heuristic: high-confidence steps get B=1 (linear), uncertain
steps get B up to ``shape.branching``. This is the cost-aware part:
we don't pay for branches where the drafter is already sure of the
right token.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..draft import DraftModel
from ..rollback import truncate_past_kv
from .shape import TreeShape
from .structure import DraftTree

log = logging.getLogger(__name__)


# ── Branching heuristic ───────────────────────────────────────────────────────


def confidence_branching(top1_prob: float, max_branching: int) -> int:
    """Pick per-node branching factor from drafter top-1 confidence.

    Bounds returned in ``[1, max_branching]``. Thresholds tuned for
    Qwen-aligned drafters: top1>0.9 ⇒ B=1 (drafter is sure, don't
    waste compute branching); 0.6-0.9 ⇒ B=2; 0.3-0.6 ⇒ B=3; below ⇒
    B=4. Cap at ``max_branching``.
    """
    if top1_prob >= 0.9:
        return 1
    if top1_prob >= 0.6:
        return min(2, max_branching)
    if top1_prob >= 0.3:
        return min(3, max_branching)
    return min(4, max_branching)


# ── KV fork helpers ───────────────────────────────────────────────────────────


def _shallow_clone_kv(past_kv: Any) -> Any:
    """Cheap clone for KV forking.

    For tuple-of-tuples: returns a new tuple containing references to
    the same tensors. Safe because we never mutate KV in place; the
    next ``truncate_past_kv`` produces fresh slices, and the next
    forward produces new tensors.

    For ``DynamicCache``: ``copy.copy`` gives a new container; the
    per-layer tensor lists are still shared until the next mutation.
    Since each branch's next op is a ``truncate`` (slice) + forward
    (new tensors), the underlying storage is shared until then.
    """
    if past_kv is None:
        return None
    if hasattr(past_kv, "get_seq_length"):
        new = copy.copy(past_kv)
        if hasattr(new, "key_cache"):
            new.key_cache = list(new.key_cache)
        if hasattr(new, "value_cache"):
            new.value_cache = list(new.value_cache)
        return new
    # tuple-of-tuples: outer tuple gets recreated, inner tensors shared.
    return tuple((k, v) for k, v in past_kv)


# ── Tree drafter ──────────────────────────────────────────────────────────────


@dataclass
class TreeDraftResult:
    """One ``draft_tree`` call's outputs."""
    tree: DraftTree
    fork_registry: Dict[int, Any]   # node_id → past_kv at the node
    n_forwards: int
    elapsed_s: float


class TreeDraftModel:
    """Wraps an existing :class:`DraftModel` to produce trees.

    Reuses everything from ``DraftModel`` for the model load, prime,
    rollback, and per-step forward. The only new logic is the
    multi-branch frontier loop and the fork registry.

    Thread-safety contract: the wrapped ``DraftModel`` is single-
    threaded (same as the rest of the engine). Tree drafting is
    serial across requests.
    """

    def __init__(self, base: DraftModel):
        self.base = base

    # ── lifecycle delegates ────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return self.base.device

    def prime(self, input_ids: torch.Tensor) -> None:
        return self.base.prime(input_ids)

    def rollback(self, keep_n: int) -> None:
        return self.base.rollback(keep_n)

    # ── tree drafting ───────────────────────────────────────────────

    @torch.no_grad()
    def draft_tree(
        self,
        *,
        last_token: int,
        shape: TreeShape,
    ) -> TreeDraftResult:
        """Build a draft tree rooted at ``last_token``.

        The base drafter's ``_past_kv`` is captured as the root's KV
        (it must already be primed to position
        ``committed_length - 1``). All forks share this prefix
        initially; each forward extends a per-node copy.

        Returns the tree + fork registry mapping node_id → its
        post-forward ``past_kv`` (the KV after the drafter has seen
        that node's token). The caller commits one path and uses the
        deepest accepted node's fork as the next round's drafter KV.
        """
        if self.base._past_kv is None:
            raise RuntimeError("TreeDraftModel.draft_tree called before prime()")

        t0 = time.perf_counter()
        tree = DraftTree()
        tree.add_root(token_id=last_token)

        # The root has no forward associated with it — its "fork" is
        # the drafter's pre-call KV state. Future draft_step's for
        # children of root will feed root's token against this KV.
        fork_registry: Dict[int, Any] = {0: _shallow_clone_kv(self.base._past_kv)}

        # BFS: frontier holds nodes whose children we still owe.
        frontier: List[int] = [0]
        n_forwards = 0

        for d in range(shape.depth):
            next_frontier: List[int] = []
            for node_idx in frontier:
                # Stop expanding if we'd blow the node budget.
                if tree.n_nodes >= shape.node_budget:
                    break

                # Draft one step from this node: feed node's token,
                # using node's KV. Get logits over V → top-B.
                parent_token = tree.token_ids[node_idx]
                parent_kv = fork_registry[node_idx]
                step_logits = self._draft_step(parent_token, parent_kv)
                # `_past_kv` after the forward represents the KV that
                # *includes* the parent's token; children inherit this.
                child_kv = _shallow_clone_kv(self.base._past_kv)
                n_forwards += 1

                # Softmax (fp32 for numerical safety on tiny probs).
                probs = F.softmax(step_logits.float(), dim=-1)
                top_p, top_i = torch.topk(probs, k=min(shape.branching, probs.numel()))
                top_p_list = top_p.tolist()
                top_i_list = top_i.tolist()

                # Pick effective B from drafter's top1 confidence.
                B = confidence_branching(top_p_list[0], shape.branching)
                for j in range(B):
                    if tree.n_nodes >= shape.node_budget:
                        break
                    child_id = tree.add_child(
                        parent_idx=node_idx,
                        token_id=top_i_list[j],
                        prob=top_p_list[j],
                    )
                    # All children of `node_idx` share the same KV
                    # (the result of feeding parent's token); they only
                    # diverge when we forward from THEM at the next
                    # depth. Sharing the dict value is safe because
                    # `truncate_past_kv` returns fresh slices.
                    fork_registry[child_id] = child_kv
                    next_frontier.append(child_id)

            frontier = self._prune_frontier(
                tree, next_frontier, fork_registry, shape.node_budget,
            )
            if not frontier:
                break

        # Restore drafter `_past_kv` to the pre-tree-build state. The
        # caller will later promote the accepted path's fork.
        self.base._past_kv = fork_registry[0]

        return TreeDraftResult(
            tree=tree,
            fork_registry=fork_registry,
            n_forwards=n_forwards,
            elapsed_s=time.perf_counter() - t0,
        )

    # ── internals ──────────────────────────────────────────────────

    def _draft_step(self, token: int, past_kv: Any) -> torch.Tensor:
        """One drafter forward, returns logits ``(V,)``.

        Sets the base drafter's ``_past_kv`` to the supplied fork so
        the model uses the right context, then captures the output's
        past_kv into the base drafter for the caller to read.
        """
        from ...compat import last_logit_only
        self.base._past_kv = past_kv
        cur_ids = torch.tensor(
            [[token]], dtype=torch.long, device=self.base.device,
        )
        with last_logit_only(self.base.model):
            out = self.base.model(
                input_ids=cur_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
        self.base._past_kv = out.past_key_values
        return out.logits[0, -1, :]   # (V,)

    @staticmethod
    def _prune_frontier(
        tree: DraftTree,
        frontier: List[int],
        fork_registry: Dict[int, Any],
        node_budget: int,
    ) -> List[int]:
        """Keep top-``budget`` frontier nodes by path log-prob.

        Always retains the best node (defensive against degenerate
        cases where every other path has -inf logprob). Removes
        pruned nodes from the fork registry so KV gets garbage-
        collected.
        """
        if tree.n_nodes <= node_budget:
            return frontier
        # Score each frontier node by cumulative log-prob.
        scored = [(tree.path_logprob[i], i) for i in frontier]
        scored.sort(key=lambda kv: kv[0], reverse=True)
        # Keep room for at least 1 child per surviving frontier node at
        # the next depth. Cap = remaining budget.
        cap = max(1, node_budget - tree.n_nodes + len(frontier))
        keep = {i for _, i in scored[:cap]}
        # Drop the discarded forks.
        for _, i in scored[cap:]:
            fork_registry.pop(i, None)
        return [i for i in frontier if i in keep]
