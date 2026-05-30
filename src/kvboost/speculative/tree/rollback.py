"""Commit-path KV reconciliation for tree speculative decoding.

After the target verifier forward, ``past_kv`` contains ``N``
speculative columns at positions ``[committed_length, committed_length
+ N)`` — one per tree node, in BFS order. The acceptance walk picks a
subset of those nodes forming a path; we need to collapse the cache so
only those columns remain, in the order they appear along the path.

For the drafter, each accepted node has a stashed ``past_kv`` in the
fork registry (the drafter walked the path one token at a time). The
deepest accepted node's KV becomes the new drafter state — every other
fork is discarded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..rollback import gather_kv_columns, truncate_past_kv

log = logging.getLogger(__name__)


def commit_path_target_kv(
    target_past_kv: Any,
    *,
    committed_length: int,
    accepted_node_ids: List[int],
) -> Any:
    """Compact target ``past_kv`` to the accepted path's columns.

    Mirrors the flat engine's KV invariant: the engine's bridge rolls
    target ``past_kv`` back by 1 column (drops ``last_committed``) before
    handing it to the verifier. The verifier then writes ``N`` columns
    — one per tree node, including the root which holds
    ``last_committed``. So the root's column refills the slot that
    rollback dropped.

    ``accepted_node_ids[0]`` is always the root (= 0). We keep ALL
    nodes on the accepted path including the root. Result seq length
    = ``committed_length + len(accepted_node_ids)``. For an
    all-rejected walk where only the root survives, this collapses to
    ``committed_length + 1`` — the same length the flat engine ends
    up at after committing only the correction.
    """
    if not accepted_node_ids:
        return truncate_past_kv(target_past_kv, committed_length)

    return gather_kv_columns(
        target_past_kv,
        base_columns=committed_length,
        tail_indices=list(accepted_node_ids),
    )


def commit_path_draft_kv(
    fork_registry: Dict[int, Any],
    *,
    accepted_node_ids: List[int],
    deepest_node_id: int,
    committed_length_after_target: int,
) -> Optional[Any]:
    """Promote the accepted path's deepest fork to be the new drafter KV.

    Mirrors the flat engine's KV bookkeeping (engine.py around line
    275). After a round, the drafter's KV should sit at
    ``max(deepest_fork_length, committed_after - 1)`` — i.e. NEVER
    truncate up. When the deepest accepted fork is shorter than the
    new committed length (which happens whenever the round commits a
    BONUS / correction token that the drafter never forwarded for),
    leave the fork at its actual length; the next round's first
    drafter forward (over root = last_committed) catches up by one.

    When the deepest accepted fork is LONGER than committed_after - 1
    (impossible under the current accept walk; defensive), truncate it
    down so the next round starts in a consistent state.

    Returns ``None`` only when the fork registry doesn't have the
    requested node — a defensive case that should not arise in normal
    operation.
    """
    if not accepted_node_ids:
        return None
    fork = fork_registry.get(deepest_node_id)
    if fork is None:
        log.warning(
            "draft fork registry missing node %d (accepted=%s); drafter "
            "will need re-prime", deepest_node_id, accepted_node_ids,
        )
        return None
    target_len = max(0, committed_length_after_target - 1)
    cur_fork_len = _kv_len(fork)
    if cur_fork_len <= target_len:
        # Don't truncate up — leave the fork as-is. The next round's
        # _draft_step(root=last_committed) extends by 1, catching up.
        return fork
    return truncate_past_kv(fork, target_len)


def _kv_len(past_kv: Any) -> int:
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):
        return int(past_kv.get_seq_length())
    return int(past_kv[0][0].shape[2])
