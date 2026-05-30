"""Acceptance walk semantics for both greedy and sampling modes."""

from __future__ import annotations

import torch

from kvboost.speculative.tree.sampler import (
    verify_tree_greedy,
    verify_tree_sampling,
)
from kvboost.speculative.tree.structure import DraftTree


def _peaky_logits(n_nodes, vocab, peaks):
    """Build per-node logits with a sharp peak at the named token per node.

    ``peaks`` is ``{node_idx: token_id}``. Other entries default to -10.
    """
    logits = torch.full((n_nodes, vocab), -10.0)
    for node, tok in peaks.items():
        logits[node, tok] = 5.0
    return logits


def _make_tree():
    # root(999)
    # ├── 10 (n1)
    # │   └── 11 (n3)
    # └── 20 (n2)
    #     └── 21 (n4)
    t = DraftTree()
    t.add_root(999)
    n1 = t.add_child(0, 10, prob=0.7)
    n2 = t.add_child(0, 20, prob=0.3)
    n3 = t.add_child(n1, 11, prob=0.8)
    n4 = t.add_child(n2, 21, prob=0.6)
    return t, (n1, n2, n3, n4)


def test_greedy_full_path_accepted_with_bonus():
    t, (n1, _n2, n3, _n4) = _make_tree()
    logits = _peaky_logits(t.n_nodes, vocab=1000, peaks={
        0: 10,    # root → predict 10 → match n1
        n1: 11,   # n1 → predict 11 → match n3
        n3: 999,  # n3 → predict 999 as bonus
    })
    r = verify_tree_greedy(t, logits)
    assert r.accepted_node_ids == [0, n1, n3]
    assert r.correction == 999
    assert r.committed_tokens(t) == [10, 11, 999]
    assert r.n_drafted_accepted == 2


def test_greedy_mismatch_at_root_commits_correction():
    t, _ = _make_tree()
    logits = _peaky_logits(t.n_nodes, vocab=1000, peaks={0: 7})
    r = verify_tree_greedy(t, logits)
    assert r.accepted_node_ids == [0]
    assert r.correction == 7
    assert r.n_drafted_accepted == 0
    assert r.committed_tokens(t) == [7]


def test_greedy_partial_accept_then_correction():
    t, (n1, _n2, _n3, _n4) = _make_tree()
    # Accept root → n1; at n1, target predicts something not in {11}.
    logits = _peaky_logits(t.n_nodes, vocab=1000, peaks={
        0: 10,
        n1: 42,   # forces a correction at depth 1
    })
    r = verify_tree_greedy(t, logits)
    assert r.accepted_node_ids == [0, n1]
    assert r.correction == 42
    assert r.committed_tokens(t) == [10, 42]


def test_greedy_only_root_no_children_returns_root_bonus():
    # Degenerate: tree is just the root. Bonus = target's argmax at root.
    t = DraftTree()
    t.add_root(999)
    logits = _peaky_logits(1, vocab=100, peaks={0: 50})
    r = verify_tree_greedy(t, logits)
    # No children → no descent → correction is parent's argmax (50).
    assert r.accepted_node_ids == [0]
    assert r.correction == 50


def test_sampling_low_temperature_matches_greedy():
    t, (n1, _n2, n3, _n4) = _make_tree()
    logits = _peaky_logits(t.n_nodes, vocab=1000, peaks={
        0: 10, n1: 11, n3: 999,
    })
    # Very low temperature → softmax is approximately one-hot → drafter
    # probs at top-1 are ~1 → acceptance ratio min(1, 1/1)=1 → always
    # accept. Should reproduce greedy path.
    gen = torch.Generator().manual_seed(0)
    r = verify_tree_sampling(t, logits, temperature=0.01, generator=gen)
    assert r.accepted_node_ids == [0, n1, n3]


def test_sampling_returns_valid_committed_tokens_shape():
    t, _ = _make_tree()
    # Uniform target → any acceptance result is fine; we only check
    # that ``committed_tokens`` returns a list of int.
    logits = torch.zeros((t.n_nodes, 50))
    gen = torch.Generator().manual_seed(7)
    r = verify_tree_sampling(t, logits, temperature=1.0, generator=gen)
    tokens = r.committed_tokens(t)
    assert isinstance(tokens, list)
    assert all(isinstance(x, int) for x in tokens)
    assert len(tokens) >= 1   # always commits at least the correction
