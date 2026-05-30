"""DraftTree + attention mask correctness.

The attention mask is the single most error-prone piece of tree
speculative decoding — a wrong mask produces plausible-looking logits
that diverge from the true conditional distribution. These tests pin
the invariants by hand on tiny trees.
"""

from __future__ import annotations

import torch

from kvboost.speculative.tree.structure import (
    DraftTree,
    build_tree_attention_mask,
    build_tree_position_ids,
    flatten_tree_input_ids,
)


def _linear_tree(tokens):
    """Helper: build a degenerate linear tree (B=1) for ``tokens``."""
    t = DraftTree()
    t.add_root(token_id=tokens[0])
    parent = 0
    for tok in tokens[1:]:
        parent = t.add_child(parent, token_id=tok, prob=1.0)
    return t


def test_root_only_tree():
    t = DraftTree()
    t.add_root(token_id=42)
    assert t.n_nodes == 1
    assert t.token_ids == [42]
    assert t.parent == [-1]
    assert t.depth == [0]
    assert t.ancestors_of(0) == [0]
    assert t.children_of(0) == []


def test_linear_tree_ancestry():
    # root(0) → 1 → 2 → 3
    t = _linear_tree([0, 1, 2, 3])
    assert t.depth == [0, 1, 2, 3]
    assert t.ancestors_of(3) == [0, 1, 2, 3]
    assert t.path_to(3) == [1, 2, 3]
    assert t.max_depth == 3


def test_branching_tree_topology():
    # Diamond shape:
    #   root
    #   ├── 10
    #   │   ├── 11
    #   │   └── 12
    #   └── 20
    #       └── 21
    t = DraftTree()
    t.add_root(token_id=999)
    n10 = t.add_child(0, 10, prob=0.6)
    n20 = t.add_child(0, 20, prob=0.4)
    n11 = t.add_child(n10, 11, prob=0.7)
    n12 = t.add_child(n10, 12, prob=0.3)
    n21 = t.add_child(n20, 21, prob=1.0)
    assert t.n_nodes == 6
    assert t.depth == [0, 1, 1, 2, 2, 2]
    assert t.children_of(0) == [n10, n20]
    assert t.children_of(n10) == [n11, n12]
    assert t.ancestors_of(n12) == [0, n10, n12]
    assert t.path_to(n11) == [10, 11]


def test_attention_mask_shape_and_prefix_visibility():
    t = _linear_tree([0, 1, 2])
    mask = build_tree_attention_mask(
        t, committed_length=4, device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 3, 4 + 3)
    # Every query sees the entire prefix unconditionally.
    assert (mask[0, 0, :, :4] == 0.0).all()


def test_attention_mask_blocks_non_ancestors():
    # Branching tree: node 3 (depth 2, parent=1) must NOT see node 2
    # (sibling subtree).
    t = DraftTree()
    t.add_root(0)
    n1 = t.add_child(0, 10, prob=0.5)
    n2 = t.add_child(0, 20, prob=0.5)
    n3 = t.add_child(n1, 11, prob=1.0)
    mask = build_tree_attention_mask(
        t, committed_length=2, device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Tree-region columns are at offset 2..5 (root, 1, 2, 3).
    tree_cols = mask[0, 0, :, 2:].clone()
    # Node 3's row: ancestors {0, 1, 3}; must be masked at column 2 (= node 2).
    # Attention masks use ``finfo.min`` (not actual -inf) to avoid NaN
    # from softmax mixing inf with finite logits — both are equivalent
    # for downstream softmax, so the assertion is "very large negative".
    assert tree_cols[n3, 0] == 0.0   # root allowed
    assert tree_cols[n3, n1] == 0.0  # parent allowed
    assert tree_cols[n3, n3] == 0.0  # self allowed
    assert tree_cols[n3, n2] < -1e30


def test_position_ids_track_depth():
    t = DraftTree()
    t.add_root(0)
    n1 = t.add_child(0, 10, prob=1.0)
    n2 = t.add_child(0, 20, prob=1.0)
    n3 = t.add_child(n1, 11, prob=1.0)
    pos = build_tree_position_ids(
        t, committed_length=5, device=torch.device("cpu"),
    )
    # Each path looks causal in its own frame: pos = committed + depth.
    assert pos.shape == (1, 4)
    assert pos[0].tolist() == [5, 6, 6, 7]


def test_flatten_tree_input_ids():
    t = DraftTree()
    t.add_root(999)
    t.add_child(0, 10, prob=1.0)
    t.add_child(0, 20, prob=1.0)
    ids = flatten_tree_input_ids(t, device=torch.device("cpu"))
    assert ids.shape == (1, 3)
    assert ids[0].tolist() == [999, 10, 20]


def test_mask_no_prefix_handled():
    # ``committed_length == 0`` is a valid edge case (very first decode
    # step with empty cache); the mask should be (1, 1, N, N).
    t = _linear_tree([1, 2, 3])
    mask = build_tree_attention_mask(
        t, committed_length=0, device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 3, 3)
    assert mask[0, 0, 0, 0] == 0.0   # root sees itself
    assert mask[0, 0, 0, 1] < -1e30  # root can't peek forward
