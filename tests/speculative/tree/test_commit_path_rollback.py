"""``gather_kv_columns`` + ``commit_path_target_kv`` correctness.

Tests on synthetic KV tensors where each column carries an identifiable
value, so we can verify the gathered output by inspection rather than
through a full forward.
"""

from __future__ import annotations

import pytest
import torch

from kvboost.speculative.rollback import (
    gather_kv_columns,
    truncate_past_kv,
)
from kvboost.speculative.tree.rollback import (
    commit_path_target_kv,
)


def _make_marked_kv(n_layers: int, seq_len: int):
    """KV where layer ``l``'s column ``s`` is the float ``l*100 + s``.

    Lets us read which column ended up where after gather.
    """
    out = []
    for l in range(n_layers):
        col_ids = torch.arange(seq_len, dtype=torch.float32)
        # shape (batch=1, heads=1, seq, head_dim=2). Mark every head_dim
        # element with the same column id so we don't depend on
        # head/dim indexing in checks.
        k = (col_ids + 100 * l).reshape(1, 1, seq_len, 1).repeat(1, 1, 1, 2)
        v = k + 1000  # offset so K and V are distinguishable
        out.append((k, v))
    return tuple(out)


def test_truncate_only_no_tail():
    past = _make_marked_kv(n_layers=2, seq_len=10)
    result = gather_kv_columns(past, base_columns=4, tail_indices=[])
    # Equivalent to truncate_past_kv(past, 4).
    expected = truncate_past_kv(past, 4)
    for (rk, rv), (ek, ev) in zip(result, expected):
        assert torch.equal(rk, ek)
        assert torch.equal(rv, ev)


def test_gather_keeps_base_and_selected_tail():
    past = _make_marked_kv(n_layers=2, seq_len=10)
    # Keep base 0..3 (4 cols) plus tail cols at indices 2, 5 (= absolute
    # positions 4+2=6 and 4+5=9).
    result = gather_kv_columns(past, base_columns=4, tail_indices=[2, 5])
    # Result should have 4 + 2 = 6 columns; layer 0's values reflect
    # the column ids [0,1,2,3, 6, 9].
    k0 = result[0][0]
    assert k0.shape[2] == 6
    assert k0[0, 0, :, 0].tolist() == [0, 1, 2, 3, 6, 9]
    # Layer 1's values: l=1 → +100 offset.
    k1 = result[1][0]
    assert k1[0, 0, :, 0].tolist() == [100, 101, 102, 103, 106, 109]


def test_gather_rejects_out_of_range_index():
    past = _make_marked_kv(n_layers=1, seq_len=8)
    with pytest.raises(ValueError):
        gather_kv_columns(past, base_columns=4, tail_indices=[5])  # tail has 4 cols


def test_gather_negative_base_rejected():
    past = _make_marked_kv(n_layers=1, seq_len=4)
    with pytest.raises(ValueError):
        gather_kv_columns(past, base_columns=-1, tail_indices=[])


def test_commit_path_target_kv_keeps_root_and_path():
    """``commit_path_target_kv`` keeps every node on the accepted path
    INCLUDING the root. The root's column refills the slot that the
    engine's boundary-rollback dropped before verify_tree.
    """
    past = _make_marked_kv(n_layers=1, seq_len=10)
    # Imagine: committed_length=5, tree has 5 nodes at cols 5..9.
    # Accepted path: nodes [0 (root), 2, 4] — keep all 3.
    # Absolute cols: 5+0=5, 5+2=7, 5+4=9.
    result = commit_path_target_kv(
        past, committed_length=5, accepted_node_ids=[0, 2, 4],
    )
    k = result[0][0]
    assert k[0, 0, :, 0].tolist() == [0, 1, 2, 3, 4, 5, 7, 9]


def test_commit_path_target_kv_root_only_keeps_one_tail():
    """Pure-correction case: walk rejected at root → keep only root.

    This matches the flat engine's behavior: when accepted_count=0,
    the engine still commits the correction token AND keeps one KV
    column (for last_committed = root).
    """
    past = _make_marked_kv(n_layers=1, seq_len=10)
    result = commit_path_target_kv(
        past, committed_length=5, accepted_node_ids=[0],
    )
    k = result[0][0]
    # base [0..4] + root col at 5+0 = 5 → [0,1,2,3,4,5].
    assert k[0, 0, :, 0].tolist() == [0, 1, 2, 3, 4, 5]


def test_commit_path_target_kv_empty_path_truncates_to_base():
    """Edge: empty acceptance list is treated as 'commit nothing past
    committed_length'."""
    past = _make_marked_kv(n_layers=1, seq_len=10)
    result = commit_path_target_kv(
        past, committed_length=3, accepted_node_ids=[],
    )
    k = result[0][0]
    assert k[0, 0, :, 0].tolist() == [0, 1, 2]
