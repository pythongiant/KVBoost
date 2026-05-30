"""Engine-level bit-exact parity test for tree speculative decoding.

The single most important correctness gate: **tree-greedy output must
match baseline greedy output byte-for-byte**, regardless of how often
the draft matches the target. This catches:

  - Attention-mask bugs (siblings leak into each other's logits).
  - Position-id bugs (RoPE sees the wrong absolute position).
  - KV gather/commit bugs (wrong column kept in the cache).
  - Acceptance-walk bugs (wrong child descended, wrong correction).
  - Boundary-token bookkeeping bugs in the engine loop.

The mock target is a *successor-table model*: position ``i``'s argmax
is ``succ_table[input_ids[0, i].item()]``. This is a closed-form
truth function — baseline greedy is just iterating the table.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest
import torch

from kvboost.speculative.stats import SpeculativeStats
from kvboost.speculative.tree.config import TreeSpeculativeConfig
from kvboost.speculative.tree.engine import TreeSpeculativeEngine
from kvboost.speculative.verifier import TargetVerifier


# ── Successor-table mock target ──────────────────────────────────────────────


class _MockModelOutput:
    """Minimal duck-type for the HF model output we read."""
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class MockTargetModel:
    """Pseudo-HF causal LM driven by a successor table.

    Implements just enough of the API for ``TreeVerifier.verify_tree``
    to run: ``.parameters()`` (for dtype detection) and ``__call__``
    accepting ``(input_ids, past_key_values, position_ids,
    attention_mask, use_cache)``. The mock ignores the mask and KV
    state — it only reads ``input_ids`` and outputs per-position
    one-hot logits at ``succ_table[token]``.

    This is intentional. The whole point of the parity test is to
    decouple model fidelity from engine bookkeeping; any acceptance
    walk + KV gather + boundary handling bug shows up against this
    closed-form ground truth.
    """

    def __init__(self, succ_table: List[int], vocab: int):
        self.succ_table = succ_table
        self.vocab = vocab
        # A real Parameter so TreeVerifier can read its dtype/device.
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        use_cache: bool = True,
    ) -> _MockModelOutput:
        # input_ids: (1, N). Per-position one-hot logits.
        N = input_ids.shape[1]
        logits = torch.full((1, N, self.vocab), -10.0)
        for i in range(N):
            tok = int(input_ids[0, i].item())
            nxt = self.succ_table[tok]
            logits[0, i, nxt] = 10.0
        # Grow fake past_kv by N positions.
        cur = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        new_past_kv = ((
            torch.zeros(1, 1, cur + N, 1),
            torch.zeros(1, 1, cur + N, 1),
        ),)
        return _MockModelOutput(logits=logits, past_key_values=new_past_kv)


# ── Drafter mock (reused from flat parity, slightly extended) ────────────────


class MockDraftModel:
    """Duck-types the real ``DraftModel`` for the tree drafter to wrap.

    Same successor-table abstraction as the flat parity mock; what
    differs is that the tree drafter pokes at ``_past_kv`` and uses the
    base drafter's ``.model.__call__`` to do per-node forwards.
    """

    def __init__(self, succ_table: List[int], vocab: int, device: str = "cpu"):
        self.succ_table = succ_table
        self.vocab = vocab
        self.device = torch.device(device)
        self._past_kv: Any = None
        self._primed_length = 0
        # The tree drafter does `base.model(input_ids=..., past_key_values=...)`
        # for per-node forwards. Reuse the successor table's mock model.
        self.model = MockTargetModel(succ_table, vocab)
        # Mock cfg attribute (tree engine reads draft.cfg.enable_kv_rollback
        # only indirectly via flat engine; tree path doesn't touch it).
        self.cfg = type("MockCfg", (), {"enable_kv_rollback": True})()

    def prime(self, input_ids: torch.Tensor) -> None:
        seq_len = int(input_ids.size(1))
        self._past_kv = ((
            torch.zeros(1, 1, seq_len, 1),
            torch.zeros(1, 1, seq_len, 1),
        ),)
        self._primed_length = seq_len

    def rollback(self, keep_n: int) -> None:
        if self._past_kv is None:
            return
        cur = self._past_kv[0][0].shape[2]
        if keep_n >= cur:
            return
        self._past_kv = ((
            torch.zeros(1, 1, keep_n, 1),
            torch.zeros(1, 1, keep_n, 1),
        ),)

    @property
    def past_kv(self):
        return self._past_kv


# ── Baseline reference ───────────────────────────────────────────────────────


def baseline_greedy(
    start_token: int,
    succ_table: List[int],
    max_new_tokens: int,
    eos_token_id: int = None,
) -> List[int]:
    out: List[int] = []
    cur = start_token
    for _ in range(max_new_tokens):
        nxt = succ_table[cur]
        out.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
        cur = nxt
    return out


# ── Engine builder ───────────────────────────────────────────────────────────


def _make_tree_engine(
    target_succ: List[int],
    draft_succ: List[int],
    vocab: int,
    *,
    max_branching: int = 2,
    max_depth: int = 4,
    node_budget: int = 16,
) -> TreeSpeculativeEngine:
    """Wire up a TreeSpeculativeEngine with mock target + drafter."""
    cfg = TreeSpeculativeConfig(
        max_branching=max_branching,
        max_depth=max_depth,
        node_budget=node_budget,
        cold_accept=0.5,
        policy="tree",
    )
    target_model = MockTargetModel(target_succ, vocab)
    # TargetVerifier wraps the model; we feed it directly.
    verifier = TargetVerifier(target_model, device="cpu")
    draft = MockDraftModel(draft_succ, vocab)
    return TreeSpeculativeEngine(
        cfg=cfg,
        target_verifier=verifier,
        draft_model=draft,
        cost_coefficients=None,    # uses defaults
        target_step_ms=10.0,
        draft_step_ms=1.0,
        mode="greedy",
        stats=SpeculativeStats(),
    )


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("B,D", [
    (1, 4),    # linear tree — degenerate, must match flat baseline
    (2, 3),    # small branching tree
    (3, 3),    # wider tree
    (2, 5),    # deeper tree
])
def test_tree_greedy_bit_exact_with_matching_draft(B, D):
    """Drafter == target. Tree-greedy output must match baseline."""
    vocab = 50
    succ = [(t * 7 + 3) % vocab for t in range(vocab)]  # arbitrary permutation
    engine = _make_tree_engine(
        target_succ=succ, draft_succ=succ, vocab=vocab,
        max_branching=B, max_depth=D, node_budget=32,
    )

    prompt = [1, 2, 3, 4, 5]
    past_kv = ((
        torch.zeros(1, 1, len(prompt), 1),
        torch.zeros(1, 1, len(prompt), 1),
    ),)
    gen, _final_kv = engine.decode_from(
        prompt_ids=prompt,
        target_past_kv=past_kv,
        cached_length=len(prompt),
        max_new_tokens=20,
    )

    expected = baseline_greedy(prompt[-1], succ, max_new_tokens=20)
    assert gen == expected, (
        f"tree greedy diverged from baseline for B={B}, D={D}\n"
        f"  got:      {gen}\n  expected: {expected}"
    )


@pytest.mark.parametrize("B,D", [(1, 4), (2, 3), (3, 4)])
def test_tree_greedy_bit_exact_with_wrong_draft(B, D):
    """Drafter completely disagrees with target. The committed sequence
    must STILL match the baseline — the drafter only affects speed."""
    vocab = 50
    target_succ = [(t * 7 + 3) % vocab for t in range(vocab)]
    # Drafter predicts arbitrary garbage that should always be wrong.
    draft_succ = [(t * 11 + 17) % vocab for t in range(vocab)]
    engine = _make_tree_engine(
        target_succ=target_succ, draft_succ=draft_succ, vocab=vocab,
        max_branching=B, max_depth=D, node_budget=32,
    )

    prompt = [10, 11, 12]
    past_kv = ((
        torch.zeros(1, 1, len(prompt), 1),
        torch.zeros(1, 1, len(prompt), 1),
    ),)
    gen, _ = engine.decode_from(
        prompt_ids=prompt,
        target_past_kv=past_kv,
        cached_length=len(prompt),
        max_new_tokens=15,
    )
    expected = baseline_greedy(prompt[-1], target_succ, max_new_tokens=15)
    assert gen == expected, (
        f"tree-greedy with bad drafter B={B} D={D}: {gen} != {expected}"
    )


def test_tree_greedy_respects_eos():
    """EOS-token id stops the loop at first emission."""
    vocab = 30
    # Build a successor table where token 5 leads to EOS (=7).
    succ = list(range(vocab))   # identity
    succ[5] = 7                 # 5 → 7 = EOS
    EOS = 7

    engine = _make_tree_engine(
        target_succ=succ, draft_succ=succ, vocab=vocab,
        max_branching=2, max_depth=3, node_budget=16,
    )
    prompt = [1, 2, 5]
    past_kv = ((
        torch.zeros(1, 1, len(prompt), 1),
        torch.zeros(1, 1, len(prompt), 1),
    ),)
    gen, _ = engine.decode_from(
        prompt_ids=prompt,
        target_past_kv=past_kv,
        cached_length=len(prompt),
        max_new_tokens=20,
        eos_token_id=EOS,
    )
    # Baseline: 5 → 7 (EOS). Loop stops here.
    assert gen == [7]


def test_tree_greedy_respects_max_new_tokens():
    vocab = 30
    succ = [(t + 1) % vocab for t in range(vocab)]   # cyclic +1
    engine = _make_tree_engine(
        target_succ=succ, draft_succ=succ, vocab=vocab,
        max_branching=2, max_depth=4, node_budget=16,
    )
    prompt = [0]
    past_kv = ((
        torch.zeros(1, 1, len(prompt), 1),
        torch.zeros(1, 1, len(prompt), 1),
    ),)
    # Tight budget that almost certainly slices mid-round.
    gen, _ = engine.decode_from(
        prompt_ids=prompt,
        target_past_kv=past_kv,
        cached_length=len(prompt),
        max_new_tokens=3,
    )
    assert len(gen) == 3
    assert gen == [1, 2, 3]
