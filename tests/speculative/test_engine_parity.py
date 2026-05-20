"""Engine-level parity tests for speculative decoding.

These tests bypass real model loading and use deterministic mock draft +
verifier objects that duck-type the real interfaces. The mocks
parameterize the model as a "successor table" — ``succ_table[t]`` is the
deterministic next token after consuming token ``t``. This makes the
expected output a closed-form iteration: ``baseline_greedy(start, succ)``
just walks the table.

The invariant we test::

    SpeculativeEngine(target=succ_t, draft=succ_d).decode_from(prompt, K)
    == baseline_greedy(prompt[-1], succ_t)   for any succ_d

regardless of how often the draft matches the target. This is the core
correctness guarantee for greedy speculative decoding.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest
import torch

from kvboost.speculative.config import SpeculativeConfig
from kvboost.speculative.engine import SpeculativeEngine
from kvboost.speculative.stats import SpeculativeStats


# ── Mocks ───────────────────────────────────────────────────────────────────


def _fake_past_kv(seq_len: int):
    """Single-layer tuple-of-tuples KV with shape (1, 1, S, 1).

    Matches the format KVCacheManager.kv_seq_len understands, so the
    SpeculativeEngine's KV-length helpers work unchanged.
    """
    if seq_len == 0:
        return ((torch.zeros(1, 1, 0, 1), torch.zeros(1, 1, 0, 1)),)
    return ((torch.zeros(1, 1, seq_len, 1), torch.zeros(1, 1, seq_len, 1)),)


class MockDraftModel:
    """Duck-types ``speculative.draft.DraftModel``.

    Stores ``_past_kv`` as a fake length-counter tensor; ``draft`` produces
    deterministic argmax tokens from the successor table.
    """

    def __init__(self, succ_table: List[int], vocab: int, device: str = "cpu") -> None:
        self.succ_table = succ_table
        self.vocab = vocab
        self.device = device
        self._past_kv: Any = None
        self._primed_length = 0
        # Mock the cfg attribute so SpeculativeEngine.cfg.enable_kv_rollback works.
        self.cfg = type("MockCfg", (), {"enable_kv_rollback": True})()

    def prime(self, input_ids: torch.Tensor) -> None:
        seq_len = int(input_ids.size(1))
        self._past_kv = _fake_past_kv(seq_len)
        self._primed_length = seq_len

    def draft(self, last_token: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids: List[int] = []
        probs = torch.zeros(k, self.vocab)
        cur = last_token
        for i in range(k):
            cur = self.succ_table[cur]
            ids.append(cur)
            probs[i, cur] = 1.0
        # Grow fake past_kv by k positions.
        cur_len = self._past_kv[0][0].shape[2] if self._past_kv else 0
        self._past_kv = _fake_past_kv(cur_len + k)
        return torch.tensor(ids, dtype=torch.long), probs

    def rollback(self, keep_n: int) -> None:
        if self._past_kv is None:
            return
        cur_len = self._past_kv[0][0].shape[2]
        if keep_n >= cur_len:
            return
        self._past_kv = _fake_past_kv(keep_n)

    def reset(self) -> None:
        self._past_kv = None
        self._primed_length = 0

    @property
    def past_kv(self) -> Any:
        return self._past_kv

    @property
    def primed_length(self) -> int:
        return self._primed_length


class MockTargetVerifier:
    """Duck-types ``speculative.verifier.TargetVerifier``.

    Returns one-hot logits at each of the K+1 positions: row i's argmax
    is the successor of the i-th token in [last_committed] + draft_ids.
    """

    def __init__(self, succ_table: List[int], vocab: int) -> None:
        self.succ_table = succ_table
        self.vocab = vocab

    def verify(
        self,
        last_committed_token: int,
        draft_ids: torch.Tensor,
        past_kv: Any,
        committed_length: int,
    ) -> Tuple[torch.Tensor, Any]:
        k = int(draft_ids.shape[0])
        logits = torch.full((k + 1, self.vocab), -10.0)
        seq = [last_committed_token] + draft_ids.tolist()
        for i, tok in enumerate(seq):
            logits[i, self.succ_table[tok]] = 10.0
        # Grow fake past_kv by k+1.
        cur_len = past_kv[0][0].shape[2] if past_kv else 0
        new_past_kv = _fake_past_kv(cur_len + k + 1)
        return logits, new_past_kv

    def rollback(self, past_kv: Any, keep_n: int) -> Any:
        if past_kv is None:
            return None
        cur_len = past_kv[0][0].shape[2]
        if keep_n > cur_len:
            raise ValueError(
                f"keep_n={keep_n} exceeds current seq_len={cur_len}"
            )
        return _fake_past_kv(keep_n)


# ── Baseline reference ──────────────────────────────────────────────────────


def baseline_greedy(
    start_token: int,
    succ_table: List[int],
    max_new_tokens: int,
    eos_token_id: int = None,
) -> List[int]:
    """Closed-form baseline: iterate the successor table from start_token."""
    out: List[int] = []
    cur = start_token
    for _ in range(max_new_tokens):
        nxt = succ_table[cur]
        out.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
        cur = nxt
    return out


# ── Test scaffolding ────────────────────────────────────────────────────────


def _make_engine(
    target_succ: List[int],
    draft_succ: List[int],
    vocab: int,
    k: int,
    mode: str = "greedy",
) -> SpeculativeEngine:
    """Build a SpeculativeEngine with mock draft+verifier."""
    cfg = SpeculativeConfig(
        draft_model_id="mock://draft",   # validated only as non-empty
        draft_k=k,
        mode=mode,
    )
    draft = MockDraftModel(draft_succ, vocab)
    verifier = MockTargetVerifier(target_succ, vocab)
    return SpeculativeEngine(
        cfg=cfg,
        target_verifier=verifier,
        draft_model=draft,
        stats=SpeculativeStats(),
    )


def _run_speculative(
    engine: SpeculativeEngine,
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_token_id: int = None,
) -> List[int]:
    """Drive the speculative engine with a synthetic prefilled state.

    The "prefill" here just means: target_past_kv starts at length
    len(prompt_ids), draft_past_kv will be primed by the engine on entry.
    """
    target_past_kv = _fake_past_kv(len(prompt_ids))
    generated, _ = engine.decode_from(
        prompt_ids=prompt_ids,
        target_past_kv=target_past_kv,
        cached_length=len(prompt_ids),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    return generated


# ── Tests ───────────────────────────────────────────────────────────────────


def test_parity_when_draft_matches_target_exactly():
    """Draft == target: every draft token is accepted → all-bonus rounds.

    Acceptance rate should be 100% and output should match baseline exactly.
    """
    vocab = 20
    succ = [(i + 1) % vocab for i in range(vocab)]   # token i → token i+1 mod vocab
    prompt = [3, 4, 5]   # last token = 5; baseline = 6, 7, 8, ...

    engine = _make_engine(succ, succ, vocab, k=4)
    spec_out = _run_speculative(engine, prompt, max_new_tokens=12)
    expected = baseline_greedy(prompt[-1], succ, max_new_tokens=12)

    assert spec_out == expected
    # Every round should be a full accept (bonus). acceptance_rate is an
    # estimator when K varies across rounds (the last round may be
    # truncated by budget), so we check the strong, exact guarantee:
    # bonus_rounds equals total rounds.
    s = engine.stats.summary()
    assert s["bonus_rounds"] == s["rounds"]


def test_parity_when_draft_always_mispredicts():
    """Draft never matches target: every round rejects at position 0.

    Output should STILL match baseline greedy on the target, because
    the correction is always target.argmax.
    """
    vocab = 20
    target_succ = [(i + 1) % vocab for i in range(vocab)]
    # Draft predicts an offset: token i → token i+7 mod vocab
    draft_succ = [(i + 7) % vocab for i in range(vocab)]
    prompt = [3, 4, 5]

    engine = _make_engine(target_succ, draft_succ, vocab, k=4)
    spec_out = _run_speculative(engine, prompt, max_new_tokens=10)
    expected = baseline_greedy(prompt[-1], target_succ, max_new_tokens=10)

    assert spec_out == expected
    # All rounds reject at position 0.
    s = engine.stats.summary()
    assert s["accepted_total"] == 0
    assert s["bonus_rounds"] == 0


def test_parity_with_partial_match():
    """Draft matches target on even tokens, mismatches on odd.

    Exercises mid-round rejection paths. Output still equals baseline.
    """
    vocab = 30
    target_succ = [(i + 1) % vocab for i in range(vocab)]
    # Draft's succ: same as target on even tokens, off-by-one on odd
    draft_succ = list(target_succ)
    for i in range(1, vocab, 2):
        draft_succ[i] = (target_succ[i] + 1) % vocab
    prompt = [2, 3, 4]   # alternating even/odd

    engine = _make_engine(target_succ, draft_succ, vocab, k=5)
    spec_out = _run_speculative(engine, prompt, max_new_tokens=15)
    expected = baseline_greedy(prompt[-1], target_succ, max_new_tokens=15)

    assert spec_out == expected


def test_parity_stops_on_eos():
    """When target predicts EOS, the loop must stop exactly at EOS.

    The token list must end with EOS and contain no further tokens.
    """
    vocab = 10
    eos = 7
    # Target: walks 5 → 6 → 7 (EOS). Draft same as target.
    succ = list(range(1, vocab + 1))
    succ[vocab - 1] = 0
    # Force EOS path: target_succ[5] = 6, succ[6] = 7 (eos)
    succ[5] = 6
    succ[6] = eos
    prompt = [5]

    engine = _make_engine(succ, succ, vocab, k=4)
    spec_out = _run_speculative(engine, prompt, max_new_tokens=20, eos_token_id=eos)
    expected = baseline_greedy(prompt[-1], succ, max_new_tokens=20, eos_token_id=eos)

    assert spec_out == expected
    assert spec_out[-1] == eos
    # No tokens generated past EOS.
    assert eos not in spec_out[:-1]


def test_parity_respects_max_new_tokens_budget():
    """Loop must not generate more than max_new_tokens, even when a
    speculative round would commit accepted+1 tokens that overshoot."""
    vocab = 20
    succ = [(i + 1) % vocab for i in range(vocab)]
    prompt = [3]

    for budget in (1, 3, 5, 7, 11):
        engine = _make_engine(succ, succ, vocab, k=4)
        spec_out = _run_speculative(engine, prompt, max_new_tokens=budget)
        assert len(spec_out) == budget, (
            f"budget={budget}: got {len(spec_out)} tokens: {spec_out}"
        )


def test_parity_varying_k():
    """Output is invariant to the value of K; only acceptance rate /
    round count change."""
    vocab = 30
    target_succ = [(i + 1) % vocab for i in range(vocab)]
    draft_succ = [(i + 3) % vocab for i in range(vocab)]  # always wrong
    prompt = [2, 3, 4]

    expected = baseline_greedy(prompt[-1], target_succ, max_new_tokens=20)
    for k in (1, 2, 3, 5, 8):
        engine = _make_engine(target_succ, draft_succ, vocab, k=k)
        spec_out = _run_speculative(engine, prompt, max_new_tokens=20)
        assert spec_out == expected, f"k={k} broke parity"


def test_parity_single_token_prompt():
    """Edge case: prompt of length 1. We still need to roll back by 1
    to get the boundary state, which leaves an empty primed draft KV.
    Engine handles this via a fresh prime in the loop."""
    vocab = 20
    succ = [(i + 1) % vocab for i in range(vocab)]
    prompt = [5]

    engine = _make_engine(succ, succ, vocab, k=3)
    spec_out = _run_speculative(engine, prompt, max_new_tokens=6)
    expected = baseline_greedy(prompt[-1], succ, max_new_tokens=6)

    assert spec_out == expected
