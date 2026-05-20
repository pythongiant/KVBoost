"""Unit tests for kvboost.speculative.sampler.

Pure-function tests — no model loading, no GPU, no slow marker.
"""

from __future__ import annotations

import pytest
import torch

from kvboost.speculative.sampler import verify_greedy, verify_sampling


# ── verify_greedy ────────────────────────────────────────────────────────────


def _logits_with_argmax_at(positions: list[int], vocab: int) -> torch.Tensor:
    """Build a (len(positions), vocab) logits tensor where row i has its
    argmax at column positions[i]."""
    out = torch.zeros(len(positions), vocab)
    for i, pos in enumerate(positions):
        out[i, pos] = 100.0
    return out


def test_greedy_all_accept_returns_K_and_bonus():
    draft = torch.tensor([3, 7, 1, 4, 9])
    target_logits = _logits_with_argmax_at([3, 7, 1, 4, 9, 2], vocab=10)
    accepted, token = verify_greedy(draft, target_logits)
    assert accepted == 5
    assert token == 2  # bonus from position K


def test_greedy_first_reject_returns_correction():
    draft = torch.tensor([3, 7, 1, 4, 9])
    # mismatch at position 2 (target wants 8, draft proposed 1)
    target_logits = _logits_with_argmax_at([3, 7, 8, 4, 9, 0], vocab=10)
    accepted, token = verify_greedy(draft, target_logits)
    assert accepted == 2
    assert token == 8  # correction at the reject position


def test_greedy_reject_at_position_zero():
    draft = torch.tensor([3, 7, 1])
    target_logits = _logits_with_argmax_at([5, 7, 1, 9], vocab=10)
    accepted, token = verify_greedy(draft, target_logits)
    assert accepted == 0
    assert token == 5


def test_greedy_rejects_shape_errors():
    with pytest.raises(ValueError, match="draft_ids must be 1-D"):
        verify_greedy(torch.zeros(2, 3), torch.zeros(4, 5))

    with pytest.raises(ValueError, match="target_logits must be 2-D"):
        verify_greedy(torch.zeros(3), torch.zeros(4))

    # K=3 but only K rows in target logits (should be K+1=4)
    with pytest.raises(ValueError, match="K\\+1 rows"):
        verify_greedy(torch.zeros(3, dtype=torch.long), torch.zeros(3, 10))


# ── verify_sampling ──────────────────────────────────────────────────────────


def test_sampling_high_target_prob_accepts_all():
    """When target heavily favors the draft's choices, accept rate should
    be near 1."""
    torch.manual_seed(0)
    K, V = 5, 16
    draft_ids = torch.tensor([3, 7, 1, 4, 9])

    # Draft probs: one-hot on the chosen token, but with eps elsewhere so
    # divisions are well-defined.
    draft_probs = torch.full((K, V), 1e-6)
    for i in range(K):
        draft_probs[i, int(draft_ids[i])] = 1.0 - (V - 1) * 1e-6

    # Target: same tokens with very high probability.
    target_logits = torch.full((K + 1, V), -10.0)
    for i in range(K):
        target_logits[i, int(draft_ids[i])] = 10.0
    target_logits[K, 5] = 10.0

    accepted, token = verify_sampling(
        draft_ids, draft_probs, target_logits, temperature=1.0
    )
    # With these distributions, accept_prob ~= 1 for every position.
    assert accepted == K


def test_sampling_zero_target_prob_always_rejects():
    """When target gives 0 probability to draft's choice, accept_prob = 0."""
    torch.manual_seed(0)
    K, V = 3, 8
    draft_ids = torch.tensor([0, 1, 2])
    draft_probs = torch.zeros(K, V)
    for i in range(K):
        draft_probs[i, int(draft_ids[i])] = 1.0

    # Target's argmax is somewhere else with overwhelming logits; the
    # draft's token has near-zero softmax prob.
    target_logits = torch.full((K + 1, V), -100.0)
    target_logits[0, 5] = 100.0  # wants token 5 at position 0
    target_logits[1, 5] = 100.0
    target_logits[2, 5] = 100.0
    target_logits[3, 5] = 100.0

    accepted, token = verify_sampling(
        draft_ids, draft_probs, target_logits, temperature=1.0
    )
    # First reject at position 0; correction should be from residual which
    # concentrates around token 5.
    assert accepted == 0
    assert token == 5


def test_sampling_invalid_temperature_raises():
    K, V = 2, 4
    draft_ids = torch.tensor([0, 1])
    draft_probs = torch.ones(K, V) / V
    target_logits = torch.zeros(K + 1, V)

    with pytest.raises(ValueError, match="temperature must be > 0"):
        verify_sampling(draft_ids, draft_probs, target_logits, temperature=0.0)


def test_sampling_shape_validation():
    with pytest.raises(ValueError, match="draft_probs must be 2-D"):
        verify_sampling(
            torch.tensor([0]), torch.zeros(5), torch.zeros(2, 4), temperature=1.0
        )

    with pytest.raises(ValueError, match="K rows"):
        verify_sampling(
            torch.tensor([0, 1]), torch.zeros(3, 4), torch.zeros(3, 4), temperature=1.0
        )


def test_sampling_position_0_marginal_matches_target():
    """Core correctness invariant from Leviathan et al. 2023, Theorem 3.1:
    the distribution of the committed token at position 0 must exactly
    match the target distribution at position 0.

    The committed token at position 0 is:
      - draft_ids[0] when accepted_count >= 1
      - the returned correction token when accepted_count == 0

    NOTE: this is NOT the same as the distribution of verify_sampling's
    second return value alone. The return value is the *output* token —
    which is the bonus (sampled from position K) when all K were accepted,
    or the correction (residual at the reject position) otherwise. Mixing
    those two carriers gives the wrong empirical distribution.
    """
    torch.manual_seed(42)
    K, V = 1, 4
    # Target wants token 2 with prob ~0.475 at position 0
    target_logits = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    target_probs = torch.softmax(target_logits, dim=-1)
    # Draft is uniform
    draft_probs = torch.full((K, V), 1.0 / V)

    N = 10_000
    counts = torch.zeros(V)
    for _ in range(N):
        draft_token = torch.multinomial(draft_probs[0], 1).item()
        draft_ids = torch.tensor([draft_token])
        accepted, out = verify_sampling(
            draft_ids, draft_probs, target_logits, temperature=1.0
        )
        # Position-0 committed token: draft's choice if accepted, else
        # the correction from the residual distribution at position 0.
        committed_pos0 = draft_token if accepted >= 1 else out
        counts[committed_pos0] += 1

    empirical = counts / N
    diff = (empirical - target_probs[0]).abs().max().item()
    assert diff < 0.02, (
        f"empirical {empirical.tolist()} far from target "
        f"{target_probs[0].tolist()} (max diff {diff:.3f})"
    )
