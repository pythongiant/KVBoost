"""Unit tests for kvboost.speculative.stats."""

from __future__ import annotations

import pytest

from kvboost.speculative.stats import SpeculativeStats


def test_initial_state_is_zero():
    s = SpeculativeStats()
    summary = s.summary()
    assert summary["rounds"] == 0
    assert summary["accepted_total"] == 0
    assert summary["committed_total"] == 0
    assert summary["bonus_rounds"] == 0
    assert summary["target_forwards"] == 0
    assert summary["acceptance_rate"] == 0.0
    assert summary["avg_committed_per_round"] == 0.0
    assert summary["histogram"] == []


def test_record_round_updates_counters():
    s = SpeculativeStats()
    s.record_round(accepted_count=3, draft_k=4)
    summary = s.summary()
    assert summary["rounds"] == 1
    assert summary["accepted_total"] == 3
    assert summary["committed_total"] == 4  # 3 + 1 correction
    assert summary["bonus_rounds"] == 0
    assert summary["target_forwards"] == 1


def test_record_full_accept_increments_bonus():
    s = SpeculativeStats()
    s.record_round(accepted_count=4, draft_k=4)
    assert s.summary()["bonus_rounds"] == 1
    assert s.summary()["committed_total"] == 5  # K + bonus


def test_avg_committed_per_round_matches_baseline_at_zero_accept():
    """When draft is always wrong, every round commits exactly 1 token —
    the same as non-speculative decode."""
    s = SpeculativeStats()
    for _ in range(10):
        s.record_round(accepted_count=0, draft_k=4)
    assert s.summary()["avg_committed_per_round"] == 1.0


def test_avg_committed_per_round_maxes_at_K_plus_1():
    """When draft always hits all K, every round commits K+1 tokens."""
    s = SpeculativeStats()
    K = 4
    for _ in range(5):
        s.record_round(accepted_count=K, draft_k=K)
    assert s.summary()["avg_committed_per_round"] == K + 1


def test_histogram_distribution():
    s = SpeculativeStats()
    s.record_round(accepted_count=4, draft_k=4)
    s.record_round(accepted_count=2, draft_k=4)
    s.record_round(accepted_count=2, draft_k=4)
    s.record_round(accepted_count=0, draft_k=4)
    hist = s.summary()["histogram"]
    # buckets 0..4
    assert hist == [1, 0, 2, 0, 1]


def test_acceptance_rate_calculation():
    s = SpeculativeStats()
    # 4 rounds with K=4 each: 4 + 2 + 0 + 3 = 9 accepted out of 16 attempts
    s.record_round(accepted_count=4, draft_k=4)
    s.record_round(accepted_count=2, draft_k=4)
    s.record_round(accepted_count=0, draft_k=4)
    s.record_round(accepted_count=3, draft_k=4)
    assert s.summary()["acceptance_rate"] == 0.5625


def test_record_round_rejects_out_of_range():
    s = SpeculativeStats()
    with pytest.raises(ValueError, match="out of range"):
        s.record_round(accepted_count=5, draft_k=4)
    with pytest.raises(ValueError, match="out of range"):
        s.record_round(accepted_count=-1, draft_k=4)


def test_reset_clears_all_state():
    s = SpeculativeStats()
    s.record_round(accepted_count=2, draft_k=4)
    s.record_round(accepted_count=3, draft_k=4)
    assert s.rounds == 2
    s.reset()
    assert s.summary()["rounds"] == 0
    assert s.summary()["histogram"] == []
