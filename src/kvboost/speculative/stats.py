# src/kvboost/speculative/stats.py

"""Per-generation acceptance bookkeeping for speculative decoding.

Tracks how many draft tokens were accepted per verification round,
exposes acceptance rate and bonus-token wins. Surfaced through
``KVBoost.speculative_stats()`` and (via the server integration) the
``/v1/stats`` endpoint, so operators can tune ``draft_k`` against actual
runtime acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class SpeculativeStats:
    """Running totals across all verification rounds of a session.

    Counters are cheap to update (one ``+=`` per round) and the summary is
    computed on demand. Reset between generations is optional — leaving
    them accumulating gives lifetime stats per engine instance.
    """

    # Number of verification rounds (i.e. target-model multi-token forwards)
    rounds: int = 0

    # Sum of accepted_count across rounds. accepted_count is in [0, K].
    accepted_total: int = 0

    # Sum of (accepted_count + 1) — the committed tokens per round, since
    # each round commits ``accepted + 1`` (the correction OR bonus token).
    committed_total: int = 0

    # Rounds where all K draft tokens were accepted (bonus-token rounds).
    bonus_rounds: int = 0

    # Theoretical "tokens we would have generated under non-speculative
    # baseline given the same wall-clock budget" — tracked by counting
    # target-model forward passes (each non-spec forward = 1 token).
    target_forwards: int = 0

    # Number of single-token draft forwards. Each round contributes
    # ``draft_k`` of these. Lets us check whether draft is dominating.
    draft_forwards: int = 0

    # Cumulative wall time inside ``draft.draft()`` and ``verifier.verify()``
    # across all rounds, in seconds. Difference between
    # (total_decode_time) and (draft + verify) is overhead in the engine
    # loop itself (sampler, rollback, list bookkeeping).
    draft_time_s: float = 0.0
    verify_time_s: float = 0.0
    rollback_time_s: float = 0.0

    # Histogram of acceptance counts per round; index i = times we saw
    # accepted_count == i. Allocated lazily because draft_k may vary.
    _hist: List[int] = field(default_factory=list)

    def record_round(
        self,
        accepted_count: int,
        draft_k: int,
        *,
        draft_time_s: float = 0.0,
        verify_time_s: float = 0.0,
        rollback_time_s: float = 0.0,
    ) -> None:
        """Update counters after one verifier round."""
        if accepted_count < 0 or accepted_count > draft_k:
            raise ValueError(
                f"accepted_count={accepted_count} out of range [0, {draft_k}]"
            )
        self.rounds += 1
        self.accepted_total += accepted_count
        self.committed_total += accepted_count + 1
        self.target_forwards += 1  # one target forward per round
        self.draft_forwards += draft_k
        self.draft_time_s += draft_time_s
        self.verify_time_s += verify_time_s
        self.rollback_time_s += rollback_time_s
        if accepted_count == draft_k:
            self.bonus_rounds += 1
        # Grow histogram if needed.
        if len(self._hist) <= draft_k:
            self._hist.extend([0] * (draft_k + 1 - len(self._hist)))
        self._hist[accepted_count] += 1

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that were accepted.

        Denominator is rounds * mean_draft_k, approximated as
        ``len(_hist) - 1`` for the most recent draft_k. For varying K
        across rounds this is an estimator, not exact.
        """
        if self.rounds == 0:
            return 0.0
        # Sum over histogram weighted by per-bucket K.
        # Simpler proxy: accepted_total / (rounds * inferred_max_k).
        max_k = max(0, len(self._hist) - 1)
        denom = self.rounds * max_k
        if denom <= 0:
            return 0.0
        return self.accepted_total / denom

    @property
    def avg_committed_per_round(self) -> float:
        """Mean tokens committed per target-model forward (the speed-up).

        Non-speculative baseline = 1.0. Higher is better; theoretical max
        is K+1 (all K accepted plus bonus)."""
        if self.rounds == 0:
            return 0.0
        return self.committed_total / self.rounds

    def summary(self) -> Dict[str, float]:
        """Dict suitable for JSON serialization in ``/v1/stats``."""
        avg_draft_ms = (
            (self.draft_time_s / self.draft_forwards) * 1000.0
            if self.draft_forwards else 0.0
        )
        avg_verify_ms = (
            (self.verify_time_s / self.target_forwards) * 1000.0
            if self.target_forwards else 0.0
        )
        avg_rollback_ms = (
            (self.rollback_time_s / self.rounds) * 1000.0
            if self.rounds else 0.0
        )
        return {
            "rounds": self.rounds,
            "accepted_total": self.accepted_total,
            "committed_total": self.committed_total,
            "bonus_rounds": self.bonus_rounds,
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_committed_per_round": round(self.avg_committed_per_round, 4),
            "draft_time_s": round(self.draft_time_s, 4),
            "verify_time_s": round(self.verify_time_s, 4),
            "rollback_time_s": round(self.rollback_time_s, 4),
            "avg_draft_ms_per_forward": round(avg_draft_ms, 3),
            "avg_verify_ms_per_forward": round(avg_verify_ms, 3),
            "avg_rollback_ms_per_round": round(avg_rollback_ms, 3),
            "histogram": list(self._hist),
        }

    def reset(self) -> None:
        self.rounds = 0
        self.accepted_total = 0
        self.committed_total = 0
        self.bonus_rounds = 0
        self.target_forwards = 0
        self.draft_forwards = 0
        self.draft_time_s = 0.0
        self.verify_time_s = 0.0
        self.rollback_time_s = 0.0
        self._hist = []
