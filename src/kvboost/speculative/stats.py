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

    # Histogram of acceptance counts per round; index i = times we saw
    # accepted_count == i. Allocated lazily because draft_k may vary.
    _hist: List[int] = field(default_factory=list)

    def record_round(self, accepted_count: int, draft_k: int) -> None:
        """Update counters after one verifier round."""
        if accepted_count < 0 or accepted_count > draft_k:
            raise ValueError(
                f"accepted_count={accepted_count} out of range [0, {draft_k}]"
            )
        self.rounds += 1
        self.accepted_total += accepted_count
        self.committed_total += accepted_count + 1
        self.target_forwards += 1  # one target forward per round
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
        return {
            "rounds": self.rounds,
            "accepted_total": self.accepted_total,
            "committed_total": self.committed_total,
            "bonus_rounds": self.bonus_rounds,
            "target_forwards": self.target_forwards,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_committed_per_round": round(self.avg_committed_per_round, 4),
            "histogram": list(self._hist),
        }

    def reset(self) -> None:
        self.rounds = 0
        self.accepted_total = 0
        self.committed_total = 0
        self.bonus_rounds = 0
        self.target_forwards = 0
        self._hist = []
