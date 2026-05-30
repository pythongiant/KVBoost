"""Per-request auto-selection between flat, tree, and no speculation.

Scoring formula: expected tokens per wall second, for each candidate.
Pick the highest. The framework is the same one ``tree/shape.py``
already uses for tree-shape selection; we extend it here to also
score the flat-K candidate and the no-spec baseline so the three
modes are directly comparable in the same units.

The selector is **fail-safe**: any exception or nonsense return falls
back to whichever mode is statically available (preferring tree if
configured, then flat, then none). The caller wraps ``choose(...)`` in
a try/except in ``bridge.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

from .tree.config import TreeSpeculativeConfig
from .tree.shape import AcceptanceEWMA, TreeShape, pick_shape

log = logging.getLogger(__name__)


Mode = Literal["none", "flat", "tree"]


@dataclass
class ChosenMode:
    """The selector's decision for one request.

    ``shape`` is populated only when ``mode == "tree"``. Score is the
    estimated tokens/s for the winning mode; the other candidates'
    scores are in ``alternatives`` for telemetry.
    """
    mode: Mode
    score_tokens_per_s: float
    shape: Optional[TreeShape] = None
    flat_k: Optional[int] = None
    alternatives: Optional[dict] = None
    reason: str = ""


# ── Scoring formulas ─────────────────────────────────────────────────────────


def _score_none(target_step_ms: float) -> float:
    """Baseline: one target forward per token."""
    if target_step_ms <= 0:
        return 0.0
    return 1000.0 / target_step_ms


def _score_flat(
    flat_k: int,
    flat_accept_p: float,
    *,
    target_step_ms: float,
    draft_step_ms: float,
) -> float:
    """Flat-K speculative: K draft forwards + 1 target forward per round.

    Expected committed per round = ``1 + K × accept_p`` (the +1 is the
    correction/bonus). Wall time per round = ``K × draft + target``.
    """
    if target_step_ms <= 0 or flat_k <= 0:
        return 0.0
    expected_committed = 1.0 + flat_k * max(0.0, min(1.0, flat_accept_p))
    wall_ms = flat_k * draft_step_ms + target_step_ms
    if wall_ms <= 0:
        return 0.0
    return 1000.0 * expected_committed / wall_ms


# ── Selector ──────────────────────────────────────────────────────────────────


class ModeSelector:
    """Auto mode selection. Idempotent w.r.t. config — instances are cheap.

    Construct one selector per server (it owns the EWMA used by all
    requests). ``choose()`` is called per request; it scores the three
    available modes (filtered by what's configured) and returns the
    winner.
    """

    def __init__(
        self,
        *,
        target_step_ms: float,
        draft_step_ms: float,
        flat_available: bool,
        tree_available: bool,
        tree_config: Optional[TreeSpeculativeConfig] = None,
        flat_k: int = 5,
        flat_cold_accept: float = 0.4,
        tree_ewma: Optional[AcceptanceEWMA] = None,
        cost_coefficients: Any = None,
    ):
        self.target_step_ms = float(target_step_ms)
        self.draft_step_ms = float(draft_step_ms)
        self.flat_available = bool(flat_available)
        self.tree_available = bool(tree_available)
        self.tree_config = tree_config
        self.flat_k = int(flat_k)
        self.flat_cold_accept = float(flat_cold_accept)
        self.cc = cost_coefficients
        self.tree_ewma = tree_ewma or AcceptanceEWMA(
            alpha=tree_config.ewma_alpha if tree_config else 0.2,
            cold_accept=tree_config.cold_accept if tree_config else 0.5,
        )
        # Track flat acceptance separately. We don't have a flat-k-
        # cohort EWMA yet (would need to instrument the flat engine);
        # use cold_accept as a constant for now. Future work: thread
        # through.

    # ── Policy: which modes are available? ───────────────────────────

    def _available_modes(self, policy: str) -> Tuple[bool, bool, bool]:
        """Return (none_ok, flat_ok, tree_ok) given policy + statics."""
        none_ok = True
        flat_ok = self.flat_available
        tree_ok = self.tree_available
        if policy == "auto":
            pass
        elif policy == "flat":
            tree_ok = False
            none_ok = False
        elif policy == "tree":
            flat_ok = False
            none_ok = False
        elif policy == "none":
            flat_ok = False
            tree_ok = False
        else:
            log.warning(
                "unknown policy %r; falling back to 'auto'", policy
            )
        return none_ok, flat_ok, tree_ok

    # ── Core ────────────────────────────────────────────────────────

    def choose(
        self,
        *,
        free_vram_mb: Optional[float] = None,
        policy: str = "auto",
    ) -> ChosenMode:
        """Return the best mode under current state.

        Never raises — any internal failure logs a WARN and returns
        the configured fallback. Callers should still wrap in
        try/except as a belt-and-suspenders.
        """
        try:
            return self._choose_impl(
                free_vram_mb=free_vram_mb, policy=policy,
            )
        except Exception as exc:
            log.warning(
                "ModeSelector.choose failed (%s); falling back to "
                "static-available mode", exc,
            )
            return self._static_fallback()

    def _choose_impl(
        self,
        *,
        free_vram_mb: Optional[float],
        policy: str,
    ) -> ChosenMode:
        none_ok, flat_ok, tree_ok = self._available_modes(policy)

        # Score every mode that's available; skip the others.
        scores: dict = {}

        if none_ok:
            scores["none"] = (
                _score_none(self.target_step_ms),
                {"mode": "none"},
            )

        if flat_ok:
            scores["flat"] = (
                _score_flat(
                    self.flat_k, self.flat_cold_accept,
                    target_step_ms=self.target_step_ms,
                    draft_step_ms=self.draft_step_ms,
                ),
                {"mode": "flat", "flat_k": self.flat_k},
            )

        if tree_ok and self.tree_config is not None:
            try:
                shape, tree_score = pick_shape(
                    cost_coefficients=self.cc,
                    ewma=self.tree_ewma,
                    target_step_ms=self.target_step_ms,
                    draft_step_ms=self.draft_step_ms,
                    free_vram_mb=free_vram_mb,
                    max_branching=self.tree_config.max_branching,
                    max_depth=self.tree_config.max_depth,
                    node_budget=self.tree_config.node_budget,
                    verify_extra_per_node=(
                        self.tree_config.verify_extra_per_node
                    ),
                )
                scores["tree"] = (
                    tree_score, {"mode": "tree", "shape": shape},
                )
            except Exception as exc:
                log.warning(
                    "pick_shape failed inside ModeSelector (%s); tree "
                    "candidate skipped", exc,
                )

        if not scores:
            return self._static_fallback()

        best_mode = max(scores.keys(), key=lambda k: scores[k][0])
        best_score, best_meta = scores[best_mode]
        alternatives = {
            k: round(v[0], 3) for k, v in scores.items() if k != best_mode
        }

        return ChosenMode(
            mode=best_mode,
            score_tokens_per_s=float(best_score),
            shape=best_meta.get("shape"),
            flat_k=best_meta.get("flat_k"),
            alternatives=alternatives,
            reason=f"policy={policy}, scores={scores}",
        )

    def _static_fallback(self) -> ChosenMode:
        """Conservative fallback ignoring scoring."""
        if self.tree_available and self.tree_config is not None:
            return ChosenMode(
                mode="tree",
                score_tokens_per_s=0.0,
                shape=TreeShape(
                    branching=self.tree_config.max_branching,
                    depth=self.tree_config.max_depth,
                    node_budget=self.tree_config.node_budget,
                ),
                reason="static fallback (no scoring)",
            )
        if self.flat_available:
            return ChosenMode(
                mode="flat",
                score_tokens_per_s=0.0,
                flat_k=self.flat_k,
                reason="static fallback (no scoring)",
            )
        return ChosenMode(
            mode="none", score_tokens_per_s=0.0,
            reason="static fallback (no scoring)",
        )

    def snapshot(self) -> dict:
        """For ``/v1/stats`` — serializable state."""
        return {
            "target_step_ms": self.target_step_ms,
            "draft_step_ms": self.draft_step_ms,
            "flat_available": self.flat_available,
            "tree_available": self.tree_available,
            "flat_k": self.flat_k,
            "flat_cold_accept": self.flat_cold_accept,
            "tree_ewma": self.tree_ewma.snapshot(),
        }
