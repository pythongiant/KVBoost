"""Tree-speculative-decoding configuration.

Sits next to (does not replace) the existing flat-spec ``SpeculativeConfig``.
The two are independent at the engine layer; ``ModeSelector`` decides which
runs per request when both are present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Force a single mode for the whole session, or let ``ModeSelector`` decide
# per request. ``auto`` only makes sense when both flat and tree are
# available.
ModePolicy = Literal["auto", "flat", "tree", "none"]


@dataclass(slots=True)
class TreeSpeculativeConfig:
    """Configuration for tree-based speculative decoding.

    Caps the search space the cost-aware ``pick_shape`` walks. Defaults
    are deliberately small — the entire tree of 32 nodes at depth 6 with
    branching up to 4 still fits comfortably in a single target forward
    on an 8B model.

    The seed prior ``cold_accept`` is the initial guess at "probability
    of accepting any one drafted token" before the ``AcceptanceEWMA``
    has enough samples to take over. 0.5 is conservative — matches
    Qwen-aligned drafter+target pairs roughly; lower it to 0.3 for
    cross-family pairs.
    """

    # Tree shape caps. Concrete shape is picked per-request by
    # ``pick_shape`` subject to these caps.
    max_branching: int = 4
    max_depth: int = 6
    node_budget: int = 32

    # Cold-start prior for ``AcceptanceEWMA`` until ≥16 observations
    # per (B, D) cohort have accumulated.
    cold_accept: float = 0.5

    # EWMA smoothing for per-cohort acceptance estimates. 0.2 gives a
    # rolling half-life of ~3 observations.
    ewma_alpha: float = 0.2

    # Per-node target verifier overhead factor. ``tree_verify_s ≈
    # step_latency_ms × (1 + N × c_extra)`` — see ``shape.py``.
    # 0.02 was the seed; on real hardware ``CalibrationTracker``-style
    # adjustment is future work.
    verify_extra_per_node: float = 0.02

    # Mode policy. ``auto`` requires both flat config and this config to
    # be present; otherwise the available one wins.
    policy: ModePolicy = "auto"

    def validate(self) -> None:
        if self.max_branching < 1:
            raise ValueError(
                f"max_branching must be >= 1, got {self.max_branching}"
            )
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.node_budget < 1:
            raise ValueError(
                f"node_budget must be >= 1, got {self.node_budget}"
            )
        if not 0.0 < self.cold_accept < 1.0:
            raise ValueError(
                f"cold_accept must be in (0, 1), got {self.cold_accept}"
            )
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError(
                f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}"
            )
        if self.verify_extra_per_node < 0.0:
            raise ValueError(
                "verify_extra_per_node must be non-negative, got "
                f"{self.verify_extra_per_node}"
            )
        if self.policy not in ("auto", "flat", "tree", "none"):
            raise ValueError(
                f"policy must be auto/flat/tree/none, got {self.policy!r}"
            )

    def summary(self) -> str:
        return (
            f"TreeSpeculativeConfig(B≤{self.max_branching}, "
            f"D≤{self.max_depth}, budget={self.node_budget}, "
            f"cold_accept={self.cold_accept}, policy={self.policy})"
        )
