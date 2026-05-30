"""Tree shape selection (cost-aware) + acceptance EWMA.

``pick_shape`` walks the small grid ``B ∈ [1..max_branching] × D ∈
[2..max_depth]`` and picks the ``(B, D)`` that maximizes expected
committed-tokens per wall second, given:

  - probed hardware coefficients (``cost_model.CostCoefficients``)
  - rolling EWMA of acceptance per ``(B, D)`` cohort (this module)
  - live free-VRAM snapshot

This is the cost-aware part of SpecBlock's deployment-time adaptation,
adapted to use kvboost's existing AWQ drafter (no co-trained rank head;
we use a confidence-based branching heuristic in ``draft.py`` instead).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)


# Below this many observations we don't trust a cohort's EWMA and fall
# back to the cold-start prior. 16 is enough for ~half-life=3 EWMA to
# converge meaningfully without over-reacting to one or two outliers.
_MIN_OBSERVATIONS_FOR_EWMA = 16


@dataclass(frozen=True)
class TreeShape:
    """Concrete per-request tree shape.

    ``branching`` is the *cap* on per-node children; ``draft.py`` may
    pick a smaller B per node based on drafter confidence. ``depth`` is
    the cap on tree depth. ``node_budget`` truncates the tree to keep
    target-verifier cost bounded.
    """
    branching: int
    depth: int
    node_budget: int

    def __str__(self) -> str:
        return (
            f"TreeShape(B={self.branching}, D={self.depth}, "
            f"budget={self.node_budget})"
        )


@dataclass
class _CohortStats:
    """Per-``(B, D)`` rolling stats: acceptance + wall time.

    ``ewma_accept`` is the per-step probability of accepting any one
    drafted token along the eventual path. ``ewma_committed`` is the
    expected committed-tokens-per-round (a wall-time proxy). ``n``
    counts observations so callers can decide whether to trust the EMAs.
    """
    n: int = 0
    ewma_accept: float = 0.0
    ewma_committed: float = 0.0
    ewma_wall_ms: float = 0.0


class AcceptanceEWMA:
    """Per-``(B, D)`` exponentially-weighted moving averages.

    Single-writer (engine thread) / multi-reader (stats endpoint thread)
    safety via a single ``threading.Lock``. The lock is acquired only
    for the few-microsecond update/read; never held across IO.

    All updates are clamped to non-negative inputs. EWMA seed is the
    first observation (rather than 0) so cold-start estimates aren't
    biased toward zero.
    """

    def __init__(self, alpha: float = 0.2, cold_accept: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 < cold_accept < 1.0:
            raise ValueError(f"cold_accept must be in (0, 1), got {cold_accept}")
        self.alpha = alpha
        self.cold_accept = cold_accept
        self._cohorts: Dict[Tuple[int, int], _CohortStats] = {}
        self._lock = threading.Lock()

    def record(
        self,
        branching: int,
        depth: int,
        *,
        accepted: int,
        drafted_path_len: int,
        committed: int,
        wall_ms: float,
    ) -> None:
        """Update the ``(branching, depth)`` cohort with one round's outcome.

        ``accepted`` is the number of drafted tokens accepted along the
        chosen path (between 0 and ``drafted_path_len``). The
        per-position accept probability is ``accepted /
        max(drafted_path_len, 1)``.
        """
        if accepted < 0 or drafted_path_len < 0 or committed < 0 or wall_ms < 0:
            return  # silently ignore nonsense — selector falls back to prior
        accept_p = accepted / max(drafted_path_len, 1)
        key = (int(branching), int(depth))
        with self._lock:
            cs = self._cohorts.get(key)
            if cs is None:
                cs = _CohortStats(
                    n=1,
                    ewma_accept=accept_p,
                    ewma_committed=float(committed),
                    ewma_wall_ms=float(wall_ms),
                )
                self._cohorts[key] = cs
                return
            a = self.alpha
            cs.n += 1
            cs.ewma_accept = a * accept_p + (1.0 - a) * cs.ewma_accept
            cs.ewma_committed = a * committed + (1.0 - a) * cs.ewma_committed
            cs.ewma_wall_ms = a * wall_ms + (1.0 - a) * cs.ewma_wall_ms

    def accept_prob(self, branching: int, depth: int) -> float:
        """Best estimate of per-step accept probability for this cohort.

        Returns the cold-start prior until the cohort has accumulated
        ``_MIN_OBSERVATIONS_FOR_EWMA`` observations. After that, the EMA
        value is returned (clamped to (0, 1) for downstream safety).
        """
        key = (int(branching), int(depth))
        with self._lock:
            cs = self._cohorts.get(key)
            if cs is None or cs.n < _MIN_OBSERVATIONS_FOR_EWMA:
                return self.cold_accept
            return min(0.999, max(0.001, cs.ewma_accept))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Read-only view for ``/v1/stats`` — JSON-safe keys."""
        with self._lock:
            return {
                f"B={b},D={d}": {
                    "n": cs.n,
                    "ewma_accept": cs.ewma_accept,
                    "ewma_committed": cs.ewma_committed,
                    "ewma_wall_ms": cs.ewma_wall_ms,
                }
                for (b, d), cs in self._cohorts.items()
            }


# ── Tree shape scoring ────────────────────────────────────────────────────────


def _est_nodes(branching: int, depth: int) -> int:
    """Total nodes in a perfect ``branching``-ary tree of given depth.

    Real trees prune via confidence-based branching, but for cost
    estimation we assume the worst case (cap × cap). ``+1`` accounts for
    the synthetic root node.
    """
    if branching == 1:
        return depth + 1
    # Geometric sum: 1 + B + B² + ... + B^D = (B^(D+1) - 1) / (B - 1).
    return (branching ** (depth + 1) - 1) // (branching - 1)


def _est_committed(accept_p: float, depth: int) -> float:
    """Expected length of the accepted path (number of committed tokens).

    Random-walk model: each step has independent accept probability
    ``accept_p``. The path length is ``∑_{d=0..D} accept_p^d`` ≈
    geometric series. Add 1 for the correction/bonus token that the
    target always commits.
    """
    if accept_p <= 0.0:
        return 1.0
    if accept_p >= 1.0:
        return float(depth + 1)
    # 1 + p + p² + ... + p^D + correction
    geometric = (1.0 - accept_p ** (depth + 1)) / (1.0 - accept_p)
    return geometric


def _est_wall_s(
    n_nodes: int,
    *,
    target_step_s: float,
    draft_step_s: float,
    verify_extra_per_node: float,
) -> float:
    """Predicted wall time for one tree round, in seconds.

    Components:
      - drafter: one forward per node (existing AWQ drafter can't fuse
        them without retraining), ``n_nodes × draft_step_s``.
      - target: one forward over the flattened tree, modeled as
        ``target_step_s × (1 + N × c_extra)``. The ``c_extra`` term
        captures attention compute scaling with sequence length.
      - commit/rollback: empirically small (≤ 1 ms on 8B); rolled into
        ``c_extra`` for simplicity.
    """
    draft_total = n_nodes * draft_step_s
    target_total = target_step_s * (1.0 + n_nodes * verify_extra_per_node)
    return draft_total + target_total


def pick_shape(
    *,
    cost_coefficients,
    ewma: AcceptanceEWMA,
    target_step_ms: float,
    draft_step_ms: float,
    free_vram_mb: Optional[float],
    max_branching: int,
    max_depth: int,
    node_budget: int,
    verify_extra_per_node: float = 0.02,
) -> Tuple[TreeShape, float]:
    """Pick the ``(B, D)`` that maximizes expected tokens/s.

    Returns ``(shape, expected_tokens_per_second)`` for the winner.
    ``cost_coefficients`` is included for future extensions (currently
    only ``target_step_ms`` and ``draft_step_ms`` are consumed); pass
    the same instance probed at server start.

    The optimization is over a small grid (≤ 4 × 7 = 28 candidates) so
    we just enumerate. ``free_vram_mb`` gates shapes whose predicted
    per-tree KV would exceed 5% of remaining VRAM.
    """
    if max_branching < 1 or max_depth < 1 or node_budget < 1:
        raise ValueError(
            f"invalid caps: B≤{max_branching}, D≤{max_depth}, "
            f"budget={node_budget}"
        )

    target_step_s = target_step_ms / 1000.0
    draft_step_s = draft_step_ms / 1000.0

    # Per-tree KV size estimate. Each speculative column adds one KV
    # entry across all layers. Real values are ~100 KB/token for fp16
    # 8B-class models, ~50 KB at int8. The OOMPlanner already enforces
    # the hard memory budget; this gate is only a sanity bound to avoid
    # picking absurd tree sizes when free_vram is genuinely tight. Use
    # a conservative 1 MiB/token constant — well above realistic values.
    per_node_kv_mb = 1.0

    vram_cap_mb = (
        free_vram_mb * 0.05 if free_vram_mb is not None else None
    )

    best_score = -math.inf
    best_shape: Optional[TreeShape] = None

    # Start at D=2 (D=1 is essentially a single draft token; flat-spec
    # is the natural mode for that).
    for B in range(1, max_branching + 1):
        for D in range(2, max_depth + 1):
            n_nodes = min(_est_nodes(B, D), node_budget)
            if vram_cap_mb is not None and n_nodes * per_node_kv_mb > vram_cap_mb:
                continue
            accept_p = ewma.accept_prob(B, D)
            expected_commit = _est_committed(accept_p, D)
            wall_s = _est_wall_s(
                n_nodes,
                target_step_s=target_step_s,
                draft_step_s=draft_step_s,
                verify_extra_per_node=verify_extra_per_node,
            )
            if wall_s <= 0:
                continue
            score = expected_commit / wall_s
            if score > best_score:
                best_score = score
                best_shape = TreeShape(
                    branching=B, depth=D, node_budget=n_nodes,
                )

    if best_shape is None:
        # Every candidate was filtered (VRAM cap too tight, or all
        # walls were zero). Return the smallest viable shape so we
        # never hand back ``None`` to a caller; the auto-selector will
        # likely route to flat or none anyway.
        log.warning(
            "pick_shape found no viable candidate (free_vram=%s, "
            "node_budget=%d) — returning conservative shape (B=1,D=2)",
            free_vram_mb, node_budget,
        )
        return TreeShape(branching=1, depth=2, node_budget=3), 0.0

    return best_shape, best_score
