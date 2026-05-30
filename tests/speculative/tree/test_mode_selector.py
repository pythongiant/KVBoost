"""ModeSelector + AcceptanceEWMA + pick_shape contracts."""

from __future__ import annotations

import pytest

from kvboost.cost_model import CostCoefficients
from kvboost.speculative.mode_selector import ChosenMode, ModeSelector
from kvboost.speculative.tree.config import TreeSpeculativeConfig
from kvboost.speculative.tree.shape import (
    AcceptanceEWMA,
    TreeShape,
    pick_shape,
)


@pytest.fixture
def cc():
    return CostCoefficients(
        total_vram_mb=12288, per_layer_mb=150, num_layers=36,
        pcie_h2d_gibps=12, hbm_bandwidth_gibps=750,
        step_latency_ms=20.0,
    )


# ── AcceptanceEWMA ────────────────────────────────────────────────────────────


def test_ewma_returns_cold_prior_before_min_samples():
    e = AcceptanceEWMA(alpha=0.2, cold_accept=0.5)
    for _ in range(15):
        e.record(2, 4, accepted=3, drafted_path_len=4,
                 committed=4, wall_ms=30)
    assert e.accept_prob(2, 4) == 0.5  # still cold


def test_ewma_uses_ema_after_min_samples():
    e = AcceptanceEWMA(alpha=0.2, cold_accept=0.5)
    for _ in range(16):
        e.record(2, 4, accepted=3, drafted_path_len=4,
                 committed=4, wall_ms=30)
    # EWMA seeded at first obs (0.75), 15 more identical obs keep it.
    p = e.accept_prob(2, 4)
    assert 0.6 < p < 0.85


def test_ewma_cohorts_are_independent():
    e = AcceptanceEWMA(alpha=0.2, cold_accept=0.5)
    for _ in range(20):
        e.record(1, 2, accepted=2, drafted_path_len=2,
                 committed=3, wall_ms=10)
    # Cohort (1,2) trained; cohort (3,5) should still be cold.
    assert e.accept_prob(1, 2) > 0.5
    assert e.accept_prob(3, 5) == 0.5


def test_ewma_record_rejects_nonsense():
    e = AcceptanceEWMA()
    # Negative inputs silently dropped; no crash, cohort stays empty.
    e.record(1, 2, accepted=-1, drafted_path_len=2,
             committed=3, wall_ms=10)
    assert e.accept_prob(1, 2) == 0.5


# ── pick_shape ────────────────────────────────────────────────────────────────


def test_pick_shape_returns_some_shape(cc):
    e = AcceptanceEWMA(cold_accept=0.5)
    shape, score = pick_shape(
        cost_coefficients=cc, ewma=e,
        target_step_ms=50.0, draft_step_ms=8.0,
        free_vram_mb=2000.0,
        max_branching=4, max_depth=6, node_budget=32,
    )
    assert isinstance(shape, TreeShape)
    assert 1 <= shape.branching <= 4
    assert 2 <= shape.depth <= 6
    assert score > 0


def test_pick_shape_picks_deeper_when_acceptance_is_high(cc):
    e = AcceptanceEWMA(cold_accept=0.5)
    # Train (1, 6) cohort to ~95% acceptance.
    for _ in range(40):
        e.record(1, 6, accepted=6, drafted_path_len=6,
                 committed=7, wall_ms=80)
    shape, _ = pick_shape(
        cost_coefficients=cc, ewma=e,
        target_step_ms=50.0, draft_step_ms=8.0,
        free_vram_mb=2000.0,
        max_branching=4, max_depth=6, node_budget=32,
    )
    # With high accept on (1,6), the linear deep tree should win.
    assert shape.branching == 1
    assert shape.depth == 6


# ── ModeSelector ──────────────────────────────────────────────────────────────


def test_mode_selector_returns_none_when_nothing_wired():
    sel = ModeSelector(
        target_step_ms=50.0, draft_step_ms=8.0,
        flat_available=False, tree_available=False,
    )
    c = sel.choose()
    assert c.mode == "none"


def test_mode_selector_picks_flat_when_only_flat_wired():
    sel = ModeSelector(
        target_step_ms=50.0, draft_step_ms=8.0,
        flat_available=True, tree_available=False,
    )
    c = sel.choose()
    assert c.mode == "flat"
    assert c.flat_k is not None


def test_mode_selector_picks_tree_when_only_tree_wired():
    sel = ModeSelector(
        target_step_ms=50.0, draft_step_ms=8.0,
        flat_available=False, tree_available=True,
        tree_config=TreeSpeculativeConfig(),
    )
    c = sel.choose()
    assert c.mode == "tree"
    assert isinstance(c.shape, TreeShape)


def test_mode_selector_policy_force_overrides_scoring(cc):
    sel = ModeSelector(
        target_step_ms=50.0, draft_step_ms=8.0,
        flat_available=True, tree_available=True,
        tree_config=TreeSpeculativeConfig(),
        cost_coefficients=cc,
    )
    assert sel.choose(policy="flat").mode == "flat"
    assert sel.choose(policy="tree").mode == "tree"
    assert sel.choose(policy="none").mode == "none"


def test_mode_selector_fail_safe_on_internal_error(monkeypatch, cc):
    sel = ModeSelector(
        target_step_ms=50.0, draft_step_ms=8.0,
        flat_available=True, tree_available=True,
        tree_config=TreeSpeculativeConfig(),
        cost_coefficients=cc,
    )

    # Inject a bad attribute that breaks scoring internals.
    def _boom(*a, **kw):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(sel, "_choose_impl", _boom)

    c = sel.choose()
    # Should fall back to the static-available mode (tree or flat).
    assert c.mode in ("tree", "flat", "none")
    # Must never crash the engine.
    assert isinstance(c, ChosenMode)
