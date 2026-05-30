# src/kvboost/speculative/bridge.py

"""Glue layer between ``KVBoost._decode_with_kv`` and the speculative
engine family (flat + tree).

This is the single function the engine calls when speculative decoding
is enabled. Kept separate from ``KVBoost`` so the dispatch is testable
in isolation against mock engines.

Cache-commit invariant
----------------------
The returned ``target_past_kv`` has length ``cached_length - 1 +
len(generated)``. KVBoost's chunk-commit path re-encodes from token IDs
rather than reading the live ``past_kv``, so the chunk cache correctness
depends only on the committed token sequence — not on the precise final
KV state. Both speculative engines guarantee the token sequence is
identical to what a non-speculative run would have produced (in greedy
mode) or distributionally equivalent (in sampling mode).

Auto mode selection
-------------------
When both flat and tree engines are configured AND a ``ModeSelector`` is
provided, the bridge consults the selector per request to choose
between them. The selector itself is fail-safe (any exception falls
back to a known-good mode); the bridge wraps the call again as a
second-line defence so a buggy selector never crashes a request.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

from .engine import SpeculativeEngine

log = logging.getLogger(__name__)


def run_speculative_decode(
    *,
    full_token_ids: List[int],
    target_past_kv: Any,
    cached_length: int,
    spec_engine: Optional[SpeculativeEngine] = None,
    tree_engine: Any = None,
    mode_selector: Any = None,
    policy: str = "auto",
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    on_token: Optional[Callable[[int], None]] = None,
    free_vram_mb: Optional[float] = None,
) -> Tuple[List[int], Any]:
    """Dispatch the right speculative engine for this request.

    Parameters
    ----------
    full_token_ids: full input prompt token ids (target prefilled on these)
    target_past_kv: post-prefill target KV cache (length == cached_length)
    cached_length: ``len(full_token_ids)`` — sanity-checked
    spec_engine: flat speculative engine, or None to disable flat mode
    tree_engine: tree speculative engine, or None to disable tree mode
    mode_selector: optional ``ModeSelector`` that picks between modes
    policy: forced policy override (``auto`` / ``flat`` / ``tree`` /
        ``none``); ignored when only one engine is available
    max_new_tokens: token budget for this call
    eos_token_id: early-stop token id
    on_token: per-COMMITTED-token callback (for streaming)
    free_vram_mb: live free VRAM, passed to the selector when known

    Returns
    -------
    (generated, target_past_kv): list of generated token ids and the
    final KV cache. Length contract matches the baseline decode loop.
    """
    if cached_length != len(full_token_ids):
        raise ValueError(
            f"cached_length ({cached_length}) != len(full_token_ids) "
            f"({len(full_token_ids)}); target must be fully prefilled"
        )

    # ── Decide which engine to invoke ──
    mode = _resolve_mode(
        spec_engine=spec_engine, tree_engine=tree_engine,
        mode_selector=mode_selector, policy=policy,
        free_vram_mb=free_vram_mb,
    )

    if mode == "tree" and tree_engine is not None:
        return tree_engine.decode_from(
            prompt_ids=full_token_ids,
            target_past_kv=target_past_kv,
            cached_length=cached_length,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            on_token=on_token,
        )

    if mode == "flat" and spec_engine is not None:
        return spec_engine.decode_from(
            prompt_ids=full_token_ids,
            target_past_kv=target_past_kv,
            cached_length=cached_length,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            on_token=on_token,
        )

    # ``mode == "none"``: no speculative engine — return empty so the
    # caller falls back to its baseline AR loop.
    return [], target_past_kv


def _resolve_mode(
    *,
    spec_engine: Any,
    tree_engine: Any,
    mode_selector: Any,
    policy: str,
    free_vram_mb: Optional[float],
) -> str:
    """Decide which mode to use for this request.

    Order of resolution:
      1. If only one engine is wired, use it.
      2. If both are wired and we have a ``ModeSelector``, ask it.
      3. Else default to flat (the historical behavior).

    Any selector failure logs at WARN and falls back to flat.
    """
    flat_ok = spec_engine is not None
    tree_ok = tree_engine is not None

    if not flat_ok and not tree_ok:
        return "none"
    if flat_ok and not tree_ok:
        return "flat" if policy != "none" else "none"
    if tree_ok and not flat_ok:
        return "tree" if policy != "none" else "none"

    if mode_selector is None:
        return "flat" if policy != "tree" else "tree"

    try:
        choice = mode_selector.choose(
            free_vram_mb=free_vram_mb, policy=policy,
        )
        # Telemetry: log the choice into stats if the selector owns one.
        stats = getattr(tree_engine, "stats", None) or getattr(spec_engine, "stats", None)
        if stats is not None and hasattr(stats, "record_mode_choice"):
            stats.record_mode_choice(choice)
        return choice.mode
    except Exception as exc:
        log.warning(
            "ModeSelector.choose raised %r; falling back to flat", exc,
        )
        return "flat"
