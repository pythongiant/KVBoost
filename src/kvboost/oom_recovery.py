"""
CUDA-OOM recovery for KVBoost.

On ``torch.cuda.OutOfMemoryError`` (or a RuntimeError with "out of memory" in
the message), inspect *which* knob is the likely culprit and adjust it:

  * KV cache "high" (used > HIGH_FRAC × budget) → lower ``max_cache_bytes``
    and evict. Cheapest fix: trims residency without touching the model.
  * KV cache "low" → lower streaming residency (``keep_first_k`` /
    ``keep_last_k``). Layers drop out of VRAM, costing streaming overhead
    but unblocking the request.

A repeated OOM with the *same* knob just adjusted flips to the other knob,
so a single persistent bottleneck still gets fully addressed. Each knob has
a floor (``MIN_CACHE_BYTES`` / ``MIN_KEEP``); when both are exhausted the
original exception re-raises.

This module is used by:
  * the kvboost inference server (``kvboost.server.engine_worker``)
  * the benchmark runner (``benchmarks_and_experiments/sharegpt_3way/run_kvboost.py``)

so the same logic applies to single-call generation, batched dispatch, and
SSE streaming. Streaming callers pass ``can_retry`` to disable retry once a
token has been emitted (mid-stream retry would re-emit tokens to the client).
"""

from __future__ import annotations

import gc
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class Diagnosis:
    """One-shot OOM diagnosis: which tier triggered, what action to take, why.

    Tiers (ordered by preference, see policy notes in ``OOMRecovery``):
      * ``"fragmentation"`` — reserved-but-unallocated ≥ tried-alloc;
        ``empty_cache()`` alone should resolve. No knob change. Fired at most
        once per request to avoid masking real OOMs.
      * ``"cache_dominant"`` — cache holds ≥ 1.5× tried-alloc; eviction will
        free more than the failed allocation needs.
      * ``"residency_bound"`` — cache holds < 0.5× tried-alloc; cache can't
        plausibly close the gap, must lower streaming residency.
      * ``"mixed"`` — neither knob is the obvious culprit; pick the one with
        more remaining headroom (% from floor), cache wins ties (cheaper).
    """

    tier: str
    action: str  # "empty_cache_only" | "lower_cache" | "lower_streaming"
    reason: str
    parsed_oom: Dict[str, Optional[float]] = field(default_factory=dict)
    cache_used_mb: float = 0.0
    cache_budget_mb: float = 0.0
    cache_headroom_frac: float = 0.0
    stream_headroom_frac: float = 0.0


# Regexes for torch's CUDA OOM message format. Defensive: any field that
# doesn't match falls back to None — diagnosis still works on partial info.
_RE_TRIED = re.compile(r"Tried to allocate\s+([\d.]+)\s*(MiB|GiB|KiB|B)", re.IGNORECASE)
_RE_FREE = re.compile(r"([\d.]+)\s*(MiB|GiB|KiB|B)\s+is free", re.IGNORECASE)
_RE_RESERVED_UNALLOC = re.compile(
    r"([\d.]+)\s*(MiB|GiB|KiB|B)\s+is reserved by PyTorch but unallocated",
    re.IGNORECASE,
)
_RE_ALLOCATED = re.compile(
    r"([\d.]+)\s*(MiB|GiB|KiB|B)\s+is allocated by PyTorch",
    re.IGNORECASE,
)
_RE_TOTAL = re.compile(r"total capacity of\s+([\d.]+)\s*(MiB|GiB|KiB|B)", re.IGNORECASE)


def _to_mib(value: float, unit: str) -> float:
    u = unit.lower()
    if u == "gib":
        return value * 1024.0
    if u == "mib":
        return value
    if u == "kib":
        return value / 1024.0
    if u == "b":
        return value / (1024.0 ** 2)
    return value  # unknown — best-effort


def parse_oom_message(msg: str) -> Dict[str, Optional[float]]:
    """Extract numeric fields from a CUDA OOM error message. Units: MiB.

    Returns a dict with five keys; values are floats (MiB) or None if the
    field wasn't found in the message. Kept module-level so callers (and
    tests) can hit it directly without an engine instance.
    """
    out: Dict[str, Optional[float]] = {
        "tried_alloc_mb": None,
        "free_mb": None,
        "reserved_unalloc_mb": None,
        "allocated_mb": None,
        "total_capacity_mb": None,
    }
    for key, rx in (
        ("tried_alloc_mb", _RE_TRIED),
        ("free_mb", _RE_FREE),
        ("reserved_unalloc_mb", _RE_RESERVED_UNALLOC),
        ("allocated_mb", _RE_ALLOCATED),
        ("total_capacity_mb", _RE_TOTAL),
    ):
        m = rx.search(msg)
        if m:
            try:
                out[key] = _to_mib(float(m.group(1)), m.group(2))
            except (TypeError, ValueError):
                pass
    return out


class OOMRecovery:
    # Tier thresholds (multipliers against tried_alloc_mb)
    TIER_CACHE_DOMINANT = 1.5   # cache_used ≥ 1.5 × tried → tier 2
    TIER_RESIDENCY_BOUND = 0.5  # cache_used < 0.5 × tried → tier 3

    CACHE_SHRINK = 0.7
    STREAM_SHRINK = 0.5
    MIN_CACHE_BYTES = int(2.5e8)   # 250 MB floor
    MIN_KEEP = 1                   # absolute floor — at least 1 resident layer each side
    SAFETY_CAP = 16                # absolute attempt cap to avoid pathological loops

    def __init__(
        self,
        engine,
        *,
        initial_max_cache_bytes: int,
        initial_keep_first_k: Optional[int],
        initial_keep_last_k: Optional[int],
        streaming_enabled: bool,
        max_retries: Optional[int] = None,
    ):
        """Initialise recovery state.

        ``max_retries`` is a soft cap. If None, we keep retrying until either
        the call succeeds or every knob has hit its floor (capped by
        ``SAFETY_CAP`` to prevent runaway loops). Set an integer to bound it
        more tightly.
        """
        self.engine = engine
        self.max_cache_bytes = int(initial_max_cache_bytes)
        self.keep_first_k = initial_keep_first_k
        self.keep_last_k = initial_keep_last_k
        self.streaming_enabled = streaming_enabled
        self.max_retries = max_retries
        self.events: List[Dict[str, Any]] = []
        # Lifetime counter (across all calls) for fragmentation diagnoses;
        # the once-per-request gate lives inside attempt() itself.
        self._fragmentation_diagnoses = 0

    # ── Introspection ──
    def _cache_bytes_used(self) -> int:
        try:
            cm = self.engine.cache_manager
            fn = getattr(cm, "current_bytes", None)
            if callable(fn):
                return int(fn())
            if isinstance(fn, (int, float)):
                return int(fn)
            chunks = getattr(cm, "_chunks", None)
            if chunks:
                total = 0
                for c in (chunks.values() if isinstance(chunks, dict) else chunks):
                    nb = getattr(c, "nbytes", None)
                    if nb is not None:
                        total += int(nb)
                return total
        except Exception:
            pass
        return 0

    def _cache_headroom_frac(self) -> float:
        """How much room is left to shrink the cache, as a fraction of current
        budget. 0 means at floor; 1 means budget is fully shrinkable."""
        if self.max_cache_bytes <= self.MIN_CACHE_BYTES:
            return 0.0
        return (self.max_cache_bytes - self.MIN_CACHE_BYTES) / self.max_cache_bytes

    def _stream_headroom_frac(self) -> float:
        """How much room is left to shrink streaming residency. Returns -1.0
        when streaming isn't enabled at all (so mixed-tier tiebreak picks cache)."""
        if not self.streaming_enabled or not self.keep_first_k:
            return -1.0
        if self.keep_first_k <= self.MIN_KEEP:
            return 0.0
        return (self.keep_first_k - self.MIN_KEEP) / self.keep_first_k

    # ── 4-tier diagnosis ──
    def _diagnose(self, err: BaseException, *, allow_fragmentation_tier: bool) -> Diagnosis:
        """Classify an OOM into one of four tiers and pick the cheapest action.

        Tier order is fixed by reasoning, not by data — fragmentation comes
        first because it's free to fix; cache_dominant before residency_bound
        because cache eviction is faster than reshuffling resident layers.
        """
        parsed = parse_oom_message(str(err))
        cache_used_mb = self._cache_bytes_used() / (1024.0 ** 2)
        cache_budget_mb = self.max_cache_bytes / (1024.0 ** 2)
        cache_h = self._cache_headroom_frac()
        stream_h = self._stream_headroom_frac()

        tried = parsed["tried_alloc_mb"]
        frag = parsed["reserved_unalloc_mb"]

        # ── Tier 1: Fragmentation ──
        # Only fire when the message is parseable AND the caller hasn't used
        # this tier on this request yet. Repeated "fragmentation" diagnoses
        # would mask a real residency/cache OOM.
        if (allow_fragmentation_tier and tried is not None and frag is not None
                and frag >= tried):
            return Diagnosis(
                tier="fragmentation",
                action="empty_cache_only",
                reason=(
                    f"reserved-but-unallocated {frag:.0f} MiB ≥ tried-alloc "
                    f"{tried:.0f} MiB — defragmenting alone should resolve"
                ),
                parsed_oom=parsed,
                cache_used_mb=cache_used_mb,
                cache_budget_mb=cache_budget_mb,
                cache_headroom_frac=cache_h,
                stream_headroom_frac=stream_h,
            )

        # ── No tried_alloc → fall back to a simple budget-fraction heuristic ──
        # Old behaviour: if cache > HIGH_FRAC × budget, lower cache; else streaming.
        if tried is None:
            cache_dominant = cache_used_mb > 0.5 * cache_budget_mb
            action = "lower_cache" if cache_dominant or not self.streaming_enabled else "lower_streaming"
            return Diagnosis(
                tier="mixed",
                action=action,
                reason=(
                    f"OOM message unparseable; legacy heuristic: cache_used "
                    f"{cache_used_mb:.0f}/{cache_budget_mb:.0f} MiB → {action}"
                ),
                parsed_oom=parsed,
                cache_used_mb=cache_used_mb,
                cache_budget_mb=cache_budget_mb,
                cache_headroom_frac=cache_h,
                stream_headroom_frac=stream_h,
            )

        # ── Tier 2: Cache dominant ──
        if cache_used_mb >= self.TIER_CACHE_DOMINANT * tried:
            return Diagnosis(
                tier="cache_dominant",
                action="lower_cache",
                reason=(
                    f"cache_used {cache_used_mb:.0f} MiB ≥ "
                    f"{self.TIER_CACHE_DOMINANT:.1f}× tried-alloc {tried:.0f} MiB — "
                    f"eviction frees > needed (cheap, no quality hit)"
                ),
                parsed_oom=parsed,
                cache_used_mb=cache_used_mb,
                cache_budget_mb=cache_budget_mb,
                cache_headroom_frac=cache_h,
                stream_headroom_frac=stream_h,
            )

        # ── Tier 3: Residency bound ──
        if cache_used_mb < self.TIER_RESIDENCY_BOUND * tried and self.streaming_enabled:
            return Diagnosis(
                tier="residency_bound",
                action="lower_streaming",
                reason=(
                    f"cache_used {cache_used_mb:.0f} MiB < "
                    f"{self.TIER_RESIDENCY_BOUND:.1f}× tried-alloc {tried:.0f} MiB — "
                    f"cache can't close the gap, must lower residency"
                ),
                parsed_oom=parsed,
                cache_used_mb=cache_used_mb,
                cache_budget_mb=cache_budget_mb,
                cache_headroom_frac=cache_h,
                stream_headroom_frac=stream_h,
            )

        # ── Tier 4: Mixed ──
        # Cache wins ties because eviction is cheaper (and stream headroom is
        # -1.0 when streaming is off, so cache also wins by default there).
        action = "lower_streaming" if stream_h > cache_h else "lower_cache"
        return Diagnosis(
            tier="mixed",
            action=action,
            reason=(
                f"ambiguous (cache_used {cache_used_mb:.0f} MiB ≈ tried "
                f"{tried:.0f} MiB): cache_headroom={cache_h:.0%}, "
                f"stream_headroom={stream_h:.0%} → {action}"
            ),
            parsed_oom=parsed,
            cache_used_mb=cache_used_mb,
            cache_budget_mb=cache_budget_mb,
            cache_headroom_frac=cache_h,
            stream_headroom_frac=stream_h,
        )

    # ── Knob adjusters ──
    # Both must STRICTLY DECREASE the relevant value. Returning None when no
    # decrease is possible (clamped at floor, or initial value was already
    # below the floor) is the signal to the caller that this knob is exhausted.
    def _lower_cache(self) -> Optional[Dict[str, Any]]:
        old_bytes = self.max_cache_bytes
        # Apply shrink, then clamp UP to the floor so we never go below it,
        # then require strict decrease relative to old (which prevents the
        # floor clamp from ever raising the value).
        new_bytes = max(self.MIN_CACHE_BYTES, int(old_bytes * self.CACHE_SHRINK))
        if new_bytes >= old_bytes:
            return None
        cm = self.engine.cache_manager
        for attr in ("max_cache_bytes", "_max_cache_bytes", "max_bytes"):
            if hasattr(cm, attr):
                try:
                    setattr(cm, attr, new_bytes)
                except Exception:
                    pass
        self.max_cache_bytes = new_bytes
        try:
            cm.clear()
        except Exception:
            pass
        return {
            "action": "lower_cache",
            "old_max_cache_bytes": old_bytes,
            "new_max_cache_bytes": new_bytes,
        }

    def _lower_streaming(self) -> Optional[Dict[str, Any]]:
        if not self.streaming_enabled or self.keep_first_k is None:
            return None
        old_first, old_last = self.keep_first_k, (self.keep_last_k or 0)
        # Same pattern as _lower_cache: clamp up to floor, then require strict
        # decrease. If the user started below the floor (e.g. keep_first_k=9
        # with MIN_KEEP previously 16), we must not promote upward.
        new_first = max(self.MIN_KEEP, int(old_first * self.STREAM_SHRINK))
        new_last = max(self.MIN_KEEP, int(old_last * self.STREAM_SHRINK))
        # Require strict decrease on at least one of the two so we always
        # make forward progress; if both are stuck, this knob is exhausted.
        if new_first >= old_first and new_last >= old_last:
            return None
        # Clamp each side independently — never raise either value.
        new_first = min(new_first, old_first)
        new_last = min(new_last, old_last)

        applied = False
        model = self.engine.model
        for owner in (
            getattr(model, "streaming_model", None),
            model,
            getattr(model, "config", None),
        ):
            if owner is None:
                continue
            cfg = getattr(owner, "streaming_config", None) or owner
            if hasattr(cfg, "keep_first_k") and hasattr(cfg, "keep_last_k"):
                try:
                    cfg.keep_first_k = new_first
                    cfg.keep_last_k = new_last
                    applied = True
                except Exception:
                    pass
            for hook_name in (
                "rebalance_residency", "refresh_residency", "_recompute_residency"
            ):
                hook = getattr(owner, hook_name, None)
                if callable(hook):
                    try:
                        hook()
                    except Exception:
                        pass
        if not applied:
            return None

        self.keep_first_k = new_first
        self.keep_last_k = new_last
        return {
            "action": "lower_streaming",
            "old_keep_first_k": old_first,
            "old_keep_last_k": old_last,
            "keep_first_k": new_first,
            "keep_last_k": new_last,
        }

    # ── Driver ──
    def attempt(
        self,
        fn: Callable[..., Any],
        *args,
        can_retry: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> Any:
        """Call ``fn(*args, **kwargs)`` with OOM-aware retry.

        On every OOM we run ``_diagnose(err)`` to pick one of four tiers,
        then ``_apply_diagnosis(...)`` to execute it. Tier 1 (fragmentation)
        is allowed at most once per request — repeated "fragmentation"
        diagnoses would mask a real residency/cache OOM. Tiers 2-4 each
        shrink a knob (or skip when both are exhausted).

        Loop terminates when:
          * ``fn`` returns normally
          * Both knobs hit floor AND fragmentation has been used (no remaining
            recovery action)
          * ``can_retry()`` returns False (mid-stream after partial output) —
            the knob still gets adjusted for the next request, then re-raise.
          * The safety cap is reached.
        """
        import torch
        last_err: Optional[BaseException] = None
        oom_events: List[Dict[str, Any]] = []
        fragmentation_used = False  # per-call gate for tier 1

        soft_cap = self.SAFETY_CAP if self.max_retries is None else self.max_retries
        loop_cap = min(soft_cap, self.SAFETY_CAP) + 1

        attempt_idx = 0
        for attempt_idx in range(loop_cap):
            try:
                result = fn(*args, **kwargs)
                if oom_events and isinstance(result, dict):
                    bt = result.setdefault("backend_telemetry", {})
                    bt["oom_events"] = oom_events
                return result
            except torch.cuda.OutOfMemoryError as e:
                last_err = e
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" not in msg and "cuda oom" not in msg:
                    raise
                last_err = e

            diagnosis = self._diagnose(
                last_err, allow_fragmentation_tier=not fragmentation_used,
            )
            if diagnosis.tier == "fragmentation":
                fragmentation_used = True
                self._fragmentation_diagnoses += 1

            change = self._apply_diagnosis(diagnosis, attempt_idx)
            if change is None:
                # All recovery paths exhausted (knobs at floor, fragmentation already used).
                log.error(
                    "OOM recovery cannot reduce further: cache=%.2f GB (floor=%.2f GB), "
                    "keep_first_k=%s keep_last_k=%s (floor=%d), fragmentation_used=%s. "
                    "Re-raising.",
                    self.max_cache_bytes / 1e9, self.MIN_CACHE_BYTES / 1e9,
                    self.keep_first_k, self.keep_last_k, self.MIN_KEEP,
                    fragmentation_used,
                )
                break
            oom_events.append(change)

            if can_retry is not None and not can_retry():
                log.error(
                    "OOM recovery: %s — caller forbids retry "
                    "(can_retry()=False, likely mid-stream after partial output). "
                    "Re-raising; the adjustment will apply to the next request.",
                    change.get("summary") or change.get("action"),
                )
                break

        if last_err is not None and attempt_idx + 1 >= loop_cap:
            log.error(
                "OOM recovery hit attempt cap %d without recovering. "
                "cache=%.2f/%.2f GB, keep=%s/%s. "
                "Bump --oom-max-retries to keep going.",
                loop_cap, self.max_cache_bytes / 1e9,
                self.MIN_CACHE_BYTES / 1e9,
                self.keep_first_k, self.keep_last_k,
            )

        assert last_err is not None
        raise last_err

    # ── Apply a Diagnosis ──
    def _apply_diagnosis(
        self, dx: Diagnosis, attempt_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Execute the chosen tier action. Returns an event dict on success,
        or None when all recovery paths are exhausted.

        Always runs gc + empty_cache() because that's free and frequently
        unblocks the OOM. Only resets the chunk cache when actively shrinking
        the cache budget — preserves chunks on fragmentation / residency-bound
        recoveries where the cache isn't the issue.
        """
        import torch

        log.warning(
            "OOM #%d on attempt %d: tier=%s action=%s — %s",
            len(self.events) + 1, attempt_idx, dx.tier, dx.action, dx.reason,
        )

        # Always defragment — this is the "free" recovery step that often
        # resolves things without touching any knob.
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ── Tier 1: Fragmentation — empty_cache alone, no knob change ──
        if dx.action == "empty_cache_only":
            event = {
                "action": "empty_cache_only",
                "tier": dx.tier,
                "attempt": attempt_idx,
                "reason": dx.reason,
                "summary": "empty_cache() only (no knob change)",
                "parsed_oom": dx.parsed_oom,
                "cache_used_mb": dx.cache_used_mb,
                "cache_budget_mb": dx.cache_budget_mb,
            }
            self.events.append(event)
            log.warning("OOM recovery → defragment only (no knob change). Will retry.")
            return event

        # ── Tiers 2-4: knob change ──
        # For cache-action tiers, _lower_cache calls cm.clear() itself, so we
        # don't pre-emptively reset_cache (that would lose chunks before the
        # knob even applies). For residency tiers, we keep the cache intact —
        # losing it doesn't help residency pressure and gives up valuable reuse.
        primary_name = dx.action
        secondary_name = "lower_streaming" if primary_name == "lower_cache" else "lower_cache"
        primary = self._lower_cache if primary_name == "lower_cache" else self._lower_streaming
        secondary = self._lower_streaming if primary_name == "lower_cache" else self._lower_cache

        change = primary()
        flipped = False
        if change is None:
            log.warning(
                "OOM recovery: diagnosed action '%s' unavailable (floor reached or disabled); "
                "falling back to '%s'.",
                primary_name, secondary_name,
            )
            change = secondary()
            flipped = True

        if change is None:
            log.error(
                "OOM recovery EXHAUSTED at attempt %d (tier=%s): "
                "cache=%.2f GB (floor=%.2f GB), keep=%s/%s (floor=%d). "
                "Re-raising original CUDA OOM.",
                attempt_idx, dx.tier,
                self.max_cache_bytes / 1e9, self.MIN_CACHE_BYTES / 1e9,
                self.keep_first_k, self.keep_last_k, self.MIN_KEEP,
            )
            return None

        # Annotate the event with full diagnosis context.
        change["tier"] = dx.tier
        change["attempt"] = attempt_idx
        change["reason"] = dx.reason
        change["parsed_oom"] = dx.parsed_oom
        change["cache_used_mb"] = dx.cache_used_mb
        change["cache_budget_mb"] = dx.cache_budget_mb
        change["cache_headroom_frac"] = dx.cache_headroom_frac
        change["stream_headroom_frac"] = dx.stream_headroom_frac
        change["flipped_to_secondary"] = flipped
        self.events.append(change)

        if change["action"] == "lower_cache":
            summary = (
                f"lower_cache: max_cache_bytes "
                f"{change['old_max_cache_bytes'] / 1e9:.2f} GB → "
                f"{change['new_max_cache_bytes'] / 1e9:.2f} GB"
            )
            change["summary"] = summary
            log.warning(
                "OOM recovery → %s (×%.2f, tier=%s, attempt=%d). Will retry.",
                summary, self.CACHE_SHRINK, dx.tier, attempt_idx,
            )
        else:
            summary = (
                f"lower_streaming: keep_first_k {change['old_keep_first_k']}→"
                f"{change['keep_first_k']}, keep_last_k "
                f"{change['old_keep_last_k'] or 0}→{change['keep_last_k']}"
            )
            change["summary"] = summary
            log.warning(
                "OOM recovery → %s (×%.2f, tier=%s, attempt=%d). Will retry.",
                summary, self.STREAM_SHRINK, dx.tier, attempt_idx,
            )
        return change

    def snapshot(self) -> Dict[str, Any]:
        """Run-end summary, suitable for the JSON payload."""
        return {
            "n_events": len(self.events),
            "fragmentation_diagnoses": self._fragmentation_diagnoses,
            "max_cache_bytes": self.max_cache_bytes,
            "keep_first_k": self.keep_first_k,
            "keep_last_k": self.keep_last_k,
            "events": list(self.events),
        }
