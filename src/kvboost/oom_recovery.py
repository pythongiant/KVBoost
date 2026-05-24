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
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class OOMRecovery:
    HIGH_FRAC = 0.5
    CACHE_SHRINK = 0.7
    STREAM_SHRINK = 0.5
    MIN_CACHE_BYTES = int(2.5e8)   # 250 MB floor
    MIN_KEEP = 16                  # at least 16 fully-resident layers each side

    def __init__(
        self,
        engine,
        *,
        initial_max_cache_bytes: int,
        initial_keep_first_k: Optional[int],
        initial_keep_last_k: Optional[int],
        streaming_enabled: bool,
        max_retries: int = 2,
    ):
        self.engine = engine
        self.max_cache_bytes = int(initial_max_cache_bytes)
        self.keep_first_k = initial_keep_first_k
        self.keep_last_k = initial_keep_last_k
        self.streaming_enabled = streaming_enabled
        self.max_retries = max_retries
        self.events: List[Dict[str, Any]] = []
        self._last_action: Optional[str] = None

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

    def _cache_high(self) -> bool:
        used = self._cache_bytes_used()
        budget = max(self.max_cache_bytes, 1)
        return used > self.HIGH_FRAC * budget

    # ── Knob adjusters ──
    def _lower_cache(self) -> Optional[Dict[str, Any]]:
        old_bytes = self.max_cache_bytes
        new_bytes = int(old_bytes * self.CACHE_SHRINK)
        if new_bytes < self.MIN_CACHE_BYTES:
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
        old_first, old_last = self.keep_first_k, self.keep_last_k
        new_first = max(self.MIN_KEEP, int(old_first * self.STREAM_SHRINK))
        new_last = max(self.MIN_KEEP, int((old_last or 0) * self.STREAM_SHRINK))
        if new_first == old_first and new_last == old_last:
            return None

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

        On CUDA OOM:
          1. Free everything we can (reset cache, gc, empty_cache).
          2. Pick a knob (cache vs streaming) based on cache occupancy.
          3. If ``can_retry`` is None or returns True, retry up to ``max_retries`` times.
             Else: adjust the knob ONCE so the *next* call benefits, then re-raise.

        The dict returned by ``fn`` is decorated with ``backend_telemetry.oom_events``
        when recovery succeeded (callers that don't return a dict are fine — telemetry
        is only attached when ``isinstance(result, dict)``).
        """
        import torch
        last_err: Optional[BaseException] = None
        oom_events: List[Dict[str, Any]] = []

        for attempt_idx in range(self.max_retries + 1):
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

            change, log_msg = self._diagnose_and_adjust(attempt_idx)
            if change is None:
                break
            oom_events.append(change)

            # Streaming consumer can't safely retry once tokens are out;
            # we still adjust the knob for the NEXT request, then re-raise.
            if can_retry is not None and not can_retry():
                log.error(
                    "OOM recovery: %s — caller forbids retry "
                    "(can_retry()=False, likely mid-stream after partial output). "
                    "Re-raising; the adjustment will apply to the next request.",
                    log_msg,
                )
                break

        assert last_err is not None
        raise last_err

    def _diagnose_and_adjust(self, attempt_idx: int):
        """Log + apply ONE knob change. Returns (change_dict, summary_str) or (None, msg)
        if both knobs are exhausted."""
        cache_high = self._cache_high()
        cache_used = self._cache_bytes_used()
        cache_frac = cache_used / max(self.max_cache_bytes, 1)
        reason_str = (
            f"cache HIGH ({cache_used / 1e9:.2f}/{self.max_cache_bytes / 1e9:.2f} GB, "
            f"{cache_frac:.0%} of budget ≥ {self.HIGH_FRAC:.0%} → cache is the suspect)"
            if cache_high else
            f"cache LOW  ({cache_used / 1e9:.2f}/{self.max_cache_bytes / 1e9:.2f} GB, "
            f"{cache_frac:.0%} of budget < {self.HIGH_FRAC:.0%} → resident layers are the suspect)"
        )
        last_action_str = (
            f"previous action was '{self._last_action}'"
            if self._last_action else "first OOM"
        )
        log.warning(
            "OOM #%d on attempt %d: %s — %s.",
            len(self.events) + 1, attempt_idx, reason_str, last_action_str,
        )

        # Free everything we can before deciding.
        try:
            self.engine.reset_cache()
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        primary_name, secondary_name = (
            ("lower_cache", "lower_streaming") if cache_high
            else ("lower_streaming", "lower_cache")
        )
        primary, secondary = (
            (self._lower_cache, self._lower_streaming)
            if cache_high
            else (self._lower_streaming, self._lower_cache)
        )
        change = primary()
        flipped_reason: Optional[str] = None
        if change is None:
            flipped_reason = f"{primary_name} unavailable (floor reached or disabled)"
        elif change.get("action") == self._last_action:
            flipped_reason = f"{primary_name} was already tried last attempt without recovering"
        if flipped_reason is not None:
            log.warning(
                "OOM recovery: skipping primary knob '%s' — %s; trying secondary '%s'.",
                primary_name, flipped_reason, secondary_name,
            )
            alt = secondary()
            if alt is not None:
                change = alt

        if change is None:
            log.error(
                "OOM recovery EXHAUSTED at attempt %d: "
                "cache=%.2f GB (floor=%.2f GB), keep=%s/%s (floor=%d). "
                "Re-raising original CUDA OOM.",
                attempt_idx,
                self.max_cache_bytes / 1e9, self.MIN_CACHE_BYTES / 1e9,
                self.keep_first_k, self.keep_last_k, self.MIN_KEEP,
            )
            return None, "knobs exhausted"

        change["attempt"] = attempt_idx
        change["reason"] = "cache_high" if cache_high else "cache_low"
        change["cache_used_gb"] = round(cache_used / 1e9, 3)
        change["cache_budget_gb"] = round(self.max_cache_bytes / 1e9, 3)
        change["flipped_to_secondary"] = flipped_reason is not None
        self.events.append(change)
        self._last_action = change["action"]

        if change["action"] == "lower_cache":
            summary = (
                f"lower_cache: max_cache_bytes "
                f"{change['old_max_cache_bytes'] / 1e9:.2f} GB → "
                f"{change['new_max_cache_bytes'] / 1e9:.2f} GB"
            )
            log.warning(
                "OOM recovery → %s (×%.2f, reason=%s, attempt=%d). Will retry.",
                summary, self.CACHE_SHRINK, change["reason"], attempt_idx,
            )
        else:
            summary = (
                f"lower_streaming: keep_first_k {change['old_keep_first_k']}→"
                f"{change['keep_first_k']}, keep_last_k "
                f"{change['old_keep_last_k'] or 0}→{change['keep_last_k']}"
            )
            log.warning(
                "OOM recovery → %s (×%.2f, reason=%s, attempt=%d). Will retry.",
                summary, self.STREAM_SHRINK, change["reason"], attempt_idx,
            )
        return change, summary

    def snapshot(self) -> Dict[str, Any]:
        """Run-end summary, suitable for the JSON payload."""
        return {
            "n_events": len(self.events),
            "max_cache_bytes": self.max_cache_bytes,
            "keep_first_k": self.keep_first_k,
            "keep_last_k": self.keep_last_k,
            "events": list(self.events),
        }
