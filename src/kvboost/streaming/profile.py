"""Streaming forward profiler.

Process-global, opt-in region timer for the streaming pipeline. Captures
per-region wall-clock time (via ``torch.cuda.Event`` pairs when CUDA is
available, ``time.perf_counter`` otherwise) plus the change in
``torch.cuda.memory_allocated()`` across each region.

Activation:

- ``KVBOOST_PROFILE=1`` enables capture for the current process. Default
  is off — disabled, every ``region()`` call is a no-op short-circuit so
  production runs pay nothing.
- ``KVBOOST_PROFILE_OUT=/path/to/trace.jsonl`` chooses the output file
  (defaults to ``/tmp/kvboost_trace.jsonl``). ``flush()`` appends — call
  :meth:`reset` between runs if you want a clean file.

Usage::

    from kvboost.streaming.profile import get_profiler

    prof = get_profiler()
    with prof.region("qlinear.forward", layer_idx=12, sub_path="self_attn.q_proj"):
        out = qlin(x)

    # Or for split start/end (pre-hook / post-hook pattern):
    handle = prof.start("model.forward.total")
    ...
    prof.end(handle)

    prof.flush()  # called automatically by demo_partial_8b / profile_run

Each record carries an ``iteration`` counter — bump it by calling
:meth:`bump_iteration` at the top of each forward so the analyzer can
discard the first iteration (TTFT / pipeline-prime noise) and report
steady-state stats.

CUDA event timings are deferred: ``elapsed_time`` requires the event to
be reached, which would force a sync mid-forward. We append the event
pair to a pending list and resolve them all in :meth:`materialize` —
either explicitly or when :meth:`flush` is called.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator, Optional

import torch

logger = logging.getLogger(__name__)

_ENV_FLAG = "KVBOOST_PROFILE"
_ENV_OUT = "KVBOOST_PROFILE_OUT"
_DEFAULT_OUT = "/tmp/kvboost_trace.jsonl"


@dataclass
class _Record:
    name: str
    layer_idx: Optional[int]
    sub_path: Optional[str]
    iteration: int
    mem_delta_bytes: int
    dt_ms: Optional[float] = None
    # CUDA path only: events kept until materialize() resolves dt_ms.
    _start_event: Any = field(default=None, repr=False)
    _end_event: Any = field(default=None, repr=False)


class StreamingProfiler:
    """Opt-in per-region timer with CUDA-event and CPU fallbacks.

    Thread-safe for record append; not safe for concurrent flush vs
    region capture (call ``flush`` from the same thread that owned the
    forward).
    """

    _instance: Optional["StreamingProfiler"] = None
    _instance_lock = Lock()

    def __init__(self, *, enabled: bool, out_path: str) -> None:
        self.enabled = enabled
        self.out_path = out_path
        self._records: list[_Record] = []
        self._pending: list[_Record] = []
        self._iteration: int = 0
        self._cuda = torch.cuda.is_available()
        self._lock = Lock()

    # ── Singleton accessor ──────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "StreamingProfiler":
        """Return the process-global profiler, constructing it from the
        current env on first access. Re-reading env on later calls is
        intentionally NOT done — tests that want to flip the flag should
        call :meth:`_reset_singleton` first.
        """
        with cls._instance_lock:
            if cls._instance is None:
                enabled = os.environ.get(_ENV_FLAG, "0") == "1"
                out_path = os.environ.get(_ENV_OUT, _DEFAULT_OUT)
                cls._instance = cls(enabled=enabled, out_path=out_path)
            return cls._instance

    @classmethod
    def _reset_singleton(cls) -> None:
        """Test hook: drop the cached singleton so the next ``instance()``
        re-reads env. Never call from production code.
        """
        with cls._instance_lock:
            cls._instance = None

    # ── Iteration bookkeeping ───────────────────────────────────────────────

    def bump_iteration(self) -> None:
        """Increment the iteration counter. Call at the top of each
        top-level forward so the analyzer can split first-token (TTFT)
        from steady-state stats.
        """
        if not self.enabled:
            return
        self._iteration += 1

    @property
    def current_iteration(self) -> int:
        return self._iteration

    # ── Region capture ──────────────────────────────────────────────────────

    @contextmanager
    def region(
        self,
        name: str,
        *,
        layer_idx: Optional[int] = None,
        sub_path: Optional[str] = None,
    ) -> Iterator[None]:
        """Time the wrapped block. No-op when the profiler is disabled."""
        if not self.enabled:
            yield
            return

        handle = self.start(name, layer_idx=layer_idx, sub_path=sub_path)
        try:
            yield
        finally:
            self.end(handle)

    def start(
        self,
        name: str,
        *,
        layer_idx: Optional[int] = None,
        sub_path: Optional[str] = None,
    ) -> Optional[_Record]:
        """Begin a region. Returns an opaque handle to pass to :meth:`end`.

        Use this when a region spans two different hooks (pre/post) and
        can't be expressed as a single ``with`` block. When ``enabled``
        is False, returns ``None`` and :meth:`end` is a no-op.
        """
        if not self.enabled:
            return None

        mem_before = (
            torch.cuda.memory_allocated() if self._cuda else 0
        )
        rec = _Record(
            name=name,
            layer_idx=layer_idx,
            sub_path=sub_path,
            iteration=self._iteration,
            mem_delta_bytes=-mem_before,  # finalized in end()
        )
        if self._cuda:
            rec._start_event = torch.cuda.Event(enable_timing=True)
            rec._end_event = torch.cuda.Event(enable_timing=True)
            rec._start_event.record()
        else:
            # Stash CPU start time in dt_ms field temporarily; end() will
            # convert to elapsed ms.
            rec.dt_ms = time.perf_counter()
        return rec

    def end(self, handle: Optional[_Record]) -> None:
        if handle is None or not self.enabled:
            return

        if self._cuda:
            handle._end_event.record()
            mem_after = torch.cuda.memory_allocated()
            handle.mem_delta_bytes += mem_after
            with self._lock:
                self._pending.append(handle)
        else:
            t0 = handle.dt_ms or 0.0
            handle.dt_ms = (time.perf_counter() - t0) * 1000.0
            handle.mem_delta_bytes = 0
            with self._lock:
                self._records.append(handle)

    # ── Materialize / flush ─────────────────────────────────────────────────

    def materialize(self) -> None:
        """Resolve all pending CUDA event pairs into ``dt_ms`` values.

        Forces ``torch.cuda.synchronize()`` because ``elapsed_time``
        requires the end event to have been reached on the GPU. Cheap if
        the forward is already done; do NOT call mid-forward.
        """
        if not self._cuda or not self._pending:
            return
        torch.cuda.synchronize()
        with self._lock:
            for rec in self._pending:
                try:
                    rec.dt_ms = rec._start_event.elapsed_time(rec._end_event)
                except Exception as exc:  # pragma: no cover - cuda quirk
                    logger.warning(
                        "profiler: could not resolve event pair for %s: %s",
                        rec.name, exc,
                    )
                    rec.dt_ms = None
                rec._start_event = None
                rec._end_event = None
                self._records.append(rec)
            self._pending.clear()

    def flush(self) -> int:
        """Materialize pending records and append all to the JSONL file.

        Returns the number of records written. Clears the in-memory
        buffer afterward so repeated calls accumulate to disk without
        re-emitting the same rows.
        """
        if not self.enabled:
            return 0
        self.materialize()
        with self._lock:
            records = self._records
            self._records = []
        if not records:
            return 0

        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)) or ".", exist_ok=True)
        with open(self.out_path, "a") as f:
            for rec in records:
                f.write(
                    json.dumps(
                        {
                            "iteration": rec.iteration,
                            "name": rec.name,
                            "layer_idx": rec.layer_idx,
                            "sub_path": rec.sub_path,
                            "dt_ms": rec.dt_ms,
                            "mem_delta_bytes": rec.mem_delta_bytes,
                        }
                    )
                    + "\n"
                )
        return len(records)

    def reset(self) -> None:
        """Drop all in-memory records and delete the output file.

        Call before a measurement run so the JSONL only contains records
        from the run you care about.
        """
        with self._lock:
            self._records.clear()
            self._pending.clear()
            self._iteration = 0
        if self.enabled and self.out_path and os.path.exists(self.out_path):
            try:
                os.remove(self.out_path)
            except OSError as exc:  # pragma: no cover - perm errors
                logger.warning("could not delete %s: %s", self.out_path, exc)

    def records_snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of the materialized records (for tests).

        Forces :meth:`materialize` first. Does not clear the buffer.
        """
        self.materialize()
        with self._lock:
            return [
                {
                    "iteration": r.iteration,
                    "name": r.name,
                    "layer_idx": r.layer_idx,
                    "sub_path": r.sub_path,
                    "dt_ms": r.dt_ms,
                    "mem_delta_bytes": r.mem_delta_bytes,
                }
                for r in self._records
            ]


def get_profiler() -> StreamingProfiler:
    """Return the process-global profiler singleton."""
    return StreamingProfiler.instance()


__all__ = ["StreamingProfiler", "get_profiler"]
