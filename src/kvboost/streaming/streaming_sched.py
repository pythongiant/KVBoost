"""Python-side proxy for the Rust ``StreamingSchedulerBackend``.

The actual implementation lives in [crates/kvboost_native/src/streaming_sched.rs](../../../crates/kvboost_native/src/streaming_sched.rs)
and is exposed through the ``kvboost_native`` PyO3 module. This file just
re-exports the class so callers can do::

    from kvboost.streaming.streaming_sched import StreamingSchedulerBackend

without having to know about the native module layout. If the Rust extension
hasn't been built (``maturin develop`` in ``crates/kvboost_native``), the
import below raises :class:`ImportError` with a hint.
"""

from __future__ import annotations

try:
    from kvboost_native import (  # type: ignore[import-not-found]
        SlotAssignment,
        StreamingSchedulerBackend,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kvboost_native extension not built. Run `maturin develop --release` "
        "from crates/kvboost_native to enable the Rust streaming scheduler."
    ) from exc


__all__ = ["StreamingSchedulerBackend", "SlotAssignment"]
