"""
Pinned host memory pool for streaming weights.

This is a thin handle-keyed wrapper over ``torch.empty(..., pin_memory=True)``
plus an optional LRU eviction policy. We deliberately do NOT call
``cudaHostAlloc`` directly: PyTorch's allocator already uses it for pinned
tensors, and going through torch keeps the tensors usable with the standard
``non_blocking=True`` ``copy_`` path the staging arena relies on.

The pool exists for two reasons:
1. Give callers an opaque integer "handle" to a tensor so dictionaries and
   the Rust scheduler can refer to weights without holding tensor refs.
2. Optionally cap pinned memory and evict least-recently-used buffers when a
   model exceeds the pinned-RAM budget (LRU pin/unpin from the plan).

Pinned memory is a system-wide resource (``ulimit -l``); evicting just means
dropping the tensor and re-staging it on next access from the safetensors
shard. Eviction is opt-in.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class PinnedMemoryPool:
    """Handle-indexed pool of pinned host tensors with optional LRU eviction.

    The pool does not allocate eagerly: it provides ``register`` to take
    ownership of an already-allocated pinned tensor and ``allocate`` as a
    convenience that creates one of a given shape/dtype. Either path returns
    an opaque ``int`` handle the caller can store wherever it likes.
    """

    def __init__(
        self,
        max_bytes: Optional[int] = None,
        *,
        lru_enabled: bool = False,
        rehydrate_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> None:
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive or None")
        if lru_enabled and max_bytes is None:
            raise ValueError("lru_enabled requires a max_bytes budget")
        if lru_enabled and rehydrate_fn is None:
            raise ValueError(
                "lru_enabled requires a rehydrate_fn so evicted handles can "
                "be reloaded on demand"
            )

        self._max_bytes = max_bytes
        self._lru_enabled = lru_enabled
        self._rehydrate_fn = rehydrate_fn

        self._lock = threading.Lock()
        self._next_handle = 1
        # OrderedDict gives us O(1) move-to-end for LRU touch.
        self._tensors: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._nbytes: dict[int, int] = {}
        self._total_bytes = 0

    # ── Allocation API ──────────────────────────────────────────────────────

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> int:
        """Allocate a pinned host tensor and return its opaque handle."""
        tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
        return self.register(tensor)

    def register(self, tensor: torch.Tensor) -> int:
        """Take ownership of an already-allocated tensor."""
        if tensor.device.type != "cpu":
            raise ValueError("PinnedMemoryPool only stores CPU tensors")
        nbytes = tensor.numel() * tensor.element_size()

        with self._lock:
            self._evict_until_fits_locked(nbytes)
            handle = self._next_handle
            self._next_handle += 1
            self._tensors[handle] = tensor
            self._nbytes[handle] = nbytes
            self._total_bytes += nbytes
            return handle

    # ── Access API ──────────────────────────────────────────────────────────

    def get(self, handle: int) -> torch.Tensor:
        """Return the tensor for ``handle``, rehydrating it if LRU-evicted."""
        with self._lock:
            tensor = self._tensors.get(handle)
            if tensor is not None:
                self._tensors.move_to_end(handle)
                return tensor

        if not self._lru_enabled or self._rehydrate_fn is None:
            raise KeyError(f"unknown pinned-pool handle: {handle}")

        # Drop the lock before invoking the user callback — it may itself
        # call back into the pool (e.g. via ``register``).
        rehydrated = self._rehydrate_fn(handle)
        if rehydrated.device.type != "cpu":
            raise ValueError("rehydrate_fn must return a CPU tensor")

        with self._lock:
            self._evict_until_fits_locked(
                rehydrated.numel() * rehydrated.element_size(),
                pinned_handle=handle,
            )
            self._tensors[handle] = rehydrated
            self._nbytes[handle] = rehydrated.numel() * rehydrated.element_size()
            self._total_bytes += self._nbytes[handle]
            return rehydrated

    def get_pointer(self, handle: int) -> int:
        """Return ``data_ptr()`` for the tensor backing ``handle``."""
        return self.get(handle).data_ptr()

    def release(self, handle: int) -> None:
        with self._lock:
            tensor = self._tensors.pop(handle, None)
            if tensor is None:
                return
            self._total_bytes -= self._nbytes.pop(handle, 0)

    # ── Introspection ───────────────────────────────────────────────────────

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def max_bytes(self) -> Optional[int]:
        return self._max_bytes

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, handle: int) -> bool:
        return handle in self._tensors

    # ── LRU helpers ─────────────────────────────────────────────────────────

    def _evict_until_fits_locked(
        self,
        incoming_bytes: int,
        *,
        pinned_handle: Optional[int] = None,
    ) -> None:
        """Caller must hold ``self._lock``."""
        if self._max_bytes is None:
            return
        if incoming_bytes > self._max_bytes:
            raise RuntimeError(
                f"single allocation ({incoming_bytes} B) exceeds pinned pool "
                f"budget ({self._max_bytes} B)"
            )
        if not self._lru_enabled:
            if self._total_bytes + incoming_bytes > self._max_bytes:
                raise RuntimeError(
                    f"pinned pool full: {self._total_bytes} + {incoming_bytes} "
                    f"> {self._max_bytes}; enable lru_enabled to evict"
                )
            return

        while self._total_bytes + incoming_bytes > self._max_bytes:
            try:
                victim, _ = next(iter(self._tensors.items()))
            except StopIteration:
                break
            if victim == pinned_handle:
                # Don't evict the handle we're currently rehydrating.
                self._tensors.move_to_end(victim)
                continue
            logger.debug("pinned-pool evicting handle=%d", victim)
            self._tensors.pop(victim, None)
            self._total_bytes -= self._nbytes.pop(victim, 0)
