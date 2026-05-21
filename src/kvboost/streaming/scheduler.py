from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .awq_loader import LayerSpec, TensorSpec
from .profile import get_profiler
from .staging import SlotLayout, StagingArena

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamingCounters:
    """Cumulative scheduler health counters. Cheap to maintain (one int+=
    per event) and decisive for diagnosing slow streaming runs.

    Read order of operations:
    - ``forwards``: how many model forward passes we've seen.
    - ``layer_before_calls``: total ``before_layer`` calls (resident layers
      short-circuit and don't increment).
    - ``prefetch_hits``: ``before_layer`` found the slot already populated
      by an async prefetch — the desired path.
    - ``prefetch_misses``: had to issue a synchronous prefetch from inside
      ``before_layer`` (the pipeline was behind). Each miss adds a full
      H2D wait to the critical path.
    - ``prefetches_async`` / ``prefetches_sync``: count of prefetch issues
      by mode. Sum is ``async + sync`` which should equal ``hits + misses``
      modulo final-layer cleanup.
    - ``prefetch_source_time_s``: cumulative wall time the CPU spent inside
      ``prefetch_source_fn`` (typically ``loader.pin_layer``). High values
      here mean disk / host-side work, not GPU-side DMA.
    """
    forwards: int = 0
    layer_before_calls: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetches_async: int = 0
    prefetches_sync: int = 0
    prefetch_source_time_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        denom = max(1, self.layer_before_calls)
        return {
            "forwards": self.forwards,
            "layer_before_calls": self.layer_before_calls,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "hit_rate": round(self.prefetch_hits / denom, 4),
            "prefetches_async": self.prefetches_async,
            "prefetches_sync": self.prefetches_sync,
            "prefetch_source_time_s": round(self.prefetch_source_time_s, 4),
        }

    def reset(self) -> None:
        self.forwards = 0
        self.layer_before_calls = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prefetches_async = 0
        self.prefetches_sync = 0
        self.prefetch_source_time_s = 0.0

PrefetchSourceFn = Callable[[int], Mapping[str, torch.Tensor]]
RunLayerFn = Callable[
    [
        int,  # layer_idx
        torch.Tensor,  # hidden_states
        Any,  # past_kv entry for this layer
        Optional[dict[str, torch.Tensor]],  # slot views if streamed
        Optional[int],  # slot_id if streamed
        LayerSpec,  # plan
    ],
    torch.Tensor,
]


@dataclass(slots=True)
class StreamingLayerPlan:
    layer_idx: int
    resident: bool
    tensors: dict[str, TensorSpec]


class StreamingScheduler:
    """CUDA stream scheduler for layer-by-layer weight streaming.

    Two ways to drive it:

    1. **Hook-driven** (production path used by ``StreamingCausalLM``):
       call :meth:`begin_forward` before HF's forward, then
       :meth:`before_layer` from each streamed layer's pre-hook and
       :meth:`after_layer` from each post-hook. The hooks own when to
       rebind weights into their executing module.

    2. **All-in-one** (legacy / unit-test path): call :meth:`forward`
       with the hidden state and per-layer past_kv. The scheduler walks
       all layers itself and calls ``run_layer_fn``.

    Both paths share the same prefetch + slot-recycle plumbing.
    """

    def __init__(
        self,
        layer_specs: Sequence[LayerSpec],
        *,
        prefetch_source_fn: PrefetchSourceFn,
        run_layer_fn: Optional[RunLayerFn] = None,
        device: str | torch.device = "cuda",
        num_slots: int = 2,
        alignment: int = 16,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                "StreamingScheduler is CUDA-only; use a resident fallback on MPS/CPU."
            )

        if num_slots != 2:
            logger.warning(
                "StreamingScheduler is optimized for 2 slots; continuing with num_slots=%d",
                num_slots,
            )

        self.prefetch_source_fn = prefetch_source_fn
        self.run_layer_fn = run_layer_fn

        self.layer_plans: list[LayerSpec] = list(layer_specs)
        self.num_layers = len(self.layer_plans)

        self.streamed_indices = [
            i for i, plan in enumerate(self.layer_plans) if not plan.resident
        ]

        self.layout = SlotLayout.from_layer_specs(
            self.layer_plans,
            alignment=alignment,
            streamed_only=True,
        )
        self.arena = StagingArena(
            self.layout,
            device=self.device,
            num_slots=num_slots,
        )

        self.transfer_stream = torch.cuda.Stream(device=self.device, priority=-1)
        self.compute_stream: Optional[torch.cuda.Stream] = None

        self.xfer_done = [
            torch.cuda.Event(blocking=False, interprocess=False)
            for _ in range(self.num_layers)
        ]
        self.compute_done = [
            torch.cuda.Event(blocking=False, interprocess=False)
            for _ in range(self.num_layers)
        ]

        # Maps streamed layer index -> slot id currently holding its weights.
        self._layer_to_slot: dict[int, int] = {}

        # Cumulative health counters. Read via ``self.counters.summary()``.
        # Always-on; each event is one int+= so overhead is negligible.
        self.counters = StreamingCounters()

    # ── Hook-driven primitives ──────────────────────────────────────────────

    def begin_forward(self) -> None:
        """Reset per-forward state and prime the first ``num_slots`` prefetches.

        Must be called from compute-stream context (typically the main
        stream) before any layer pre-hook fires.
        """
        with get_profiler().region("scheduler.begin_forward"):
            self.counters.forwards += 1
            self.compute_stream = torch.cuda.current_stream(device=self.device)
            self._layer_to_slot.clear()
            if self.streamed_indices:
                self._prime_initial_prefetches()

    def before_layer(self, layer_idx: int) -> Optional[dict[str, torch.Tensor]]:
        """Block compute on this layer's transfer completion and return its
        slot views. Returns ``None`` for resident layers (caller should skip).
        """
        plan = self.layer_plans[layer_idx]
        if plan.resident:
            return None

        with get_profiler().region("scheduler.before_layer", layer_idx=layer_idx):
            self.counters.layer_before_calls += 1
            slot_id = self._layer_to_slot.get(layer_idx)
            if slot_id is None:
                # Late prefetch (single-slot config, or layer was missed). Stage
                # synchronously into slot 0 so forward can proceed.
                self.counters.prefetch_misses += 1
                slot_id = self._fallback_synchronous_prefetch(layer_idx)
            else:
                self.counters.prefetch_hits += 1

            assert self.compute_stream is not None, "begin_forward not called"
            self.compute_stream.wait_event(self.xfer_done[layer_idx])
            return self.arena.slot_views(slot_id)

    def after_layer(self, layer_idx: int) -> None:
        """Record the compute-done event for this layer and schedule the
        prefetch of the next-but-one streamed layer (i.e. ``i + num_slots``)
        into the slot we just released.
        """
        plan = self.layer_plans[layer_idx]
        if plan.resident:
            return

        with get_profiler().region("scheduler.after_layer", layer_idx=layer_idx):
            slot_id = self._layer_to_slot.get(layer_idx)
            if slot_id is None:
                return

            assert self.compute_stream is not None
            self.compute_stream.record_event(self.compute_done[layer_idx])

            # Look ahead by num_slots; that's the next streamed layer eligible to
            # reuse the slot we're about to free.
            next_layer = self._next_streamed_after(layer_idx, hops=self.arena.num_slots)
            if next_layer is not None and next_layer not in self._layer_to_slot:
                self._prefetch_streamed_layer_into_slot(
                    layer_idx=next_layer,
                    slot_id=slot_id,
                    wait_event=self.compute_done[layer_idx],
                )

    # ── Legacy all-in-one driver ────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv: Sequence[Any],
    ) -> torch.Tensor:
        if self.run_layer_fn is None:
            raise RuntimeError(
                "StreamingScheduler.forward requires run_layer_fn at construction; "
                "use begin_forward/before_layer/after_layer for hook-driven mode."
            )
        if hidden_states.device.type != "cuda":
            raise ValueError("StreamingScheduler.forward expects CUDA tensors")

        if len(past_kv) < self.num_layers:
            raise ValueError(
                f"past_kv has length {len(past_kv)} but {self.num_layers} layers were provided"
            )

        self.begin_forward()

        for layer_idx, plan in enumerate(self.layer_plans):
            if plan.resident:
                hidden_states = self.run_layer_fn(
                    layer_idx,
                    hidden_states,
                    past_kv[layer_idx],
                    None,
                    None,
                    plan,
                )
                continue

            slot_views = self.before_layer(layer_idx)
            slot_id = self._layer_to_slot[layer_idx]
            hidden_states = self.run_layer_fn(
                layer_idx,
                hidden_states,
                past_kv[layer_idx],
                slot_views,
                slot_id,
                plan,
            )
            self.after_layer(layer_idx)

        return hidden_states

    # ── Internals ───────────────────────────────────────────────────────────

    def _prime_initial_prefetches(self) -> None:
        initial = self.streamed_indices[: self.arena.num_slots]
        for slot_id, layer_idx in enumerate(initial):
            self._prefetch_streamed_layer_into_slot(
                layer_idx=layer_idx,
                slot_id=slot_id,
                wait_event=None,
            )

    def _next_streamed_after(self, current_layer_idx: int, *, hops: int = 1) -> int | None:
        """Return the streamed layer that's ``hops`` positions after
        ``current_layer_idx`` in the streamed-only sequence. ``hops=1`` is
        the immediate next streamed layer.
        """
        try:
            pos = self.streamed_indices.index(current_layer_idx)
        except ValueError:
            return None
        target = pos + hops
        if 0 <= target < len(self.streamed_indices):
            return self.streamed_indices[target]
        return None

    def _prefetch_streamed_layer_into_slot(
        self,
        *,
        layer_idx: int,
        slot_id: int,
        wait_event: Optional[torch.cuda.Event],
        is_sync_fallback: bool = False,
    ) -> None:
        if layer_idx in self._layer_to_slot:
            return

        if wait_event is not None:
            self.transfer_stream.wait_event(wait_event)
        if is_sync_fallback:
            self.counters.prefetches_sync += 1
        else:
            self.counters.prefetches_async += 1

        t0 = time.perf_counter()
        with get_profiler().region("scheduler.prefetch_source_fn", layer_idx=layer_idx):
            layer_tensors = self.prefetch_source_fn(layer_idx)
        self.counters.prefetch_source_time_s += time.perf_counter() - t0

        with torch.cuda.stream(self.transfer_stream):
            self.arena.copy_layer_into_slot(slot_id, layer_tensors)
            self.xfer_done[layer_idx].record(self.transfer_stream)

        self._layer_to_slot[layer_idx] = slot_id

    def _fallback_synchronous_prefetch(self, layer_idx: int) -> int:
        """Stage a layer into slot 0 synchronously. Used when the pipeline
        has only 1 slot or the caller skipped priming.
        """
        slot_id = 0
        self._prefetch_streamed_layer_into_slot(
            layer_idx=layer_idx,
            slot_id=slot_id,
            wait_event=None,
            is_sync_fallback=True,
        )
        return slot_id

    def debug_state(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "num_layers": self.num_layers,
            "streamed_indices": list(self.streamed_indices),
            "slot_bytes": self.layout.slot_bytes,
            "slot_count": self.arena.num_slots,
            "layer_to_slot": dict(self._layer_to_slot),
        }
