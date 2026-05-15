from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .awq_loader import LayerSpec, TensorSpec
from .staging import SlotLayout, StagingArena

logger = logging.getLogger(__name__)

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
    """
    CUDA stream scheduler for layer-by-layer weight streaming.

    This class only orchestrates:
    - transfer stream prefetch
    - compute stream waits
    - two-slot reuse
    - resident layer bypass

    The actual layer execution stays in the caller's run_layer_fn.
    """

    def __init__(
        self,
        layer_specs: Sequence[LayerSpec],
        *,
        prefetch_source_fn: PrefetchSourceFn,
        run_layer_fn: RunLayerFn,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv: Sequence[Any],
    ) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            raise ValueError("StreamingScheduler.forward expects CUDA tensors")

        if len(past_kv) < self.num_layers:
            raise ValueError(
                f"past_kv has length {len(past_kv)} but {self.num_layers} layers were provided"
            )

        self.compute_stream = torch.cuda.current_stream(device=self.device)
        self._layer_to_slot.clear()

        if not self.streamed_indices:
            # Fully resident model.
            for i, plan in enumerate(self.layer_plans):
                hidden_states = self.run_layer_fn(
                    i,
                    hidden_states,
                    past_kv[i],
                    None,
                    None,
                    plan,
                )
            return hidden_states

        self._prime_initial_prefetches()

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

            slot_id = self._layer_to_slot[layer_idx]
            self.compute_stream.wait_event(self.xfer_done[layer_idx])

            hidden_states = self.run_layer_fn(
                layer_idx,
                hidden_states,
                past_kv[layer_idx],
                self.arena.slot_views(slot_id),
                slot_id,
                plan,
            )

            self.compute_stream.record_event(self.compute_done[layer_idx])

            # Reuse the same slot for the next streamed layer after the current
            # compute finishes. This is the slot-reuse hazard boundary.
            next_streamed_idx = self._next_unprefetched_streamed_layer(layer_idx)
            if next_streamed_idx is not None:
                self._prefetch_streamed_layer_into_slot(
                    layer_idx=next_streamed_idx,
                    slot_id=slot_id,
                    wait_event=self.compute_done[layer_idx],
                )

        return hidden_states

    def _prime_initial_prefetches(self) -> None:
        """
        Prefetch the first two streamed layers into slot 0 and slot 1.
        """
        initial = self.streamed_indices[: self.arena.num_slots]
        for slot_id, layer_idx in enumerate(initial):
            self._prefetch_streamed_layer_into_slot(
                layer_idx=layer_idx,
                slot_id=slot_id,
                wait_event=None,
            )

    def _next_unprefetched_streamed_layer(self, current_layer_idx: int) -> int | None:
        """
        Return the next streamed layer index after current_layer_idx that has not
        been scheduled into a slot yet.
        """
        seen = False
        for layer_idx in self.streamed_indices:
            if seen:
                return layer_idx
            if layer_idx == current_layer_idx:
                seen = True
        return None

    def _prefetch_streamed_layer_into_slot(
        self,
        *,
        layer_idx: int,
        slot_id: int,
        wait_event: Optional[torch.cuda.Event],
    ) -> None:
        """
        Copy the given layer's weights into the selected slot on the transfer stream.
        """
        if layer_idx in self._layer_to_slot:
            # Already prefetched.
            return

        if wait_event is not None:
            self.transfer_stream.wait_event(wait_event)

        layer_tensors = self.prefetch_source_fn(layer_idx)

        with torch.cuda.stream(self.transfer_stream):
            self.arena.copy_layer_into_slot(slot_id, layer_tensors)
            self.xfer_done[layer_idx].record(self.transfer_stream)

        self._layer_to_slot[layer_idx] = slot_id

    def debug_state(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "num_layers": self.num_layers,
            "streamed_indices": list(self.streamed_indices),
            "slot_bytes": self.layout.slot_bytes,
            "slot_count": self.arena.num_slots,
            "layer_to_slot": dict(self._layer_to_slot),
        }