from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Mapping, Sequence

import torch

from .awq_loader import LayerSpec, TensorSpec


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(slots=True, frozen=True)
class TensorPlacement:
    """
    A tensor's location inside one contiguous uint8 staging slot.
    """

    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def numel(self) -> int:
        return int(prod(self.shape)) if self.shape else 1

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()


@dataclass(slots=True)
class SlotLayout:
    """
    Static layout for one streamed layer slot.

    The layout is derived from a representative streamed layer and validated
    against the other streamed layers. Offsets are byte-aligned so the slot
    can be backed by a raw uint8 CUDA tensor while tensor views are created
    via stride-safe dtype reinterpretation.
    """

    slot_bytes: int
    placements: dict[str, TensorPlacement] = field(default_factory=dict)
    alignment: int = 16

    @classmethod
    def from_layer_specs(
        cls,
        layer_specs: Sequence[LayerSpec],
        *,
        alignment: int = 16,
        streamed_only: bool = True,
    ) -> "SlotLayout":
        relevant = [
            layer for layer in layer_specs if not streamed_only or not layer.resident
        ]
        if not relevant:
            return cls(slot_bytes=0, placements={}, alignment=alignment)

        reference = relevant[0]
        ref_keys = set(reference.tensors.keys())

        # Validate the tensor schema is consistent across streamed layers.
        for layer in relevant[1:]:
            keys = set(layer.tensors.keys())
            if keys != ref_keys:
                raise ValueError(
                    f"Layer {layer.layer_idx} tensor schema differs from reference "
                    f"layer {reference.layer_idx}: {sorted(keys ^ ref_keys)}"
                )
            for name, spec in layer.tensors.items():
                ref_spec = reference.tensors[name]
                if spec.shape != ref_spec.shape or spec.dtype != ref_spec.dtype:
                    raise ValueError(
                        f"Tensor mismatch for {name}: "
                        f"layer {layer.layer_idx} has {spec.shape}/{spec.dtype}, "
                        f"reference {reference.layer_idx} has {ref_spec.shape}/{ref_spec.dtype}"
                    )

        # Build a deterministic placement map from the reference layer.
        placements: dict[str, TensorPlacement] = {}
        cursor = 0
        for name in sorted(reference.tensors.keys()):
            spec = reference.tensors[name]
            cursor = align_up(cursor, alignment)
            placements[name] = TensorPlacement(
                name=name,
                offset=cursor,
                nbytes=spec.nbytes,
                shape=spec.shape,
                dtype=spec.dtype,
            )
            cursor += spec.nbytes

        # Compute the maximum required bytes across streamed layers.
        slot_bytes = 0
        for layer in relevant:
            required = 0
            for name in sorted(layer.tensors.keys()):
                spec = layer.tensors[name]
                required = align_up(required, alignment)
                required += spec.nbytes
            slot_bytes = max(slot_bytes, align_up(required, alignment))

        return cls(slot_bytes=slot_bytes, placements=placements, alignment=alignment)

    def placement_for(self, name: str) -> TensorPlacement:
        try:
            return self.placements[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tensor placement: {name}") from exc


class StagingSlot:
    """
    Raw CUDA byte buffer for one streamed layer.

    Tensor views are reconstructed by byte offset + dtype reinterpretation.
    """

    def __init__(
        self,
        slot_id: int,
        slot_bytes: int,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("StagingSlot requires a CUDA device")

        self.slot_id = slot_id
        self.slot_bytes = slot_bytes
        self.device = device

        self.buffer = torch.empty(
            slot_bytes,
            dtype=torch.uint8,
            device=device,
        )

    def view(self, placement: TensorPlacement) -> torch.Tensor:
        """
        Return a zero-copy tensor view into the slot.

        This is safe because the staging buffer is raw bytes and the layout
        enforces dtype-aligned offsets.
        """
        if placement.offset + placement.nbytes > self.slot_bytes:
            raise ValueError(
                f"Placement {placement.name} exceeds slot bounds "
                f"({placement.offset + placement.nbytes} > {self.slot_bytes})"
            )

        byte_slice = self.buffer.narrow(0, placement.offset, placement.nbytes)
        typed = byte_slice.view(placement.dtype)
        return typed.reshape(placement.shape)

    def copy_from_host(
        self,
        placement: TensorPlacement,
        host_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Copy host tensor contents into the slot.

        Expected source:
        - CPU tensor for streamed weights
        - pinned CPU tensor for async CUDA copies
        """
        dst = self.view(placement)

        if host_tensor.shape != placement.shape:
            raise ValueError(
                f"Shape mismatch for {placement.name}: "
                f"expected {placement.shape}, got {tuple(host_tensor.shape)}"
            )
        if host_tensor.dtype != placement.dtype:
            raise ValueError(
                f"Dtype mismatch for {placement.name}: "
                f"expected {placement.dtype}, got {host_tensor.dtype}"
            )

        non_blocking = host_tensor.device.type == "cpu"
        dst.copy_(host_tensor, non_blocking=non_blocking)
        return dst

    def views(self, layout: SlotLayout) -> dict[str, torch.Tensor]:
        return {name: self.view(placement) for name, placement in layout.placements.items()}


class StagingArena:
    """
    Two-slot CUDA staging arena for ping-pong layer streaming.
    """

    def __init__(
        self,
        layout: SlotLayout,
        *,
        device: torch.device,
        num_slots: int = 2,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("StagingArena is CUDA-only")

        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")

        self.layout = layout
        self.device = device
        self.num_slots = num_slots

        self.slots = [
            StagingSlot(slot_id=i, slot_bytes=layout.slot_bytes, device=device)
            for i in range(num_slots)
        ]

    def slot(self, slot_id: int) -> StagingSlot:
        return self.slots[slot_id]

    def slot_views(self, slot_id: int) -> dict[str, torch.Tensor]:
        return self.slots[slot_id].views(self.layout)

    def copy_layer_into_slot(
        self,
        slot_id: int,
        layer_tensors: Mapping[str, torch.Tensor],
    ) -> None:
        slot = self.slots[slot_id]
        for name, tensor in layer_tensors.items():
            placement = self.layout.placement_for(name)
            slot.copy_from_host(placement, tensor)

    def clear(self) -> None:
        # Not strictly necessary, but useful for debugging / determinism.
        self.slots.clear()
        self.slots.extend(
            [
                StagingSlot(slot_id=i, slot_bytes=self.layout.slot_bytes, device=self.device)
                for i in range(self.num_slots)
            ]
        )