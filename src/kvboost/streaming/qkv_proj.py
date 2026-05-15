"""Streaming AWQ linear layer.

``StreamingQLinear`` is a parameterless ``nn.Module`` that holds *references*
to slot-relative views (``qweight``, ``scales``, ``qzeros``, ``bias``) and
dispatches to whichever AWQ GEMM kernel is available. The slot base pointer is
stable across forwards (only the bytes inside the slot change), which keeps
the Marlin launch-config cache valid — see the plan's M2 risk note.

For resident layers the same module is used: ``rebind`` is called once at load
time with already-on-GPU tensors and never again.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .kernels import awq_linear


class StreamingQLinear(nn.Module):
    """AWQ linear backed by tensor views (slot-relative or resident).

    The module deliberately stores tensor references in plain attributes —
    NOT ``nn.Parameter`` — because the underlying storage may belong to a
    staging slot that gets overwritten between forwards. Treating them as
    parameters would mislead PyTorch's autograd / state-dict machinery.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int,
        prefer: str = "auto",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.prefer = prefer

        self._qweight: Optional[torch.Tensor] = None
        self._scales: Optional[torch.Tensor] = None
        self._qzeros: Optional[torch.Tensor] = None
        self._bias: Optional[torch.Tensor] = None

    # ── Binding ─────────────────────────────────────────────────────────────

    def rebind(
        self,
        *,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Point this module at fresh slot-relative views (or resident weights).

        Validates shapes once so a misaligned slot layout fails loudly here
        rather than as a CUDA illegal-memory access deep inside Marlin.
        """
        pack = 8  # 32 / 4
        expected_qweight = (self.in_features, self.out_features // pack)
        if tuple(qweight.shape) != expected_qweight:
            raise ValueError(
                f"qweight shape {tuple(qweight.shape)} != expected {expected_qweight}"
            )
        if scales.shape[-1] != self.out_features:
            raise ValueError(
                f"scales out-dim {scales.shape[-1]} != out_features {self.out_features}"
            )

        self._qweight = qweight
        self._scales = scales
        self._qzeros = qzeros
        self._bias = bias

    @property
    def is_bound(self) -> bool:
        return self._qweight is not None

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._qweight is None or self._scales is None or self._qzeros is None:
            raise RuntimeError(
                "StreamingQLinear.forward called before rebind(); the streaming "
                "scheduler must populate weights for this layer first."
            )

        return awq_linear(
            x,
            self._qweight,
            self._scales,
            self._qzeros,
            self._bias,
            group_size=self.group_size,
            prefer=self.prefer,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"group_size={self.group_size}, "
            f"bound={self.is_bound}"
        )


__all__ = ["StreamingQLinear"]
