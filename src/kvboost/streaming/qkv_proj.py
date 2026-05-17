"""Streaming AWQ linear layer.

``StreamingQLinear`` is a parameterless ``nn.Module`` that holds *references*
to slot-relative views (``qweight``, ``scales``, ``qzeros``, ``bias``) and
dispatches to whichever AWQ GEMM kernel is available. The slot base pointer is
stable across forwards (only the bytes inside the slot change), which keeps
the Marlin launch-config cache valid — see the plan's M2 risk note.

For resident layers the same module is used: ``rebind`` is called once at load
time with already-on-GPU tensors and never again.

Two binding modes:

- **Streaming (default)**: ``rebind`` just stores references to the four
  AWQ tensors; ``forward`` runs the AWQ kernel against them every call.
  Required when the underlying bytes change between forwards (CUDA slot
  recycling).
- **Cached dense** (``cache_dense=True``): ``rebind`` dequantizes once
  into a dense fp16 weight; ``forward`` is a plain matmul. Used by the
  MPS / unified-memory path where weights are permanent and there's no
  benefit to keeping them packed. Costs ~4× memory per layer vs packed
  but eliminates per-forward dequant.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .kernels import awq_dequantize_reference, awq_linear
from .profile import get_profiler


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
        cache_dense: bool = False,
        layer_idx: Optional[int] = None,
        sub_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.prefer = prefer
        self.cache_dense = cache_dense
        # Tags for the profiler — purely metadata, never read by forward().
        self.layer_idx = layer_idx
        self.sub_path = sub_path

        self._qweight: Optional[torch.Tensor] = None
        self._scales: Optional[torch.Tensor] = None
        self._qzeros: Optional[torch.Tensor] = None
        self._bias: Optional[torch.Tensor] = None
        # Set only in cache_dense mode; holds the dequantized (in, out) weight.
        self._dense_weight: Optional[torch.Tensor] = None

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

        In ``cache_dense`` mode, the four AWQ tensors are immediately
        dequantized into a single dense ``(in_features, out_features)``
        fp16 weight and the packed tensors are dropped. Use this only when
        weights are permanent — otherwise the cached dense weight will
        silently shadow subsequent rebinds.
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

        if self.cache_dense:
            # Dequantize once, store only the dense matrix. ``scales.dtype``
            # is the AWQ-blessed compute dtype (fp16 in practice).
            dense = awq_dequantize_reference(qweight, scales, qzeros, self.group_size)
            self._dense_weight = dense.contiguous()
            # Drop the packed tensors so they can be freed.
            self._qweight = None
            self._scales = None
            self._qzeros = None
            self._bias = bias
        else:
            self._qweight = qweight
            self._scales = scales
            self._qzeros = qzeros
            self._bias = bias

    @property
    def is_bound(self) -> bool:
        return self._dense_weight is not None or self._qweight is not None

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with get_profiler().region(
            "qlinear.forward",
            layer_idx=self.layer_idx,
            sub_path=self.sub_path,
        ):
            if self._dense_weight is not None:
                # Cached-dense fast path: one matmul, no dequant per forward.
                out = x.to(self._dense_weight.dtype) @ self._dense_weight
                if self._bias is not None:
                    out = out + self._bias.to(out.dtype)
                return out

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
        mode = "dense" if self.cache_dense else "streaming"
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"group_size={self.group_size}, "
            f"mode={mode}, "
            f"bound={self.is_bound}"
        )


__all__ = ["StreamingQLinear"]
