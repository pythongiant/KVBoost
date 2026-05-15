from __future__ import annotations

"""Streaming layer runner for KVBoost.

This module implements a fresh nn.Module that mirrors the forward contract of
Hugging Face LlamaDecoderLayer without monkey-patching any HF internals.

It is intentionally thin:
- no trainable parameters
- no persistent weights
- all quantized weights are supplied as slot-relative pointers / views
- attention dispatch reuses the existing FlashAttention path

The module is designed to be paired with:
- a streaming scheduler that preloads one layer blob per slot
- a qlinear backend (Marlin or exllamav2) exposed as torch-tensor functions
- resident submodules for layers that remain permanently on GPU

This file is written to be conservative about HF version drift. It accepts
common LlamaDecoderLayer.forward argument patterns and passes through outputs
compatible with the standard `(hidden_states, present_key_value, attn_weights)`
style used by Transformers.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch
import torch.nn as nn

try:  # optional at import time for local unit tests
    from kvboost.flash_attn_ext import flash_attn_func
except Exception:  # pragma: no cover
    flash_attn_func = None  # type: ignore[assignment]


@runtime_checkable
class SupportsForward(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class LayerRunnerConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    use_flash_attn: bool = True
    is_decoder: bool = True


@dataclass(frozen=True)
class ResidentLayerRefs:
    """Resident (fully-GPU) submodules.

    These are the only learnable / persistent components attached to the module.
    They may be standard HF submodules or lightweight replacements.
    """

    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module
    self_attn: nn.Module
    mlp: nn.Module


@dataclass(frozen=True)
class StreamedLayerSpec:
    """Metadata describing the currently staged layer blob.

    The `slot` is managed externally by the staging scheduler. All pointers are
    slot-relative and must remain stable for the lifetime of the slot storage.
    """

    slot: int
    layer_idx: int
    q_proj: Any
    k_proj: Any
    v_proj: Any
    o_proj: Any
    gate_proj: Any
    up_proj: Any
    down_proj: Any
    input_layernorm: Any
    post_attention_layernorm: Any
    self_attn: Any
    mlp: Any


class _IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return x


class StreamingDecoderLayer(nn.Module):
    """Stateless decoder layer that mirrors LlamaDecoderLayer.forward.

    The instance itself holds no parameters. It consumes a per-forward
    `StreamedLayerSpec` for streamed layers or `ResidentLayerRefs` for layers
    pinned in VRAM.
    """

    def __init__(
        self,
        config: LayerRunnerConfig,
        *,
        resident: Optional[ResidentLayerRefs] = None,
        attention_impl: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.resident = resident
        self.attention_impl = attention_impl or flash_attn_func
        self._norm_fallback = _IdentityNorm()

    @property
    def has_resident_weights(self) -> bool:
        return self.resident is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        layer_spec: Optional[StreamedLayerSpec] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        """Run one decoder layer.

        The signature is intentionally broad to survive HF minor API drift.
        Extra keyword args are ignored unless a downstream backend uses them.
        """

        del kwargs

        if self.resident is not None:
            return self._forward_resident(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        if layer_spec is None:
            raise ValueError("layer_spec is required for streamed layers")

        return self._forward_streamed(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            layer_spec=layer_spec,
        )

    def _forward_resident(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Any],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.resident.input_layernorm(hidden_states)

        attn_out = self._call_attention(
            self.resident.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states, self_attn_weights, present_key_value = self._unpack_attention_output(attn_out)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.resident.post_attention_layernorm(hidden_states)
        hidden_states = self.resident.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, self_attn_weights

    def _forward_streamed(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Any],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        layer_spec: StreamedLayerSpec,
    ) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        del cache_position

        residual = hidden_states
        hidden_states = self._call_norm(layer_spec.input_layernorm, hidden_states)

        attn_out = self._call_attention(
            layer_spec.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=None,
            position_embeddings=position_embeddings,
        )
        hidden_states, self_attn_weights, present_key_value = self._unpack_attention_output(attn_out)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self._call_norm(layer_spec.post_attention_layernorm, hidden_states)
        hidden_states = self._call_mlp(layer_spec, hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, self_attn_weights

    def _call_norm(self, norm: Any, x: torch.Tensor) -> torch.Tensor:
        if isinstance(norm, nn.Module):
            return norm(x)
        if callable(norm):
            return norm(x)
        return self._norm_fallback(x)

    def _call_mlp(self, layer_spec: StreamedLayerSpec, x: torch.Tensor) -> torch.Tensor:
        mlp = layer_spec.mlp
        if callable(mlp):
            try:
                return mlp(x)
            except TypeError:
                pass

        # Fallback path for explicit tensor-backed kernels.
        gate = self._call_qlinear(layer_spec.gate_proj, x)
        up = self._call_qlinear(layer_spec.up_proj, x)
        hidden = torch.nn.functional.silu(gate) * up
        return self._call_qlinear(layer_spec.down_proj, hidden)

    def _call_qlinear(self, proj: Any, x: torch.Tensor) -> torch.Tensor:
        if callable(proj):
            return proj(x)
        if isinstance(proj, tuple) and len(proj) == 5:
            # (qweight_ptr, scales_ptr, qzeros_ptr, bias_ptr, kernel)
            qweight_ptr, scales_ptr, qzeros_ptr, bias_ptr, kernel = proj
            return kernel(x, qweight_ptr, scales_ptr, qzeros_ptr, bias_ptr)
        raise TypeError(f"unsupported qlinear descriptor: {type(proj)!r}")

    def _call_attention(
        self,
        attn: Any,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Any],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Any:
        if callable(attn):
            try:
                return attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            except TypeError:
                return attn(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)

        if self.attention_impl is None:
            raise RuntimeError("flash_attn_func is unavailable and no attention_impl was provided")

        # Conservative fallback for tensor-only attention backends.
        return self.attention_impl(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask=attention_mask,
            causal=True,
        )

    @staticmethod
    def _unpack_attention_output(out: Any) -> Tuple[torch.Tensor, Any, Optional[Any]]:
        if isinstance(out, tuple):
            if len(out) == 3:
                return out[0], out[1], out[2]
            if len(out) == 2:
                return out[0], None, out[1]
            if len(out) == 1:
                return out[0], None, None
        return out, None, None


__all__ = [
    "LayerRunnerConfig",
    "ResidentLayerRefs",
    "StreamedLayerSpec",
    "StreamingDecoderLayer",
]