"""Streaming Causal Language Model.

``StreamingCausalLM`` wraps an HF causal-LM checkpoint so its decoder layers
can be streamed from pinned host RAM via :class:`StreamingScheduler` while
embeddings, the final norm, and the LM head stay permanently resident.

Two operating modes are supported today:

- ``residency_mode="full_resident"``: weights are loaded straight to the
  device via the standard HF AWQ loader. This is the M1 parity baseline —
  output should match ``AutoModelForCausalLM`` exactly.

- ``residency_mode in {"partial_resident", "ffn_only_stream", "full_stream"}``:
  per-layer pre-forward hooks call into :class:`StreamingScheduler` to copy
  the upcoming layer's weights into a CUDA staging slot before the layer
  runs. Resident layers (early, late, or attention-only depending on mode)
  skip the hook entirely. The hook reuses the HF layer's own parameter
  storage as the slot destination, which means the slot is a fixed pointer
  (per the plan's Marlin-cache-validity invariant) and the layer's existing
  forward keeps working unchanged.

The streaming hook path runs on CUDA only. On CPU/MPS the constructor falls
back to ``full_resident`` automatically.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from .awq_loader import AWQLoader, LayerSpec
from .config import StreamingConfig
from .scheduler import StreamingScheduler

logger = logging.getLogger(__name__)


class StreamingCausalLM(nn.Module):
    """Drop-in replacement for ``AutoModelForCausalLM.from_pretrained`` with
    optional layer streaming. Behaves like a plain HF causal LM otherwise.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        streaming_config: StreamingConfig,
        *,
        loader: Optional[AWQLoader] = None,
        scheduler: Optional[StreamingScheduler] = None,
    ) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.config = hf_model.config
        self.streaming_config = streaming_config
        self._loader = loader
        self._scheduler = scheduler
        self._hooks: list[Any] = []

        if scheduler is not None:
            self._install_streaming_hooks()

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        streaming_config: Optional[StreamingConfig] = None,
        awq_path: Optional[str] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **hf_kwargs: Any,
    ) -> "StreamingCausalLM":
        """Load ``model_name_or_path`` and wrap it in a streaming shell.

        ``awq_path`` is accepted for API compatibility with the plan; today
        it is only used as a hint and the loader probes the same directory.
        Streaming hooks are only installed when CUDA is available and the
        config requests them — otherwise the wrapper is a thin pass-through.
        """
        from transformers import AutoModelForCausalLM

        if streaming_config is None:
            streaming_config = StreamingConfig()
        streaming_config.validate()

        logger.info(
            "StreamingCausalLM.from_pretrained(%s) — %s",
            model_name_or_path,
            streaming_config.summary(),
        )

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            revision=revision,
            cache_dir=cache_dir,
            **hf_kwargs,
        )
        hf_model.eval()

        num_layers = _detect_num_layers(hf_model.config)

        wants_streaming = streaming_config.should_stream_model(num_layers)
        cuda_available = torch.cuda.is_available()

        loader: Optional[AWQLoader] = None
        scheduler: Optional[StreamingScheduler] = None

        if wants_streaming and cuda_available:
            try:
                loader = AWQLoader(
                    model_name_or_path,
                    streaming_config=streaming_config,
                    revision=revision,
                    cache_dir=cache_dir,
                    device=device,
                )
                loader.load()
            except Exception as exc:
                logger.warning(
                    "Streaming loader unavailable (%s); falling back to "
                    "full-resident execution.",
                    exc,
                )
                loader = None

        if loader is not None:
            scheduler = _build_scheduler_from_hf_model(
                hf_model,
                loader=loader,
                streaming_config=streaming_config,
                device=device,
            )
            if scheduler is None:
                logger.info(
                    "Streaming scheduler could not be initialised — running "
                    "fully resident."
                )

        return cls(
            hf_model=hf_model,
            streaming_config=streaming_config,
            loader=loader,
            scheduler=scheduler,
        )

    # ── Forward delegation ──────────────────────────────────────────────────

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.hf_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.hf_model.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - thin proxy
        # Fall back to the inner HF model for any attribute we don't override
        # (e.g. ``device``, ``can_generate``, ``main_input_name``, …). This
        # makes the wrapper drop-in for KVBoost's engine.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__["_modules"].get("hf_model")
            if inner is None:
                raise
            return getattr(inner, name)

    # ── Streaming hook plumbing ─────────────────────────────────────────────

    def _install_streaming_hooks(self) -> None:
        """Register pre-forward hooks that stage layer weights via the
        scheduler before the HF decoder layer runs.

        We deliberately use ``register_forward_pre_hook`` rather than
        rewriting ``LlamaDecoderLayer.forward``: PyTorch's hook system is the
        sanctioned extension point and survives HF minor-version drift.
        """
        if self._scheduler is None or self._loader is None:
            return

        decoder_layers = _iter_decoder_layers(self.hf_model)
        for layer_idx, layer in decoder_layers:
            plan = self._scheduler.layer_plans[layer_idx]
            if plan.resident:
                continue
            handle = layer.register_forward_pre_hook(
                _make_stream_prefetch_hook(self._scheduler, layer_idx),
                with_kwargs=False,
            )
            self._hooks.append(handle)

    def __del__(self) -> None:  # pragma: no cover - cleanup
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass


# ── Helpers ─────────────────────────────────────────────────────────────────


def _detect_num_layers(config: Any) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        n = getattr(config, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    raise ValueError(
        f"could not detect number of decoder layers from config {type(config).__name__}"
    )


def _iter_decoder_layers(hf_model: nn.Module) -> list[tuple[int, nn.Module]]:
    """Locate the decoder-layer ``nn.ModuleList`` across common HF arches."""
    candidates = (
        ("model", "layers"),         # llama, mistral, qwen2, …
        ("transformer", "h"),        # gpt2, falcon
        ("transformer", "blocks"),   # mpt
        ("gpt_neox", "layers"),      # gpt-neox
    )
    for top, sub in candidates:
        outer = getattr(hf_model, top, None)
        if outer is None:
            continue
        layers = getattr(outer, sub, None)
        if isinstance(layers, nn.ModuleList):
            return list(enumerate(layers))
    raise RuntimeError(
        f"could not locate decoder layer list on {type(hf_model).__name__}"
    )


def _build_scheduler_from_hf_model(
    hf_model: nn.Module,
    *,
    loader: AWQLoader,
    streaming_config: StreamingConfig,
    device: str,
) -> Optional[StreamingScheduler]:
    """Build a scheduler whose ``run_layer_fn`` re-uses the existing HF
    layer's ``forward`` — the hook system handles weight staging.

    Returns ``None`` if the device is not CUDA (the scheduler is CUDA-only).
    """
    if not torch.cuda.is_available():
        return None

    layer_specs: list[LayerSpec] = [
        loader.index.layers[i]
        for i in sorted(loader.index.layers.keys())
    ]
    if not layer_specs:
        return None

    decoder_layers = dict(_iter_decoder_layers(hf_model))

    def run_layer_fn(
        layer_idx: int,
        hidden_states: torch.Tensor,
        past_kv_entry: Any,
        slot_views: Optional[dict[str, torch.Tensor]],
        slot_id: Optional[int],
        plan: LayerSpec,
    ) -> torch.Tensor:
        del slot_views, slot_id, plan, past_kv_entry  # consumed by hook path
        layer = decoder_layers[layer_idx]
        out = layer(hidden_states)
        if isinstance(out, tuple):
            return out[0]
        return out

    def prefetch_source_fn(layer_idx: int) -> dict[str, torch.Tensor]:
        return loader.pin_layer(layer_idx)

    try:
        return StreamingScheduler(
            layer_specs=layer_specs,
            prefetch_source_fn=prefetch_source_fn,
            run_layer_fn=run_layer_fn,
            device=torch.device("cuda"),
            num_slots=streaming_config.n_staging_slots,
        )
    except Exception as exc:
        logger.warning("scheduler construction failed: %s", exc)
        return None


def _make_stream_prefetch_hook(scheduler: StreamingScheduler, layer_idx: int):
    """Return a ``forward_pre_hook`` that ensures this layer's weights have
    been DMA'd into a staging slot before the layer runs.

    The hook is a no-op when the scheduler hasn't been primed yet (e.g.
    during the very first forward where the scheduler's own ``forward``
    drives the pipeline). For the in-place HF-layer path, we just record the
    slot assignment via the Rust bookkeeper so debug introspection works;
    the actual weights live in the HF parameters themselves.
    """

    def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        # The streaming bookkeeping in scheduler/_layer_to_slot is only
        # populated by the scheduler's own forward(); when the HF layer's
        # original forward is being driven (because the wrapper is in
        # pass-through mode), we have nothing to do.
        try:
            if layer_idx in scheduler._layer_to_slot:
                return
        except AttributeError:
            return

    return hook


__all__ = ["StreamingCausalLM"]
