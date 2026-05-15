"""Streaming Causal Language Model.

``StreamingCausalLM`` wraps an HF causal-LM checkpoint so its decoder layers
can be streamed from pinned host RAM via :class:`StreamingScheduler` while
embeddings, the final norm, and the LM head stay permanently resident.

Modes:

- ``residency_mode="full_resident"``: weights are loaded straight to the
  device via the standard HF AWQ loader. M1 parity baseline.

- ``residency_mode in {"partial_resident", "ffn_only_stream", "full_stream"}``:
  the model is instantiated with ``accelerate.init_empty_weights`` so all
  parameters land on the ``meta`` device. The AWQLoader then materializes
  only the resident parameters onto GPU, and the quantized linear modules
  in *streamed* decoder layers are replaced with parameterless
  :class:`StreamingQLinear` instances. A forward-pre-hook on each streamed
  layer asks the scheduler to stage the next layer's weights into a CUDA
  slot and rebinds the layer's StreamingQLinear children to that slot's
  views. The slot pointers are constant across forwards (only the bytes
  change), preserving Marlin's launch-config cache invariant.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from .awq_loader import (
    AWQLoader,
    LayerSpec,
    _assign_dotted_attribute,
    _resolve_dotted_attribute,
)
from .config import StreamingConfig
from .qkv_proj import StreamingQLinear
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
        streamed_qlinears: Optional[dict[int, dict[str, StreamingQLinear]]] = None,
    ) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.config = hf_model.config
        self.streaming_config = streaming_config
        self._loader = loader
        self._scheduler = scheduler
        self._streamed_qlinears = streamed_qlinears or {}
        self._hook_handles: list[Any] = []

        if scheduler is not None and streamed_qlinears:
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
        """Load ``model_name_or_path`` and wrap it in a streaming shell."""
        from transformers import AutoConfig, AutoModelForCausalLM

        del awq_path  # accepted for API compat; not currently used as a hint

        if streaming_config is None:
            streaming_config = StreamingConfig()
        streaming_config.validate()

        logger.info(
            "StreamingCausalLM.from_pretrained(%s) — %s",
            model_name_or_path,
            streaming_config.summary(),
        )

        cfg = AutoConfig.from_pretrained(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        num_layers = _detect_num_layers(cfg)

        want_streaming = (
            streaming_config.should_stream_model(num_layers)
            and torch.cuda.is_available()
        )

        if not want_streaming:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                low_cpu_mem_usage=True,
                revision=revision,
                cache_dir=cache_dir,
                **hf_kwargs,
            )
            hf_model.eval()
            return cls(hf_model=hf_model, streaming_config=streaming_config)

        # ── Streaming path ────────────────────────────────────────────────
        loader = AWQLoader(
            model_name_or_path,
            streaming_config=streaming_config,
            revision=revision,
            cache_dir=cache_dir,
            device=device,
        )
        loader.load()

        try:
            from accelerate import init_empty_weights
        except ImportError as exc:
            raise ImportError(
                "Streaming inference requires `accelerate`. "
                "Install with `pip install kvboost[streaming]`."
            ) from exc

        with init_empty_weights():
            hf_model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
        hf_model.eval()

        # 1. Replace streamed layers' quant linears BEFORE materializing —
        #    that way we don't accidentally materialize tensors we're about
        #    to throw away.
        streamed_indices = set(loader.streamed_layer_indices())
        group_size = int(loader.index.quant_config.get("group_size", 128))
        streamed_qlinears = _replace_streamed_linears(
            hf_model,
            layer_indices=streamed_indices,
            group_size=group_size,
            prefer=streaming_config.quant_kernel,
        )

        # 2. Materialize resident tensors onto GPU.
        loader.materialize_into_module(hf_model, only_resident=True)

        # 3. Move any remaining meta params on resident submodules off meta.
        _materialize_meta_buffers(hf_model, device=loader.device_spec.device, dtype=dtype)

        # 4. Build the scheduler that drives the staged DMA.
        scheduler = _build_scheduler(
            hf_model,
            loader=loader,
            streaming_config=streaming_config,
        )

        return cls(
            hf_model=hf_model,
            streaming_config=streaming_config,
            loader=loader,
            scheduler=scheduler,
            streamed_qlinears=streamed_qlinears,
        )

    # ── Forward delegation ──────────────────────────────────────────────────

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._scheduler is not None:
            self._scheduler.begin_forward()
        out = self.hf_model(*args, **kwargs)
        if self._scheduler is not None:
            torch.cuda.synchronize(self._scheduler.device)
        return out

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.hf_model.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - thin proxy
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__["_modules"].get("hf_model")
            if inner is None:
                raise
            return getattr(inner, name)

    # ── Streaming hook plumbing ─────────────────────────────────────────────

    def _install_streaming_hooks(self) -> None:
        """Attach a pre/post hook to each streamed HF decoder layer.

        Pre-hook: asks the scheduler to ensure this layer's weights are
        staged into a slot, then rebinds the layer's StreamingQLinear
        children to that slot's views.

        Post-hook: records the compute-done event and schedules the
        next-but-one streamed layer's prefetch into the freed slot.
        """
        if self._scheduler is None:
            return

        layers = dict(_iter_decoder_layers(self.hf_model))
        for layer_idx, qlinears in self._streamed_qlinears.items():
            hf_layer = layers[layer_idx]
            pre = hf_layer.register_forward_pre_hook(
                _make_pre_hook(self._scheduler, layer_idx, qlinears),
                with_kwargs=False,
            )
            post = hf_layer.register_forward_hook(
                _make_post_hook(self._scheduler, layer_idx),
                with_kwargs=False,
            )
            self._hook_handles.extend([pre, post])

    def __del__(self) -> None:  # pragma: no cover - cleanup
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass


# ── Module-tree helpers ─────────────────────────────────────────────────────


def _detect_num_layers(config: Any) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        n = getattr(config, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    raise ValueError(
        f"could not detect number of decoder layers from config {type(config).__name__}"
    )


def _iter_decoder_layers(hf_model: nn.Module) -> list[tuple[int, nn.Module]]:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("transformer", "blocks"),
        ("gpt_neox", "layers"),
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


def _is_quant_linear(module: nn.Module) -> bool:
    """Duck-type check for autoawq's ``WQLinear_GEMM`` / transformers'
    ``AwqLinear`` / similar. Anything with the three AWQ tensor attributes
    qualifies; we don't care what the class is.
    """
    return all(hasattr(module, attr) for attr in ("qweight", "scales", "qzeros"))


_QLINEAR_NAMES = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


def _iter_quant_linears(layer: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return ``(dotted_path_within_layer, module)`` for each quant linear
    inside a single decoder layer.
    """
    found: list[tuple[str, nn.Module]] = []
    for name, child in layer.named_modules():
        if not name:
            continue
        if not _is_quant_linear(child):
            continue
        found.append((name, child))
    return found


def _set_submodule(root: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    parent.add_module(parts[-1], new_module)


def _detect_in_out(quant_linear: nn.Module) -> tuple[int, int]:
    """Read ``(in_features, out_features)`` from a quant-linear module."""
    # Most autoawq/HF quant linears expose these directly.
    in_features = getattr(quant_linear, "in_features", None)
    out_features = getattr(quant_linear, "out_features", None)
    if in_features is None or out_features is None:
        qw = quant_linear.qweight
        in_features = qw.shape[0]
        out_features = qw.shape[1] * 8  # 4-bit pack=8
    return int(in_features), int(out_features)


def _replace_streamed_linears(
    hf_model: nn.Module,
    *,
    layer_indices: Iterable[int],
    group_size: int,
    prefer: str,
) -> dict[int, dict[str, StreamingQLinear]]:
    """Replace each streamed layer's quant linears with StreamingQLinear.

    Returns ``{layer_idx: {sub_path: streaming_module}}`` so callers can
    drive rebinds without re-walking the tree.
    """
    out: dict[int, dict[str, StreamingQLinear]] = {}
    layers = dict(_iter_decoder_layers(hf_model))
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        replacements: dict[str, StreamingQLinear] = {}
        for sub_path, qlin in _iter_quant_linears(layer):
            in_f, out_f = _detect_in_out(qlin)
            new = StreamingQLinear(
                in_features=in_f,
                out_features=out_f,
                group_size=group_size,
                prefer=prefer if prefer != "auto" else "auto",
            )
            _set_submodule(layer, sub_path, new)
            replacements[sub_path] = new
        out[layer_idx] = replacements
    return out


def _materialize_meta_buffers(
    hf_model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """After resident materialization, any parameter still on the ``meta``
    device that belongs to a non-streamed submodule needs *something*
    finite. We instantiate them as zero tensors. Streamed-layer
    StreamingQLinear submodules have no parameters, so they're untouched.

    Layernorms inside streamed layers are already resident (loaded by
    ``materialize_into_module``); this pass is a safety net for biases or
    auxiliary buffers the residency policy didn't flag.
    """
    for name, param in list(hf_model.named_parameters()):
        if param.device.type != "meta":
            continue
        new = torch.zeros(param.shape, dtype=dtype, device=device)
        try:
            _assign_dotted_attribute(hf_model, name, new)
        except AttributeError:
            logger.debug("Could not materialize meta param %s", name)


def _build_scheduler(
    hf_model: nn.Module,
    *,
    loader: AWQLoader,
    streaming_config: StreamingConfig,
) -> Optional[StreamingScheduler]:
    if not torch.cuda.is_available():
        return None

    layer_specs: list[LayerSpec] = [
        loader.index.layers[i] for i in sorted(loader.index.layers.keys())
    ]
    if not layer_specs:
        return None

    streamed_set = set(loader.streamed_layer_indices())
    for spec in layer_specs:
        spec.resident = spec.layer_idx not in streamed_set

    # Only stream the *projection* tensors. Layernorms and biases for
    # streamed layers are tiny and already resident; keep them out of the
    # slot layout so per-DMA bytes stay equal to the proj sum.
    layer_specs_streaming: list[LayerSpec] = []
    for spec in layer_specs:
        if spec.resident:
            layer_specs_streaming.append(spec)
            continue
        proj_only = LayerSpec(
            layer_idx=spec.layer_idx,
            tensors={
                k: v for k, v in spec.tensors.items()
                if "proj" in k and not v.is_resident
            },
            resident=False,
        )
        layer_specs_streaming.append(proj_only)

    def prefetch_source_fn(layer_idx: int) -> dict[str, torch.Tensor]:
        # AWQLoader.pin_layer returns full safetensors keys
        # ("model.layers.{i}.self_attn.q_proj.qweight"). The slot layout
        # uses the same keys (built from the same TensorSpec.name), so no
        # renaming is needed — return as-is filtering to proj tensors only.
        raw = loader.pin_layer(layer_idx)
        return {k: v for k, v in raw.items() if "proj" in k}

    try:
        return StreamingScheduler(
            layer_specs=layer_specs_streaming,
            prefetch_source_fn=prefetch_source_fn,
            device=torch.device("cuda"),
            num_slots=streaming_config.n_staging_slots,
        )
    except Exception as exc:
        logger.warning("scheduler construction failed: %s", exc)
        return None


def _make_pre_hook(
    scheduler: StreamingScheduler,
    layer_idx: int,
    qlinears: dict[str, StreamingQLinear],
):
    """Pre-hook: stage this layer's weights and rebind StreamingQLinears.

    ``qlinears`` maps the sub-path within the decoder layer (e.g.
    ``"self_attn.q_proj"``) to the StreamingQLinear we installed. We use
    that path to derive the full safetensors key
    (``"model.layers.{i}.{sub_path}.{kind}"``) that indexes into the slot
    views the arena returns.
    """

    def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        slot_views = scheduler.before_layer(layer_idx)
        if slot_views is None:
            return
        prefix = f"model.layers.{layer_idx}."
        for sub_path, qlin in qlinears.items():
            try:
                qweight = slot_views[f"{prefix}{sub_path}.qweight"]
                scales = slot_views[f"{prefix}{sub_path}.scales"]
                qzeros = slot_views[f"{prefix}{sub_path}.qzeros"]
            except KeyError as exc:
                raise RuntimeError(
                    f"slot views missing tensor for {sub_path}: {exc}"
                ) from exc
            bias_key = f"{prefix}{sub_path}.bias"
            bias = slot_views.get(bias_key)
            qlin.rebind(qweight=qweight, scales=scales, qzeros=qzeros, bias=bias)

    return hook


def _make_post_hook(scheduler: StreamingScheduler, layer_idx: int):
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        scheduler.after_layer(layer_idx)

    return hook


__all__ = ["StreamingCausalLM"]
