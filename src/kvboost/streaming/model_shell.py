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

import copy
import logging
import re
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from .awq_loader import (
    AWQLoader,
    LayerSpec,
    TensorSpec,
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

        cuda_available = torch.cuda.is_available()
        mps_available = _mps_available()
        want_streaming = (
            streaming_config.should_stream_model(num_layers) and cuda_available
        )

        # MPS unified-memory path: streaming makes no sense (CPU and GPU
        # share RAM), but HF's AWQ loader won't work either (autoawq's
        # CUDA kernels are missing). Build a quant-stripped skeleton,
        # replace every decoder projection with StreamingQLinear(prefer="torch"),
        # materialize everything resident on MPS.
        if (not cuda_available) and mps_available and streaming_config is not None:
            return cls._from_pretrained_mps(
                model_name_or_path,
                cfg=cfg,
                streaming_config=streaming_config,
                dtype=dtype,
                revision=revision,
                cache_dir=cache_dir,
            )

        if not want_streaming:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
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
        #
        #    We use the *index-based* walker (not the duck-type walker)
        #    because ``AutoModelForCausalLM.from_config`` under
        #    ``init_empty_weights`` does NOT always trigger HF's AWQ
        #    integration — depending on the transformers version, the
        #    skeleton's projections may be plain ``nn.Linear`` rather than
        #    ``WQLinear_GEMM``. The index walker discovers projection paths
        #    from the safetensors keys, which works for both.
        streamed_indices = set(loader.streamed_layer_indices())
        group_size = int(loader.index.quant_config.get("group_size", 128))
        streamed_qlinears = _replace_linears_for_quant_paths(
            hf_model,
            loader=loader,
            group_size=group_size,
            prefer=streaming_config.quant_kernel,
            cache_dense=False,  # CUDA: weights change per slot, can't cache
            layer_indices=streamed_indices,
        )

        # Diagnostic: if zero replacements happened, the streaming pipeline
        # is silently a no-op. Fail loudly rather than producing wrong logits.
        total_replaced = sum(len(v) for v in streamed_qlinears.values())
        if streamed_indices and total_replaced == 0:
            raise RuntimeError(
                f"Streaming path tried to replace projections in "
                f"{len(streamed_indices)} layers but replaced 0 modules. "
                f"The safetensors index either has no qweight tensors or no "
                f"layers were matched. Aborting before producing garbage."
            )
        logger.info(
            "Replaced %d quant projections across %d streamed layers",
            total_replaced,
            len(streamed_qlinears),
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

    @classmethod
    def _from_pretrained_mps(
        cls,
        model_name_or_path: str,
        *,
        cfg: Any,
        streaming_config: StreamingConfig,
        dtype: torch.dtype,
        revision: Optional[str],
        cache_dir: Optional[str],
    ) -> "StreamingCausalLM":
        """Unified-memory path for Apple Silicon.

        Builds a quant-config-stripped skeleton (so HF instantiates plain
        ``nn.Linear`` at the AWQ projection paths instead of autoawq's
        CUDA-only WQLinear_GEMM), replaces those Linears with
        :class:`StreamingQLinear` sized by the safetensors index, and
        materializes everything onto MPS.

        No scheduler runs — there is no separate VRAM to amortize transfers
        against.
        """
        from transformers import AutoModelForCausalLM

        try:
            from accelerate import init_empty_weights
        except ImportError as exc:
            raise ImportError(
                "MPS streaming path requires `accelerate`. "
                "Install with `pip install kvboost[streaming]`."
            ) from exc

        # 1. Index the checkpoint (metadata only — no tensors loaded yet).
        loader = AWQLoader(
            model_name_or_path,
            streaming_config=streaming_config,
            revision=revision,
            cache_dir=cache_dir,
            device="mps",
        )
        loader.load()

        # 2. Strip the quantization_config so HF builds plain Linear at
        #    the q_proj / k_proj / ... paths. The autoawq class path
        #    requires CUDA kernels at __init__ on some versions and is
        #    pointless here anyway.
        cfg_no_quant = copy.deepcopy(cfg)
        for attr in ("quantization_config", "_quantization_config"):
            if hasattr(cfg_no_quant, attr):
                try:
                    setattr(cfg_no_quant, attr, None)
                except Exception:
                    pass

        with init_empty_weights():
            hf_model = AutoModelForCausalLM.from_config(cfg_no_quant, torch_dtype=dtype)
        hf_model.eval()

        # 3. Replace standard Linears at AWQ projection paths with
        #    StreamingQLinear sized from the safetensors index. Binds are
        #    permanent on MPS (no slot recycling), so request the cached-
        #    dense fast path: dequant happens once at bind, forwards are
        #    plain matmuls. Trade: ~4× memory per projection vs packed.
        #
        #    The env var ``KVBOOST_MPS_CACHE_DENSE=0`` flips this off for
        #    A/B benchmarking. Default behavior is unchanged.
        import os

        cache_dense = os.environ.get("KVBOOST_MPS_CACHE_DENSE", "1") != "0"
        group_size = int(loader.index.quant_config.get("group_size", 128))
        replacements = _replace_linears_for_quant_paths(
            hf_model,
            loader=loader,
            group_size=group_size,
            prefer="torch",
            cache_dense=cache_dense,
        )

        # 4. Materialize the resident parts (embeddings, lm_head, norms,
        #    per-layer layernorms) onto MPS.
        loader.materialize_into_module(hf_model, only_resident=True)

        # 5. Materialize any leftover meta params (biases that aren't in
        #    safetensors, etc.) as zeros on MPS.
        mps_device = torch.device("mps")
        _materialize_meta_buffers(hf_model, device=mps_device, dtype=dtype)

        # 6. Bind every StreamingQLinear with its permanent weights on MPS.
        loader.bind_streaming_qlinears(replacements, device=mps_device)

        # Empty streamed_qlinears dict → no hooks installed → no scheduler.
        return cls(
            hf_model=hf_model,
            streaming_config=streaming_config,
            loader=loader,
            scheduler=None,
            streamed_qlinears={},
        )

    # ── Forward delegation ──────────────────────────────────────────────────

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Scheduler priming + sync happen via pre/post hooks installed on
        # ``hf_model`` (see ``_install_streaming_hooks``), so they fire on
        # every forward regardless of call path — including the internal
        # forwards driven by ``hf_model.generate``.
        return self.hf_model(*args, **kwargs)

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
        """Attach the per-forward scheduler-priming hook on ``hf_model`` and
        the per-streamed-layer pre/post hooks on each decoder layer.

        Model-level pre-hook: calls ``scheduler.begin_forward()`` so the
        staging pipeline is reset and the first ``num_slots`` prefetches are
        issued. Fires on *every* forward of ``hf_model``, including the
        token-by-token forwards driven by ``hf_model.generate`` — that's the
        whole point of attaching here rather than overriding
        ``StreamingCausalLM.forward``.

        Model-level post-hook: ``torch.cuda.synchronize`` on the scheduler's
        device so the caller doesn't see stale results from the transfer
        stream.

        Layer pre-hook: asks the scheduler to ensure this layer's weights
        are staged into a slot, then rebinds the layer's StreamingQLinear
        children to that slot's views.

        Layer post-hook: records the compute-done event and schedules the
        next-but-one streamed layer's prefetch into the freed slot.
        """
        if self._scheduler is None:
            return

        sched = self._scheduler
        device = sched.device

        def _model_pre(_mod: nn.Module, _inputs: tuple[Any, ...]) -> None:
            sched.begin_forward()

        def _model_post(_mod: nn.Module, _inputs: tuple[Any, ...], _out: Any) -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        # Register the model-level priming hooks first, BEFORE walking for
        # per-layer hooks — otherwise an exception in the walker would
        # leave the wrapper without the begin_forward hook, silently
        # breaking generate() on CUDA.
        self._hook_handles.append(
            self.hf_model.register_forward_pre_hook(_model_pre, with_kwargs=False)
        )
        self._hook_handles.append(
            self.hf_model.register_forward_hook(_model_post, with_kwargs=False)
        )

        if not self._streamed_qlinears:
            return

        try:
            layers = dict(_iter_decoder_layers(self.hf_model))
        except Exception as exc:
            logger.warning(
                "could not locate decoder layers for per-layer hooks: %s", exc
            )
            return

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


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_built() and backend.is_available())


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


def _replace_linears_for_quant_paths(
    hf_model: nn.Module,
    *,
    loader: AWQLoader,
    group_size: int,
    prefer: str = "torch",
    cache_dense: bool = False,
    layer_indices: Optional[Iterable[int]] = None,
) -> dict[int, dict[str, StreamingQLinear]]:
    """Replace every decoder-layer projection whose AWQ quant tensors exist
    in the loader's index with a :class:`StreamingQLinear`.

    Unlike :func:`_replace_streamed_linears` (which requires the skeleton to
    contain autoawq-style quant linears already), this variant walks the
    safetensors index to discover the projection paths and replaces whatever
    module is currently at that path — typically plain ``nn.Linear`` from a
    quant-config-stripped skeleton (MPS) **or** a skeleton built via
    ``AutoModelForCausalLM.from_config`` where HF's AWQ integration didn't
    fire (CUDA streaming).

    ``layer_indices`` restricts replacement to those layer ids. If ``None``,
    walks every decoder layer (the MPS / full-replace pattern). For CUDA
    streaming, pass the set of streamed-only indices.

    Pass ``cache_dense=True`` when binds are permanent (no slot recycling).
    The replacement modules will dequantize once on first rebind and run
    forward as a dense matmul. Costs ~4× memory per layer but eliminates
    per-forward dequant.
    """
    assert loader.index is not None
    out: dict[int, dict[str, StreamingQLinear]] = {}
    decoder_layers = dict(_iter_decoder_layers(hf_model))

    target_layers = (
        set(layer_indices)
        if layer_indices is not None
        else set(decoder_layers.keys())
    )

    for layer_idx, layer in decoder_layers.items():
        if layer_idx not in target_layers:
            continue
        layer_tensors = loader.index.layers.get(layer_idx)
        if layer_tensors is None:
            out[layer_idx] = {}
            continue

        prefix = f"model.layers.{layer_idx}."
        sub_paths: list[tuple[str, Any]] = []
        for tensor_name, spec in layer_tensors.tensors.items():
            if not tensor_name.endswith(".qweight"):
                continue
            if not tensor_name.startswith(prefix):
                continue
            sub_path = tensor_name[len(prefix):-len(".qweight")]
            sub_paths.append((sub_path, spec))

        replacements: dict[str, StreamingQLinear] = {}
        for sub_path, qw_spec in sorted(sub_paths):
            in_features = qw_spec.shape[0]
            out_features = qw_spec.shape[1] * 8  # 4-bit pack=8
            new = StreamingQLinear(
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
                prefer=prefer,
                cache_dense=cache_dense,
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
    #
    # IMPORTANT: SlotLayout cross-validates that every streamed layer has
    # the same tensor schema. The keys in spec.tensors are full safetensors
    # paths ("model.layers.{i}.self_attn.q_proj.qweight"), so they differ
    # between layers by the layer index alone. Strip that prefix so the
    # schema check sees identical keys across layers — and remember to do
    # the matching strip in prefetch_source_fn and the pre-hook below.
    layer_specs_streaming: list[LayerSpec] = []
    for spec in layer_specs:
        if spec.resident:
            layer_specs_streaming.append(spec)
            continue
        normalized: dict[str, TensorSpec] = {}
        for full_key, tspec in spec.tensors.items():
            if "proj" not in full_key:
                continue
            if tspec.is_resident:
                continue
            sub_path = _strip_layer_prefix(full_key, spec.layer_idx)
            normalized[sub_path] = TensorSpec(
                name=sub_path,
                path=tspec.path,
                shape=tspec.shape,
                dtype=tspec.dtype,
                layer_idx=tspec.layer_idx,
                is_quantized=tspec.is_quantized,
                is_resident=tspec.is_resident,
                nbytes=tspec.nbytes,
            )
        proj_only = LayerSpec(
            layer_idx=spec.layer_idx,
            tensors=normalized,
            resident=False,
        )
        layer_specs_streaming.append(proj_only)

    def prefetch_source_fn(layer_idx: int) -> dict[str, torch.Tensor]:
        # AWQLoader.pin_layer returns full safetensors keys
        # ("model.layers.{i}.self_attn.q_proj.qweight"). Normalize to the
        # same layer-relative form the slot layout was built with.
        raw = loader.pin_layer(layer_idx)
        return {
            _strip_layer_prefix(k, layer_idx): v
            for k, v in raw.items()
            if "proj" in k
        }

    try:
        return StreamingScheduler(
            layer_specs=layer_specs_streaming,
            prefetch_source_fn=prefetch_source_fn,
            device=torch.device("cuda"),
            num_slots=streaming_config.n_staging_slots,
        )
    except Exception as exc:
        # Hard fail. A silent fallback here leaves the model with mostly
        # meta-device parameters and produces garbage logits at forward
        # time — much better to surface the real error.
        raise RuntimeError(
            f"streaming scheduler construction failed: {exc}. "
            "The streamed model would have undefined weights; refusing to "
            "return a broken wrapper."
        ) from exc


def _strip_layer_prefix(name: str, layer_idx: int) -> str:
    """``model.layers.{i}.self_attn.q_proj.qweight`` → ``self_attn.q_proj.qweight``.

    The streaming pipeline keys slot views by *layer-relative* paths so the
    slot layout's per-layer schema check sees identical keys regardless of
    which layer is staged.
    """
    prefix = f"model.layers.{layer_idx}."
    return name[len(prefix):] if name.startswith(prefix) else name


def _make_pre_hook(
    scheduler: StreamingScheduler,
    layer_idx: int,
    qlinears: dict[str, StreamingQLinear],
):
    """Pre-hook: stage this layer's weights and rebind StreamingQLinears.

    ``qlinears`` maps the sub-path within the decoder layer (e.g.
    ``"self_attn.q_proj"``) to the StreamingQLinear we installed. Slot
    views are keyed by layer-relative paths
    (``"{sub_path}.{kind}"``), matching what the arena layout was built
    with in :func:`_build_scheduler`.
    """

    def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        slot_views = scheduler.before_layer(layer_idx)
        if slot_views is None:
            return
        for sub_path, qlin in qlinears.items():
            try:
                qweight = slot_views[f"{sub_path}.qweight"]
                scales = slot_views[f"{sub_path}.scales"]
                qzeros = slot_views[f"{sub_path}.qzeros"]
            except KeyError as exc:
                raise RuntimeError(
                    f"slot views missing tensor for {sub_path}: {exc}"
                ) from exc
            bias = slot_views.get(f"{sub_path}.bias")
            qlin.rebind(qweight=qweight, scales=scales, qzeros=qzeros, bias=bias)

    return hook


def _make_post_hook(scheduler: StreamingScheduler, layer_idx: int):
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        scheduler.after_layer(layer_idx)

    return hook


__all__ = ["StreamingCausalLM"]
