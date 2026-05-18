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
    fuse_gate_up_layer_spec,
    fuse_gate_up_tensors,
)
from .config import StreamingConfig
from .profile import get_profiler
from .qkv_proj import StreamingQLinear, StreamingQLinearGateUp
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

        # Strip quantization_config before from_config. Otherwise HF's AWQ
        # integration fires and constructs autoawq's WQLinear_GEMM at every
        # projection path — and that class allocates real CUDA buffers in
        # __init__ regardless of init_empty_weights (it passes an explicit
        # device). Even after we replace the streamed-layer modules, the
        # transient CUDA allocation taints peak-memory measurement (no real
        # VRAM savings) and pulls in autoawq's per-layer kernel state.
        #
        # With the config stripped, from_config builds plain nn.Linear on
        # meta. We then replace ALL projection paths (resident + streamed)
        # with StreamingQLinear, sized via the safetensors index. Resident
        # layers are bound once via bind_streaming_qlinears; streamed
        # layers are bound per-forward by the scheduler hooks.
        cfg_no_quant = _strip_quantization_config(cfg)

        with init_empty_weights():
            hf_model = AutoModelForCausalLM.from_config(cfg_no_quant, torch_dtype=dtype)
        hf_model.eval()

        streamed_indices = set(loader.streamed_layer_indices())
        resident_indices = set(loader.index.layers.keys()) - streamed_indices
        group_size = int(loader.index.quant_config.get("group_size", 128))

        # 1. Replace streamed-layer projections with StreamingQLinear
        #    (cache_dense=False so the scheduler can rebind per-forward).
        streamed_qlinears = _replace_linears_for_quant_paths(
            hf_model,
            loader=loader,
            group_size=group_size,
            prefer=streaming_config.quant_kernel,
            cache_dense=False,
            layer_indices=streamed_indices,
        )
        # 2. Replace resident-layer projections too — also StreamingQLinear,
        #    but bound once permanently. cache_dense=False keeps the weights
        #    packed (~4× less VRAM than fp16 dense); forward pays torch
        #    dequant per call, which on resident layers is the price of not
        #    depending on autoawq's CUDA kernel.
        resident_qlinears = _replace_linears_for_quant_paths(
            hf_model,
            loader=loader,
            group_size=group_size,
            prefer="torch",
            cache_dense=False,
            layer_indices=resident_indices,
        )

        total_streamed = sum(len(v) for v in streamed_qlinears.values())
        total_resident = sum(len(v) for v in resident_qlinears.values())
        if (streamed_indices or resident_indices) and (total_streamed + total_resident) == 0:
            # Show the user what we actually found so they can see whether
            # this is an architecture mismatch (hybrid attention, MoE
            # routers at non-standard paths, etc.) vs a real bug.
            sample_keys: list[str] = []
            for name in loader.index.tensors.keys():
                sample_keys.append(name)
                if len(sample_keys) >= 12:
                    break
            qweight_keys = [k for k in loader.index.tensors.keys() if k.endswith(".qweight")]
            raise RuntimeError(
                "Streaming path replaced 0 projection modules.\n"
                f"  Expected pattern: model.layers.{{i}}.{{self_attn|mlp}}.{{proj}}.qweight\n"
                f"  Total qweight tensors found in safetensors: {len(qweight_keys)}\n"
                f"  Sample qweight paths: {qweight_keys[:5]}\n"
                f"  First {len(sample_keys)} index entries: {sample_keys}\n"
                "Hybrid architectures (linear-attention / Mamba / MoE with "
                "non-standard router paths) aren't supported by the current "
                "walker. Try a standard transformer AWQ model — e.g. "
                "casperhansen/llama-3-8b-instruct-awq or Qwen/Qwen2.5-7B-Instruct-AWQ."
            )
        logger.info(
            "Replaced projections: %d resident across %d layers, "
            "%d streamed across %d layers",
            total_resident, len(resident_qlinears),
            total_streamed, len(streamed_qlinears),
        )

        # 2b. SwiGLU gate+up fusion (opt-in via StreamingConfig). Must run
        #     BEFORE bind_streaming_qlinears (which inspects which sub_paths
        #     are present) and BEFORE _build_scheduler (which builds the
        #     slot layout from the per-layer tensor schema). The fusion
        #     mutates the hf_model tree (mlp.gate_up_proj installed,
        #     .mlp wrapped in StreamingMLP) and returns updated qlinears
        #     dicts keyed on the fused sub_path.
        if streaming_config.fuse_gate_up:
            streamed_qlinears = _apply_gate_up_fusion(
                hf_model,
                layer_replacements=streamed_qlinears,
                group_size=group_size,
            )
            resident_qlinears = _apply_gate_up_fusion(
                hf_model,
                layer_replacements=resident_qlinears,
                group_size=group_size,
            )

        # 3. Materialize non-quant resident tensors (embeds, lm_head,
        #    final norm, per-layer layernorms, qkv biases on resident
        #    layers) onto GPU. Quant projection tensors (qweight/scales/
        #    qzeros) are skipped here — they're bound directly into the
        #    StreamingQLinear modules below, which don't have parameter
        #    slots for `_assign_dotted_attribute` to write into.
        loader.materialize_into_module(hf_model, only_resident=True)

        # 4. Zero out any remaining meta params (rotary inv_freq is a
        #    buffer initialized in __init__, not a meta param; safety net
        #    for things we didn't enumerate).
        _materialize_meta_buffers(hf_model, device=loader.device_spec.device, dtype=dtype)

        # 5. One-shot bind for the resident-layer StreamingQLinears: load
        #    their packed AWQ tensors from disk and call .rebind() once.
        if resident_qlinears:
            loader.bind_streaming_qlinears(
                resident_qlinears,
                device=loader.device_spec.device,
            )

        # 6. Build the scheduler that drives the per-forward DMA for
        #    streamed layers. begin_forward/before_layer/after_layer fire
        #    from the hooks installed below.
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
        cfg_no_quant = _strip_quantization_config(cfg)

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

        # 3b. SwiGLU fusion before bind — same reasoning as the CUDA
        #     path. bind_streaming_qlinears below knows how to assemble
        #     the fused mlp.gate_up_proj from gate_proj+up_proj source
        #     tensors in the checkpoint.
        if streaming_config.fuse_gate_up:
            replacements = _apply_gate_up_fusion(
                hf_model,
                layer_replacements=replacements,
                group_size=group_size,
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
        #
        # The profiler region wraps the full forward — but the
        # iteration bump and the inner regions live in
        # _install_streaming_hooks's model-pre hook so that
        # hf_model.generate's per-token forwards are counted too.
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
        profiler = get_profiler()
        # Pair a forward.total start-handle across pre/post hooks. Stored
        # as a mutable cell because nested closures can't rebind in py3.
        _forward_handle: dict[str, Any] = {"h": None}

        def _model_pre(_mod: nn.Module, _inputs: tuple[Any, ...]) -> None:
            profiler.bump_iteration()
            _forward_handle["h"] = profiler.start("model.forward.total")
            sched.begin_forward()

        def _model_post(_mod: nn.Module, _inputs: tuple[Any, ...], _out: Any) -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            profiler.end(_forward_handle["h"])
            _forward_handle["h"] = None

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

            # Per-block timing for the breakdown table: only installed
            # when the profiler is enabled (the hooks are cheap pre/post
            # callables but PyTorch still walks the hook list per
            # forward — keep them off the critical path otherwise).
            if profiler.enabled:
                for attr_name, region in (("self_attn", "attn.forward"), ("mlp", "mlp.forward")):
                    block = getattr(hf_layer, attr_name, None)
                    if block is None:
                        continue
                    pre_fn, handle_cell = _make_block_pre_hook(region, layer_idx)
                    self._hook_handles.append(
                        block.register_forward_pre_hook(pre_fn, with_kwargs=False)
                    )
                    self._hook_handles.append(
                        block.register_forward_hook(
                            _make_block_post_hook(handle_cell), with_kwargs=False
                        )
                    )

    def __del__(self) -> None:  # pragma: no cover - cleanup
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass


# ── Module-tree helpers ─────────────────────────────────────────────────────


def _strip_quantization_config(cfg: Any) -> Any:
    """Return a deep-copy of ``cfg`` with ``quantization_config`` fully
    removed from its ``__dict__``.

    Setting ``cfg.quantization_config = None`` is **not** sufficient:
    :meth:`PretrainedConfig.to_dict` (which ``GenerationConfig.from_model_config``
    invokes during ``from_config``) checks ``if "quantization_config" in
    output`` and then calls ``.to_dict()`` on the value unconditionally,
    crashing with ``AttributeError`` when the value is None. We have to
    remove the key from ``__dict__`` entirely.

    Also strips ``_pre_quantization_dtype`` and the private ``_quantization_config``
    if present, so HF's AWQ integration has no breadcrumbs to follow.
    """
    stripped = copy.deepcopy(cfg)
    for attr in (
        "quantization_config",
        "_quantization_config",
        "_pre_quantization_dtype",
    ):
        stripped.__dict__.pop(attr, None)
    return stripped


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_built() and backend.is_available())


_LAYER_COUNT_ATTRS = (
    "num_hidden_layers",
    "num_decoder_layers",
    "n_layer",
    "n_layers",
    "num_layers",
    "decoder_layers",
)
_NESTED_CONFIG_ATTRS = (
    "text_config",
    "llm_config",
    "decoder_config",
    "language_config",
    "thinker_config",
)


def _detect_num_layers(config: Any) -> int:
    """Walk a HF ``PretrainedConfig`` looking for the decoder layer count.

    Tries a wide set of attribute names first (different architectures use
    different ones), then recurses one level into known sub-config
    attributes for multi-modal / nested configs (Qwen3.5, Llama-4, etc.).
    Falls back to scanning ``config.to_dict()`` for any ``num_hidden_layers``
    / ``num_layers`` / ``n_layer`` key at any depth.
    """
    # 1. Direct attribute lookup
    for attr in _LAYER_COUNT_ATTRS:
        n = getattr(config, attr, None)
        if isinstance(n, int) and n > 0:
            return n

    # 2. Recurse into nested sub-configs (one level)
    for sub_attr in _NESTED_CONFIG_ATTRS:
        sub = getattr(config, sub_attr, None)
        if sub is None:
            continue
        for attr in _LAYER_COUNT_ATTRS:
            n = getattr(sub, attr, None)
            if isinstance(n, int) and n > 0:
                return n

    # 3. Last resort: dict scan at any depth
    try:
        cfg_dict = config.to_dict()
    except Exception:
        cfg_dict = {}

    def _scan(d: Any) -> Optional[int]:
        if isinstance(d, dict):
            for key in _LAYER_COUNT_ATTRS:
                v = d.get(key)
                if isinstance(v, int) and v > 0:
                    return v
            for v in d.values():
                hit = _scan(v)
                if hit is not None:
                    return hit
        elif isinstance(d, list):
            for v in d:
                hit = _scan(v)
                if hit is not None:
                    return hit
        return None

    hit = _scan(cfg_dict)
    if hit is not None:
        return hit

    raise ValueError(
        f"could not detect number of decoder layers from config "
        f"{type(config).__name__}. Tried attrs={_LAYER_COUNT_ATTRS}, "
        f"sub-configs={_NESTED_CONFIG_ATTRS}, and a recursive dict scan. "
        f"Top-level config keys: {sorted(cfg_dict.keys())[:20]}…"
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
                layer_idx=layer_idx,
                sub_path=sub_path,
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
                layer_idx=layer_idx,
                sub_path=sub_path,
            )
            _set_submodule(layer, sub_path, new)
            replacements[sub_path] = new
        out[layer_idx] = replacements
    return out


class StreamingMLP(nn.Module):
    """SwiGLU MLP wired to a fused ``gate_up_proj`` + ``down_proj``.

    Replaces HF's ``Qwen2MLP`` / ``LlamaMLP`` on any layer where we
    fused gate_proj and up_proj. Forward pattern is the standard
    SwiGLU contract — ``down(silu(gate(x)) * up(x))`` — only the
    matmul-and-elementwise sequence is collapsed into one fused
    call via :meth:`StreamingQLinearGateUp.forward_silu_mul`.

    The replacement carries no parameters of its own; both child
    projections are :class:`StreamingQLinear` instances that hold
    references to slot views (streamed layers) or resident weights.
    """

    def __init__(
        self,
        gate_up_proj: StreamingQLinearGateUp,
        down_proj: StreamingQLinear,
    ) -> None:
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_up_proj.forward_silu_mul(x))


def _apply_gate_up_fusion(
    hf_model: nn.Module,
    *,
    layer_replacements: dict[int, dict[str, StreamingQLinear]],
    group_size: int,
) -> dict[int, dict[str, StreamingQLinear]]:
    """For every layer in ``layer_replacements``, merge ``mlp.gate_proj``
    and ``mlp.up_proj`` into a single ``mlp.gate_up_proj`` of type
    :class:`StreamingQLinearGateUp`, then swap the layer's ``.mlp``
    submodule for a :class:`StreamingMLP` that drives the fused path.

    Returns a NEW replacements dict where each layer's ``mlp.gate_proj``
    and ``mlp.up_proj`` entries are gone, replaced by a single
    ``mlp.gate_up_proj`` entry. Layers without both projections (resident
    layers in modes where MLP isn't touched, or hybrid architectures)
    are passed through unchanged.

    The caller still uses the returned dict to drive the loader bind /
    scheduler rebind — both have been taught to handle ``mlp.gate_up_proj``
    as a fused module.
    """
    fused_replacements: dict[int, dict[str, StreamingQLinear]] = {}
    decoder_layers = dict(_iter_decoder_layers(hf_model))

    for layer_idx, qlinears in layer_replacements.items():
        gate = qlinears.get("mlp.gate_proj")
        up = qlinears.get("mlp.up_proj")
        if gate is None or up is None:
            # Nothing to fuse on this layer — keep its qlinears as-is.
            fused_replacements[layer_idx] = dict(qlinears)
            continue

        if gate.in_features != up.in_features:
            raise ValueError(
                f"layer {layer_idx}: gate.in_features ({gate.in_features}) "
                f"!= up.in_features ({up.in_features}); cannot fuse"
            )

        fused = StreamingQLinearGateUp(
            in_features=gate.in_features,
            gate_out=gate.out_features,
            up_out=up.out_features,
            group_size=group_size,
            prefer=gate.prefer,
            cache_dense=gate.cache_dense,
            layer_idx=layer_idx,
            sub_path="mlp.gate_up_proj",
        )

        hf_layer = decoder_layers[layer_idx]
        # Plant the fused module at mlp.gate_up_proj. The walker doesn't
        # need to delete the old gate_proj / up_proj attrs — wrapping
        # ``.mlp`` in StreamingMLP below makes them unreachable.
        _set_submodule(hf_layer, "mlp.gate_up_proj", fused)

        down = qlinears.get("mlp.down_proj")
        if down is None:
            raise RuntimeError(
                f"layer {layer_idx}: gate_up fused but mlp.down_proj is "
                f"missing — refusing to leave the MLP in a half-built state."
            )

        # Drop in StreamingMLP — HF's forward will now call our compact
        # path instead of its own gate/silu/up/mul/down sequence.
        _set_submodule(hf_layer, "mlp", StreamingMLP(fused, down))

        new_qlinears = {
            k: v for k, v in qlinears.items()
            if k not in ("mlp.gate_proj", "mlp.up_proj")
        }
        new_qlinears["mlp.gate_up_proj"] = fused
        fused_replacements[layer_idx] = new_qlinears

    return fused_replacements


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
        # SwiGLU fusion: collapse mlp.gate_proj + mlp.up_proj into a
        # single mlp.gate_up_proj TensorSpec so the slot layout
        # allocates one contiguous region for the fused tensors.
        if streaming_config.fuse_gate_up:
            proj_only = fuse_gate_up_layer_spec(proj_only)
        layer_specs_streaming.append(proj_only)

    def prefetch_source_fn(layer_idx: int) -> dict[str, torch.Tensor]:
        # AWQLoader.pin_layer returns full safetensors keys
        # ("model.layers.{i}.self_attn.q_proj.qweight"). Normalize to the
        # same layer-relative form the slot layout was built with.
        raw = loader.pin_layer(layer_idx)
        layer_relative = {
            _strip_layer_prefix(k, layer_idx): v
            for k, v in raw.items()
            if "proj" in k
        }
        if streaming_config.fuse_gate_up:
            # Concat mlp.gate_proj/up_proj into mlp.gate_up_proj host-side
            # before the DMA — matches the fused slot layout key schema.
            layer_relative = fuse_gate_up_tensors(layer_relative)
        return layer_relative

    device = torch.device("cuda")
    num_slots = _resolve_num_slots(
        layer_specs_streaming,
        device=device,
        streaming_config=streaming_config,
    )

    try:
        return StreamingScheduler(
            layer_specs=layer_specs_streaming,
            prefetch_source_fn=prefetch_source_fn,
            device=device,
            num_slots=num_slots,
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


def _resolve_num_slots(
    layer_specs_streaming: list[LayerSpec],
    *,
    device: torch.device,
    streaming_config: StreamingConfig,
) -> int:
    """Decide how many staging slots the scheduler should allocate.

    Behavior:

    - If ``streaming_config.n_staging_slots > 0``, use it verbatim — explicit
      user choice always wins.
    - If ``0``, probe free VRAM via ``torch.cuda.mem_get_info``, divide by
      per-slot bytes (computed from the same SlotLayout the scheduler will
      build), reserve ``auto_slots_margin_gb`` for KV cache + activations,
      and clamp to ``[2, auto_slots_max]``.

    The "auto" path errs toward fewer slots when VRAM is tight — clamps at 2
    (the minimum for the double-buffer pipeline), at most ``auto_slots_max``
    (default 4; past that the look-ahead gain falls off fast).
    """
    explicit = streaming_config.n_staging_slots
    if explicit > 0:
        logger.info("staging slots: %d (explicit)", explicit)
        return explicit

    # Build the SAME layout the scheduler will use, so slot_bytes matches
    # exactly. SlotLayout.from_layer_specs is pure metadata — no allocation.
    from .staging import SlotLayout

    layout = SlotLayout.from_layer_specs(
        layer_specs_streaming, alignment=16, streamed_only=True
    )
    slot_bytes = layout.slot_bytes

    if slot_bytes == 0:
        # No streamed tensors → no streaming. Return the minimum so the
        # scheduler still constructs cleanly.
        logger.info("staging slots: 2 (no streamed tensors, minimum)")
        return 2

    free_bytes = torch.cuda.mem_get_info(device)[0]
    margin_bytes = int(streaming_config.auto_slots_margin_gb * (1 << 30))
    usable = max(0, free_bytes - margin_bytes)
    affordable = usable // slot_bytes

    chosen = max(2, min(int(affordable), streaming_config.auto_slots_max))
    logger.info(
        "staging slots: %d (auto; slot_bytes=%.1f MB, "
        "free=%.2f GB, margin=%.2f GB, affordable=%d, cap=%d)",
        chosen,
        slot_bytes / (1 << 20),
        free_bytes / (1 << 30),
        margin_bytes / (1 << 30),
        affordable,
        streaming_config.auto_slots_max,
    )
    return chosen


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

    profiler = get_profiler()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        slot_views = scheduler.before_layer(layer_idx)
        if slot_views is None:
            return
        with profiler.region("hook.rebind", layer_idx=layer_idx):
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


def _make_block_pre_hook(name: str, layer_idx: int):
    """Pre-hook that opens a profiler region for an inner block (attn or
    mlp). Paired with :func:`_make_block_post_hook`.
    """
    profiler = get_profiler()
    handle_cell: dict[str, Any] = {"h": None}

    def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        handle_cell["h"] = profiler.start(name, layer_idx=layer_idx)

    return hook, handle_cell


def _make_block_post_hook(handle_cell: dict[str, Any]):
    profiler = get_profiler()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        profiler.end(handle_cell["h"])
        handle_cell["h"] = None

    return hook


__all__ = ["StreamingCausalLM"]
