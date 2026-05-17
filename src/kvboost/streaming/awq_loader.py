# src/kvboost/streaming/awq_loader.py

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .config import StreamingConfig

logger = logging.getLogger(__name__)

MARLIN_CACHE_DIR = Path.home() / ".cache" / "kvboost" / "marlin"
MARLIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Device detection
# =============================================================================


@dataclass(slots=True, frozen=True)
class DeviceSpec:
    device: torch.device
    kind: str  # cuda | mps | cpu

    use_pinned_memory: bool
    non_blocking: bool

    supports_marlin: bool
    supports_async_transfer: bool


def detect_device(prefer: str = "auto") -> DeviceSpec:
    prefer = prefer.lower()

    #
    # Explicit CUDA
    #

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        return DeviceSpec(
            device=torch.device("cuda"),
            kind="cuda",
            use_pinned_memory=True,
            non_blocking=True,
            supports_marlin=True,
            supports_async_transfer=True,
        )

    #
    # Explicit CPU
    #

    if prefer == "cpu":
        return DeviceSpec(
            device=torch.device("cpu"),
            kind="cpu",
            use_pinned_memory=False,
            non_blocking=False,
            supports_marlin=False,
            supports_async_transfer=False,
        )

    #
    # Explicit MPS
    #

    if prefer == "mps":
        if not (
            torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but unavailable")

        return DeviceSpec(
            device=torch.device("mps"),
            kind="mps",
            use_pinned_memory=False,
            non_blocking=False,
            supports_marlin=False,
            supports_async_transfer=False,
        )

    #
    # Auto preference order:
    # CUDA -> MPS -> CPU
    #

    if torch.cuda.is_available():
        return DeviceSpec(
            device=torch.device("cuda"),
            kind="cuda",
            use_pinned_memory=True,
            non_blocking=True,
            supports_marlin=True,
            supports_async_transfer=True,
        )

    if (
        torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        return DeviceSpec(
            device=torch.device("mps"),
            kind="mps",
            use_pinned_memory=False,
            non_blocking=False,
            supports_marlin=False,
            supports_async_transfer=False,
        )

    return DeviceSpec(
        device=torch.device("cpu"),
        kind="cpu",
        use_pinned_memory=False,
        non_blocking=False,
        supports_marlin=False,
        supports_async_transfer=False,
    )


# =============================================================================
# Metadata
# =============================================================================


@dataclass(slots=True)
class TensorSpec:
    name: str
    path: Path

    shape: tuple[int, ...]
    dtype: torch.dtype

    layer_idx: Optional[int]

    is_quantized: bool = False
    is_resident: bool = False

    nbytes: int = 0


@dataclass(slots=True)
class LayerSpec:
    layer_idx: int
    tensors: dict[str, TensorSpec] = field(default_factory=dict)

    resident: bool = False


@dataclass(slots=True)
class AWQIndex:
    model_dir: Path

    config: dict[str, Any]
    quant_config: dict[str, Any]

    tensors: dict[str, TensorSpec]
    layers: dict[int, LayerSpec]

    tied_embeddings: bool


# =============================================================================
# Loader
# =============================================================================


class AWQLoader:
    """
    Production-grade AWQ loader for KVBoost streaming.

    Goals:
    - metadata-only indexing
    - pinned host staging for CUDA
    - fast grouped shard loading
    - optional Marlin repack caching
    - CUDA/MPS portability
    """

    def __init__(
        self,
        model_name_or_path: str,
        streaming_config: StreamingConfig,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "auto",
        max_workers: int = 4,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.streaming_config = streaming_config
        self.revision = revision
        self.cache_dir = cache_dir

        self.device_spec = detect_device(device)

        self.max_workers = max_workers

        self.model_dir: Optional[Path] = None
        self.index: Optional[AWQIndex] = None

        #
        # Runtime tensor stores
        #

        self._resident_tensors: dict[str, torch.Tensor] = {}
        self._pinned_tensors: dict[str, torch.Tensor] = {}

    # =========================================================================
    # Public API
    # =========================================================================

    def load(self) -> AWQIndex:
        """
        Download + index checkpoint metadata.
        """

        logger.info(
            "Loading model with backend=%s",
            self.device_spec.kind,
        )

        self.model_dir = Path(
            snapshot_download(
                repo_id=self.model_name_or_path,
                revision=self.revision,
                cache_dir=self.cache_dir,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )

        config = self._load_json("config.json")
        quant_config = self._load_quant_config()

        tensors = self._build_tensor_index()
        layers = self._build_layer_index(tensors)

        self.index = AWQIndex(
            model_dir=self.model_dir,
            config=config,
            quant_config=quant_config,
            tensors=tensors,
            layers=layers,
            tied_embeddings=config.get(
                "tie_word_embeddings",
                False,
            ),
        )

        self._apply_residency_policy()

        return self.index

    @torch.inference_mode()
    def materialize_resident_tensors(self) -> None:
        """
        Load resident tensors directly onto device.
        """

        assert self.index is not None

        resident_specs = [
            spec
            for spec in self.index.tensors.values()
            if spec.is_resident
        ]

        #
        # Group by shard for fewer file opens
        #

        grouped: dict[Path, list[TensorSpec]] = {}

        for spec in resident_specs:
            grouped.setdefault(spec.path, []).append(spec)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
        ) as pool:

            futures = []

            for path, specs in grouped.items():
                futures.append(
                    pool.submit(
                        self._load_shard_resident,
                        path,
                        specs,
                    )
                )

            for future in futures:
                loaded = future.result()
                self._resident_tensors.update(loaded)

        self._alias_tied_embeddings()

    @torch.inference_mode()
    def pin_layer(
        self,
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Return a layer's streamed tensors as pinned-host tensors.

        Cached: on the second call for the same layer we hand back the
        already-pinned tensors and skip the safetensors read entirely. The
        streaming scheduler calls this once per layer per token, so missing
        the cache pays full disk I/O on every decode step — that's a 3+
        second per-token regression on a 32B model.
        """

        assert self.index is not None

        layer = self.index.layers[layer_idx]
        needed = [s for s in layer.tensors.values() if not s.is_resident]

        # Cache hit: every needed tensor is already in _pinned_tensors.
        # Strict all-or-nothing — partial hits force a re-read so we don't
        # mix tensors from different load passes (defensive; in practice
        # we either pinned the whole layer or none of it).
        if needed and all(s.name in self._pinned_tensors for s in needed):
            return {s.name: self._pinned_tensors[s.name] for s in needed}

        grouped: dict[Path, list[TensorSpec]] = {}
        for spec in needed:
            grouped.setdefault(spec.path, []).append(spec)

        out: dict[str, torch.Tensor] = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
        ) as pool:

            futures = []

            for path, specs in grouped.items():
                futures.append(
                    pool.submit(
                        self._load_shard_pinned,
                        path,
                        specs,
                    )
                )

            for future in futures:
                out.update(future.result())

        self._pinned_tensors.update(out)

        return out

    def get_resident_tensor(
        self,
        name: str,
    ) -> torch.Tensor:
        return self._resident_tensors[name]

    def streamed_layer_indices(self) -> list[int]:
        """Layer indices whose projection tensors are NOT resident."""
        assert self.index is not None
        streamed: list[int] = []
        for layer_idx, layer in sorted(self.index.layers.items()):
            has_streamed_proj = any(
                ("proj" in name) and (not spec.is_resident)
                for name, spec in layer.tensors.items()
            )
            if has_streamed_proj:
                streamed.append(layer_idx)
        return streamed

    @torch.inference_mode()
    def materialize_into_module(
        self,
        hf_model: "torch.nn.Module",
        *,
        only_resident: bool = True,
        skip_quant_projections: bool = True,
    ) -> None:
        """Write resident tensors into the matching submodules of ``hf_model``.

        Walks each tensor that ``_apply_residency_policy`` flagged resident,
        navigates the dotted path on ``hf_model``, and assigns the loaded
        tensor to the leaf attribute (either as a fresh ``nn.Parameter`` or a
        plain attribute, depending on what's there).

        This is the bridge that lets us use ``accelerate.init_empty_weights``
        for the skeleton and then selectively materialize only the layers that
        should live in VRAM.

        ``skip_quant_projections`` (default True) skips ``*.qweight``,
        ``*.scales``, ``*.qzeros`` tensors. When the streaming pipeline
        replaces projection modules with :class:`StreamingQLinear` (which
        has no ``qweight`` parameter slot), naively assigning those tensors
        via ``setattr`` creates **orphaned duplicate** allocations — they
        sit on the new module as bare attributes while the real binding
        happens later via :meth:`bind_streaming_qlinears` into the
        ``_qweight`` / ``_scales`` / ``_qzeros`` slots. Skipping them here
        avoids that double-allocation (~2 GiB for a 32B model with 8
        resident layers).
        """
        assert self.index is not None

        def _is_quant_proj(name: str) -> bool:
            return (
                name.endswith(".qweight")
                or name.endswith(".scales")
                or name.endswith(".qzeros")
            )

        wanted = [
            spec for spec in self.index.tensors.values()
            if (spec.is_resident if only_resident else True)
            and not (skip_quant_projections and _is_quant_proj(spec.name))
        ]

        by_shard: dict[Path, list[TensorSpec]] = {}
        for spec in wanted:
            by_shard.setdefault(spec.path, []).append(spec)

        for shard_path, specs in by_shard.items():
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for spec in specs:
                    tensor = f.get_tensor(spec.name)
                    tensor = tensor.to(
                        self.device_spec.device,
                        non_blocking=False,
                    )
                    _assign_dotted_attribute(hf_model, spec.name, tensor)

        # Re-alias tied embeddings if they were materialized separately.
        if self.index.tied_embeddings:
            try:
                embed = _resolve_dotted_attribute(hf_model, "model.embed_tokens.weight")
                _assign_dotted_attribute(hf_model, "lm_head.weight", embed)
            except AttributeError:
                pass

    @torch.inference_mode()
    def bind_streaming_qlinears(
        self,
        layer_replacements: dict[int, dict[str, Any]],
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """One-shot bind: load each StreamingQLinear's quant tensors from
        disk and rebind permanently.

        Used by the unified-memory (MPS / CPU) path where no scheduler runs
        — weights are materialized once on the target device and the
        rebind never changes across forwards.

        ``layer_replacements`` is ``{layer_idx: {sub_path: StreamingQLinear}}``
        where ``sub_path`` is the within-layer dotted path (e.g.
        ``"self_attn.q_proj"``). The corresponding safetensors keys are
        derived as ``model.layers.{layer_idx}.{sub_path}.{kind}`` for
        ``kind in {qweight, scales, qzeros, bias}``.
        """
        assert self.index is not None
        target_device = device if device is not None else self.device_spec.device

        # Group every needed tensor by its shard for one open per shard.
        by_shard: dict[Path, list[tuple[int, str, str, TensorSpec]]] = {}
        for layer_idx, qlinears in layer_replacements.items():
            for sub_path in qlinears:
                base = f"model.layers.{layer_idx}.{sub_path}"
                for kind in ("qweight", "scales", "qzeros", "bias"):
                    tensor_name = f"{base}.{kind}"
                    spec = self.index.tensors.get(tensor_name)
                    if spec is None:
                        continue  # bias is often absent in AWQ checkpoints
                    by_shard.setdefault(spec.path, []).append(
                        (layer_idx, sub_path, kind, spec)
                    )

        loaded: dict[tuple[int, str, str], torch.Tensor] = {}
        for shard_path, items in by_shard.items():
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for layer_idx, sub_path, kind, spec in items:
                    tensor = f.get_tensor(spec.name).to(target_device)
                    loaded[(layer_idx, sub_path, kind)] = tensor

        for layer_idx, qlinears in layer_replacements.items():
            for sub_path, qlin in qlinears.items():
                key = (layer_idx, sub_path)
                qlin.rebind(
                    qweight=loaded[(*key, "qweight")],
                    scales=loaded[(*key, "scales")],
                    qzeros=loaded[(*key, "qzeros")],
                    bias=loaded.get((*key, "bias")),
                )

    # =========================================================================
    # Indexing
    # =========================================================================

    def _build_tensor_index(
        self,
    ) -> dict[str, TensorSpec]:

        assert self.model_dir is not None

        tensor_specs: dict[str, TensorSpec] = {}

        safetensor_files = sorted(
            self.model_dir.glob("*.safetensors")
        )

        for shard_path in safetensor_files:

            with safe_open(
                shard_path,
                framework="pt",
                device="cpu",
            ) as f:

                for key in f.keys():

                    tensor = f.get_tensor(key)

                    spec = TensorSpec(
                        name=key,
                        path=shard_path,
                        shape=tuple(tensor.shape),
                        dtype=tensor.dtype,
                        layer_idx=self._extract_layer_idx(key),
                        is_quantized=(
                            "qweight" in key
                            or "qzeros" in key
                        ),
                        nbytes=tensor.numel()
                        * tensor.element_size(),
                    )

                    tensor_specs[key] = spec

        logger.info(
            "Indexed %d tensors",
            len(tensor_specs),
        )

        return tensor_specs

    def _build_layer_index(
        self,
        tensors: dict[str, TensorSpec],
    ) -> dict[int, LayerSpec]:

        layers: dict[int, LayerSpec] = {}

        for spec in tensors.values():

            if spec.layer_idx is None:
                continue

            if spec.layer_idx not in layers:
                layers[spec.layer_idx] = LayerSpec(
                    layer_idx=spec.layer_idx,
                )

            layers[spec.layer_idx].tensors[spec.name] = spec

        return layers

    # =========================================================================
    # Residency policy
    # =========================================================================

    def _apply_residency_policy(self) -> None:
        """
        Mark resident tensors/layers.
        """

        assert self.index is not None

        num_layers = len(self.index.layers)

        resident_layers = set(
            range(self.streaming_config.keep_first_k)
        )

        resident_layers.update(
            range(
                max(
                    0,
                    num_layers
                    - self.streaming_config.keep_last_k,
                ),
                num_layers,
            )
        )

        for spec in self.index.tensors.values():

            resident = False

            #
            # Global always-resident tensors
            #

            if any(
                x in spec.name
                for x in [
                    "embed_tokens",
                    "lm_head",
                    "norm",
                ]
            ):
                resident = True

            #
            # Early/late layers
            #

            if (
                spec.layer_idx is not None
                and spec.layer_idx in resident_layers
            ):
                resident = True

            #
            # FFN-only streaming mode
            #

            if (
                self.streaming_config.use_ffn_only_streaming
                and spec.layer_idx is not None
            ):
                if any(
                    x in spec.name
                    for x in [
                        "self_attn",
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                    ]
                ):
                    resident = True

            spec.is_resident = resident

    # =========================================================================
    # Shard loading
    # =========================================================================

    @torch.inference_mode()
    def _load_shard_resident(
        self,
        shard_path: Path,
        specs: list[TensorSpec],
    ) -> dict[str, torch.Tensor]:

        out: dict[str, torch.Tensor] = {}

        with safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as f:

            for spec in specs:

                tensor = f.get_tensor(spec.name)

                tensor = tensor.to(
                    self.device_spec.device,
                    non_blocking=self.device_spec.non_blocking,
                )

                out[spec.name] = tensor

        return out

    @torch.inference_mode()
    def _load_shard_pinned(
        self,
        shard_path: Path,
        specs: list[TensorSpec],
    ) -> dict[str, torch.Tensor]:

        out: dict[str, torch.Tensor] = {}

        with safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as f:

            for spec in specs:

                #
                # Reuse cached pinned tensors
                #

                if spec.name in self._pinned_tensors:
                    out[spec.name] = self._pinned_tensors[
                        spec.name
                    ]
                    continue

                tensor = f.get_tensor(spec.name)

                #
                # CUDA:
                # pinned host memory for async DMA
                #
                # MPS/CPU:
                # pageable host tensor
                #

                host = torch.empty_like(
                    tensor,
                    pin_memory=self.device_spec.use_pinned_memory,
                )

                host.copy_(tensor)

                #
                # Marlin repack
                #

                if (
                    self.device_spec.supports_marlin
                    and "qweight" in spec.name
                ):
                    host = self._maybe_repack_marlin(
                        spec,
                        host,
                    )

                out[spec.name] = host

        return out

    # =========================================================================
    # Marlin repack
    # =========================================================================

    def _maybe_repack_marlin(
        self,
        spec: TensorSpec,
        tensor: torch.Tensor,
    ) -> torch.Tensor:

        cache_path = self._marlin_cache_path(spec)

        #
        # Fast path
        #

        if cache_path.exists():
            return torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )

        logger.info(
            "Repacking AWQ tensor for Marlin: %s",
            spec.name,
        )

        repacked = self._call_marlin_repack(spec, tensor)

        torch.save(
            repacked,
            cache_path,
        )

        return repacked

    def _call_marlin_repack(
        self,
        spec: TensorSpec,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the vendored Marlin repack kernel if it's importable;
        otherwise return the tensor unchanged. Repack is shape-preserving from
        the caller's perspective — it only reorders bits inside the int32
        packs to match Marlin's expected layout.
        """
        try:
            from .kernels.marlin import awq_marlin_repack, marlin_awq_available
        except Exception:  # pragma: no cover
            return tensor.contiguous()

        if not marlin_awq_available():
            return tensor.contiguous()

        pack = 8
        in_features = spec.shape[0]
        out_features = spec.shape[1] * pack
        try:
            return awq_marlin_repack(
                tensor,
                in_features=in_features,
                out_features=out_features,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Marlin repack kernel rejected %s (%s); using raw layout",
                spec.name,
                exc,
            )
            return tensor.contiguous()

    def _marlin_cache_path(
        self,
        spec: TensorSpec,
    ) -> Path:

        h = hashlib.sha256()

        h.update(str(spec.path).encode())
        h.update(str(spec.name).encode())
        h.update(str(spec.shape).encode())

        sha = h.hexdigest()[:16]

        safe_name = spec.name.replace(".", "_")

        return MARLIN_CACHE_DIR / f"{safe_name}_{sha}.pt"

    # =========================================================================
    # Tied embeddings
    # =========================================================================

    def _alias_tied_embeddings(self) -> None:

        assert self.index is not None

        if not self.index.tied_embeddings:
            return

        embed_key = None
        lm_head_key = None

        for key in self._resident_tensors:

            if "embed_tokens.weight" in key:
                embed_key = key

            if "lm_head.weight" in key:
                lm_head_key = key

        if embed_key and lm_head_key:
            self._resident_tensors[lm_head_key] = (
                self._resident_tensors[embed_key]
            )

            logger.info(
                "Aliased tied embeddings"
            )

    # =========================================================================
    # Config helpers
    # =========================================================================

    def _load_json(
        self,
        filename: str,
    ) -> dict[str, Any]:

        assert self.model_dir is not None

        path = self.model_dir / filename

        with open(path) as f:
            return json.load(f)

    def _load_quant_config(
        self,
    ) -> dict[str, Any]:

        assert self.model_dir is not None

        # Legacy: standalone quantize_config.json next to the model.
        for filename in [
            "quantize_config.json",
            "quant_config.json",
        ]:
            path = self.model_dir / filename
            if path.exists():
                with open(path) as f:
                    return json.load(f)

        # Modern transformers/AutoAWQ format: quantization config is
        # embedded inside config.json under the ``quantization_config`` key.
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                full_config = json.load(f)
            embedded = full_config.get("quantization_config")
            if isinstance(embedded, dict):
                return embedded

        raise FileNotFoundError(
            "No AWQ quantization config found (looked for quantize_config.json, "
            "quant_config.json, and config.json::quantization_config)."
        )

    # =========================================================================
    # Utils
    # =========================================================================

    @staticmethod
    def _extract_layer_idx(
        tensor_name: str,
    ) -> Optional[int]:

        #
        # model.layers.12.self_attn.q_proj.qweight
        #

        parts = tensor_name.split(".")

        for i, part in enumerate(parts):
            if part == "layers":
                return int(parts[i + 1])

        return None


# =============================================================================
# Module-tree helpers (free functions — used by materialize_into_module and
# the streaming layer-replacement walker in model_shell.py)
# =============================================================================


def _resolve_dotted_attribute(root: "torch.nn.Module", dotted: str) -> Any:
    parts = dotted.split(".")
    obj: Any = root
    for p in parts:
        obj = getattr(obj, p)
    return obj


def _assign_dotted_attribute(
    root: "torch.nn.Module",
    dotted: str,
    value: torch.Tensor,
) -> None:
    """Navigate ``root`` to the parent of the dotted path and set the leaf.

    If the leaf currently exists as an ``nn.Parameter``, wrap ``value`` in a
    fresh ``nn.Parameter`` so PyTorch's parameter machinery (and state_dict)
    continues to see it. Otherwise plain ``setattr``.

    Handles meta-device parameters from ``accelerate.init_empty_weights``:
    those land in ``module._parameters`` and we replace them in-place.
    """
    import torch.nn as nn

    parts = dotted.split(".")
    parent: Any = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf = parts[-1]

    existing = None
    if isinstance(parent, nn.Module):
        existing = parent._parameters.get(leaf)
    if existing is None and hasattr(parent, leaf):
        existing = getattr(parent, leaf)

    if isinstance(existing, nn.Parameter):
        parent._parameters[leaf] = nn.Parameter(value, requires_grad=False)
    elif isinstance(parent, nn.Module) and leaf in parent._buffers:
        parent._buffers[leaf] = value
    else:
        setattr(parent, leaf, value)