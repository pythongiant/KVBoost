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


# Process-wide latch: tripped the first time a pinned host alloc fails.
# Pinning isn't binary "works or doesn't" on locked-down containers —
# cudaHostAlloc can succeed for small sizes and fail under accumulated
# pressure when RLIMIT_MEMLOCK is tight (it surfaces as cudaErrorInvalidValue
# from the underlying mlock). A reactive latch + per-call fallback handles
# both "totally disabled" and "works until it doesn't" cases.
import threading as _threading

_PINNING_AVAILABLE = True
_PINNING_LATCH_LOCK = _threading.Lock()


def _alloc_host_like(stub: torch.Tensor) -> torch.Tensor:
    """Allocate a tensor shaped/typed like ``stub``, pinned if possible,
    pageable otherwise. Trips ``_PINNING_AVAILABLE`` False on first failure
    so subsequent calls skip the pinned attempt entirely.
    """
    global _PINNING_AVAILABLE
    if _PINNING_AVAILABLE:
        try:
            return torch.empty_like(stub, pin_memory=True)
        except Exception as exc:
            memlock_note = ""
            try:
                import resource

                soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
                if soft == resource.RLIM_INFINITY:
                    memlock_note = " (RLIMIT_MEMLOCK=unlimited)"
                else:
                    memlock_note = (
                        f" (RLIMIT_MEMLOCK soft={soft} bytes, hard={hard})"
                    )
            except Exception:
                pass
            with _PINNING_LATCH_LOCK:
                if _PINNING_AVAILABLE:
                    _PINNING_AVAILABLE = False
                    logger.warning(
                        "Pinned host allocation failed (%s: %s)%s — disabling "
                        "pinned memory for the remainder of this process. "
                        "Streaming H2D copies will fall back to pageable + "
                        "synchronous; streaming overlap is lost. To restore "
                        "async DMA, raise RLIMIT_MEMLOCK (e.g. "
                        "`ulimit -l unlimited`, or container flags "
                        "--ulimit memlock=-1 / --cap-add IPC_LOCK).",
                        type(exc).__name__,
                        exc,
                        memlock_note,
                    )
    return torch.empty_like(stub, pin_memory=False)


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

    @torch.no_grad()
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

    @torch.no_grad()
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

        When ``streaming_config.fuse_gate_up`` is set, the returned dict
        contains a pre-fused, pinned ``mlp.gate_up_proj.{kind}`` entry in
        place of the source ``mlp.gate_proj.{kind}`` + ``mlp.up_proj.{kind}``
        entries. The fusion buffer is allocated once per layer and the
        source entries in ``_pinned_tensors`` are rewritten as ``.narrow()``
        views into it, so subsequent calls are pure dict lookups.
        """

        assert self.index is not None

        layer = self.index.layers[layer_idx]
        needed = [s for s in layer.tensors.values() if not s.is_resident]

        fuse = self.streaming_config.fuse_gate_up

        # Cache hit: every needed tensor is already in _pinned_tensors.
        # Strict all-or-nothing — partial hits force a re-read so we don't
        # mix tensors from different load passes (defensive; in practice
        # we either pinned the whole layer or none of it).
        if needed and all(s.name in self._pinned_tensors for s in needed):
            cached = {s.name: self._pinned_tensors[s.name] for s in needed}
            if fuse:
                return self._with_fused_view(cached, layer_idx)
            return cached

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

        if fuse:
            return self._with_fused_view(out, layer_idx)
        return out

    def _with_fused_view(
        self,
        tensors: dict[str, torch.Tensor],
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Replace source ``mlp.gate_proj.{kind}`` + ``mlp.up_proj.{kind}``
        entries with a single pre-fused, pinned ``mlp.gate_up_proj.{kind}``
        entry per kind.

        The fused buffer is allocated once per (layer, kind) on first call
        and stashed in ``_pinned_tensors`` under the fused key. The source
        entries in ``_pinned_tensors`` are rebound to ``.narrow()`` views
        of the fused buffer; this keeps the cache-hit check valid (source
        names still resolve) while freeing the standalone source pinned
        allocations.

        Returned dict has the fused key and no gate/up sources, so the
        downstream slot layout (built with the fused schema) finds the
        tensors it expects without any per-token ``torch.cat``.
        """
        gate_prefix = f"model.layers.{layer_idx}.{_GATE_SUB_PATH}."
        up_prefix = f"model.layers.{layer_idx}.{_UP_SUB_PATH}."
        fused_prefix = f"model.layers.{layer_idx}.{_FUSED_SUB_PATH}."

        gate_by_kind: dict[str, torch.Tensor] = {}
        up_by_kind: dict[str, torch.Tensor] = {}
        passthrough: dict[str, torch.Tensor] = {}

        for name, tensor in tensors.items():
            if name.startswith(gate_prefix):
                gate_by_kind[name[len(gate_prefix):]] = tensor
            elif name.startswith(up_prefix):
                up_by_kind[name[len(up_prefix):]] = tensor
            else:
                passthrough[name] = tensor

        if not gate_by_kind and not up_by_kind:
            return passthrough

        if set(gate_by_kind.keys()) != set(up_by_kind.keys()):
            raise ValueError(
                f"gate/up kind mismatch — gate has {sorted(gate_by_kind)}, "
                f"up has {sorted(up_by_kind)}. Did you skip the bias on one side?"
            )

        result = dict(passthrough)
        for kind, gate in gate_by_kind.items():
            up = up_by_kind[kind]
            fused_name = fused_prefix + kind
            fused = self._pinned_tensors.get(fused_name)
            if fused is None:
                fused = self._build_fused_pinned(gate, up)
                self._pinned_tensors[fused_name] = fused
                # Rebind source entries to views of the fused buffer so the
                # cache-hit path in pin_layer() still resolves them, and
                # release the original standalone source allocations.
                gate_size = gate.shape[-1]
                up_size = up.shape[-1]
                self._pinned_tensors[gate_prefix + kind] = fused.narrow(-1, 0, gate_size)
                self._pinned_tensors[up_prefix + kind] = fused.narrow(-1, gate_size, up_size)
            result[fused_name] = fused

        return result

    def _build_fused_pinned(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        """Allocate a pinned tensor sized for ``cat([gate, up], dim=-1)``
        and copy both halves into it. One pinned alloc + two memcpys total,
        amortized across the lifetime of the loader.
        """
        if gate.shape[:-1] != up.shape[:-1]:
            raise ValueError(
                f"gate/up shape mismatch on non-cat dims: "
                f"{tuple(gate.shape)} vs {tuple(up.shape)}"
            )
        if gate.dtype != up.dtype:
            raise ValueError(
                f"gate/up dtype mismatch: {gate.dtype} vs {up.dtype}"
            )

        fused_shape = list(gate.shape)
        fused_shape[-1] = gate.shape[-1] + up.shape[-1]
        stub = torch.empty(fused_shape, dtype=gate.dtype, device="cpu")
        if self.device_spec.use_pinned_memory:
            fused = _alloc_host_like(stub)
        else:
            fused = stub
        fused.narrow(-1, 0, gate.shape[-1]).copy_(gate)
        fused.narrow(-1, gate.shape[-1], up.shape[-1]).copy_(up)
        return fused

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

    @torch.no_grad()
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

    @torch.no_grad()
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

        # Resolve each sub_path to the safetensors source paths it needs.
        # For the SwiGLU-fused module ``mlp.gate_up_proj``, the checkpoint
        # has no key under that name — instead pull from ``mlp.gate_proj``
        # and ``mlp.up_proj`` and concat along ``dim=-1`` on the host
        # before binding. Bias is rare in AWQ MLPs, so handle the all-or-
        # nothing case (both present → concat; both absent → None).
        def _source_sub_paths(sub_path: str) -> list[str]:
            if sub_path == _FUSED_SUB_PATH:
                return [_GATE_SUB_PATH, _UP_SUB_PATH]
            return [sub_path]

        # Group every needed tensor by its shard for one open per shard.
        # The tuple now carries (layer_idx, fused_or_real_sub_path,
        # source_sub_path, kind, spec) so the loader knows which fused
        # slot the loaded tensor will eventually concat into.
        by_shard: dict[Path, list[tuple[int, str, str, str, TensorSpec]]] = {}
        for layer_idx, qlinears in layer_replacements.items():
            for sub_path in qlinears:
                for src in _source_sub_paths(sub_path):
                    base = f"model.layers.{layer_idx}.{src}"
                    for kind in ("qweight", "scales", "qzeros", "bias"):
                        tensor_name = f"{base}.{kind}"
                        spec = self.index.tensors.get(tensor_name)
                        if spec is None:
                            continue  # bias is often absent in AWQ
                        by_shard.setdefault(spec.path, []).append(
                            (layer_idx, sub_path, src, kind, spec)
                        )

        # Keyed by (layer_idx, target_sub_path, source_sub_path, kind) so
        # we can pull both halves of a fused module back out in order.
        loaded: dict[tuple[int, str, str, str], torch.Tensor] = {}
        for shard_path, items in by_shard.items():
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for layer_idx, sub_path, src, kind, spec in items:
                    tensor = f.get_tensor(spec.name).to(target_device)
                    loaded[(layer_idx, sub_path, src, kind)] = tensor

        for layer_idx, qlinears in layer_replacements.items():
            for sub_path, qlin in qlinears.items():
                sources = _source_sub_paths(sub_path)

                def _gather(kind: str) -> Optional[torch.Tensor]:
                    parts = [
                        loaded.get((layer_idx, sub_path, src, kind))
                        for src in sources
                    ]
                    if all(p is None for p in parts):
                        return None
                    if any(p is None for p in parts):
                        raise RuntimeError(
                            f"layer {layer_idx} {sub_path}.{kind}: partial "
                            f"presence across sources {sources} — refuse to "
                            f"bind a half-fused tensor."
                        )
                    if len(parts) == 1:
                        return parts[0]
                    return torch.cat(parts, dim=-1)  # type: ignore[arg-type]

                qlin.rebind(
                    qweight=_gather("qweight"),
                    scales=_gather("scales"),
                    qzeros=_gather("qzeros"),
                    bias=_gather("bias"),
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

    @torch.no_grad()
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

    @torch.no_grad()
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

                if self.device_spec.use_pinned_memory:
                    host = _alloc_host_like(tensor)
                else:
                    host = torch.empty_like(tensor)

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
            loaded = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
            return self._ensure_pinned(loaded)

        logger.info(
            "Repacking AWQ tensor for Marlin: %s",
            spec.name,
        )

        repacked = self._call_marlin_repack(spec, tensor)

        torch.save(
            repacked,
            cache_path,
        )

        return self._ensure_pinned(repacked)

    def _ensure_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin ``tensor`` if pinned memory is enabled and it isn't already.

        ``torch.save`` / ``torch.load`` round-trips drop the pinned flag, and
        ``tensor.contiguous()`` returns a non-pinned copy when the input is
        already contiguous-but-pinned in some PyTorch builds. Without this,
        the streaming H2D copy at ``staging.copy_from_host`` would silently
        downgrade ``non_blocking=True`` to a synchronous transfer.
        """
        if not self.device_spec.use_pinned_memory:
            return tensor
        if tensor.is_pinned():
            return tensor
        # _alloc_host_like falls back to pageable if pinning has been
        # latched off (e.g. RLIMIT_MEMLOCK exhausted mid-run).
        host = _alloc_host_like(tensor)
        host.copy_(tensor)
        return host

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


# =============================================================================
# SwiGLU fusion helpers
# =============================================================================


_GATE_SUB_PATH = "mlp.gate_proj"
_UP_SUB_PATH = "mlp.up_proj"
_FUSED_SUB_PATH = "mlp.gate_up_proj"


def fuse_gate_up_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    key_prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Concat ``mlp.gate_proj.{kind}`` + ``mlp.up_proj.{kind}`` along
    ``dim=1`` into a single ``mlp.gate_up_proj.{kind}`` entry, for each
    AWQ tensor kind present (qweight, scales, qzeros, bias).

    AWQ packs 8 four-bit nibbles per int32 along ``out_features``, so
    concatenating ``dim=1`` is byte-safe when both gate_out and up_out
    are multiples of 8 (always true for SwiGLU intermediate sizes).

    Tensors not belonging to gate/up are passed through unchanged.

    ``key_prefix`` is prepended to all three of ``{gate, up, fused}``
    sub-paths when the loader is using full safetensors keys
    (``model.layers.{i}.mlp.gate_proj.qweight``). For layer-relative
    keys (``mlp.gate_proj.qweight``), pass ``""``.
    """
    gate_prefix = f"{key_prefix}{_GATE_SUB_PATH}."
    up_prefix = f"{key_prefix}{_UP_SUB_PATH}."
    fused_prefix = f"{key_prefix}{_FUSED_SUB_PATH}."

    # Collect gate/up per kind so we can decide which fused keys to emit.
    gate_by_kind: dict[str, torch.Tensor] = {}
    up_by_kind: dict[str, torch.Tensor] = {}
    passthrough: dict[str, torch.Tensor] = {}

    for key, tensor in tensors.items():
        if key.startswith(gate_prefix):
            gate_by_kind[key[len(gate_prefix):]] = tensor
        elif key.startswith(up_prefix):
            up_by_kind[key[len(up_prefix):]] = tensor
        else:
            passthrough[key] = tensor

    if not gate_by_kind and not up_by_kind:
        return passthrough

    if set(gate_by_kind.keys()) != set(up_by_kind.keys()):
        # Defensive: a partial schema would silently emit a half-fused
        # tensor and we'd hit a CUDA shape error mid-forward. Better
        # to fail with the names so the user can fix the bind input.
        raise ValueError(
            f"gate/up kind mismatch — gate has {sorted(gate_by_kind)}, "
            f"up has {sorted(up_by_kind)}. Did you skip the bias on one side?"
        )

    fused: dict[str, torch.Tensor] = dict(passthrough)
    for kind, g_tensor in gate_by_kind.items():
        u_tensor = up_by_kind[kind]
        # bias is 1-D ``(out_features,)``; everything else is 2-D with
        # out_features on dim=1. Concat on the last dim either way.
        fused[fused_prefix + kind] = torch.cat([g_tensor, u_tensor], dim=-1)
    return fused


def fuse_gate_up_layer_spec(layer: "LayerSpec") -> "LayerSpec":
    """Return a new ``LayerSpec`` with gate_proj/up_proj TensorSpecs
    merged into a single ``mlp.gate_up_proj.{kind}`` entry.

    Slot layout drives byte placement from these specs; if the layout
    sees two separate gate/up entries while the loader emits a fused
    tensor, the arena will try to memcpy a too-big tensor into a too-
    small slot region. Keep them in lockstep by transforming both
    sides via this helper.

    Layer-relative keys only (sub_path form, no ``model.layers.{i}.``).
    """
    gate_specs: dict[str, "TensorSpec"] = {}
    up_specs: dict[str, "TensorSpec"] = {}
    passthrough: dict[str, "TensorSpec"] = {}

    for name, spec in layer.tensors.items():
        if name.startswith(_GATE_SUB_PATH + "."):
            gate_specs[name[len(_GATE_SUB_PATH) + 1:]] = spec
        elif name.startswith(_UP_SUB_PATH + "."):
            up_specs[name[len(_UP_SUB_PATH) + 1:]] = spec
        else:
            passthrough[name] = spec

    if not gate_specs or not up_specs:
        return layer

    merged: dict[str, "TensorSpec"] = dict(passthrough)
    for kind, g_spec in gate_specs.items():
        u_spec = up_specs[kind]
        if g_spec.dtype != u_spec.dtype:
            raise ValueError(
                f"gate/up dtype mismatch for {kind}: "
                f"{g_spec.dtype} vs {u_spec.dtype}"
            )
        # Stack along dim=-1 (out_features). Both 1-D bias and 2-D
        # qweight/scales/qzeros work the same way — last dim grows.
        new_shape = list(g_spec.shape)
        new_shape[-1] = g_spec.shape[-1] + u_spec.shape[-1]
        merged_name = f"{_FUSED_SUB_PATH}.{kind}"
        merged[merged_name] = TensorSpec(
            name=merged_name,
            path=g_spec.path,  # not load-source-of-truth post-fusion
            shape=tuple(new_shape),
            dtype=g_spec.dtype,
            layer_idx=g_spec.layer_idx,
            is_quantized=g_spec.is_quantized,
            is_resident=g_spec.is_resident,
            nbytes=g_spec.nbytes + u_spec.nbytes,
        )

    return LayerSpec(
        layer_idx=layer.layer_idx,
        tensors=merged,
        resident=layer.resident,
    )