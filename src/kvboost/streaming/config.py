# src/kvboost/streaming/config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ResidencyMode = Literal[
    "full_stream",
    "partial_resident",
    "ffn_only_stream",
    "full_resident",
]


@dataclass(slots=True)
class StreamingConfig:
    """
    Configuration for KVBoost streaming inference.

    Notes
    -----
    Streaming primarily reduces VRAM usage. It is NOT expected to outperform
    fully-resident inference during autoregressive decode.

    Throughput is heavily dependent on:
    - PCIe bandwidth
    - overlap efficiency
    - residency strategy
    - quant kernel efficiency

    Recommended defaults:
        keep_first_k = 4
        keep_last_k = 4
        residency_mode = "partial_resident"

    Common strategies
    -----------------
    full_stream:
        Every decoder layer streams from host RAM.

    partial_resident:
        Early + late layers remain resident in VRAM.

    ffn_only_stream:
        Attention weights remain resident while FFNs stream.
        Usually the best VRAM/perf tradeoff.

    full_resident:
        Disables streaming entirely.
    """

    #
    # Residency strategy
    #

    residency_mode: ResidencyMode = "partial_resident"

    #
    # Layer residency
    #

    keep_first_k: int = 4
    keep_last_k: int = 4

    #
    # Streaming pipeline
    #

    # 0 = auto-size at load time: probe free VRAM, divide by per-slot bytes,
    # leave ``auto_slots_margin_gb`` free for KV cache + activations, then
    # clamp to ``[2, auto_slots_max]``. Any positive int is taken as an
    # explicit user override and skips auto-sizing entirely.
    n_staging_slots: int = 0
    auto_slots_margin_gb: float = 1.0
    auto_slots_max: int = 4

    enable_double_buffering: bool = True
    enable_async_prefetch: bool = True

    #
    # CUDA streams
    #

    transfer_stream_priority: int = -1

    #
    # Memory
    #

    use_pinned_memory: bool = True
    enable_pinned_lru: bool = False

    #
    # Quant kernels
    #

    quant_kernel: Literal["marlin", "exllama_v2", "auto"] = "auto"

    #
    # Fusion
    #

    # Merge SwiGLU's gate_proj and up_proj into a single matmul with
    # 2× out_features, then split + silu+mul afterward. Saves one kernel
    # launch and one HBM read of the activation per layer. Profile-driven
    # win when ``qlinear.forward::{gate_proj,up_proj}`` exceeds 25% of
    # total per-token time. Default on; set False to A/B against the
    # unfused path with the same trace.
    fuse_gate_up: bool = True

    #
    # Runtime behavior
    #

    prefetch_layers_ahead: int = 2
    streaming_disable_below_layers: int = 12

    #
    # Validation / debugging
    #

    verify_layer_shapes: bool = False
    enable_nvtx: bool = False
    verbose_scheduler: bool = False

    #
    # Experimental
    #

    speculative_decode: bool = False
    paged_kv_cache: bool = False

    #
    # Optional limits
    #

    max_pinned_host_memory_gb: Optional[float] = None

    def should_stream_model(self, num_hidden_layers: int) -> bool:
        """
        Determine whether streaming should activate for a model.

        Small models often regress with streaming due to orchestration overhead.
        """
        if self.residency_mode == "full_resident":
            return False

        return num_hidden_layers >= self.streaming_disable_below_layers

    @property
    def use_partial_residency(self) -> bool:
        return self.residency_mode in {
            "partial_resident",
            "ffn_only_stream",
        }

    @property
    def use_ffn_only_streaming(self) -> bool:
        return self.residency_mode == "ffn_only_stream"

    @property
    def use_full_streaming(self) -> bool:
        return self.residency_mode == "full_stream"

    def validate(self) -> None:
        """
        Validate config values early.
        """

        if self.keep_first_k < 0:
            raise ValueError("keep_first_k must be >= 0")

        if self.keep_last_k < 0:
            raise ValueError("keep_last_k must be >= 0")

        if self.n_staging_slots < 0:
            raise ValueError(
                "n_staging_slots must be >= 0 (0 = auto-size at load time)"
            )

        if self.prefetch_layers_ahead < 0:
            raise ValueError("prefetch_layers_ahead must be >= 0")

        # 0 (auto) is fine for double buffering — auto-sizing clamps the
        # minimum to 2. Only reject an explicit 1 here.
        if (
            self.enable_double_buffering
            and 0 < self.n_staging_slots < 2
        ):
            raise ValueError(
                "double buffering requires at least 2 staging slots"
            )

        if self.auto_slots_margin_gb < 0:
            raise ValueError("auto_slots_margin_gb must be >= 0")
        if self.auto_slots_max < 2:
            raise ValueError("auto_slots_max must be >= 2")

        if self.max_pinned_host_memory_gb is not None:
            if self.max_pinned_host_memory_gb <= 0:
                raise ValueError(
                    "max_pinned_host_memory_gb must be positive"
                )

    def summary(self) -> str:
        """
        Human-readable runtime summary.
        """

        slots_str = "auto" if self.n_staging_slots == 0 else str(self.n_staging_slots)
        return (
            "StreamingConfig("
            f"mode={self.residency_mode}, "
            f"keep_first_k={self.keep_first_k}, "
            f"keep_last_k={self.keep_last_k}, "
            f"slots={slots_str}, "
            f"kernel={self.quant_kernel}, "
            f"fuse_gate_up={self.fuse_gate_up}"
            ")"
        )