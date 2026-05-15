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

    n_staging_slots: int = 2
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

        if self.n_staging_slots < 1:
            raise ValueError("n_staging_slots must be >= 1")

        if self.prefetch_layers_ahead < 0:
            raise ValueError("prefetch_layers_ahead must be >= 0")

        if (
            self.enable_double_buffering
            and self.n_staging_slots < 2
        ):
            raise ValueError(
                "double buffering requires at least 2 staging slots"
            )

        if self.max_pinned_host_memory_gb is not None:
            if self.max_pinned_host_memory_gb <= 0:
                raise ValueError(
                    "max_pinned_host_memory_gb must be positive"
                )

    def summary(self) -> str:
        """
        Human-readable runtime summary.
        """

        return (
            "StreamingConfig("
            f"mode={self.residency_mode}, "
            f"keep_first_k={self.keep_first_k}, "
            f"keep_last_k={self.keep_last_k}, "
            f"slots={self.n_staging_slots}, "
            f"kernel={self.quant_kernel}"
            ")"
        )