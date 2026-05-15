# src/kvboost/streaming/__init__.py

"""
KVBoost streaming inference backend.

This package provides an AirLLM-style layer streaming runtime for running
large AWQ-quantized causal LLMs with limited VRAM by:

- Keeping quantized weights in pinned host RAM
- Streaming decoder layers into GPU staging buffers
- Reusing KVBoost's existing DynamicCache + chunk reuse pipeline
- Using fused AWQ kernels (Marlin / ExLlamaV2)
- Supporting partial layer residency

Primary entrypoints:

- StreamingConfig
- StreamingCausalLM
"""

from .config import StreamingConfig
from .model_shell import StreamingCausalLM
from .loader import load_streaming_model

__all__ = [
    "StreamingConfig",
    "StreamingCausalLM",
    "load_streaming_model",
]

__version__ = "0.1.0"