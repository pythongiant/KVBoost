"""
KVBoost — Chunk-level KV cache reuse for faster HuggingFace inference.

Usage:
    from kvboost import KVBoost

    engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")
    engine.warm("You are a helpful assistant. ...")
    result = engine.generate("You are a helpful assistant. ...\n\nUser: Hello")
    print(result.output_text)

Recompute strategies:
    from kvboost import KVBoost, RecomputeStrategy

    # Original: fix last R tokens at each chunk seam
    engine = KVBoost.from_pretrained("...", recompute_strategy="selective")

    # CacheBlend: fix only the ~15% most deviated tokens (smarter, faster)
    engine = KVBoost.from_pretrained("...", recompute_strategy="cacheblend")

    # None: skip recompute entirely (fastest, slight quality risk)
    engine = KVBoost.from_pretrained("...", recompute_strategy="none")
"""

from .engine import InferenceEngine as KVBoost, GenerationMode, GenerationResult, RecomputeStrategy
from .models import CachedChunk, AssembledPrompt, WarmResult, content_hash_from_tokens, chained_hash
from .cache_manager import KVCacheManager
from .chunk_registry import ChunkRegistry, ChunkStrategy
from .prompt_assembler import PromptAssembler, AssemblyMode
from .selective_recompute import SelectiveRecompute
from .cacheblend import CacheBlendRecompute
from .kv_quantize import quantize_kv, dequantize_kv, QuantizedKV
from .disk_tier import DiskTier
from .batch import find_common_chunk_prefix, broadcast_kv, group_by_prefix
from .compat import SUPPORTED_ARCHITECTURES, UNSUPPORTED_ARCHITECTURES, check_model_compatibility, default_device
from .flash_attn_ext import (
    install_flash_attention, uninstall_flash_attention,
    install_paged_attention, uninstall_paged_attention,
    flash_attention_available, get_tier as get_flash_attn_tier,
)
from .cpu_paged import CPUPagedEngine, BlockAllocator, paged_attention_fwd, ChunkBlockMapper

__version__ = "0.4.0"

__all__ = [
    # Core engine
    "KVBoost",
    "GenerationMode",
    "GenerationResult",
    "RecomputeStrategy",
    # CPU paged engine
    "CPUPagedEngine",
    "BlockAllocator",
    "paged_attention_fwd",
    "ChunkBlockMapper",
    # Flash attention
    "install_flash_attention",
    "uninstall_flash_attention",
    "install_paged_attention",
    "uninstall_paged_attention",
    "flash_attention_available",
    "get_flash_attn_tier",
    # Data structures
    "CachedChunk",
    "AssembledPrompt",
    "WarmResult",
    # Sub-systems
    "KVCacheManager",
    "ChunkRegistry",
    "ChunkStrategy",
    "PromptAssembler",
    "AssemblyMode",
    "SelectiveRecompute",
    "CacheBlendRecompute",
    # Compat
    "SUPPORTED_ARCHITECTURES",
    "UNSUPPORTED_ARCHITECTURES",
    "check_model_compatibility",
    # Quantization
    "quantize_kv",
    "dequantize_kv",
    "QuantizedKV",
]
