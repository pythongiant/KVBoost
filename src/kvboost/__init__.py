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
from .models import CachedChunk, AssembledPrompt, content_hash_from_tokens, chained_hash
from .cache_manager import KVCacheManager
from .chunk_registry import ChunkRegistry, ChunkStrategy
from .prompt_assembler import PromptAssembler, AssemblyMode
from .selective_recompute import SelectiveRecompute
from .cacheblend import CacheBlendRecompute
from .compat import SUPPORTED_ARCHITECTURES, UNSUPPORTED_ARCHITECTURES, check_model_compatibility

__version__ = "0.1.0"

__all__ = [
    "KVBoost",
    "GenerationMode",
    "GenerationResult",
    "RecomputeStrategy",
    "CachedChunk",
    "AssembledPrompt",
    "KVCacheManager",
    "ChunkRegistry",
    "ChunkStrategy",
    "PromptAssembler",
    "AssemblyMode",
    "SelectiveRecompute",
    "CacheBlendRecompute",
    "SUPPORTED_ARCHITECTURES",
    "UNSUPPORTED_ARCHITECTURES",
    "check_model_compatibility",
]
