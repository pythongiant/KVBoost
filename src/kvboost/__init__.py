"""
KVBoost — Chunk-level KV cache reuse for faster HuggingFace inference.

Usage:
    from kvboost import KVBoost

    engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")
    engine.warm("You are a helpful assistant. ...")
    result = engine.generate("You are a helpful assistant. ...\n\nUser: Hello")
    print(result.output_text)
"""

from .engine import InferenceEngine as KVBoost, GenerationMode, GenerationResult
from .models import CachedChunk, AssembledPrompt
from .cache_manager import KVCacheManager
from .chunk_registry import ChunkRegistry, ChunkStrategy
from .prompt_assembler import PromptAssembler, AssemblyMode
from .selective_recompute import SelectiveRecompute

__version__ = "0.1.0"

__all__ = [
    "KVBoost",
    "GenerationMode",
    "GenerationResult",
    "CachedChunk",
    "AssembledPrompt",
    "KVCacheManager",
    "ChunkRegistry",
    "ChunkStrategy",
    "PromptAssembler",
    "AssemblyMode",
    "SelectiveRecompute",
]
