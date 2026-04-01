"""
Shared utilities for benchmarks and experiments.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from kvboost import KVBoost as InferenceEngine, GenerationMode, GenerationResult

RESULTS_DIR = Path(__file__).resolve().parent / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_results(data: Any, filename: str) -> Path:
    out_dir = ensure_results_dir()
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_engine(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    chunk_size: int = 128,
    recompute_overlap: int = 16,
    max_chunks: int = 256,
    **kwargs,
) -> InferenceEngine:
    """Load an InferenceEngine with standard defaults."""
    log = get_logger("utils")
    log.info("Loading engine: model=%s chunk_size=%d", model_name, chunk_size)
    engine = InferenceEngine.from_pretrained(
        model_name=model_name,
        chunk_size=chunk_size,
        recompute_overlap=recompute_overlap,
        max_chunks=max_chunks,
        **kwargs,
    )
    log.info("Engine ready. Device: %s", engine.device)
    return engine


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, median, stdev, min, max, p95, p99 for a list of values."""
    if not values:
        return {}
    sorted_v = sorted(values)
    n = len(sorted_v)
    result = {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }
    if n > 1:
        result["stdev"] = round(statistics.stdev(values), 3)
    if n >= 20:
        result["p95"] = round(sorted_v[int(n * 0.95)], 3)
        result["p99"] = round(sorted_v[int(n * 0.99)], 3)
    return result


def timed(fn):
    """Decorator that returns (result, elapsed_ms)."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed
    return wrapper


# Standard prompts reused across experiments
SYSTEM_PROMPT = """\
You are an expert technical assistant specializing in distributed systems, \
machine learning infrastructure, and high-performance computing. \
You provide precise, actionable answers backed by production experience. \
When asked a question, first briefly summarize the key considerations, \
then provide a concrete recommendation or implementation strategy. \
Always mention trade-offs and when to use alternative approaches. \
If the question involves code, include minimal but complete examples. \
Prioritize correctness over brevity, but avoid unnecessary padding.\
"""

RAG_DOCUMENT = """\
## KV Cache Architecture in LLM Inference

A KV (Key-Value) cache stores the intermediate key and value tensors produced by the \
attention mechanism during the forward pass of a transformer model. During autoregressive \
generation, each new token needs to attend to all previous tokens. Without caching, this \
requires recomputing keys and values for every past token at each generation step, giving \
O(n^2) complexity. With KV caching, previously computed K/V tensors are stored and reused, \
reducing generation to O(n) per step.

### Memory Layout
KV tensors have shape [batch, heads, seq_len, head_dim]. For a model with 22 layers, \
32 attention heads, head dimension 64, and a sequence of 512 tokens, the KV cache for \
one sequence requires 22 * 2 * 32 * 512 * 64 * 2 bytes = 92MB in float16.

### Prefix Caching
Prefix caching stores the KV tensors for a fixed prefix (e.g., system prompt) and reuses \
them across requests. All requests sharing the same prefix avoid recomputing that prefix's \
KV tensors. This is effective when prompts share a long, stable prefix.

### Chunk-Level Caching
Chunk-level caching generalizes prefix caching by breaking prompts into fixed-size chunks \
and caching each chunk independently. This allows reuse of arbitrary subsets of chunks \
across requests that don't share the same prefix but do share some chunks. For example, \
a RAG pipeline that injects multiple documents can cache each document chunk independently \
and reuse them across queries.

### Positional Encodings
The main challenge with chunk caching is handling positional encodings correctly. RoPE \
(Rotary Position Embedding) encodes absolute positions directly into K/V tensors. When \
reusing a cached chunk, the position_ids for new tokens must be set to continue from the \
end of the cached region so that relative positions remain consistent.\
"""

FEW_SHOT_EXAMPLES = """\
Here are examples of the input/output format:

Input: Classify the sentiment: "The product exceeded all my expectations!"
Output: POSITIVE

Input: Classify the sentiment: "Delivery was late and the item was broken."
Output: NEGATIVE

Input: Classify the sentiment: "It's okay, nothing special about it."
Output: NEUTRAL

Input: Classify the sentiment: "Absolutely terrible experience, never again."
Output: NEGATIVE

Input: Classify the sentiment: "Pretty good for the price point."
Output: POSITIVE\
"""

QUERIES = [
    "What is the best strategy for reducing P99 latency in a high-throughput gRPC service?",
    "How should I design a rate limiter for a multi-tenant API gateway?",
    "Explain the trade-offs between synchronous and asynchronous model serving.",
    "What are the key considerations when choosing between Redis and Memcached for session storage?",
    "How do I implement circuit breakers in a microservices architecture?",
]

RAG_QUERIES = [
    "Based on the above, what is the memory formula for KV cache?",
    "When is chunk-level caching more beneficial than prefix caching?",
    "What complexity does KV caching reduce generation to?",
    "How does RoPE affect chunk caching according to the document?",
    "Summarize the difference between prefix caching and chunk-level caching.",
]
