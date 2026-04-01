#!/usr/bin/env python3
"""
KVBoost vs MLX LLM Benchmark
=============================
Head-to-head comparison of:
  1. HuggingFace baseline (no caching)
  2. KVBoost (chunk-level KV cache reuse)
  3. MLX LLM (Apple's Metal-optimized inference)

Measures TTFT, total latency, and throughput across multiple workloads.

Usage:
    python benchmarks_and_experiments/benchmark_vs_mlx.py
    python benchmarks_and_experiments/benchmark_vs_mlx.py --model Qwen/Qwen2.5-3B
    python benchmarks_and_experiments/benchmark_vs_mlx.py --workload multiturn
    python benchmarks_and_experiments/benchmark_vs_mlx.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger("bench_mlx")


# ── Result container ────────────────────────────────────────────────
@dataclass
class BenchResult:
    engine: str
    workload: str
    query_index: int
    query: str
    ttft_ms: float
    total_ms: float
    tokens_generated: int
    tok_per_sec: float
    prompt_tokens: int
    cache_reuse_pct: float
    output_preview: str


# ── MLX runner ──────────────────────────────────────────────────────
class MLXRunner:
    def __init__(self, model_name: str):
        from mlx_lm import load
        log.info("Loading MLX model: %s", model_name)
        self.model, self.tokenizer = load(model_name)
        log.info("MLX model loaded")

    def generate(self, prompt: str, max_tokens: int = 64) -> BenchResult:
        from mlx_lm import stream_generate

        t0 = time.perf_counter()
        first_token_time = None
        last_resp = None
        token_count = 0

        for resp in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
            last_resp = resp

        t1 = time.perf_counter()

        ttft = (first_token_time - t0) * 1000 if first_token_time else 0
        total = (t1 - t0) * 1000
        text = last_resp.text if last_resp else ""
        tps = token_count / max(t1 - t0, 1e-6)
        prompt_tokens = last_resp.prompt_tokens if last_resp else 0

        return BenchResult(
            engine="mlx",
            workload="",
            query_index=0,
            query=prompt[:80],
            ttft_ms=ttft,
            total_ms=total,
            tokens_generated=token_count,
            tok_per_sec=tps,
            prompt_tokens=prompt_tokens,
            cache_reuse_pct=0.0,
            output_preview=text[:100],
        )


# ── KVBoost runner ──────────────────────────────────────────────────
class KVBoostRunner:
    def __init__(self, model_name: str, chunk_size: int = 128):
        from kvboost import KVBoost
        log.info("Loading KVBoost model: %s", model_name)
        self.engine = KVBoost.from_pretrained(
            model_name, chunk_size=chunk_size, recompute_overlap=8,
        )
        log.info("KVBoost loaded on: %s", self.engine.device)

    def warm(self, text: str) -> int:
        return self.engine.warm(text)

    def generate(
        self, prompt: str, max_tokens: int = 64, mode: str = "chunk_kv_reuse"
    ) -> BenchResult:
        from kvboost import GenerationMode

        mode_enum = {
            "baseline": GenerationMode.BASELINE,
            "chunk_kv_reuse": GenerationMode.CHUNK_KV_REUSE,
        }[mode]

        r = self.engine.generate(
            prompt, max_new_tokens=max_tokens, mode=mode_enum, do_sample=False
        )

        engine_name = "kvboost" if mode == "chunk_kv_reuse" else "hf_baseline"
        return BenchResult(
            engine=engine_name,
            workload="",
            query_index=0,
            query=prompt[:80],
            ttft_ms=r.ttft_ms,
            total_ms=r.total_ms,
            tokens_generated=r.generated_tokens,
            tok_per_sec=r.tokens_per_sec,
            prompt_tokens=r.prompt_tokens,
            cache_reuse_pct=r.kv_reuse_ratio * 100,
            output_preview=r.output_text[:100],
        )


# ── Workloads ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert coding assistant specializing in software engineering, "
    "algorithms, data structures, and distributed system design. You provide "
    "concise, accurate, and well-structured answers. When relevant, include a "
    "brief code example in Python unless the user specifies another language. "
    "Keep answers under 3 sentences unless the topic requires more depth. "
    "Always consider edge cases and performance implications in your answers. "
    "When discussing algorithms, mention time and space complexity using Big-O "
    "notation. When discussing system design, consider scalability, fault "
    "tolerance, and consistency trade-offs. If a question is ambiguous, state "
    "your assumptions before answering. Prefer practical, production-ready "
    "advice over theoretical descriptions."
)

CODE_CONTEXT = '''
"""Thread-safe LRU cache with TTL expiration."""

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional


class CacheEntry:
    __slots__ = ("key", "value", "created_at", "last_accessed", "ttl", "hit_count")

    def __init__(self, key: Hashable, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.monotonic()
        self.last_accessed = self.created_at
        self.ttl = ttl
        self.hit_count = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.monotonic() - self.created_at) > self.ttl

    def touch(self) -> None:
        self.last_accessed = time.monotonic()
        self.hit_count += 1


class LRUCache:
    def __init__(self, capacity: int = 128, default_ttl: Optional[float] = None):
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._store: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return default
            if entry.is_expired:
                del self._store[key]
                self._misses += 1
                return default
            self._store.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def put(self, key: Hashable, value: Any, ttl: Optional[float] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self._capacity:
                self._evict_one()
            self._store[key] = CacheEntry(key, value, effective_ttl)
            self._store.move_to_end(key)

    def delete(self, key: Hashable) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def _evict_one(self) -> None:
        if self._store:
            self._store.popitem(last=False)
            self._evictions += 1

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "capacity": self._capacity,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "evictions": self._evictions,
        }
'''

MULTITURN_HISTORY = [
    ("What is the derivative of x^2?",
     "The derivative of x^2 is 2x. Using the limit definition: "
     "d/dx[x^2] = lim(h->0) [(x+h)^2 - x^2] / h = 2x. "
     "Intuitively, this tells us the slope of y=x^2 at any point x."),

    ("What about x^3?",
     "The derivative of x^3 is 3x^2. Expanding (x+h)^3 and taking "
     "the limit gives 3x^2. The exponent comes down as a coefficient "
     "and decreases by one."),

    ("Now explain the power rule in general.",
     "The Power Rule: d/dx[x^n] = n*x^(n-1) for any real n. "
     "Works for positive, negative, and fractional exponents. "
     "Example: d/dx[sqrt(x)] = (1/2)*x^(-1/2)."),

    ("Give me an example with x^5.",
     "f(x) = x^5, so f'(x) = 5x^4. At x=2: f(2)=32, f'(2)=80. "
     "The second derivative f''(x) = 20x^3."),

    ("How does the chain rule work?",
     "Chain Rule: d/dx[f(g(x))] = f'(g(x)) * g'(x). "
     "Example: d/dx[(2x+1)^7] = 7*(2x+1)^6 * 2 = 14*(2x+1)^6."),

    ("Walk me through differentiating (3x^2 + 1)^4.",
     "Outer: u^4, inner: u = 3x^2 + 1. "
     "d/du[u^4] = 4u^3, d/dx[3x^2+1] = 6x. "
     "Result: 4*(3x^2+1)^3 * 6x = 24x*(3x^2+1)^3."),

    ("What about the product rule?",
     "Product Rule: d/dx[f*g] = f'*g + f*g'. "
     "Example: d/dx[x^2 * sin(x)] = 2x*sin(x) + x^2*cos(x). "
     "Use when two functions are multiplied, not composed."),
]


def build_multiturn_prompts(system: str) -> List[str]:
    """Build progressively longer multi-turn prompts."""
    prompts = []
    history = system
    for user_msg, assistant_reply in MULTITURN_HISTORY:
        history += f"\n\nUser: {user_msg}\nAssistant: {assistant_reply}"
        prompt = history + "\n\nUser: Continue.\nAssistant:"
        prompts.append(prompt)
    return prompts


WORKLOADS = {
    "chatbot": {
        "desc": "System prompt reuse (~250 tokens)",
        "warmup_text": SYSTEM_PROMPT,
        "prompts": [
            SYSTEM_PROMPT + "\n\nUser: How do I reverse a linked list?\nAssistant:",
            SYSTEM_PROMPT + "\n\nUser: What is the time complexity of quicksort?\nAssistant:",
            SYSTEM_PROMPT + "\n\nUser: Explain the difference between a mutex and a semaphore.\nAssistant:",
        ],
    },
    "code": {
        "desc": "Code context reuse (~800 tokens)",
        "warmup_text": CODE_CONTEXT,
        "prompts": [
            f"Given this code:\n{CODE_CONTEXT}\n\nQuestion: What is the time complexity of the get method?\nAnswer:",
            f"Given this code:\n{CODE_CONTEXT}\n\nQuestion: How would you add a bulk insert method?\nAnswer:",
            f"Given this code:\n{CODE_CONTEXT}\n\nQuestion: Is this implementation thread-safe? Why?\nAnswer:",
        ],
    },
    "multiturn": {
        "desc": "Multi-turn conversation (232 -> 1100+ tokens)",
        "warmup_text": SYSTEM_PROMPT,
        "prompts": build_multiturn_prompts(SYSTEM_PROMPT),
    },
}


# ── Benchmark runner ────────────────────────────────────────────────
def run_benchmark(
    model_name: str,
    workload_name: str,
    max_tokens: int = 64,
    chunk_size: int = 128,
    warmup_runs: int = 1,
) -> List[BenchResult]:
    workload = WORKLOADS[workload_name]
    prompts = workload["prompts"]
    warmup_text = workload["warmup_text"]
    results: List[BenchResult] = []

    print(f"\n{'='*70}")
    print(f"  Workload: {workload_name} — {workload['desc']}")
    print(f"  Model: {model_name} | max_tokens: {max_tokens}")
    print(f"{'='*70}\n")

    # ── Load engines ────────────────────────────────────────────
    kvboost = KVBoostRunner(model_name, chunk_size=chunk_size)
    mlx = MLXRunner(model_name)

    # ── Warmup (KVBoost only — MLX has no explicit cache warming) ──
    n_warmed = kvboost.warm(warmup_text)
    log.info("KVBoost warmed %d chunks", n_warmed)

    # Warmup run to compile MPS/MLX kernels
    for _ in range(warmup_runs):
        _ = kvboost.generate(prompts[0], max_tokens=5, mode="baseline")
        _ = mlx.generate(prompts[0], max_tokens=5)

    # ── Run each prompt ─────────────────────────────────────────
    for i, prompt in enumerate(prompts):
        print(f"  --- Query {i+1} (prompt ~{len(prompt.split())} words) ---")

        # HF Baseline
        r_base = kvboost.generate(prompt, max_tokens, mode="baseline")
        r_base.workload = workload_name
        r_base.query_index = i + 1
        results.append(r_base)

        # KVBoost
        r_kv = kvboost.generate(prompt, max_tokens, mode="chunk_kv_reuse")
        r_kv.workload = workload_name
        r_kv.query_index = i + 1
        results.append(r_kv)

        # MLX
        r_mlx = mlx.generate(prompt, max_tokens)
        r_mlx.workload = workload_name
        r_mlx.query_index = i + 1
        results.append(r_mlx)

        # Print comparison
        print(
            f"    {'HF Baseline':>12s}  TTFT {r_base.ttft_ms:7.1f}ms | "
            f"total {r_base.total_ms:7.1f}ms | {r_base.tok_per_sec:5.1f} tok/s"
        )
        print(
            f"    {'KVBoost':>12s}  TTFT {r_kv.ttft_ms:7.1f}ms | "
            f"total {r_kv.total_ms:7.1f}ms | {r_kv.tok_per_sec:5.1f} tok/s | "
            f"reuse {r_kv.cache_reuse_pct:.0f}%"
        )
        print(
            f"    {'MLX':>12s}  TTFT {r_mlx.ttft_ms:7.1f}ms | "
            f"total {r_mlx.total_ms:7.1f}ms | {r_mlx.tok_per_sec:5.1f} tok/s"
        )

        # Speedups
        kv_vs_base = r_base.ttft_ms / r_kv.ttft_ms if r_kv.ttft_ms > 0 else 0
        mlx_vs_base = r_base.ttft_ms / r_mlx.ttft_ms if r_mlx.ttft_ms > 0 else 0
        kv_vs_mlx = r_mlx.ttft_ms / r_kv.ttft_ms if r_kv.ttft_ms > 0 else 0
        print(
            f"    {'':>12s}  KVBoost vs baseline: {kv_vs_base:.1f}x | "
            f"MLX vs baseline: {mlx_vs_base:.1f}x | "
            f"KVBoost vs MLX: {kv_vs_mlx:.1f}x"
        )
        print()

    return results


# ── Summary table ───────────────────────────────────────────────────
def print_summary(results: List[BenchResult]):
    print(f"\n{'='*70}")
    print("  SUMMARY — TTFT Comparison")
    print(f"{'='*70}")
    print(f"  {'Query':>8s} | {'HF Base':>10s} | {'KVBoost':>10s} | {'MLX':>10s} | {'KV vs HF':>8s} | {'KV vs MLX':>9s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}")

    queries = sorted(set((r.workload, r.query_index) for r in results))
    for wl, qi in queries:
        base = next((r for r in results if r.workload == wl and r.query_index == qi and r.engine == "hf_baseline"), None)
        kv = next((r for r in results if r.workload == wl and r.query_index == qi and r.engine == "kvboost"), None)
        mlx = next((r for r in results if r.workload == wl and r.query_index == qi and r.engine == "mlx"), None)
        if not (base and kv and mlx):
            continue

        kv_vs_hf = base.ttft_ms / kv.ttft_ms if kv.ttft_ms > 0 else 0
        kv_vs_mlx = mlx.ttft_ms / kv.ttft_ms if kv.ttft_ms > 0 else 0

        label = f"{wl[:6]} Q{qi}"
        print(
            f"  {label:>8s} | {base.ttft_ms:8.1f}ms | {kv.ttft_ms:8.1f}ms | {mlx.ttft_ms:8.1f}ms | "
            f"{kv_vs_hf:6.1f}x | {kv_vs_mlx:7.1f}x"
        )

    print(f"{'='*70}\n")


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="KVBoost vs MLX LLM benchmark",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--workload", default=None, choices=list(WORKLOADS.keys()))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    workloads = [args.workload] if args.workload else list(WORKLOADS.keys())
    all_results: List[BenchResult] = []

    print(f"\n  Model: {args.model}")
    print(f"  Workloads: {', '.join(workloads)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Chunk size: {args.chunk_size}")

    for wl in workloads:
        results = run_benchmark(
            model_name=args.model,
            workload_name=wl,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
            warmup_runs=args.warmup_runs,
        )
        all_results.extend(results)

    print_summary(all_results)

    if args.output:
        out = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "chunk_size": args.chunk_size,
            "results": [asdict(r) for r in all_results],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
