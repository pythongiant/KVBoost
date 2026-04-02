#!/usr/bin/env python3
"""
KVBoost vs vLLM-MLX Prefix Caching Benchmark
=============================================
Head-to-head on Apple Silicon using vllm-mlx (MLX port of vLLM).

Three axes that matter:
  1. Non-prefix interior reuse — document in the middle, not the start.
     vLLM prefix caching misses here; KVBoost should hit.
  2. Cold-start overhead — performance with empty cache.
  3. Break-even prompt length — sweep to find crossover point.

Both engines run locally on MPS/Metal — no CUDA required.

Note: vllm-mlx uses MLX-format models (e.g. mlx-community/Qwen2.5-3B-4bit)
while KVBoost uses HF models (Qwen/Qwen2.5-3B). The comparison is
apples-to-oranges on quantization — 4-bit MLX vs float16 HF. This is the
realistic deployment comparison: each system uses its optimal format.

Setup
-----
    pip install kvboost vllm-mlx

Usage
-----
    python benchmarks_and_experiments/benchmark_vs_vllm.py
    python benchmarks_and_experiments/benchmark_vs_vllm.py --axis non_prefix
    python benchmarks_and_experiments/benchmark_vs_vllm.py --axis sweep
    python benchmarks_and_experiments/benchmark_vs_vllm.py --skip-vllm
    python benchmarks_and_experiments/benchmark_vs_vllm.py --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger("bench_vllm")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Result container ────────────────────────────────────────────────

@dataclass
class BenchResult:
    engine: str           # "kvboost", "vllm_mlx", "hf_baseline"
    axis: str             # "non_prefix", "cold_start", "sweep"
    pattern: str          # "exact_prefix", "interior", "no_reuse"
    prompt_tokens: int
    cache_state: str      # "cold", "warm", "q1", etc.
    ttft_ms: float
    total_ms: float
    tokens_generated: int
    tok_per_sec: float
    reuse_pct: float
    output_preview: str


# ── Prompt generators ───────────────────────────────────────────────

SHARED_DOCUMENT = (
    "Key-Value caching stores intermediate attention tensors during autoregressive "
    "generation. Without caching, each new token requires recomputing keys and values "
    "for all previous tokens, giving O(n^2) complexity per step. With KV caching, "
    "previously computed K/V tensors are stored and reused, reducing each step to O(n). "
    "The cache grows linearly with sequence length: for a model with L layers, H heads, "
    "and head dimension D, the KV cache for a sequence of length S requires "
    "2 * L * H * S * D * sizeof(dtype) bytes. "
    "Prefix caching extends this across requests. When multiple prompts share the same "
    "prefix, the KV tensors for that prefix can be computed once and reused. "
    "Chunk-level caching generalizes prefix caching by allowing reuse of non-contiguous "
    "chunks. A document that appears in the middle of a prompt can still be cached and "
    "reused across requests with different preambles. This is the key differentiator "
    "of chunk-based systems over pure prefix-caching approaches. "
    "The main challenge with chunk caching is positional encoding correctness. RoPE "
    "embeds absolute position into K/V tensors, so chunks cached at one position cannot "
    "be naively reused at another without fixing the position encoding mismatch. "
    "Memory management is critical at scale. PagedAttention allocates KV cache in "
    "fixed-size pages rather than contiguous blocks, reducing fragmentation and enabling "
    "efficient memory sharing between sequences."
)

SYSTEM_PROMPT = (
    "You are an expert technical assistant specializing in distributed systems, "
    "machine learning infrastructure, and high-performance computing. "
    "You provide precise, actionable answers backed by production experience. "
    "When asked a question, first briefly summarize the key considerations, "
    "then provide a concrete recommendation. Always mention trade-offs."
)

QUERIES = [
    "What is the difference between prefix caching and chunk-level caching?",
    "How does PagedAttention reduce memory fragmentation?",
    "What is the memory formula for KV cache at sequence length S?",
    "Why does RoPE make chunk caching challenging?",
]


def _filler(n_words: int) -> str:
    base = (
        "The transformer architecture uses self-attention to process sequences "
        "in parallel. Each layer computes query key and value projections from "
        "the input representations. The attention scores are computed as the "
        "scaled dot product of queries and keys. Multi-head attention splits "
        "the representation into multiple subspaces. "
    )
    repeated = (base * (n_words // len(base.split()) + 2))
    return " ".join(repeated.split()[:n_words])


def make_exact_prefix_prompt(query: str) -> str:
    return f"{SHARED_DOCUMENT}\n\nQuestion: {query}\nAnswer:"


def make_interior_prompt(query: str, preamble_words: int = 150) -> str:
    preamble = _filler(preamble_words)
    return f"{preamble}\n\n{SHARED_DOCUMENT}\n\nQuestion: {query}\nAnswer:"


def make_no_reuse_prompt(query: str) -> str:
    unique = f"Request {time.time_ns()}. " + _filler(300)
    return f"{unique}\n\nQuestion: {query}\nAnswer:"


def make_length_prompt(target_words: int, query: str) -> str:
    ctx = _filler(max(0, target_words - 20))
    return f"{ctx}\n\nQuestion: {query}\nAnswer:"


# ── vLLM-MLX runner ────────────────────────────────────────────────

def _patch_vllm_mlx_loader():
    """
    Fix vllm-mlx 0.2.x bug: load_model_with_fallback() doesn't return
    the (model, tokenizer) tuple on the success path — it falls through
    the try block without a return statement.
    """
    try:
        import vllm_mlx.utils.tokenizer as tok_mod
        from mlx_lm import load as mlx_load

        _original = tok_mod.load_model_with_fallback

        def _fixed(model_name, tokenizer_config=None):
            tokenizer_config = tokenizer_config or {}
            try:
                return mlx_load(model_name, tokenizer_config=tokenizer_config)
            except Exception:
                return _original(model_name, tokenizer_config)

        tok_mod.load_model_with_fallback = _fixed
        log.debug("Patched vllm-mlx load_model_with_fallback")
    except ImportError:
        pass


class VLLMMLXRunner:
    def __init__(self, model_name: str = "mlx-community/Qwen2.5-3B-4bit"):
        _patch_vllm_mlx_loader()
        from vllm_mlx.engine.simple import SimpleEngine
        log.info("Loading vLLM-MLX: %s", model_name)
        self._engine = SimpleEngine(model_name)
        self._started = False
        self.model_name = model_name

    async def _ensure_started(self):
        if not self._started:
            await self._engine.start()
            self._started = True
            log.info("vLLM-MLX ready")

    async def generate(self, prompt: str, max_tokens: int = 64) -> BenchResult:
        await self._ensure_started()

        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0
        text = ""
        prompt_tokens = 0

        async for chunk in self._engine.stream_generate(
            prompt, max_tokens=max_tokens, temperature=0.0,
        ):
            if first_token_time is None and chunk.new_text:
                first_token_time = time.perf_counter()
            if chunk.new_text:
                token_count += 1
            text = chunk.text
            prompt_tokens = chunk.prompt_tokens
            if chunk.finished:
                break

        t1 = time.perf_counter()
        ttft = (first_token_time - t0) * 1000 if first_token_time else (t1 - t0) * 1000

        return BenchResult(
            engine="vllm_mlx",
            axis="", pattern="", cache_state="",
            prompt_tokens=prompt_tokens,
            ttft_ms=ttft,
            total_ms=(t1 - t0) * 1000,
            tokens_generated=token_count,
            tok_per_sec=token_count / max(t1 - t0, 1e-6),
            reuse_pct=0.0,
            output_preview=text[:100],
        )

    async def generate_chat(self, system: str, user: str, max_tokens: int = 64) -> BenchResult:
        """Chat completion — uses vLLM-MLX's system prompt KV caching."""
        await self._ensure_started()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0
        text = ""
        prompt_tokens = 0

        async for chunk in self._engine.stream_chat(
            messages, max_tokens=max_tokens, temperature=0.0,
        ):
            if first_token_time is None and chunk.new_text:
                first_token_time = time.perf_counter()
            if chunk.new_text:
                token_count += 1
            text = chunk.text
            prompt_tokens = chunk.prompt_tokens
            if chunk.finished:
                break

        t1 = time.perf_counter()
        ttft = (first_token_time - t0) * 1000 if first_token_time else (t1 - t0) * 1000

        return BenchResult(
            engine="vllm_mlx",
            axis="", pattern="", cache_state="",
            prompt_tokens=prompt_tokens,
            ttft_ms=ttft,
            total_ms=(t1 - t0) * 1000,
            tokens_generated=token_count,
            tok_per_sec=token_count / max(t1 - t0, 1e-6),
            reuse_pct=0.0,
            output_preview=text[:100],
        )

    def get_stats(self) -> dict:
        return self._engine.get_stats()

    async def stop(self):
        if self._started:
            await self._engine.stop()


# ── KVBoost runner ──────────────────────────────────────────────────

class KVBoostRunner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B", chunk_size: int = 128):
        from kvboost import KVBoost
        log.info("Loading KVBoost: %s", model_name)
        self.engine = KVBoost.from_pretrained(
            model_name, chunk_size=chunk_size, recompute_overlap=8,
        )
        log.info("KVBoost ready on %s", self.engine.device)

    def warm(self, text: str) -> int:
        return self.engine.warm(text)

    def clear_cache(self):
        self.engine.cache_manager._hot.clear()
        self.engine.cache_manager._content_index.clear()
        self.engine.cache_manager.hits = 0
        self.engine.cache_manager.misses = 0
        self.engine.cache_manager.approximate_hits = 0

    def generate(self, prompt: str, max_tokens: int = 64, mode: str = "chunk_kv_reuse") -> BenchResult:
        from kvboost import GenerationMode
        mode_enum = {
            "baseline": GenerationMode.BASELINE,
            "chunk_kv_reuse": GenerationMode.CHUNK_KV_REUSE,
        }[mode]
        r = self.engine.generate(prompt, max_new_tokens=max_tokens, mode=mode_enum, do_sample=False)
        return BenchResult(
            engine="kvboost" if mode == "chunk_kv_reuse" else "hf_baseline",
            axis="", pattern="", cache_state="",
            prompt_tokens=r.prompt_tokens,
            ttft_ms=r.ttft_ms,
            total_ms=r.total_ms,
            tokens_generated=r.generated_tokens,
            tok_per_sec=r.tokens_per_sec,
            reuse_pct=r.kv_reuse_ratio * 100,
            output_preview=r.output_text[:100],
        )


# ── Benchmark axes ──────────────────────────────────────────────────

async def axis_non_prefix(
    kv: KVBoostRunner,
    vllm: Optional[VLLMMLXRunner],
    max_tokens: int = 64,
) -> List[BenchResult]:
    """
    Axis 1: Non-prefix interior reuse.
    Document in the middle — vLLM prefix cache misses, KVBoost should hit.
    """
    results = []
    print(f"\n{'='*70}")
    print("  AXIS 1: Non-Prefix Interior Reuse")
    print(f"  Document in the middle, different preamble each time")
    print(f"{'='*70}")

    kv.warm(SHARED_DOCUMENT)

    for pattern_name, make_fn in [
        ("exact_prefix", make_exact_prefix_prompt),
        ("interior", lambda q: make_interior_prompt(q, preamble_words=150)),
        ("no_reuse", make_no_reuse_prompt),
    ]:
        print(f"\n  --- Pattern: {pattern_name} ---")

        for i, query in enumerate(QUERIES[:3]):
            prompt = make_fn(query)

            # HF Baseline
            r = kv.generate(prompt, max_tokens, mode="baseline")
            r.axis, r.pattern, r.cache_state = "non_prefix", pattern_name, f"q{i+1}"
            results.append(r)

            # KVBoost
            r = kv.generate(prompt, max_tokens)
            r.axis, r.pattern, r.cache_state = "non_prefix", pattern_name, f"q{i+1}"
            results.append(r)

            # vLLM-MLX
            if vllm:
                r = await vllm.generate(prompt, max_tokens)
                r.axis, r.pattern, r.cache_state = "non_prefix", pattern_name, f"q{i+1}"
                results.append(r)

            # Print
            kv_r = results[-2] if not vllm else results[-3]
            if kv_r.engine != "kvboost":
                kv_r = results[-2]
            line = f"    Q{i+1}: KVBoost TTFT {kv_r.ttft_ms:7.1f}ms (reuse {kv_r.reuse_pct:.0f}%)"
            if vllm:
                vllm_r = results[-1]
                line += f" | vLLM {vllm_r.ttft_ms:7.1f}ms"
                if kv_r.ttft_ms > 0:
                    line += f" | KV vs vLLM: {vllm_r.ttft_ms / kv_r.ttft_ms:.1f}x"
            print(line)

    return results


async def axis_cold_start(
    kv: KVBoostRunner,
    vllm: Optional[VLLMMLXRunner],
    max_tokens: int = 64,
) -> List[BenchResult]:
    """Axis 2: Cold-start overhead — empty cache, no reuse."""
    results = []
    print(f"\n{'='*70}")
    print("  AXIS 2: Cold-Start Overhead")
    print(f"{'='*70}")

    kv.clear_cache()

    for i, query in enumerate(QUERIES[:3]):
        prompt = make_exact_prefix_prompt(query)
        state = "cold" if i == 0 else f"warm_q{i+1}"

        r_base = kv.generate(prompt, max_tokens, mode="baseline")
        r_base.axis, r_base.pattern, r_base.cache_state = "cold_start", "exact_prefix", state
        results.append(r_base)

        r_kv = kv.generate(prompt, max_tokens)
        r_kv.axis, r_kv.pattern, r_kv.cache_state = "cold_start", "exact_prefix", state
        results.append(r_kv)

        r_vllm = None
        if vllm:
            r_vllm = await vllm.generate(prompt, max_tokens)
            r_vllm.axis, r_vllm.pattern, r_vllm.cache_state = "cold_start", "exact_prefix", state
            results.append(r_vllm)

        line = f"    {state:>10s}: Baseline {r_base.ttft_ms:7.1f}ms | KVBoost {r_kv.ttft_ms:7.1f}ms (reuse {r_kv.reuse_pct:.0f}%)"
        if r_vllm:
            line += f" | vLLM {r_vllm.ttft_ms:7.1f}ms"
        print(line)

    return results


async def axis_sweep(
    kv: KVBoostRunner,
    vllm: Optional[VLLMMLXRunner],
    max_tokens: int = 64,
    lengths: Optional[List[int]] = None,
) -> List[BenchResult]:
    """Axis 3: Break-even prompt length sweep."""
    results = []
    if lengths is None:
        lengths = [100, 250, 500, 1000, 2000]

    print(f"\n{'='*70}")
    print("  AXIS 3: Break-Even Prompt Length Sweep")
    print(f"  Two queries per length: cold (populates cache) then warm (reuses)")
    print(f"{'='*70}")

    query = QUERIES[0]

    for n_words in lengths:
        print(f"\n  --- ~{n_words} words ---")
        kv.clear_cache()

        prompt = make_length_prompt(n_words, query)

        # Cold
        r_base = kv.generate(prompt, max_tokens, mode="baseline")
        r_base.axis, r_base.pattern, r_base.cache_state = "sweep", f"{n_words}w", "cold"
        results.append(r_base)

        r_kv_cold = kv.generate(prompt, max_tokens)
        r_kv_cold.axis, r_kv_cold.pattern, r_kv_cold.cache_state = "sweep", f"{n_words}w", "cold"
        results.append(r_kv_cold)

        if vllm:
            r_v = await vllm.generate(prompt, max_tokens)
            r_v.axis, r_v.pattern, r_v.cache_state = "sweep", f"{n_words}w", "cold"
            results.append(r_v)

        # Warm
        r_kv_warm = kv.generate(prompt, max_tokens)
        r_kv_warm.axis, r_kv_warm.pattern, r_kv_warm.cache_state = "sweep", f"{n_words}w", "warm"
        results.append(r_kv_warm)

        if vllm:
            r_v2 = await vllm.generate(prompt, max_tokens)
            r_v2.axis, r_v2.pattern, r_v2.cache_state = "sweep", f"{n_words}w", "warm"
            results.append(r_v2)

        line = (
            f"    cold: Baseline {r_base.ttft_ms:7.1f}ms | KVBoost {r_kv_cold.ttft_ms:7.1f}ms"
        )
        if vllm:
            line += f" | vLLM {r_v.ttft_ms:7.1f}ms"
        print(line)

        line = f"    warm: KVBoost {r_kv_warm.ttft_ms:7.1f}ms ({r_kv_warm.reuse_pct:.0f}%)"
        if vllm:
            line += f" | vLLM {r_v2.ttft_ms:7.1f}ms"
        print(line)

    return results


# ── Summary ─────────────────────────────────────────────────────────

def print_summary(results: List[BenchResult]):
    print(f"\n{'='*70}")
    print("  SUMMARY — TTFT by Engine")
    print(f"{'='*70}")

    engines = sorted(set(r.engine for r in results))
    axes = sorted(set(r.axis for r in results))

    for axis in axes:
        axis_results = [r for r in results if r.axis == axis]
        print(f"\n  [{axis}]")
        patterns = sorted(set((r.pattern, r.cache_state) for r in axis_results))
        for pattern, state in patterns:
            for eng in engines:
                eng_results = [r for r in axis_results if r.engine == eng and r.pattern == pattern and r.cache_state == state]
                if eng_results:
                    avg_ttft = sum(r.ttft_ms for r in eng_results) / len(eng_results)
                    avg_reuse = sum(r.reuse_pct for r in eng_results) / len(eng_results)
                    reuse_str = f" reuse {avg_reuse:.0f}%" if avg_reuse > 0 else ""
                    print(f"    {pattern:>15s} {state:>8s} | {eng:>12s} | TTFT {avg_ttft:8.1f}ms{reuse_str}")

    # vLLM stats if available
    vllm_results = [r for r in results if r.engine == "vllm_mlx"]
    if vllm_results:
        kv_results = [r for r in results if r.engine == "kvboost"]
        if kv_results:
            avg_kv = sum(r.ttft_ms for r in kv_results) / len(kv_results)
            avg_vllm = sum(r.ttft_ms for r in vllm_results) / len(vllm_results)
            print(f"\n  Overall mean TTFT: KVBoost {avg_kv:.1f}ms | vLLM-MLX {avg_vllm:.1f}ms")

    print(f"{'='*70}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KVBoost vs vLLM-MLX benchmark")
    parser.add_argument("--kvboost-model", default="Qwen/Qwen2.5-3B",
                        help="HF model for KVBoost (float16)")
    parser.add_argument("--vllm-model", default="mlx-community/Qwen2.5-3B-4bit",
                        help="MLX model for vLLM-MLX (quantized)")
    parser.add_argument("--axis", default=None,
                        choices=["non_prefix", "cold_start", "sweep"])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM-MLX (KVBoost only)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"\n  KVBoost model: {args.kvboost_model} (float16)")
    print(f"  vLLM model:    {'skipped' if args.skip_vllm else args.vllm_model + ' (4-bit)'}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Chunk size:    {args.chunk_size}")

    kv = KVBoostRunner(args.kvboost_model, chunk_size=args.chunk_size)
    vllm = None
    if not args.skip_vllm:
        try:
            vllm = VLLMMLXRunner(args.vllm_model)
        except Exception as e:
            log.warning("vLLM-MLX failed to init: %s — running KVBoost only", e)

    async def run():
        all_results: List[BenchResult] = []
        axes = [args.axis] if args.axis else ["non_prefix", "cold_start", "sweep"]

        for axis in axes:
            if axis == "non_prefix":
                all_results.extend(await axis_non_prefix(kv, vllm, args.max_tokens))
            elif axis == "cold_start":
                all_results.extend(await axis_cold_start(kv, vllm, args.max_tokens))
            elif axis == "sweep":
                all_results.extend(await axis_sweep(kv, vllm, args.max_tokens))

        print_summary(all_results)

        if vllm:
            print(f"\n  vLLM-MLX cache stats: {vllm.get_stats().get('system_kv_cache', 'N/A')}")
            await vllm.stop()

        if args.output:
            out_path = Path(args.output)
            out = {
                "kvboost_model": args.kvboost_model,
                "vllm_model": None if args.skip_vllm else args.vllm_model,
                "max_tokens": args.max_tokens,
                "chunk_size": args.chunk_size,
                "results": [asdict(r) for r in all_results],
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n  Results saved to {out_path}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
