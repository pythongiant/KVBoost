#!/usr/bin/env python3
"""
Benchmark harness for KV-cache reuse system.

Compares three modes across three workloads:
  BASELINE       — standard HF generate, no caching
  PREFIX_CACHE   — exact prefix caching only
  CHUNK_KV_REUSE — full chunk-level KV reuse + selective recompute

Workloads:
  1. system_prompt_reuse   — same system prompt, different user queries
  2. rag_doc_reuse         — same documents, different questions
  3. few_shot_reuse        — same few-shot examples, different inputs

Usage:
  python benchmark.py [--model MODEL] [--chunk-size N] [--runs N] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict
from typing import Dict, List

from kv_cache_system import InferenceEngine, GenerationMode, GenerationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


# -----------------------------------------------------------------------
# Workload definitions
# -----------------------------------------------------------------------

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
O(n²) complexity. With KV caching, previously computed K/V tensors are stored and reused, \
reducing generation to O(n) per step.

### Memory Layout
KV tensors have shape [batch, heads, seq_len, head_dim]. For a model with 22 layers, \
32 attention heads, head dimension 64, and a sequence of 512 tokens, the KV cache for \
one sequence requires 22 × 2 × 32 × 512 × 64 × 2 bytes ≈ 92MB in float16.

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

WORKLOADS = {
    "system_prompt_reuse": {
        "description": "Same system prompt, different user questions",
        "fixed_prefix": SYSTEM_PROMPT,
        "queries": [
            "What is the best strategy for reducing P99 latency in a high-throughput gRPC service?",
            "How should I design a rate limiter for a multi-tenant API gateway?",
            "Explain the trade-offs between synchronous and asynchronous model serving.",
            "What are the key considerations when choosing between Redis and Memcached for session storage?",
            "How do I implement circuit breakers in a microservices architecture?",
        ],
    },
    "rag_doc_reuse": {
        "description": "Same document context, different questions",
        "fixed_prefix": RAG_DOCUMENT,
        "queries": [
            "Based on the above, what is the memory formula for KV cache?",
            "According to the document, when is chunk-level caching more beneficial than prefix caching?",
            "What complexity does KV caching reduce generation to?",
            "How does RoPE affect chunk caching according to the document?",
            "Summarize the difference between prefix caching and chunk-level caching.",
        ],
    },
    "few_shot_reuse": {
        "description": "Same few-shot examples, different inputs",
        "fixed_prefix": FEW_SHOT_EXAMPLES,
        "queries": [
            "\nInput: Classify the sentiment: \"I love this so much!\"\nOutput:",
            "\nInput: Classify the sentiment: \"Complete waste of money.\"\nOutput:",
            "\nInput: Classify the sentiment: \"Works as advertised.\"\nOutput:",
            "\nInput: Classify the sentiment: \"Mind-blowingly good service!\"\nOutput:",
            "\nInput: Classify the sentiment: \"Not what I was expecting at all.\"\nOutput:",
        ],
    },
}


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

class BenchmarkRunner:
    def __init__(self, engine: InferenceEngine, n_runs: int = 3, max_new_tokens: int = 48):
        self.engine = engine
        self.n_runs = n_runs
        self.max_new_tokens = max_new_tokens
        self.results: List[Dict] = []

    def run_all(self) -> Dict:
        summary = {}

        for workload_name, workload in WORKLOADS.items():
            log.info("=" * 60)
            log.info("WORKLOAD: %s", workload_name)
            log.info("  %s", workload["description"])
            log.info("=" * 60)
            summary[workload_name] = self._run_workload(workload_name, workload)

        return summary

    def _run_workload(self, name: str, workload: Dict) -> Dict:
        prefix = workload["fixed_prefix"]
        queries = workload["queries"]

        modes = [
            GenerationMode.BASELINE,
            GenerationMode.PREFIX_CACHE,
            GenerationMode.CHUNK_KV_REUSE,
        ]

        workload_results: Dict[str, List[GenerationResult]] = {m.value: [] for m in modes}

        for mode in modes:
            log.info("  Mode: %s", mode.value)

            # Warm cache for non-baseline modes
            if mode != GenerationMode.BASELINE:
                n = self.engine.warm_chunks(prefix, position_offset=0)
                log.info("    Warmed %d chunks for prefix (%d tokens)",
                         n, len(self.engine._encode(prefix)))

            for run_i in range(self.n_runs):
                for q_i, query in enumerate(queries):
                    full_prompt = prefix + "\n\n" + query
                    result = self.engine.generate(
                        full_prompt,
                        max_new_tokens=self.max_new_tokens,
                        mode=mode,
                        do_sample=False,
                    )
                    workload_results[mode.value].append(result)

                    if run_i == 0:
                        log.info(
                            "    Q%d | TTFT=%.1fms | total=%.1fms | "
                            "tps=%.1f | reuse=%.0f%% | out=%r",
                            q_i + 1,
                            result.ttft_ms,
                            result.total_ms,
                            result.tokens_per_sec,
                            result.kv_reuse_ratio * 100,
                            result.output_text[:60],
                        )

        return self._compute_stats(workload_results)

    def _compute_stats(
        self, results: Dict[str, List[GenerationResult]]
    ) -> Dict:
        stats = {}
        for mode, runs in results.items():
            if not runs:
                continue
            ttfts = [r.ttft_ms for r in runs]
            totals = [r.total_ms for r in runs]
            tpss = [r.tokens_per_sec for r in runs]
            reuses = [r.kv_reuse_ratio for r in runs]
            stats[mode] = {
                "ttft_ms": {
                    "mean": round(statistics.mean(ttfts), 2),
                    "median": round(statistics.median(ttfts), 2),
                    "stdev": round(statistics.stdev(ttfts) if len(ttfts) > 1 else 0, 2),
                    "min": round(min(ttfts), 2),
                },
                "total_ms": {
                    "mean": round(statistics.mean(totals), 2),
                    "median": round(statistics.median(totals), 2),
                },
                "tokens_per_sec": {
                    "mean": round(statistics.mean(tpss), 2),
                },
                "kv_reuse_ratio": {
                    "mean": round(statistics.mean(reuses), 3),
                },
                "n_samples": len(runs),
            }
        return stats


def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for workload, modes in summary.items():
        print(f"\n{'─'*50}")
        print(f"Workload: {workload}")
        print(f"{'─'*50}")
        print(f"{'Mode':<20} {'TTFT(ms)':<12} {'Total(ms)':<12} {'TPS':<10} {'Cache%':<8}")
        print(f"{'─'*20} {'─'*11} {'─'*11} {'─'*9} {'─'*7}")

        baseline_ttft = modes.get("baseline", {}).get("ttft_ms", {}).get("mean", 1)

        for mode in ["baseline", "prefix_cache", "chunk_kv_reuse"]:
            if mode not in modes:
                continue
            m = modes[mode]
            ttft = m["ttft_ms"]["mean"]
            total = m["total_ms"]["mean"]
            tps = m["tokens_per_sec"]["mean"]
            reuse = m["kv_reuse_ratio"]["mean"] * 100
            speedup = baseline_ttft / max(ttft, 0.1)
            tag = f"({speedup:.1f}x)" if mode != "baseline" else ""
            print(f"{mode:<20} {ttft:<12.1f} {total:<12.1f} {tps:<10.1f} {reuse:<8.0f} {tag}")

    print()


def main():
    parser = argparse.ArgumentParser(description="KV Cache Reuse Benchmark")
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=128,
        help="Tokens per chunk (default: 128)",
    )
    parser.add_argument(
        "--recompute-overlap", type=int, default=16,
        help="Tokens to recompute at seams (default: 16)",
    )
    parser.add_argument(
        "--runs", type=int, default=2,
        help="Repetitions per query (default: 2)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=48,
        help="Tokens to generate per query (default: 48)",
    )
    parser.add_argument(
        "--output", default="",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--workload", default="all",
        choices=["all", "system_prompt_reuse", "rag_doc_reuse", "few_shot_reuse"],
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load engine
    log.info("Loading engine with model=%s chunk_size=%d", args.model, args.chunk_size)
    engine = InferenceEngine.from_pretrained(
        model_name=args.model,
        chunk_size=args.chunk_size,
        recompute_overlap=args.recompute_overlap,
        max_chunks=256,
    )
    log.info("Engine ready. Device: %s", engine.device)

    # Select workloads
    if args.workload == "all":
        workloads = WORKLOADS
    else:
        workloads = {args.workload: WORKLOADS[args.workload]}

    runner = BenchmarkRunner(
        engine=engine,
        n_runs=args.runs,
        max_new_tokens=args.max_new_tokens,
    )

    t_start = time.perf_counter()

    # Patch runner to use selected workloads
    summary = {}
    for wname, wdata in workloads.items():
        log.info("=" * 60)
        log.info("WORKLOAD: %s", wname)
        log.info("  %s", wdata["description"])
        log.info("=" * 60)
        summary[wname] = runner._run_workload(wname, wdata)

    t_end = time.perf_counter()

    print_summary(summary)
    log.info("Total benchmark time: %.1fs", t_end - t_start)
    log.info("Cache stats: %s", engine.cache_stats())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
