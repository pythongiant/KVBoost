#!/usr/bin/env python3
"""
Experiment 5: Realistic Workloads
==================================
Tests the system under production-like conditions:

  1. Long-context RAG: 4K-8K token prompts with multiple retrieved documents
  2. Multi-turn conversation: growing history, amortized TTFT
  3. Server simulation: N concurrent users sharing system prompt

Usage:
  python 05_realistic_workloads.py
  python 05_realistic_workloads.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List

from utils import (
    get_logger, load_engine, save_results, compute_stats,
    SYSTEM_PROMPT, RAG_DOCUMENT,
    GenerationMode,
)

log = get_logger("05_realistic_workloads")

# Multiple documents for long-context RAG
RAG_DOCUMENTS = [
    # Document 1: Transformer architecture
    (
        "## Document 1: Transformer Architecture\n\n"
        "The transformer architecture was introduced in 'Attention Is All You Need' (2017). "
        "It consists of an encoder-decoder structure where both components use self-attention "
        "and feed-forward layers. The encoder processes the input sequence in parallel, while "
        "the decoder generates output tokens autoregressively. Multi-head attention allows the "
        "model to jointly attend to information from different representation subspaces. "
        "The attention function maps a query and a set of key-value pairs to an output, "
        "computed as a weighted sum of the values where weights are determined by compatibility "
        "between the query and keys. Layer normalization and residual connections are used "
        "throughout the architecture for training stability. Positional encodings provide "
        "sequence order information since attention is permutation-invariant."
    ),
    # Document 2: KV cache optimization
    (
        "## Document 2: KV Cache Optimization\n\n"
        "During autoregressive generation, the KV cache stores previously computed key and "
        "value tensors to avoid redundant computation. Without caching, generating each new "
        "token requires attending to all previous tokens, leading to quadratic complexity. "
        "With KV caching, only the new token's query needs to attend to all cached keys. "
        "Memory management for KV caches is critical: PagedAttention divides the KV cache "
        "into blocks and manages them like virtual memory pages. Prefix caching stores KV "
        "tensors for common prefixes (system prompts) and shares them across requests. "
        "Quantization of KV caches (e.g., to INT8 or INT4) can reduce memory by 2-4x "
        "with minimal quality loss. Speculative decoding uses a smaller draft model to "
        "propose multiple tokens, verified in parallel by the target model."
    ),
    # Document 3: Distributed inference
    (
        "## Document 3: Distributed Inference\n\n"
        "Large language models often exceed the memory capacity of a single GPU, requiring "
        "distributed inference strategies. Tensor parallelism splits individual layers across "
        "GPUs, with each GPU computing a portion of the attention heads or feed-forward "
        "neurons. Pipeline parallelism assigns different layers to different GPUs, creating "
        "a processing pipeline. Expert parallelism distributes mixture-of-experts (MoE) "
        "layers across GPUs. Load balancing across multiple replicas uses techniques like "
        "least-connections routing or hash-based affinity for cache locality. Batching "
        "strategies include static batching (fixed batch, wait for completion) and continuous "
        "batching (in-flight batching) which starts processing new requests as soon as "
        "existing ones complete. Continuous batching significantly improves GPU utilization."
    ),
    # Document 4: Model compression
    (
        "## Document 4: Model Compression\n\n"
        "Model compression techniques reduce the size and computational requirements of "
        "large language models. Quantization reduces precision from FP16 to INT8 or INT4, "
        "with methods like GPTQ, AWQ, and SqueezeLLM achieving near-lossless compression. "
        "Knowledge distillation trains a smaller student model to mimic a larger teacher "
        "model's behavior. Pruning removes redundant weights or attention heads based on "
        "importance scores. Low-rank factorization approximates weight matrices as products "
        "of smaller matrices using techniques like LoRA. Dynamic sparsity activates only a "
        "subset of parameters for each input. Structured pruning removes entire neurons, "
        "heads, or layers for hardware-friendly speedups. The quality-compression trade-off "
        "varies by task: mathematical reasoning is more sensitive to quantization than "
        "general text generation."
    ),
]

# Conversation turns for multi-turn test
CONVERSATION_TURNS = [
    {"role": "user", "content": "What is the difference between TCP and UDP?"},
    {"role": "assistant", "content": "TCP provides reliable, ordered delivery with connection setup and flow control. UDP is connectionless and faster but unreliable."},
    {"role": "user", "content": "When would I use UDP over TCP?"},
    {"role": "assistant", "content": "Use UDP for real-time applications like gaming, video streaming, or DNS lookups where speed matters more than guaranteed delivery."},
    {"role": "user", "content": "How does TCP handle congestion?"},
    {"role": "assistant", "content": "TCP uses algorithms like slow start, congestion avoidance, fast retransmit, and fast recovery. Modern variants include CUBIC and BBR."},
    {"role": "user", "content": "What is BBR and how does it differ from CUBIC?"},
    {"role": "assistant", "content": "BBR (Bottleneck Bandwidth and RTT) estimates available bandwidth directly rather than using loss as a signal like CUBIC. BBR performs better on lossy links."},
    {"role": "user", "content": "Can you explain how BBR estimates bandwidth?"},
    {"role": "assistant", "content": "BBR periodically probes for bandwidth by increasing send rate and measuring delivery rate. It also probes for RTT by briefly reducing the send rate to drain queues."},
    {"role": "user", "content": "What are the practical deployment considerations for BBR vs CUBIC?"},
    {"role": "assistant", "content": "BBR can be unfair to CUBIC flows in shared networks. BBRv2 addresses this. For deployment, test on representative traffic and monitor retransmission rates."},
    {"role": "user", "content": "How do modern CDNs handle TCP optimization?"},
    {"role": "assistant", "content": "CDNs use edge servers close to users, persistent connections, connection pooling, and TCP optimizations like TFO (TCP Fast Open) and custom congestion control."},
    {"role": "user", "content": "Summarize the key takeaways from our conversation about networking."},
]

# Simulated user queries for server simulation
SERVER_QUERIES = [
    "What is the time complexity of quicksort?",
    "How does garbage collection work in Java?",
    "Explain the difference between processes and threads.",
    "What is a mutex and when should I use one?",
    "How does DNS resolution work?",
    "What is the difference between REST and GraphQL?",
    "Explain how HTTPS encryption works.",
    "What is a load balancer and how does it work?",
    "How do database indexes improve query performance?",
    "What is eventual consistency in distributed systems?",
    "Explain the observer design pattern.",
    "What is dependency injection and why use it?",
    "How does a hash table handle collisions?",
    "What is the CAP theorem?",
    "Explain how virtual memory works.",
    "What is a microservice architecture?",
    "How does Git branching work internally?",
    "What is WebSocket and when to use it over HTTP?",
    "Explain the difference between SQL and NoSQL databases.",
    "What is container orchestration with Kubernetes?",
]


def experiment_long_context_rag(engine, runs: int, max_new_tokens: int) -> Dict:
    """
    Test with 2-4 retrieved documents concatenated into the prompt.
    Measures TTFT as context length grows.
    """
    log.info("=== Long-context RAG ===")

    questions = [
        "Compare the compression techniques mentioned across the documents.",
        "How do KV cache optimizations relate to distributed inference strategies?",
        "What are the trade-offs between quantization and pruning for model compression?",
    ]

    results = []
    for n_docs in [1, 2, 3, 4]:
        context = "\n\n".join(RAG_DOCUMENTS[:n_docs])

        # Warm cache for documents
        engine.warm_chunks(context, position_offset=0)

        for mode in [GenerationMode.BASELINE, GenerationMode.CHUNK_KV_REUSE]:
            ttfts = []
            for run_i in range(runs):
                for q in questions:
                    prompt = context + "\n\nQuestion: " + q
                    r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                        mode=mode, do_sample=False)
                    if run_i > 0 or runs == 1:
                        ttfts.append(r.ttft_ms)

            prompt_tokens = len(engine._encode(context + "\n\nQuestion: " + questions[0]))
            results.append({
                "n_documents": n_docs,
                "mode": mode.value,
                "prompt_tokens": prompt_tokens,
                "ttft_ms": compute_stats(ttfts),
                "n_samples": len(ttfts),
            })
            log.info("  %d docs, %s: TTFT mean=%.1fms (%d tokens)",
                     n_docs, mode.value,
                     results[-1]["ttft_ms"].get("mean", 0), prompt_tokens)

    return {"long_context_rag": results}


def experiment_multi_turn(engine, max_new_tokens: int) -> Dict:
    """
    Simulate a multi-turn conversation where history grows.
    Measure amortized TTFT across turns.
    """
    log.info("=== Multi-turn conversation ===")

    results = []
    history = SYSTEM_PROMPT + "\n\n"

    for turn_idx in range(0, len(CONVERSATION_TURNS), 2):
        user_turn = CONVERSATION_TURNS[turn_idx]
        history += f"User: {user_turn['content']}\nAssistant: "

        # Warm cache for history so far
        engine.warm_chunks(history, position_offset=0)

        for mode in [GenerationMode.BASELINE, GenerationMode.CHUNK_KV_REUSE]:
            r = engine.generate(history, max_new_tokens=max_new_tokens,
                                mode=mode, do_sample=False)
            results.append({
                "turn": turn_idx // 2 + 1,
                "mode": mode.value,
                "history_tokens": len(engine._encode(history)),
                "ttft_ms": r.ttft_ms,
                "kv_reuse_ratio": r.kv_reuse_ratio,
            })
            log.info("  Turn %d, %s: TTFT=%.1fms, reuse=%.0f%%, history=%d tokens",
                     turn_idx // 2 + 1, mode.value,
                     r.ttft_ms, r.kv_reuse_ratio * 100,
                     len(engine._encode(history)))

        # Add assistant response to history
        if turn_idx + 1 < len(CONVERSATION_TURNS):
            assistant_turn = CONVERSATION_TURNS[turn_idx + 1]
            history += f"{assistant_turn['content']}\n"

    return {"multi_turn": results}


def experiment_server_simulation(engine, n_users: int, queries_per_user: int,
                                 max_new_tokens: int) -> Dict:
    """
    Simulate N users sending queries with a shared system prompt.
    Measure aggregate throughput and per-request TTFT.
    """
    log.info("=== Server simulation: %d users, %d queries each ===", n_users, queries_per_user)

    engine.warm_chunks(SYSTEM_PROMPT, position_offset=0)

    results = {"baseline": [], "chunk_kv_reuse": []}
    total_queries = n_users * queries_per_user
    query_pool = SERVER_QUERIES

    for mode in [GenerationMode.BASELINE, GenerationMode.CHUNK_KV_REUSE]:
        t_start = time.perf_counter()
        query_count = 0

        for user_id in range(n_users):
            # Each user sends queries_per_user requests
            user_queries = random.sample(query_pool, min(queries_per_user, len(query_pool)))

            for query in user_queries:
                prompt = SYSTEM_PROMPT + "\n\n" + query
                r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                    mode=mode, do_sample=False)
                results[mode.value].append({
                    "user_id": user_id,
                    "ttft_ms": r.ttft_ms,
                    "total_ms": r.total_ms,
                    "tokens_per_sec": r.tokens_per_sec,
                    "kv_reuse_ratio": r.kv_reuse_ratio,
                })
                query_count += 1

        elapsed = time.perf_counter() - t_start
        log.info("  %s: %d queries in %.1fs (%.1f QPS)",
                 mode.value, query_count, elapsed, query_count / elapsed)

    # Compute aggregate stats
    summary = {}
    for mode in ["baseline", "chunk_kv_reuse"]:
        runs = results[mode]
        summary[mode] = {
            "ttft_ms": compute_stats([r["ttft_ms"] for r in runs]),
            "total_ms": compute_stats([r["total_ms"] for r in runs]),
            "tokens_per_sec": compute_stats([r["tokens_per_sec"] for r in runs]),
            "total_queries": len(runs),
            "total_time_sec": round(sum(r["total_ms"] for r in runs) / 1000, 2),
            "aggregate_qps": round(
                len(runs) / max(sum(r["total_ms"] for r in runs) / 1000, 0.001), 2
            ),
        }

    return {"server_simulation": summary, "server_details": results}


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Realistic workloads")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--n-users", type=int, default=5)
    parser.add_argument("--queries-per-user", type=int, default=4)
    parser.add_argument("--output", type=str, default="05_realistic_workloads.json")
    args = parser.parse_args()

    engine = load_engine(model_name=args.model, chunk_size=args.chunk_size)

    all_results = {}

    # 1. Long-context RAG
    r1 = experiment_long_context_rag(engine, args.runs, args.max_new_tokens)
    all_results.update(r1)

    # 2. Multi-turn conversation
    r2 = experiment_multi_turn(engine, args.max_new_tokens)
    all_results.update(r2)

    # 3. Server simulation
    r3 = experiment_server_simulation(
        engine, args.n_users, args.queries_per_user, args.max_new_tokens
    )
    all_results.update(r3)

    # Print summary
    print("\n" + "=" * 70)
    print("REALISTIC WORKLOAD SUMMARY")
    print("=" * 70)

    # Long-context RAG
    print("\n--- Long-Context RAG ---")
    print(f"{'Docs':<6} {'Tokens':<10} {'Baseline TTFT':<16} {'ChunkKV TTFT':<16} {'Speedup':<10}")
    rag = all_results["long_context_rag"]
    for i in range(0, len(rag), 2):
        b = rag[i]
        c = rag[i + 1] if i + 1 < len(rag) else None
        b_ttft = b["ttft_ms"].get("mean", 0)
        c_ttft = c["ttft_ms"].get("mean", 0) if c else 0
        speedup = b_ttft / max(c_ttft, 0.01) if c else 0
        print(f"{b['n_documents']:<6} {b['prompt_tokens']:<10} "
              f"{b_ttft:<16.1f} {c_ttft:<16.1f} {speedup:<10.1f}x")

    # Multi-turn
    print("\n--- Multi-Turn Conversation ---")
    mt = all_results["multi_turn"]
    for i in range(0, len(mt), 2):
        b = mt[i]
        c = mt[i + 1] if i + 1 < len(mt) else None
        print(f"  Turn {b['turn']}: baseline={b['ttft_ms']:.1f}ms, "
              f"chunk_kv={c['ttft_ms']:.1f}ms (reuse={c['kv_reuse_ratio']*100:.0f}%)"
              if c else f"  Turn {b['turn']}: baseline={b['ttft_ms']:.1f}ms")

    # Server sim
    if "server_simulation" in all_results:
        print("\n--- Server Simulation ---")
        ss = all_results["server_simulation"]
        for mode in ["baseline", "chunk_kv_reuse"]:
            s = ss[mode]
            print(f"  {mode}: TTFT={s['ttft_ms'].get('mean', 0):.1f}ms, "
                  f"QPS={s['aggregate_qps']}")

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
