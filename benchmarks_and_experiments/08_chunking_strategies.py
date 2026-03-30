#!/usr/bin/env python3
"""
Experiment 8: Chunking Strategy Comparison
============================================
Compares the three chunking strategies (FIXED, SEMANTIC, DOCUMENT)
across workloads, measuring:

  - Cache hit rate
  - TTFT
  - Output quality (match vs baseline)
  - Memory efficiency (chunks stored per prompt)

Usage:
  python 08_chunking_strategies.py
  python 08_chunking_strategies.py --chunk-size 128 --runs 3
"""

from __future__ import annotations

import argparse
import gc
from typing import Dict, List

import torch

from utils import (
    get_logger, save_results, compute_stats,
    SYSTEM_PROMPT, RAG_DOCUMENT, QUERIES, RAG_QUERIES,
    GenerationMode,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import InferenceEngine
from chunk_registry import ChunkStrategy

log = get_logger("08_chunking_strategies")

# RAG workload: multiple documents, reused across different queries
DOCUMENTS = [
    (
        "## Transformer Architecture\n\n"
        "The transformer model uses self-attention to process sequences in parallel. "
        "Multi-head attention allows attending to different representation subspaces. "
        "The architecture includes encoder and decoder stacks with residual connections "
        "and layer normalization. Feed-forward networks process each position independently.\n\n"
        "Key innovations include positional encodings for sequence order and scaled "
        "dot-product attention for computing relevance between tokens."
    ),
    (
        "## Distributed Systems Fundamentals\n\n"
        "The CAP theorem establishes that distributed systems cannot simultaneously "
        "guarantee consistency, availability, and partition tolerance. Consensus protocols "
        "like Raft and Paxos solve the distributed agreement problem. Vector clocks track "
        "causality in distributed systems without synchronized clocks.\n\n"
        "Gossip protocols provide eventual consistency with low overhead. CRDTs "
        "(Conflict-free Replicated Data Types) enable concurrent updates without "
        "coordination by designing data structures that commute."
    ),
    (
        "## Database Indexing\n\n"
        "B-tree indexes organize data in balanced tree structures for efficient range "
        "queries and point lookups. Hash indexes provide O(1) lookups but don't support "
        "range queries. LSM (Log-Structured Merge) trees optimize write performance by "
        "batching writes in memory before flushing to sorted disk files.\n\n"
        "Bloom filters provide probabilistic membership testing with no false negatives. "
        "They are commonly used to avoid unnecessary disk reads in LSM-tree based "
        "storage engines like RocksDB and LevelDB."
    ),
]

DOCUMENT_QUERIES = [
    "What is multi-head attention and why is it useful?",
    "Explain the CAP theorem and its practical implications.",
    "How do LSM trees optimize write performance?",
    "Compare B-tree and hash indexes for different query patterns.",
    "What are CRDTs and when should they be used?",
]


def build_engine_with_strategy(model_name: str, chunk_size: int,
                                strategy: ChunkStrategy) -> InferenceEngine:
    """Build an engine with a specific chunking strategy."""
    engine = InferenceEngine.from_pretrained(
        model_name=model_name,
        chunk_size=chunk_size,
        max_chunks=256,
    )
    # Override the chunk strategy
    engine.chunk_registry.strategy = strategy
    engine.assembler.registry.strategy = strategy
    return engine


def run_strategy_benchmark(model_name: str, chunk_size: int,
                           strategy: ChunkStrategy,
                           runs: int, max_new_tokens: int) -> Dict:
    """Run benchmark for one chunking strategy."""
    log.info("  Strategy: %s", strategy.value)
    engine = build_engine_with_strategy(model_name, chunk_size, strategy)

    results = {"strategy": strategy.value, "workloads": {}}

    # Workload 1: System prompt reuse
    prefix = SYSTEM_PROMPT
    queries = QUERIES[:3]

    # Get baseline outputs
    baseline_outputs = []
    for query in queries:
        prompt = prefix + "\n\n" + query
        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=GenerationMode.BASELINE, do_sample=False)
        baseline_outputs.append(r.output_text)

    # Warm and test
    n_chunks = engine.warm_chunks(prefix, position_offset=0)
    token_ids = engine._encode(prefix)
    chunk_splits = engine.chunk_registry.split(token_ids, prefix)

    ttfts, matches = [], []
    for run_i in range(runs):
        for qi, query in enumerate(queries):
            prompt = prefix + "\n\n" + query
            r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)
            if run_i > 0 or runs == 1:
                ttfts.append(r.ttft_ms)
                matches.append(1.0 if r.output_text == baseline_outputs[qi] else 0.0)

    stats = engine.cache_stats()
    results["workloads"]["system_prompt"] = {
        "ttft_ms": compute_stats(ttfts),
        "output_match_rate": round(sum(matches) / max(len(matches), 1), 3),
        "chunks_created": n_chunks,
        "chunk_splits": len(chunk_splits),
        "chunk_sizes": [len(s[2]) for s in chunk_splits],
        "cache_hit_rate": stats["hit_rate"],
        "hot_memory_mb": stats["hot_memory_mb"],
    }

    # Workload 2: Multi-document RAG
    all_docs = "\n\n".join(DOCUMENTS)

    baseline_outputs_rag = []
    for query in DOCUMENT_QUERIES[:3]:
        prompt = all_docs + "\n\nQuestion: " + query
        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=GenerationMode.BASELINE, do_sample=False)
        baseline_outputs_rag.append(r.output_text)

    # Warm documents
    for doc in DOCUMENTS:
        engine.warm_chunks(doc, position_offset=0)

    token_ids_rag = engine._encode(all_docs)
    chunk_splits_rag = engine.chunk_registry.split(token_ids_rag, all_docs)

    ttfts_rag, matches_rag = [], []
    for run_i in range(runs):
        for qi, query in enumerate(DOCUMENT_QUERIES[:3]):
            prompt = all_docs + "\n\nQuestion: " + query
            r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)
            if run_i > 0 or runs == 1:
                ttfts_rag.append(r.ttft_ms)
                matches_rag.append(1.0 if r.output_text == baseline_outputs_rag[qi] else 0.0)

    stats2 = engine.cache_stats()
    results["workloads"]["multi_doc_rag"] = {
        "ttft_ms": compute_stats(ttfts_rag),
        "output_match_rate": round(sum(matches_rag) / max(len(matches_rag), 1), 3),
        "chunk_splits": len(chunk_splits_rag),
        "chunk_sizes": [len(s[2]) for s in chunk_splits_rag],
        "cache_hit_rate": stats2["hit_rate"],
        "hot_memory_mb": stats2["hot_memory_mb"],
    }

    # Workload 3: Document reordering (tests if strategy handles document
    # appearing at different positions)
    reorder_results = []
    for i, doc in enumerate(DOCUMENTS):
        # Put one document first, others after
        reordered = doc + "\n\n" + "\n\n".join(
            d for j, d in enumerate(DOCUMENTS) if j != i
        )
        prompt = reordered + "\n\nQuestion: " + DOCUMENT_QUERIES[i]

        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)
        reorder_results.append({
            "primary_doc_idx": i,
            "ttft_ms": r.ttft_ms,
            "kv_reuse_ratio": r.kv_reuse_ratio,
        })

    results["workloads"]["document_reordering"] = {
        "results": reorder_results,
        "avg_reuse_ratio": round(
            sum(r["kv_reuse_ratio"] for r in reorder_results) / len(reorder_results), 3
        ),
    }

    del engine
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 8: Chunking strategies")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=str, default="08_chunking_strategies.json")
    args = parser.parse_args()

    strategies = [ChunkStrategy.FIXED, ChunkStrategy.SEMANTIC, ChunkStrategy.DOCUMENT]
    all_results = []

    for strategy in strategies:
        result = run_strategy_benchmark(
            args.model, args.chunk_size, strategy,
            args.runs, args.max_new_tokens,
        )
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 90)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 90)

    for wname in ["system_prompt", "multi_doc_rag"]:
        print(f"\n--- {wname} ---")
        print(f"{'Strategy':<12} {'TTFT(ms)':<12} {'Match%':<10} "
              f"{'HitRate':<10} {'Chunks':<10} {'MemMB':<10}")
        print("-" * 64)
        for r in all_results:
            w = r["workloads"].get(wname, {})
            print(
                f"{r['strategy']:<12} "
                f"{w.get('ttft_ms', {}).get('mean', 0):<12.1f} "
                f"{w.get('output_match_rate', 0) * 100:<10.1f} "
                f"{w.get('cache_hit_rate', 0):<10.3f} "
                f"{w.get('chunk_splits', 0):<10} "
                f"{w.get('hot_memory_mb', 0):<10.1f}"
            )

    print(f"\n--- Document Reordering (avg reuse ratio) ---")
    for r in all_results:
        reorder = r["workloads"].get("document_reordering", {})
        print(f"  {r['strategy']}: {reorder.get('avg_reuse_ratio', 0) * 100:.1f}%")

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
