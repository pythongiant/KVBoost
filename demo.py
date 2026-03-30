#!/usr/bin/env python3
"""
Quick correctness + sanity demo.
Run before the full benchmark to verify KV reuse produces coherent output.

  python demo.py [--model MODEL] [--chunk-size N]
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger("demo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    from engine import InferenceEngine, GenerationMode

    log.info("Loading %s ...", args.model)
    engine = InferenceEngine.from_pretrained(
        model_name=args.model,
        chunk_size=args.chunk_size,
        recompute_overlap=8,
    )
    log.info("Loaded. Device: %s", engine.device)

    # ----------------------------------------------------------------
    # Demo 1: Same system prompt, three different queries
    # ----------------------------------------------------------------
    system = (
        "You are a concise technical assistant. "
        "Answer in one sentence. "
        "Be direct and factual. "
        "Do not use bullet points. "
        "Always state your answer first, then briefly explain why."
    )

    queries = [
        "What is a KV cache?",
        "What is a transformer attention head?",
        "What is prefill vs decode in LLM inference?",
    ]

    print("\n" + "=" * 60)
    print("DEMO 1: System prompt reuse")
    print("=" * 60)

    # Warm the cache with system prompt
    n_chunks = engine.warm_chunks(system, position_offset=0)
    log.info("Warmed %d chunks for system prompt.", n_chunks)

    for i, q in enumerate(queries):
        prompt = system + "\n\n" + q
        print(f"\nQuery {i+1}: {q}")
        print("-" * 40)

        for mode in [GenerationMode.BASELINE, GenerationMode.CHUNK_KV_REUSE]:
            r = engine.generate(prompt, max_new_tokens=args.max_new_tokens,
                                mode=mode, do_sample=False)
            tag = "BASELINE " if mode == GenerationMode.BASELINE else "CHUNK_KV "
            print(f"  [{tag}] TTFT={r.ttft_ms:6.1f}ms  reuse={r.kv_reuse_ratio*100:4.0f}%  "
                  f"output: {r.output_text!r}")

    # ----------------------------------------------------------------
    # Demo 2: RAG-style (doc + question)
    # ----------------------------------------------------------------
    doc = (
        "The transformer architecture was introduced in the paper "
        "'Attention Is All You Need' (Vaswani et al., 2017). "
        "It replaces recurrence with self-attention, allowing parallelization "
        "during training. The encoder processes input sequences into continuous "
        "representations; the decoder generates output sequences auto-regressively. "
        "Multi-head attention computes attention over different representation subspaces "
        "simultaneously, improving the model's ability to capture varied relationships "
        "in the data. Positional encodings inject sequence order information since the "
        "architecture itself is permutation-invariant."
    )

    rag_queries = [
        "What paper introduced the transformer?",
        "What does multi-head attention do?",
    ]

    print("\n\n" + "=" * 60)
    print("DEMO 2: RAG document reuse")
    print("=" * 60)

    n_chunks = engine.warm_chunks(doc, position_offset=0)
    log.info("Warmed %d chunks for document.", n_chunks)

    for i, q in enumerate(rag_queries):
        prompt = doc + "\n\nQuestion: " + q + "\nAnswer:"
        print(f"\nQuestion {i+1}: {q}")
        print("-" * 40)

        for mode in [GenerationMode.BASELINE, GenerationMode.CHUNK_KV_REUSE]:
            r = engine.generate(prompt, max_new_tokens=args.max_new_tokens,
                                mode=mode, do_sample=False)
            tag = "BASELINE " if mode == GenerationMode.BASELINE else "CHUNK_KV "
            print(f"  [{tag}] TTFT={r.ttft_ms:6.1f}ms  reuse={r.kv_reuse_ratio*100:4.0f}%  "
                  f"output: {r.output_text!r}")

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("Cache stats:", engine.cache_stats())
    print("=" * 60)


if __name__ == "__main__":
    main()
