#!/usr/bin/env python3
"""
Experiment 1: Scale Beyond TinyLlama
=====================================
Measures TTFT speedup across multiple model sizes to show that the
KV-cache reuse benefit grows with model scale.

Models tested (configurable):
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0   (1.1B)
  - Qwen/Qwen2-0.5B                      (0.5B)
  - meta-llama/Llama-3.2-1B              (1B)
  - meta-llama/Llama-3.2-3B              (3B)
  - meta-llama/Meta-Llama-3-8B           (8B, needs ~16GB VRAM)

Usage:
  python 01_scale_models.py
  python 01_scale_models.py --models "TinyLlama/TinyLlama-1.1B-Chat-v1.0,Qwen/Qwen2-0.5B"
  python 01_scale_models.py --runs 5 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

from utils import (
    get_logger, load_engine, save_results, compute_stats,
    SYSTEM_PROMPT, RAG_DOCUMENT, QUERIES, RAG_QUERIES,
    GenerationMode,
)

log = get_logger("01_scale_models")

DEFAULT_MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2-0.5B",
]

WORKLOADS = {
    "system_prompt_reuse": {
        "prefix": SYSTEM_PROMPT,
        "queries": QUERIES,
    },
    "rag_doc_reuse": {
        "prefix": RAG_DOCUMENT,
        "queries": RAG_QUERIES,
    },
}


def benchmark_model(model_name: str, chunk_size: int, runs: int, max_new_tokens: int) -> dict:
    log.info("=" * 60)
    log.info("MODEL: %s", model_name)
    log.info("=" * 60)

    try:
        engine = load_engine(model_name=model_name, chunk_size=chunk_size)
    except Exception as e:
        log.error("Failed to load %s: %s", model_name, e)
        return {"model": model_name, "error": str(e)}

    # Count parameters
    n_params = sum(p.numel() for p in engine.model.parameters())
    n_layers = engine.model.config.num_hidden_layers
    n_heads = engine.model.config.num_attention_heads
    head_dim = engine.model.config.hidden_size // n_heads

    model_results = {
        "model": model_name,
        "params_billions": round(n_params / 1e9, 2),
        "num_layers": n_layers,
        "num_heads": n_heads,
        "head_dim": head_dim,
        "device": engine.device,
        "chunk_size": chunk_size,
        "workloads": {},
    }

    modes = [GenerationMode.BASELINE, GenerationMode.PREFIX_CACHE, GenerationMode.CHUNK_KV_REUSE]

    for wname, wdata in WORKLOADS.items():
        log.info("  Workload: %s", wname)
        prefix = wdata["prefix"]
        queries = wdata["queries"]

        mode_results = {}
        for mode in modes:
            ttfts, totals, tps_list = [], [], []

            if mode != GenerationMode.BASELINE:
                engine.warm_chunks(prefix, position_offset=0)

            for run_i in range(runs):
                for query in queries:
                    prompt = prefix + "\n\n" + query
                    r = engine.generate(
                        prompt, max_new_tokens=max_new_tokens,
                        mode=mode, do_sample=False,
                    )
                    # Skip first run as warmup
                    if run_i > 0 or runs == 1:
                        ttfts.append(r.ttft_ms)
                        totals.append(r.total_ms)
                        tps_list.append(r.tokens_per_sec)

            mode_results[mode.value] = {
                "ttft_ms": compute_stats(ttfts),
                "total_ms": compute_stats(totals),
                "tokens_per_sec": compute_stats(tps_list),
                "n_samples": len(ttfts),
            }
            log.info("    %s: TTFT mean=%.1fms", mode.value,
                     mode_results[mode.value]["ttft_ms"].get("mean", 0))

        # Compute speedups
        baseline_ttft = mode_results["baseline"]["ttft_ms"].get("mean", 1)
        for mode_name in ["prefix_cache", "chunk_kv_reuse"]:
            if mode_name in mode_results:
                mode_ttft = mode_results[mode_name]["ttft_ms"].get("mean", 1)
                mode_results[mode_name]["ttft_speedup"] = round(
                    baseline_ttft / max(mode_ttft, 0.01), 2
                )

        model_results["workloads"][wname] = mode_results

    # Cleanup to free GPU memory before next model
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_results


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Scale across model sizes")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model names")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per query (first is warmup unless runs=1)")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=str, default="01_scale_models.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    all_results = []

    for model_name in models:
        result = benchmark_model(model_name, args.chunk_size, args.runs, args.max_new_tokens)
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("SCALING SUMMARY")
    print("=" * 80)
    print(f"{'Model':<45} {'Params':<8} {'Baseline TTFT':<15} {'ChunkKV TTFT':<15} {'Speedup':<8}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<45} ERROR: {r['error']}")
            continue
        for wname, modes in r["workloads"].items():
            b_ttft = modes.get("baseline", {}).get("ttft_ms", {}).get("mean", 0)
            c_ttft = modes.get("chunk_kv_reuse", {}).get("ttft_ms", {}).get("mean", 0)
            speedup = modes.get("chunk_kv_reuse", {}).get("ttft_speedup", 0)
            print(f"{r['model']:<45} {r['params_billions']:<8} {b_ttft:<15.1f} {c_ttft:<15.1f} {speedup:<8.1f}x")

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
