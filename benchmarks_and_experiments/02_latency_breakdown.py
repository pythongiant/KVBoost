#!/usr/bin/env python3
"""
Experiment 2: Latency Breakdown
================================
Instruments the generation pipeline to measure where time is spent:

  1. KV cache lookup time
  2. CPU -> GPU transfer time
  3. Selective recompute time at seams
  4. Live-token prefill time
  5. Per-token decode throughput

Produces a stacked bar chart data breakdown for each mode.

Usage:
  python 02_latency_breakdown.py
  python 02_latency_breakdown.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runs 5
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import torch

from utils import (
    get_logger, load_engine, save_results, compute_stats,
    SYSTEM_PROMPT, RAG_DOCUMENT, QUERIES, RAG_QUERIES,
    GenerationMode,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache_manager import KVCacheManager
from chunk_registry import ChunkRegistry
from prompt_assembler import PromptAssembler, AssemblyMode
from selective_recompute import SelectiveRecompute
from models import AssembledPrompt
from transformers import DynamicCache

log = get_logger("02_latency_breakdown")


def breakdown_generate(engine, prompt: str, max_new_tokens: int = 48) -> Dict:
    """Generate with fine-grained timing of each pipeline stage."""
    timings = {}

    token_ids = engine._encode(prompt)
    timings["prompt_tokens"] = len(token_ids)

    # Stage 1: Cache lookup + assembly
    t0 = time.perf_counter()
    assembled = engine.assembler.assemble(token_ids)
    timings["cache_lookup_ms"] = (time.perf_counter() - t0) * 1000
    timings["cached_tokens"] = assembled.cached_length
    timings["live_tokens"] = len(assembled.live_token_ids)
    timings["cache_hit_ratio"] = assembled.cache_hit_ratio
    timings["num_seams"] = max(0, len(assembled.chunk_boundaries) - 1)

    # Stage 2: Selective recompute
    t0 = time.perf_counter()
    if len(assembled.chunk_boundaries) > 1:
        assembled = engine.selective_recompute.apply(assembled, engine.model)
    timings["selective_recompute_ms"] = (time.perf_counter() - t0) * 1000

    # Stage 3: CPU -> GPU transfer
    past_kv = assembled.cached_past_kv
    t0 = time.perf_counter()
    if past_kv is not None:
        past_kv = tuple(
            (layer[0].to(engine.device), layer[1].to(engine.device))
            for layer in past_kv
        )
    timings["cpu_to_gpu_transfer_ms"] = (time.perf_counter() - t0) * 1000

    # Stage 4: Live-token prefill
    live_ids = assembled.live_token_ids
    cached_len = assembled.cached_length
    generated = []

    t0 = time.perf_counter()
    if live_ids:
        input_ids = torch.tensor([live_ids], dtype=torch.long, device=engine.device)
        pos_ids = torch.arange(
            cached_len, cached_len + len(live_ids),
            dtype=torch.long, device=engine.device,
        ).unsqueeze(0)

        with torch.no_grad():
            out = engine.model(
                input_ids=input_ids,
                past_key_values=engine._as_cache(past_kv),
                position_ids=pos_ids,
                use_cache=True,
            )
        past_kv = engine._normalize_past_kv(out.past_key_values)
        next_token = engine._sample(out.logits[:, -1, :], 1.0, False)
        generated.append(next_token)
    timings["live_prefill_ms"] = (time.perf_counter() - t0) * 1000

    # Stage 5: Autoregressive decode
    t0 = time.perf_counter()
    cur_pos = cached_len + len(live_ids)
    while len(generated) < max_new_tokens:
        if generated[-1] == engine.tokenizer.eos_token_id:
            break
        cur_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device=engine.device)
        pos_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=engine.device)
        with torch.no_grad():
            out = engine.model(
                input_ids=cur_ids,
                past_key_values=engine._as_cache(past_kv),
                position_ids=pos_ids,
                use_cache=True,
            )
        past_kv = engine._normalize_past_kv(out.past_key_values)
        next_token = engine._sample(out.logits[:, -1, :], 1.0, False)
        generated.append(next_token)
        cur_pos += 1
    timings["decode_ms"] = (time.perf_counter() - t0) * 1000
    timings["decode_tokens"] = len(generated) - (1 if live_ids else 0)
    timings["decode_tokens_per_sec"] = (
        timings["decode_tokens"] / max(timings["decode_ms"] / 1000, 1e-6)
    )

    # Total
    timings["total_ms"] = (
        timings["cache_lookup_ms"]
        + timings["selective_recompute_ms"]
        + timings["cpu_to_gpu_transfer_ms"]
        + timings["live_prefill_ms"]
        + timings["decode_ms"]
    )
    timings["ttft_ms"] = (
        timings["cache_lookup_ms"]
        + timings["selective_recompute_ms"]
        + timings["cpu_to_gpu_transfer_ms"]
        + timings["live_prefill_ms"]
    )

    return timings


def breakdown_baseline(engine, prompt: str, max_new_tokens: int = 48) -> Dict:
    """Baseline with timing breakdown (just prefill + decode)."""
    timings = {}
    token_ids = engine._encode(prompt)
    timings["prompt_tokens"] = len(token_ids)
    timings["cached_tokens"] = 0
    timings["live_tokens"] = len(token_ids)
    timings["cache_hit_ratio"] = 0.0
    timings["num_seams"] = 0
    timings["cache_lookup_ms"] = 0.0
    timings["selective_recompute_ms"] = 0.0
    timings["cpu_to_gpu_transfer_ms"] = 0.0

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=engine.device)

    # Prefill
    t0 = time.perf_counter()
    generated = []
    with torch.no_grad():
        out = engine.model(input_ids=input_ids, past_key_values=None, use_cache=True)
    past = engine._normalize_past_kv(out.past_key_values)
    next_token = engine._sample(out.logits[:, -1, :], 1.0, False)
    generated.append(next_token)
    timings["live_prefill_ms"] = (time.perf_counter() - t0) * 1000

    # Decode
    t0 = time.perf_counter()
    while len(generated) < max_new_tokens:
        if generated[-1] == engine.tokenizer.eos_token_id:
            break
        cur_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device=engine.device)
        with torch.no_grad():
            out = engine.model(input_ids=cur_ids, past_key_values=engine._as_cache(past), use_cache=True)
        past = engine._normalize_past_kv(out.past_key_values)
        next_token = engine._sample(out.logits[:, -1, :], 1.0, False)
        generated.append(next_token)
    timings["decode_ms"] = (time.perf_counter() - t0) * 1000
    timings["decode_tokens"] = len(generated) - 1
    timings["decode_tokens_per_sec"] = (
        timings["decode_tokens"] / max(timings["decode_ms"] / 1000, 1e-6)
    )
    timings["total_ms"] = timings["live_prefill_ms"] + timings["decode_ms"]
    timings["ttft_ms"] = timings["live_prefill_ms"]

    return timings


def run_breakdown(engine, prefix: str, queries: List[str],
                  runs: int, max_new_tokens: int) -> Dict:
    """Run breakdown for both baseline and chunk_kv modes."""
    engine.warm_chunks(prefix, position_offset=0)

    results = {"baseline": [], "chunk_kv_reuse": []}

    for run_i in range(runs):
        for query in queries:
            prompt = prefix + "\n\n" + query

            # Baseline
            b = breakdown_baseline(engine, prompt, max_new_tokens)
            if run_i > 0 or runs == 1:
                results["baseline"].append(b)

            # Chunk KV
            c = breakdown_generate(engine, prompt, max_new_tokens)
            if run_i > 0 or runs == 1:
                results["chunk_kv_reuse"].append(c)

    # Aggregate
    summary = {}
    for mode, runs_data in results.items():
        stage_keys = [
            "cache_lookup_ms", "selective_recompute_ms",
            "cpu_to_gpu_transfer_ms", "live_prefill_ms",
            "decode_ms", "total_ms", "ttft_ms",
            "decode_tokens_per_sec",
        ]
        summary[mode] = {}
        for key in stage_keys:
            values = [r[key] for r in runs_data if key in r]
            summary[mode][key] = compute_stats(values)
        summary[mode]["n_samples"] = len(runs_data)
        if runs_data:
            summary[mode]["avg_cached_tokens"] = round(
                sum(r["cached_tokens"] for r in runs_data) / len(runs_data), 1
            )
            summary[mode]["avg_live_tokens"] = round(
                sum(r["live_tokens"] for r in runs_data) / len(runs_data), 1
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Latency breakdown")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=str, default="02_latency_breakdown.json")
    args = parser.parse_args()

    engine = load_engine(model_name=args.model, chunk_size=args.chunk_size)

    all_results = {}
    workloads = {
        "system_prompt_reuse": (SYSTEM_PROMPT, QUERIES),
        "rag_doc_reuse": (RAG_DOCUMENT, RAG_QUERIES),
    }

    for wname, (prefix, queries) in workloads.items():
        log.info("Workload: %s", wname)
        all_results[wname] = run_breakdown(
            engine, prefix, queries, args.runs, args.max_new_tokens,
        )

    # Print summary
    print("\n" + "=" * 90)
    print("LATENCY BREAKDOWN (mean ms)")
    print("=" * 90)
    for wname, modes in all_results.items():
        print(f"\n--- {wname} ---")
        print(f"{'Mode':<18} {'Lookup':<10} {'Recomp':<10} {'Transfer':<10} "
              f"{'Prefill':<10} {'Decode':<10} {'Total':<10} {'TTFT':<10}")
        print("-" * 90)
        for mode in ["baseline", "chunk_kv_reuse"]:
            m = modes[mode]
            print(
                f"{mode:<18} "
                f"{m['cache_lookup_ms'].get('mean', 0):<10.2f} "
                f"{m['selective_recompute_ms'].get('mean', 0):<10.2f} "
                f"{m['cpu_to_gpu_transfer_ms'].get('mean', 0):<10.2f} "
                f"{m['live_prefill_ms'].get('mean', 0):<10.2f} "
                f"{m['decode_ms'].get('mean', 0):<10.2f} "
                f"{m['total_ms'].get('mean', 0):<10.2f} "
                f"{m['ttft_ms'].get('mean', 0):<10.2f}"
            )

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
