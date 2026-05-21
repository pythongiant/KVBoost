#!/usr/bin/env python3
"""
3-way ShareGPT benchmark — KVBoost runner.

Stack under test:
  * Qwen2.5-7B-Instruct target  (HF, fp16)
  * Qwen2.5-1.5B-Instruct draft (HF, fp16)
  * CacheBlend KV reuse across conversation turns
  * Speculative decoding (gamma=5 by default)

Outputs results/kvboost.json in the schema shared with run_vllm.py and
run_llamacpp.py so compare.py can plot all three side-by-side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _common as common
from _common import (
    ConvResult, TurnResult, add_common_args, checkpoint_key,
    compute_metrics, load_sharegpt, print_summary,
    replay_conversations, setup_logging,
)

RESULTS_DIR    = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"


def build_engine(args):
    from kvboost import KVBoost
    from kvboost.speculative import SpeculativeConfig

    spec_cfg = SpeculativeConfig(
        draft_model_id=args.draft_model,
        draft_k=args.gamma,
        mode="greedy",   # bit-identical to non-spec greedy
    )
    return KVBoost.from_pretrained(
        args.model,
        # ── KV reuse + CacheBlend prefill ──
        chunk_size=args.chunk_size,
        recompute_strategy="cacheblend",
        chunk_boundary_window=16,
        overlap_k=16,
        sink_tokens=32,
        recompute_overlap=16,
        max_cache_bytes=int(args.max_cache_bytes),
        recency_window_chunks=args.recency_window_chunks,
        # ── Speculative decoding ──
        speculative_config=spec_cfg,
    )


def make_run_turn(engine, max_new_tokens: int):
    from kvboost import GenerationMode

    def _spec_snapshot() -> dict:
        s = engine.speculative_stats() or {}
        return {
            "rounds":         s.get("rounds", 0),
            "accepted_total": s.get("accepted_total", 0),
            "draft_forwards": s.get("draft_forwards", 0),
        }

    def run_turn(prompt: str) -> dict:
        engine.warm_chunks(prompt, position_offset=0)
        pre = _spec_snapshot()

        t0 = time.perf_counter()
        result = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
        )
        wall_total_ms = (time.perf_counter() - t0) * 1000.0

        post = _spec_snapshot()
        d_rounds   = post["rounds"] - pre["rounds"]
        d_accepted = post["accepted_total"] - pre["accepted_total"]
        d_proposed = post["draft_forwards"] - pre["draft_forwards"]

        return {
            "ttft_ms":        float(result.ttft_ms),
            "total_ms":       float(result.total_ms) if result.total_ms else wall_total_ms,
            "output_text":    result.output_text,
            "output_tokens":  int(result.generated_tokens),
            "prompt_tokens":  int(result.prompt_tokens),
            "cached_tokens":  int(result.cached_tokens),
            "spec_accepted":  int(d_accepted) if d_rounds > 0 else None,
            "spec_proposed":  int(d_proposed) if d_rounds > 0 else None,
            "spec_rounds":    int(d_rounds)   if d_rounds > 0 else None,
        }

    return run_turn


def main():
    parser = argparse.ArgumentParser(description="KVBoost 3-way ShareGPT runner")
    add_common_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-cache-bytes", type=float, default=3.0e9)
    parser.add_argument("--recency-window-chunks", type=int, default=16)
    args = parser.parse_args()

    setup_logging(args.verbose, args.debug)
    print(f"\n{'=' * 72}\n  KVBoost (cacheblend + spec) — ShareGPT 3-way\n{'=' * 72}")
    print(f"  target={args.model}")
    print(f"  draft ={args.draft_model}  gamma={args.gamma}")
    print(f"  n_samples={args.n_samples}  turns={args.min_turns}-{args.max_turns}")
    print(f"{'=' * 72}\n")

    engine = build_engine(args)
    conversations = load_sharegpt(
        n_conversations=args.n_samples,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        tokenizer=engine.tokenizer,
        max_context_tokens=args.max_context_tokens,
    )
    if not conversations:
        sys.exit("No conversations after filtering.")

    ck_path = CHECKPOINT_DIR / f"kvboost_{checkpoint_key('kvboost', args.model, args.n_samples, args.max_turns)}.json"
    meta = {"backend": "kvboost", "model": args.model, "draft": args.draft_model, "gamma": args.gamma}

    t0 = time.perf_counter()
    results = replay_conversations(
        run_turn=make_run_turn(engine, args.max_new_tokens),
        count_tokens=lambda s: len(engine.tokenizer.encode(s, add_special_tokens=True)),
        reset_between_convs=engine.reset_cache,
        conversations=conversations,
        ck_path=ck_path,
        meta=meta,
        no_checkpoint=args.no_checkpoint,
    )
    wall_s = time.perf_counter() - t0

    metrics = compute_metrics(results, total_wall_s=wall_s)
    print_summary("kvboost", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / "kvboost.json"
    payload = {
        "backend": "kvboost",
        "model": args.model,
        "draft_model": args.draft_model,
        "config": {
            "gamma": args.gamma,
            "recompute_strategy": "cacheblend",
            "chunk_size": args.chunk_size,
            "max_new_tokens": args.max_new_tokens,
            "n_samples": args.n_samples,
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "max_context_tokens": args.max_context_tokens,
        },
        "wall_s": wall_s,
        "metrics": metrics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results written: {out_path}")

    if ck_path.exists():
        ck_path.unlink()


if __name__ == "__main__":
    main()
