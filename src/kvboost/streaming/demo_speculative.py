"""Speculative-decode demo over the AWQ streaming pipeline.

Mirrors :mod:`demo_partial_8b` — same flag conventions, same per-token
streaming output, same peak-VRAM accounting — but runs the decode loop
through :class:`kvboost.speculative.SpeculativeEngine` so a small draft
model can amortize the cost of each target-model forward.

Run::

    python -m kvboost.streaming.demo_speculative \\
        --model        Qwen/Qwen2.5-32B-Instruct-AWQ \\
        --draft-model  Qwen/Qwen2.5-1.5B-Instruct-AWQ \\
        --keep-first-k 11 --keep-last-k 11 --n-staging-slots 8 \\
        --gamma 5 --mode greedy \\
        --prompt "Explain entropy in two sentences." \\
        --max-new-tokens 60

Pairs the existing partial-residency streaming target with a fully-
resident small draft. After generation, prints the acceptance histogram
and average-committed-per-round so you can tune ``--gamma`` against
real workload acceptance.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import torch


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="KVBoost speculative-decoding demo over AWQ streaming."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ",
                        help="Target model (large, AWQ-quantized).")
    parser.add_argument("--draft-model", required=True,
                        help="Draft model id. Must share tokenizer with the "
                             "target (e.g. Qwen2.5-1.5B for Qwen2.5-32B).")
    parser.add_argument("--prompt", default="Explain entropy in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--keep-first-k", type=int, default=4)
    parser.add_argument("--keep-last-k", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("partial_resident", "ffn_only_stream", "full_stream", "full_resident"),
        default="partial_resident",
        help="Target-model residency mode (same as demo_partial_8b).",
    )
    parser.add_argument(
        "--n-staging-slots", type=int, default=0,
        help="0 = auto-size at load time.",
    )
    parser.add_argument(
        "--gamma", type=int, default=5,
        help="Tokens drafted per verification round (default: 5).",
    )
    parser.add_argument(
        "--spec-mode", default="greedy", choices=("greedy", "sampling"),
        help="Sampler mode. 'greedy' matches non-spec greedy bit-for-bit.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if not torch.cuda.is_available():
        print("CUDA required for the streaming demo. Aborting.", file=sys.stderr)
        return 2

    from kvboost import KVBoost
    from kvboost.speculative import SpeculativeConfig
    from kvboost.streaming import StreamingConfig

    streaming_cfg = StreamingConfig(
        residency_mode=args.mode,
        keep_first_k=args.keep_first_k,
        keep_last_k=args.keep_last_k,
        n_staging_slots=args.n_staging_slots,
    )
    spec_cfg = SpeculativeConfig(
        draft_model_id=args.draft_model,
        draft_k=args.gamma,
        mode=args.spec_mode,
        temperature=args.temperature,
    )

    print(
        f"Loading target={args.model} ({streaming_cfg.summary()}) ; "
        f"{spec_cfg.summary()} …",
        file=sys.stderr,
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    engine = KVBoost.from_pretrained(
        args.model,
        streaming_config=streaming_cfg,
        speculative_config=spec_cfg,
        max_cache_bytes=int(2e9),
    )

    # InferenceEngine skips its own .to(device) for quantized models because
    # weight movers in bnb/accelerate can break. In streaming full_resident
    # mode there are no per-layer hooks, so nothing else moves the model.
    # Mirror demo_partial_8b.py and do it explicitly.
    if args.mode == "full_resident":
        engine.model.hf_model.to("cuda")

    load_s = time.perf_counter() - t0
    peak_after_load = torch.cuda.max_memory_allocated()
    print(f"  load_time: {load_s:.1f}s", file=sys.stderr)
    print(f"  peak_vram_after_load: {peak_after_load / 1e9:.2f} GB", file=sys.stderr)

    print("--- generation ---", file=sys.stderr)
    t1 = time.perf_counter()
    result = engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=(args.spec_mode == "sampling"),
    )
    decode_s = time.perf_counter() - t1
    peak_decode = torch.cuda.max_memory_allocated()

    # Summary
    new_tokens = result.generated_tokens
    tps = new_tokens / decode_s if decode_s > 0 else 0.0

    print()
    print("--- output ---", file=sys.stderr)
    print(result.output_text)
    print()

    print("--- summary ---", file=sys.stderr)
    print(f"  new_tokens:              {new_tokens}", file=sys.stderr)
    print(f"  total_decode_time:       {decode_s:.2f}s", file=sys.stderr)
    print(f"  avg_tok_per_s:           {tps:.2f}", file=sys.stderr)
    print(f"  peak_vram_during_decode: {peak_decode / 1e9:.2f} GB", file=sys.stderr)

    spec_stats = engine.speculative_stats()
    if spec_stats:
        print()
        print("--- speculative stats ---", file=sys.stderr)
        print(f"  rounds:                  {spec_stats['rounds']}", file=sys.stderr)
        print(f"  accepted_total:          {spec_stats['accepted_total']}", file=sys.stderr)
        print(f"  committed_total:         {spec_stats['committed_total']}", file=sys.stderr)
        print(f"  bonus_rounds:            {spec_stats['bonus_rounds']}", file=sys.stderr)
        print(f"  acceptance_rate:         {spec_stats['acceptance_rate']:.3f}", file=sys.stderr)
        print(f"  avg_committed/round:     {spec_stats['avg_committed_per_round']:.2f}", file=sys.stderr)
        print(f"  histogram (K=0..{len(spec_stats['histogram'])-1}): "
              f"{spec_stats['histogram']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
