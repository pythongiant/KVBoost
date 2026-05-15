"""Manual VRAM-savings demo.

Loads an AWQ model that does NOT fit fully into the local GPU and runs
greedy decode using :class:`StreamingCausalLM` with partial residency.
Prints peak VRAM, tok/s, and the generated text.

Run::

    python -m kvboost.streaming.demo_partial_8b \\
        --model casperhansen/llama-3-8b-instruct-awq \\
        --prompt "Explain entropy in two sentences."

The defaults target the 4 GB GPU class described in the plan. With
``keep_first_k=2 keep_last_k=2`` only 4 layers stay resident; the rest
stream from pinned host RAM. Expect ~7–10 tok/s on PCIe 4.0 x16 — this is
a VRAM-savings tool, not a throughput tool.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KVBoost streaming demo (8B).")
    parser.add_argument("--model", default="casperhansen/llama-3-8b-instruct-awq")
    parser.add_argument("--prompt", default="Explain entropy in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--keep-first-k", type=int, default=2)
    parser.add_argument("--keep-last-k", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("partial_resident", "ffn_only_stream", "full_stream", "full_resident"),
        default="partial_resident",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if not torch.cuda.is_available():
        print("CUDA required for the streaming demo. Aborting.")
        return 2

    from transformers import AutoTokenizer

    from kvboost.streaming import StreamingCausalLM, StreamingConfig

    cfg = StreamingConfig(
        residency_mode=args.mode,
        keep_first_k=args.keep_first_k,
        keep_last_k=args.keep_last_k,
    )

    print(f"Loading {args.model} ({cfg.summary()}) …")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = StreamingCausalLM.from_pretrained(
        args.model,
        streaming_config=cfg,
        dtype=torch.float16,
    )
    if args.mode == "full_resident":
        model.hf_model.cuda()

    load_s = time.perf_counter() - t0
    peak_after_load = torch.cuda.max_memory_allocated()
    print(f"  load_time: {load_s:.1f}s")
    print(f"  peak_vram_after_load: {peak_after_load / 1e9:.2f} GB")

    inputs = tok(args.prompt, return_tensors="pt").to("cuda")

    # Warm-up forward (primes the staging pipeline; first decode token is
    # slower).
    with torch.inference_mode():
        _ = model(**inputs)

    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_s = time.perf_counter() - t1
    peak_decode = torch.cuda.max_memory_allocated()

    new_tokens = out_ids.shape[1] - inputs["input_ids"].shape[1]
    tps = new_tokens / gen_s if gen_s > 0 else 0.0

    print(f"  decode_time: {gen_s:.2f}s  ({new_tokens} new tokens, {tps:.1f} tok/s)")
    print(f"  peak_vram_during_decode: {peak_decode / 1e9:.2f} GB")
    print()
    print(tok.decode(out_ids[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
