"""Parity-baseline demo: Qwen2.5-3B-AWQ in full_resident mode.

Used as the throughput / correctness reference that the streaming
``demo_partial_8b`` is compared against. Same I/O, same KV cache; only the
residency policy differs.

Run::

    python -m kvboost.streaming.demo_full_resident_3b
"""

from __future__ import annotations

import argparse
import time

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    parser.add_argument("--prompt", default="Explain entropy in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA required. Aborting.")
        return 2

    from transformers import AutoTokenizer

    from kvboost.streaming import StreamingCausalLM, StreamingConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = StreamingCausalLM.from_pretrained(
        args.model,
        streaming_config=StreamingConfig(residency_mode="full_resident"),
        dtype=torch.float16,
    )
    model.hf_model.cuda()
    load_s = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    print(f"loaded in {load_s:.1f}s, peak_vram={peak / 1e9:.2f} GB")

    inputs = tok(args.prompt, return_tensors="pt").to("cuda")

    t1 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_s = time.perf_counter() - t1
    new_tokens = out_ids.shape[1] - inputs["input_ids"].shape[1]
    print(f"decode: {gen_s:.2f}s, {new_tokens / gen_s:.1f} tok/s")
    print(tok.decode(out_ids[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
