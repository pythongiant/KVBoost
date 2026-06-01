"""Isolate the decode-step cost vs context length — no server, no KVBoost.

Answers one question: is the 24→2.6 tok/s decode cliff the *raw model
forward* on this GPU, or is it KVBoost / streaming / sampling harness
overhead?

Times a single-token decode at increasing KV-context lengths, straight
through the HF model with a plain DynamicCache. If the per-step time
balloons here too, the cliff is the GPU/model (a slow or contended card,
or a non-flash decode kernel). If it stays flat here but the server is
slow, the cliff is in the serving path, not attention.

Run on the GPU box:
    python tests/integration/decode_step_bench.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --contexts 80 1040 9360 23400
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[80, 1040, 9360, 23400])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — run this on the GPU box.")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    dev = torch.device("cuda")
    print(f"Loading {args.model} ({args.dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
    ).to(dev).eval()

    vocab = int(model.config.vocab_size)
    impl = getattr(model.config, "_attn_implementation", "?")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  {props.total_memory / 1024**3:.1f} GiB  "
          f"SM {props.major}.{props.minor}")
    print(f"attn_implementation: {impl}")
    print(f"{'ctx':>8} {'ms/step':>10} {'tok/s':>9} {'peak_MiB':>10}")
    print("-" * 40)

    for ctx in args.contexts:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        ids = torch.randint(0, vocab, (1, ctx), device=dev)
        with torch.no_grad():
            out = model(ids, use_cache=True)
            past = out.past_key_values
            nxt = torch.tensor([[ctx % vocab]], device=dev)
            for _ in range(args.warmup):
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(args.steps):
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
            torch.cuda.synchronize(dev)
            ms = (time.perf_counter() - t0) / args.steps * 1000.0
        peak = torch.cuda.max_memory_allocated(dev) / 1024**2
        print(f"{ctx:>8} {ms:>10.1f} {1000.0 / ms:>9.1f} {peak:>10.0f}")

    print()
    print("Read: if ms/step climbs steeply with ctx HERE, the cliff is the")
    print("GPU/model itself (slow/contended card or non-flash decode). If it")
    print("stays ~flat but the server is slow, the cliff is in the serving")
    print("path (streaming, sampling, or KVBoost's decode loop).")


if __name__ == "__main__":
    main()
