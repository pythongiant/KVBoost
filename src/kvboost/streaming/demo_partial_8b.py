"""Manual VRAM-savings demo.

Loads an AWQ model that does NOT fit fully into the local GPU and runs
greedy decode using :class:`StreamingCausalLM` with partial residency.
Streams output token-by-token, prints peak VRAM, per-token Δt, running
tok/s, and the generated text.

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
import sys
import time
from typing import Optional

import torch


def _greedy_decode_streaming(
    model,
    tok,
    inputs,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    *,
    quiet_per_token: bool = False,
) -> tuple[torch.Tensor, float, int, list[float]]:
    """Manual greedy decode loop with per-token streaming output and timing.

    Returns ``(out_ids, total_decode_seconds, new_token_count, per_token_dt)``.

    Uses :class:`DynamicCache` so subsequent tokens only feed the latest id —
    the same KV-reuse contract HF's ``generate`` relies on, stripped down so
    we can intercept between tokens for timing + printing.
    """
    from transformers import DynamicCache

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    device = input_ids.device

    past = DynamicCache()
    cur_ids = input_ids
    cur_mask = attention_mask
    full_ids = input_ids.clone()

    t_start = time.perf_counter()
    last_t = t_start
    per_token_dt: list[float] = []

    print()
    print("--- generation ---", file=sys.stderr, flush=True)

    if not quiet_per_token:
        # Bare-stream the prompt prefix so the live output reads like a
        # continuation (no leading double-blank).
        sys.stdout.write("")
        sys.stdout.flush()

    with torch.inference_mode():
        for step in range(max_new_tokens):
            out = model(
                input_ids=cur_ids,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = int(next_token.item())

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            per_token_dt.append(dt)
            running_tps = (step + 1) / (now - t_start)

            piece = tok.decode([token_id], skip_special_tokens=True)

            if quiet_per_token:
                # Compact one-line-per-token form, easy to grep.
                preview = piece.replace("\n", "\\n").replace("\r", "\\r")
                print(
                    f"[{step:3d}] tok={token_id:>6d} '{preview}' "
                    f"Δ={dt*1000:7.1f}ms tps={running_tps:5.2f}",
                    flush=True,
                )
            else:
                # Live-stream the piece to stdout; print the stats line to
                # stderr so the two streams stay separable when redirected.
                sys.stdout.write(piece)
                sys.stdout.flush()
                if (step + 1) % 8 == 0 or step == 0:
                    print(
                        f"\n  [{step+1:3d}/{max_new_tokens}] "
                        f"Δ_last={dt*1000:6.0f}ms  running={running_tps:5.2f} tok/s",
                        file=sys.stderr,
                        flush=True,
                    )

            full_ids = torch.cat([full_ids, next_token], dim=1)

            if eos_token_id is not None and token_id == eos_token_id:
                break

            cur_ids = next_token
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((cur_mask.shape[0], 1), dtype=cur_mask.dtype, device=device)],
                    dim=1,
                )

    print()
    total_s = time.perf_counter() - t_start
    return full_ids, total_s, len(per_token_dt), per_token_dt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KVBoost streaming demo (8B).")
    parser.add_argument("--model", default="casperhansen/llama-3-8b-instruct-awq")
    parser.add_argument("--prompt", default="Explain entropy in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--keep-first-k", type=int, default=2)
    parser.add_argument("--keep-last-k", type=int, default=2)
    parser.add_argument(
        "--n-staging-slots",
        type=int,
        default=0,
        help="Number of staging slots; 0 = auto (clamped by auto_slots_max).",
    )
    parser.add_argument(
        "--mode",
        choices=("partial_resident", "ffn_only_stream", "full_stream", "full_resident"),
        default="partial_resident",
    )
    parser.add_argument(
        "--quiet-stream",
        action="store_true",
        help="Print one line per token (id + Δt + running tok/s) instead of "
             "live-streaming text. Useful for piping to logs.",
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
        n_staging_slots=args.n_staging_slots,
    )

    print(f"Loading {args.model} ({cfg.summary()}) …", file=sys.stderr)
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
    print(f"  load_time: {load_s:.1f}s", file=sys.stderr)
    print(f"  peak_vram_after_load: {peak_after_load / 1e9:.2f} GB", file=sys.stderr)

    inputs = tok(args.prompt, return_tensors="pt").to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]
    print(f"  prompt_tokens: {prompt_tokens}", file=sys.stderr)

    # Warm-up prefill (primes the staging pipeline; first decode token is
    # otherwise much slower). Time it separately so the user can see TTFT
    # cost vs steady-state decode cost.
    print("--- warm-up prefill ---", file=sys.stderr)
    t_warm = time.perf_counter()
    with torch.inference_mode():
        _ = model(**inputs)
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t_warm
    print(f"  prefill_time: {warm_s:.2f}s", file=sys.stderr)

    torch.cuda.reset_peak_memory_stats()

    eos = tok.eos_token_id
    out_ids, decode_s, new_tokens, per_token_dt = _greedy_decode_streaming(
        model,
        tok,
        inputs,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos,
        quiet_per_token=args.quiet_stream,
    )
    peak_decode = torch.cuda.max_memory_allocated()

    # Summary stats
    tps = new_tokens / decode_s if decode_s > 0 else 0.0
    if per_token_dt:
        # Drop the first token (TTFT) from the steady-state stats — its
        # cost is dominated by the prefill, not per-token streaming.
        steady = per_token_dt[1:] if len(per_token_dt) > 1 else per_token_dt
        steady_mean_ms = 1000 * sum(steady) / len(steady)
        steady_tps = 1.0 / (sum(steady) / len(steady))
        first_token_ms = per_token_dt[0] * 1000
    else:
        steady_mean_ms = 0.0
        steady_tps = 0.0
        first_token_ms = 0.0

    print()
    print("--- summary ---", file=sys.stderr)
    print(f"  new_tokens:              {new_tokens}", file=sys.stderr)
    print(f"  total_decode_time:       {decode_s:.2f}s", file=sys.stderr)
    print(f"  avg_tok_per_s:           {tps:.2f}", file=sys.stderr)
    print(f"  first_token_latency:     {first_token_ms:.0f}ms", file=sys.stderr)
    print(f"  steady_state_ms_per_tok: {steady_mean_ms:.0f}ms", file=sys.stderr)
    print(f"  steady_state_tok_per_s:  {steady_tps:.2f}", file=sys.stderr)
    print(f"  peak_vram_during_decode: {peak_decode / 1e9:.2f} GB", file=sys.stderr)

    # In quiet mode the live-stream above only emits timing lines, not the
    # actual text. Print the decoded continuation so the user has a readable
    # answer to inspect.
    if args.quiet_stream:
        new_text = tok.decode(
            out_ids[0][prompt_tokens:], skip_special_tokens=True
        )
        print()
        print("--- output ---")
        print(new_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
