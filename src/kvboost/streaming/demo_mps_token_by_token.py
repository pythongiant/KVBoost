"""Token-by-token MPS demo for ``StreamingCausalLM``.

Loads an AWQ model on Apple Silicon (unified memory, no real streaming —
see :mod:`kvboost.streaming.model_shell` ``_from_pretrained_mps``) and
prints each generated token as it's produced. Two modes:

- ``--mode streamer``: uses HF's :class:`TextStreamer` for a one-liner
  streaming output. Tokens print as a continuous string.
- ``--mode manual``: explicit greedy decode loop with per-token timing.
  Prints token id, decoded piece, elapsed ms, running tok/s. Useful for
  measuring exactly where time is going on the pure-torch AWQ path.

Run::

    python -m kvboost.streaming.demo_mps_token_by_token
    python -m kvboost.streaming.demo_mps_token_by_token --mode manual --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import torch


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    backend = getattr(torch.backends, "mps", None)
    if backend is not None and backend.is_built() and backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _format_prompt(tok, raw_prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return raw_prompt
    try:
        messages = [{"role": "user", "content": raw_prompt}]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return raw_prompt


def _run_streamer(model, tok, inputs, max_new_tokens: int) -> None:
    from transformers import TextStreamer

    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )


def _run_manual(model, tok, inputs, max_new_tokens: int, eos_token_id: Optional[int]) -> None:
    """Explicit greedy decode loop using ``DynamicCache`` for KV reuse.

    Prints each token as ``[i] tok=<id> '<piece>' Δ=<ms>ms running=<tok/s>``.
    """
    from transformers import DynamicCache

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    device = input_ids.device

    past = DynamicCache()
    cur_ids = input_ids
    cur_mask = attention_mask

    t_start = time.perf_counter()
    last_t = t_start
    total_new = 0

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
            dt_ms = (now - last_t) * 1000
            last_t = now
            total_new += 1
            tps = total_new / (now - t_start)

            piece = tok.decode([token_id], skip_special_tokens=True)
            # Sanitize for terminal: replace any control chars in the
            # piece preview but keep it readable.
            preview = piece.replace("\n", "\\n").replace("\r", "\\r")
            print(
                f"[{step:3d}] tok={token_id:>6d} '{preview}' "
                f"Δ={dt_ms:6.1f}ms running={tps:5.2f} tok/s",
                flush=True,
            )

            if eos_token_id is not None and token_id == eos_token_id:
                print(f"  (eos at step {step})")
                break

            # Feed only the new token next round; KV cache holds the rest.
            cur_ids = next_token
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((cur_mask.shape[0], 1), dtype=cur_mask.dtype, device=device)],
                    dim=1,
                )

    total_s = time.perf_counter() - t_start
    print()
    print(f"generated {total_new} tokens in {total_s:.2f}s ({total_new/total_s:.2f} tok/s)")


def _load_model(model_id: str, cache_dense: bool) -> "object":
    """Helper for the benchmark path. Toggles the MPS cache_dense flag via
    the ``KVBOOST_MPS_CACHE_DENSE`` env var the loader reads. Returns a
    freshly-constructed wrapper each call.
    """
    import gc
    import os

    from kvboost.streaming import StreamingCausalLM, StreamingConfig

    os.environ["KVBOOST_MPS_CACHE_DENSE"] = "1" if cache_dense else "0"
    gc.collect()
    return StreamingCausalLM.from_pretrained(
        model_id,
        streaming_config=StreamingConfig(),
        dtype=torch.float16,
    )


def _measure_decode(model, tok, inputs, max_new_tokens: int) -> tuple[float, int, str]:
    """Run a greedy decode and return ``(seconds, new_tokens, output_text)``.
    Quiet — no per-token print.
    """
    in_len = inputs["input_ids"].shape[1]
    t = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    dt = time.perf_counter() - t
    new_tokens = out_ids.shape[1] - in_len
    return dt, new_tokens, tok.decode(out_ids[0][in_len:], skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="casperhansen/llama-3.2-1b-instruct-awq")
    parser.add_argument(
        "--prompt",
        default="Explain entropy in two sentences.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=("streamer", "manual", "benchmark"),
        default="streamer",
        help=(
            "streamer: HF TextStreamer (compact). "
            "manual: per-token timing. "
            "benchmark: run cache_dense ON vs OFF and report tok/s ratio."
        ),
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat-template formatting (use raw prompt).",
    )
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from kvboost.streaming import StreamingCausalLM, StreamingConfig

    device = _pick_device()
    print(f"device: {device}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.mode == "benchmark":
        prompt = _format_prompt(
            tok, args.prompt, use_chat_template=not args.no_chat_template
        )
        inputs = tok(prompt, return_tensors="pt").to(device)
        print(f"prompt_tokens: {inputs['input_ids'].shape[1]}", file=sys.stderr)
        print("--- benchmark (cache_dense ON vs OFF) ---", file=sys.stderr)

        results = []
        for label, flag in (("cache_dense=ON ", True), ("cache_dense=OFF", False)):
            print(f"loading with {label} …", file=sys.stderr)
            t0 = time.perf_counter()
            model = _load_model(args.model, cache_dense=flag)
            load_s = time.perf_counter() - t0

            decode_s, n, text = _measure_decode(
                model, tok, inputs, args.max_new_tokens
            )
            tps = n / decode_s if decode_s > 0 else 0.0
            results.append((label, load_s, decode_s, n, tps, text))
            print(
                f"  {label}: load={load_s:.1f}s  decode={decode_s:.2f}s  "
                f"tokens={n}  tok/s={tps:.2f}",
                file=sys.stderr,
            )
            del model

        print()
        for label, load_s, decode_s, n, tps, text in results:
            print(f"=== {label} ===")
            print(f"  load:   {load_s:6.2f}s")
            print(f"  decode: {decode_s:6.2f}s ({n} tokens, {tps:.2f} tok/s)")
            print(f"  output: {text[:120]!r}{'…' if len(text) > 120 else ''}")
        if len(results) == 2 and results[0][4] > 0 and results[1][4] > 0:
            on, off = results[0][4], results[1][4]
            print()
            print(
                f"speedup from cache_dense: {on / off:.2f}× "
                f"({on:.2f} → {off:.2f} tok/s)"
            )
        return 0

    print(f"loading {args.model} …", file=sys.stderr)
    t0 = time.perf_counter()
    model = StreamingCausalLM.from_pretrained(
        args.model,
        streaming_config=StreamingConfig(),
        dtype=torch.float16,
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    prompt = _format_prompt(tok, args.prompt, use_chat_template=not args.no_chat_template)
    inputs = tok(prompt, return_tensors="pt").to(device)
    print(f"prompt: {args.prompt!r}", file=sys.stderr)
    print(f"prompt_tokens: {inputs['input_ids'].shape[1]}", file=sys.stderr)
    print(f"--- generation ({args.mode}) ---", file=sys.stderr)

    if args.mode == "streamer":
        _run_streamer(model, tok, inputs, args.max_new_tokens)
    else:
        eos = tok.eos_token_id
        _run_manual(model, tok, inputs, args.max_new_tokens, eos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
