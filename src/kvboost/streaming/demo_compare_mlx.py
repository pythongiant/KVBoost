"""Side-by-side comparison: KVBoost (MPS, pure-torch AWQ) vs mlx-lm.

Runs the same prompt through both stacks and reports load time, peak
RSS, and decode tok/s. Not a controlled experiment — the two paths use
*different* quantization formats and *different* runtimes. This is a
ballpark sanity check, not a benchmark.

Run::

    pip install mlx-lm
    python -m kvboost.streaming.demo_compare_mlx
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RunResult:
    backend: str
    model_id: str
    load_s: float
    decode_s: float
    new_tokens: int
    tok_per_s: float
    peak_rss_gb: float
    text: str


def _rss_gb() -> float:
    # macOS resource returns bytes; Linux returns KB. Detect via sys.platform.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / 1e9
    return r * 1024 / 1e9


def _run_kvboost(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool,
) -> RunResult:
    from transformers import AutoTokenizer

    from kvboost.streaming import StreamingCausalLM, StreamingConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.perf_counter()
    model = StreamingCausalLM.from_pretrained(
        model_id,
        streaming_config=StreamingConfig(),
        dtype=torch.float16,
    )
    load_s = time.perf_counter() - t0

    if use_chat_template:
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt
    else:
        text = prompt

    inputs = tok(text, return_tensors="pt").to("mps")
    in_len = inputs["input_ids"].shape[1]

    t1 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    decode_s = time.perf_counter() - t1
    new_tokens = out_ids.shape[1] - in_len
    output_text = tok.decode(out_ids[0][in_len:], skip_special_tokens=True)

    return RunResult(
        backend="kvboost-mps",
        model_id=model_id,
        load_s=load_s,
        decode_s=decode_s,
        new_tokens=new_tokens,
        tok_per_s=new_tokens / decode_s if decode_s > 0 else 0.0,
        peak_rss_gb=_rss_gb(),
        text=output_text,
    )


def _run_mlx(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool,
) -> RunResult:
    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load
    except ImportError as exc:
        raise SystemExit(
            "mlx-lm not installed. Run `pip install mlx-lm` and try again."
        ) from exc

    t0 = time.perf_counter()
    mlx_model, tok = mlx_load(model_id)
    load_s = time.perf_counter() - t0

    if use_chat_template:
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt
    else:
        text = prompt

    in_ids = tok.encode(text)
    in_len = len(in_ids)

    t1 = time.perf_counter()
    output_text = mlx_generate(
        mlx_model,
        tok,
        prompt=text,
        max_tokens=max_new_tokens,
        verbose=False,
    )
    decode_s = time.perf_counter() - t1

    # mlx_lm.generate returns only the new text; count its tokens.
    out_ids = tok.encode(output_text)
    new_tokens = max(1, len(out_ids))

    return RunResult(
        backend="mlx-lm",
        model_id=model_id,
        load_s=load_s,
        decode_s=decode_s,
        new_tokens=new_tokens,
        tok_per_s=new_tokens / decode_s if decode_s > 0 else 0.0,
        peak_rss_gb=_rss_gb(),
        text=output_text,
    )


def _print_result(r: RunResult) -> None:
    print(f"=== {r.backend} ===")
    print(f"  model:       {r.model_id}")
    print(f"  load:        {r.load_s:6.2f}s")
    print(f"  decode:      {r.decode_s:6.2f}s for {r.new_tokens} new tokens")
    print(f"  tok/s:       {r.tok_per_s:6.2f}")
    print(f"  peak RSS:    {r.peak_rss_gb:6.2f} GB")
    print(f"  output:      {r.text[:200]!r}{'…' if len(r.text) > 200 else ''}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kvboost-model",
        default="casperhansen/llama-3.2-1b-instruct-awq",
    )
    parser.add_argument(
        "--mlx-model",
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
    )
    parser.add_argument(
        "--prompt", default="Explain entropy in two sentences."
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat-template formatting (use raw prompt).",
    )
    parser.add_argument(
        "--only",
        choices=("kvboost", "mlx", "both"),
        default="both",
    )
    args = parser.parse_args(argv)

    use_chat = not args.no_chat_template

    results: list[RunResult] = []

    if args.only in ("kvboost", "both"):
        print("--- running kvboost (MPS, pure-torch AWQ) ---", file=sys.stderr)
        r = _run_kvboost(args.kvboost_model, args.prompt, args.max_new_tokens, use_chat)
        results.append(r)
        # Drop the model so RSS measurement for MLX run isn't tainted.
        gc.collect()

    if args.only in ("mlx", "both"):
        print("--- running mlx-lm (Metal, MLX 4-bit) ---", file=sys.stderr)
        r = _run_mlx(args.mlx_model, args.prompt, args.max_new_tokens, use_chat)
        results.append(r)

    print()
    for r in results:
        _print_result(r)

    if len(results) == 2:
        a, b = results
        if a.tok_per_s > 0 and b.tok_per_s > 0:
            ratio = b.tok_per_s / a.tok_per_s
            faster = b.backend if ratio > 1 else a.backend
            print(
                f"speed ratio: {b.backend} is "
                f"{ratio:.2f}× {'faster' if ratio > 1 else 'slower'} than {a.backend}"
            )
            print(
                f"  ({faster} wins on tok/s — different quant format + kernel "
                f"path, not an apples-to-apples runtime comparison)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
