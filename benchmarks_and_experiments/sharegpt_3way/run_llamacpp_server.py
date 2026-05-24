#!/usr/bin/env python3
"""
3-way ShareGPT benchmark — llama.cpp **server-mode** runner (parallel).

Talks to a running ``llama-server`` (the llama.cpp HTTP daemon) over its
OpenAI-compatible ``/v1/completions`` endpoint and fires N conversations
concurrently. ``llama-server`` uses internal "slots" for parallelism; if the
slots are full it returns 503 (when configured) or just queues — this client
handles either with backoff and retry.

Telemetry
---------
TTFT comes from streaming SSE (time to first content chunk). llama-server
does not expose per-request KV-reuse counters in its OpenAI-compat usage
field, so ``cached_tokens`` will typically be 0. The ITL and decode-tok/s
numbers are the meaningful comparison signal here.

Quick start
-----------
    # shell 1: start the server (assumes you've built llama.cpp with CUDA)
    ./llama-server \\
        -m ~/models/qwen2.5-7b-instruct-q4_k_m.gguf \\
        --model-draft ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \\
        -ngl 99 \\
        --ctx-size 4096 \\
        --parallel 8 \\
        --port 8002

    # shell 2: parallel replay
    python run_llamacpp_server.py --server-url http://localhost:8002 \\
        --concurrency 8 --n-samples 500

The ``--parallel N`` flag on llama-server creates N slots; matching
``--concurrency`` to ``--parallel`` is the right starting point. Beyond that,
the server queues / rejects, and the client backs off.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import httpx

import _server_common as srv
from _common import (
    ConvResult, TurnResult, add_common_args, capture_run_metadata,
    checkpoint_key, compute_metrics, is_run_complete, load_sharegpt,
    log, print_summary, setup_logging,
)

RESULTS_DIR    = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"


# ── Per-conversation worker ─────────────────────────────────────────────────

async def _process_conversation_llamacpp(
    *,
    conv: dict,
    client: httpx.AsyncClient,
    completions_url: str,
    model_name: str,
    max_new_tokens: int,
    count_tokens,
    save_output_text: bool,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> ConvResult:
    conv_id = conv["id"]
    turns = conv["turns"]
    n_human = sum(1 for t in turns if t.get("from") == "human")

    conv_start = time.perf_counter()
    conv_result = ConvResult(
        conv_id=conv_id,
        n_turns=n_human,
        start_iso=datetime.now(timezone.utc).isoformat(),
    )

    history = ""
    human_turn_idx = 0
    for turn in turns:
        if turn.get("from") != "human":
            continue

        prompt = history + f"Human: {turn['value']}\nAssistant:"
        history_tokens = count_tokens(prompt)
        turn_start_iso = datetime.now(timezone.utc).isoformat()

        body = {
            "model":          model_name,
            "prompt":         prompt,
            "max_tokens":     max_new_tokens,
            "temperature":    0.0,
            "cache_prompt":   True,            # llama-server: reuse KV across requests
            "stream_options": {"include_usage": True},
        }

        error_msg: Optional[str] = None
        output_text = ""
        ttft_ms = 0.0
        total_ms = 0.0
        usage: Optional[dict] = None

        try:
            output_text, ttft_ms, total_ms, _hdrs, usage = await srv.post_stream_with_retry(
                client, completions_url, body,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            log.error("Conv %s turn %d failed: %s", conv_id, human_turn_idx, error_msg)

        if error_msg is not None:
            tr = TurnResult(
                conv_id=conv_id, turn_idx=human_turn_idx, n_turns_total=n_human,
                history_tokens=history_tokens, prompt_tokens=history_tokens,
                cached_tokens=0, cache_hit_ratio=0.0,
                ttft_ms=0.0, total_ms=0.0, decode_ms=0.0,
                output_tokens=0, itl_ms=0.0, decode_tps=0.0,
                error=error_msg, stop_reason="error",
                turn_start_iso=turn_start_iso,
            )
            conv_result.turns.append(tr)
            conv_result.error_count += 1
            break

        # llama-server doesn't expose per-request KV reuse in OpenAI usage.
        # If a future build adds it under prompt_tokens_details, pick it up.
        details = (usage or {}).get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        prompt_tokens = int((usage or {}).get("prompt_tokens") or history_tokens)
        out_tok       = int((usage or {}).get("completion_tokens") or 0)

        decode_ms = max(total_ms - ttft_ms, 0.0)
        itl_ms = decode_ms / max(out_tok - 1, 1)
        decode_tps = (out_tok / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        preview = output_text[:200] + "…" if len(output_text) > 200 else output_text
        stop_reason = "max_tokens" if out_tok >= max_new_tokens else "eos"

        tr = TurnResult(
            conv_id=conv_id,
            turn_idx=human_turn_idx,
            n_turns_total=n_human,
            history_tokens=history_tokens,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached,
            cache_hit_ratio=cached / max(prompt_tokens, 1),
            ttft_ms=float(ttft_ms),
            total_ms=float(total_ms),
            decode_ms=decode_ms,
            output_tokens=out_tok,
            itl_ms=itl_ms,
            decode_tps=decode_tps,
            stop_reason=stop_reason,
            output_text_preview=preview,
            output_text=output_text if save_output_text else None,
            turn_start_iso=turn_start_iso,
            backend_telemetry={
                "usage": usage,
            },
        )
        conv_result.turns.append(tr)
        history = prompt + output_text + "\n"
        human_turn_idx += 1

    conv_result.end_iso = datetime.now(timezone.utc).isoformat()
    conv_result.wall_s = time.perf_counter() - conv_start
    return conv_result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="llama.cpp server-mode 3-way ShareGPT runner (parallel)",
    )
    add_common_args(parser)
    parser.add_argument("--server-url", default="http://localhost:8002",
                        help="llama-server base URL (default: http://localhost:8002)")
    parser.add_argument("--model", default="default",
                        help="Model id sent in request (llama-server doesn't validate "
                             "strictly; default 'default' usually works).")
    parser.add_argument("--tokenizer", required=True,
                        help="HF tokenizer id for prompt token counting / dataset "
                             "filtering. llama-server uses its own tokenizer for "
                             "the model but this client needs HF for the filter "
                             "pipeline. Use the model's HF id, e.g. "
                             "Qwen/Qwen2.5-7B-Instruct.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--base-delay", type=float, default=0.5)
    parser.add_argument("--max-delay", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    args = parser.parse_args()

    setup_logging(args.verbose, args.debug)
    print(f"\n{'=' * 72}\n  llama.cpp server-mode (parallel) — ShareGPT 3-way\n{'=' * 72}")
    print(f"  server      = {args.server_url}")
    print(f"  concurrency = {args.concurrency}")
    print(f"  tokenizer   = {args.tokenizer}")

    # llama-server exposes /v1/models — use it to fail fast.
    try:
        srv.check_openai_server(args.server_url)
    except SystemExit:
        raise
    print(f"  n_samples   = {args.n_samples}  turns={args.min_turns}-{args.max_turns}")
    print(f"{'=' * 72}\n")

    out_path = Path(args.output) if args.output else RESULTS_DIR / "llamacpp_server.json"
    if not args.no_checkpoint and is_run_complete(out_path, args.n_samples):
        print(f"[skip] {out_path} already covers {args.n_samples} conversations.")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    conversations = load_sharegpt(
        n_conversations=args.n_samples,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        tokenizer=tokenizer,
        max_context_tokens=args.max_context_tokens,
    )
    if not conversations:
        sys.exit("No conversations after filtering.")

    ck_path = CHECKPOINT_DIR / (
        f"llamacpp_server_{checkpoint_key('llamacpp_server', args.tokenizer, args.n_samples, args.max_turns)}.json"
    )
    meta = {
        "backend":     "llamacpp_server",
        "tokenizer":   args.tokenizer,
        "server_url":  args.server_url,
        "concurrency": args.concurrency,
    }
    config = {
        "concurrency":         args.concurrency,
        "max_new_tokens":      args.max_new_tokens,
        "n_samples":           args.n_samples,
        "min_turns":           args.min_turns,
        "max_turns":           args.max_turns,
        "max_context_tokens":  args.max_context_tokens,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "save_output_text":    args.save_output_text,
        "max_retries":         args.max_retries,
        "base_delay":          args.base_delay,
        "max_delay":           args.max_delay,
        "request_timeout":     args.request_timeout,
        "reset_between_conversations": False,
    }
    run_metadata = capture_run_metadata("llamacpp_server", config)

    def count_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=True))

    completions_url = args.server_url.rstrip("/") + "/v1/completions"

    t0 = time.perf_counter()
    results = asyncio.run(srv.replay_parallel(
        process_conv_fn=_process_conversation_llamacpp,
        process_conv_kwargs=dict(
            completions_url=completions_url,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            count_tokens=count_tokens,
            save_output_text=args.save_output_text,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
            max_delay=args.max_delay,
        ),
        conversations=conversations,
        concurrency=args.concurrency,
        ck_path=ck_path,
        meta=meta,
        run_metadata=run_metadata,
        no_checkpoint=args.no_checkpoint,
        progress_every=args.progress_every,
        request_timeout=args.request_timeout,
        backend_label="llamacpp_server",
    ))
    wall_s = time.perf_counter() - t0
    run_metadata.end_iso = datetime.now(timezone.utc).isoformat()

    metrics = compute_metrics(results, total_wall_s=wall_s)
    print_summary("llamacpp_server", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend":      "llamacpp_server",
        "tokenizer":    args.tokenizer,
        "model":        args.model,
        "server_url":   args.server_url,
        "concurrency":  args.concurrency,
        "config":       config,
        "run_metadata": asdict(run_metadata),
        "wall_s":       wall_s,
        "metrics":      metrics,
        "results":      [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results written: {out_path}")

    if ck_path.exists() and not args.no_checkpoint:
        ck_path.unlink()
    live = ck_path.with_name(ck_path.stem + ".live.json")
    if live.exists():
        live.unlink()


if __name__ == "__main__":
    main()
