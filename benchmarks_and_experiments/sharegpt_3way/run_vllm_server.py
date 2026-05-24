#!/usr/bin/env python3
"""
3-way ShareGPT benchmark — vLLM **server-mode** runner (parallel).

Talks to a running ``vllm serve`` (OpenAI-compatible API) over HTTP and fires
N conversations concurrently. vLLM's async engine already does continuous
batching with prefix caching across requests; this client just keeps the
queue full and falls back to sequential under back-pressure.

Telemetry
---------
TTFT comes from streaming SSE (time to first content chunk — accurate).
Total wall time from stream completion. Cached-tokens from
``usage.prompt_tokens_details.cached_tokens`` if vLLM populates it; else 0.

Quick start
-----------
    # shell 1: start the server
    vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --enable-prefix-caching \\
        --gpu-memory-utilization 0.85 \\
        --max-model-len 4096 \\
        --port 8001

    # shell 2: parallel replay
    python run_vllm_server.py --server-url http://localhost:8001 \\
        --concurrency 8 --n-samples 500

Methodology caveat
------------------
Same as run_kvboost_server.py: prefix cache is shared across concurrent
clients, so per-conversation cold-start is no longer measurable. The numbers
describe multi-tenant warm steady-state. Recorded as
``reset_between_conversations=false`` in metadata.
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


# ── vLLM-specific telemetry ─────────────────────────────────────────────────

def _vllm_cached_tokens(usage: Optional[dict]) -> int:
    """vLLM populates ``usage.prompt_tokens_details.cached_tokens`` when
    prefix caching is on. Older versions may use a flat ``cached_tokens``
    field. Returns 0 if neither is present."""
    if not usage:
        return 0
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)


# ── Per-conversation worker ─────────────────────────────────────────────────

async def _process_conversation_vllm(
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

        # vLLM accepts the OpenAI text-completion schema (single "prompt").
        body = {
            "model":         model_name,
            "prompt":        prompt,
            "max_tokens":    max_new_tokens,
            "temperature":   0.0,
            "stream_options": {"include_usage": True},
        }

        error_msg: Optional[str] = None
        output_text = ""
        ttft_ms = 0.0
        total_ms = 0.0
        usage: Optional[dict] = None

        try:
            output_text, ttft_ms, total_ms, _headers, usage = await srv.post_stream_with_retry(
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

        cached = _vllm_cached_tokens(usage)
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
        description="vLLM server-mode 3-way ShareGPT runner (parallel)",
    )
    add_common_args(parser)
    parser.add_argument("--server-url", default="http://localhost:8001",
                        help="vllm serve base URL (default: http://localhost:8001)")
    parser.add_argument("--model", default=None,
                        help="Model id; default: GET /v1/models")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer id (default: --model)")
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--base-delay", type=float, default=0.5)
    parser.add_argument("--max-delay", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    args = parser.parse_args()

    setup_logging(args.verbose, args.debug)
    print(f"\n{'=' * 72}\n  vLLM server-mode (parallel) — ShareGPT 3-way\n{'=' * 72}")
    print(f"  server      = {args.server_url}")
    print(f"  concurrency = {args.concurrency}")

    models_resp = srv.check_openai_server(args.server_url)
    loaded = (models_resp.get("data") or [{}])[0].get("id", "<unknown>")
    model_name = args.model or loaded
    print(f"  model       = {model_name}  (server loaded: {loaded})")
    print(f"  n_samples   = {args.n_samples}  turns={args.min_turns}-{args.max_turns}")
    print(f"{'=' * 72}\n")

    out_path = Path(args.output) if args.output else RESULTS_DIR / "vllm_server.json"
    if not args.no_checkpoint and is_run_complete(out_path, args.n_samples):
        print(f"[skip] {out_path} already covers {args.n_samples} conversations.")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model_name)

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
        f"vllm_server_{checkpoint_key('vllm_server', model_name, args.n_samples, args.max_turns)}.json"
    )
    meta = {
        "backend":     "vllm_server",
        "model":       model_name,
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
    run_metadata = capture_run_metadata("vllm_server", config)

    def count_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=True))

    completions_url = args.server_url.rstrip("/") + "/v1/completions"

    t0 = time.perf_counter()
    results = asyncio.run(srv.replay_parallel(
        process_conv_fn=_process_conversation_vllm,
        process_conv_kwargs=dict(
            completions_url=completions_url,
            model_name=model_name,
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
        backend_label="vllm_server",
    ))
    wall_s = time.perf_counter() - t0
    run_metadata.end_iso = datetime.now(timezone.utc).isoformat()

    metrics = compute_metrics(results, total_wall_s=wall_s)
    print_summary("vllm_server", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend":      "vllm_server",
        "model":        model_name,
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
