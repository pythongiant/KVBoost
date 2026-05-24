#!/usr/bin/env python3
"""
3-way ShareGPT benchmark — KVBoost **server-mode** runner.

Unlike run_kvboost.py (which loads the engine in-process and serializes one
conversation at a time), this script talks to a running `kvboost-server` over
HTTP and fires N conversations concurrently. The server's BatchQueue handles
batching and back-pressure: when --max-queue-size is exceeded the server
returns 503, and this client backs off and retries — so concurrency
gracefully drops to sequential under saturation, which is the "OOM ⇒ queue
sequentially" behaviour we want.

Quick start
-----------
    # 1) Start the server (in another shell or via tmux)
    kvboost-server \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --recompute-strategy cacheblend \\
        --max-batch-size 8 --max-queue-size 64

    # 2) Run the parallel benchmark
    python run_kvboost_server.py \\
        --server-url http://localhost:8000 \\
        --concurrency 8 \\
        --n-samples 500

Methodology caveat
------------------
The in-process variant resets the KV cache between conversations to measure
the cold→warm climb. This variant CANNOT reset (the cache is server-wide and
shared across concurrent clients) — the numbers describe multi-tenant warm
steady-state, closer to production traffic, not the single-conversation
cold-start curve. The metadata records `concurrency` and
`reset_between_conversations=false` so analysis can tell the two apart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

import _common as common
from _common import (
    ConvResult, TurnResult, add_common_args, capture_run_metadata,
    checkpoint_key, compute_metrics, is_run_complete, load_checkpoint,
    load_sharegpt, log, print_summary, save_checkpoint, setup_logging,
)

RESULTS_DIR    = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"


# ── HTTP helpers ────────────────────────────────────────────────────────────

class ServerError(Exception):
    """Non-retryable server error (4xx that isn't 503, 5xx that isn't transient)."""


async def _post_completion(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> Tuple[dict, Dict[str, str]]:
    """POST to /v1/completions with 503-aware retry.

    Returns (json_body, response_headers). Raises ServerError on
    non-retryable failures or after `max_retries` 503s.
    """
    delay = base_delay
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(url, json=body)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError,
                httpx.RemoteProtocolError) as e:
            last_exc = e
            log.warning("HTTP transport error (attempt %d/%d): %s",
                        attempt + 1, max_retries + 1, e)
        else:
            if resp.status_code == 200:
                return resp.json(), dict(resp.headers)

            if resp.status_code == 503:
                # Server queue full — exactly the back-pressure case
                last_exc = ServerError(f"503 queue full: {resp.text[:200]}")
            elif resp.status_code == 504:
                last_exc = ServerError(f"504 timeout: {resp.text[:200]}")
            elif 500 <= resp.status_code < 600:
                # 5xx — retry conservatively (could be transient OOM cleanup)
                last_exc = ServerError(f"{resp.status_code}: {resp.text[:200]}")
            else:
                # 4xx — not our fault to retry
                raise ServerError(
                    f"HTTP {resp.status_code} (non-retryable): {resp.text[:300]}"
                )

        # Backoff with jitter, then loop
        if attempt < max_retries:
            jitter = random.uniform(0, delay * 0.3)
            await asyncio.sleep(min(delay + jitter, max_delay))
            delay = min(delay * 2.0, max_delay)

    raise ServerError(f"Exhausted {max_retries} retries: {last_exc}")


def _parse_kvboost_headers(headers: Dict[str, str]) -> Dict[str, float]:
    """Pull X-KVBoost-* response headers populated by app.py.

    HTTP headers are case-insensitive on the wire; httpx normalizes to
    lowercase keys, so we lowercase once and look up that way.
    """
    lc = {k.lower(): v for k, v in headers.items()}

    def _num(name: str) -> Optional[float]:
        v = lc.get(name.lower())
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "ttft_ms":         _num("X-KVBoost-Ttft-Ms"),
        "total_ms":        _num("X-KVBoost-Total-Ms"),
        "prompt_tokens":   _num("X-KVBoost-Prompt-Tokens"),
        "cached_tokens":   _num("X-KVBoost-Cached-Tokens"),
        "generated_tokens":_num("X-KVBoost-Generated-Tokens"),
        "kv_reuse_ratio":  _num("X-KVBoost-Kv-Reuse-Ratio"),
    }


# ── Conversation worker ─────────────────────────────────────────────────────

async def _process_conversation(
    *,
    conv: dict,
    conv_idx: int,
    n_total: int,
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
    """Replay one conversation turn-by-turn against the server.

    Turns within a conversation are still serial — turn N's prompt depends on
    turn N-1's output. Parallelism is purely *across* conversations.
    """
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
        t_turn = time.perf_counter()

        body = {
            "model":       model_name,
            "prompts":     [prompt],
            "max_tokens":  max_new_tokens,
            "temperature": 0.0,
            "do_sample":   False,
            "stream":      False,
        }

        error_msg: Optional[str] = None
        kv: Dict[str, float] = {}
        output_text = ""
        out_tok = 0
        prompt_tokens = history_tokens
        cached = 0
        ttft_ms = 0.0
        total_ms = 0.0

        try:
            resp, headers = await _post_completion(
                client, completions_url, body,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            )
            kv = _parse_kvboost_headers(headers)
            choices = resp.get("choices") or []
            output_text = (choices[0].get("text") if choices else "") or ""

            # Prefer engine-reported numbers (from X-KVBoost-* headers); fall
            # back to client-side wall time and usage stats if headers absent.
            wall_total_ms = (time.perf_counter() - t_turn) * 1000.0
            ttft_ms  = kv.get("ttft_ms")  if kv.get("ttft_ms")  is not None else wall_total_ms
            total_ms = kv.get("total_ms") if kv.get("total_ms") is not None else wall_total_ms
            usage    = resp.get("usage") or {}
            out_tok  = int(kv.get("generated_tokens") or usage.get("completion_tokens") or 0)
            prompt_tokens = int(kv.get("prompt_tokens") or usage.get("prompt_tokens") or history_tokens)
            cached   = int(kv.get("cached_tokens") or 0)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            log.error("Conv %s turn %d failed: %s", conv_id, human_turn_idx, error_msg)

        if error_msg is not None:
            tr = TurnResult(
                conv_id=conv_id,
                turn_idx=human_turn_idx,
                n_turns_total=n_human,
                history_tokens=history_tokens,
                prompt_tokens=history_tokens,
                cached_tokens=0,
                cache_hit_ratio=0.0,
                ttft_ms=0.0,
                total_ms=0.0,
                decode_ms=0.0,
                output_tokens=0,
                itl_ms=0.0,
                decode_tps=0.0,
                error=error_msg,
                stop_reason="error",
                turn_start_iso=turn_start_iso,
            )
            conv_result.turns.append(tr)
            conv_result.error_count += 1
            break  # history is broken — abort the rest of this conv

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
                "kv_reuse_ratio_header": kv.get("kv_reuse_ratio"),
                "wall_total_ms":         (time.perf_counter() - t_turn) * 1000.0,
            },
        )
        conv_result.turns.append(tr)
        history = prompt + output_text + "\n"
        human_turn_idx += 1

    conv_result.end_iso = datetime.now(timezone.utc).isoformat()
    conv_result.wall_s = time.perf_counter() - conv_start
    return conv_result


# ── Top-level parallel replay ───────────────────────────────────────────────

async def replay_parallel(
    *,
    conversations: List[dict],
    server_url: str,
    model_name: str,
    concurrency: int,
    max_new_tokens: int,
    count_tokens,
    ck_path: Path,
    meta: dict,
    run_metadata,
    no_checkpoint: bool,
    save_output_text: bool,
    progress_every: int,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    request_timeout: float,
) -> List[ConvResult]:
    """Fire N conversation workers concurrently against the server.

    Conversation independence: each worker owns its own history string.
    Server-side KV cache is shared (intentionally — that's the realistic
    multi-tenant case), so the per-conv reset present in the in-process
    variant is not performed here.
    """
    if no_checkpoint:
        all_results: List[ConvResult] = []
        processed_ids: List[str] = []
    else:
        all_results, processed_ids = load_checkpoint(ck_path)
    processed_set = set(processed_ids)

    pending = [c for c in conversations if c["id"] not in processed_set]
    if not pending:
        log.info("All %d conversations already complete; nothing to do.", len(conversations))
        return all_results

    log.info(
        "Parallel replay: %d conversations pending, concurrency=%d, server=%s",
        len(pending), concurrency, server_url,
    )

    sem = asyncio.Semaphore(concurrency)
    completions_url = server_url.rstrip("/") + "/v1/completions"

    timeout = httpx.Timeout(connect=10.0, read=request_timeout, write=30.0, pool=10.0)
    limits = httpx.Limits(
        max_keepalive_connections=concurrency * 2,
        max_connections=concurrency * 4,
    )

    completed = {"n": 0}
    interrupted = {"flag": False}

    def _save_now(reason: str) -> None:
        if no_checkpoint:
            return
        try:
            save_checkpoint(
                all_results, [r.conv_id for r in all_results], ck_path,
                meta, run_metadata=run_metadata,
            )
            log.info("Checkpoint saved on %s (%d conversations)", reason, len(all_results))
        except Exception as e:
            log.error("Failed to save checkpoint: %s", e)

    def _signal_handler(signum, frame):  # noqa: ARG001
        log.warning("Signal %s caught; finishing in-flight convs then exiting.", signum)
        interrupted["flag"] = True

    old_sigint = signal.signal(signal.SIGINT, _signal_handler)
    try:
        old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        old_sigterm = None

    total_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=False) as client:

        async def _run_one(idx: int, conv: dict) -> None:
            if interrupted["flag"]:
                return
            async with sem:
                if interrupted["flag"]:
                    return
                try:
                    cr = await _process_conversation(
                        conv=conv,
                        conv_idx=idx,
                        n_total=len(conversations),
                        client=client,
                        completions_url=completions_url,
                        model_name=model_name,
                        max_new_tokens=max_new_tokens,
                        count_tokens=count_tokens,
                        save_output_text=save_output_text,
                        max_retries=max_retries,
                        base_delay=base_delay,
                        max_delay=max_delay,
                    )
                except Exception as e:
                    log.error("Conversation %s crashed: %s", conv["id"], e)
                    cr = ConvResult(
                        conv_id=conv["id"],
                        n_turns=0,
                        start_iso=datetime.now(timezone.utc).isoformat(),
                        error_count=1,
                    )

            all_results.append(cr)
            completed["n"] += 1
            n = completed["n"]
            if n % progress_every == 0 or n == 1:
                elapsed = time.perf_counter() - total_start
                ttfts = [t.ttft_ms for r in all_results for t in r.turns if not t.error]
                ttft_p50 = (sorted(ttfts)[len(ttfts) // 2] if ttfts else 0.0)
                log.info(
                    "  [%d/%d done] conv=%s turns=%d errs=%d ttft_p50=%.0fms elapsed=%.0fs",
                    n, len(pending), cr.conv_id, cr.n_turns, cr.error_count,
                    ttft_p50, elapsed,
                )
            if n % max(progress_every * 4, 20) == 0:
                _save_now("periodic")

        tasks = [asyncio.create_task(_run_one(i, c)) for i, c in enumerate(pending)]
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            interrupted["flag"] = True
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    signal.signal(signal.SIGINT, old_sigint)
    if old_sigterm is not None:
        try:
            signal.signal(signal.SIGTERM, old_sigterm)
        except Exception:
            pass

    if not no_checkpoint:
        _save_now("finalize")

    log.info(
        "Parallel replay completed: %d conversations in %.1fs (concurrency=%d)",
        len(all_results), time.perf_counter() - total_start, concurrency,
    )
    return all_results


# ── CLI ─────────────────────────────────────────────────────────────────────

def _check_server_alive(server_url: str) -> dict:
    """Hit /health and /v1/models so we fail fast with a clear error."""
    base = server_url.rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as c:
            health = c.get(base + "/health")
            health.raise_for_status()
            models = c.get(base + "/v1/models")
            models.raise_for_status()
            return models.json()
    except Exception as e:
        sys.exit(
            f"Cannot reach kvboost-server at {server_url}: {e}\n"
            f"Start it with:  kvboost-server --model <hf-id> --port 8000"
        )


def main():
    parser = argparse.ArgumentParser(
        description="KVBoost server-mode 3-way ShareGPT runner (parallel)",
    )
    add_common_args(parser)
    parser.add_argument("--server-url", default="http://localhost:8000",
                        help="kvboost-server base URL (default: http://localhost:8000)")
    parser.add_argument("--model", default=None,
                        help="Model id (must match what kvboost-server loaded). "
                             "If omitted, taken from GET /v1/models.")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of conversations in flight at once (default: 8).")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer id for token counting / filtering. "
                             "Defaults to --model.")
    parser.add_argument("--max-retries", type=int, default=20,
                        help="Per-request retry budget on 503/5xx (default: 20).")
    parser.add_argument("--base-delay", type=float, default=0.5,
                        help="Initial backoff delay in seconds (default: 0.5).")
    parser.add_argument("--max-delay", type=float, default=30.0,
                        help="Backoff ceiling in seconds (default: 30).")
    parser.add_argument("--request-timeout", type=float, default=600.0,
                        help="Per-request HTTP read timeout in seconds (default: 600). "
                             "Generation on a streamed 32B can take minutes.")
    args = parser.parse_args()

    setup_logging(args.verbose, args.debug)

    print(f"\n{'=' * 72}\n  KVBoost server-mode (parallel) — ShareGPT 3-way\n{'=' * 72}")
    print(f"  server      = {args.server_url}")
    print(f"  concurrency = {args.concurrency}")

    models_resp = _check_server_alive(args.server_url)
    loaded = (models_resp.get("data") or [{}])[0].get("id", "<unknown>")
    model_name = args.model or loaded
    if args.model and args.model != loaded:
        log.warning("--model=%s but server has %s loaded; using %s.",
                    args.model, loaded, args.model)
    print(f"  model       = {model_name}  (server loaded: {loaded})")
    print(f"  n_samples   = {args.n_samples}  turns={args.min_turns}-{args.max_turns}")
    print(f"{'=' * 72}\n")

    out_path = Path(args.output) if args.output else RESULTS_DIR / "kvboost_server.json"
    if not args.no_checkpoint and is_run_complete(out_path, args.n_samples):
        print(f"[skip] {out_path} already covers {args.n_samples} conversations; "
              "delete it or pass --no-checkpoint to force re-run.")
        return

    # Tokenizer is needed for filtering / per-turn token counts.
    from transformers import AutoTokenizer
    tok_id = args.tokenizer or model_name
    log.info("Loading tokenizer: %s", tok_id)
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

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
        f"kvboost_server_{checkpoint_key('kvboost_server', model_name, args.n_samples, args.max_turns)}.json"
    )
    meta = {
        "backend":     "kvboost_server",
        "model":       model_name,
        "server_url":  args.server_url,
        "concurrency": args.concurrency,
    }
    config = {
        "concurrency":          args.concurrency,
        "max_new_tokens":       args.max_new_tokens,
        "n_samples":            args.n_samples,
        "min_turns":            args.min_turns,
        "max_turns":            args.max_turns,
        "max_context_tokens":   args.max_context_tokens,
        "max_tokens_per_turn":  args.max_tokens_per_turn,
        "save_output_text":     args.save_output_text,
        "max_retries":          args.max_retries,
        "base_delay":           args.base_delay,
        "max_delay":            args.max_delay,
        "request_timeout":      args.request_timeout,
        # ⚠ Methodology flag: cache is server-wide and shared. Downstream
        # analysis should not compare these numbers directly to in-process
        # numbers that DO reset between convs.
        "reset_between_conversations": False,
    }
    run_metadata = capture_run_metadata("kvboost_server", config)

    def count_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=True))

    t0 = time.perf_counter()
    results = asyncio.run(replay_parallel(
        conversations=conversations,
        server_url=args.server_url,
        model_name=model_name,
        concurrency=args.concurrency,
        max_new_tokens=args.max_new_tokens,
        count_tokens=count_tokens,
        ck_path=ck_path,
        meta=meta,
        run_metadata=run_metadata,
        no_checkpoint=args.no_checkpoint,
        save_output_text=args.save_output_text,
        progress_every=args.progress_every,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        max_delay=args.max_delay,
        request_timeout=args.request_timeout,
    ))
    wall_s = time.perf_counter() - t0
    run_metadata.end_iso = datetime.now(timezone.utc).isoformat()

    metrics = compute_metrics(results, total_wall_s=wall_s)
    print_summary("kvboost_server", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend":      "kvboost_server",
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
