"""
Shared HTTP-client infrastructure for the server-mode replay runners
(``run_kvboost_server.py``, ``run_vllm_server.py``, ``run_llamacpp_server.py``).

What lives here
---------------
* ``ServerError``        — raised on non-retryable HTTP responses.
* ``post_with_retry``    — POST one completion with exponential-backoff retry
                           on 503 (queue full / back-pressure) and other 5xx.
* ``replay_parallel``    — orchestrate N concurrent conversation workers with
                           an ``asyncio.Semaphore``; checkpoint, signal-handle,
                           progress-log.  Backends supply their own per-turn
                           coroutine (extracted telemetry differs per backend).

What does NOT live here
-----------------------
Per-backend response parsing (TTFT source, cached-tokens header / usage field,
streaming chunk format).  Each backend script defines its own
``_process_conversation`` and passes that as ``process_conv_fn`` here.
"""

from __future__ import annotations

import asyncio
import json
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from _common import (
    ConvResult, load_checkpoint, log, save_checkpoint,
)


# ── Exceptions ──────────────────────────────────────────────────────────────

class ServerError(Exception):
    """Non-retryable server error or retries exhausted."""


# ── HTTP POST with retry ────────────────────────────────────────────────────

async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> Tuple[dict, Dict[str, str]]:
    """POST a JSON body with 503-aware exponential-backoff retry.

    The 503 path is exactly what gives us the "OOM ⇒ queue sequentially"
    behaviour: when the server's queue is full it returns 503, we sleep,
    we retry — concurrency naturally falls back to whatever the server can
    actually absorb.

    Returns ``(parsed_json, headers_dict)``.  Raises ``ServerError`` on
    non-retryable 4xx, or after ``max_retries`` retries of 5xx/transport
    errors.
    """
    delay = base_delay
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(url, json=body)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError,
                httpx.RemoteProtocolError, httpx.ConnectError) as e:
            last_exc = e
            log.warning("HTTP transport error (attempt %d/%d): %s",
                        attempt + 1, max_retries + 1, e)
        else:
            if resp.status_code == 200:
                return resp.json(), dict(resp.headers)

            if resp.status_code == 503:
                # Server queue full — back-pressure case
                last_exc = ServerError(f"503 queue full: {resp.text[:200]}")
            elif resp.status_code == 504:
                last_exc = ServerError(f"504 timeout: {resp.text[:200]}")
            elif 500 <= resp.status_code < 600:
                # 5xx — retry conservatively (could be transient OOM cleanup)
                last_exc = ServerError(f"{resp.status_code}: {resp.text[:200]}")
            else:
                # 4xx — client error, do not retry
                raise ServerError(
                    f"HTTP {resp.status_code} (non-retryable): {resp.text[:300]}"
                )

        if attempt < max_retries:
            jitter = random.uniform(0, delay * 0.3)
            await asyncio.sleep(min(delay + jitter, max_delay))
            delay = min(delay * 2.0, max_delay)

    raise ServerError(f"Exhausted {max_retries} retries: {last_exc}")


# ── Streaming POST (for TTFT measurement) ──────────────────────────────────

async def post_stream_with_retry(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> Tuple[str, float, float, Dict[str, str], Optional[dict]]:
    """POST with ``stream=true`` and return (output_text, ttft_ms, total_ms,
    headers, usage_dict).

    Used for backends (vLLM, llama-server) whose non-streaming response has no
    TTFT field. We open an SSE stream, time the first delta chunk to get
    accurate TTFT, then drain the rest to get total_ms and the assembled
    output text. ``usage_dict`` is best-effort: present only if the server
    emits a usage chunk (e.g. ``stream_options.include_usage=true``).

    Retries on 503 / 5xx / transport errors *before* the stream starts.
    Mid-stream failures propagate.
    """
    body = dict(body)
    body["stream"] = True

    delay = base_delay
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    text = text.decode("utf-8", errors="replace")[:300]
                    if resp.status_code == 503:
                        last_exc = ServerError(f"503 queue full: {text}")
                    elif resp.status_code == 504:
                        last_exc = ServerError(f"504 timeout: {text}")
                    elif 500 <= resp.status_code < 600:
                        last_exc = ServerError(f"{resp.status_code}: {text}")
                    else:
                        raise ServerError(
                            f"HTTP {resp.status_code} (non-retryable): {text}"
                        )
                else:
                    # Stream open — read SSE lines, time first delta.
                    output_text = ""
                    ttft_ms: Optional[float] = None
                    usage: Optional[dict] = None
                    headers = dict(resp.headers)

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except Exception:
                            continue
                        # Usage chunk (OpenAI stream_options.include_usage)
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        for ch in choices:
                            # vLLM/llama-server text completions: choice has
                            # "text"; chat: "delta": {"content": "..."}.
                            delta = ch.get("text") or (
                                (ch.get("delta") or {}).get("content")
                            )
                            if delta:
                                if ttft_ms is None:
                                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                                output_text += delta

                    total_ms = (time.perf_counter() - t0) * 1000.0
                    if ttft_ms is None:
                        ttft_ms = total_ms
                    return output_text, ttft_ms, total_ms, headers, usage

        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError,
                httpx.RemoteProtocolError, httpx.ConnectError) as e:
            last_exc = e
            log.warning("HTTP transport error (attempt %d/%d): %s",
                        attempt + 1, max_retries + 1, e)
        except ServerError:
            raise
        except Exception as e:
            # Mid-stream failure — not retryable, surface it
            raise ServerError(f"stream error: {type(e).__name__}: {e}")

        if attempt < max_retries:
            jitter = random.uniform(0, delay * 0.3)
            await asyncio.sleep(min(delay + jitter, max_delay))
            delay = min(delay * 2.0, max_delay)

    raise ServerError(f"Exhausted {max_retries} retries: {last_exc}")


# ── Parallel replay orchestrator ────────────────────────────────────────────

ProcessConvFn = Callable[..., Awaitable[ConvResult]]


async def replay_parallel(
    *,
    process_conv_fn: ProcessConvFn,
    process_conv_kwargs: dict,
    conversations: List[dict],
    concurrency: int,
    ck_path: Path,
    meta: dict,
    run_metadata,
    no_checkpoint: bool,
    progress_every: int,
    request_timeout: float,
    backend_label: str,
) -> List[ConvResult]:
    """Drive ``process_conv_fn`` over conversations with bounded concurrency.

    ``process_conv_fn`` is an async callable expected to consume the
    ``process_conv_kwargs`` plus three positional / keyword args we supply per
    invocation:  ``conv=...``, ``client=...``, ``count_tokens`` is expected to
    be inside the kwargs already (it's per-backend stable).
    """
    if no_checkpoint:
        all_results: List[ConvResult] = []
        processed_ids: List[str] = []
    else:
        all_results, processed_ids = load_checkpoint(ck_path)
    processed_set = set(processed_ids)

    pending = [c for c in conversations if c["id"] not in processed_set]
    if not pending:
        log.info("All %d conversations already complete; nothing to do.",
                 len(conversations))
        return all_results

    log.info(
        "[%s] parallel replay: %d conversations pending, concurrency=%d",
        backend_label, len(pending), concurrency,
    )

    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(connect=10.0, read=request_timeout,
                            write=30.0, pool=10.0)
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
            log.info("Checkpoint saved on %s (%d conversations)",
                     reason, len(all_results))
        except Exception as e:
            log.error("Failed to save checkpoint: %s", e)

    def _signal_handler(signum, frame):  # noqa: ARG001
        log.warning("Signal %s caught; finishing in-flight convs then exiting.",
                    signum)
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
                    cr = await process_conv_fn(
                        conv=conv,
                        client=client,
                        **process_conv_kwargs,
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
                ttfts = [t.ttft_ms for r in all_results for t in r.turns
                         if not t.error]
                ttft_p50 = (sorted(ttfts)[len(ttfts) // 2] if ttfts else 0.0)
                log.info(
                    "  [%s %d/%d done] conv=%s turns=%d errs=%d "
                    "ttft_p50=%.0fms elapsed=%.0fs",
                    backend_label, n, len(pending), cr.conv_id, cr.n_turns,
                    cr.error_count, ttft_p50, elapsed,
                )
            if n % max(progress_every * 4, 20) == 0:
                _save_now("periodic")

        tasks = [asyncio.create_task(_run_one(i, c))
                 for i, c in enumerate(pending)]
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
        "[%s] parallel replay completed: %d conversations in %.1fs (concurrency=%d)",
        backend_label, len(all_results),
        time.perf_counter() - total_start, concurrency,
    )
    return all_results


# ── Server liveness helpers ─────────────────────────────────────────────────

def check_openai_server(server_url: str, expect_path: str = "/v1/models") -> dict:
    """Hit ``/v1/models`` (and ``/health`` if available) so we fail fast with
    a clear error if the server isn't up. Returns the parsed JSON of
    ``expect_path``."""
    import sys as _sys
    base = server_url.rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as c:
            # /health is convention but not all servers expose it
            try:
                c.get(base + "/health").raise_for_status()
            except Exception:
                pass
            resp = c.get(base + expect_path)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        _sys.exit(
            f"Cannot reach server at {server_url}{expect_path}: {e}\n"
            f"  - Is the backend running on the expected port?\n"
            f"  - Is the URL correct (http vs https, host, port)?"
        )
