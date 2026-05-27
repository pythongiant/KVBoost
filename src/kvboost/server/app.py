"""
FastAPI application — OpenAI-compatible inference server for KVBoost.

Endpoints
---------
GET  /health                    — liveness probe
GET  /v1/models                 — list loaded model
POST /v1/completions            — text completion (streaming + non-streaming)
POST /v1/chat/completions       — chat completion (streaming + non-streaming)
GET  /v1/stats                  — server / cache / queue diagnostics
POST /v1/warm                   — pre-warm the KV cache with a text string

Streaming
---------
When stream=True the response is a text/event-stream (SSE) where each
chunk is a JSON-encoded CompletionChunk / ChatCompletionChunk followed by
a final "data: [DONE]" sentinel — identical to the OpenAI streaming format.

Error handling
--------------
HTTP 400  — invalid request (Pydantic validation failure)
HTTP 503  — request queue full (back-pressure)
HTTP 504  — request timed out waiting for the worker
HTTP 500  — unexpected model error
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    ModelCard,
    ModelList,
    UsageStats,
)
from .engine_worker import EngineWorker
from . import tool_parsers

log = logging.getLogger(__name__)
io_log = logging.getLogger("kvboost.server.io")


def _truncate(text: str, limit: int = 500) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…<+{len(text) - limit} chars>"


def _attach_kvboost_headers(response: Response, result, request_id: str) -> None:
    """Expose per-request KVBoost telemetry via X-KVBoost-* response headers.

    OpenAI's response shape has no slot for ttft/cached_tokens/kv_reuse, so we
    return them as headers — OpenAI clients ignore them, and our benchmark
    client (run_kvboost_server.py) reads them to populate TurnResult fields
    without needing in-process access to the engine.
    """
    def _set(name: str, value):
        if value is None:
            return
        response.headers[name] = str(value)

    _set("X-KVBoost-Request-Id", request_id)
    _set("X-KVBoost-Ttft-Ms", f"{getattr(result, 'ttft_ms', 0.0):.3f}")
    total_ms = getattr(result, "total_ms", None)
    if total_ms is not None:
        _set("X-KVBoost-Total-Ms", f"{total_ms:.3f}")
    _set("X-KVBoost-Prompt-Tokens", getattr(result, "prompt_tokens", None))
    _set("X-KVBoost-Cached-Tokens", getattr(result, "cached_tokens", None))
    _set("X-KVBoost-Generated-Tokens", getattr(result, "generated_tokens", None))
    reuse = getattr(result, "kv_reuse_ratio", None)
    if reuse is not None:
        _set("X-KVBoost-Kv-Reuse-Ratio", f"{float(reuse):.6f}")


def build_app(
    worker: EngineWorker,
    model_name: Optional[str] = None,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str = "hermes",
    max_tokens_cap: Optional[int] = None,
) -> FastAPI:
    """
    Construct and return the FastAPI application.

    Parameters
    ----------
    worker                   : a started EngineWorker instance
    model_name               : override for the model id shown in /v1/models
    enable_auto_tool_choice  : if True, parse model output for tool calls when
                               the request includes `tools`
    tool_call_parser         : parser name (see tool_parsers.PARSERS)
    max_tokens_cap           : if set, clamp each request's ``max_tokens`` down
                               to this value before dispatching to the engine.
                               Independent of the schema-level cap (131072) so
                               the operator can pick a safe ceiling for the
                               actual VRAM/KV-cache budget on this server.
    """
    _model_name = model_name or worker._model_name
    _auto_tools = enable_auto_tool_choice
    _parser_name = tool_call_parser
    _max_tokens_cap = max_tokens_cap

    def _clamp_max_tokens(req) -> None:
        """Mutate ``req.max_tokens`` down to ``_max_tokens_cap`` if set.
        Logs once per clamp so operators can see when clients are asking for
        more than this server will serve."""
        if _max_tokens_cap is None:
            return
        if req.max_tokens > _max_tokens_cap:
            io_log.info(
                "max_tokens clamped: %d → %d (server cap)",
                req.max_tokens, _max_tokens_cap,
            )
            req.max_tokens = _max_tokens_cap

    app = FastAPI(
        title="KVBoost Inference Server",
        description="OpenAI-compatible API powered by KVBoost chunk-level KV caching.",
        version="0.4.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request/response access log ───────────────────────────────────────────

    @app.middleware("http")
    async def _log_io(request: Request, call_next):
        req_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        client = f"{request.client.host}:{request.client.port}" if request.client else "-"
        io_log.info(
            "REQ  id=%s %s %s client=%s",
            req_id, request.method, request.url.path, client,
        )
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            io_log.exception("ERR  id=%s %s %s elapsed=%.1fms",
                             req_id, request.method, request.url.path, elapsed_ms)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        io_log.info(
            "RES  id=%s %s %s status=%d elapsed=%.1fms",
            req_id, request.method, request.url.path, response.status_code, elapsed_ms,
        )
        return response

    # ── Startup / shutdown ────────────────────────────────────────────────────

    @app.on_event("startup")
    async def _startup():
        await worker.start()
        log.info("KVBoost server ready — model=%s", _model_name)

    @app.on_event("shutdown")
    async def _shutdown():
        await worker.stop()

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health", tags=["utility"])
    async def health():
        return {"status": "ok", "model": _model_name}

    # ── Models ────────────────────────────────────────────────────────────────

    @app.get("/v1/models", response_model=ModelList, tags=["models"])
    async def list_models():
        return ModelList(data=[ModelCard(id=_model_name)])

    # ── Stats ─────────────────────────────────────────────────────────────────

    @app.get("/v1/stats", tags=["utility"])
    async def stats():
        return worker.stats()

    # ── Warm ──────────────────────────────────────────────────────────────────

    @app.post("/v1/warm", tags=["utility"])
    async def warm(request: Request):
        body = await request.json()
        text = body.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Field 'text' is required.")
        io_log.info("WARM in: chars=%d text=%r", len(text), _truncate(text))
        await worker.warm(text)
        io_log.info("WARM out: chars=%d", len(text))
        return {"status": "warmed", "chars": len(text)}

    # ── /v1/completions ───────────────────────────────────────────────────────

    @app.post("/v1/completions", tags=["completions"])
    async def completions(req: CompletionRequest, response: Response):
        _validate_model(req.model, _model_name)
        _clamp_max_tokens(req)

        io_log.info(
            "COMPLETIONS in: model=%s n_prompts=%d max_tokens=%d temp=%s stream=%s",
            req.model, len(req.prompts), req.max_tokens, req.temperature, req.stream,
        )
        for i, p in enumerate(req.prompts):
            io_log.info("  prompt[%d]=%r", i, _truncate(p))

        if req.stream:
            return StreamingResponse(
                _stream_completions(req, worker, _model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        results = await _run_completions(req, worker, _model_name)
        prompt_tokens = sum(len(worker._tokenize(p)) for p in req.prompts)
        completion_tokens = sum(len(worker._tokenize(r.output_text)) for r in results)

        for i, r in enumerate(results):
            io_log.info("COMPLETIONS out[%d]=%r", i, _truncate(r.output_text))
        io_log.info(
            "COMPLETIONS done: prompt_tokens=%d completion_tokens=%d total=%d",
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
        )

        # Telemetry headers — for a multi-prompt batch we expose the first
        # result's stats. Single-prompt is the dominant case (chat & bench).
        if results:
            _attach_kvboost_headers(response, results[0], request_id=f"cmpl-{uuid.uuid4().hex[:12]}")

        choices = [
            CompletionChoice(text=r.output_text, index=i)
            for i, r in enumerate(results)
        ]
        return CompletionResponse(
            model=_model_name,
            choices=choices,
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ── /v1/chat/completions ──────────────────────────────────────────────────

    @app.post("/v1/chat/completions", tags=["chat"])
    async def chat_completions(req: ChatCompletionRequest, response: Response):
        _validate_model(req.model, _model_name)
        _clamp_max_tokens(req)

        prompt = req.to_prompt(worker.engine.tokenizer)

        io_log.info(
            "CHAT in: model=%s n_messages=%d max_tokens=%d temp=%s stream=%s",
            req.model, len(req.messages), req.max_tokens, req.temperature, req.stream,
        )
        for i, m in enumerate(req.messages):
            io_log.info("  msg[%d] role=%s content=%r", i, m.role, _truncate(m.content))
        io_log.debug("CHAT prompt=%r", _truncate(prompt, 1000))

        if req.stream:
            return StreamingResponse(
                _stream_chat(
                    req, prompt, worker, _model_name,
                    auto_tools=_auto_tools, parser_name=_parser_name,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await _run_single(prompt, req.max_tokens, req.temperature,
                                   req.do_sample, worker, _model_name)

        prompt_tokens = len(worker._tokenize(prompt))
        completion_tokens = len(worker._tokenize(result.output_text))

        _attach_kvboost_headers(response, result, request_id=f"chat-{uuid.uuid4().hex[:12]}")

        io_log.info("CHAT out=%r", _truncate(result.output_text))
        io_log.info(
            "CHAT done: prompt_tokens=%d completion_tokens=%d total=%d generated=%d",
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
            result.generated_tokens,
        )

        tool_calls = None
        cleaned_text = result.output_text
        tools_active = (
            _auto_tools
            and bool(req.tools)
            and req.tool_choice != "none"
        )
        if tools_active:
            # Build an allowlist of declared tool names — the parser uses it
            # to drop calls naming hallucinated / unrelated functions, which
            # also prevents misclassifying incidental ```json blocks as calls.
            tool_names = {
                t.function.name for t in (req.tools or [])
                if t.function and t.function.name
            }
            cleaned_text, parsed_calls = tool_parsers.parse(
                result.output_text, _parser_name, tool_names=tool_names,
            )
            parsed_calls = _filter_tool_choice(parsed_calls, req.tool_choice)
            if parsed_calls:
                tool_calls = parsed_calls
                io_log.info("CHAT tool_calls=%d names=%s",
                            len(parsed_calls),
                            [tc.function.name for tc in parsed_calls])
            elif req.tool_choice == "required":
                io_log.warning(
                    "CHAT tool_choice=required but model emitted no tool calls"
                )

        if tool_calls:
            finish_reason = "tool_calls"
            message = ChatMessage(
                role="assistant",
                content=cleaned_text or None,
                tool_calls=tool_calls,
            )
        else:
            finish_reason = "stop" if result.generated_tokens < req.max_tokens else "length"
            message = ChatMessage(role="assistant", content=cleaned_text)

        return ChatCompletionResponse(
            model=_model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(asyncio.QueueFull)
    async def _queue_full(_req, exc):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": {"message": str(exc), "type": "server_error", "code": 503}},
        )

    @app.exception_handler(asyncio.TimeoutError)
    async def _timeout(_req, exc):
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"error": {"message": "Request timed out.", "type": "timeout", "code": 504}},
        )

    return app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_tool_choice(calls, tool_choice):
    """
    Apply OpenAI tool_choice semantics to a parsed list of ToolCall objects.

    - None / "auto" : pass through
    - "none"        : drop everything (callers should also skip parsing)
    - "required"    : pass through (enforcement is best-effort; we can't force
                      the model post-hoc, just flag the absence at the call site)
    - {"type":"function","function":{"name":"X"}} : keep only matching name
    """
    if not calls:
        return calls
    if tool_choice in (None, "auto", "required"):
        return calls
    if tool_choice == "none":
        return []
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        wanted = fn.get("name")
        if wanted:
            return [c for c in calls if c.function.name == wanted]
    return calls


def _validate_model(requested: str, available: str) -> None:
    if requested != available:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{requested}' not loaded. Available: '{available}'.",
        )


async def _run_single(prompt, max_tokens, temperature, do_sample, worker, model_name):
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    try:
        result = await worker.generate(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            stream=False,
            model_name=model_name,
        )
    except asyncio.QueueFull:
        raise
    except asyncio.TimeoutError:
        raise
    except Exception as exc:
        log.exception("Generation error for request %s", request_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return result


async def _run_completions(req, worker, model_name):
    tasks = [
        _run_single(p, req.max_tokens, req.temperature, req.do_sample, worker, model_name)
        for p in req.prompts
    ]
    return await asyncio.gather(*tasks)


async def _stream_completions(
    req: CompletionRequest, worker: EngineWorker, model_name: str
) -> AsyncGenerator[str, None]:
    """
    SSE generator for /v1/completions with stream=True.

    We run the full generation (non-streaming in the engine) and emit the
    text as a single content delta followed by [DONE].  A future improvement
    would hook into the engine's decode loop to emit tokens one-by-one.
    """
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    for i, prompt in enumerate(req.prompts):
        try:
            result = await worker.generate(
                request_id=request_id,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=req.do_sample,
                stream=True,
                model_name=model_name,
            )
        except Exception as exc:
            error_chunk = json.dumps({"error": {
            "message": str(exc) or exc.__class__.__name__,
            "type": "server_error",
            "code": 500,
        }})
            yield f"data: {error_chunk}\n\n"
            return

        chunk = CompletionChunk(
            id=request_id,
            model=model_name,
            choices=[{
                "text": result.output_text,
                "index": i,
                "finish_reason": "stop",
                "logprobs": None,
            }],
        )
        io_log.info("COMPLETIONS stream out[%d] id=%s text=%r",
                    i, request_id, _truncate(result.output_text))
        yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


async def _stream_chat(
    req: ChatCompletionRequest,
    prompt: str,
    worker: EngineWorker,
    model_name: str,
    *,
    auto_tools: bool = False,
    parser_name: str = "hermes",
) -> AsyncGenerator[str, None]:
    """SSE generator for /v1/chat/completions with stream=True.

    Emits real token-by-token deltas. When `auto_tools` is on and the request
    includes `tools`, the assistant text stream is run through a streaming
    Hermes parser: plain content is emitted as `delta.content`, and complete
    `<tool_call>{...}</tool_call>` blocks are emitted as OpenAI-format
    `delta.tool_calls`. Partial markup is held back so the client never sees
    half-tags.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    tokenizer = worker.engine.tokenizer

    tools_active = (
        auto_tools and bool(req.tools) and req.tool_choice != "none"
    )
    stream_tool_names = (
        {
            t.function.name for t in (req.tools or [])
            if t.function and t.function.name
        }
        if tools_active else None
    )
    stream_parser = (
        tool_parsers.make_streaming_parser(parser_name, tool_names=stream_tool_names)
        if tools_active else None
    )
    emitted_tool_calls = []  # list of ToolCall, used to set finish_reason
    tool_call_index = 0       # OpenAI delta indexing within this completion

    def _content_chunk(text: str) -> str:
        chunk = ChatCompletionChunk(
            id=request_id,
            model=model_name,
            choices=[{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }],
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    def _tool_call_chunk(idx: int, call) -> str:
        # Single-shot delta: emit id, name, and full arguments together.
        # OpenAI clients (openai-python, LangChain, LiteLLM) accept this.
        chunk = ChatCompletionChunk(
            id=request_id,
            model=model_name,
            choices=[{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": idx,
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }],
                },
                "finish_reason": None,
            }],
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    # Role delta first — tells the client a response is starting.
    role_chunk = ChatCompletionChunk(
        id=request_id,
        model=model_name,
        choices=[{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
    )
    yield f"data: {role_chunk.model_dump_json()}\n\n"

    all_tokens: list[int] = []
    prev_text: str = ""
    final_result = None

    try:
        async for kind, payload in worker.stream_generate(
            request_id=request_id,
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.do_sample,
            model_name=model_name,
        ):
            if kind == "token":
                all_tokens.append(payload)
                cur_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
                delta = cur_text[len(prev_text):]
                if not delta:
                    # Multi-byte char not yet complete — wait for next token.
                    continue
                prev_text = cur_text

                if stream_parser is None:
                    yield _content_chunk(delta)
                else:
                    for ev_kind, ev_payload in stream_parser.feed(delta):
                        if ev_kind == "text":
                            if ev_payload:
                                yield _content_chunk(ev_payload)
                        else:  # "tool_call"
                            call = _filter_tool_choice([ev_payload], req.tool_choice)
                            if not call:
                                continue
                            yield _tool_call_chunk(tool_call_index, call[0])
                            emitted_tool_calls.append(call[0])
                            tool_call_index += 1
            elif kind == "done":
                final_result = payload
            elif kind == "error":
                error_chunk = json.dumps({"error": {
                    "message": str(payload) or payload.__class__.__name__,
                    "type": "server_error",
                    "code": 500,
                }})
                yield f"data: {error_chunk}\n\n"
                return
    except Exception as exc:
        error_chunk = json.dumps({"error": {
            "message": str(exc) or exc.__class__.__name__,
            "type": "server_error",
            "code": 500,
        }})
        yield f"data: {error_chunk}\n\n"
        return

    # Drain any held-back text/tool_call from the parser.
    if stream_parser is not None:
        for ev_kind, ev_payload in stream_parser.flush():
            if ev_kind == "text":
                if ev_payload:
                    yield _content_chunk(ev_payload)
            else:
                call = _filter_tool_choice([ev_payload], req.tool_choice)
                if call:
                    yield _tool_call_chunk(tool_call_index, call[0])
                    emitted_tool_calls.append(call[0])
                    tool_call_index += 1

    if emitted_tool_calls:
        finish_reason = "tool_calls"
    elif final_result is not None and final_result.generated_tokens >= req.max_tokens:
        finish_reason = "length"
    else:
        finish_reason = "stop"

    if tools_active and not emitted_tool_calls and req.tool_choice == "required":
        io_log.warning(
            "CHAT stream tool_choice=required but model emitted no tool calls"
        )

    stop_chunk = ChatCompletionChunk(
        id=request_id,
        model=model_name,
        choices=[{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    )
    generated = final_result.generated_tokens if final_result is not None else len(all_tokens)
    io_log.info(
        "CHAT stream out id=%s finish=%s generated_tokens=%d tool_calls=%d text=%r",
        request_id, finish_reason, generated, len(emitted_tool_calls),
        _truncate(prev_text),
    )
    yield f"data: {stop_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
