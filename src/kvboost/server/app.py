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

from fastapi import FastAPI, HTTPException, Request, status
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

log = logging.getLogger(__name__)


def build_app(worker: EngineWorker, model_name: Optional[str] = None) -> FastAPI:
    """
    Construct and return the FastAPI application.

    Parameters
    ----------
    worker      : a started EngineWorker instance
    model_name  : override for the model id shown in /v1/models
    """
    _model_name = model_name or worker._model_name

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
        await worker.warm(text)
        return {"status": "warmed", "chars": len(text)}

    # ── /v1/completions ───────────────────────────────────────────────────────

    @app.post("/v1/completions", tags=["completions"])
    async def completions(req: CompletionRequest):
        _validate_model(req.model, _model_name)

        if req.stream:
            return StreamingResponse(
                _stream_completions(req, worker, _model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        results = await _run_completions(req, worker, _model_name)
        prompt_tokens = sum(len(worker._tokenize(p)) for p in req.prompts)
        completion_tokens = sum(len(worker._tokenize(r.output_text)) for r in results)

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
    async def chat_completions(req: ChatCompletionRequest):
        _validate_model(req.model, _model_name)

        prompt = req.to_prompt(worker.engine.tokenizer)

        if req.stream:
            return StreamingResponse(
                _stream_chat(req, prompt, worker, _model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await _run_single(prompt, req.max_tokens, req.temperature,
                                   req.do_sample, worker, _model_name)

        prompt_tokens = len(worker._tokenize(prompt))
        completion_tokens = len(worker._tokenize(result.output_text))

        return ChatCompletionResponse(
            model=_model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.output_text),
                    finish_reason="stop" if result.generated_tokens < req.max_tokens else "length",
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
            error_chunk = json.dumps({"error": str(exc)})
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
        yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


async def _stream_chat(
    req: ChatCompletionRequest,
    prompt: str,
    worker: EngineWorker,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """SSE generator for /v1/chat/completions with stream=True."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

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
        error_chunk = json.dumps({"error": str(exc)})
        yield f"data: {error_chunk}\n\n"
        return

    # Role delta (first chunk)
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

    # Content delta
    content_chunk = ChatCompletionChunk(
        id=request_id,
        model=model_name,
        choices=[{
            "index": 0,
            "delta": {"content": result.output_text},
            "finish_reason": None,
        }],
    )
    yield f"data: {content_chunk.model_dump_json()}\n\n"

    # Stop chunk
    stop_chunk = ChatCompletionChunk(
        id=request_id,
        model=model_name,
        choices=[{
            "index": 0,
            "delta": {},
            "finish_reason": "stop" if result.generated_tokens < req.max_tokens else "length",
        }],
    )
    yield f"data: {stop_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
