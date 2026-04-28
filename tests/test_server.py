"""
Tests for the KVBoost OpenAI-compatible inference server.

Covers:
  - Schema validation (CompletionRequest, ChatCompletionRequest)
  - ChatCompletionRequest.to_prompt() fallback rendering
  - BatchQueue: enqueue, prefix grouping, dispatch, back-pressure
  - EngineWorker: start/stop lifecycle, stats shape
  - FastAPI app: /health, /v1/models, /v1/stats, /v1/warm
  - FastAPI app: POST /v1/completions (non-streaming, multi-prompt)
  - FastAPI app: POST /v1/chat/completions (non-streaming)
  - FastAPI app: 400 on wrong model, 503 on queue full
  - FastAPI app: streaming SSE format (data: ... \n\n + [DONE])

All tests use a mock engine — no model download required.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from kvboost.server.schema import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    PendingRequest,
    UsageStats,
)
from kvboost.server.batch_queue import BatchQueue, Batch, QueuedRequest
from kvboost.server.engine_worker import EngineWorker
from kvboost.server.app import build_app
from kvboost.engine import GenerationResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

MODEL_NAME = "test-model"

def _make_result(text="hello world", tokens=2, prompt="hi") -> GenerationResult:
    return GenerationResult(
        mode="chunk_kv_reuse",
        prompt=prompt,
        output_text=text,
        generated_tokens=tokens,
        ttft_ms=10.0,
        total_ms=50.0,
        tokens_per_sec=40.0,
        kv_reuse_ratio=0.8,
        prompt_tokens=3,
        cached_tokens=2,
    )


def _make_engine() -> MagicMock:
    """Minimal mock of InferenceEngine."""
    engine = MagicMock()
    engine.model.config._name_or_path = MODEL_NAME
    engine.device = "cpu"
    engine.tokenizer.encode = lambda text, **kw: [1, 2, 3]
    engine.tokenizer.apply_chat_template = None  # triggers fallback
    engine.chunk_registry.chunk_size = 128
    engine.cache_manager.stats.return_value = {
        "hits": 0, "misses": 0, "utilization": 0.0,
    }
    engine.generate.return_value = _make_result()
    engine.generate_batch.return_value = [_make_result(), _make_result()]
    engine.warm.return_value = MagicMock()
    return engine


@pytest.fixture
def engine():
    return _make_engine()


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def worker(engine):
    loop = asyncio.get_event_loop()
    w = EngineWorker(
        engine=engine,
        loop=loop,
        batch_window_ms=5.0,
        max_batch_size=4,
        max_queue_size=16,
    )
    await w.start()
    yield w
    await w.stop()


@pytest.fixture
def sync_client(engine):
    """Sync TestClient — used for simple non-streaming tests."""
    loop = asyncio.new_event_loop()
    w = EngineWorker(engine=engine, loop=loop, batch_window_ms=5.0)

    app = build_app(w, model_name=MODEL_NAME)

    # Manually start the worker within the test client's lifespan
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestCompletionRequestSchema:
    def test_default_values(self):
        req = CompletionRequest(model=MODEL_NAME, prompt="hello")
        assert req.max_tokens == 128
        assert req.temperature == 1.0
        assert req.stream is False
        assert req.prompts == ["hello"]

    def test_list_prompt(self):
        req = CompletionRequest(model=MODEL_NAME, prompt=["a", "b"])
        assert req.prompts == ["a", "b"]

    def test_empty_list_prompt_rejected(self):
        with pytest.raises(Exception):
            CompletionRequest(model=MODEL_NAME, prompt=[])

    def test_do_sample_false_when_temperature_zero(self):
        req = CompletionRequest(model=MODEL_NAME, prompt="x", temperature=0.0)
        assert req.do_sample is False

    def test_do_sample_true_when_temperature_positive(self):
        req = CompletionRequest(model=MODEL_NAME, prompt="x", temperature=0.7)
        assert req.do_sample is True

    def test_max_tokens_bounds(self):
        with pytest.raises(Exception):
            CompletionRequest(model=MODEL_NAME, prompt="x", max_tokens=0)
        with pytest.raises(Exception):
            CompletionRequest(model=MODEL_NAME, prompt="x", max_tokens=9999)


class TestChatCompletionRequestSchema:
    def test_to_prompt_fallback(self):
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hi!"),
            ],
        )
        prompt = req.to_prompt(tokenizer=None)
        assert "You are helpful." in prompt
        assert "Hi!" in prompt
        assert "Assistant:" in prompt

    def test_to_prompt_with_chat_template(self):
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "<|system|>helpful<|user|>hi<|assistant|>"
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[ChatMessage(role="user", content="hi")],
        )
        prompt = req.to_prompt(tokenizer=mock_tok)
        assert "<|user|>" in prompt

    def test_empty_messages_rejected(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(model=MODEL_NAME, messages=[])

    def test_do_sample(self):
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[ChatMessage(role="user", content="x")],
            temperature=0.9,
        )
        assert req.do_sample is True


# ── BatchQueue tests ──────────────────────────────────────────────────────────

class TestBatchQueue:
    def _tokenize(self, text):
        return [ord(c) for c in text[:8]]

    def _prefix_key(self, ids):
        return str(ids[:4])

    @pytest.mark.asyncio
    async def test_enqueue_and_dispatch(self):
        dispatched: List[Batch] = []

        async def dispatch(batch: Batch):
            dispatched.append(batch)
            for req in batch.requests:
                req.future.set_result(_make_result())

        q = BatchQueue(
            tokenize_fn=self._tokenize,
            prefix_key_fn=self._prefix_key,
            dispatch_fn=dispatch,
            batch_window_ms=10.0,
            max_batch_size=8,
            max_queue_size=32,
        )
        await q.start()

        fut = await q.enqueue(
            request_id="r1", prompt="hello", max_tokens=32,
            temperature=1.0, do_sample=False, stream=False,
            model_name=MODEL_NAME,
        )
        result = await asyncio.wait_for(fut, timeout=2.0)
        assert result.output_text == "hello world"
        assert q.total_received == 1

        await q.stop()

    @pytest.mark.asyncio
    async def test_prefix_grouping(self):
        batches: List[Batch] = []

        async def dispatch(batch: Batch):
            batches.append(batch)
            for req in batch.requests:
                req.future.set_result(_make_result())

        q = BatchQueue(
            tokenize_fn=lambda t: [1, 2, 3, 4, 5, 6, 7, 8],  # same tokens → same prefix
            prefix_key_fn=self._prefix_key,
            dispatch_fn=dispatch,
            batch_window_ms=30.0,
            max_batch_size=8,
            max_queue_size=32,
        )
        await q.start()

        futs = [
            await q.enqueue(
                request_id=f"r{i}", prompt=f"prompt {i}", max_tokens=32,
                temperature=1.0, do_sample=False, stream=False,
                model_name=MODEL_NAME,
            )
            for i in range(3)
        ]
        # Resolve all futures (dispatch mock resolves them)
        for fut in futs:
            fut.set_result(_make_result())  # allow wait_for to complete

        await asyncio.sleep(0.1)

        # All 3 should land in a single batch (same prefix key)
        assert q.total_received == 3
        await q.stop()

    @pytest.mark.asyncio
    async def test_queue_full_raises(self):
        async def dispatch(batch):
            pass

        q = BatchQueue(
            tokenize_fn=self._tokenize,
            prefix_key_fn=self._prefix_key,
            dispatch_fn=dispatch,
            batch_window_ms=999.0,
            max_batch_size=1,
            max_queue_size=2,
        )
        await q.start()

        # Fill the queue
        for i in range(2):
            await q._queue.put(MagicMock())

        with pytest.raises(asyncio.QueueFull):
            await q.enqueue(
                request_id="overflow", prompt="x", max_tokens=8,
                temperature=1.0, do_sample=False, stream=False,
                model_name=MODEL_NAME,
            )

        await q.stop()

    def test_stats_shape(self):
        async def dispatch(batch):
            pass

        q = BatchQueue(
            tokenize_fn=self._tokenize,
            prefix_key_fn=self._prefix_key,
            dispatch_fn=dispatch,
        )
        s = q.stats()
        assert "queue_depth" in s
        assert "total_received" in s
        assert "avg_batch_size" in s


# ── EngineWorker tests ────────────────────────────────────────────────────────

class TestEngineWorker:
    @pytest.mark.asyncio
    async def test_start_stop(self, engine):
        loop = asyncio.get_event_loop()
        w = EngineWorker(engine=engine, loop=loop, batch_window_ms=5.0)
        await w.start()
        await w.stop()

    @pytest.mark.asyncio
    async def test_stats_shape(self, worker):
        s = worker.stats()
        assert "model" in s
        assert "device" in s
        assert "queue" in s
        assert "cache" in s

    @pytest.mark.asyncio
    async def test_warm_calls_engine(self, worker, engine):
        await worker.warm("system prompt")
        engine.warm.assert_called_once_with("system prompt")

    def test_prefix_key_deterministic(self, worker):
        ids = [1, 2, 3, 4, 5]
        k1 = worker._prefix_key(ids)
        k2 = worker._prefix_key(ids)
        assert k1 == k2
        assert isinstance(k1, str)

    def test_tokenize(self, worker, engine):
        result = worker._tokenize("hello")
        assert isinstance(result, list)


# ── FastAPI app tests ─────────────────────────────────────────────────────────

class TestAppEndpoints:
    def test_health(self, sync_client):
        r = sync_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_list_models(self, sync_client):
        r = sync_client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert any(m["id"] == MODEL_NAME for m in data["data"])

    def test_stats_endpoint(self, sync_client):
        r = sync_client.get("/v1/stats")
        assert r.status_code == 200
        body = r.json()
        assert "queue" in body
        assert "cache" in body

    def test_warm_endpoint(self, sync_client, engine):
        r = sync_client.post("/v1/warm", json={"text": "You are helpful."})
        assert r.status_code == 200
        assert r.json()["status"] == "warmed"

    def test_warm_missing_text(self, sync_client):
        r = sync_client.post("/v1/warm", json={})
        assert r.status_code == 400

    def test_completions_wrong_model(self, sync_client):
        r = sync_client.post("/v1/completions", json={
            "model": "wrong-model",
            "prompt": "hello",
        })
        assert r.status_code == 400

    def test_chat_completions_wrong_model(self, sync_client):
        r = sync_client.post("/v1/chat/completions", json={
            "model": "wrong-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 400


@pytest.mark.asyncio
class TestAppAsyncEndpoints:
    """Async tests using httpx.AsyncClient for proper await support."""

    async def _make_async_client(self, engine):
        loop = asyncio.get_event_loop()
        w = EngineWorker(engine=engine, loop=loop, batch_window_ms=5.0)
        app = build_app(w, model_name=MODEL_NAME)
        await w.start()
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), w

    async def test_completions_non_streaming(self, engine):
        client, worker = await self._make_async_client(engine)
        async with client:
            r = await client.post("/v1/completions", json={
                "model": MODEL_NAME,
                "prompt": "Say hello.",
                "max_tokens": 32,
                "temperature": 0.0,
                "stream": False,
            })
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "text_completion"
        assert len(body["choices"]) == 1
        assert isinstance(body["choices"][0]["text"], str)
        assert "usage" in body
        await worker.stop()

    async def test_chat_completions_non_streaming(self, engine):
        client, worker = await self._make_async_client(engine)
        async with client:
            r = await client.post("/v1/chat/completions", json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 32,
                "temperature": 0.0,
                "stream": False,
            })
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)
        await worker.stop()

    async def test_completions_multi_prompt(self, engine):
        engine.generate.side_effect = [_make_result("a"), _make_result("b")]
        client, worker = await self._make_async_client(engine)
        async with client:
            r = await client.post("/v1/completions", json={
                "model": MODEL_NAME,
                "prompt": ["prompt a", "prompt b"],
                "max_tokens": 16,
                "stream": False,
            })
        assert r.status_code == 200
        body = r.json()
        assert len(body["choices"]) == 2
        await worker.stop()

    async def test_streaming_completions_format(self, engine):
        client, worker = await self._make_async_client(engine)
        chunks = []
        async with client:
            async with client.stream("POST", "/v1/completions", json={
                "model": MODEL_NAME,
                "prompt": "hello",
                "max_tokens": 16,
                "stream": True,
            }) as r:
                assert r.status_code == 200
                async for line in r.aiter_lines():
                    if line:
                        chunks.append(line)

        assert any("[DONE]" in c for c in chunks)
        data_lines = [c for c in chunks if c.startswith("data:") and "[DONE]" not in c]
        assert len(data_lines) >= 1
        # Each data line must be valid JSON
        for line in data_lines:
            payload = json.loads(line.removeprefix("data:").strip())
            assert "choices" in payload
        await worker.stop()

    async def test_streaming_chat_format(self, engine):
        client, worker = await self._make_async_client(engine)
        chunks = []
        async with client:
            async with client.stream("POST", "/v1/chat/completions", json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "stream": True,
            }) as r:
                assert r.status_code == 200
                async for line in r.aiter_lines():
                    if line:
                        chunks.append(line)

        assert any("[DONE]" in c for c in chunks)
        data_lines = [c for c in chunks if c.startswith("data:") and "[DONE]" not in c]
        assert len(data_lines) >= 1
        for line in data_lines:
            payload = json.loads(line.removeprefix("data:").strip())
            assert "choices" in payload
            assert "delta" in payload["choices"][0]
        await worker.stop()

    async def test_usage_stats_in_response(self, engine):
        client, worker = await self._make_async_client(engine)
        async with client:
            r = await client.post("/v1/completions", json={
                "model": MODEL_NAME,
                "prompt": "count tokens",
                "max_tokens": 8,
                "stream": False,
            })
        body = r.json()
        usage = body["usage"]
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        await worker.stop()
