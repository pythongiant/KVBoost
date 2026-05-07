"""
EngineWorker
============
Runs the KVBoost InferenceEngine in a dedicated background thread and
exposes an async interface so FastAPI handlers can await results without
blocking the event loop.

Architecture
------------
- The engine (model + tokenizer + KVCacheManager) lives in one thread.
  PyTorch is not async-safe; all model.forward() calls must happen in
  the same OS thread to avoid CUDA context conflicts.
- FastAPI handlers submit work via asyncio Futures and await them.
  The worker thread resolves the futures from the executor thread using
  loop.call_soon_threadsafe().
- Streaming responses are handled via asyncio.Queue: the worker pushes
  tokens as they are generated; the handler reads from the queue and
  yields SSE events.

Batching
--------
The worker's dispatch() coroutine receives a Batch (from BatchQueue) and
calls engine.generate_batch() when the batch has >1 request, or
engine.generate() for singletons.  Results are mapped back to per-request
futures.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from ..engine import InferenceEngine, GenerationResult
from ..batch import group_by_prefix
from .batch_queue import Batch, BatchQueue, QueuedRequest
from .schema import PendingRequest

log = logging.getLogger(__name__)


class EngineWorker:
    """
    Wraps a KVBoost InferenceEngine for async use from FastAPI.

    Parameters
    ----------
    engine      : a fully initialised InferenceEngine (or subclass)
    max_workers : thread-pool size (default 1 — model is not thread-safe)
    batch_window_ms  : collection window for the BatchQueue
    max_batch_size   : max requests per batch dispatch
    max_queue_size   : queue capacity before 503

    The event loop is captured automatically when ``start()`` is awaited,
    so the worker binds to whichever loop FastAPI/uvicorn is actually
    running on.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        max_workers: int = 1,
        batch_window_ms: float = 20.0,
        max_batch_size: int = 8,
        max_queue_size: int = 256,
        release_cache_after_request: bool = False,
    ) -> None:
        self.engine = engine
        self.loop = loop  # may be overridden in start() with the running loop
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="kvboost-worker",
        )
        self._release_cache = release_cache_after_request

        self.queue = BatchQueue(
            tokenize_fn=self._tokenize,
            prefix_key_fn=self._prefix_key,
            dispatch_fn=self._dispatch_batch,
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
            max_queue_size=max_queue_size,
        )

        self._model_name = getattr(engine.model.config, "_name_or_path", "kvboost-model")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        # Bind to the actual running loop (uvicorn creates its own when
        # started with loop="none"). Doing this lazily avoids cross-loop
        # Future errors if a stale loop was passed at construction time.
        self.loop = asyncio.get_running_loop()
        await self.queue.start()
        log.info("EngineWorker started (model=%s)", self._model_name)

    async def stop(self) -> None:
        await self.queue.stop()
        self._executor.shutdown(wait=False)
        log.info("EngineWorker stopped.")

    # ── Public async API ──────────────────────────────────────────────────────

    async def generate(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        do_sample: bool,
        stream: bool,
        model_name: str,
        timeout_s: float = 600.0,
    ) -> GenerationResult:
        """
        Submit a single generation request.  Returns when the result is ready.
        Raises asyncio.TimeoutError if not completed within timeout_s.
        """
        fut = await self.queue.enqueue(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            stream=stream,
            model_name=model_name,
        )
        return await asyncio.wait_for(fut, timeout=timeout_s)

    async def stream_generate(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        do_sample: bool,
        model_name: str,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Token-by-token streaming. Bypasses the batch queue — streaming
        requests run as singletons through the executor.

        Yields ("token", token_id) per generated token, then exactly one
        ("done", GenerationResult) on success, or ("error", Exception) on
        failure. Consumers should stop iterating after the terminal event.
        """
        token_q: asyncio.Queue = asyncio.Queue()
        loop = self.loop

        def _on_token(tok: int) -> None:
            # Called from the worker thread — hand off to the loop.
            loop.call_soon_threadsafe(token_q.put_nowait, ("token", tok))

        def _run() -> None:
            try:
                result = self.engine.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    on_token=_on_token,
                )
                loop.call_soon_threadsafe(token_q.put_nowait, ("done", result))
            except Exception as exc:
                log.exception("Streaming generation failed for %s", request_id)
                loop.call_soon_threadsafe(token_q.put_nowait, ("error", exc))
            finally:
                self._release_gpu_memory()

        loop.run_in_executor(self._executor, _run)

        while True:
            kind, payload = await token_q.get()
            yield kind, payload
            if kind in ("done", "error"):
                return

    async def warm(self, text: str) -> None:
        """Warm the KV cache with a prefix string (runs in worker thread)."""
        await self.loop.run_in_executor(self._executor, self.engine.warm, text)

    def _release_gpu_memory(self) -> None:
        """
        Drop the KV cache and return CUDA blocks to free memory between requests.

        No-op unless `release_cache_after_request=True` was passed at
        construction (CLI: --release-cache-after-request). Resets the
        chunk cache (so request N+1 starts cold) and then runs
        gc.collect() + torch.cuda.empty_cache() to actually return the
        freed tensors to the allocator. Useful on 8 GB-class GPUs where
        a populated cache + activations otherwise OOMs request N+1.
        Trades cache reuse for headroom; skip it on bigger GPUs.
        """
        if not self._release_cache:
            return
        try:
            self.engine.reset_cache()
        except Exception as exc:
            log.debug("reset_cache failed: %s", exc)
        try:
            import torch
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as exc:
            log.debug("empty_cache failed: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tokenize(self, prompt: str) -> List[int]:
        return self.engine.tokenizer.encode(prompt, add_special_tokens=False)

    def _prefix_key(self, token_ids: List[int]) -> str:
        from ..batch import group_by_prefix
        from ..models import content_hash_from_tokens
        chunk_size = self.engine.chunk_registry.chunk_size
        n_chunks = 3
        end = min(len(token_ids), n_chunks * chunk_size)
        prefix = token_ids[:end]
        return content_hash_from_tokens(prefix) if prefix else "empty"

    async def _dispatch_batch(self, batch: Batch) -> None:
        """
        Called by BatchQueue for each collected batch.
        Runs the engine in the thread pool and resolves per-request futures.
        """
        try:
            results = await self.loop.run_in_executor(
                self._executor, self._run_batch, batch
            )
            for req, result in zip(batch.requests, results):
                self._resolve(req.future, result)
        except Exception as exc:
            log.exception("Batch dispatch failed: %s", exc)
            for req in batch.requests:
                self._reject(req.future, exc)

    def _run_batch(self, batch: Batch) -> List[GenerationResult]:
        """Runs in the worker thread (blocking)."""
        t0 = time.perf_counter()
        reqs = batch.requests

        try:
            if len(reqs) == 1:
                r = reqs[0]
                result = self.engine.generate(
                    prompt=r.prompt,
                    max_new_tokens=r.max_tokens,
                    temperature=r.temperature,
                    do_sample=r.do_sample,
                )
                log.debug(
                    "Singleton generate: req=%s ttft=%.0fms",
                    r.request_id, result.ttft_ms,
                )
                return [result]

            # Batched path — prompts share a prefix (grouped by BatchQueue)
            prompts = [r.prompt for r in reqs]
            max_tokens = max(r.max_tokens for r in reqs)
            temperature = reqs[0].temperature   # use first req's params for batch
            do_sample = reqs[0].do_sample

            results = self.engine.generate_batch(
                prompts=prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        finally:
            self._release_gpu_memory()

        elapsed = (time.perf_counter() - t0) * 1000
        log.debug(
            "Batch generate: size=%d elapsed=%.0fms", len(reqs), elapsed
        )
        return results

    def _resolve(self, future: asyncio.Future, result: Any) -> None:
        if future is None or future.done():
            return
        self.loop.call_soon_threadsafe(future.set_result, result)

    def _reject(self, future: asyncio.Future, exc: Exception) -> None:
        if future is None or future.done():
            return
        self.loop.call_soon_threadsafe(future.set_exception, exc)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        cache_stats = self.engine.cache_manager.stats()
        return {
            "model": self._model_name,
            "device": self.engine.device,
            "queue": self.queue.stats(),
            "cache": cache_stats,
        }
