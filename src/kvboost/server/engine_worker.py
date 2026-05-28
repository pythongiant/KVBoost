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
from ..oom_planner import (
    OOMPlanner, RequestPlan, RequestTooLargeError,
    gpu_mem_snapshot, format_snapshot,
)
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
        rewarm_text: Optional[str] = None,
        planner: Optional[OOMPlanner] = None,
    ) -> None:
        self.engine = engine
        self.loop = loop  # may be overridden in start() with the running loop
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="kvboost-worker",
        )
        self._release_cache = release_cache_after_request
        self._rewarm_text = rewarm_text
        self.planner = planner

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
            loop.call_soon_threadsafe(token_q.put_nowait, ("token", tok))

        def _run() -> None:
            try:
                effective_prompt = prompt
                plan: Optional[RequestPlan] = None
                if self.planner is not None:
                    prompt_tokens = len(self._tokenize(prompt))
                    plan = self.planner.plan(prompt_tokens, max_new_tokens=max_tokens)
                    if plan.truncated_from is not None:
                        toks = self._tokenize(prompt)[:plan.prompt_tokens]
                        effective_prompt = self.engine.tokenizer.decode(toks)
                    self.planner.log_pre_request(plan)

                def _do_generate():
                    return self.engine.generate(
                        prompt=effective_prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        on_token=_on_token,
                        # Per-call overrides — engine restores its own state
                        # in a try/finally. Planner doesn't mutate anything.
                        prefill_chunk_size=plan.chunk_size if plan is not None else None,
                        kv_cache_bits=plan.kv_bits if plan is not None else None,
                    )

                try:
                    result = self._run_with_oom_logging(_do_generate, request_id, plan)
                finally:
                    if plan is not None:
                        prompt_tokens = plan.prompt_tokens
                        self.planner.log_post_request(plan, prompt_tokens)

                loop.call_soon_threadsafe(token_q.put_nowait, ("done", result))
            except RequestTooLargeError as exc:
                log.warning("Request %s rejected by planner: %s", request_id, exc)
                loop.call_soon_threadsafe(token_q.put_nowait, ("error", exc))
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

        Logs pre/post GPU memory so operators can quantify how much
        cleanup actually freed — useful for deciding whether the
        `--release-cache-after-request` overhead is worth the headroom.
        """
        if not self._release_cache:
            return
        device = getattr(self.engine, "device", None)
        pre = gpu_mem_snapshot(device)
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
        post = gpu_mem_snapshot(device)
        if pre and post:
            freed = post.get("free_mb", 0) - pre.get("free_mb", 0)
            log.info(
                "Cleanup | pre: %s | post: %s | freed=%.0fMiB",
                format_snapshot(pre), format_snapshot(post), freed,
            )

        if self._rewarm_text:
            try:
                self.engine.warm(self._rewarm_text)
            except Exception as exc:
                log.warning("rewarm after release failed: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_with_oom_logging(self, fn, request_id, plan):
        """Run ``fn()`` and log a full GPU memory snapshot if it OOMs.

        The planner promised the request would fit, so a CUDA OOM here is a
        planner mis-prediction (or an external memory pressure source). We
        capture the live memory state at the moment of failure, attempt an
        emergency ``empty_cache()``, capture the post-cleanup state, then
        re-raise so the request fails cleanly.

        The point of the logging is operator visibility — these events should
        be rare, and when they happen the snapshot tells us *what fragmented*
        and *whether cleanup recovered anything*. That's how we tune the
        planner's safety margin over time.
        """
        import torch
        device = getattr(self.engine, "device", None)
        try:
            return fn()
        except torch.cuda.OutOfMemoryError as exc:
            self._log_oom_event(request_id, exc, plan, device)
            raise
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda oom" in msg:
                self._log_oom_event(request_id, exc, plan, device)
            raise

    def _log_oom_event(self, request_id, exc, plan, device) -> None:
        """Emit the OOM telemetry the operator needs to debug a slip-through.

        Three sections of state get logged: the GPU memory at the moment of
        failure, the OOM error text itself (PyTorch's message has the tried-
        alloc / reserved / free breakdown), and what the planner had
        committed to. We then attempt one emergency ``empty_cache()`` and log
        the post-cleanup snapshot so the operator can see whether the
        allocator was holding fragmented blocks vs truly out of memory.
        """
        import gc, torch
        pre = gpu_mem_snapshot(device)
        log.error(
            "OOM slipped past planner for %s | mem-at-failure: %s | error: %s",
            request_id, format_snapshot(pre), str(exc).splitlines()[0],
        )
        if plan is not None:
            log.error("  plan was: %s", plan)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.debug("emergency empty_cache failed: %s", e)
        post = gpu_mem_snapshot(device)
        if pre and post:
            freed = post.get("free_mb", 0) - pre.get("free_mb", 0)
            log.error(
                "  post-cleanup: %s | freed=%.0fMiB %s",
                format_snapshot(post), freed,
                "(fragmentation, not true OOM)" if freed > 100 else "(true OOM — request was too big)",
            )

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
        """Runs in the worker thread (blocking).

        Non-streaming. The planner sees the longest prompt in the batch
        and picks one configuration for the entire dispatch — batch
        members share KV bits and prefill chunk size for the duration.
        """
        t0 = time.perf_counter()
        reqs = batch.requests

        # Re-bind prompts; planner may truncate them in place for this batch.
        prompts: List[str] = [r.prompt for r in reqs]

        # Plan against the largest prompt — that's the one driving peak memory.
        plan: Optional[RequestPlan] = None
        if self.planner is not None:
            token_lengths = [len(self._tokenize(p)) for p in prompts]
            max_prompt_tokens = max(token_lengths)
            max_new = max(r.max_tokens for r in reqs)
            plan = self.planner.plan(max_prompt_tokens, max_new_tokens=max_new)
            if plan.truncated_from is not None:
                # Truncate every prompt in the batch to the planned cap.
                # Simpler than per-request planning and preserves batch-
                # invariant config across members.
                cap = plan.prompt_tokens
                for i, (p, n) in enumerate(zip(prompts, token_lengths)):
                    if n > cap:
                        toks = self._tokenize(p)[:cap]
                        prompts[i] = self.engine.tokenizer.decode(toks)
            self.planner.log_pre_request(plan)

        # Build the actual generate calls with explicit per-call overrides.
        plan_chunk = plan.chunk_size if plan is not None else None
        plan_bits = plan.kv_bits if plan is not None else None

        def _singleton():
            r = reqs[0]
            result = self.engine.generate(
                prompt=prompts[0],
                max_new_tokens=r.max_tokens,
                temperature=r.temperature,
                do_sample=r.do_sample,
                prefill_chunk_size=plan_chunk,
                kv_cache_bits=plan_bits,
            )
            log.debug(
                "Singleton generate: req=%s ttft=%.0fms",
                r.request_id, result.ttft_ms,
            )
            return [result]

        def _batched():
            max_tokens = max(r.max_tokens for r in reqs)
            temperature = reqs[0].temperature
            do_sample = reqs[0].do_sample
            return self.engine.generate_batch(
                prompts=prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                prefill_chunk_size=plan_chunk,
                kv_cache_bits=plan_bits,
            )

        try:
            target = _singleton if len(reqs) == 1 else _batched
            batch_id = reqs[0].request_id if len(reqs) == 1 else f"batch[{len(reqs)}]"
            try:
                results = self._run_with_oom_logging(target, batch_id, plan)
            finally:
                if plan is not None:
                    self.planner.log_post_request(plan, plan.prompt_tokens)
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
        out = {
            "model": self._model_name,
            "device": self.engine.device,
            "queue": self.queue.stats(),
            "cache": cache_stats,
        }
        spec_stats = self.engine.speculative_stats()
        if spec_stats:
            out["speculative"] = spec_stats
        if self.planner is not None:
            out["planner"] = self.planner.snapshot()
        return out
