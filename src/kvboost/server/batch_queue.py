"""
Async prefix-grouped batch queue.

Design
------
Incoming requests are held in an asyncio.Queue.  A background coroutine
(the collector) drains the queue every `batch_window_ms` milliseconds,
groups requests by their shared prefix hash (using KVBoost's
group_by_prefix), and submits each group as a batch to the EngineWorker
running in a thread pool.

Prefix grouping strategy
-------------------------
1. Tokenize every request as it arrives and compute the hash of its first
   N chunk-aligned prefix chunks (same logic as engine.group_by_prefix).
2. At the end of the collection window, requests with the same prefix hash
   are batched together so the engine can broadcast the shared KV once and
   run a single batched prefill.
3. Requests that don't share a prefix with anyone form singleton batches.

Back-pressure
-------------
If `max_queue_size` requests are pending, new requests are rejected with
HTTP 503.  The caller should retry.

Timeouts
---------
Each request has a `timeout_s` deadline.  If the worker hasn't returned
by then the future is cancelled and the client receives HTTP 504.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    prefix_key: str          # hash of first N prefix chunks
    max_tokens: int
    temperature: float
    do_sample: bool
    stream: bool
    model_name: str
    arrived_at: float = field(default_factory=time.monotonic)
    future: Optional[asyncio.Future] = None  # resolved by worker


@dataclass
class Batch:
    """A group of requests that share a common prefix, ready to dispatch."""
    prefix_key: str
    requests: List[QueuedRequest]

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def prompts(self) -> List[str]:
        return [r.prompt for r in self.requests]


class BatchQueue:
    """
    Asyncio-based request queue with prefix grouping.

    Parameters
    ----------
    tokenize_fn         : callable(prompt) -> List[int]
    prefix_key_fn       : callable(token_ids) -> str  (hash of first N chunks)
    batch_window_ms     : how long to wait before dispatching a collected batch
    max_batch_size      : hard cap on requests per batch
    max_queue_size      : max total pending requests before 503
    dispatch_fn         : called with each Batch; must be a coroutine
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[int]],
        prefix_key_fn: Callable[[List[int]], str],
        dispatch_fn: Callable[[Batch], "Coroutine"],
        batch_window_ms: float = 20.0,
        max_batch_size: int = 8,
        max_queue_size: int = 256,
    ) -> None:
        self._tokenize = tokenize_fn
        self._prefix_key = prefix_key_fn
        self._dispatch = dispatch_fn
        self._window = batch_window_ms / 1000.0
        self._max_batch = max_batch_size
        self._max_queue = max_queue_size

        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._collector_task: Optional[asyncio.Task] = None

        # Stats
        self.total_received = 0
        self.total_batches = 0
        self.total_dispatched = 0
        self.total_rejected = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._collector_task = asyncio.create_task(self._collector_loop())
        log.info(
            "BatchQueue started: window=%.0fms max_batch=%d max_queue=%d",
            self._window * 1000, self._max_batch, self._max_queue,
        )

    async def stop(self) -> None:
        self._running = False
        if self._collector_task:
            self._collector_task.cancel()
            try:
                await self._collector_task
            except asyncio.CancelledError:
                pass
        log.info("BatchQueue stopped.")

    # ── Public API ────────────────────────────────────────────────────────────

    async def enqueue(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        do_sample: bool,
        stream: bool,
        model_name: str,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> asyncio.Future:
        """
        Add a request to the queue.  Returns a Future that resolves to a
        GenerationResult when the request completes.

        Raises asyncio.QueueFull (→ HTTP 503) if the queue is at capacity.
        """
        if self._queue.full():
            self.total_rejected += 1
            raise asyncio.QueueFull("Request queue is full — try again later.")

        # Tokenize inline so we can compute the prefix key immediately.
        # This is fast (CPU tokenisation) and must happen in the event loop.
        token_ids = await asyncio.get_event_loop().run_in_executor(
            None, self._tokenize, prompt
        )
        prefix_key = self._prefix_key(token_ids)

        fut = asyncio.get_event_loop().create_future()

        req = QueuedRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=token_ids,
            prefix_key=prefix_key,
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            stream=stream,
            model_name=model_name,
            future=fut,
        )

        await self._queue.put(req)
        self.total_received += 1
        log.debug("Enqueued request %s (prefix=%s)", request_id, prefix_key[:8])
        return fut

    # ── Collector loop ────────────────────────────────────────────────────────

    async def _collector_loop(self) -> None:
        """
        Wait for the batch window, then drain the queue and group by prefix.
        Dispatches one batch per prefix group.
        """
        while self._running:
            # Wait for at least one request
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Collect everything that arrives within the window
            window_start = time.monotonic()
            collected: List[QueuedRequest] = [first]

            while (time.monotonic() - window_start) < self._window:
                try:
                    req = self._queue.get_nowait()
                    collected.append(req)
                    if len(collected) >= self._max_batch * 8:
                        # Safety valve: don't let the window grow unbounded
                        break
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)

            # Group by prefix key
            groups: Dict[str, List[QueuedRequest]] = {}
            for req in collected:
                groups.setdefault(req.prefix_key, []).append(req)

            # Dispatch each group as a batch, split into max_batch_size chunks
            for prefix_key, reqs in groups.items():
                for chunk_start in range(0, len(reqs), self._max_batch):
                    chunk = reqs[chunk_start: chunk_start + self._max_batch]
                    batch = Batch(prefix_key=prefix_key, requests=chunk)
                    self.total_batches += 1
                    self.total_dispatched += len(chunk)
                    log.debug(
                        "Dispatching batch: prefix=%s size=%d",
                        prefix_key[:8], len(chunk),
                    )
                    asyncio.create_task(self._dispatch(batch))

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "queue_depth": self._queue.qsize(),
            "total_received": self.total_received,
            "total_dispatched": self.total_dispatched,
            "total_batches": self.total_batches,
            "total_rejected": self.total_rejected,
            "avg_batch_size": (
                self.total_dispatched / self.total_batches
                if self.total_batches else 0.0
            ),
        }
