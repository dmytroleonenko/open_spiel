"""
Async inference batching service used by remote actors.

This module intentionally keeps the transport-agnostic core (batching +
timeout logic) separate from any specific RPC wiring so we can reuse it for
local tests, gRPC servers, or other IPC layers.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, List, Sequence, Tuple


InferFn = Callable[[Sequence[Any]], Awaitable[Sequence[Any]] | Sequence[Any]]


class BatchingInferenceServer:
    """Batches inference requests and runs them through a shared callable."""

    def __init__(
        self,
        infer_fn: InferFn,
        *,
        batch_size: int = 32,
        max_wait_ms: int = 10,
    ):
        """
        Args:
            infer_fn: Callable that accepts a sequence of inputs and returns a
                sequence of outputs of the same length. May be async or sync.
            batch_size: Number of requests to accumulate before forcing a run.
            max_wait_ms: Maximum time to wait before flushing a partial batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self._infer_fn = infer_fn
        self._batch_size = batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._lock = asyncio.Lock()
        self._pending: List[Tuple[Any, asyncio.Future]] = []
        self._timer_task: asyncio.Task | None = None
        self._closed = False

    async def submit(self, payload: Any) -> Any:
        """Queue a request and await the inference result."""
        if self._closed:
            raise RuntimeError("Inference server already shut down")

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async with self._lock:
            self._pending.append((payload, future))
            should_flush = len(self._pending) >= self._batch_size
            should_schedule_timer = (
                not should_flush
                and self._timer_task is None
                and self._max_wait_s > 0
            )

        if should_flush:
            await self._flush_pending()
        elif should_schedule_timer:
            self._timer_task = asyncio.create_task(self._flush_after_timeout())

        return await future

    async def shutdown(self):
        """Flush any remaining requests and mark the server as closed."""
        self._closed = True
        await self._flush_pending()

    async def _flush_after_timeout(self):
        await asyncio.sleep(self._max_wait_s)
        await self._flush_pending()

    async def _flush_pending(self):
        async with self._lock:
            if not self._pending:
                if self._timer_task:
                    self._timer_task.cancel()
                    self._timer_task = None
                return

            pending = self._pending
            self._pending = []
            timer = self._timer_task
            self._timer_task = None

        if timer:
            timer.cancel()

        inputs = [payload for payload, _ in pending]
        results = await _maybe_await(self._infer_fn(inputs))

        if len(results) != len(pending):
            raise ValueError(
                "Inference function returned different number of results "
                f"(expected {len(pending)}, got {len(results)})"
            )

        for result, (_, fut) in zip(results, pending):
            if not fut.done():
                fut.set_result(result)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
