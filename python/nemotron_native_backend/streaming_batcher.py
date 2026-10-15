from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .backend import BackendError, BackendIdentity
from .batcher import QueueFull, RequestAborted, exact_partitions
from .lifecycle import record
from .streaming_models import (
    StreamingAudioChunk,
    StreamingResult,
    StreamingSynthesisRequest,
)

StreamChunkEmitter = Callable[[StreamingAudioChunk], Awaitable[bool]]


PARTIAL_HOLD_COHORT_DEADLINES = 5.0


PARTIAL_IDLE_SETTLE_FRACTION = 1.0 / 20.0


class StreamingSynthesisBackend(Protocol):
    @property
    def identity(self) -> BackendIdentity: ...

    async def stream_batch(
        self,
        requests: Sequence[StreamingSynthesisRequest],
        emit_chunk: StreamChunkEmitter,
    ) -> list[StreamingResult]: ...

    async def shutdown(self) -> None: ...


@dataclass(slots=True)
class StreamingHandle:
    request_id: str
    result: asyncio.Future[StreamingResult]
    _chunks: asyncio.Queue[Any]

    async def next_chunk(self) -> StreamingAudioChunk:
        value = await self._chunks.get()
        if isinstance(value, BaseException):
            raise value
        if not isinstance(value, StreamingAudioChunk):
            raise RuntimeError("streaming batcher queue ABI drift")
        return value


@dataclass(slots=True)
class _Pending:
    request: StreamingSynthesisRequest
    handle: StreamingHandle
    enqueued_at: float = 0.0
    state: str = "queued"
    aborted: bool = False
    forced_error: BaseException | None = None
    next_sequence: int = 0
    chunk_count: int = 0
    total_samples: int = 0
    final_emitted: bool = False
    client_terminal_enqueued: bool = False
    terminal_error_enqueued: bool = False
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)


class StreamingExactBatcher:
    def __init__(
        self,
        backend: StreamingSynthesisBackend,
        *,
        batch_window_ms: float,
        cohort_timeout_ms: float = 1000.0,
        expected_cohort_size: int | None = None,
        max_queue_depth: int,
        max_inflight_batches: int = 1,
        chunk_queue_depth: int = 4,
        cohort_hold: bool = True,
        partial_hold_ms: float | None = None,
        partial_idle_settle_ms: float | None = None,
    ) -> None:
        self.backend = backend
        self.batch_window_seconds = float(batch_window_ms) / 1000.0
        self.cohort_timeout_seconds = float(cohort_timeout_ms) / 1000.0
        self.expected_cohort_size = expected_cohort_size
        self.partial_hold_seconds = (
            PARTIAL_HOLD_COHORT_DEADLINES * self.cohort_timeout_seconds
            if partial_hold_ms is None
            else float(partial_hold_ms) / 1000.0
        )
        if self.partial_hold_seconds <= 0.0:
            raise ValueError("partial_hold_ms must be positive")
        self.partial_idle_settle_seconds = (
            PARTIAL_IDLE_SETTLE_FRACTION * self.cohort_timeout_seconds
            if partial_idle_settle_ms is None
            else float(partial_idle_settle_ms) / 1000.0
        )
        if self.partial_idle_settle_seconds < 0.0:
            raise ValueError("partial_idle_settle_ms must be nonnegative")
        if (
            expected_cohort_size is not None
            and expected_cohort_size not in backend.identity.enabled_buckets
        ):
            raise ValueError("expected cohort size has no enabled exact backend plan")
        self.max_queue_depth = int(max_queue_depth)
        self.max_inflight_batches = int(max_inflight_batches)
        if self.max_inflight_batches not in (1, 2):
            raise ValueError("max_inflight_batches must be one or two")
        self.chunk_queue_depth = int(chunk_queue_depth)
        if self.chunk_queue_depth < 1:
            raise ValueError("chunk_queue_depth must be positive")
        self._queue: deque[_Pending] = deque()
        self._pending: dict[str, _Pending] = {}
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self._dispatch_slots = asyncio.Semaphore(self.max_inflight_batches)
        self._closed = False
        self._active_batches = 0

        self._committed_batches = 0

        self._last_completion_at = 0.0

        self.cohort_hold = bool(cohort_hold)
        self._holding = (
            self.cohort_hold
            and self.expected_cohort_size is not None
            and not self.batch_window_seconds
        )

    def start(self) -> None:
        if self._runner is not None:
            raise RuntimeError("streaming batcher already started")
        self._last_completion_at = asyncio.get_running_loop().time()
        self._runner = asyncio.create_task(self._run(), name="nemotron-streaming-exact-batcher")

    def submit(self, request: StreamingSynthesisRequest) -> StreamingHandle:
        if self._closed:
            raise BackendError("worker_stopping", "worker is stopping", retryable=True)
        if request.request_id in self._pending:
            raise BackendError("duplicate_request_id", "request ID is already pending")
        if len(self._pending) >= self.max_queue_depth:
            raise QueueFull()
        loop = asyncio.get_running_loop()
        handle = StreamingHandle(
            request_id=request.request_id,
            result=loop.create_future(),
            _chunks=asyncio.Queue(maxsize=self.chunk_queue_depth),
        )
        pending = _Pending(request=request, handle=handle, enqueued_at=loop.time())
        self._pending[request.request_id] = pending
        self._queue.append(pending)
        self._wake.set()
        return handle

    def abort(self, request_id: str) -> bool:
        pending = self._pending.get(request_id)
        if pending is None:
            return False
        pending.aborted = True
        pending.abort_event.set()
        if pending.state == "queued":
            self._finish_queued_error(pending, RequestAborted())
        elif pending.state in {"scheduled", "active"}:
            error = RequestAborted()
            pending.forced_error = error
            self._force_terminal_error(pending, error)
        self._wake.set()
        return True

    @property
    def queue_depth(self) -> int:
        return sum(1 for item in self._queue if not item.aborted)

    @property
    def active_batches(self) -> int:
        return self._active_batches

    async def _run(self) -> None:
        try:
            if self._holding:
                await self._run_holding()
                return
            while not self._closed:
                await self._wake.wait()
                self._wake.clear()
                deadline = asyncio.get_running_loop().time() + self.cohort_timeout_seconds
                if self.batch_window_seconds:
                    await asyncio.sleep(self.batch_window_seconds)
                while self.expected_cohort_size is not None:
                    counts: dict[tuple[Any, ...], int] = defaultdict(int)
                    for item in self._queue:
                        if not item.aborted:
                            counts[item.request.compatibility_key] += 1
                    if max(counts.values(), default=0) >= self.expected_cohort_size:
                        break
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                    except TimeoutError:
                        break
                snapshot: list[_Pending] = []
                while self._queue:
                    item = self._queue.popleft()
                    if not item.aborted:
                        snapshot.append(item)
                groups: dict[tuple[Any, ...], list[_Pending]] = defaultdict(list)
                for item in snapshot:
                    groups[item.request.compatibility_key].append(item)
                for compatible in groups.values():
                    self._launch_compatible(compatible)
                if self._queue:
                    self._wake.set()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            for item in tuple(self._pending.values()):
                self._finish_queued_error(item, error)
            raise

    async def _run_holding(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            self._purge_aborted()
            groups = self._grouped_queue()
            if not groups:
                self._wake.clear()
                await self._wake.wait()
                continue
            plan, wait_seconds = self._plan_cohorts(groups, loop.time())
            if plan:
                for cohort in plan:
                    self._launch_cohort(cohort)
                continue
            self._wake.clear()
            if wait_seconds is None:
                await self._wake.wait()
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=wait_seconds)
            except TimeoutError:
                continue

    def _purge_aborted(self) -> None:
        if any(item.aborted for item in self._queue):
            self._queue = deque(item for item in self._queue if not item.aborted)

    def _grouped_queue(self) -> dict[tuple[Any, ...], list[_Pending]]:
        groups: dict[tuple[Any, ...], list[_Pending]] = defaultdict(list)
        for item in self._queue:
            groups[item.request.compatibility_key].append(item)
        return groups

    def _largest_bucket_within(self, depth: int) -> int | None:
        buckets = [
            int(value) for value in self.backend.identity.enabled_buckets if int(value) <= depth
        ]
        return max(buckets) if buckets else None

    def _plan_cohorts(
        self, groups: dict[tuple[Any, ...], list[_Pending]], now: float
    ) -> tuple[list[list[_Pending]], float | None]:
        expected = self.expected_cohort_size
        assert expected is not None
        plan: list[list[_Pending]] = []
        committed = self._committed_batches
        waits: list[float] = []
        stalled_for = now - self._last_completion_at
        backend_is_wedged = stalled_for >= self.partial_hold_seconds

        settled = stalled_for >= self.partial_idle_settle_seconds
        for items in groups.values():
            overdue = now - items[0].enqueued_at - self.cohort_timeout_seconds
            if len(items) < expected and overdue >= 0.0:
                if self._largest_bucket_within(len(items)) is None:
                    self._reject_unrepresentable(items)
                    continue
            if committed >= self.max_inflight_batches:
                continue
            if len(items) >= expected:
                plan.append(items[:expected])
                committed += 1
                continue
            if overdue < 0.0:
                waits.append(-overdue)
                continue

            if committed == 0 and not settled and not backend_is_wedged:
                waits.append(self.partial_idle_settle_seconds - stalled_for)
                continue
            if (committed == 0 and settled) or backend_is_wedged:
                size = self._largest_bucket_within(len(items))
                if size is None:
                    self._reject_unrepresentable(items)
                    continue
                plan.append(items[:size])
                committed += 1
                continue
            waits.append(self.partial_hold_seconds - stalled_for)
        if plan:
            return plan, None
        return plan, min(waits) if waits else None

    def _reject_unrepresentable(self, items: list[_Pending]) -> None:
        error = BackendError(
            "unsupported_exact_batch",
            f"{len(items)} request(s) cannot be represented by enabled exact plans "
            f"{list(self.backend.identity.enabled_buckets)}",
            retryable=True,
        )
        chosen = {id(item) for item in items}
        self._queue = deque(item for item in self._queue if id(item) not in chosen)
        for item in items:
            self._finish_queued_error(item, error)

    def _launch_cohort(self, cohort: list[_Pending]) -> None:
        chosen = {id(item) for item in cohort}
        self._queue = deque(item for item in self._queue if id(item) not in chosen)
        for item in cohort:
            item.state = "scheduled"
        self._start_dispatch(cohort)

    def _start_dispatch(self, cohort: list[_Pending]) -> None:
        self._committed_batches += 1
        task = asyncio.create_task(
            self._dispatch(cohort),
            name=f"nemotron-streaming-dispatch-b{len(cohort)}",
        )
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_done)

    def _launch_compatible(self, items: list[_Pending]) -> None:
        partitions, remainder = exact_partitions(len(items), self.backend.identity.enabled_buckets)
        cursor = 0
        for batch in partitions:
            cohort = items[cursor : cursor + batch]
            cursor += batch
            for item in cohort:
                item.state = "scheduled"
            self._start_dispatch(cohort)
        if remainder:
            error = BackendError(
                "unsupported_exact_batch",
                f"{remainder} request(s) cannot be represented by enabled exact plans "
                f"{list(self.backend.identity.enabled_buckets)}",
                retryable=True,
            )
            for item in items[cursor:]:
                self._finish_queued_error(item, error)

    def _dispatch_done(self, task: asyncio.Task[None]) -> None:
        self._dispatch_tasks.discard(task)
        self._committed_batches -= 1
        self._last_completion_at = asyncio.get_running_loop().time()
        if self._holding:
            self._wake.set()
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and self._runner is not None and not self._runner.done():
            self._runner.cancel(error)

    async def _dispatch(self, cohort: list[_Pending]) -> None:
        async with self._dispatch_slots:
            for item in cohort:
                item.state = "active"
            self._active_batches += 1
            by_id = {item.request.request_id: item for item in cohort}

            async def emit(chunk: StreamingAudioChunk) -> bool:
                chunk.validate()
                item = by_id.get(chunk.request_id)
                if item is None:
                    raise BackendError(
                        "backend_contract", "streaming chunk request ID differs from cohort"
                    )
                if item.final_emitted:
                    raise BackendError(
                        "backend_contract", "streaming backend emitted after final chunk"
                    )
                if chunk.sequence != item.next_sequence:
                    raise BackendError(
                        "backend_contract",
                        f"streaming sequence {chunk.sequence} != {item.next_sequence}",
                    )
                item.next_sequence += 1
                item.chunk_count += 1
                item.total_samples += chunk.samples
                item.final_emitted = chunk.is_final
                return await self._enqueue_audio_or_abort(item, chunk)

            try:
                if all(item.aborted or item.forced_error is not None for item in cohort):
                    for item in cohort:
                        await self._finish_active_error(item, item.forced_error or RequestAborted())
                    return
                results = await self.backend.stream_batch(
                    tuple(item.request for item in cohort), emit
                )
                if len(results) != len(cohort):
                    raise BackendError("backend_contract", "streaming backend result count drift")
                result_by_id = {result.request_id: result for result in results}
                if len(result_by_id) != len(results) or set(result_by_id) != set(by_id):
                    raise BackendError("backend_contract", "streaming backend result IDs differ")
                for item in cohort:
                    result = result_by_id[item.request.request_id]
                    result.validate()
                    if not item.final_emitted:
                        raise BackendError(
                            "backend_contract", "streaming backend completed without final audio"
                        )
                    if (
                        result.chunk_count != item.chunk_count
                        or result.total_samples != item.total_samples
                    ):
                        raise BackendError("backend_contract", "streaming result accounting drift")
                    if item.aborted:
                        await self._finish_active_error(item, item.forced_error or RequestAborted())
                    else:
                        self._pending.pop(item.request.request_id, None)
                        item.state = "done"
                        if not item.handle.result.done():
                            item.handle.result.set_result(result)
            except asyncio.CancelledError as error:
                for item in cohort:
                    await self._finish_active_error(item, error)
                raise
            except Exception as error:
                for item in cohort:
                    await self._finish_active_error(item, error)
            finally:
                self._active_batches -= 1

    def _finish_queued_error(self, item: _Pending, error: BaseException) -> None:
        self._pending.pop(item.request.request_id, None)
        item.state = "done"
        if not item.handle.result.done():
            item.handle.result.set_exception(error)
        self._force_terminal_error(item, error)

    def _force_terminal_error(self, item: _Pending, error: BaseException) -> None:
        if item.terminal_error_enqueued:
            return
        while True:
            try:
                item.handle._chunks.get_nowait()
            except asyncio.QueueEmpty:
                break
        item.handle._chunks.put_nowait(error)
        item.terminal_error_enqueued = True
        item.client_terminal_enqueued = False

    async def _enqueue_audio_or_abort(self, item: _Pending, chunk: StreamingAudioChunk) -> bool:
        if item.aborted:
            return False
        try:
            item.handle._chunks.put_nowait(chunk)
        except asyncio.QueueFull:
            put_task = asyncio.create_task(item.handle._chunks.put(chunk))
            abort_task = asyncio.create_task(item.abort_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {put_task, abort_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_task in done or item.aborted:
                    put_task.cancel()
                    await asyncio.gather(put_task, return_exceptions=True)
                    self._force_terminal_error(item, item.forced_error or RequestAborted())
                    return False
                abort_task.cancel()
                await asyncio.gather(abort_task, return_exceptions=True)
                put_task.result()
            finally:
                for task in (put_task, abort_task):
                    if not task.done():
                        task.cancel()
        if item.aborted:
            self._force_terminal_error(item, item.forced_error or RequestAborted())
            return False
        if chunk.is_final:
            item.client_terminal_enqueued = True
        return True

    async def _finish_active_error(self, item: _Pending, error: BaseException) -> None:
        self._pending.pop(item.request.request_id, None)
        item.state = "done"
        if not item.handle.result.done():
            item.handle.result.set_exception(error)
        self._force_terminal_error(item, error)

    async def shutdown(self) -> None:
        record("batcher_drain_begin")
        self._closed = True
        self._wake.set()
        if self._runner is not None:
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
        stopping = BackendError("worker_stopping", "worker stopped", retryable=True)
        for item in tuple(self._pending.values()):
            item.forced_error = stopping
            item.aborted = True
            item.abort_event.set()
            if item.state == "queued":
                self._finish_queued_error(item, stopping)
            elif item.state in {"scheduled", "active"}:
                self._force_terminal_error(item, stopping)
        if self._dispatch_tasks:
            await asyncio.gather(*tuple(self._dispatch_tasks), return_exceptions=True)
        for item in tuple(self._pending.values()):
            await self._finish_active_error(item, stopping)
        self._pending.clear()
        record("batcher_drain_end")
        await self.backend.shutdown()


__all__ = [
    "StreamChunkEmitter",
    "StreamingExactBatcher",
    "StreamingHandle",
    "StreamingSynthesisBackend",
]
