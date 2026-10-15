from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .backend import BackendError, SynthesisBackend, validate_exact16_execution_evidence
from .models import AudioResult, SynthesisRequest
from .receipts import ReceiptWriter


class QueueFull(BackendError):
    def __init__(self) -> None:
        super().__init__("queue_full", "worker request queue is full", retryable=True)


class RequestAborted(BackendError):
    def __init__(self) -> None:
        super().__init__("aborted", "request was aborted", retryable=False)


@dataclass(slots=True)
class _Pending:
    request: SynthesisRequest
    future: asyncio.Future[AudioResult]
    state: str = "queued"
    aborted: bool = False
    forced_error: BaseException | None = None


def exact_partitions(count: int, enabled_buckets: Sequence[int]) -> tuple[list[int], int]:

    if count < 0:
        raise ValueError("count must be nonnegative")
    remainder = count
    result: list[int] = []
    for bucket in sorted({int(value) for value in enabled_buckets}, reverse=True):
        if bucket < 1:
            raise ValueError("bucket sizes must be positive")
        while remainder >= bucket:
            result.append(bucket)
            remainder -= bucket
    return result, remainder


class ExactBatcher:
    def __init__(
        self,
        backend: SynthesisBackend,
        *,
        batch_window_ms: float,
        cohort_timeout_ms: float = 1000.0,
        expected_cohort_size: int | None = None,
        max_queue_depth: int,
        max_inflight_batches: int = 1,
        receipt_writer: ReceiptWriter | None = None,
    ) -> None:
        self.backend = backend
        self.batch_window_seconds = float(batch_window_ms) / 1000.0
        self.cohort_timeout_seconds = float(cohort_timeout_ms) / 1000.0
        self.expected_cohort_size = expected_cohort_size
        if (
            expected_cohort_size is not None
            and expected_cohort_size not in backend.identity.enabled_buckets
        ):
            raise ValueError("expected cohort size has no enabled exact backend plan")
        self.max_queue_depth = int(max_queue_depth)
        self.max_inflight_batches = int(max_inflight_batches)
        if self.max_inflight_batches not in (1, 2):
            raise ValueError("max_inflight_batches must be one or two")
        self.receipt_writer = receipt_writer
        self._queue: deque[_Pending] = deque()
        self._pending: dict[str, _Pending] = {}
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self._dispatch_slots = asyncio.Semaphore(self.max_inflight_batches)
        self._closed = False
        self._active_batches = 0

    def start(self) -> None:
        if self._runner is not None:
            raise RuntimeError("batcher already started")
        self._runner = asyncio.create_task(self._run(), name="nemotron-exact-batcher")

    def submit(self, request: SynthesisRequest) -> asyncio.Future[AudioResult]:
        if self._closed:
            raise BackendError("worker_stopping", "worker is stopping", retryable=True)
        if request.request_id in self._pending:
            raise BackendError("duplicate_request_id", "request ID is already pending")
        if len(self._pending) >= self.max_queue_depth:
            raise QueueFull()
        loop = asyncio.get_running_loop()
        pending = _Pending(request=request, future=loop.create_future())
        self._pending[request.request_id] = pending
        self._queue.append(pending)
        self._wake.set()
        return pending.future

    def abort(self, request_id: str) -> bool:
        pending = self._pending.get(request_id)
        if pending is None:
            return False
        pending.aborted = True
        if pending.state == "queued" and not pending.future.done():
            pending.future.set_exception(RequestAborted())
            self._pending.pop(request_id, None)
        self._wake.set()
        return True

    @property
    def queue_depth(self) -> int:
        return sum(1 for item in self._queue if not item.aborted)

    @property
    def active_batches(self) -> int:
        return self._active_batches

    @property
    def dispatch_task_count(self) -> int:
        return len(self._dispatch_tasks)

    async def _run(self) -> None:
        try:
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
                    if item.aborted:
                        continue
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
            for item in list(self._pending.values()):
                if not item.future.done():
                    item.future.set_exception(error)
            self._pending.clear()
            raise

    def _launch_compatible(self, items: list[_Pending]) -> None:
        partitions, remainder = exact_partitions(len(items), self.backend.identity.enabled_buckets)
        cursor = 0
        for batch in partitions:
            cohort = items[cursor : cursor + batch]
            cursor += batch
            for item in cohort:
                item.state = "scheduled"
            task = asyncio.create_task(
                self._dispatch(cohort),
                name=f"nemotron-exact-dispatch-b{batch}",
            )
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_done)
        if remainder:
            error = BackendError(
                "unsupported_exact_batch",
                f"{remainder} request(s) cannot be represented by enabled exact plans "
                f"{list(self.backend.identity.enabled_buckets)}",
                retryable=True,
            )
            for item in items[cursor:]:
                self._finish_error(item, error)

    def _dispatch_done(self, task: asyncio.Task[None]) -> None:
        self._dispatch_tasks.discard(task)
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
            try:
                if all(item.aborted or item.forced_error is not None for item in cohort):
                    for item in cohort:
                        terminal = self._terminal_error(item)
                        if terminal is None:
                            raise BackendError(
                                "batcher_state", "terminal cohort has no terminal error"
                            )
                        self._finish_error(item, terminal)
                    return
                requests = [item.request for item in cohort]
                results = await self.backend.synthesize_batch(requests)
                evidence = self.backend.consume_execution_evidence(requests, results)
                validate_exact16_execution_evidence(
                    self.backend.identity, requests, results, evidence
                )
                if len(results) != len(cohort):
                    raise BackendError(
                        "backend_contract",
                        f"backend returned {len(results)} results for B{len(cohort)}",
                    )
                by_id = {result.request_id: result for result in results}
                if len(by_id) != len(results) or set(by_id) != {
                    item.request.request_id for item in cohort
                }:
                    raise BackendError("backend_contract", "backend result IDs differ from cohort")
                for result in results:
                    result.validate()
                if self.receipt_writer is not None:
                    await self.receipt_writer.write_success(requests, evidence)
                for item in cohort:
                    terminal = self._terminal_error(item)
                    if terminal is not None:
                        self._finish_error(item, terminal)
                    else:
                        self._finish_result(item, by_id[item.request.request_id])
            except asyncio.CancelledError as error:
                for item in cohort:
                    terminal = self._terminal_error(item, default=error)
                    self._finish_error(item, terminal or error)
                raise
            except Exception as error:
                for item in cohort:
                    terminal = self._terminal_error(item, default=error)
                    if terminal is None:
                        raise BackendError(
                            "batcher_state", "failed cohort has no terminal error"
                        ) from error
                    self._finish_error(item, terminal)
            finally:
                self._active_batches -= 1

    @staticmethod
    def _terminal_error(
        item: _Pending, *, default: BaseException | None = None
    ) -> BaseException | None:
        if item.forced_error is not None:
            return item.forced_error
        if item.aborted:
            return RequestAborted()
        return default

    def _finish_result(self, item: _Pending, result: AudioResult) -> None:
        self._pending.pop(item.request.request_id, None)
        item.state = "done"
        if not item.future.done():
            item.future.set_result(result)

    def _finish_error(self, item: _Pending, error: BaseException) -> None:
        self._pending.pop(item.request.request_id, None)
        item.state = "done"
        if not item.future.done():
            item.future.set_exception(error)

    async def shutdown(self) -> None:
        self._closed = True
        self._wake.set()
        if self._runner is not None:
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
        stopping = BackendError("worker_stopping", "worker stopped", retryable=True)
        for item in list(self._pending.values()):
            item.forced_error = stopping
            if item.state == "queued":
                self._finish_error(item, stopping)
        if self._dispatch_tasks:
            await asyncio.gather(*tuple(self._dispatch_tasks), return_exceptions=True)
        for item in list(self._pending.values()):
            self._finish_error(item, stopping)
        self._pending.clear()
        await self.backend.shutdown()
