from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass

from .backend import BackendError
from .batcher import QueueFull, RequestAborted
from .execution.native_assets import WIDTHS
from .streaming_batcher import StreamingHandle
from .streaming_contract import OneChunkLookaheadPublisher
from .streaming_models import StreamingAudioChunk, StreamingResult


class InflightHandle(StreamingHandle):
    def __init__(self, request_id, result, chunks, wake):
        super().__init__(request_id, result, chunks)
        self._wake = wake

    async def next_chunk(self):
        try:
            return await super().next_chunk()
        finally:
            self._wake.set()


@dataclass
class PendingRequest:
    request: object
    handle: InflightHandle
    submitted_ns: int
    slot: int | None = None
    epoch: int = 0
    publisher: object = None
    admitted_ns: int | None = None
    first_audio_ns: int | None = None
    chunks: int = 0
    samples: int = 0
    aborted: bool = False


class InflightBatcher:
    def __init__(
        self,
        backend,
        *,
        mode="inflight",
        max_queue_depth=None,
        chunk_queue_depth=4,
        max_active_requests=None,
    ):
        if mode not in ("inflight", "fixed_request") or backend.batch not in WIDTHS:
            raise ValueError("In-flight execution requires a supported physical width")
        self.batch = backend.batch
        if max_active_requests is not None and (
            mode != "inflight"
            or type(max_active_requests) is not int
            or not 1 <= max_active_requests <= self.batch
        ):
            raise ValueError("active admission cap requires inflight mode and 1..physical width")
        self.max_active_requests = max_active_requests
        max_queue_depth = max(128, 4 * self.batch) if max_queue_depth is None else max_queue_depth
        if max_queue_depth < self.batch or chunk_queue_depth < 2:
            raise ValueError("Insufficient bounded request or chunk capacity")
        self.backend = backend
        self.mode = mode
        self._admission_signal = (
            getattr(backend.inflight_engine, "admission_requested", None)
            if mode == "inflight"
            else None
        )
        self.max_queue_depth = max_queue_depth
        self.chunk_queue_depth = chunk_queue_depth
        self._pending = {}
        self._queue = deque()
        self._slots = [None] * self.batch
        self._epochs = [0] * self.batch
        self._retire = []
        self._defer_admission_reset = mode == "inflight" and getattr(
            getattr(backend.inflight_engine, "policy", None), "defer_admission_reset", False
        )
        self._deferred_admissions = {}
        self._deferred_admission_count = 0
        self._wake = asyncio.Event()
        self._runner = None
        self._closed = False
        self._failure = None
        self._shutdown_task = None
        self._readiness_done = False
        self._records = []
        self.on_fatal = None

    @property
    def queue_depth(self):
        return sum(not item.aborted for item in self._queue)

    @property
    def active_batches(self):
        return int(any(item is not None for item in self._slots))

    def start(self):
        if self._runner is not None:
            raise RuntimeError("In-flight controller is already started")
        self._runner = asyncio.create_task(self._run(), name=f"native-b{self.batch}-inflight")

    def submit(self, request):
        if self._closed or self._failure is not None:
            raise BackendError(
                "worker_stopping", "In-flight execution is unavailable", retryable=True
            )
        if request.request_id in self._pending:
            raise BackendError("duplicate_request_id", "Request ID is already pending")
        if not 1 <= len(request.token_ids) <= 512:
            raise BackendError(
                "unsupported_text_length", "Native execution supports 1..512 text tokens"
            )
        if len(self._pending) >= self.max_queue_depth:
            raise QueueFull()
        loop = asyncio.get_running_loop()
        handle = InflightHandle(
            request.request_id,
            loop.create_future(),
            asyncio.Queue(maxsize=self.chunk_queue_depth),
            self._wake,
        )
        item = PendingRequest(request, handle, time.monotonic_ns())
        self._pending[request.request_id] = item
        self._queue.append(item)
        if getattr(getattr(self.backend.inflight_engine, "policy", None), "diagnostics", False):
            self._record(item, "queued")
        if self.max_active_requests is not None:
            self._refresh_admission_signal()
        elif self._admission_signal is not None:
            self._admission_signal.set()
        self._wake.set()
        return handle

    def abort(self, request_id):
        item = self._pending.get(request_id)
        if item is None:
            return False
        item.aborted = True
        self._error(item, RequestAborted())
        if self.max_active_requests is not None:
            self._refresh_admission_signal()
        self._wake.set()
        return True

    def _error(self, item, error):
        while not item.handle._chunks.empty():
            item.handle._chunks.get_nowait()
        item.handle._chunks.put_nowait(error)
        if not item.handle.result.done():
            item.handle.result.set_exception(error)
        if item.slot is None:
            if isinstance(error, RequestAborted) and getattr(
                getattr(self.backend.inflight_engine, "policy", None), "diagnostics", False
            ):
                self._record(item, "cancelled", error=str(error))
            self._pending.pop(item.request.request_id, None)

    def _record(self, item, status, **extra):
        row = dict(
            request_id=item.request.request_id,
            seed=item.request.seed,
            normalized_text=item.request.normalized_text,
            token_count=len(item.request.token_ids),
            mode=self.mode,
            physical_batch=self.batch,
            slot=item.slot,
            epoch=item.epoch,
            submitted_ns=item.submitted_ns,
            admitted_ns=item.admitted_ns,
            first_audio_ns=item.first_audio_ns,
            observed_ns=time.monotonic_ns(),
            chunks=item.chunks,
            samples=item.samples,
            status=status,
            **extra,
        )
        if self.max_active_requests is not None:
            row["max_active_requests"] = self.max_active_requests
        self._records.append(row)

    def _flush_records(self):
        if self._records:
            payload = "".join(
                json.dumps(row, separators=(",", ":")) + "\n" for row in self._records
            )
            with (self.backend.receipt_dir / "inflight-requests.jsonl").open("a") as stream:
                stream.write(payload)
            self._records.clear()
        engine = self.backend.inflight_engine
        if getattr(getattr(engine, "policy", None), "diagnostics", False):
            rows = engine.drain_diagnostics()
            if rows:
                payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
                with (self.backend.receipt_dir / "inflight-diagnostics.jsonl").open("a") as stream:
                    stream.write(payload)

    def _free(self, item):
        self._pending.pop(item.request.request_id, None)
        if item.slot is not None:
            self._slots[item.slot] = None
            self._retire.append((item.slot, item.epoch))
        if self.max_active_requests is not None:
            self._refresh_admission_signal()

    def _publish(self, item, chunk):
        audio = StreamingAudioChunk(
            chunk.request_id, chunk.sequence, chunk.payload, chunk.samples, chunk.is_final
        )
        audio.validate()
        item.handle._chunks.put_nowait(audio)
        item.chunks += 1
        item.samples += chunk.samples
        if chunk.samples and item.first_audio_ns is None:
            item.first_audio_ns = time.monotonic_ns()
        if getattr(getattr(self.backend.inflight_engine, "policy", None), "diagnostics", False):
            self._record(item, "publication_enqueued", chunk_samples=chunk.samples)

    def _consume(self, events):
        for event in events:
            slot = event["slot"]
            if not 0 <= slot < self.batch:
                raise RuntimeError("Execution returned an invalid physical slot")
            item = self._slots[slot]
            if item is None or item.epoch != event["epoch"]:
                continue
            if item.request.request_id != event["request_id"]:
                raise RuntimeError("Execution returned another request's slot ownership")
            if item.aborted:
                continue
            if event["type"] == "chunk":
                chunks = item.publisher.decoded(
                    item.request.request_id, event["pcm_f32_le"], event["samples"]
                )
                for chunk in chunks:
                    self._publish(item, chunk)
            elif event["type"] == "finish":
                if not item.samples:
                    self._error(item, BackendError("empty_audio", "Request ended without audio"))
                    self._record(item, "failed", error="empty_audio")
                    self._free(item)
                    continue
                self._publish(item, item.publisher.finish(item.request.request_id))
                result = StreamingResult(item.request.request_id, item.chunks, item.samples)
                result.validate()
                if not item.handle.result.done():
                    item.handle.result.set_result(result)
                self._record(
                    item,
                    "finished",
                    termination=event.get("reason"),
                    engine_startup_priority_launches=event.get("receipt", {}).get(
                        "engine_startup_priority_launches"
                    ),
                    engine_admission_yields=event.get("receipt", {}).get("engine_admission_yields"),
                )
                self._free(item)
                self._readiness_done = True
            else:
                raise RuntimeError("Unsupported in-flight execution event")

    def _refresh_admission_signal(self):
        if self._admission_signal is not None:
            waiting = any(not item.aborted for item in self._queue)
            owners = sum(item is not None for item in self._slots)
            if waiting and owners < self.max_active_requests:
                self._admission_signal.set()
            else:
                self._admission_signal.clear()

    def _select_admissions(self):
        if not self._queue:
            if self._admission_signal is not None:
                self._admission_signal.clear()
            return []
        self._queue = deque(item for item in self._queue if not item.aborted)
        if not self._queue and self._admission_signal is not None:
            self._admission_signal.clear()
        free = [slot for slot, item in enumerate(self._slots) if item is None]
        if self.mode == "fixed_request":
            if len(free) != self.batch:
                return []
            if self._readiness_done and len(self._queue) < self.batch:
                return []
            if (
                not self._readiness_done
                and len(self._queue) not in (1,)
                and len(self._queue) < self.batch
            ):
                return []
        selected = []
        peers = self.batch - len(free)
        for slot in free:
            if self.max_active_requests is not None and peers >= self.max_active_requests:
                break
            if not self._queue:
                break
            item = self._queue.popleft()
            self._epochs[slot] += 1
            item.slot = slot
            item.epoch = self._epochs[slot]
            item.admitted_ns = time.monotonic_ns()
            item.publisher = OneChunkLookaheadPublisher(
                [item.request.request_id], eager=True, completion_marker=True
            )
            self._slots[slot] = item
            selected.append(item)
            self._record(item, "admitted", peers=peers)
            peers += 1
        if self.max_active_requests is not None:
            self._refresh_admission_signal()
        elif not self._queue and self._admission_signal is not None:
            self._admission_signal.clear()
        return selected

    def _tick_deferred(self, retirements, cancellations, admissions, progress):
        engine = self.backend.inflight_engine
        dropped = set()
        for slot, epoch in cancellations:
            key = (slot, epoch)
            if self._deferred_admissions.pop(key, None) is not None:
                dropped.add(key)
            else:
                engine.cancel(slot, epoch)
        initialized_retirements = []
        for slot, epoch in retirements:
            key = (slot, epoch)
            if key in dropped or self._deferred_admissions.pop(key, None) is not None:
                dropped.add(key)
            else:
                initialized_retirements.append(key)
        if getattr(getattr(engine, "policy", None), "retire_batch", False):
            engine.retire_many(initialized_retirements)
        else:
            for slot, epoch in initialized_retirements:
                engine.retire(slot, epoch)
        pending_slots = {slot for slot, _ in self._deferred_admissions}
        for item in admissions:
            if item.slot in pending_slots:
                raise RuntimeError("physical slot has conflicting deferred admission ownership")
            if self._slots[item.slot] is not item or self._epochs[item.slot] != item.epoch:
                raise RuntimeError("deferred admission lost slot ownership")
            self._deferred_admissions[item.slot, item.epoch] = dict(
                slot=item.slot,
                epoch=item.epoch,
                text=item.request.normalized_text,
                token_ids=tuple(item.request.token_ids),
                seed=item.request.seed,
                request_id=item.request.request_id,
            )
            pending_slots.add(item.slot)
            self._deferred_admission_count += 1
        if not progress or self._closed:
            return []
        if (
            admissions
            and self._deferred_admission_count < self.batch
            and self._admission_signal is not None
            and self._admission_signal.is_set()
            and any(item is None for item in self._slots)
        ):
            return []
        if self._deferred_admissions:
            records = list(self._deferred_admissions.values())
            engine.admit_many(records)
            self._deferred_admissions.clear()
        before = (engine.producer_launches, engine.codec_launches)
        events = engine.advance(progress, allow_admission_yield=False)
        if before != (engine.producer_launches, engine.codec_launches):
            self._deferred_admission_count = 0
        return events

    def _tick(self, retirements, cancellations, admissions, progress):
        if self._defer_admission_reset:
            return self._tick_deferred(retirements, cancellations, admissions, progress)
        engine = self.backend.inflight_engine
        for slot, epoch in cancellations:
            engine.cancel(slot, epoch)
        if getattr(getattr(engine, "policy", None), "retire_batch", False):
            engine.retire_many(retirements)
        else:
            for slot, epoch in retirements:
                engine.retire(slot, epoch)
        if admissions:
            records = [
                dict(
                    slot=item.slot,
                    epoch=item.epoch,
                    text=item.request.normalized_text,
                    token_ids=item.request.token_ids,
                    seed=item.request.seed,
                    request_id=item.request.request_id,
                )
                for item in admissions
            ]
            admit_many = getattr(engine, "admit_many", None)
            if admit_many is not None:
                admit_many(records)
            else:
                for record in records:
                    engine.admit(**record)
        return engine.advance(progress) if progress else []

    def _progress_slots(self):
        return [
            slot
            for slot, item in enumerate(self._slots)
            if item is not None
            and not item.aborted
            and item.handle._chunks.qsize() <= self.chunk_queue_depth - 2
        ]

    async def _run(self):
        loop = asyncio.get_running_loop()
        try:
            while not self._closed:
                self._wake.clear()
                cancellations = []
                for item in self._slots:
                    if item is not None and item.aborted:
                        cancellations.append((item.slot, item.epoch))
                        self._record(item, "cancelled")
                        self._free(item)
                retirements, self._retire = self._retire, []
                admissions = self._select_admissions()
                progress = self._progress_slots()
                if not (retirements or admissions or progress or cancellations):
                    await self._wake.wait()
                    continue
                while admissions and not self._defer_admission_reset:
                    await loop.run_in_executor(
                        self.backend.executor,
                        self._tick,
                        retirements,
                        cancellations,
                        admissions,
                        [],
                    )
                    retirements, cancellations, admissions = [], [], []
                    if not self._closed:
                        await asyncio.sleep(0)
                        admissions = self._select_admissions()
                    progress = [] if self._closed else self._progress_slots()
                events = await loop.run_in_executor(
                    self.backend.executor,
                    self._tick,
                    retirements,
                    cancellations,
                    admissions,
                    progress,
                )
                self._consume(events)
                self._flush_records()
                if events:
                    await asyncio.sleep(0)
        except Exception as error:
            self._failure = error
            self._closed = True
            for item in tuple(self._pending.values()):
                self._error(item, error)
                self._record(item, "failed", error=str(error))
            self._pending.clear()
            if self.on_fatal is not None:
                self.on_fatal()
        finally:
            self._flush_records()
            active = [(slot, item.epoch) for slot, item in enumerate(self._slots) if item]
            if self._failure is None:
                await loop.run_in_executor(
                    self.backend.executor, self._tick, self._retire + active, active, [], []
                )
            self._flush_records()
            self._slots = [None] * self.batch
            self._retire = []
            self._deferred_admissions.clear()
            self._deferred_admission_count = 0

    async def shutdown(self):
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self):
        self._closed = True
        for item in tuple(self._pending.values()):
            item.aborted = True
            self._error(item, RequestAborted())
            if item.slot is not None:
                self._record(item, "cancelled")
        self._wake.set()
        try:
            if self._runner is not None:
                await self._runner
        finally:
            self._pending.clear()
            self._queue.clear()
            await self.backend.shutdown()
