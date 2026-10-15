import asyncio
import contextlib
import json
from collections import deque
from pathlib import Path
from typing import Any

from ..backend import BackendError
from ..initial_publication import InitialAudioHold, InitialHoldBudget, initial_hold_seconds
from ..protocol import ProtocolError, message, read_frame, validate_envelope, write_frame
from ..server import CompletedRequestIdWindow
from ..streaming_models import StreamingSynthesisRequest
from . import first_fence
from .ready_writer import ReadyWriter, ack_records
from .slab_audit import SlabAudit
from .slab_cancel_v2 import abort_response
from .slab_pool import SlabPool


class SlabServerMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initial_hold_seconds = initial_hold_seconds()
        if self._initial_hold_seconds and (
            self.native_host_options != {"f5": True} or self.native_dma is not None
        ):
            raise RuntimeError("initial publication hold requires exact F5 without DMA")

        def identify(value):
            return dict(module=value.__module__, qualname=value.__qualname__)

        binding = dict(
            class_identity=identify(type(self)),
            mro=[identify(value) for value in type(self).__mro__],
            handler=identify(type(self)._handle_client),
            initializer=identify(type(self).__init__),
            hold_initializer=identify(SlabServerMixin.__init__),
            transport_options=dict(self.native_host_options),
            dma_is_none=self.native_dma is None,
        )
        self._initial_hold_budget = InitialHoldBudget(
            binding=binding, configured_hold_ms=int(self._initial_hold_seconds * 1000)
        )

    async def _deliver_stream(self, handle, send, request_ids, completed_ids, dispatched=None):
        try:
            result = await super()._deliver_stream(
                handle, send, request_ids, completed_ids, dispatched
            )
            if isinstance(handle, InitialAudioHold):
                while handle.terminal_sent and handle.late_abort_count:
                    handle.late_abort_count -= 1
                    await send(message("aborted", request_id=handle.request_id))
            return result
        finally:
            if isinstance(handle, InitialAudioHold):
                await handle.close()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        async with self._client_lock:
            if self._stopping or self._client_active:
                writer.close()
                await writer.wait_closed()
                return
            self._client_active = True
            self._client_writer = writer
        pool = None
        audit = None
        send_lock = asyncio.Lock()
        held_requests = {}
        held_tasks = set()
        request_ids: set[str] = set()
        completed_ids = CompletedRequestIdWindow(self.completed_request_id_capacity)
        first_leases = {}
        config = self.native_host_options
        queued_input = deque()
        input_batches = 0
        input_records = 0
        maximum_input_group = 0
        ready_writer = None
        dma = self.native_dma

        async def write(value):
            self.native_trace.event(
                "descriptor_write_begin", request_id=value.get("request_id"), kind=value["kind"]
            )
            await write_frame(writer, value, self.max_frame_bytes)
            self.native_trace.event(
                "descriptor_write_end", request_id=value.get("request_id"), kind=value["kind"]
            )

        if config.get("f2"):
            ready_writer = ReadyWriter(write)

        async def send_held(value, held):
            held.before_send(value)
            original = value
            lease = None
            advertised = False
            try:
                if value.get("kind") == "audio":
                    payload = value["data"]
                    self.native_trace.event(
                        "slab_publish_begin",
                        request_id=value["request_id"],
                        sequence=value["sequence"],
                        bytes=len(payload),
                    )
                    lease = await held.wait_owned(pool.publish(payload))
                    self.native_trace.event(
                        "slab_publish_end",
                        request_id=value["request_id"],
                        sequence=value["sequence"],
                        slot=lease["slot"],
                        generation=lease["generation"],
                    )
                    value = {key: item for key, item in value.items() if key != "data"}
                    value.update(kind="audio_slab", **lease)
                self.native_trace.event(
                    "descriptor_lock_begin", request_id=value.get("request_id"), kind=value["kind"]
                )
                async with send_lock:
                    held.before_send(value)
                    if lease is not None:
                        if original["sequence"] == 0:
                            first_leases[(lease["arena"], lease["slot"], lease["generation"])] = (
                                original["request_id"]
                            )
                        if audit is not None:
                            audit.audio(original)
                    held.begin_send(value)
                    advertised = True
                    await write(value)
                    held.after_send(value)
            finally:
                if lease is not None and not advertised and not pool.closed:
                    first_leases.pop((lease["arena"], lease["slot"], lease["generation"]), None)
                    pool.acknowledge(lease["arena"], lease["slot"], lease["generation"])
                    pool.counts["locally_reclaimed_unsent"] += 1

        def abort_held(request_id):
            held = held_requests.get(request_id) if isinstance(request_id, str) else None
            if held is not None and held.terminal_started and not held.delivery_task.done():
                held.late_abort_count += 1
                return "KNOWN_TERMINAL_PENDING"

            def abort_owned(ident):
                owned = held_requests.get(ident)
                if owned is not None and owned.terminal_sent:
                    return False
                accepted = owned.abort() if owned is not None else False
                return self.batcher.abort(ident) or accepted

            return abort_response(request_id, completed_ids, abort_owned)

        def held_done(task, handle):
            held_tasks.discard(task)
            if held_requests.get(handle.request_id) is handle:
                held_requests.pop(handle.request_id)

        async def send(value: dict[str, Any]) -> None:
            held = held_requests.get(value.get("request_id"))
            if held is not None and asyncio.current_task() is held.delivery_task:
                return await send_held(value, held)
            if value.get("kind") == "audio":
                payload = value["data"]
                self.native_trace.event(
                    "slab_publish_begin",
                    request_id=value["request_id"],
                    sequence=value["sequence"],
                    bytes=len(payload),
                )
                lease = dma.publish_direct(payload) if dma is not None else None
                if lease is None:
                    lease = await pool.publish(payload)
                self.native_trace.event(
                    "slab_publish_end",
                    request_id=value["request_id"],
                    sequence=value["sequence"],
                    slot=lease["slot"],
                    generation=lease["generation"],
                )
                if value["sequence"] == 0:
                    first_leases[(lease["arena"], lease["slot"], lease["generation"])] = value[
                        "request_id"
                    ]
                if audit is not None:
                    audit.audio(value)
                value = {k: v for k, v in value.items() if k != "data"}
                value.update(kind="audio_slab", **lease)
            if ready_writer is not None:
                await ready_writer.send(value)
                return
            self.native_trace.event(
                "descriptor_lock_begin", request_id=value.get("request_id"), kind=value["kind"]
            )
            async with send_lock:
                await write(value)

        try:
            pool = SlabPool(slots=1024)
            if dma is not None:
                dma.establish(pool, self.backend)
            if self.slab_audit_enabled:
                audit = SlabAudit(
                    Path(self.backend.receipt_dir, pool.arena + "-expected-pcm16.jsonl")
                )
            hello = validate_envelope(
                await read_frame(reader, self.max_frame_bytes), from_client=True
            )
            self._validate_hello(hello)
            await send(
                message(
                    "worker_ready",
                    worker_instance_id=self.worker_instance_id,
                    **self.manifest.worker_ready_fields(),
                    cuda_device_name=self.backend.identity.cuda_device_name,
                    cuda_compute_capability=list(self.backend.identity.cuda_compute_capability),
                )
            )
            while True:
                incoming = (
                    queued_input.popleft()
                    if queued_input
                    else await read_frame(reader, self.max_frame_bytes)
                )
                if isinstance(incoming, dict) and incoming.get("kind") == "submit_batch":
                    if (
                        not config.get("f4")
                        or set(incoming) != {"kind", "protocol_version", "records"}
                        or incoming["protocol_version"] != 1
                    ):
                        raise ProtocolError("invalid submit batch envelope")
                    records = incoming["records"]
                    if (
                        not isinstance(records, list)
                        or not 1 <= len(records) <= 64
                        or any(
                            not isinstance(item, dict) or item.get("kind") != "submit"
                            for item in records
                        )
                    ):
                        raise ProtocolError("invalid submit batch records")
                    queued_input.extend(records)
                    input_batches += 1
                    input_records += len(records)
                    maximum_input_group = max(maximum_input_group, len(records))
                    continue
                if isinstance(incoming, dict) and incoming.get("kind") in (
                    "audio_ack",
                    "audio_ack_batch",
                ):
                    if incoming["kind"] == "audio_ack_batch" and not config.get("f2"):
                        raise ProtocolError("unregistered ACK batch")
                    try:
                        records = ack_records(incoming)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ProtocolError(str(error)) from error
                    for ack in records:
                        handled = (
                            dma.acknowledge_direct(ack["arena"], ack["slot"], ack["generation"])
                            if dma is not None
                            else False
                        )
                        if not handled:
                            pool.acknowledge(ack["arena"], ack["slot"], ack["generation"])
                        request_id = first_leases.pop(
                            (ack["arena"], ack["slot"], ack["generation"]), None
                        )
                        self.native_trace.event(
                            "slab_ack",
                            request_id=request_id,
                            slot=ack["slot"],
                            generation=ack["generation"],
                        )
                        if request_id is not None:
                            first_fence.acknowledge(request_id)
                    continue
                incoming = validate_envelope(incoming, from_client=True)
                kind = incoming["kind"]
                if kind == "submit":
                    self.native_trace.event("submit_read", request_id=incoming.get("request_id"))
                    request = StreamingSynthesisRequest.from_message(incoming)
                    if request.request_id in request_ids or request.request_id in completed_ids:
                        await send(
                            self._failed(
                                request.request_id,
                                BackendError(
                                    "duplicate_request_id",
                                    "request ID was already used on this connection",
                                ),
                            )
                        )
                        continue
                    try:
                        handle = self.batcher.submit(request)
                    except BackendError as error:
                        await send(self._failed(request.request_id, error))
                        continue
                    request_ids.add(request.request_id)
                    if self._initial_hold_seconds:
                        handle = InitialAudioHold(
                            handle, self._initial_hold_seconds, self._initial_hold_budget
                        )
                        held_requests[request.request_id] = handle
                    self.native_trace.event("submit_enqueued", request_id=request.request_id)
                    lookup = getattr(self.batcher, "r31_dispatch_event", None)
                    dispatched = lookup(request.request_id) if lookup is not None else None
                    if dispatched is None:
                        await send(message("accepted", request_id=request.request_id))
                    task = asyncio.create_task(
                        self._deliver_stream(handle, send, request_ids, completed_ids, dispatched),
                        name=f"deliver-stream-{request.request_id}",
                    )
                    self._delivery_tasks.add(task)
                    task.add_done_callback(self._delivery_done)
                    if self._initial_hold_seconds:
                        handle.delivery_task = task
                        held_tasks.add(task)
                        task.add_done_callback(lambda task, handle=handle: held_done(task, handle))
                elif kind == "abort":
                    self._validate_exact_keys(incoming, {"request_id"})
                    request_id = incoming["request_id"]
                    action = (
                        abort_held(request_id)
                        if self._initial_hold_seconds
                        else abort_response(request_id, completed_ids, self.batcher.abort)
                    )
                    if action != "UNKNOWN_REQUEST":
                        first_fence.retire(request_id)
                    if audit is not None:
                        with Path(
                            self.backend.receipt_dir, pool.arena + "-cancellation.jsonl"
                        ).open("a") as stream:
                            stream.write(
                                json.dumps(dict(request_id=request_id, action=action)) + "\n"
                            )
                    if action == "KNOWN_TERMINAL":
                        await send(message("aborted", request_id=request_id))
                    elif action == "UNKNOWN_REQUEST":
                        await send(
                            self._failed(
                                str(request_id),
                                BackendError("unknown_request", "request is not pending"),
                            )
                        )
                elif kind == "ping":
                    self._validate_exact_keys(incoming, {"nonce"})
                    nonce = self._validate_ping_nonce(incoming["nonce"])
                    await send(
                        message(
                            "pong",
                            nonce=nonce,
                            queue_depth=self.batcher.queue_depth,
                            active_batches=self.batcher.active_batches,
                        )
                    )
                else:
                    raise ProtocolError(f"unexpected post-handshake kind {kind!r}")
        except (asyncio.IncompleteReadError, ConnectionError, BrokenPipeError):
            pass
        finally:
            if self._initial_hold_seconds:
                for request_id in tuple(request_ids):
                    if request_id in held_requests:
                        held_requests[request_id].abort()
                    first_fence.retire(request_id)
                    self.batcher.abort(request_id)
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
                remaining = tuple(held_tasks)
                if remaining:
                    done, pending = await asyncio.wait(remaining, timeout=5.0)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*remaining, return_exceptions=True)
                self._initial_hold_budget.snapshot(self.backend.receipt_dir)
            if ready_writer is not None:
                await ready_writer.close()
            if dma is not None:
                dma.retire_direct()
            for request_id in request_ids:
                first_fence.retire(request_id)
            if audit is not None:
                audit.close()
            if pool is not None:
                pool.close()
                directory = getattr(self.backend, "receipt_dir", None)
                if directory is not None:
                    receipt = pool.receipt()
                    receipt["r21"] = dict(
                        options=config,
                        input_batch_records=input_records,
                        input_batches=input_batches,
                        maximum_input_batch=maximum_input_group,
                        output_frames=ready_writer.frames if ready_writer else None,
                        output_records=ready_writer.records if ready_writer else None,
                        maximum_output_batch=ready_writer.maximum_group if ready_writer else None,
                    )
                    Path(directory, pool.arena + ".json").write_text(
                        json.dumps(receipt, indent=2) + "\n"
                    )
            for request_id in tuple(request_ids):
                self.batcher.abort(request_id)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            async with self._client_lock:
                if self._client_writer is writer:
                    self._client_writer = None
                self._client_active = False


def install_server(base):
    base.NativeServer = type("R20LeasedPCMServer", (SlabServerMixin, base.NativeServer), {})
