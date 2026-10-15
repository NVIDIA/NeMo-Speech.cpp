from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from . import r31
from .backend import BackendError
from .batcher import RequestAborted
from .manifest import WorkerManifest
from .protocol import (
    TWO_PHASE_OUTPUT_CONTRACT,
    ProtocolError,
    message,
    read_frame,
    validate_envelope,
    write_frame,
)
from .server import CompletedRequestIdWindow, WorkerServer
from .streaming_batcher import StreamingExactBatcher, StreamingHandle
from .streaming_models import (
    STREAMING_FLEET_EXECUTION_CONTRACT,
    STREAMING_OUTPUT_CONTRACT,
    StreamingSynthesisRequest,
)


class StreamingWorkerServer(WorkerServer):
    def __init__(
        self,
        *,
        socket_path: Path,
        manifest: WorkerManifest,
        backend: Any,
        batcher: StreamingExactBatcher,
        max_frame_bytes: int,
    ) -> None:
        super().__init__(
            socket_path=socket_path,
            manifest=manifest,
            backend=backend,
            batcher=batcher,
            max_frame_bytes=max_frame_bytes,
        )
        if manifest.raw.get("manifest_status") != ("functional_v2_fleet_streaming_frozen"):
            raise RuntimeError("streaming worker requires the frozen v2 manifest")
        if manifest.worker_ready_fields()["output_contract"] not in (
            STREAMING_OUTPUT_CONTRACT,
            TWO_PHASE_OUTPUT_CONTRACT,
        ):
            raise RuntimeError("streaming worker output contract drift")
        identity = backend.identity
        if (
            not identity.real_weights
            or identity.execution_contract != STREAMING_FLEET_EXECUTION_CONTRACT
        ):
            raise RuntimeError(
                "streaming fleet manifest requires real weights and the v2 streaming contract"
            )

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
        send_lock = asyncio.Lock()
        request_ids: set[str] = set()
        completed_ids = CompletedRequestIdWindow(self.completed_request_id_capacity)

        async def send(value: dict[str, Any]) -> None:
            async with send_lock:
                await write_frame(writer, value, self.max_frame_bytes)

        try:
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
                incoming = validate_envelope(
                    await read_frame(reader, self.max_frame_bytes), from_client=True
                )
                kind = incoming["kind"]
                if kind == "submit":
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
                elif kind == "abort":
                    self._validate_exact_keys(incoming, {"request_id"})
                    request_id = incoming["request_id"]
                    if not isinstance(request_id, str) or not self.batcher.abort(request_id):
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
            for request_id in tuple(request_ids):
                self.batcher.abort(request_id)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            async with self._client_lock:
                if self._client_writer is writer:
                    self._client_writer = None
                self._client_active = False

    async def _deliver_stream(
        self,
        handle: StreamingHandle,
        send: Any,
        request_ids: set[str],
        completed_ids: CompletedRequestIdWindow,
        dispatched: asyncio.Event | None = None,
    ) -> None:
        request_id = handle.request_id
        handle.result.add_done_callback(self._drain_result_exception)
        completed = False
        chunk_count = 0
        total_samples = 0
        try:
            if dispatched is not None:
                waiter = asyncio.ensure_future(dispatched.wait())
                try:
                    await asyncio.wait((waiter, handle.result), return_when=asyncio.FIRST_COMPLETED)
                finally:
                    waiter.cancel()
                if not dispatched.is_set():
                    await handle.result
                    raise BackendError(
                        "backend_contract", "request finished before its cohort launched"
                    )
                await send(message("accepted", request_id=request_id))
            output_contract = self.manifest.worker_ready_fields()["output_contract"]
            if output_contract not in (STREAMING_OUTPUT_CONTRACT, TWO_PHASE_OUTPUT_CONTRACT):
                raise BackendError("backend_contract", "unsupported streaming output contract")
            two_phase = output_contract == TWO_PHASE_OUTPUT_CONTRACT
            marker_allowed = two_phase and r31.enabled("eager_publication")
            final_chunk = None
            audio_records = 0
            while True:
                chunk = await handle.next_chunk()
                chunk.validate()
                marker = marker_allowed and chunk.is_final and chunk.samples == 0
                if two_phase and chunk.samples == 0 and not marker:
                    raise BackendError("backend_contract", "empty PCM is not a completion message")
                if chunk.request_id != request_id or chunk.sequence != chunk_count:
                    raise BackendError(
                        "backend_contract", "streaming audio identity/sequence drift"
                    )
                chunk_count += 1
                total_samples += chunk.samples
                if chunk.is_final:
                    final_chunk = chunk
                    if not two_phase or marker:
                        break
                await send(
                    message(
                        "audio",
                        request_id=request_id,
                        sequence=chunk.sequence,
                        is_final=False,
                        encoding="pcm_f32_le",
                        sample_rate_hz=chunk.sample_rate_hz,
                        channels=chunk.channels,
                        samples=chunk.samples,
                        data=chunk.pcm_f32_le,
                    )
                )
                audio_records += 1
                if chunk.is_final:
                    break
            result = await handle.result
            result.validate()
            if (
                final_chunk is None
                or result.request_id != request_id
                or result.chunk_count != chunk_count
                or result.total_samples != total_samples
            ):
                raise BackendError(
                    "backend_contract",
                    "streaming final/result accounting drift",
                )
            if two_phase:
                await send(
                    message(
                        "stream_complete_v1",
                        request_id=request_id,
                        next_sequence=audio_records,
                        chunk_count=audio_records,
                        total_samples=total_samples,
                    )
                )
            else:
                await send(
                    message(
                        "audio",
                        request_id=request_id,
                        sequence=final_chunk.sequence,
                        is_final=True,
                        encoding="pcm_f32_le",
                        sample_rate_hz=final_chunk.sample_rate_hz,
                        channels=final_chunk.channels,
                        samples=final_chunk.samples,
                        data=final_chunk.pcm_f32_le,
                    )
                )
            completed = True
        except StopAsyncIteration:
            with contextlib.suppress(ConnectionError, BrokenPipeError):
                await send(
                    self._failed(
                        request_id,
                        BackendError(
                            "backend_contract",
                            "stream ended without a final audio chunk",
                        ),
                    )
                )
        except asyncio.CancelledError:
            raise
        except RequestAborted:
            with contextlib.suppress(ConnectionError, BrokenPipeError):
                await send(message("aborted", request_id=request_id))
        except BackendError as error:
            with contextlib.suppress(ConnectionError, BrokenPipeError):
                await send(self._failed(request_id, error))
        except Exception as error:
            with contextlib.suppress(ConnectionError, BrokenPipeError):
                await send(
                    self._failed(
                        request_id,
                        BackendError("backend_internal", str(error), retryable=False),
                    )
                )
        finally:
            if not completed:
                self.batcher.abort(request_id)
            request_ids.discard(request_id)
            completed_ids.add(request_id)

    @staticmethod
    def _drain_result_exception(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()


__all__ = ["StreamingWorkerServer"]
