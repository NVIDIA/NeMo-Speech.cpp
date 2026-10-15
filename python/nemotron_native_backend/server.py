from __future__ import annotations

import asyncio
import contextlib
import os
import stat
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from .backend import BackendError, SynthesisBackend
from .batcher import ExactBatcher, RequestAborted
from .lifecycle import record
from .manifest import SHA256_RE, WorkerManifest
from .models import SynthesisRequest
from .protocol import (
    ProtocolError,
    message,
    read_frame,
    validate_envelope,
    write_frame,
)

COMPLETED_REQUEST_ID_WINDOW = 4096


class CompletedRequestIdWindow:
    def __init__(self, capacity: int = COMPLETED_REQUEST_ID_WINDOW) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("completed request ID capacity must be a positive integer")
        self._capacity = capacity
        self._ordered: deque[str] = deque()
        self._ids: set[str] = set()

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._ids

    def add(self, request_id: str) -> None:
        if request_id in self._ids:
            return
        if len(self._ordered) == self._capacity:
            self._ids.remove(self._ordered.popleft())
        self._ordered.append(request_id)
        self._ids.add(request_id)


class WorkerServer:
    def __init__(
        self,
        *,
        socket_path: Path,
        manifest: WorkerManifest,
        backend: SynthesisBackend,
        batcher: ExactBatcher,
        max_frame_bytes: int,
    ) -> None:
        self.socket_path = socket_path
        self.manifest = manifest
        self.backend = backend
        self.batcher = batcher
        if type(max_frame_bytes) is not int or not (1024 <= max_frame_bytes <= 64 * 1024 * 1024):
            raise ValueError("max_frame_bytes must be an integer in [1KiB,64MiB]")
        self.max_frame_bytes = max_frame_bytes
        self.worker_instance_id = str(uuid.uuid4())
        self.completed_request_id_capacity = COMPLETED_REQUEST_ID_WINDOW
        self._server: asyncio.AbstractServer | None = None
        self._client_lock = asyncio.Lock()
        self._client_active = False
        self._client_writer: asyncio.StreamWriter | None = None
        self._stopping = False
        self._listener_closed = False
        self._clients_closed = False
        self._socket_unlinked = False
        self._delivery_tasks: set[asyncio.Task[None]] = set()
        self._batcher_shutdown = False
        self._shutdown_complete = False
        self._shutdown_lock = asyncio.Lock()
        self._socket_creation_attempted = False

    async def start(self) -> None:
        if self.backend.identity.enabled_buckets != self.manifest.enabled_buckets:
            raise RuntimeError(
                "backend/manifest enabled bucket mismatch: "
                f"{self.backend.identity.enabled_buckets} != {self.manifest.enabled_buckets}"
            )
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise RuntimeError(f"refusing to replace existing socket path: {self.socket_path}")
        if not self.socket_path.parent.is_dir():
            raise RuntimeError(f"socket parent does not exist: {self.socket_path.parent}")
        self.batcher.start()
        self._socket_creation_attempted = True
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client, path=str(self.socket_path)
            )
            os.chmod(self.socket_path, 0o660)
        except BaseException as error:
            try:
                await self.shutdown()
            except BaseException as cleanup_error:
                error.add_note(f"startup rollback failed: {cleanup_error}")
            raise

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("server is not started")
        async with self._server:
            await self._server.serve_forever()

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
            ready = message(
                "worker_ready",
                worker_instance_id=self.worker_instance_id,
                **self.manifest.worker_ready_fields(),
                cuda_device_name=self.backend.identity.cuda_device_name,
                cuda_compute_capability=list(self.backend.identity.cuda_compute_capability),
            )
            await send(ready)
            while True:
                incoming = validate_envelope(
                    await read_frame(reader, self.max_frame_bytes), from_client=True
                )
                kind = incoming["kind"]
                if kind == "submit":
                    request = SynthesisRequest.from_message(incoming)
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
                        future = self.batcher.submit(request)
                    except BackendError as error:
                        await send(self._failed(request.request_id, error))
                        continue
                    request_ids.add(request.request_id)
                    await send(message("accepted", request_id=request.request_id))
                    task = asyncio.create_task(
                        self._deliver(
                            request.request_id,
                            future,
                            send,
                            request_ids,
                            completed_ids,
                        ),
                        name=f"deliver-{request.request_id}",
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

    async def _deliver(
        self,
        request_id: str,
        future: asyncio.Future[Any],
        send: Any,
        request_ids: set[str],
        completed_ids: CompletedRequestIdWindow,
    ) -> None:
        try:
            result = await future
            await send(
                message(
                    "audio",
                    request_id=request_id,
                    sequence=0,
                    is_final=True,
                    encoding="pcm_f32_le",
                    sample_rate_hz=result.sample_rate_hz,
                    channels=result.channels,
                    samples=result.samples,
                    data=result.pcm_f32_le,
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
            request_ids.discard(request_id)
            completed_ids.add(request_id)

    def _validate_hello(self, value: dict[str, Any]) -> None:
        self._validate_exact_keys(value, {"expected_manifest_sha256", "client_instance_id"})
        expected = value["expected_manifest_sha256"]
        if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
            raise ProtocolError("expected_manifest_sha256 is malformed")
        if expected != self.manifest.sha256:
            raise ProtocolError("client/worker manifest mismatch")
        client_id = value["client_instance_id"]
        if type(client_id) is not str or not client_id or len(client_id.encode("utf-8")) > 1024:
            raise ProtocolError("client_instance_id must contain 1..1024 UTF-8 bytes")

    @staticmethod
    def _validate_ping_nonce(value: Any) -> int:
        if type(value) is not int or not (0 <= value <= 0xFFFFFFFFFFFFFFFF):
            raise ProtocolError("ping nonce must be a uint64 integer")
        return value

    def _delivery_done(self, task: asyncio.Task[None]) -> None:
        self._delivery_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _validate_exact_keys(value: dict[str, Any], fields: set[str]) -> None:
        expected = {"protocol_version", "kind", *fields}
        if set(value) != expected:
            raise ProtocolError(
                f"{value.get('kind')} fields differ: missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )

    @staticmethod
    def _failed(request_id: str, error: BackendError) -> dict[str, Any]:
        return message(
            "failed",
            request_id=request_id,
            code=error.code,
            message=str(error),
            retryable=error.retryable,
        )

    def begin_shutdown(self) -> None:
        self._stopping = True
        if self._server is not None and not self._listener_closed:
            self._server.close()
            self._listener_closed = True
            record("listener_closed")
        if self._client_writer is not None:
            self._client_writer.close()

    async def shutdown(self) -> None:
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            errors: list[BaseException] = []
            self.begin_shutdown()
            if self._server is not None:
                try:
                    await self._server.wait_closed()
                except BaseException as error:
                    errors.append(error)
            if not self._clients_closed:
                self._clients_closed = True
                record("clients_closed")
            for task in tuple(self._delivery_tasks):
                task.cancel()
            if self._delivery_tasks:
                await asyncio.gather(*self._delivery_tasks, return_exceptions=True)
            if not self._batcher_shutdown:
                self._batcher_shutdown = True
                try:
                    await self.batcher.shutdown()
                except BaseException as error:
                    errors.append(error)
            socket_stat = None
            if self._socket_creation_attempted:
                try:
                    socket_stat = self.socket_path.lstat()
                except FileNotFoundError:
                    pass
            if socket_stat is not None:
                if stat.S_ISSOCK(socket_stat.st_mode):
                    try:
                        self.socket_path.unlink()
                    except BaseException as error:
                        errors.append(error)
                else:
                    errors.append(
                        RuntimeError(f"refusing to unlink non-socket path: {self.socket_path}")
                    )
            if not self.socket_path.exists() and not self.socket_path.is_symlink():
                if not self._socket_unlinked:
                    self._socket_unlinked = True
                    record("socket_unlinked")
            self._shutdown_complete = True
            if errors:
                raise RuntimeError("worker shutdown failed: " + "; ".join(map(str, errors)))
