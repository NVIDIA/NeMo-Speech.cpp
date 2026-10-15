# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import collections
import contextlib
import hashlib
import json
import logging
import os
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from nemotron_native_backend import protocol as native_protocol
from nemotron_native_backend.host.serialization import install as install_serialization
from nemotron_native_backend.protocol import message, read_frame, write_frame

from .protocol import audio_frame, control_frame, float_audio_frame, pcm16, read_control
from .slabs import SlabReader

LOG = logging.getLogger(__name__)


@dataclass(eq=False)
class Request:
    request_id: int
    native_id: str
    text: str
    token_count: int = 0
    next_sequence: int = 0
    total_samples: int = 0
    submitted: bool = False
    accepted: bool = False
    canceled: bool = False
    terminal: bool = False
    scheduled: bool = False
    preparation_task: asyncio.Task | None = None
    preparation_started: bool = False
    output: collections.deque = field(default_factory=collections.deque)
    trace: dict = field(default_factory=dict)


class Session:
    def __init__(self, bridge, reader, writer):
        self.bridge = bridge
        self.reader = reader
        self.writer = writer
        self.native_reader = None
        self.native_writer = None
        self.native_lock = asyncio.Lock()
        self.requests = {}
        self.native_requests = {}
        self.retired = collections.deque(maxlen=4096)
        self.ready = asyncio.Queue(maxsize=bridge.args.max_pending_requests + 1)
        self.output_bytes = 0
        self.preparations = set()
        self.running_preparations = set()
        self.background = set()
        self.slabs = SlabReader()
        self.closed = False

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.background.add(task)
        task.add_done_callback(self.background.discard)

        def done(task):
            if not task.cancelled() and task.exception() is not None:
                self.writer.close()
                if self.native_writer is not None:
                    self.native_writer.close()

        task.add_done_callback(done)
        return task

    async def send_native(self, kind, **fields):
        async with self.native_lock:
            if self.native_writer is None or self.closed:
                raise ConnectionError("native connection is unavailable")
            async with asyncio.timeout(self.bridge.args.write_timeout):
                await write_frame(self.native_writer, message(kind, **fields))

    def enqueue(self, request, payload, *, terminal=False):
        if self.closed or request.terminal:
            return False
        if (
            len(request.output) >= self.bridge.args.request_queue_chunks
            or self.output_bytes + len(payload) > self.bridge.args.max_output_bytes
        ):
            return False
        request.output.append((payload, terminal))
        self.output_bytes += len(payload)
        request.terminal = terminal
        if not request.scheduled:
            request.scheduled = True
            self.ready.put_nowait(request)
        return True

    def discard_output(self, request):
        while request.output:
            self.output_bytes -= len(request.output.popleft()[0])

    async def fail(self, request, code, detail, *, abort=True, aborted=False):
        if request.terminal:
            return
        request.canceled = True
        if request.preparation_task is not None and not request.preparation_started:
            request.preparation_task.cancel()
        self.discard_output(request)
        if aborted:
            payload = control_frame("aborted", request_id=request.request_id)
        else:
            payload = control_frame(
                "failed",
                request_id=request.request_id,
                code=code,
                message=detail,
                retryable=code in {"output_backpressure", "worker_stopping", "queue_full"},
            )
        if not self.enqueue(request, payload, terminal=True):
            raise ConnectionError("client output budget exhausted")
        if abort and request.submitted:
            await self.send_native("abort", request_id=request.native_id)

    def retire(self, request):
        self.mark(request, "retired")
        if self.bridge.diagnostics is not None:
            self.bridge.diagnostics.write(
                json.dumps(
                    {
                        "request_id": request.request_id,
                        "native_id": request.native_id,
                        "events_ns": request.trace,
                    }
                )
                + "\n"
            )
        self.requests.pop(request.request_id, None)
        self.native_requests.pop(request.native_id, None)
        self.retired.append(request.native_id)

    def mark(self, request, event):
        if self.bridge.diagnostics is not None and event not in request.trace:
            request.trace[event] = time.monotonic_ns()

    async def write_output(self):
        while True:
            request = await self.ready.get()
            if not request.output:
                request.scheduled = False
                continue
            payload, terminal = request.output.popleft()
            self.output_bytes -= len(payload)
            self.writer.write(payload)
            async with asyncio.timeout(self.bridge.args.write_timeout):
                await self.writer.drain()
            if payload[4:5] in (b"A", b"F"):
                self.mark(request, "first_audio_written")
            request.scheduled = False
            if terminal:
                self.retire(request)
            elif request.output:
                request.scheduled = True
                self.ready.put_nowait(request)

    async def prepare(self, request, value):
        try:
            if request.canceled or self.closed:
                return
            loop = asyncio.get_running_loop()
            async with self.bridge.preparation_slots:
                if request.canceled or self.closed:
                    return
                request.preparation_started = True
                self.running_preparations.add(request.preparation_task)
                self.mark(request, "preparation_started")
                try:
                    text, tokens = await loop.run_in_executor(
                        self.bridge.executor, self.bridge.prepare_text, value
                    )
                finally:
                    self.running_preparations.discard(request.preparation_task)
                self.mark(request, "preparation_finished")
            if request.canceled or self.closed:
                return
            request.text = text
            request.token_count = len(tokens)
            request.submitted = True
            await self.send_native(
                "submit",
                request_id=request.native_id,
                session_id=None,
                normalized_text=text,
                token_ids=tokens,
                speaker_id="english_sample",
                max_tokens=512,
                temperature=0.7,
                top_k=80,
                sample_rate_hz=22050,
                seed=value.get("seed", 0),
            )
            self.mark(request, "native_submitted")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.fail(request, "preparation_failed", str(error))

    async def read_client(self):
        while True:
            value = await read_control(self.reader)
            kind = value.get("kind")
            ident = value.get("request_id")
            if type(ident) is not int or not 0 <= ident <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError("request_id must be a uint64 integer")
            if kind == "submit":
                if ident in self.requests:
                    raise ValueError("request_id is already active")
                if (
                    len(self.requests) >= self.bridge.args.max_pending_requests
                    or len(self.preparations) >= self.bridge.args.max_pending_requests
                ):
                    raise ValueError("client pending request limit exceeded")
                request = Request(ident, uuid.uuid4().hex, "")
                self.mark(request, "received")
                self.requests[ident] = request
                self.native_requests[request.native_id] = request
                try:
                    self.bridge.validate_submit(value)
                except ValueError as error:
                    await self.fail(request, "invalid_request", str(error), abort=False)
                    continue
                task = self.spawn(self.prepare(request, value))
                request.preparation_task = task
                self.preparations.add(task)
                task.add_done_callback(self.preparations.discard)
            elif kind == "abort":
                request = self.requests.get(ident)
                if request is not None:
                    await self.fail(request, "aborted", "request aborted", aborted=True)
            else:
                raise ValueError("unsupported bridge client message")

    async def read_native(self):
        while True:
            value = await read_frame(self.native_reader)
            if value.get("protocol_version") != 1:
                raise ValueError("native protocol version changed")
            kind = value.get("kind")
            ident = value.get("request_id")
            request = self.native_requests.get(ident)
            payload = None
            if kind == "audio_slab":
                if request is not None:
                    self.mark(request, "first_audio_received")
                payload = self.slabs.copy(value)
                await self.send_native(
                    "audio_ack",
                    arena=value["arena"],
                    slot=value["slot"],
                    generation=value["generation"],
                )
            if request is None:
                if ident in self.retired:
                    continue
                raise ValueError("native output belongs to an unknown request")
            if request.canceled or request.terminal:
                continue
            if kind == "accepted":
                if request.accepted:
                    raise ValueError("native request was accepted twice")
                request.accepted = True
                if not self.enqueue(
                    request,
                    control_frame(
                        "accepted",
                        request_id=request.request_id,
                        processed_text=request.text,
                        token_count=request.token_count,
                    ),
                ):
                    await self.fail(request, "output_backpressure", "client output queue is full")
            elif kind == "audio_slab":
                if not request.accepted or value["sequence"] != request.next_sequence:
                    raise ValueError("native audio sequence mismatch")
                request.next_sequence += 1
                request.total_samples += value["samples"]
                forwarded = (
                    float_audio_frame(request.request_id, value["sequence"], payload)
                    if os.environ.get("NEMOTRON_BRIDGE_FORWARD_F32") == "1"
                    else audio_frame(request.request_id, value["sequence"], pcm16(payload))
                )
                if not self.enqueue(request, forwarded):
                    await self.fail(request, "output_backpressure", "client output queue is full")
            elif kind == "stream_complete_v1":
                if (
                    type(value.get("next_sequence")) is not int
                    or type(value.get("chunk_count")) is not int
                    or type(value.get("total_samples")) is not int
                    or value["next_sequence"] != request.next_sequence
                    or value["chunk_count"] != request.next_sequence
                    or value["total_samples"] != request.total_samples
                    or request.next_sequence < 1
                    or request.total_samples < 1
                ):
                    raise ValueError("native completion accounting mismatch")
                if not self.enqueue(
                    request,
                    control_frame(
                        "complete",
                        request_id=request.request_id,
                        total_samples=request.total_samples,
                        chunk_count=request.next_sequence,
                    ),
                    terminal=True,
                ):
                    await self.fail(request, "output_backpressure", "client output queue is full")
            elif kind == "failed":
                await self.fail(
                    request,
                    str(value.get("code", "backend_failed")),
                    str(value.get("message", "native request failed")),
                    abort=False,
                )
            elif kind == "aborted":
                await self.fail(
                    request, "aborted", "native request aborted", abort=False, aborted=True
                )
            else:
                raise ValueError("unsupported native output message")

    async def run(self):
        tasks = []
        try:
            hello = await asyncio.wait_for(read_control(self.reader), 10)
            if hello != {"kind": "hello", "protocol_version": 1}:
                raise ValueError("invalid bridge handshake")
            self.native_reader, self.native_writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.bridge.args.native_worker_socket)),
                self.bridge.args.connect_timeout,
            )
            await self.send_native(
                "client_hello",
                expected_manifest_sha256=self.bridge.manifest_hash,
                client_instance_id="nemo-speech-cpp-" + uuid.uuid4().hex,
            )
            ready = await asyncio.wait_for(read_frame(self.native_reader), 10)
            self.bridge.validate_ready(ready)
            self.writer.write(
                control_frame(
                    "ready",
                    model_name="nemotron-tts",
                    sample_rate=22050,
                    voices=["english_sample"],
                    batch_size=self.bridge.args.width,
                    mode=self.bridge.args.mode,
                    text_normalization_enabled=self.bridge.preparer.enabled,
                )
            )
            await asyncio.wait_for(self.writer.drain(), self.bridge.args.write_timeout)
            tasks = [
                asyncio.create_task(fn())
                for fn in (self.read_client, self.read_native, self.write_output)
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (asyncio.IncompleteReadError, ConnectionError, BrokenPipeError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("bridge connection failed")
        finally:
            self.closed = True
            for task in (*tasks, *self.background):
                if task not in self.running_preparations:
                    task.cancel()
            await asyncio.gather(*tasks, *self.background, return_exceptions=True)
            if self.native_writer is not None:
                self.native_writer.close()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.native_writer.wait_closed(), 5)
            self.slabs.close()
            self.writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.writer.wait_closed(), 5)


class Bridge:
    def __init__(self, args, *, tokenizer=None, preparer=None):
        self.args = args
        raw = args.manifest.read_bytes()
        self.manifest_hash = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest["physical_width"] != args.width:
            raise ValueError("bridge width differs from the native serving plan")
        if self.manifest["host"].get("batching_mode", "fixed_cohort") != args.mode:
            raise ValueError("bridge mode differs from the native serving plan")
        if self.manifest["transport_compatibility"]["output_contract"] != (
            "pcm_f32_le_22050_mono_variable_streaming_v3"
        ):
            raise ValueError("bridge requires the native streaming-v3 output contract")
        if tokenizer is None:
            from tokenizers import Tokenizer

            seal_path = args.model_seal or args.model / "model-seal.json"
            seal = json.loads(seal_path.read_text())
            tokenizer_path = (args.model / "tokenizer.json").resolve(strict=True)
            tokenizer_seal = seal["files"]["tokenizer.json"]
            if (
                seal["model_identity"] != self.manifest["model"]["checkpoint_sha256"]
                or not tokenizer_path.is_relative_to(args.model.resolve(strict=True))
                or tokenizer_path.stat().st_size != tokenizer_seal["bytes"]
                or hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
                != tokenizer_seal["sha256"]
            ):
                raise ValueError("tokenizer does not match the selected sealed model")
            tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        if preparer is None:
            from .normalization import TextPreparer

            preparer = TextPreparer(args.normalization_config)
        self.tokenizer = tokenizer
        self.preparer = preparer
        self.executor = ThreadPoolExecutor(
            max_workers=args.preparation_threads, thread_name_prefix="native-text"
        )
        self.preparation_slots = asyncio.Semaphore(args.preparation_threads)
        self.session = None
        self.tasks = set()
        diagnostic_path = os.environ.get("NEMOTRON_BRIDGE_DIAGNOSTICS")
        self.diagnostics = open(diagnostic_path, "x") if diagnostic_path else None

    @staticmethod
    def validate_submit(value):
        allowed = {
            "kind",
            "protocol_version",
            "request_id",
            "text",
            "voice",
            "language",
            "seed",
            "apply_normalize",
        }
        if set(value) - allowed:
            raise ValueError("unsupported submit fields")
        text = value.get("text")
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 1_000_000:
            raise ValueError("text must contain 1..1000000 UTF-8 bytes")
        if value.get("voice", "english_sample") != "english_sample":
            raise ValueError("native TTS supports only voice english_sample")
        if value.get("language", "en-US") not in ("en", "en-US", "english", "English", ""):
            raise ValueError("the native package supports English only")
        seed = value.get("seed", 0)
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must be a uint64 integer")
        if type(value.get("apply_normalize", True)) is not bool:
            raise ValueError("apply_normalize must be boolean")

    def prepare_text(self, value):
        text = self.preparer.prepare(value["text"], "en-US", value.get("apply_normalize", True))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text preparation returned empty text")
        tokens = self.tokenizer.encode_batch([text], add_special_tokens=False)[0].ids + [131073]
        if not 1 <= len(tokens) <= 512:
            raise ValueError("native TTS supports at most 512 text tokens including EOS")
        if any(type(token) is not int or not 0 <= token < 131075 for token in tokens):
            raise ValueError("tokenizer returned an out-of-vocabulary token")
        return text, tokens

    def validate_ready(self, value):
        expected = self.manifest["transport_compatibility"]
        if (
            value.get("kind") != "worker_ready"
            or value.get("protocol_version") != 1
            or value.get("manifest_sha256") != self.manifest_hash
            or value.get("model_sha256") != self.manifest["model"]["checkpoint_sha256"]
            or value.get("output_contract") != expected["output_contract"]
            or value.get("enabled_buckets") != sorted({1, self.args.width})
            or any(
                value.get(key) != item
                for key, item in expected["installed_transport_versions"].items()
            )
        ):
            raise ValueError("native worker identity differs from the selected package")

    async def accept(self, reader, writer):
        if self.session is not None:
            writer.close()
            await writer.wait_closed()
            return
        self.session = Session(self, reader, writer)
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            await self.session.run()
        finally:
            self.session = None
            self.tasks.discard(task)

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
        self.preparer.close()
        if self.diagnostics is not None:
            self.diagnostics.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="C++ frontend bridge for native Nemotron TTS")
    for name in ("native-worker-socket", "manifest", "model", "listen"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("fixed_cohort", "fixed_request", "inflight"), default="inflight"
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--normalization-config", type=Path)
    parser.add_argument("--model-seal", type=Path)
    parser.add_argument("--preparation-threads", type=int, default=16)
    parser.add_argument("--max-pending-requests", type=int, default=4096)
    parser.add_argument("--request-queue-chunks", type=int, default=32)
    parser.add_argument("--max-output-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--write-timeout", type=float, default=10)
    parser.add_argument("--connect-timeout", type=float, default=10)
    args = parser.parse_args(argv)
    if args.width not in [1 << exponent for exponent in range(13)]:
        parser.error("width must be a supported power of two from 1 through 4096")
    for key in (
        "preparation_threads",
        "max_pending_requests",
        "request_queue_chunks",
        "max_output_bytes",
        "write_timeout",
        "connect_timeout",
    ):
        if getattr(args, key) <= 0:
            parser.error(key + " must be positive")
    return args


async def serve(args):
    bridge = Bridge(args)
    server = None
    created = False
    try:
        if args.listen.exists() or args.listen.is_symlink():
            raise ValueError("refusing to replace an existing bridge socket")
        args.listen.parent.mkdir(parents=True, exist_ok=True)
        server = await asyncio.start_unix_server(bridge.accept, path=str(args.listen))
        created = True
        args.listen.chmod(0o600)
        stopping = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopping.set)
        print(json.dumps({"status": "BRIDGE_LISTENING", "socket": str(args.listen)}), flush=True)
        await stopping.wait()
    finally:
        if server is not None:
            server.close()
            await server.wait_closed()
        await bridge.close()
        if created:
            args.listen.unlink(missing_ok=True)


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    print(json.dumps({"serialization": install_serialization(native_protocol)}), flush=True)
    asyncio.run(serve(parse_args(argv)))


if __name__ == "__main__":
    main()
