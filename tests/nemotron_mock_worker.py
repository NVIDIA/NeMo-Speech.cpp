# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import struct
import threading


class MockWorker:
    def __init__(self, path, float_audio=False, preparation=None):
        self.path = str(path)
        self.float_audio = float_audio
        self.preparation = preparation
        self.ready = threading.Event()
        self.release_stream = threading.Event()
        self.parallel_ready = threading.Event()
        self.aborted = threading.Event()
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.submissions = []
        self.completed = []
        self.requests = {}
        self.errors = []
        self.threads = []
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.path)
        self.listener.listen(1)
        self.listener.settimeout(0.2)
        self.connection = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _send(self, payload):
        with self.send_lock:
            self.connection.sendall(struct.pack(">I", len(payload)) + payload)

    def _json(self, **value):
        self._send(b"J" + json.dumps(value).encode())

    def _read(self, size):
        result = bytearray()
        while len(result) < size:
            data = self.connection.recv(size - len(result))
            if not data:
                raise EOFError()
            result.extend(data)
        return bytes(result)

    def _run(self):
        try:
            while not self.closed.is_set():
                try:
                    self.connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if self.connection is None:
                return
            while not self.closed.is_set():
                length = struct.unpack(">I", self._read(4))[0]
                if not 1 <= length <= 1024 * 1024:
                    raise ValueError("invalid control frame length")
                payload = self._read(length)
                if payload[:1] != b"J":
                    raise ValueError("expected JSON control frame")
                message = json.loads(payload[1:])
                kind = message["kind"]
                if kind == "hello":
                    if message["protocol_version"] != 1:
                        raise ValueError("unexpected protocol version")
                    self._json(
                        kind="ready",
                        protocol_version=1,
                        model_name="nemotron-tts",
                        sample_rate=22050,
                        voices=["english_sample"],
                        batch_size=32,
                        mode="inflight",
                        text_normalization_enabled=False,
                        **({"text_preparation": self.preparation} if self.preparation else {}),
                    )
                    self.ready.set()
                elif kind in ("submit", "submit_prepared"):
                    if kind != ("submit_prepared" if self.preparation else "submit"):
                        raise ValueError("incorrect preparation dispatch")
                    request_id = message["request_id"]
                    event = threading.Event()
                    with self.lock:
                        if request_id in self.requests:
                            raise ValueError("duplicate request ID")
                        self.requests[request_id] = event
                        self.submissions.append(message)
                        if sum(x["text"].startswith("parallel") for x in self.submissions) >= 2:
                            self.parallel_ready.set()
                    thread = threading.Thread(
                        target=self._synthesize, args=(message, event), daemon=True
                    )
                    self.threads.append(thread)
                    thread.start()
                elif kind == "abort":
                    with self.lock:
                        event = self.requests.get(message["request_id"])
                    if event is not None:
                        event.set()
                    self.aborted.set()
                else:
                    raise ValueError(f"unexpected control kind {kind}")
        except (EOFError, OSError):
            pass
        except Exception as error:
            self.errors.append(str(error))

    def _synthesize(self, message, abort):
        request_id, text = message["request_id"], message["text"]
        try:
            if text == "failure":
                self._json(
                    kind="failed",
                    request_id=request_id,
                    message="injected preparation failure",
                )
                return
            self._json(
                kind="accepted",
                request_id=request_id,
                processed_text=text,
                token_count=len(message.get("token_ids", [0, 0, 0])),
            )
            audio_kind = b"F" if self.float_audio else b"A"
            if text == "malformed":
                invalid = (
                    struct.pack("<9f", *([0.5] * 8 + [float("nan")])) if self.float_audio else b"x"
                )
                self._send(audio_kind + struct.pack(">QI", request_id, 0) + invalid)
                return
            if text.startswith("parallel") and not self.parallel_ready.wait(5):
                raise RuntimeError("server serialized concurrent requests")
            count = 0
            for sequence in range(1000 if text == "cancel" else 2):
                if abort.is_set() or self.closed.is_set():
                    self._json(kind="failed", request_id=request_id, message="cancelled")
                    return
                pcm = (
                    struct.pack("<256f", *([0.5 if sequence % 2 else -0.5] * 256))
                    if self.float_audio
                    else struct.pack("<256h", *([1000 if sequence % 2 else -1000] * 256))
                )
                self._send(audio_kind + struct.pack(">QI", request_id, sequence) + pcm)
                count += 256
                if text == "stream" and sequence == 0 and not self.release_stream.wait(5):
                    raise RuntimeError("HTTP withheld first audio until completion")
                if text == "cancel":
                    abort.wait(0.01)
            self._json(
                kind="complete",
                request_id=request_id,
                total_samples=count,
                chunk_count=2,
            )
            with self.lock:
                self.completed.append(request_id)
        except (OSError, EOFError):
            pass
        except Exception as error:
            self.errors.append(str(error))
            try:
                self._json(kind="failed", request_id=request_id, message=str(error))
            except OSError:
                pass

    def close(self):
        self.closed.set()
        self.release_stream.set()
        self.parallel_ready.set()
        for event in self.requests.values():
            event.set()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        self.listener.close()
        self.thread.join(timeout=2)
        for thread in self.threads:
            thread.join(timeout=2)
