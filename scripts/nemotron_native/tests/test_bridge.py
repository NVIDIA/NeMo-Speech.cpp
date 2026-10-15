# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from nemotron_cpp_bridge.protocol import (
    control_frame,
    float_audio_frame,
    pcm16,
    pcm16_scalar,
    read_control,
)
from nemotron_cpp_bridge.server import (
    Bridge,
    Request,
    Session,
    initialize_preparation_worker,
    parse_args,
    preparation_cpus,
)
from nemotron_cpp_bridge.slabs import SlabReader
from nemotron_native_backend.protocol import message, read_frame, write_frame


class PreparationAffinityTests(unittest.TestCase):
    def test_default_does_not_change_affinity(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(preparation_cpus())
        with mock.patch.object(os, "sched_setaffinity", create=True) as setter:
            initialize_preparation_worker(None)
            setter.assert_not_called()

    def test_cpu_list_validation(self):
        with mock.patch.object(os, "sched_getaffinity", return_value={1, 2, 4, 5}, create=True):
            for invalid in ("", "-1", "1,,2", "2-1", "0", "1-4", "9999999999", "1.5"):
                with self.subTest(value=invalid), mock.patch.dict(
                    os.environ, {"NEMOTRON_BRIDGE_PREPARATION_CPUS": invalid}
                ), self.assertRaises(ValueError):
                    preparation_cpus()
            with mock.patch.dict(os.environ, {"NEMOTRON_BRIDGE_PREPARATION_CPUS": "1-2, 4,5"}):
                self.assertEqual(preparation_cpus(), frozenset({1, 2, 4, 5}))

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux affinity required")
    def test_only_executor_workers_receive_affinity(self):
        original = os.sched_getaffinity(0)
        chosen = frozenset({min(original)})
        with ThreadPoolExecutor(
            max_workers=1, initializer=initialize_preparation_worker, initargs=(chosen,)
        ) as executor:
            self.assertEqual(executor.submit(os.sched_getaffinity, 0).result(), chosen)
            self.assertEqual(os.sched_getaffinity(0), original)


class Tokenizer:
    def encode_batch(self, texts, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("target text must not add special tokens")
        if len(texts) != 1:
            raise AssertionError("text preparation must not wait for a request batch")
        return [SimpleNamespace(ids=[ord(char) for char in texts[0]])]


class Preparer:
    enabled = False

    def prepare(self, text, language, apply_normalize=True):
        return text.strip()

    def close(self):
        pass


async def receive(reader):
    size = struct.unpack(">I", await reader.readexactly(4))[0]
    payload = await reader.readexactly(size)
    if payload[:1] == b"J":
        return json.loads(payload[1:])
    request_id, sequence = struct.unpack(">QI", payload[1:13])
    return {
        "kind": "audio",
        "request_id": request_id,
        "sequence": sequence,
        "data": payload[13:],
        "encoding": payload[:1],
    }


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_float_audio_frame_preserves_extent_and_header(self):
        pcm = struct.pack("<4f", -1, 0, 0.5, 1)
        encoded = float_audio_frame(123, 7, pcm)
        self.assertEqual(struct.unpack(">I", encoded[:4])[0], 13 + len(pcm))
        self.assertEqual(encoded[4:17], b"F" + struct.pack(">QI", 123, 7))
        self.assertEqual(encoded[17:], pcm)
        for malformed in (b"", b"x", b"abc", b"abcde"):
            with self.assertRaises(ValueError):
                float_audio_frame(123, 7, malformed)

    async def test_valid_text_budget_survives_json_escaping(self):
        text = "\x01" * 1_000_000
        encoded = control_frame("submit", request_id=1, text=text)
        self.assertGreater(len(encoded), 2 * 1024 * 1024)
        reader = asyncio.StreamReader()
        reader.feed_data(encoded)
        reader.feed_eof()
        self.assertEqual((await read_control(reader))["text"], text)


class PCMTests(unittest.TestCase):
    def test_matches_asymmetric_half_away_rounding_and_saturation(self):
        values = [-2, -1, -0.5, -0.5 / 32768, -0.0, 0, 0.5 / 32767, 0.5, 1, 2]
        payload = struct.pack("<" + "f" * len(values), *values)
        expected = [-32768, -32768, -16384, -1, 0, 0, 1, 16384, 32767, 32767]
        self.assertEqual(list(struct.unpack("<10h", pcm16_scalar(payload))), expected)
        self.assertEqual(pcm16(payload), pcm16_scalar(payload))

    def test_rejects_nan_infinity_and_misaligned_input(self):
        for payload in (
            b"abc",
            *(struct.pack("<f", value) for value in (math.nan, math.inf, -math.inf)),
        ):
            with self.assertRaises(ValueError):
                pcm16(payload)


class SlabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.slab = SlabReader(self.temp.name)
        self.addCleanup(self.slab.close)
        self.arena = "r20-pcm-" + "a" * 32
        self.path = Path(self.temp.name, self.arena)
        self.path.write_bytes(struct.pack("<4f", -1, 0, 0.5, 1) + bytes(SlabReader.SLOT_BYTES - 16))
        self.descriptor = dict(
            slab_version=1,
            arena=self.arena,
            slot=0,
            generation=1,
            arena_bytes=SlabReader.SLOT_BYTES,
            slot_bytes=SlabReader.SLOT_BYTES,
            offset=0,
            data_bytes=16,
            samples=4,
            sequence=0,
            sample_rate_hz=22050,
            channels=1,
            encoding="pcm_f32_le",
            is_final=False,
        )

    def test_copy_is_owned_and_duplicate_lease_rejected(self):
        payload = self.slab.copy(self.descriptor)
        with self.path.open("r+b") as stream:
            stream.write(bytes(16))
        self.assertEqual(payload, struct.pack("<4f", -1, 0, 0.5, 1))
        with self.assertRaises(ValueError):
            self.slab.copy(self.descriptor)

    def test_rejects_invalid_extent_type_identity_and_symlink(self):
        for key, value in (
            ("offset", 1),
            ("data_bytes", 20),
            ("slot", 1),
            ("generation", 0),
            ("samples", True),
            ("arena", "../secret"),
            ("arena_bytes", 1 << 40),
            ("encoding", "pcm16"),
            ("sample_rate_hz", 24000),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.slab.copy({**self.descriptor, key: value})
        target = self.path.with_suffix(".backing")
        self.path.rename(target)
        self.path.symlink_to(target)
        with self.assertRaises(OSError):
            self.slab.copy(self.descriptor)


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        environment = mock.patch.dict(os.environ, {"NEMOTRON_BRIDGE_FORWARD_F32": "0"})
        environment.start()
        self.addCleanup(environment.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.native_path = self.root / "worker.sock"
        self.client_path = self.root / "bridge.sock"
        catalog = json.loads((ROOT / "deploy/nemotron_native/programs.json").read_text())
        plan = catalog["programs"]["1"]["serving"]
        plan["host"]["batching_mode"] = "inflight"
        self.manifest = self.root / "serving.json"
        self.manifest.write_text(json.dumps(plan))
        self.args = parse_args(
            [
                "--native-worker-socket",
                str(self.native_path),
                "--manifest",
                str(self.manifest),
                "--model",
                str(self.root),
                "--listen",
                str(self.client_path),
                "--width",
                "1",
            ]
        )
        self.bridge = Bridge(self.args, tokenizer=Tokenizer(), preparer=Preparer())
        self.bridge.frontend_preparation = getattr(self, "frontend_metadata", None)
        self.submitted = []
        self.aborted = []
        self.acknowledged = []
        self.bad_completion = False
        self.worker_done = asyncio.Event()
        self.worker_errors = []
        self.native = await asyncio.start_unix_server(self.worker, path=str(self.native_path))
        self.client_server = await asyncio.start_unix_server(
            self.bridge.accept, path=str(self.client_path)
        )
        self.slab_patch = mock.patch(
            "nemotron_cpp_bridge.server.SlabReader", lambda: SlabReader(self.root)
        )
        self.slab_patch.start()
        self.reader, self.writer = await asyncio.open_unix_connection(str(self.client_path))
        self.writer.write(control_frame("hello"))
        self.ready = await asyncio.wait_for(receive(self.reader), 3)

    async def asyncTearDown(self):
        self.writer.close()
        await self.writer.wait_closed()
        await self.bridge.close()
        self.client_server.close()
        self.native.close()
        await self.client_server.wait_closed()
        await self.native.wait_closed()
        await asyncio.wait_for(self.worker_done.wait(), 3)
        self.slab_patch.stop()
        self.temp.cleanup()
        self.assertEqual(self.worker_errors, [])

    async def worker(self, reader, writer):
        arena = "r20-pcm-" + "b" * 32
        path = self.root / arena
        path.write_bytes(bytes(SlabReader.SLOT_BYTES * 4))
        requests = {}
        try:
            hello = await read_frame(reader)
            if hello["expected_manifest_sha256"] != self.bridge.manifest_hash:
                raise AssertionError("manifest identity mismatch")
            plan = self.bridge.manifest
            await write_frame(
                writer,
                message(
                    "worker_ready",
                    manifest_sha256=self.bridge.manifest_hash,
                    model_sha256=plan["model"]["checkpoint_sha256"],
                    enabled_buckets=[1],
                    output_contract=plan["transport_compatibility"]["output_contract"],
                    **plan["transport_compatibility"]["installed_transport_versions"],
                ),
            )
            while True:
                incoming = await read_frame(reader)
                kind = incoming["kind"]
                if kind == "submit":
                    self.submitted.append(incoming)
                    slot = len(self.submitted) - 1
                    requests[slot] = incoming["request_id"]
                    await write_frame(
                        writer, message("accepted", request_id=incoming["request_id"])
                    )
                    if incoming["normalized_text"] == "cancel":
                        continue
                    with path.open("r+b") as stream:
                        stream.seek(slot * SlabReader.SLOT_BYTES)
                        stream.write(struct.pack("<4f", -1, 0, 0.5, 1))
                    await write_frame(
                        writer,
                        message(
                            "audio_slab",
                            request_id=incoming["request_id"],
                            sequence=0,
                            encoding="pcm_f32_le",
                            sample_rate_hz=22050,
                            channels=1,
                            samples=4,
                            is_final=False,
                            slab_version=1,
                            arena=arena,
                            slot=slot,
                            generation=1,
                            arena_bytes=SlabReader.SLOT_BYTES * 4,
                            slot_bytes=SlabReader.SLOT_BYTES,
                            offset=slot * SlabReader.SLOT_BYTES,
                            data_bytes=16,
                        ),
                    )
                elif kind == "audio_ack":
                    self.acknowledged.append(incoming)
                    with path.open("r+b") as stream:
                        stream.seek(incoming["slot"] * SlabReader.SLOT_BYTES)
                        stream.write(bytes(16))
                    await write_frame(
                        writer,
                        message(
                            "stream_complete_v1",
                            request_id=requests[incoming["slot"]],
                            next_sequence=1,
                            chunk_count=1,
                            total_samples=5 if self.bad_completion else 4,
                        ),
                    )
                elif kind == "abort":
                    self.aborted.append(incoming["request_id"])
                    await write_frame(writer, message("aborted", request_id=incoming["request_id"]))
                else:
                    raise AssertionError("unexpected worker command")
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except BaseException as error:
            self.worker_errors.append(repr(error))
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            self.worker_done.set()

    async def test_multiplexes_tokenized_requests_and_copies_audio_before_ack(self):
        self.assertEqual(self.ready["sample_rate"], 22050)
        self.assertEqual(self.ready["mode"], "inflight")
        for ident, text in ((5, " hello "), (9, "world")):
            self.writer.write(
                control_frame(
                    "submit",
                    request_id=ident,
                    text=text,
                    voice="english_sample",
                    language="en-US",
                    seed=0,
                )
            )
        observed = {5: [], 9: []}
        complete = set()
        while len(complete) < 2:
            event = await asyncio.wait_for(receive(self.reader), 3)
            observed[event["request_id"]].append(event)
            if event["kind"] == "complete":
                complete.add(event["request_id"])
        for events in observed.values():
            self.assertEqual([event["kind"] for event in events], ["accepted", "audio", "complete"])
            self.assertEqual(events[1]["data"], struct.pack("<4h", -32768, 0, 16384, 32767))
            self.assertEqual(events[2]["total_samples"], 4)
        self.assertEqual({item["normalized_text"] for item in self.submitted}, {"hello", "world"})
        self.assertTrue(all(item["token_ids"][-1] == 131073 for item in self.submitted))
        self.assertEqual(len(self.acknowledged), 2)

    async def test_float_forwarding_copies_before_ack_and_preserves_sample_count(self):
        with mock.patch.dict(os.environ, {"NEMOTRON_BRIDGE_FORWARD_F32": "1"}):
            self.writer.write(control_frame("submit", request_id=11, text="float audio"))
            events = [await asyncio.wait_for(receive(self.reader), 3) for _ in range(3)]
        self.assertEqual([event["kind"] for event in events], ["accepted", "audio", "complete"])
        self.assertEqual(events[1]["encoding"], b"F")
        self.assertEqual(events[1]["data"], struct.pack("<4f", -1, 0, 0.5, 1))
        self.assertEqual(events[2]["total_samples"], 4)
        self.assertEqual(events[2]["chunk_count"], 1)
        self.assertEqual(len(self.acknowledged), 1)

    async def test_cancel_propagates_and_unknown_abort_is_ignored(self):
        self.writer.write(control_frame("submit", request_id=7, text="cancel"))
        self.assertEqual((await asyncio.wait_for(receive(self.reader), 3))["kind"], "accepted")
        self.writer.write(control_frame("abort", request_id=7))
        self.assertEqual((await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted")
        for _ in range(100):
            if self.aborted:
                break
            await asyncio.sleep(0.001)
        self.assertEqual(self.aborted, [self.submitted[0]["request_id"]])
        self.writer.write(control_frame("abort", request_id=123))
        self.writer.write(control_frame("submit", request_id=8, text="next"))
        events = [await asyncio.wait_for(receive(self.reader), 3) for _ in range(3)]
        self.assertEqual([event["kind"] for event in events], ["accepted", "audio", "complete"])

    async def test_waiting_cancellation_reclaims_preparation_tasks(self):
        self.args.max_pending_requests = 2
        self.bridge.preparation_slots = asyncio.Semaphore(0)
        session = self.bridge.session
        for ident in (1, 2):
            self.writer.write(control_frame("submit", request_id=ident, text="waiting"))
        await wait_until(lambda: len(session.preparations) == 2)
        for ident in (1, 2):
            self.writer.write(control_frame("abort", request_id=ident))
        for _ in range(2):
            self.assertEqual((await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted")
        await wait_until(lambda: not session.preparations)
        for ident in range(3, 53):
            self.writer.write(control_frame("submit", request_id=ident, text="waiting"))
            self.writer.write(control_frame("abort", request_id=ident))
            self.assertEqual((await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted")
            await wait_until(lambda: not session.preparations)
        self.assertFalse(session.background)
        self.assertFalse(session.requests)
        self.assertFalse(self.submitted)
        self.bridge.preparation_slots = asyncio.Semaphore(1)
        self.writer.write(control_frame("submit", request_id=53, text="recovery"))
        events = [await asyncio.wait_for(receive(self.reader), 3) for _ in range(3)]
        self.assertEqual([event["kind"] for event in events], ["accepted", "audio", "complete"])

    async def test_running_cancellation_retains_slot_until_normalizer_finishes(self):
        self.args.max_pending_requests = 2
        self.bridge.preparation_slots = asyncio.Semaphore(1)
        session = self.bridge.session
        started, release = threading.Event(), threading.Event()
        original = self.bridge.prepare_text

        def blocked(value):
            started.set()
            if not release.wait(5):
                raise TimeoutError("normalizer was not released")
            return original(value)

        with mock.patch.object(self.bridge, "prepare_text", side_effect=blocked) as prepare:
            try:
                self.writer.write(control_frame("submit", request_id=1, text="running"))
                await wait_until(started.is_set)
                self.writer.write(control_frame("abort", request_id=1))
                self.assertEqual(
                    (await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted"
                )
                self.assertFalse(session.requests)
                self.assertEqual(len(session.preparations), 1)
                self.assertEqual(len(session.running_preparations), 1)
                self.assertTrue(self.bridge.preparation_slots.locked())
                self.writer.write(control_frame("submit", request_id=2, text="waiting"))
                await wait_until(lambda: len(session.preparations) == 2)
                self.writer.write(control_frame("abort", request_id=2))
                self.assertEqual(
                    (await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted"
                )
                await wait_until(lambda: len(session.preparations) == 1)
                self.assertEqual(prepare.call_count, 1)
                self.assertTrue(self.bridge.preparation_slots.locked())
            finally:
                release.set()
            await wait_until(lambda: not session.preparations)
            self.assertFalse(session.running_preparations)
            self.assertFalse(self.bridge.preparation_slots.locked())
            self.assertFalse(self.submitted)
            self.writer.write(control_frame("submit", request_id=3, text="recovery"))
            events = [await asyncio.wait_for(receive(self.reader), 3) for _ in range(3)]
            self.assertEqual([event["kind"] for event in events], ["accepted", "audio", "complete"])

    async def test_canceled_running_preparation_still_counts_toward_admission(self):
        self.args.max_pending_requests = 1
        self.bridge.preparation_slots = asyncio.Semaphore(1)
        session = self.bridge.session
        started, release = threading.Event(), threading.Event()
        original = self.bridge.prepare_text

        def blocked(value):
            started.set()
            if not release.wait(5):
                raise TimeoutError("normalizer was not released")
            return original(value)

        with mock.patch.object(self.bridge, "prepare_text", side_effect=blocked):
            try:
                self.writer.write(control_frame("submit", request_id=1, text="running"))
                await wait_until(started.is_set)
                self.writer.write(control_frame("abort", request_id=1))
                self.assertEqual(
                    (await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted"
                )
                with self.assertLogs("nemotron_cpp_bridge.server", "ERROR") as errors:
                    self.writer.write(control_frame("submit", request_id=2, text="over limit"))
                    await wait_until(lambda: session.closed)
                self.assertIn("client pending request limit exceeded", "\n".join(errors.output))
                self.assertEqual(len(session.preparations), 1)
                self.assertEqual(len(session.running_preparations), 1)
                self.assertTrue(self.bridge.preparation_slots.locked())
            finally:
                release.set()
            self.assertEqual(await asyncio.wait_for(self.reader.read(), 3), b"")
            self.assertFalse(session.preparations)
            self.assertFalse(session.running_preparations)
            self.assertFalse(self.bridge.preparation_slots.locked())
            self.assertFalse(self.submitted)

    async def test_prepared_submit_is_rejected_without_negotiation(self):
        self.assertNotIn("text_preparation", self.ready)
        self.writer.write(
            control_frame("submit_prepared", request_id=1, text="hello", token_ids=[42, 131073])
        )
        event = await asyncio.wait_for(receive(self.reader), 3)
        self.assertEqual((event["kind"], event["code"]), ("failed", "invalid_request"))
        self.assertIn("negotiated mode", event["message"])
        self.assertFalse(self.submitted)

    async def test_invalid_voice_fails_without_native_submission(self):
        self.writer.write(control_frame("submit", request_id=10, text="hello", voice="other"))
        event = await asyncio.wait_for(receive(self.reader), 3)
        self.assertEqual(event["kind"], "failed")
        self.assertEqual(event["code"], "invalid_request")
        self.assertEqual(self.submitted, [])

    async def test_tokenizer_identity_is_verified_before_loading(self):
        tokenizer_path = self.root / "tokenizer.json"
        tokenizer_path.write_bytes(b"sealed-tokenizer")
        (self.root / "model-seal.json").write_text(
            json.dumps(
                {
                    "model_identity": self.bridge.manifest["model"]["checkpoint_sha256"],
                    "files": {
                        "tokenizer.json": {
                            "bytes": tokenizer_path.stat().st_size,
                            "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                        }
                    },
                }
            )
        )
        factory = mock.Mock(return_value=Tokenizer())
        module = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=factory))
        with mock.patch.dict(sys.modules, {"tokenizers": module}):
            valid = Bridge(self.args, preparer=Preparer())
            await valid.close()
            self.assertEqual(factory.call_count, 1)
            tokenizer_path.write_bytes(b"changed-tokenizer")
            with self.assertRaisesRegex(ValueError, "tokenizer does not match"):
                Bridge(self.args, preparer=Preparer())
            self.assertEqual(factory.call_count, 1)

    async def test_continuous_workers_advance_without_event_loop_permits(self):
        self.bridge.continuous_preparation = True
        self.bridge.executor.shutdown()
        self.bridge.executor = ThreadPoolExecutor(max_workers=1)
        first_started, release, second_started = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = self.bridge.prepare_text

        def work(value):
            if value["text"] == "first":
                first_started.set()
                if not release.wait(5):
                    raise TimeoutError("first preparation was not released")
            else:
                second_started.set()
            return original(value)

        with mock.patch.object(self.bridge, "prepare_text", side_effect=work):
            try:
                self.writer.write(control_frame("submit", request_id=1, text="first"))
                await wait_until(first_started.is_set)
                self.writer.write(control_frame("submit", request_id=2, text="second"))
                await wait_until(lambda: len(self.bridge.session.dispatched_preparations) == 2)
                release.set()
                self.assertTrue(second_started.wait(1))
                events = [await asyncio.wait_for(receive(self.reader), 3) for _ in range(6)]
                self.assertEqual(sum(event["kind"] == "complete" for event in events), 2)
            finally:
                release.set()
        self.assertEqual({item["normalized_text"] for item in self.submitted}, {"first", "second"})

    async def test_continuous_canceled_queue_is_bounded_and_skips_normalization(self):
        self.bridge.continuous_preparation = True
        self.bridge.executor.shutdown()
        self.bridge.executor = ThreadPoolExecutor(max_workers=1)
        self.args.max_pending_requests = 2
        session = self.bridge.session
        started, release = threading.Event(), threading.Event()
        original = self.bridge.prepare_text

        def work(value):
            started.set()
            if not release.wait(5):
                raise TimeoutError("preparation was not released")
            return original(value)

        with mock.patch.object(self.bridge, "prepare_text", side_effect=work) as prepare:
            try:
                self.writer.write(control_frame("submit", request_id=1, text="running"))
                await wait_until(started.is_set)
                self.writer.write(control_frame("submit", request_id=2, text="queued"))
                await wait_until(lambda: len(session.dispatched_preparations) == 2)
                self.writer.write(control_frame("abort", request_id=2))
                self.assertEqual(
                    (await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted"
                )
                self.assertEqual(len(session.preparations), 2)
                self.assertEqual(prepare.call_count, 1)
                with self.assertLogs("nemotron_cpp_bridge.server", "ERROR") as errors:
                    self.writer.write(control_frame("submit", request_id=3, text="over limit"))
                    await wait_until(lambda: session.closed)
                self.assertIn("client pending request limit exceeded", "\n".join(errors.output))
                self.assertEqual(len(session.dispatched_preparations), 2)
                release.set()
                self.assertEqual(await asyncio.wait_for(self.reader.read(), 3), b"")
                self.assertFalse(session.dispatched_preparations)
                self.assertFalse(session.preparations)
                self.assertEqual(prepare.call_count, 1)
                self.assertFalse(self.submitted)
            finally:
                release.set()

    async def test_continuous_shutdown_drains_active_preparation(self):
        self.bridge.continuous_preparation = True
        started, release = threading.Event(), threading.Event()
        original = self.bridge.prepare_text
        session = self.bridge.session

        def work(value):
            started.set()
            if not release.wait(5):
                raise TimeoutError("preparation was not released")
            return original(value)

        with mock.patch.object(self.bridge, "prepare_text", side_effect=work):
            try:
                self.writer.write(control_frame("submit", request_id=1, text="running"))
                await wait_until(started.is_set)
                self.writer.close()
                await self.writer.wait_closed()
                await wait_until(lambda: session.closed)
                self.assertEqual(len(session.dispatched_preparations), 1)
                self.assertFalse(self.worker_done.is_set())
                release.set()
                await asyncio.wait_for(self.worker_done.wait(), 3)
                self.assertFalse(session.dispatched_preparations)
                self.assertFalse(session.preparations)
                self.assertFalse(self.submitted)
            finally:
                release.set()

    async def test_continuous_configuration_rejects_ambiguous_value(self):
        with mock.patch.dict(os.environ, {"NEMOTRON_BRIDGE_CONTINUOUS_PREPARATION": "true"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                Bridge(self.args, tokenizer=Tokenizer(), preparer=Preparer())

    async def test_completion_accounting_failure_closes_connection(self):
        self.bad_completion = True
        self.writer.write(control_frame("submit", request_id=11, text="hello"))
        terminal = []
        with self.assertLogs("nemotron_cpp_bridge.server", "ERROR"):
            try:
                while True:
                    terminal.append(await asyncio.wait_for(receive(self.reader), 3))
            except asyncio.IncompleteReadError:
                pass
        self.assertNotIn("complete", [event["kind"] for event in terminal])

    async def test_slow_request_queue_is_replaced_by_failure(self):
        session = Session(self.bridge, None, None)
        self.args.request_queue_chunks = 2
        request = Request(99, "native", "hello")
        self.assertTrue(session.enqueue(request, b"audio-one"))
        self.assertTrue(session.enqueue(request, b"audio-two"))
        self.assertFalse(session.enqueue(request, b"audio-three"))
        await session.fail(
            request, "output_backpressure", "client output queue is full", abort=False
        )
        self.assertEqual(len(request.output), 1)
        self.assertTrue(request.terminal)
        self.assertTrue(request.canceled)
        self.assertEqual(session.output_bytes, len(request.output[0][0]))


class FrontendBridgeTests(unittest.IsolatedAsyncioTestCase):
    frontend_metadata = {
        "library": "/test/preparation.so",
        "tokenizer": "/test/tokenizer.json",
        "normalization_config": "null",
    }
    asyncSetUp = BridgeTests.asyncSetUp
    asyncTearDown = BridgeTests.asyncTearDown
    worker = BridgeTests.worker

    async def test_prepared_submission_preserves_tokens_and_skips_preparation(self):
        self.assertEqual(self.ready["text_preparation"], self.frontend_metadata)
        with mock.patch.object(
            self.bridge, "prepare_text", side_effect=AssertionError("unexpected preparation")
        ) as prepare:
            self.writer.write(
                control_frame(
                    "submit_prepared",
                    request_id=1,
                    text="Already normalized.",
                    token_ids=[42, 43, 131073],
                    seed=7,
                )
            )
            events = [await asyncio.wait_for(receive(self.reader), 3) for _ in range(3)]
            prepare.assert_not_called()
        self.assertEqual([event["kind"] for event in events], ["accepted", "audio", "complete"])
        self.assertEqual(events[0]["processed_text"], "Already normalized.")
        self.assertEqual(events[0]["token_count"], 3)
        self.assertEqual(self.submitted[0]["normalized_text"], "Already normalized.")
        self.assertEqual(self.submitted[0]["token_ids"], [42, 43, 131073])
        self.assertEqual(self.submitted[0]["seed"], 7)

    async def test_ordinary_submission_is_rejected_in_prepared_mode(self):
        self.writer.write(control_frame("submit", request_id=1, text="hello"))
        event = await asyncio.wait_for(receive(self.reader), 3)
        self.assertEqual((event["kind"], event["code"]), ("failed", "invalid_request"))
        self.assertIn("negotiated mode", event["message"])
        self.assertFalse(self.submitted)
        self.writer.write(
            control_frame("submit_prepared", request_id=2, text="recovery", token_ids=[42, 131073])
        )
        events = [await asyncio.wait_for(receive(self.reader), 3) for _ in range(3)]
        self.assertEqual([event["kind"] for event in events], ["accepted", "audio", "complete"])

    async def test_invalid_prepared_token_contract_is_rejected_before_dispatch(self):
        for ident, tokens in enumerate(
            (
                None,
                [],
                [1, 2],
                [True, 131073],
                [-1, 131073],
                [131075, 131073],
                [1.0, 131073],
                ["1", 131073],
                [131073] * 513,
                {"id": 131073},
            ),
            start=1,
        ):
            with self.subTest(tokens=str(tokens)[:60]):
                self.writer.write(
                    control_frame(
                        "submit_prepared", request_id=ident, text="hello", token_ids=tokens
                    )
                )
                event = await asyncio.wait_for(receive(self.reader), 3)
                self.assertEqual((event["kind"], event["code"]), ("failed", "invalid_request"))
                self.assertIn("prepared token", event["message"])
                self.assertFalse(self.submitted)
        self.bridge.validate_input(
            {"kind": "submit_prepared", "text": "hello", "token_ids": [0] * 511 + [131073]}
        )
        with self.assertRaisesRegex(ValueError, "prepared token"):
            self.bridge.validate_input(
                {"kind": "submit_prepared", "text": "hello", "token_ids": (42, 131073)}
            )

    async def test_prepared_request_cancellation_reaches_native_worker(self):
        self.writer.write(
            control_frame("submit_prepared", request_id=1, text="cancel", token_ids=[42, 131073])
        )
        self.assertEqual((await asyncio.wait_for(receive(self.reader), 3))["kind"], "accepted")
        self.writer.write(control_frame("abort", request_id=1))
        self.assertEqual((await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted")
        await wait_until(lambda: bool(self.aborted))
        self.assertEqual(self.aborted, [self.submitted[0]["request_id"]])

    async def test_cancel_buffered_prepared_request_skips_native_submission(self):
        self.writer.write(
            control_frame("submit_prepared", request_id=1, text="hello", token_ids=[42, 131073])
            + control_frame("abort", request_id=1)
        )
        self.assertEqual((await asyncio.wait_for(receive(self.reader), 3))["kind"], "aborted")
        await wait_until(lambda: not self.bridge.session.preparations)
        self.assertFalse(self.submitted)


if __name__ == "__main__":
    unittest.main()
