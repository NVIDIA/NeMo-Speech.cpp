# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from nemotron_cpp_bridge.protocol import control_frame
from nemotron_cpp_bridge.server import Bridge, Request, Session
from nemotron_native_backend.protocol import frame, message


class CountingReader(asyncio.StreamReader):
    def __init__(self, payload):
        super().__init__()
        self.reads = 0
        self.feed_data(payload)

    async def readexactly(self, size):
        value = await super().readexactly(size)
        self.reads += 1
        return value

    @property
    def frames(self):
        return self.reads // 2


class Writer:
    def __init__(self, reader=None):
        self.reader = reader
        self.writes = []
        self.first_write = asyncio.Event()
        self.input_frames_at_first_write = None

    def write(self, payload):
        if not self.writes:
            if self.reader is not None:
                self.input_frames_at_first_write = self.reader.frames
            self.first_write.set()
        self.writes.append(payload)

    async def drain(self):
        pass


def session(fair_batch, reader=None):
    bridge = SimpleNamespace(
        fair_batch=fair_batch,
        diagnostics=None,
        frontend_preparation=None,
        validate_submit=Bridge.validate_submit,
        args=SimpleNamespace(
            max_pending_requests=1024,
            request_queue_chunks=256,
            max_output_bytes=1024 * 1024,
            write_timeout=1,
        ),
    )
    bridge.validate_input = MethodType(Bridge.validate_input, bridge)
    return Session(bridge, reader, Writer(reader))


class FairnessConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_boundaries_and_invalid_values(self):
        plan = {
            "physical_width": 1,
            "host": {"batching_mode": "inflight"},
            "transport_compatibility": {
                "output_contract": "pcm_f32_le_22050_mono_variable_streaming_v3"
            },
        }
        args = SimpleNamespace(
            manifest=SimpleNamespace(read_bytes=lambda: json.dumps(plan).encode()),
            width=1,
            mode="inflight",
            preparation_threads=1,
        )
        preparer = SimpleNamespace(enabled=False, close=lambda: None)
        for value in (None, "0", "64", "1024"):
            environment = {} if value is None else {"NEMOTRON_BRIDGE_FAIR_BATCH": value}
            with self.subTest(value=value), mock.patch.dict(os.environ, environment, clear=True):
                bridge = Bridge(args, tokenizer=object(), preparer=preparer)
                self.assertEqual(bridge.fair_batch, int(value or 0))
                await bridge.close()
        for value in ("", "-1", "1025", "1.5", "true", "1_000", "999999999"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"NEMOTRON_BRIDGE_FAIR_BATCH": value}, clear=True
            ), self.assertRaisesRegex(ValueError, "NEMOTRON_BRIDGE_FAIR_BATCH"):
                Bridge(args, tokenizer=object(), preparer=preparer)


class FairnessTests(unittest.IsolatedAsyncioTestCase):
    async def run_until_output(self, instance, consume):
        tasks = [asyncio.create_task(consume()), asyncio.create_task(instance.write_output())]
        try:
            await asyncio.wait_for(instance.writer.first_write.wait(), 1)
            for task in tasks:
                if task.done():
                    task.result()
            return instance.writer.input_frames_at_first_write
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            instance.slabs.close()

    async def test_native_output_progresses_before_buffered_burst_finishes(self):
        for fair_batch, expected in ((0, 192), (64, 64)):
            with self.subTest(fair_batch=fair_batch):
                reader = CountingReader(
                    b"".join(frame(message("accepted", request_id=str(i))) for i in range(192))
                )
                instance = session(fair_batch, reader)
                instance.native_reader = reader
                for i in range(192):
                    request = Request(i, str(i), "text", token_count=1)
                    instance.requests[i] = request
                    instance.native_requests[str(i)] = request
                self.assertEqual(
                    await self.run_until_output(instance, instance.read_native), expected
                )

    async def test_invalid_client_frames_cannot_starve_ready_failures(self):
        for fair_batch, expected in ((0, 192), (64, 64)):
            with self.subTest(fair_batch=fair_batch):
                reader = CountingReader(
                    b"".join(control_frame("submit", request_id=i, text="") for i in range(192))
                )
                instance = session(fair_batch, reader)
                self.assertEqual(
                    await self.run_until_output(instance, instance.read_client), expected
                )

    async def test_retired_native_frames_still_yield_to_ready_output(self):
        reader = CountingReader(frame(message("aborted", request_id="retired")) * 192)
        instance = session(64, reader)
        instance.native_reader = reader
        instance.retired.append("retired")
        instance.enqueue(Request(1, "ready", "text"), control_frame("accepted", request_id=1))
        self.assertEqual(await self.run_until_output(instance, instance.read_native), 64)

    async def test_output_yields_after_terminal_and_empty_queue_entries(self):
        for empty in (False, True):
            for fair_batch, expected in ((0, 192), (64, 64)):
                with self.subTest(empty=empty, fair_batch=fair_batch):
                    instance = session(fair_batch)
                    payloads = []
                    for i in range(192):
                        request = Request(i, str(i), "text")
                        if empty:
                            request.scheduled = True
                            instance.ready.put_nowait(request)
                        else:
                            payload = control_frame("complete", request_id=i)
                            payloads.append(payload)
                            instance.enqueue(request, payload, terminal=True)
                    task = asyncio.create_task(instance.write_output())
                    try:
                        await asyncio.sleep(0)
                        if task.done():
                            task.result()
                        processed = 192 - instance.ready.qsize()
                        self.assertEqual(processed, expected)
                        self.assertEqual(instance.writer.writes, payloads[:expected])
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        instance.slabs.close()


if __name__ == "__main__":
    unittest.main()
