from __future__ import annotations

import asyncio
import json
import math
import os
from collections import Counter
from pathlib import Path

from .backend import BackendError
from .batcher import RequestAborted

MAX_REQUEST_PACKETS = 8
MAX_HELD_BYTES = 128 * 1024 * 1024


def initial_hold_seconds():
    value = float(os.environ.get("NEMOTRON_INITIAL_AUDIO_HOLD_MS", "0"))
    if not math.isfinite(value) or value not in (0.0, 40.0, 300.0):
        raise ValueError("initial audio hold must be 0, 40 or 300 milliseconds")
    return value / 1000.0


class InitialHoldBudget:
    def __init__(self, limit=MAX_HELD_BYTES, binding=None, configured_hold_ms=0):
        if type(configured_hold_ms) is not int or configured_hold_ms not in (0, 40, 300):
            raise ValueError("invalid initial audio hold receipt duration")
        self.configured_hold_ms = configured_hold_ms
        if type(limit) is not int or limit < 0:
            raise ValueError("invalid initial audio staging limit")
        self.limit = limit
        self.binding = binding
        self.used = 0
        self.peak = 0
        self.reasons = Counter()
        self.delay_histogram_ms = Counter()
        self.count = 0
        self.delay_sum = 0.0
        self.delay_min = None
        self.delay_max = 0.0

    def reserve(self, size):
        if size < 0 or self.used + size > self.limit:
            return False
        self.used += size
        self.peak = max(self.peak, self.used)
        return True

    def release(self, size):
        if not 0 <= size <= self.used:
            raise RuntimeError("initial audio staging accounting drift")
        self.used -= size

    def record(self, delay, reason):
        self.count += 1
        self.reasons[reason] += 1
        self.delay_sum += delay
        self.delay_min = delay if self.delay_min is None else min(self.delay_min, delay)
        self.delay_max = max(self.delay_max, delay)
        self.delay_histogram_ms[min(1000, int(delay * 1000) // 5 * 5)] += 1

    def snapshot(self, directory):
        row = {
            "schema": "nemotron.initial-publication-hold.v6",
            "activation_binding": self.binding,
            "configured_hold_ms": self.configured_hold_ms,
            "max_request_packets": MAX_REQUEST_PACKETS,
            "extra_staging_limit_bytes": self.limit,
            "extra_staging_current_bytes": self.used,
            "extra_staging_peak_bytes": self.peak,
            "release_count": self.count,
            "release_reasons": dict(self.reasons),
            "hold_seconds_sum": self.delay_sum,
            "hold_seconds_min": self.delay_min,
            "hold_seconds_max": self.delay_max,
            "hold_histogram_5ms_lower_edges_1000_overflow": dict(self.delay_histogram_ms),
            "scope": "cumulative process hold-release delays; excludes subsequent send time",
        }
        with (Path(directory) / "initial-publication-hold.jsonl").open("a") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


class InitialAudioHold:
    def __init__(self, handle, seconds, budget):
        if not 0 < handle._chunks.maxsize < MAX_REQUEST_PACKETS:
            raise RuntimeError("initial hold requires an existing queue below eight packets")
        self.handle = handle
        self.request_id = handle.request_id
        self.result = handle.result
        self.seconds = seconds
        self.budget = budget
        self.aborted = asyncio.Event()
        self.terminal_sent = False
        self.terminal_started = False
        self.late_abort_count = 0
        self.delivery_task = None
        self.reading = None
        self.first = True
        self.held_bytes = 0

    def abort(self):
        if self.terminal_sent:
            return False
        already_aborted = self.aborted.is_set()
        self.aborted.set()
        if not already_aborted and self.reading is not None and not self.reading.done():
            self.reading.cancel(self)
        return True

    def check(self):
        if self.aborted.is_set():
            raise RequestAborted()
        if self.result.cancelled():
            raise RequestAborted()
        if self.result.done():
            error = self.result.exception()
            if error is not None:
                raise error

    def before_send(self, value):
        if value["kind"] in ("audio", "audio_slab", "stream_complete_v1"):
            self.check()

    def begin_send(self, value):
        if value["kind"] in ("stream_complete_v1", "failed", "aborted") or (
            value["kind"] in ("audio", "audio_slab") and value.get("is_final")
        ):
            self.terminal_started = True

    def after_send(self, value):
        if value["kind"] in ("stream_complete_v1", "failed", "aborted") or (
            value["kind"] in ("audio", "audio_slab") and value.get("is_final")
        ):
            self.terminal_sent = True

    def release(self):
        self.budget.release(self.held_bytes)
        self.held_bytes = 0

    async def wait_owned(self, awaitable):
        if self.reading is not None:
            raise RuntimeError("overlapping initial hold ownership wait")
        self.reading = asyncio.current_task()
        try:
            return await awaitable
        except asyncio.CancelledError as error:
            if error.args and error.args[0] is self:
                if self.reading.uncancel():
                    raise
                raise RequestAborted() from None
            raise
        finally:
            self.reading = None

    async def next_chunk(self):
        self.release()
        self.check()
        chunk = await self.wait_owned(self.handle.next_chunk())
        self.check()
        if not self.first:
            return chunk
        self.first = False
        chunk.validate()
        if chunk.request_id != self.request_id or chunk.sequence != 0:
            raise BackendError("backend_contract", "streaming audio identity/sequence drift")
        if chunk.is_final or self.result.done():
            self.budget.record(0.0, "successful_terminal")
            return chunk
        size = len(chunk.pcm_f32_le)
        if not self.budget.reserve(size):
            self.budget.record(0.0, "staging_capacity")
            return chunk
        self.held_bytes = size
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self.seconds
        released = loop.create_future()

        def wake(_result=None):
            if not released.done():
                released.set_result(None)

        def deadline_reached():
            nonlocal timer
            if loop.time() < deadline:
                timer = loop.call_at(deadline, deadline_reached)
            else:
                wake()

        timer = loop.call_at(deadline, deadline_reached)
        self.result.add_done_callback(wake)
        reason = "cancelled_or_failed"
        try:
            await self.wait_owned(released)
            self.check()
            reason = "successful_terminal" if self.result.done() else "timer"
            return chunk
        finally:
            self.budget.record(loop.time() - started, reason)
            timer.cancel()
            self.result.remove_done_callback(wake)
            released.cancel()

    async def close(self):
        if self.reading is not None:
            raise RuntimeError("initial audio reader remains active during delivery cleanup")
        self.release()
