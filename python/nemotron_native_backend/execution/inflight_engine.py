from __future__ import annotations

import ctypes as C
import hashlib
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .engine import SAMPLE_RATE, SAMPLES_PER_FRAME, Stage0BuildInfo, inspect_residency
from .inflight_policy import DEFAULT_POLICY, InflightPolicy
from .native_assets import WIDTHS
from .native_inputs import Request, validate_requests
from .request_noise import RequestNoise
from .reusable_inputs import ReusableInputs


@dataclass
class Slot:
    epoch: int
    request_id: str
    seed: int
    admitted: float
    generation: int = 0
    produced: int = 0
    consumed: int = 0
    chunk_index: int = 0
    samples: int = 0
    reason: str | None = None
    finished: bool = False
    first_audio: float | None = None

    def target(self, schedule):
        return schedule[self.chunk_index] if self.chunk_index < len(schedule) else 8


def ready_groups(slots, progress_slots, schedule, *, first_packet_f2=False):
    groups = {}
    for index in progress_slots:
        slot = slots[index]
        if slot is None or slot.finished:
            continue
        pending = slot.produced - slot.consumed
        target = slot.target(schedule)
        if pending and (pending >= target or slot.reason is not None):
            frames = 3 if target <= 3 else 5 if target <= 5 else 8
            if first_packet_f2 and target <= 2:
                frames = 2
            groups.setdefault(frames, []).append(index)
    return sorted(groups.items(), key=lambda item: min(slots[i].admitted for i in item[1]))


def reset_ranges(slots, batch=32):
    if type(batch) is not int or batch not in WIDTHS:
        raise ValueError("unsupported physical reset width")
    slots = list(slots)
    if any(type(slot) is not int or not 0 <= slot < batch for slot in slots):
        raise ValueError("reset slots must be valid physical indices")
    if len(set(slots)) != len(slots):
        raise ValueError("reset slots must be unique")
    selected = sum(1 << slot for slot in slots)
    ranges = []

    def visit(start, width):
        mask = ((1 << width) - 1) << start
        if selected & mask == mask:
            ranges.append((start, start + width))
        elif selected & mask:
            visit(start, width // 2)
            visit(start + width // 2, width // 2)

    visit(0, batch)
    return ranges


def codec_row_order(selected, batch):
    if type(batch) is not int or batch not in WIDTHS:
        raise ValueError("unsupported physical codec width")
    selected = tuple(selected)
    if any(type(lane) is not int or not 0 <= lane < batch for lane in selected):
        raise ValueError("codec rows must be valid physical slots")
    active = set(selected)
    if len(active) != len(selected):
        raise ValueError("codec physical slots must be unique")
    order = tuple(sorted(active)) + tuple(lane for lane in range(batch) if lane not in active)
    inverse = [0] * batch
    for row, lane in enumerate(order):
        inverse[lane] = row
    return order, tuple(inverse)


def startup_steps(slots, progress, limit):
    if not progress or any(
        (slot := slots[index]) is None
        or slot.finished
        or slot.reason is not None
        or slot.generation >= min(5, limit)
        for index in progress
    ):
        return 1
    return max(1, min(min(5, limit) - slots[index].generation for index in progress))


def startup_progress(slots, progress, now, *, defer=False, align_phases=False):
    startup = [index for index in progress if slots[index].first_audio is None]
    if defer or not startup:
        return progress
    peers = [slots[index] for index in progress if slots[index].first_audio is not None]
    if any(
        slot.first_audio is None or slot.first_audio + slot.samples / SAMPLE_RATE - now < 0.040
        for slot in peers
    ):
        return progress
    if align_phases:
        startup_phase = [index for index in startup if slots[index].generation < 5]
        if startup_phase:
            return startup_phase
    return startup


class InflightEngine:
    @classmethod
    def from_engine(cls, engine, *, stage0_library, policy=None):
        return cls(engine, stage0_library=stage0_library, policy=policy)

    def __init__(self, engine, *, stage0_library, policy=None):
        if engine.batch not in WIDTHS or engine.steps > 511:
            raise ValueError("in-flight execution requires a supported physical width")
        if engine._closed or engine._lock.locked():
            raise RuntimeError("cannot adopt a closed engine or an active cohort")
        if tuple(engine.research_schedule) != (2, 3, 4, 5):
            raise ValueError("in-flight execution requires the selected 2/3/4/5 startup schedule")
        if engine._codec_pending is not None:
            raise RuntimeError("cannot adopt an engine with outstanding codec work")
        self.engine = engine
        self.policy = InflightPolicy.parse(policy)
        self.batch = engine.batch
        self.t = t = engine.t
        self.core = c = engine.core
        self.neural = n = c.neural
        self.schedule = tuple(engine.research_schedule)
        self.steps = engine.steps
        self.slots = [None] * self.batch
        self.epochs = [0] * self.batch
        self.bank = 0
        self.closed = False
        self.poisoned = False
        self.tick_count = 0
        self.producer_launches = 0
        self.codec_launches = 0
        self.live_refills = 0
        self.observation_pending = False
        self.admission_requested = threading.Event()
        self.admission_yields = 0
        self.codec_ready_order = deque()
        self.startup_priority_deferred = False
        self.startup_priority_launches = 0
        self.startup_priority_steps = 0
        self.startup_priority_started = None
        if self.policy.playback_reserve_guard:
            self.playback_producer_envelopes = {}
            self.playback_codec_envelopes = {}
            self.playback_producer_pending = None
            self.playback_codec_pending = None
            self.playback_reserve_last = None
        self.trace = deque(maxlen=4096)
        self.diagnostic_events = deque(maxlen=8192)
        self.diagnostic_dropped = 0
        n.stream.synchronize()
        engine.codec_stream.synchronize()
        from . import helper_graphs

        if engine.contraction is not None:
            engine.contraction.close()
            engine.contraction = None
        engine.active_lanes_config = None
        c.reset(0)
        n.start()
        n.stream.synchronize()
        self._capture_prefix()
        snapshot = getattr(engine, "_fp32_prefix_snapshot", None)
        if snapshot is not None:
            snapshot.detach()
        helper_graphs.uninstall(n)
        engine.helper_graphs = None
        cache = getattr(engine, "_r9_prefix_cache", None)
        if cache is not None:
            n.prefill_lib.mk2_native_prefill_run = cache[2]
            engine._r9_prefix_cache = None
            del cache
        if self.batch >= 128:
            t.cuda.empty_cache()
        self._replace_stage0(stage0_library)
        with t.cuda.stream(n.stream):
            self.progress = t.zeros(self.batch, dtype=t.uint8, device=n.inputs.device)
            self.progress_bool = t.zeros(self.batch, dtype=t.bool, device=n.inputs.device)
            self.progress_long = t.zeros(self.batch, dtype=t.int64, device=n.inputs.device)
            self.old_phoneme_ended = t.zeros_like(n.inputs.phoneme_ended)
            self.noise = RequestNoise(n, self.progress)
            n.bindings.gumbel = self.noise.buffer.data_ptr()
            self.progress_host = t.zeros(self.batch, dtype=t.uint8, pin_memory=True)
            self.progress_host_array = self.progress_host.numpy()
            self.meta = t.empty((2, self.batch), dtype=t.int64, device=n.inputs.device)
            self.meta_host = t.empty((2, self.batch), dtype=t.int64, pin_memory=True)
            self.meta_done = t.cuda.Event()
            self.ready = t.cuda.Event()
            self.tokens_host = t.zeros((self.batch, 512), dtype=t.int64, pin_memory=True)
            self.tokens_host_array = self.tokens_host.numpy()
            self.phoneme_long = t.zeros_like(n.inputs.last_phoneme)
            engine.counts.zero_()
            engine.consumed.zero_()
            c.done.zero_()
            n.codes.zero_()
            n.phonemes.zero_()
        setter = n.lib.mk2_set_native_progress_mask
        setter.argtypes = [C.c_void_p, C.c_void_p]
        setter.restype = C.c_int
        n._check(setter(n.stage0_handle, self.progress.data_ptr()), "in-flight progress mask")
        self._capture_helpers()
        if not hasattr(engine, "submit_pcm_transfer"):
            from . import pcm_transfer

            pcm_transfer.install(engine)
        self._allocate_codec()
        self._capture_admissions()
        n.stream.synchronize()

    def _view(self, pointer, shape, dtype):
        t = self.t
        item_size = t.empty((), dtype=dtype).element_size()
        count = 1
        for extent in shape:
            count *= extent

        class Alias:
            __cuda_array_interface__ = {
                "shape": (count * item_size,),
                "typestr": "|u1",
                "data": (int(pointer), False),
                "version": 3,
                "strides": None,
            }

        return t.as_tensor(Alias(), device=self.neural.inputs.device).view(dtype).view(shape)

    def _capture_prefix(self):
        t, n = self.t, self.neural
        blocks = n.info.blocks_per_sequence
        p = n.inputs.prompt_positions
        self.seq_len = self._view(n.bindings.seq_len, (self.batch,), t.int32)
        self.slot_map = self._view(n.bindings.slot_map, (self.batch,), t.int32)
        if not t.equal(self.slot_map, t.arange(self.batch, dtype=t.int32, device=n.inputs.device)):
            raise ValueError("in-flight prefix requires identity physical slots")
        expected_table = t.arange(
            self.batch * 2 * blocks, dtype=t.int32, device=n.inputs.device
        ).view(self.batch, 2, blocks)
        for i in range(3):
            table = self._view(n.bindings.kv_block_table[i], (self.batch, 2, blocks), t.int32)
            if not t.equal(table, expected_table):
                raise ValueError("in-flight prefix requires stable row-owned KV block tables")
        self.conv = [
            self._view(
                n.bindings.conv_state[i], (self.batch, n.bindings.conv_slot_stride[i]), t.bfloat16
            )
            for i in range(14)
        ]
        self.ssm = []
        if getattr(n, "_r8_initial_state", None) is None:
            if n.cache["ssm_dtype"] != "FP32":
                raise ValueError("uncanonical in-flight state must use FP32 storage")
            self.ssm = [
                self._view(
                    n.bindings.ssm_state[i], (self.batch, n.bindings.ssm_slot_stride[i]), t.float32
                )
                for i in range(14)
            ]
        self.kv_prefix = []
        for i in range(3):
            pool = self._view(
                n.bindings.kv_pool_base[i], (self.batch, 2, blocks, 4, 32, 128), t.bfloat16
            )
            for block in range((p + 31) // 32):
                self.kv_prefix.append(pool[:, :, block, :, : min(32, p - block * 32), :])
        with t.cuda.stream(n.stream):
            self.conv_prefix = [value[0].clone() for value in self.conv]
            self.ssm_prefix = [value[0].clone() for value in self.ssm]
            self.kv_initial = [value[0].clone() for value in self.kv_prefix]
        n.stream.synchronize()

    def _replace_stage0(self, path):
        from .split_native_runtime import bind_stage0_library

        n = self.neural
        path = Path(path).resolve(strict=True)
        lib = bind_stage0_library(path)
        version = lib.mk2_native_inflight_abi_version
        version.argtypes = []
        version.restype = C.c_int
        if version() != 1:
            raise ValueError("unsupported in-flight Stage0 ABI")
        lib.mk2_build_info.argtypes = [C.POINTER(Stage0BuildInfo)]
        lib.mk2_build_info.restype = C.c_int
        info = Stage0BuildInfo()
        n._check(lib.mk2_build_info(C.byref(info)), "in-flight resource contract", lib)
        if info.batch != self.batch or info.blocks_per_sm < 1:
            raise ValueError("in-flight kernel changed the supported physical width or residency")
        if self.engine.execution["mode"] == "overlap" and info.blocks_per_sm != 1:
            raise ValueError("in-flight overlap requires one resident producer block per SM")
        n.lib.mk2_destroy(n.stage0_handle)
        n.stage0_handle = None
        handle = lib.mk2_create_stage0(C.byref(n.pw), C.byref(n.bw), self.batch)
        n._require_handle(handle, "in-flight Stage0 allocation", lib)
        try:
            initial = getattr(n, "_r8_initial_state", None)
            if initial is not None:
                setter = lib.mk2_r8_bind_initial_state
                setter.argtypes = [
                    C.c_void_p,
                    C.POINTER(C.c_void_p),
                    C.c_void_p,
                    C.c_void_p,
                    C.c_int,
                ]
                setter.restype = C.c_int
                n._check(
                    setter(
                        handle,
                        initial["initial"],
                        initial["ready"],
                        n.bindings.seq_len,
                        n.inputs.prompt_positions,
                    ),
                    "in-flight canonical state",
                    lib,
                )
            state = getattr(n, "_r9_state", None)
            if state is not None:
                scales = (C.c_void_p * 14)(*[state["scales"][i].data_ptr() for i in range(14)])
                maps = (C.c_void_p * 14)(*[state["maps"][i].data_ptr() for i in range(14)])
                setter = lib.mk2_r9_state_bind
                setter.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
                setter.restype = C.c_int
                n._check(setter(handle, scales, maps), "in-flight state format binding", lib)
            setter = lib.mk2_set_native_predictor_active_mask
            setter.argtypes = [C.c_void_p, C.c_void_p]
            setter.restype = C.c_int
            n._check(setter(handle, n.inputs.audio_valid.data_ptr()), "in-flight startup mask", lib)
            lib.mk2_set_native_stage0_grid.argtypes = [C.c_void_p, C.c_int]
            lib.mk2_set_native_stage0_grid.restype = C.c_int
            n._check(
                lib.mk2_set_native_stage0_grid(
                    handle,
                    self.engine.execution.get(
                        "full_stage0_grid", self.engine.execution["stage0_grid"]
                    ),
                ),
                "in-flight full producer grid",
                lib,
            )
        except BaseException:
            lib.mk2_destroy(handle)
            raise
        n.lib, n.stage0_handle = lib, handle
        self.startup_entry = None
        if hasattr(n, "_r8_startup_stage0"):
            self.startup_entry = lib.mk2_run_native_startup_stage0
            self.startup_entry.argtypes = lib.mk2_run_native_stage0.argtypes
            self.startup_entry.restype = lib.mk2_run_native_stage0.restype
        n.library = path
        n.library_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.engine.paths = {**self.engine.paths, "stage0": path}
        self.engine._actual_library_records = {
            **self.engine._actual_library_records,
            "stage0": {"path": str(path), "sha256": n.library_sha256},
        }
        self.engine.residency = inspect_residency(self.core, self.engine.execution)

    def _prepare_device(self):
        t, i = self.t, self.neural.inputs
        self.progress_bool.copy_(self.progress)
        self.progress_long.copy_(self.progress)
        self.old_phoneme_ended.copy_(i.phoneme_ended)
        i.prepared = False
        i.generation = 0
        ReusableInputs.prepare_decode(i)
        t.where(self.progress_bool, i.phoneme_ended, self.old_phoneme_ended, out=i.phoneme_ended)
        t.logical_and(i.audio_valid, self.progress_bool, out=i.audio_valid)
        i.embedding.mul_(self.progress_bool[:, None])

    def _commit_device(self):
        t, n, e, c = self.t, self.neural, self.engine, self.core
        i = n.inputs
        t.eq(n.codes, 1025, out=i.code_eos)
        t.any(i.code_eos, dim=1, out=i.any_eos)
        t.logical_and(i.any_eos, i.audio_valid, out=i.stop)
        t.where(self.progress_bool[:, None], n.codes, i.last_codes, out=i.last_codes)
        self.phoneme_long.copy_(n.phonemes)
        t.where(self.progress_bool, self.phoneme_long, i.last_phoneme, out=i.last_phoneme)
        t.ge(n.codes, 0, out=e.element_valid)
        t.lt(n.codes, 1024, out=e.element_tmp)
        t.logical_and(e.element_valid, e.element_tmp, out=e.element_valid)
        t.all(e.element_valid, dim=1, out=e.valid)
        t.ne(n.codes, 0, out=e.element_tmp)
        t.any(e.element_tmp, dim=1, out=e.nonzero)
        t.logical_and(e.valid, e.nonzero, out=e.valid)
        t.logical_and(e.valid, i.audio_valid, out=e.valid)
        t.logical_not(c.done, out=e.active)
        t.logical_and(e.valid, e.active, out=e.valid)
        t.clamp(e.counts, max=self.steps - 1, out=e.write_index)
        e.queue.scatter_(1, e._write_index3, e._step_codes3)
        e.valid_i64.copy_(e.valid)
        e.counts.add_(e.valid_i64)
        t.logical_or(c.done, i.stop, out=c.done)
        i.offsets.add_(self.progress_long)
        self.meta[0].copy_(e.counts)
        self.meta[1].copy_(c.done)

    def _capture_helpers(self):
        t, n = self.t, self.neural
        names = (
            "offsets",
            "last_codes",
            "last_phoneme",
            "phoneme_ended",
            "audio_valid",
            "stop",
            "embedding",
            "combined_model",
        )
        with t.cuda.stream(n.stream):
            snapshots = {name: getattr(n.inputs, name).clone() for name in names}
            for _ in range(2):
                self._prepare_device()
                self.noise.generation()
                self._commit_device()
            n.stream.synchronize()
            self.prepare_graph = t.cuda.CUDAGraph()
            with t.cuda.graph(self.prepare_graph, stream=n.stream):
                self._prepare_device()
                self.noise.generation()
            self.commit_graph = t.cuda.CUDAGraph()
            with t.cuda.graph(self.commit_graph, stream=n.stream):
                self._commit_device()
            for name, value in snapshots.items():
                getattr(n.inputs, name).copy_(value)
            self.engine.counts.zero_()
            self.core.done.zero_()
        n.inputs.prepared = False
        n.stream.synchronize()

    def _allocate_codec(self):
        t, c, stream = self.t, self.core, self.engine.codec_stream
        self.codec_buffers = {}
        self.codec_packing = {}
        self.codec_ticket_rows = None
        self.codec_packing_launches = 0
        frames_supported = (
            (2, 3, 5, 8)
            if getattr(self, "policy", DEFAULT_POLICY).codec_first_packet_f2
            else (3, 5, 8)
        )
        for frames in frames_supported:
            packed = t.zeros((self.batch, frames, 16), dtype=t.int64, device="cuda:0")
            lengths = t.zeros(self.batch, dtype=t.int32, device="cuda:0")
            index = t.zeros((self.batch, frames), dtype=t.int64, device="cuda:0")
            offsets = t.arange(frames, dtype=t.int64, device="cuda:0").view(1, frames)
            mask = t.empty((self.batch, frames), dtype=t.bool, device="cuda:0")
            codec_slots = c.slots
            if getattr(self, "policy", DEFAULT_POLICY).codec_local_packing:
                codec_slots = t.arange(self.batch, dtype=t.int32, device="cuda:0")
                self.codec_packing[frames] = {
                    "slots": codec_slots,
                    "order": t.arange(self.batch, dtype=t.int64, device="cuda:0"),
                    "physical_codes": t.empty_like(packed),
                    "physical_lengths": t.empty_like(lengths),
                    "staged_order": tuple(range(self.batch)),
                }
            pcm = c.pcm[: self.batch * frames * SAMPLES_PER_FRAME].view(
                self.batch, frames * SAMPLES_PER_FRAME
            )
            event = t.cuda.Event()
            bindings = [
                c.codec.bind_buffers(
                    packed,
                    lengths,
                    codec_slots,
                    c.banks[bank],
                    c.banks[1 - bank],
                    pcm,
                    c.samples,
                    c.status,
                    stream=stream,
                    completion_event=event,
                )
                for bank in (0, 1)
            ]
            self.codec_buffers[frames] = (packed, lengths, index, offsets, mask, bindings)
        self.neural.stream.wait_stream(t.cuda.current_stream(0))
        stream.wait_stream(t.cuda.current_stream(0))

    def _check_open(self):
        if self.closed or self.poisoned:
            raise RuntimeError("in-flight engine is closed or failed")

    def _diagnostic(self, kind, **details):
        if getattr(self, "policy", DEFAULT_POLICY).diagnostics:
            row = {
                "type": kind,
                "physical_batch": self.batch,
                "host_monotonic_ns": time.monotonic_ns(),
                **details,
            }
            if "slots" in details:
                row["owners"] = [
                    {
                        "slot": index,
                        "epoch": self.slots[index].epoch,
                        "request_id": self.slots[index].request_id,
                    }
                    for index in details["slots"]
                    if self.slots[index] is not None
                ]
            if len(self.diagnostic_events) == self.diagnostic_events.maxlen:
                self.diagnostic_dropped += 1
            self.diagnostic_events.append(row)

    def drain_diagnostics(self):
        rows = list(self.diagnostic_events)
        self.diagnostic_events.clear()
        if self.diagnostic_dropped:
            rows.append({"type": "diagnostic_overflow", "events": self.diagnostic_dropped})
            self.diagnostic_dropped = 0
        return rows

    def _slot(self, index, epoch):
        self._check_open()
        if type(index) is not int or not 0 <= index < self.batch:
            raise ValueError("invalid physical slot")
        slot = self.slots[index]
        if slot is None or slot.epoch != epoch:
            raise ValueError("stale or absent request slot")
        return slot

    def _reset_range(self, start, end):
        n, c = self.neural, self.core
        rows = slice(start, end)
        for value, initial in zip(self.conv, self.conv_prefix, strict=True):
            value[rows].copy_(initial)
        for value, initial in zip(self.ssm, self.ssm_prefix, strict=True):
            value[rows].copy_(initial)
        for value, initial in zip(self.kv_prefix, self.kv_initial, strict=True):
            value[rows].copy_(initial)
        self.seq_len[rows].fill_(n.inputs.prompt_positions)
        self.slot_map[rows].copy_(self.slot_ids[rows])
        for bank in c.banks:
            for value in bank:
                value[rows].zero_()
        i = n.inputs
        i.text_tokens[rows].copy_(self.tokens_host[rows], non_blocking=True)
        i.text_lengths[rows].copy_(self.admission_host[0, rows], non_blocking=True)
        i.max_text_offsets[rows].copy_(self.admission_host[1, rows], non_blocking=True)
        self.noise.seeds[rows].copy_(self.admission_host[2, rows], non_blocking=True)
        for name in (
            "offsets",
            "last_codes",
            "last_phoneme",
            "phoneme_ended",
            "audio_valid",
            "stop",
        ):
            getattr(i, name)[rows].zero_()
        n.codes[rows].zero_()
        n.phonemes[rows].zero_()
        self.engine.counts[rows].zero_()
        self.engine.consumed[rows].zero_()
        c.done[rows].zero_()

    def _reset_slot(self, slot):
        self._reset_range(slot, slot + 1)

    def _capture_admissions(self):
        t, stream = self.t, self.neural.stream
        self.admission_host = t.zeros((3, self.batch), dtype=t.int64, pin_memory=True)
        self.admission_host_array = self.admission_host.numpy()
        self.admission_host[0].fill_(1)
        self.admission_graphs = {}
        self.admission_events = {}
        self.admission_done = [None] * self.batch
        self.admission_pending = [False] * self.batch
        with t.cuda.stream(stream):
            self.slot_ids = t.arange(self.batch, dtype=t.int32, device=self.neural.inputs.device)
            self._reset_range(0, self.batch)
            stream.synchronize()
            if self.batch > 32:
                from .inflight_reset import IndexedReset

                self.indexed_reset = IndexedReset(self)
                return
            for width in (1 << shift for shift in range(self.batch.bit_length())):
                for start in range(0, self.batch, width):
                    key = (start, start + width)
                    graph = t.cuda.CUDAGraph()
                    with t.cuda.graph(graph, stream=stream):
                        self._reset_range(*key)
                    self.admission_graphs[key] = graph
                    self.admission_events[key] = t.cuda.Event()

    def _validate_admission(self, slot, epoch, *, text, token_ids, seed, request_id):
        self._check_open()
        if type(slot) is not int or not 0 <= slot < self.batch or self.slots[slot] is not None:
            raise ValueError("admission requires an unused physical slot")
        if type(epoch) is not int or epoch <= self.epochs[slot]:
            raise ValueError("slot epoch must increase on every admission")
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError("request text and identity must be nonempty")
        if type(seed) is not int or not 0 <= seed < 1 << 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        request = Request(text, tuple(token_ids), tuple(self.core.requests[0].context_ids))
        validate_requests([request])

    def admit(self, slot, epoch, *, text, token_ids, seed, request_id):
        self.admit_many(
            [
                dict(
                    slot=slot,
                    epoch=epoch,
                    text=text,
                    token_ids=token_ids,
                    seed=seed,
                    request_id=request_id,
                )
            ]
        )

    def admit_many(self, records):
        self._check_open()
        self._observe()
        records = list(records)
        if len({record["slot"] for record in records}) != len(records):
            raise ValueError("admission slots must be unique")
        for record in records:
            self._validate_admission(**record)
        if not records:
            return
        ranges = reset_ranges((record["slot"] for record in records), self.batch)
        indexed = getattr(self, "indexed_reset", None)
        live = any(value is not None and value.generation > 0 for value in self.slots)
        try:
            self._diagnostic("reset_wait_start", slots=[record["slot"] for record in records])
            if indexed is not None:
                indexed.wait()
            pending = {
                self.admission_done[record["slot"]]
                for record in records
                if self.admission_pending[record["slot"]]
            }
            for event in pending:
                event.synchronize()
            self._diagnostic("reset_wait_end")
            staging_rows = (
                [record["slot"] for record in records]
                if indexed is None
                else list(range(len(records)))
            )
            lengths = [len(record["token_ids"]) for record in records]
            seeds = [
                record["seed"] if record["seed"] < 1 << 63 else record["seed"] - (1 << 64)
                for record in records
            ]
            self.admission_host_array[0, staging_rows] = lengths
            self.admission_host_array[1, staging_rows] = [length - 1 for length in lengths]
            self.admission_host_array[2, staging_rows] = seeds
            for position, record in enumerate(records):
                slot = record["slot"]
                self.admission_pending[slot] = False
                row = self.tokens_host_array[staging_rows[position]]
                row.fill(0)
                row[: len(record["token_ids"])] = record["token_ids"]
            with self.t.cuda.stream(self.neural.stream):
                if indexed is not None:
                    indexed.submit(records)
                else:
                    for key in ranges:
                        self.admission_graphs[key].replay()
                        event = self.admission_events[key]
                        event.record(self.neural.stream)
                        for slot in range(*key):
                            self.admission_done[slot] = event
                            self.admission_pending[slot] = True
                for record in records:
                    slot = record["slot"]
                    self.slots[slot] = Slot(
                        record["epoch"], record["request_id"], record["seed"], time.perf_counter()
                    )
                    self.epochs[slot] = record["epoch"]
                    self.live_refills += int(live)
                    self.trace.append(
                        {
                            "type": "admit",
                            "slot": slot,
                            "epoch": record["epoch"],
                            "request_id": record["request_id"],
                            "host_monotonic_ns": time.monotonic_ns(),
                        }
                    )
                self._diagnostic("reset_enqueued", slots=[record["slot"] for record in records])
        except BaseException:
            self.poisoned = True
            raise

    def _playback_queued_rows(self):
        pending = self.playback_codec_pending
        actual = getattr(self.core.codec, "_ticket", None)
        packed = getattr(self, "codec_ticket_rows", None)
        if pending is None:
            return actual is None and packed is None, set()
        ticket, _, frames, owners = pending
        if actual is not ticket or ticket.owner is not self.core.codec:
            return False, set()
        selected = tuple(row[0] for row in owners)
        if len(set(selected)) != len(selected):
            return False, set()
        for lane, epoch, request_id, size in owners:
            if not 0 <= lane < self.batch or not 0 < size <= frames:
                return False, set()
            slot = self.slots[lane]
            if slot is None or (slot.epoch, slot.request_id) != (epoch, request_id):
                return False, set()
        if packed is not None:
            if packed[0] is not ticket or packed[3] != selected:
                return False, set()
            if len(packed[2]) != self.batch or any(
                packed[2][lane] != size for lane, _, _, size in owners
            ):
                return False, set()
        return True, set(selected)

    def _playback_reserve_choice(self, progress, selected, repeats, now):
        if selected == progress and repeats == 1:
            self.playback_reserve_last = {"reason": "already_shared_single_step"}
            return selected, repeats
        published = [
            index
            for index, slot in enumerate(self.slots)
            if slot is not None and not slot.finished and slot.first_audio is not None
        ]
        if not published:
            self.playback_reserve_last = {"reason": "no_published_owners"}
            return selected, repeats
        valid, queued = self._playback_queued_rows()
        if valid:
            published = [
                index
                for index in published
                if self.slots[index].reason is None
                or self.slots[index].produced != self.slots[index].consumed
                or index in queued
            ]
            if not published:
                self.playback_reserve_last = {"reason": "no_pending_published_owners"}
                return selected, repeats
        ready = [
            index
            for index in published
            if index not in queued
            and self.slots[index].produced > self.slots[index].consumed
            and (
                self.slots[index].reason is not None
                or self.slots[index].produced - self.slots[index].consumed
                >= self.slots[index].target(self.schedule)
            )
        ]
        capped = bool(queued and ready)
        if capped:
            repeats = 1
        kind = (
            "startup"
            if self.startup_entry is not None
            and all(self.slots[index].generation < 5 for index in selected)
            else "full"
        )
        spans = [
            value
            for (entry, count), value in self.playback_producer_envelopes.items()
            if entry == kind and count >= repeats
        ]
        full = self.playback_producer_envelopes.get(("full", 1))
        codecs = self.playback_codec_envelopes
        calibrated = all(frames in codecs for frames in self.codec_buffers)
        reason = "reserve_clear"
        if not valid:
            reason = "queued_owner_unknown"
        elif self.observation_pending:
            reason = "producer_metadata_pending"
        elif not spans or not calibrated:
            reason = "uncalibrated_service"
        elif any(
            not math.isfinite(value) or value < 0
            for value in (*self.playback_producer_envelopes.values(), *codecs.values())
        ):
            reason = "invalid_service_envelope"
        evaluations = []
        if reason == "reserve_clear":
            quantum = max(spans)
            intervening = max(self.playback_producer_envelopes.values())
            codec = max(codecs[frames] for frames in self.codec_buffers)
            for index in published:
                slot = self.slots[index]
                pending = slot.produced - slot.consumed
                missing = (
                    0
                    if index in queued or slot.reason is not None
                    else max(0, slot.target(self.schedule) - pending)
                )
                if pending < 0 or (missing and full is None):
                    reason = "uncalibrated_service" if pending >= 0 else "invalid_owner_state"
                    break
                class_turns = len(self.codec_buffers) * (
                    2 if self.policy.codec_deadline_priority else 1
                )
                turns = 1 if index in queued else class_turns + bool(queued)
                reserve = (
                    quantum
                    + missing * (full or 0.0)
                    + (missing + turns) * codec
                    + (missing + max(0, turns - 1)) * intervening
                    + 0.040
                )
                lead = slot.first_audio + slot.samples / SAMPLE_RATE - now
                evaluations.append((index, slot.epoch, slot.request_id, missing, lead, reserve))
                if not math.isfinite(lead) or not math.isfinite(reserve) or lead < reserve:
                    reason = "playback_reserve"
        self.playback_reserve_last = {
            "reason": reason,
            "published": published,
            "queued": sorted(queued),
            "ready": ready,
            "prefetch_cap": capped,
            "owners": evaluations,
        }
        if reason != "reserve_clear":
            return progress, 1
        return selected, repeats

    def _playback_observed_producer(self):
        pending = self.playback_producer_pending
        if pending is not None:
            started, kind, repeats = pending
            key = (kind, repeats)
            elapsed = time.perf_counter() - started
            if math.isfinite(elapsed) and elapsed >= 0:
                self.playback_producer_envelopes[key] = max(
                    elapsed, self.playback_producer_envelopes.get(key, 0.0)
                )
            self.playback_producer_pending = None

    def _playback_finished_codec(self, ticket):
        pending = self.playback_codec_pending
        if pending is not None and pending[0] is ticket:
            _, started, frames, owners = pending
            if all(
                self.slots[lane] is not None
                and (self.slots[lane].epoch, self.slots[lane].request_id) == (epoch, request_id)
                for lane, epoch, request_id, _ in owners
            ):
                elapsed = time.perf_counter() - started
                if math.isfinite(elapsed) and elapsed >= 0:
                    self.playback_codec_envelopes[frames] = max(
                        elapsed,
                        self.playback_codec_envelopes.get(frames, 0.0),
                    )
        self.playback_codec_pending = None

    def _step(self, progress, *, repeats=1):
        if self.observation_pending:
            raise RuntimeError("previous producer metadata must be observed before mask reuse")
        if type(repeats) is not int or not 1 <= repeats <= startup_steps(
            self.slots, progress, self.steps
        ):
            raise ValueError("producer repetition would cross a request control boundary")
        if getattr(self, "policy", DEFAULT_POLICY).playback_reserve_guard:
            kind = (
                "startup"
                if self.startup_entry is not None
                and all(self.slots[index].generation < 5 for index in progress)
                else "full"
            )
            self.playback_producer_pending = (time.perf_counter(), kind, repeats)
        t, n = self.t, self.neural
        self.progress_host_array.fill(0)
        self.progress_host_array[progress] = 1
        with t.cuda.stream(n.stream):
            self.progress.copy_(self.progress_host, non_blocking=True)
            for _ in range(repeats):
                self.prepare_graph.replay()
                launch = n.lib.mk2_run_native_stage0
                if self.startup_entry is not None and all(
                    self.slots[index].generation < 5 for index in progress
                ):
                    launch = self.startup_entry
                n._check(
                    launch(n.stage0_handle, C.byref(n.bindings), n.stream.cuda_stream),
                    "in-flight Stage0",
                )
                self.commit_graph.replay()
                for index in progress:
                    self.slots[index].generation += 1
                self.producer_launches += 1
                self._diagnostic("producer_enqueued", slots=list(progress), repeats=repeats)
                self.trace.append(
                    {
                        "type": "producer",
                        "tick": self.tick_count,
                        "slots": list(progress),
                        "startup_batch_steps": repeats,
                        "host_monotonic_ns": time.monotonic_ns(),
                    }
                )
            self.meta_host.copy_(self.meta, non_blocking=True)
            self.meta_done.record(n.stream)
        self.observation_pending = True

    def _observe(self):
        if not self.observation_pending:
            return
        try:
            self._diagnostic("producer_wait_start")
            self.meta_done.synchronize()
            self._diagnostic("producer_wait_end")
        except BaseException:
            self.poisoned = True
            raise
        counts, done = self.meta_host.tolist()
        for index, slot in enumerate(self.slots):
            if slot is None or slot.finished:
                continue
            slot.produced = counts[index]
            if done[index]:
                slot.reason = "eos"
            elif slot.generation >= self.steps:
                slot.reason = "length_limit"
        self.observation_pending = False
        if getattr(self, "policy", DEFAULT_POLICY).playback_reserve_guard:
            self._playback_observed_producer()

    def _progress(self, requested):
        return [
            index
            for index in requested
            if (slot := self.slots[index]) is not None
            and not slot.finished
            and slot.reason is None
            and slot.produced - slot.consumed < slot.target(self.schedule)
        ]

    def _advance_producer(self, progress):
        now = time.perf_counter()
        policy = getattr(self, "policy", DEFAULT_POLICY)
        started = getattr(self, "startup_priority_started", None)
        defer = getattr(self, "startup_priority_deferred", False) or (
            started is not None and now - started >= policy.startup_budget_ms / 1000
        )
        selected = startup_progress(
            self.slots,
            progress,
            now,
            defer=defer or policy.producer_priority == "shared",
            align_phases=policy.startup_phase_alignment,
        )
        priority = selected != progress
        repeats = startup_steps(self.slots, selected, self.steps)
        if policy.partial_startup_single_step and any(slot is None for slot in self.slots):
            repeats = 1
        if policy.playback_reserve_guard:
            if priority:
                repeats = min(repeats, policy.startup_max_steps - self.startup_priority_steps)
            selected, repeats = self._playback_reserve_choice(progress, selected, repeats, now)
            priority = selected != progress
        if priority:
            if started is None:
                self.startup_priority_started = now
            remaining = policy.startup_max_steps - getattr(self, "startup_priority_steps", 0)
            repeats = min(repeats, remaining)
            self.startup_priority_steps = policy.startup_max_steps - remaining + repeats
            self.startup_priority_deferred = self.startup_priority_steps >= policy.startup_max_steps
            self.startup_priority_launches += repeats
        else:
            self.startup_priority_deferred = False
            self.startup_priority_steps = 0
            self.startup_priority_started = None
        if repeats == 1:
            self._step(selected)
        else:
            self._step(selected, repeats=repeats)

    def _submit_codec(self, frames, selected):
        reserve_started = (
            time.perf_counter()
            if getattr(self, "policy", DEFAULT_POLICY).playback_reserve_guard
            else None
        )
        t, n, e = self.t, self.neural, self.engine
        if getattr(self, "codec_ticket_rows", None) is not None:
            raise RuntimeError("codec row map cannot be reused before ticket completion")
        packed, lengths, index, offsets, mask, bindings = self.codec_buffers[frames]
        packing = getattr(self, "codec_packing", {}).get(frames)
        packed_transfer = getattr(self, "policy", DEFAULT_POLICY).codec_packed_transfer
        if packed_transfer and packing is None:
            raise RuntimeError("packed codec transfer requires an owned row map")
        order, inverse = (
            codec_row_order(selected, self.batch) if packing is not None else (None, None)
        )
        reordered = packing is not None and order != tuple(range(self.batch))
        physical_codes = packing["physical_codes"] if reordered else packed
        physical_lengths = packing["physical_lengths"] if reordered else lengths
        sizes = [0] * self.batch
        for lane in selected:
            slot = self.slots[lane]
            sizes[lane] = min(slot.target(self.schedule), slot.produced - slot.consumed)
        with t.cuda.stream(n.stream):
            if packing is not None and order != packing["staged_order"]:
                packing["order"].copy_(t.tensor(order, dtype=t.int64))
                packing["slots"].copy_(packing["order"])
                packing["staged_order"] = order
            physical_lengths.copy_(t.tensor(sizes, dtype=t.int32))
            t.add(e.consumed.view(self.batch, 1), offsets, out=index)
            index.clamp_(max=self.steps - 1)
            t.gather(
                e.queue, 1, index.unsqueeze(-1).expand(self.batch, frames, 16), out=physical_codes
            )
            t.lt(offsets, physical_lengths.view(self.batch, 1), out=mask)
            physical_codes.mul_(mask.unsqueeze(-1))
            e.consumed.add_(physical_lengths)
            if reordered:
                t.index_select(physical_codes, 0, packing["order"], out=packed)
                t.index_select(physical_lengths, 0, packing["order"], out=lengths)
            self.ready.record(n.stream)
        for lane in selected:
            self.slots[lane].consumed += sizes[lane]
        e.codec_stream.wait_event(self.ready)
        with t.cuda.stream(e.codec_stream):
            ticket = bindings[self.bank].enqueue()
        if packing is not None:
            self.codec_ticket_rows = (ticket, inverse, tuple(sizes), tuple(selected))
            self.codec_packing_launches += int(reordered)
        if packed_transfer:
            e.submit_pcm_transfer(ticket, row_count=len(selected))
        else:
            e.submit_pcm_transfer(ticket)
        self._diagnostic("codec_enqueued", slots=selected, frames=frames)
        self.codec_launches += 1
        self.trace.append(
            {
                "type": "codec",
                "tick": self.tick_count,
                "frames": frames,
                "slots": selected,
                "codec_reordered": reordered,
                "host_monotonic_ns": time.monotonic_ns(),
            }
        )
        if reserve_started is not None:
            self.playback_codec_pending = (
                ticket,
                reserve_started,
                frames,
                tuple(
                    (lane, self.slots[lane].epoch, self.slots[lane].request_id, sizes[lane])
                    for lane in selected
                ),
            )
        return ticket, sizes

    def _finish_codec(self, ticket, sizes, selected):
        import numpy as np

        owner = getattr(self, "codec_ticket_rows", None)
        inverse = range(self.batch)
        if owner is not None:
            if owner[0] is not ticket or owner[2:] != (tuple(sizes), tuple(selected)):
                raise RuntimeError("codec row map ticket ownership mismatch")
            inverse = owner[1]
        self._diagnostic("codec_wait_start", slots=selected)
        result = self.core.codec.finish(ticket, copy_pcm=True)
        self._diagnostic("codec_wait_end", slots=selected)
        pcm = result["pcm"].numpy()
        finite = result.get("finite")
        if finite is None:
            finite = bool(np.isfinite(pcm).all())
        if not finite:
            raise RuntimeError("in-flight codec produced nonfinite PCM")
        samples = result["samples"].cpu().tolist()
        if owner is not None:
            samples = [samples[row] for row in inverse]
        if samples != [size * SAMPLES_PER_FRAME for size in sizes]:
            raise RuntimeError("in-flight codec sample-count mismatch")
        if getattr(self, "policy", DEFAULT_POLICY).codec_packed_transfer:
            if pcm.ndim != 2 or any(
                not 0 <= inverse[lane] < pcm.shape[0] or samples[lane] > pcm.shape[1]
                for lane in selected
            ):
                raise RuntimeError("packed PCM prefix does not cover selected codec rows")
        self.bank = 1 - self.bank
        events = []
        for lane in selected:
            slot = self.slots[lane]
            sample_count = samples[lane]
            payload = pcm[inverse[lane], :sample_count].astype("<f4", copy=False).tobytes()
            now = time.perf_counter()
            if slot.first_audio is None:
                slot.first_audio = now
            events.append(
                {
                    "type": "chunk",
                    "slot": lane,
                    "epoch": slot.epoch,
                    "request_id": slot.request_id,
                    "chunk_index": slot.chunk_index,
                    "pcm_f32_le": payload,
                    "samples": sample_count,
                    "sample_rate": SAMPLE_RATE,
                }
            )
            slot.chunk_index += 1
            slot.samples += sample_count
        self._diagnostic("publication_ready", slots=selected)
        if owner is not None:
            self.codec_ticket_rows = None
        if getattr(self, "policy", DEFAULT_POLICY).playback_reserve_guard:
            self._playback_finished_codec(ticket)
        return events

    def _finish_events(self, requested):
        events = []
        for index in requested:
            slot = self.slots[index]
            if (
                slot is not None
                and not slot.finished
                and slot.reason is not None
                and slot.produced == slot.consumed
            ):
                slot.finished = True
                events.append(self._terminal(index, slot))
        return events

    def _terminal(self, index, slot):
        return {
            "type": "finish",
            "slot": index,
            "epoch": slot.epoch,
            "request_id": slot.request_id,
            "reason": slot.reason,
            "samples": slot.samples,
            "sample_rate": SAMPLE_RATE,
            "receipt": {
                "schema": "nemotron.inflight-request.v1",
                "physical_batch": self.batch,
                "request_id": slot.request_id,
                "slot": index,
                "epoch": slot.epoch,
                "seed": slot.seed,
                "generations": slot.generation,
                "engine_startup_priority_launches": self.startup_priority_launches,
                "engine_admission_yields": self.admission_yields,
                "audio_frames": slot.consumed,
                "samples": slot.samples,
                "reason": slot.reason,
                "rng": self.noise.provenance(),
                "stage0_sha256": self.neural.library_sha256,
                "engine_first_audio_ms": None
                if slot.first_audio is None
                else (slot.first_audio - slot.admitted) * 1000,
                "engine_completion_ms": (time.perf_counter() - slot.admitted) * 1000,
                "contraction": False,
            },
        }

    def _admission_can_progress(self):
        signal = getattr(self, "admission_requested", None)
        return (
            signal is not None
            and signal.is_set()
            and any(slot is None or slot.finished for slot in self.slots)
        )

    def _next_codec_group(self, ready):
        if getattr(self, "policy", DEFAULT_POLICY).codec_merge_ready and ready:
            self.codec_ready_order = deque()
            return [(max(frames for frames, _ in ready), [i for _, lanes in ready for i in lanes])]
        groups = dict(ready)
        order = deque(
            frames for frames in getattr(self, "codec_ready_order", ()) if frames in groups
        )
        order.extend(frames for frames, _ in ready if frames not in order)
        self.codec_ready_order = order
        deadline_priority = getattr(self, "policy", DEFAULT_POLICY).codec_deadline_priority
        if not order:
            if deadline_priority:
                self.codec_deadline_bypassed = False
            return []
        if deadline_priority:
            frames = order[0]
            if not getattr(self, "codec_deadline_bypassed", False):
                now = time.perf_counter()
                frames = min(
                    order,
                    key=lambda frame: min(
                        now
                        if self.slots[lane].first_audio is None
                        else self.slots[lane].first_audio + self.slots[lane].samples / SAMPLE_RATE
                        for lane in groups[frame]
                    ),
                )
            self.codec_deadline_bypassed = frames != order[0]
            order.remove(frames)
            return [(frames, groups[frames])]
        frames = order.popleft()
        return [(frames, groups[frames])]

    def advance(self, progress_slots, *, allow_admission_yield=True):
        self._check_open()
        if type(allow_admission_yield) is not bool:
            raise ValueError("admission yielding must be boolean")
        requested = list(progress_slots)
        if len(set(requested)) != len(requested) or any(
            type(i) is not int or not 0 <= i < self.batch for i in requested
        ):
            raise ValueError("progress slots must be unique physical indices")
        self.tick_count += 1
        events = []
        first_packet_f2 = getattr(self, "policy", DEFAULT_POLICY).codec_first_packet_f2
        try:
            self._observe()
            if allow_admission_yield and self._admission_can_progress():
                self.admission_yields += 1
                return self._finish_events(requested)
            ready = ready_groups(
                self.slots, requested, self.schedule, first_packet_f2=first_packet_f2
            )
            if not ready:
                progress = self._progress(requested)
                if progress:
                    self.engine._set_stage0_grid(overlapped=False)
                    self._advance_producer(progress)
                    self._observe()
                ready = ready_groups(
                    self.slots, requested, self.schedule, first_packet_f2=first_packet_f2
                )
            if ready:
                self._diagnostic("codec_ready", groups=ready)
            prefetched = False
            for frames, selected in self._next_codec_group(ready):
                ticket, sizes = self._submit_codec(frames, selected)
                progress = self._progress(requested)
                mode = self.engine.execution["mode"]
                serial_prefetch = (
                    mode == "serial" and getattr(self, "policy", DEFAULT_POLICY).serial_prefetch
                )
                prefetch = (
                    bool(progress)
                    and not prefetched
                    and not self._admission_can_progress()
                    and (mode == "overlap" or serial_prefetch)
                )
                if prefetch:
                    if serial_prefetch:
                        self.neural.stream.wait_event(ticket.event)
                    self.engine._set_stage0_grid(overlapped=not serial_prefetch)
                    self._advance_producer(progress)
                    self.engine._set_stage0_grid(overlapped=False)
                    prefetched = True
                events.extend(self._finish_codec(ticket, sizes, selected))
            events.extend(self._finish_events(requested))
            return events
        except BaseException:
            self.poisoned = True
            raise

    def cancel(self, slot, epoch):
        value = self._slot(slot, epoch)
        self._observe()
        self.neural.stream.synchronize()
        self.engine.codec_stream.synchronize()
        value.reason = "cancelled"
        value.finished = True
        return self._terminal(slot, value)

    def retire(self, slot, epoch):
        self.retire_many([(slot, epoch)])

    def retire_many(self, records):
        records = list(records)
        if len({slot for slot, _ in records}) != len(records):
            raise ValueError("retirement slots must be unique")
        values = [(slot, epoch, self._slot(slot, epoch)) for slot, epoch in records]
        if not values:
            return
        self._observe()
        if any(not value.finished for _, _, value in values):
            raise RuntimeError("cannot retire a live request")
        self._diagnostic("retirement_wait_start", slots=[slot for slot, _, _ in values])
        self.neural.stream.synchronize()
        self.engine.codec_stream.synchronize()
        self._diagnostic("retirement_wait_end")
        for slot, epoch, value in values:
            self.trace.append(
                {
                    "type": "retire",
                    "slot": slot,
                    "epoch": epoch,
                    "request_id": value.request_id,
                    "host_monotonic_ns": time.monotonic_ns(),
                }
            )
            self.slots[slot] = None

    def receipt(self):
        return {
            "schema": "nemotron.inflight-engine.v1",
            "physical_batch": self.batch,
            "stage0_sha256": self.neural.library_sha256,
            "producer_launches": self.producer_launches,
            "codec_launches": self.codec_launches,
            "live_refills": self.live_refills,
            "admission_yields": self.admission_yields,
            "startup_priority_launches": self.startup_priority_launches,
            "policy": self.policy.record(),
            "startup_priority": "first audio only with at least 40ms peer playback lead",
            "max_slots": self.batch,
            "admission": "indexed masked reset graph with compact pinned staging"
            if hasattr(self, "indexed_reset")
            else "exact dyadic slot-range CUDA reset graphs with owned pinned staging",
            "reset_graphs": 1 if hasattr(self, "indexed_reset") else len(self.admission_graphs),
            "noise_dispatch": "captured with input preparation",
            "trace_boundary": "host admission, enqueue and retirement; not GPU execution timing",
            "rng": self.noise.provenance(),
            "trace": list(self.trace),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.neural.stream.synchronize()
            self.engine.codec_stream.synchronize()
        finally:
            try:
                for name in ("prepare_graph", "commit_graph"):
                    graph = getattr(self, name, None)
                    if graph is not None:
                        graph.reset()
                for graph in getattr(self, "admission_graphs", {}).values():
                    graph.reset()
                indexed = getattr(self, "indexed_reset", None)
                if indexed is not None:
                    indexed.close()
            finally:
                self.engine.close()
