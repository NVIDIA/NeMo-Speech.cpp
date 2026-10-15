from __future__ import annotations

import ctypes as C
import dataclasses
import hashlib
import json
import operator
import os
import threading
import time
from pathlib import Path

SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 1764
WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
MAX_STEPS = 511
_CONSTRUCTION_LOCK = threading.RLock()


def effective_seed(seed, epoch):
    if type(seed) is not int or not 0 <= seed <= 9223372036854775807:
        raise ValueError("seed must be an unsigned 63-bit integer")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    return seed + epoch - 1 & 9223372036854775807


def generation_deficit(counts, consumed, done, target, executed, limit):
    if not len(counts) == len(consumed) == len(done):
        raise ValueError("queue metadata widths differ")
    if type(target) is not int or not 1 <= target <= 8 or not 0 <= executed <= limit:
        raise ValueError("invalid generation/chunk bounds")
    if True in done:
        pending = [
            n - used
            for (n, used, terminal) in zip(counts, consumed, done, strict=False)
            if not terminal
        ]
        need = max(0, target - min(pending)) if pending else 0
    else:
        need = max(0, target - min(map(operator.sub, counts, consumed)))
    return min(need, limit - executed)


def continuation_lanes(counts, consumed, held):
    return [
        lane
        for (lane, (count, used, pending)) in enumerate(zip(counts, consumed, held, strict=False))
        if pending and count > used
    ]


def publication_probe_steps(needed, held, done):
    return (
        min(needed, 1)
        if any(pending and (not terminal) for (pending, terminal) in zip(held, done, strict=False))
        else needed
    )


def validate_execution(row):
    mode = row.get("mode", "serial")
    policy = row.get("grid_policy", "static")
    if mode not in ("serial", "overlap", "after-first-publication"):
        raise ValueError("unsupported overlap mode")
    if policy not in ("static", "overlap-only"):
        raise ValueError("unsupported grid policy")
    out = dict(
        mode=mode,
        grid_policy=policy,
        stage0_grid=row.get("stage0_grid", 0),
        codec_grid=row.get("codec_grid", 0),
    )
    for key in ("stage0_grid", "codec_grid"):
        if type(out[key]) is not int or out[key] < 0:
            raise ValueError("grid must be a nonnegative integer")
    if mode != "serial" and (not out["stage0_grid"] or not out["codec_grid"]):
        raise ValueError("overlap requires explicit bounded grids")
    if policy == "overlap-only":
        full = row.get("full_stage0_grid")
        if type(full) is not int or full < out["stage0_grid"] or full < 1:
            raise ValueError("overlap-only requires a full grid at least as large as overlap grid")
        out["full_stage0_grid"] = full
    return out


def should_prefetch(
    mode, counts, consumed, done, held, sizes, first_published, next_frames, executed, limit
):
    if mode == "serial" or (mode == "after-first-publication" and (not all(first_published))):
        return False
    future_held = [pending or size > 0 for (pending, size) in zip(held, sizes, strict=False)]
    for lane in continuation_lanes(counts, consumed, future_held):
        future_held[lane] = False
    needed = generation_deficit(counts, consumed, done, next_frames, executed, limit)
    return publication_probe_steps(needed, future_held, done) == 1


class Stage0BuildInfo(C.Structure):
    _fields_ = [
        (name, C.c_int)
        for name in (
            "version",
            "batch",
            "batch_pad",
            "stages",
            "kchunk",
            "direct_input",
            "b1_local_barriers",
            "skip_hidden_snapshot",
            "threads",
            "dynamic_smem",
            "static_smem",
            "registers",
            "device",
            "sms",
            "blocks_per_sm",
            "grid",
        )
    ]


def inspect_residency(core, execution):
    n = core.neural
    n.lib.mk2_build_info.argtypes = [C.POINTER(Stage0BuildInfo)]
    n.lib.mk2_build_info.restype = C.c_int
    info = Stage0BuildInfo()
    n._check(n.lib.mk2_build_info(C.byref(info)), "overlap Stage0 build information")
    actual_codec = core.codec.lib.native_stage1_resident_blocks(core.codec.handle)
    props = core.t.cuda.get_device_properties(0)
    if info.sms != props.multi_processor_count:
        raise RuntimeError("stage0/runtime device SM count mismatch")
    expected_stage0 = (
        execution.get("full_stage0_grid")
        if execution["grid_policy"] == "overlap-only"
        else execution["stage0_grid"]
    )
    if expected_stage0 and info.grid != expected_stage0:
        raise RuntimeError("requested Stage0 grid was clamped or ignored")
    if execution["codec_grid"] and actual_codec != execution["codec_grid"]:
        raise RuntimeError("requested codec grid was ignored")
    if execution["mode"] != "serial":
        if info.blocks_per_sm != 1:
            raise RuntimeError("experiment currently qualifies one Stage0 CTA per SM only")
        if execution["stage0_grid"] + actual_codec > info.sms:
            raise RuntimeError("persistent grids exceed distinct-SM capacity")
    if execution["grid_policy"] == "overlap-only":
        setter = n.lib.mk2_set_native_stage0_grid
        setter.argtypes = [C.c_void_p, C.c_int]
        setter.restype = C.c_int
    return dict(
        stage0={name: getattr(info, name) for (name, _) in info._fields_},
        codec_blocks=actual_codec,
        device_name=props.name,
        distinct_sm_capacity_checked=execution["stage0_grid"] + actual_codec <= info.sms,
        progress_guarantee="finite independently resident grids; no inter-kernel device "
        "wait; actual overlap "
        "requires GPU evidence",
    )


def select_retained_rows(done, target):
    if not done or type(target) is not int or (not 0 < target < len(done)):
        raise ValueError("invalid width contraction")
    if any(type(v) is not bool for v in done):
        raise ValueError("done must contain host booleans")
    live = [i for (i, v) in enumerate(done) if not v]
    if not 0 < len(live) <= target:
        return None
    return live + [i for (i, v) in enumerate(done) if v][: target - len(live)]


def validate_active_lanes(row, batch, execution):
    if row is None or row.get("mode") == "disabled":
        return None
    if row.get("mode") != "fixed-subset" or row.get("target_batch") != 16 or batch not in (32, 64):
        raise ValueError("contraction supports B32/B64 to B16 only")
    if row.get("after_chunk0") is not True:
        raise ValueError("contraction must preserve the chunk0 path")
    grid = row.get("narrow_grid")
    if type(grid) is not int or grid <= 0:
        raise ValueError("narrow grid must be a positive integer")
    policy = row.get("grid_policy", "static")
    if policy not in ("static", "follow-stage0"):
        raise ValueError("unsupported narrow grid policy")
    if policy == "follow-stage0" and (
        execution.get("grid_policy") != "overlap-only" or grid != execution.get("stage0_grid")
    ):
        raise ValueError(
            "dynamic narrow grid must follow an explicit bounded Stage0 execution grid"
        )
    for name in ("stage0", "transfer"):
        artifact = row.get(name, {})
        if (
            set(artifact) != {"path", "sha256"}
            or not isinstance(artifact["path"], str)
            or len(artifact["sha256"]) != 64
        ):
            raise ValueError("auxiliary library requires exact path and SHA256")
    return dict(row)


class ActiveLaneContraction:
    def __init__(self, neural, config, root, codec_grid, info_type, execution):
        from .split_native_runtime import bind_stage0_library

        self.n = neural
        self.t = neural.torch
        self.config = config
        self.handle = None
        self._original_step = neural.step
        self._hook = None
        self.enabled = False
        self.selected_rows = None
        self.transition = None
        self.widths = []
        self.grids = []
        self.full_width = neural.batch
        self.target = config["target_batch"]
        self.grid = config["narrow_grid"]
        self.grid_policy = config.get("grid_policy", "static")
        self.full_grid = (
            execution.get("full_stage0_grid", self.grid)
            if self.grid_policy == "follow-stage0"
            else self.grid
        )
        self.current_grid = self.full_grid
        self.artifacts = {}
        for key in ("stage0", "transfer"):
            row = config[key]
            path = Path(row["path"])
            if not path.is_absolute():
                path = Path(root) / path
            path = path.resolve(strict=True)
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got != row["sha256"]:
                raise ValueError(f"active-lane {key} library hash mismatch")
            self.artifacts[key] = {"path": str(path), "sha256": got}
        self.lib = bind_stage0_library(self.artifacts["stage0"]["path"])
        self.transfer = C.CDLL(self.artifacts["transfer"]["path"])
        binding_type = type(neural.bindings)
        self.transfer.native_lanes_bindings_size.argtypes = []
        self.transfer.native_lanes_bindings_size.restype = C.c_size_t
        if self.transfer.native_lanes_bindings_size() != C.sizeof(binding_type):
            raise RuntimeError("lane-transfer ABI layout mismatch")
        for name in ("native_lanes_gather", "native_lanes_scatter"):
            fn = getattr(self.transfer, name)
            fn.argtypes = [
                C.POINTER(binding_type),
                C.POINTER(binding_type),
                C.c_void_p,
                C.c_int,
                C.c_int,
                C.c_void_p,
            ]
            fn.restype = C.c_int
        self.lib.mk2_build_info.argtypes = [C.POINTER(info_type)]
        self.lib.mk2_build_info.restype = C.c_int
        old_grid = os.environ.get("NEMOTRON_STAGE0_MEGAKERNEL_GRID")
        os.environ["NEMOTRON_STAGE0_MEGAKERNEL_GRID"] = str(self.full_grid)
        try:
            self.handle = self.lib.mk2_create_stage0(
                C.byref(neural.pw), C.byref(neural.bw), self.target
            )
            neural._require_handle(self.handle, "contracted Stage0 create", self.lib)
            info = info_type()
            neural._check(
                self.lib.mk2_build_info(C.byref(info)), "contracted Stage0 resources", self.lib
            )
            if (
                self.lib.mk2_batch(self.handle) != self.target
                or info.batch != self.target
                or info.grid != self.full_grid
            ):
                raise RuntimeError("contracted Stage0 width or requested grid mismatch")
            if info.blocks_per_sm != 1 or (
                execution["mode"] != "serial" and self.grid + codec_grid > info.sms
            ):
                raise RuntimeError(
                    "contracted persistent grids exceed verified distinct-SM capacity"
                )
            self.residency = {name: getattr(info, name) for (name, _) in info._fields_}
            if self.grid_policy == "follow-stage0":
                self.lib.mk2_set_native_stage0_grid.argtypes = [C.c_void_p, C.c_int]
                self.lib.mk2_set_native_stage0_grid.restype = C.c_int
        except BaseException:
            if self.handle:
                self.lib.mk2_destroy(self.handle)
                self.handle = None
            raise
        finally:
            if old_grid is None:
                os.environ.pop("NEMOTRON_STAGE0_MEGAKERNEL_GRID", None)
            else:
                os.environ["NEMOTRON_STAGE0_MEGAKERNEL_GRID"] = old_grid
        try:
            t = self.t
            width = self.target
            blocks = neural.bindings.kv_max_blocks_per_seq

            def empty(shape, dtype):
                return t.empty(shape, dtype=dtype, device="cuda:0")

            self.rows = empty((width,), t.int32)
            self.embedding = empty((width, 1536), t.bfloat16)
            self.noise = empty((width, 16, 80), t.float32)
            self.codes = empty((width, 16), t.int64)
            self.phones = empty((width,), t.int32)
            self.seq = empty((width,), t.int32)
            self.slots = empty((width,), t.int32)
            self.tables = [empty((width, 2, blocks), t.int32) for _ in range(3)]
            self.bindings = binding_type.from_buffer_copy(neural.bindings)
            for field, tensor in (
                ("embedding", self.embedding),
                ("gumbel", self.noise),
                ("codes_out", self.codes),
                ("phoneme_out", self.phones),
                ("seq_len", self.seq),
                ("slot_map", self.slots),
            ):
                setattr(self.bindings, field, tensor.data_ptr())
            for i, table in enumerate(self.tables):
                self.bindings.kv_block_table[i] = table.data_ptr()
            neural.stream.wait_stream(t.cuda.current_stream(0))
            self._hook = self.step
            neural.step = self._hook
        except BaseException:
            self.close()
            raise

    def reset(self):
        self.enabled = False
        self.selected_rows = None
        self.transition = None
        self.widths = []
        self.grids = []

    def set_requested_grid(self, grid):
        if self.grid_policy != "follow-stage0":
            return
        if grid not in (self.grid, self.full_grid):
            raise ValueError("producer requested an unqualified narrow launch grid")
        if grid != self.current_grid:
            self.n._check(
                self.lib.mk2_set_native_stage0_grid(self.handle, grid),
                "set contracted launch grid",
                self.lib,
            )
            self.current_grid = grid

    def activate(self, rows, generation, *, reason):
        if self.enabled or len(rows) != self.target or len(set(rows)) != self.target:
            raise ValueError("activation requires a fresh unique fixed subset")
        if any(type(i) is not int or not 0 <= i < self.full_width for i in rows):
            raise ValueError("invalid original logical row")
        with self.t.cuda.stream(self.n.stream):
            self.rows.copy_(self.t.tensor(rows, dtype=self.t.int32))
        self.selected_rows = list(rows)
        self.enabled = True
        self.transition = dict(
            generation=generation, original_logical_rows=list(rows), reason=reason
        )

    def observe(self, done, generation, chunk0_seen):
        if self.enabled or not chunk0_seen:
            return
        rows = select_retained_rows(done, self.target)
        if rows is not None:
            self.activate(
                rows,
                generation,
                reason="observed EOS after chunk0; retain live rows plus distinct dead padding",
            )

    def step(self):
        n = self.n
        if not self.enabled:
            result = self._original_step()
            self.widths.append(self.full_width)
            self.grids.append(None)
            return result
        if (
            n.closed
            or not n.started
            or n.inputs.generation >= min(n.cache["max_generations"], len(n.step_events))
        ):
            raise RuntimeError("contracted step exceeds prepared diagnostic capacity")
        with self.t.cuda.stream(n.stream):
            n.inputs.prepare_decode()
            n.noise.generation(n.batch)
            args = (
                C.byref(n.bindings),
                C.byref(self.bindings),
                self.rows.data_ptr(),
                self.full_width,
                self.target,
                n.stream.cuda_stream,
            )
            rc = self.transfer.native_lanes_gather(*args)
            if rc:
                raise RuntimeError(f"lane gather CUDA error {rc}")
            (a, b) = n.step_events[n.inputs.generation]
            a.record(n.stream)
            n._check(
                self.lib.mk2_run_native_stage0(
                    self.handle, C.byref(self.bindings), n.stream.cuda_stream
                ),
                "contracted full Stage0",
                self.lib,
            )
            b.record(n.stream)
            rc = self.transfer.native_lanes_scatter(*args)
            if rc:
                raise RuntimeError(f"lane scatter CUDA error {rc}")
            n.inputs.commit_outputs(n.codes, n.phonemes)
        n.history.append((a, b))
        self.widths.append(self.target)
        self.grids.append(self.current_grid)
        return (n.codes, n.phonemes, n.inputs.stop)

    def receipt(self):
        return dict(
            mode="fixed-subset",
            batch=self.full_width,
            target_batch=self.target,
            after_chunk0=True,
            narrow_grid=self.grid,
            grid_policy=self.grid_policy,
            full_narrow_grid=self.full_grid,
            transition=self.transition,
            stage0_widths=self.widths,
            stage0_grids=self.grids,
            contracted_generations=self.widths.count(self.target),
            artifacts=self.artifacts,
            residency=self.residency,
            rng="unchanged full64 draws; gather original logical rows",
            inactive_outputs="dropped dead rows codes=0 and phonemes=0; live and padding "
            "rows scattered "
            "before unchanged original-width helper graph",
            state="original conv/SSM/KV pool addresses; gathered original slot-map and KV-table "
            "rows; canonical live seq lengths scattered",
            scope="two auxiliary transfer kernels per contracted generation; one full native "
            "Stage0 neural kernel; unchanged request membership and original-width codec",
        )

    def close(self):
        if self.handle:
            self.n.stream.synchronize()
            self.lib.mk2_destroy(self.handle)
            self.handle = None
        if self._hook is not None:
            if self.n.step is not self._hook:
                raise RuntimeError("contracted step hook ownership changed")
            self.n.step = self._original_step
            self._hook = None


class NativeCohortEngine:
    def __init__(
        self,
        batch,
        model_root,
        steps=MAX_STEPS,
        retain_codes=False,
        *,
        support,
        artifact_root,
        execution=None,
        stage0_library=None,
        codec_library=None,
        prefill_library=None,
        qualification_mode=False,
        serial_match_mode="overlap",
        active_lanes=None,
        research_schedule=(3, 5),
    ):
        if batch not in WIDTHS or not 6 <= steps <= MAX_STEPS:
            raise ValueError("requires B1/B16/B32/B64 and 6..511 generation steps")
        if type(research_schedule) is not tuple or research_schedule not in (
            (3, 5),
            (2, 2, 2, 4),
            (1, 2, 2, 3),
            (2, 1, 5),
            (2, 2, 4),
            (2, 3, 4, 5),
        ):
            raise ValueError("unsupported research chunk schedule")
        if research_schedule != (3, 5) and qualification_mode is not True:
            raise ValueError("changed packet contract is qualification-only")
        self.research_schedule = research_schedule
        self.batch = batch
        self.steps = steps
        self.retain_codes = bool(retain_codes)
        self._lock = threading.Lock()
        self._closed = False
        self.last_receipt = None
        self.core = None
        if (execution is not None or stage0_library or codec_library or prefill_library) and (
            not qualification_mode
        ):
            raise ValueError("explicit unqualified overrides require qualification_mode=True")
        self.execution = validate_execution(execution if execution is not None else {})
        self.active_lanes_config = validate_active_lanes(active_lanes, batch, self.execution)
        if active_lanes is not None and (not qualification_mode):
            raise ValueError("explicit active-lane override requires qualification_mode=True")
        self.contraction = None
        self.qualification_mode = bool(qualification_mode)
        if serial_match_mode not in ("overlap", "after-first-publication"):
            raise ValueError("invalid serial grid reference mode")
        if serial_match_mode != "overlap" and (not qualification_mode):
            raise ValueError("serial grid reference override is qualification-only")
        self._serial_grid_reference_mode = serial_match_mode
        self._codec_pending = None
        self._deferred_serial_grid = False
        self._overlap_records = []
        self.s = support
        self.s.verify_sources()
        (self.paths, self.selection) = self.s.select(batch)
        self.startup_predictor_skip = False
        self.helper_graphs = None
        for name, override in (
            ("stage0", stage0_library),
            ("stage1", codec_library),
            ("prefill", prefill_library),
        ):
            if override:
                self.paths[name] = Path(override).resolve(strict=True)
        self._actual_library_records = {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for (name, path) in self.paths.items()
        }
        self._support_source_sha256 = self.s.source_sha256
        self._engine_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.assets = self.s.ModelAssets(Path(model_root))
        try:
            with _CONSTRUCTION_LOCK:
                old_grid = os.environ.get("NEMOTRON_STAGE0_MEGAKERNEL_GRID")
                initial_grid = (
                    self.execution.get("full_stage0_grid")
                    if self.execution["grid_policy"] == "overlap-only"
                    else self.execution["stage0_grid"]
                )
                os.environ["NEMOTRON_STAGE0_MEGAKERNEL_GRID"] = str(initial_grid)
                try:
                    self.core = self.s.Cohort(
                        self.assets,
                        batch,
                        self.paths,
                        "Warming the first request path.",
                        steps,
                        codec_grid=self.execution["codec_grid"],
                    )
                    self.residency = inspect_residency(self.core, self.execution)
                finally:
                    if old_grid is None:
                        os.environ.pop("NEMOTRON_STAGE0_MEGAKERNEL_GRID", None)
                    else:
                        os.environ["NEMOTRON_STAGE0_MEGAKERNEL_GRID"] = old_grid
            self.t = self.core.t
            self._allocate_queues()
            if self.active_lanes_config:
                self.contraction = ActiveLaneContraction(
                    self.core.neural,
                    self.active_lanes_config,
                    artifact_root,
                    self.residency["codec_blocks"],
                    Stage0BuildInfo,
                    self.execution,
                )
        except BaseException:
            if self.contraction is not None:
                self.contraction.close()
            if self.core is not None:
                self.core.close()
            raise

    def _allocate_queues(self):
        (t, c, B) = (self.t, self.core, self.batch)
        self.queue = t.zeros((B, self.steps, 16), dtype=t.int64, device="cuda:0")
        self.counts = t.zeros(B, dtype=t.int64, device="cuda:0")
        self.consumed = t.zeros_like(self.counts)
        self.consumed_host = [0] * B
        self.element_valid = t.empty((B, 16), dtype=t.bool, device="cuda:0")
        self.element_tmp = t.empty_like(self.element_valid)
        self.valid = t.empty(B, dtype=t.bool, device="cuda:0")
        self.nonzero = t.empty_like(self.valid)
        self.active = t.empty_like(self.valid)
        self.valid_i64 = t.empty_like(self.counts)
        self.write_index = t.empty_like(self.counts)
        self._write_index3 = self.write_index.view(B, 1, 1).expand(B, 1, 16)
        self._step_codes3 = c.neural.codes.view(B, 1, 16)
        self._done_event = t.cuda.Event()
        self._done_event.record()
        self._observe_done = t.empty(B, dtype=t.bool, pin_memory=True)
        self._observe_counts = t.empty(B, dtype=t.int64, pin_memory=True)
        self._observe_event = t.cuda.Event()
        self._observe_event.record()
        self.codec_stream = t.cuda.Stream(device=0)
        self.codec_stream.wait_stream(t.cuda.current_stream(0))
        self._origin = t.cuda.Event(enable_timing=True)
        self._origin.record()
        self._event_pool = [
            tuple(t.cuda.Event(enable_timing=True) for _ in range(3)) for _ in range(self.steps)
        ]
        for triple in self._event_pool:
            for event in triple:
                event.record()
        self._chunk_events = []
        self.bindings = {}
        for F in sorted(set((*self.research_schedule, 8))):
            packed = t.empty((B, F, 16), dtype=t.int64, device="cuda:0")
            lengths = t.empty(B, dtype=t.int32, device="cuda:0")
            index = t.empty((B, F), dtype=t.int64, device="cuda:0")
            offsets = t.arange(F, dtype=t.int64, device="cuda:0").view(1, F)
            mask = t.empty((B, F), dtype=t.bool, device="cuda:0")
            (a, b) = (t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True))
            a.record()
            b.record()
            pcm = c.pcm[: B * F * SAMPLES_PER_FRAME].view(B, F * SAMPLES_PER_FRAME)
            banks = [
                c.codec.bind_buffers(
                    packed,
                    lengths,
                    c.slots,
                    c.banks[k],
                    c.banks[1 - k],
                    pcm,
                    c.samples,
                    c.status,
                    stream=self.codec_stream,
                    completion_event=b,
                )
                for k in (0, 1)
            ]
            self.bindings[F] = (packed, lengths, index, offsets, mask, a, b, banks)
        c.neural.stream.wait_stream(t.cuda.current_stream(0))
        t.cuda.synchronize(0)

    def _requests(self, texts, token_ids, speaker):
        if speaker != "english_sample":
            raise ValueError("selected native assets support english_sample only")
        if not isinstance(texts, (list, tuple)) or len(texts) != self.batch:
            raise ValueError("texts must contain exactly the configured batch width")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must be nonempty normalized strings")
        if token_ids is None:
            return self.s.requests_from_texts(self.assets, list(texts))
        from .. import r31
        from .native_inputs import Request, validate_requests

        if len(token_ids) != self.batch:
            raise ValueError("token_ids width differs from texts")
        context = self.core.requests[0].context_ids
        requests = [
            Request(text, tuple(ids), context)
            for (text, ids) in zip(texts, token_ids, strict=False)
        ]
        if r31.enabled("staged_admission"):
            if (
                len(context) != 3
                or any(type(x) is not int or not 0 <= x < 131075 for x in context)
                or any(
                    not 1 <= len(r.target_ids) <= 512 or r.target_ids[-1] != 131073
                    for r in requests
                )
            ):
                raise ValueError("invalid native request context/target/EOS contract")
            return requests
        validate_requests(requests)
        return requests

    def _reset(self, requests, seed):
        (t, c, n) = (self.t, self.core, self.core.neural)
        tokens = t.zeros((self.batch, 512), dtype=t.int64)
        lengths = t.tensor([len(r.target_ids) for r in requests], dtype=t.int64)
        for lane, request in enumerate(requests):
            tokens[lane, : len(request.target_ids)] = t.tensor(request.target_ids, dtype=t.int64)
        c.requests = requests
        n.inputs.requests = tuple(requests)
        if self.contraction is not None:
            self.contraction.reset()
        c.reset(seed)
        with t.cuda.stream(n.stream):
            n.inputs.text_tokens.copy_(tokens)
            n.inputs.text_lengths.copy_(lengths)
            n.inputs.max_text_offsets.copy_(lengths - 1)
            self.counts.zero_()
            self.consumed.zero_()
        self.consumed_host = [0] * self.batch
        self._chunk_events = []
        self._overlap_records = []
        self._deferred_serial_grid = False
        self._origin.record(n.stream)

    def _append_step(self, generation):
        (t, c, n) = (self.t, self.core, self.core.neural)
        restore_grid = self._deferred_serial_grid
        if restore_grid:
            self._set_stage0_grid(overlapped=True)
            self._deferred_serial_grid = False
        (codes, phones, stop) = n.step()
        if restore_grid:
            self._set_stage0_grid(overlapped=False)
        c.raw_cols[generation].copy_(codes)
        c.stop_cols[generation].copy_(stop)
        c.phone_cols[generation].copy_(phones)
        t.ge(codes, 0, out=self.element_valid)
        t.lt(codes, 1024, out=self.element_tmp)
        t.logical_and(self.element_valid, self.element_tmp, out=self.element_valid)
        t.all(self.element_valid, dim=1, out=self.valid)
        t.ne(codes, 0, out=self.element_tmp)
        t.any(self.element_tmp, dim=1, out=self.nonzero)
        t.logical_and(self.valid, self.nonzero, out=self.valid)
        t.logical_not(c.done, out=self.active)
        t.logical_and(self.valid, self.active, out=self.valid)
        t.clamp(self.counts, max=self.steps - 1, out=self.write_index)
        self.queue.scatter_(1, self._write_index3, self._step_codes3)
        self.valid_i64.copy_(self.valid)
        self.counts.add_(self.valid_i64)
        t.logical_or(c.done, stop, out=c.done)

    def _observe(self, first_eos, start, end):
        (c, n) = (self.core, self.core.neural)
        if c.stops.dtype is not self.t.bool or c.done.dtype is not self.t.bool:
            raise RuntimeError("EOS observations must be boolean")
        self._done_event.record(n.stream)
        self._done_event.synchronize()
        self._observe_done.copy_(c.done, non_blocking=True)
        self._observe_counts.copy_(self.counts, non_blocking=True)
        self._observe_event.record()
        self._observe_event.synchronize()
        done = self._observe_done.tolist()
        if True in done:
            rows = c.stops[:, start:end].cpu().tolist()
            first_eos = self.s.update_first_eos(first_eos, rows, start)
        else:
            first_eos = list(first_eos)
        if done != [g is not None for g in first_eos]:
            raise RuntimeError("native EOS carry mismatch")
        if self.contraction is not None:
            self.contraction.observe(done, end, bool(self._chunk_events))
        return (first_eos, done, self._observe_counts.tolist())

    def _set_stage0_grid(self, *, overlapped):
        if self.execution["grid_policy"] != "overlap-only":
            return
        grid = self.execution["stage0_grid"] if overlapped else self.execution["full_stage0_grid"]
        n = self.core.neural
        n._check(
            n.lib.mk2_set_native_stage0_grid(n.stage0_handle, grid), "set finite Stage0 launch grid"
        )
        if self.contraction is not None:
            self.contraction.set_requested_grid(grid)

    def _submit_chunk(self, F, counts, bank):
        if self._codec_pending is not None:
            raise RuntimeError("codec buffer/scratch lease still outstanding")
        (t, n) = (self.t, self.core.neural)
        (packed, lengths, index, offsets, mask, _, _, banks) = self.bindings[F]
        (ready, a, b) = self._event_pool[len(self._chunk_events)]
        sizes = [
            min(F, available - used)
            for (available, used) in zip(counts, self.consumed_host, strict=False)
        ]
        with t.cuda.stream(n.stream):
            lengths.copy_(t.tensor(sizes, dtype=t.int32))
            t.add(self.consumed.view(self.batch, 1), offsets, out=index)
            index.clamp_(max=self.steps - 1)
            t.gather(self.queue, 1, index.unsqueeze(-1).expand(self.batch, F, 16), out=packed)
            t.lt(offsets, lengths.view(self.batch, 1), out=mask)
            packed.mul_(mask.unsqueeze(-1))
            self.consumed.add_(lengths)
            ready.record(n.stream)
        self.consumed_host = [
            used + size for (used, size) in zip(self.consumed_host, sizes, strict=False)
        ]
        self.codec_stream.wait_event(ready)
        with t.cuda.stream(self.codec_stream):
            a.record(self.codec_stream)
            banks[bank].event = b
            ticket = banks[bank].enqueue()
        pending = dict(ticket=ticket, sizes=sizes, events=(a, b))
        self._codec_pending = pending
        self._chunk_events.append((a, b))
        return pending

    def _finish_chunk(self, pending):
        if pending is not self._codec_pending:
            raise RuntimeError("wrong codec ticket")
        result = self.core.codec.finish(pending["ticket"], copy_pcm=True)
        self._codec_pending = None
        samples = result["samples"].cpu().tolist()
        if samples != [size * SAMPLES_PER_FRAME for size in pending["sizes"]]:
            raise RuntimeError("native codec sample counts differ from valid-frame queue")
        (a, b) = pending["events"]
        return (result, pending["sizes"], samples, a.elapsed_time(b))

    def _overlap_receipt(self):
        rows = []
        for row in self._overlap_records:
            (a, b) = row["codec_events"]
            (c0, c1) = (self._origin.elapsed_time(a), self._origin.elapsed_time(b))
            generation = row["stage0_generation"]
            item = dict(
                chunk_index=row["chunk_index"],
                stage0_generation=generation,
                codec_start_ms=c0,
                codec_end_ms=c1,
                event_interval_overlap_ms=0.0,
            )
            if generation is not None:
                (x, y) = self.core.neural.step_events[generation]
                (s0, s1) = (self._origin.elapsed_time(x), self._origin.elapsed_time(y))
                item.update(
                    stage0_start_ms=s0,
                    stage0_end_ms=s1,
                    event_interval_overlap_ms=max(0.0, min(c1, s1) - max(c0, s0)),
                )
            rows.append(item)
        return dict(
            rows=rows,
            prefetched_generations=sum(r["stage0_generation"] is not None for r in rows),
            event_interval_overlap_ms=sum(r["event_interval_overlap_ms"] for r in rows),
            caveat="CUDA event intervals include admission wait; use an Nsight timeline to prove "
            "simultaneous kernel residency",
        )

    def stream(self, texts, *, token_ids=None, speaker="english_sample", seed=0, epoch=1):
        seed_value = effective_seed(seed, epoch)
        requests = self._requests(texts, token_ids, speaker)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("native engine already has a live cohort")
        try:
            if self._closed:
                raise RuntimeError("native engine is closed")
            yield from self._stream(requests, speaker, seed, epoch, seed_value)
            self._r31_predraw(effective_seed(seed, epoch + 1))
        finally:
            try:
                if self.core is not None:
                    self.core.neural.stream.synchronize()
                    if self._codec_pending is not None:
                        self.core.codec.finish(self._codec_pending["ticket"], copy_pcm=False)
                        self._codec_pending = None
                    self.codec_stream.synchronize()
            finally:
                self._lock.release()

    def _r31_predraw(self, seed):
        from .. import r31

        if not r31.enabled("rng_predraw") or self.core is None:
            return
        n = self.core.neural
        if not hasattr(n.noise, "predraw"):
            return
        with self.t.cuda.stream(n.stream):
            n.noise.predraw(seed, 8)

    def _stream(self, requests, speaker, base_seed, epoch, seed):
        import numpy as np

        from .. import r31

        (t, c, n) = (self.t, self.core, self.core.neural)
        first_eos = [None] * self.batch
        finished = [False] * self.batch
        held = [False] * self.batch
        first_published = [False] * self.batch
        total_samples = [0] * self.batch
        deliveries = [[] for _ in range(self.batch)]
        pcm_hashes = [hashlib.sha256() for _ in range(self.batch)]
        fast_first_chunk = r31.enabled("fast_first_chunk")
        pending_hashes = []

        def flush_hashes():
            for lane, payload in pending_hashes:
                pcm_hashes[lane].update(payload)
            pending_hashes.clear()

        lane_chunk_indices = [0] * self.batch
        chunks = []
        continuation_observations = []
        start_wall = time.perf_counter()
        self._reset(requests, seed)
        n.start()
        with t.cuda.stream(n.stream):
            for generation in range(5):
                (codes, phones, stop) = n.step()
                c.raw_cols[generation].copy_(codes)
                c.stop_cols[generation].copy_(stop)
                c.phone_cols[generation].copy_(phones)
            t.any(c.stops[:, :5], dim=1, out=c.done)
        executed = 5
        (first_eos, done, counts) = self._observe(first_eos, 0, executed)
        bank = 0
        window_index = 0
        prefetched_generation = None

        def finish_ready():
            for lane in range(self.batch):
                terminal = done[lane] or executed == self.steps
                drained = counts[lane] == self.consumed_host[lane]
                if not finished[lane] and terminal and drained:
                    finished[lane] = True
                    first_published[lane] = True
                    held[lane] = False
                    yield {
                        "type": "finish",
                        "lane": lane,
                        "reason": "eos" if done[lane] else "length_limit",
                        "samples": total_samples[lane],
                        "sample_rate": SAMPLE_RATE,
                    }

        while True:
            F = (
                self.research_schedule[window_index]
                if window_index < len(self.research_schedule)
                else 8
            )
            while True:
                release = continuation_lanes(counts, self.consumed_host, held)
                if release:
                    for lane in release:
                        held[lane] = False
                        first_published[lane] = True
                    continuation_observations.append(
                        {
                            "executed_generations": executed,
                            "lanes": release,
                            "elapsed_ms": (time.perf_counter() - start_wall) * 1000,
                        }
                    )
                    yield {"type": "continuation", "lanes": release}
                yield from finish_ready()
                needed = generation_deficit(
                    counts, self.consumed_host, done, F, executed, self.steps
                )
                if not needed:
                    break
                needed = publication_probe_steps(needed, held, done)
                before = executed
                if prefetched_generation is not None:
                    if prefetched_generation != before or needed != 1:
                        raise RuntimeError("prefetch diverged from serial observation policy")
                    executed += 1
                    prefetched_generation = None
                else:
                    with t.cuda.stream(n.stream):
                        for generation in range(before, before + needed):
                            self._append_step(generation)
                            executed += 1
                flush_hashes()
                (first_eos, done, counts) = self._observe(first_eos, before, executed)
            pending = [
                count - used for (count, used) in zip(counts, self.consumed_host, strict=False)
            ]
            if any(pending):
                submission_generations = executed
                pending_codec = self._submit_chunk(F, counts, bank)
                sizes = pending_codec["sizes"]
                next_frames = (
                    self.research_schedule[window_index + 1]
                    if window_index + 1 < len(self.research_schedule)
                    else 8
                )
                prefetch = should_prefetch(
                    self.execution["mode"],
                    counts,
                    self.consumed_host,
                    done,
                    held,
                    sizes,
                    first_published,
                    next_frames,
                    executed,
                    self.steps,
                )
                if (
                    self.execution["mode"] == "serial"
                    and self.execution["grid_policy"] == "overlap-only"
                ):
                    self._deferred_serial_grid = should_prefetch(
                        self._serial_grid_reference_mode,
                        counts,
                        self.consumed_host,
                        done,
                        held,
                        sizes,
                        first_published,
                        next_frames,
                        executed,
                        self.steps,
                    )
                if prefetch:
                    self._set_stage0_grid(overlapped=True)
                    if prefetched_generation is not None:
                        raise RuntimeError("only one future generation may be outstanding")
                    with t.cuda.stream(n.stream):
                        self._append_step(executed)
                    prefetched_generation = executed
                    self._set_stage0_grid(overlapped=False)
                early_proof = (
                    bool(getattr(self, "r8_options", {}).get("early_proof"))
                    and window_index == 0
                    and (not prefetch)
                    and should_prefetch(
                        "overlap",
                        counts,
                        self.consumed_host,
                        done,
                        held,
                        sizes,
                        first_published,
                        next_frames,
                        executed,
                        self.steps,
                    )
                )
                if early_proof:
                    if prefetched_generation is not None:
                        raise RuntimeError("proof generation already submitted")
                    n.stream.wait_event(pending_codec["events"][1])
                    with t.cuda.stream(n.stream):
                        self._append_step(executed)
                    prefetched_generation = executed
                (result, sizes, samples, codec_ms) = self._finish_chunk(pending_codec)
                self._overlap_records.append(
                    {
                        "chunk_index": window_index,
                        "stage0_generation": submission_generations if prefetch else None,
                        "codec_events": pending_codec["events"],
                    }
                )
                data = result["pcm"].numpy()
                finite = result.get("finite")
                if not (np.isfinite(data).all() if finite is None else finite):
                    raise RuntimeError("native codec produced nonfinite PCM")
                delivery_ms = (time.perf_counter() - start_wall) * 1000
                chunks.append(
                    {
                        "index": window_index,
                        "frames": F,
                        "executed_generations": submission_generations,
                        "prefetched_generation": submission_generations if prefetch else None,
                        "true_lengths": sizes,
                        "samples": samples,
                        "delivery_ms": delivery_ms,
                        "stage1_event_ms": codec_ms,
                        "status": result["status"],
                    }
                )
                last_chunk_lane = max((i for (i, value) in enumerate(samples) if value), default=-1)
                for lane, sample_count in enumerate(samples):
                    if sample_count:
                        pcm = (
                            data[lane, :sample_count]
                            .astype(
                                "<i2" if result.get("encoding") == "pcm_s16_le_r9" else "<f4",
                                copy=False,
                            )
                            .tobytes()
                        )
                        if fast_first_chunk:
                            pending_hashes.append((lane, pcm))
                        else:
                            pcm_hashes[lane].update(pcm)
                        total_samples[lane] += sample_count
                        deliveries[lane].append(delivery_ms)
                        held[lane] = True
                        batch_end = lane == last_chunk_lane or (
                            fast_first_chunk and lane % 64 == 63
                        )
                        yield {
                            "type": "chunk",
                            "lane": lane,
                            "chunk_index": lane_chunk_indices[lane],
                            "batch_end": batch_end,
                            "pcm_f32_le": pcm,
                            "samples": sample_count,
                            "sample_rate": SAMPLE_RATE,
                        }
                        lane_chunk_indices[lane] += 1
                bank = 1 - bank
                window_index += 1
            yield from finish_ready()
            if all(finished):
                break
            if not any(pending) and executed == self.steps:
                raise RuntimeError("capped cohort did not drain")
        flush_hashes()
        elapsed_ms = (time.perf_counter() - start_wall) * 1000
        raw = c.raw[:, :executed].cpu().numpy()
        phones = c.phones[:, :executed].cpu().numpy()
        active_lengths = [g + 1 if g is not None else executed for g in first_eos]
        neural = n.receipt()
        if neural["trt_imported_modules"]:
            raise RuntimeError("TensorRT imported in native serving engine")
        receipt = {
            "schema": "native-three-mk.serving-cohort.v2",
            "status": "COMPLETE",
            "batch": self.batch,
            "base_seed": base_seed,
            "epoch": epoch,
            "effective_seed": seed,
            "speaker": speaker,
            "requests": [dataclasses.asdict(r) for r in requests],
            "text_sha256": [hashlib.sha256(r.text.encode()).hexdigest() for r in requests],
            "token_ids_sha256": [
                hashlib.sha256(json.dumps(r.target_ids).encode()).hexdigest() for r in requests
            ],
            "executed_steps": executed,
            "step_cap": self.steps,
            "first_eos_generations": first_eos,
            "samples": total_samples,
            "finish_reasons": ["eos" if value else "length_limit" for value in done],
            "completed_lanes": sum(done),
            "terminal_lanes": self.batch,
            "length_limit_lanes": self.batch - sum(done),
            "no_audio_lanes": sum(n == 0 for n in total_samples),
            "pcm_sha256": [h.hexdigest() for h in pcm_hashes],
            "active_codes_sha256": [
                hashlib.sha256(raw[i, :length].astype("<i8", copy=False).tobytes()).hexdigest()
                for (i, length) in enumerate(active_lengths)
            ],
            "active_phonemes_sha256": [
                hashlib.sha256(phones[i, :length].astype("<i4", copy=False).tobytes()).hexdigest()
                for (i, length) in enumerate(active_lengths)
            ],
            "startup_predictor_skip": self.startup_predictor_skip,
            "helper_graphs": self.helper_graphs.receipt() if self.helper_graphs else None,
            "live_codes_sha256": [
                hashlib.sha256(raw[i, 5:length].astype("<i8", copy=False).tobytes()).hexdigest()
                for (i, length) in enumerate(active_lengths)
            ],
            "deliveries_ms": deliveries,
            "elapsed_ms": elapsed_ms,
            "chunks": chunks,
            "continuation_observations": continuation_observations,
            "prefill_event_ms": neural["prefill_cuda_event_ms"],
            "stage0_event_ms": neural["stage0_cuda_event_ms"],
            "rng": neural["rng"],
            "trt_imported_modules": neural["trt_imported_modules"],
            "selection": self.selection,
            "chunk_policy": {
                "schema": "nemotron.chunk-schedule.research.r26.v1",
                "startup": list(self.research_schedule),
                "steady": 8,
                "speech_delay": 5,
                "partial_terminal_tail": True,
                "changed_payload_contract": self.research_schedule != (3, 5),
            },
            "scheduling": "serial fixed cohort, chunk0 protected; no graph replay, overlap, "
            "refill or "
            "compaction",
            "publication": "observe one new valid frame after held PCM; probe one generation "
            "at a time "
            "only until continuation or EOS is established",
            "elapsed_scope": "engine admission through lane finish events including caller "
            "backpressure; "
            "host float PCM, before frontend PCM16/lookahead/HTTP",
            "engine_source_sha256": self._engine_source_sha256,
        }
        if self.retain_codes:
            receipt["codes"] = [
                raw[i, :length].tolist() for (i, length) in enumerate(active_lengths)
            ]
            receipt["phonemes"] = [
                phones[i, :length].tolist() for (i, length) in enumerate(active_lengths)
            ]
        if self.helper_graphs:
            receipt["scheduling"] = (
                "serial fixed cohort; preparation/control graph replay; three separate "
                "neural megakernels; no overlap, refill or compaction"
            )
        receipt["scheduling"] = (
            "isolated bounded-grid "
            + self.execution["mode"]
            + "; one policy-preserving generation lookahead; immutable codec input; one "
            "outstanding codec ticket"
        )
        receipt["r8_runtime_options"] = getattr(self, "r8_options", {})
        receipt["execution"] = self.execution
        receipt["qualification_mode"] = self.qualification_mode
        receipt["residency"] = self.residency
        receipt["overlap"] = self._overlap_receipt()
        receipt["active_lanes"] = (
            self.contraction.receipt() if self.contraction else {"mode": "disabled"}
        )
        receipt["actual_libraries"] = self._actual_library_records
        receipt["support_source_sha256"] = self._support_source_sha256
        snapshot_owner = getattr(self, "_fp32_prefix_snapshot", None)
        receipt["fp32_prefix_snapshot"] = (
            dict(snapshot_owner.receipt) if snapshot_owner is not None else {"policy": "disabled"}
        )
        self.last_receipt = receipt
        yield {"type": "receipt", "receipt": receipt}

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("close the live cohort iterator before closing its engine")
        try:
            if not self._closed:
                self._closed = True
                if self.core is not None:
                    self.core.neural.stream.synchronize()
                    self.codec_stream.synchronize()
                    if self.contraction is not None:
                        self.contraction.close()
                    self.core.close()
                    self.core = None
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
