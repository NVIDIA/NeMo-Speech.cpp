from __future__ import annotations

import ctypes as C
import math


class ResetLayout(C.Structure):
    _fields_ = [
        ("row_stride", C.c_uint64),
        ("source_row_stride", C.c_uint64),
        ("elements", C.c_uint64),
        ("rank", C.c_int32),
        ("item_bytes", C.c_int32),
        ("source_compact", C.c_int32),
        ("sizes", C.c_uint64 * 6),
        ("strides", C.c_uint64 * 6),
    ]


def reset_layout(destination, source=None, *, compact=False):
    shape = tuple(destination.shape[1:]) or (1,)
    strides = tuple(destination.stride()[1:]) or (1,)
    if len(shape) > 6 or destination.element_size() not in (1, 2, 4, 8):
        raise ValueError("unsupported indexed reset layout")
    source_stride = 0
    if source is not None:
        if not source.is_contiguous() or source.element_size() != destination.element_size():
            raise ValueError("indexed reset source must have contiguous matching elements")
        if source.numel() != math.prod(shape):
            if tuple(source.shape) != tuple(destination.shape):
                raise ValueError("indexed reset source must be one row or a full physical tensor")
            source_stride = source.stride(0)
    return ResetLayout(
        destination.stride(0),
        source_stride,
        math.prod(shape),
        len(shape),
        destination.element_size(),
        int(compact),
        (C.c_uint64 * 6)(*shape),
        (C.c_uint64 * 6)(*strides),
    )


class IndexedReset:
    def __init__(self, owner):
        self.owner = owner
        t, n, b = owner.t, owner.neural, owner.batch
        self.pending = False
        self.done = t.cuda.Event()
        self.ids_host = t.empty(b, dtype=t.int32, pin_memory=True)
        self.count_host = t.empty(1, dtype=t.int32, pin_memory=True)
        self.ids_host_array = self.ids_host.numpy()
        self.count_host_array = self.count_host.numpy()
        self.launch = n.lib.mk2_inflight_reset_rows
        self.launch.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_uint64,
            C.POINTER(ResetLayout),
            C.c_void_p,
        ]
        self.launch.restype = C.c_int
        with t.cuda.stream(n.stream):
            self.ids = t.zeros(b, dtype=t.int32, device=n.inputs.device)
            self.count = t.zeros(1, dtype=t.int32, device=n.inputs.device)
            self.tokens = t.empty_like(owner.tokens_host, device=n.inputs.device)
            self.metadata = t.empty_like(owner.admission_host, device=n.inputs.device)
            self.operations = []
            for targets, sources in (
                (owner.conv, owner.conv_prefix),
                (owner.ssm, owner.ssm_prefix),
                (owner.kv_prefix, owner.kv_initial),
            ):
                for target, source in zip(targets, sources, strict=True):
                    self.add(target, source)
            self.add(owner.seq_len, fill=n.inputs.prompt_positions)
            self.add(owner.slot_map, owner.slot_ids)
            for bank in owner.core.banks:
                for value in bank:
                    self.add(value)
            inputs = n.inputs
            self.add(inputs.text_tokens, self.tokens, compact=True)
            self.add(inputs.text_lengths, self.metadata[0], compact=True)
            self.add(inputs.max_text_offsets, self.metadata[1], compact=True)
            self.add(owner.noise.seeds, self.metadata[2], compact=True)
            for name in (
                "offsets",
                "last_codes",
                "last_phoneme",
                "phoneme_ended",
                "audio_valid",
                "stop",
            ):
                self.add(getattr(inputs, name))
            for value in (
                n.codes,
                n.phonemes,
                owner.engine.counts,
                owner.engine.consumed,
                owner.core.done,
            ):
                self.add(value)
            self.run()
            n.stream.synchronize()
            self.graph = t.cuda.CUDAGraph()
            with t.cuda.graph(self.graph, stream=n.stream):
                self.run()
        n.stream.synchronize()

    def add(self, target, source=None, *, fill=0, compact=False):
        self.operations.append(
            (target, source, fill, reset_layout(target, source, compact=compact))
        )

    def run(self):
        n = self.owner.neural
        for target, source, fill, layout in self.operations:
            n._check(
                self.launch(
                    target.data_ptr(),
                    None if source is None else source.data_ptr(),
                    self.ids.data_ptr(),
                    self.count.data_ptr(),
                    self.owner.batch,
                    fill,
                    C.byref(layout),
                    n.stream.cuda_stream,
                ),
                "indexed request-state reset",
            )

    def wait(self):
        if self.pending:
            self.done.synchronize()
            self.pending = False

    def submit(self, records):
        if self.pending:
            raise RuntimeError("indexed reset staging is still in use")
        count = len(records)
        if not 0 < count <= self.owner.batch:
            raise ValueError("indexed reset requires nonempty bounded admission")
        self.ids_host_array[:count] = [record["slot"] for record in records]
        self.count_host_array[0] = count
        o = self.owner
        with o.t.cuda.stream(o.neural.stream):
            self.ids[:count].copy_(self.ids_host[:count], non_blocking=True)
            self.count.copy_(self.count_host, non_blocking=True)
            self.tokens[:count].copy_(o.tokens_host[:count], non_blocking=True)
            for index in range(3):
                self.metadata[index, :count].copy_(
                    o.admission_host[index, :count], non_blocking=True
                )
            self.graph.replay()
            self.done.record(o.neural.stream)
        self.pending = True

    def close(self):
        self.wait()
        self.graph.reset()
