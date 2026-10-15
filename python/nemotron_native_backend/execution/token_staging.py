import types

import numpy as np


def validate_lengths(requests, batch, capacity):
    if len(requests) != batch:
        raise ValueError("token staging requires the complete physical cohort")
    lengths = [len(r.target_ids) for r in requests]
    if any(length < 1 or length > capacity for length in lengths):
        raise ValueError("token length outside prepared capacity")
    return lengths


def pack_dense(requests, tokens, lengths, max_offsets):
    sizes = validate_lengths(requests, tokens.shape[0], tokens.shape[1])
    tokens.fill(0)
    lengths[:] = sizes
    np.subtract(lengths, 1, out=max_offsets)
    for row, (request, size) in enumerate(zip(requests, sizes, strict=False)):
        tokens[row, :size] = request.target_ids
    return int(sum(sizes))


class TokenStaging:
    def __init__(self, engine, mode):
        if mode != "pinned":
            raise ValueError("unsupported token staging mode")
        t = engine.t
        n = engine.core.neural
        self.mode = mode
        (self.batch, self.capacity) = n.inputs.text_tokens.shape
        self.lease = t.cuda.Event()
        self.lease.record(n.stream)
        self.calls = 0
        self.bytes_transferred = 0
        self.last_tokens = 0
        self.scatter_binding = None
        self.tokens = t.empty((self.batch, self.capacity), dtype=t.int64, pin_memory=True)
        self.lengths = t.empty(self.batch, dtype=t.int64, pin_memory=True)
        self.max_offsets = t.empty(self.batch, dtype=t.int64, pin_memory=True)
        self.arrays = (self.tokens.numpy(), self.lengths.numpy(), self.max_offsets.numpy())
        self.lease.synchronize()

    def pack(self, requests):
        self.lease.synchronize()
        self.last_tokens = pack_dense(requests, *self.arrays)

    def enqueue(self, engine):
        n = engine.core.neural
        n.inputs.text_tokens.copy_(self.tokens, non_blocking=True)
        n.inputs.text_lengths.copy_(self.lengths, non_blocking=True)
        n.inputs.max_text_offsets.copy_(self.max_offsets, non_blocking=True)
        transferred = (self.batch * self.capacity + 2 * self.batch) * 8
        self.lease.record(n.stream)
        self.calls += 1
        self.bytes_transferred += transferred

    def receipt(self):
        return dict(
            mode=self.mode,
            batch=self.batch,
            capacity=self.capacity,
            calls=self.calls,
            bytes_transferred=self.bytes_transferred,
            last_tokens=self.last_tokens,
            scatter_binding=self.scatter_binding,
            buffer_ownership="exclusive engine; lease event before CPU buffer reuse",
        )


def install(engine, mode="pinned"):
    existing = getattr(engine, "_r11_token_staging", None)
    if existing is not None:
        if existing.mode != mode:
            raise RuntimeError("token staging already installed with a different mode")
        return existing
    staging = TokenStaging(engine, mode)
    original = engine._reset

    def reset(self, requests, seed):
        (t, c, n) = (self.t, self.core, self.core.neural)
        staging.pack(requests)
        c.requests = requests
        n.inputs.requests = tuple(requests)
        if self.contraction is not None:
            self.contraction.reset()
        c.reset(seed)
        with t.cuda.stream(n.stream):
            staging.enqueue(self)
            self.counts.zero_()
            self.consumed.zero_()
        self.consumed_host = [0] * self.batch
        self._chunk_events = []
        self._overlap_records = []
        self._deferred_serial_grid = False
        self._origin.record(n.stream)

    engine._r11_token_staging = staging
    engine._r11_token_staging_original_reset = original
    engine._reset = types.MethodType(reset, engine)
    return staging
