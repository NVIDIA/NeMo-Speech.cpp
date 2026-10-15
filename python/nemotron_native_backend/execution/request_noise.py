from __future__ import annotations

import ctypes as C


class RequestNoise:
    def __init__(self, neural, progress):
        t = neural.torch
        self.batch = neural.batch
        self.neural = neural
        self.progress = progress
        self.seeds = t.zeros(self.batch, dtype=t.int64, device=neural.inputs.device)
        self.buffer = t.zeros((self.batch, 16, 80), dtype=t.float32, device=neural.inputs.device)
        self.generations = t.zeros(self.batch, dtype=t.int32, device=neural.inputs.device)
        self.launch = neural.lib.mk2_inflight_noise
        self.launch.argtypes = [C.c_void_p] * 5
        self.launch.restype = C.c_int

    def bind(self, slot, seed):
        if type(slot) is not int or not 0 <= slot < self.batch:
            raise ValueError("invalid request noise slot")
        if type(seed) is not int or not 0 <= seed < 1 << 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self.seeds[slot] = seed if seed < 1 << 63 else seed - (1 << 64)

    def generation(self):
        n = self.neural
        self.generations.copy_(n.inputs.offsets)
        n._check(
            self.launch(
                self.seeds.data_ptr(),
                self.generations.data_ptr(),
                self.progress.data_ptr(),
                self.buffer.data_ptr(),
                n.stream.cuda_stream,
            ),
            "request-local Philox noise",
        )

    @staticmethod
    def provenance():
        return {
            "schema": "nemotron.request-local-philox.v1",
            "key": "request seed; logical generation; codebook and sample index",
            "distribution": "float32 Gumbel -log(-log(U))",
            "neighbour_independent": True,
            "legacy_batch_rng_equivalent": False,
        }
