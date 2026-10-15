from __future__ import annotations

import ctypes as C

from .reusable_inputs import ReusableInputs
from .split_native_runtime import NativeNeuralRunner


class ReusableNeuralRunner(NativeNeuralRunner):
    def __init__(self, *, diagnostic_steps=30, **kwargs):
        super().__init__(**kwargs)
        try:
            self.inputs = ReusableInputs.adopt(self.inputs)
            self.step_events = [
                (
                    self.torch.cuda.Event(enable_timing=True),
                    self.torch.cuda.Event(enable_timing=True),
                )
                for _ in range(diagnostic_steps)
            ]
            for a, b in self.step_events:
                a.record()
                b.record()
            self.stream.wait_stream(self.torch.cuda.current_stream(0))
        except BaseException:
            self.close()
            raise

    def step(self):
        if (
            self.closed
            or not self.started
            or self.inputs.generation >= min(self.cache["max_generations"], len(self.step_events))
        ):
            raise RuntimeError("native step exceeds prepared diagnostic capacity")
        with self.torch.cuda.stream(self.stream):
            self.inputs.prepare_decode()
            self.noise.generation(self.batch)
            (a, b) = self.step_events[self.inputs.generation]
            a.record(self.stream)
            self._check(
                self.lib.mk2_run_native_stage0(
                    self.stage0_handle, C.byref(self.bindings), self.stream.cuda_stream
                ),
                "native Stage0",
            )
            b.record(self.stream)
            self.inputs.commit_outputs(self.codes, self.phonemes)
        self.history.append((a, b))
        return (self.codes, self.phonemes, self.inputs.stop)
