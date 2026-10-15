from __future__ import annotations

import ctypes as C
from pathlib import Path

from .native_assets import (
    BackboneWeights,
    PredictorWeights,
    Stage0Bindings,
    sha256,
    weight_bindings,
)
from .native_inputs import NativeInputs, cache_plan
from .wide_noise import CohortNoise


class PrefillConfig(C.Structure):
    _fields_ = [
        (x, C.c_int)
        for x in ("batch", "prompt_positions", "max_sequence_tokens", "tokens_per_block")
    ]


class PrefillInfo(C.Structure):
    _fields_ = [
        (x, C.c_int)
        for x in (
            "version",
            "batch",
            "prompt_positions",
            "total_prompt_rows",
            "padded_prompt_rows",
            "max_sequence_tokens",
            "tokens_per_block",
            "blocks_per_sequence",
            "conv_slot_elements",
            "ssm_slot_elements",
        )
    ] + [("kv_pool_bytes_per_layer", C.c_uint64), ("prefill_scratch_bytes", C.c_uint64)]


def bind_prefill_library(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.mk2_last_error.argtypes = []
    lib.mk2_last_error.restype = C.c_char_p
    lib.mk2_native_prefill_create.argtypes = [
        C.POINTER(PredictorWeights),
        C.POINTER(BackboneWeights),
        C.POINTER(PrefillConfig),
        C.c_void_p,
        C.c_void_p,
    ]
    lib.mk2_native_prefill_create.restype = C.c_void_p
    lib.mk2_native_prefill_run.argtypes = [C.c_void_p, C.c_void_p]
    lib.mk2_native_prefill_run.restype = C.c_int
    lib.mk2_native_prefill_get_state.argtypes = [
        C.c_void_p,
        C.POINTER(Stage0Bindings),
        C.POINTER(PrefillInfo),
    ]
    lib.mk2_native_prefill_get_state.restype = C.c_int
    lib.mk2_native_prefill_destroy.argtypes = [C.c_void_p]
    lib.mk2_native_prefill_destroy.restype = None
    return lib


def bind_stage0_library(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.mk2_last_error.argtypes = []
    lib.mk2_last_error.restype = C.c_char_p
    lib.mk2_create_stage0.argtypes = [
        C.POINTER(PredictorWeights),
        C.POINTER(BackboneWeights),
        C.c_int,
    ]
    lib.mk2_create_stage0.restype = C.c_void_p
    lib.mk2_run_native_stage0.argtypes = [C.c_void_p, C.POINTER(Stage0Bindings), C.c_void_p]
    lib.mk2_run_native_stage0.restype = C.c_int
    lib.mk2_destroy.argtypes = [C.c_void_p]
    lib.mk2_destroy.restype = None
    lib.mk2_batch.argtypes = [C.c_void_p]
    lib.mk2_batch.restype = C.c_int
    return lib


class NativeNeuralRunner:
    def __init__(
        self,
        *,
        assets,
        requests,
        library,
        prefill_library=None,
        base_seed=0,
        epoch=1,
        max_generations=511,
    ):
        import torch

        self.torch = torch
        torch.cuda.set_device(0)
        self.library = Path(library).resolve()
        self.library_sha256 = sha256(self.library)
        self.lib = bind_stage0_library(self.library)
        self.prefill_library = Path(prefill_library or library).resolve()
        self.prefill_library_sha256 = sha256(self.prefill_library)
        self.prefill_lib = bind_prefill_library(self.prefill_library)
        self.prefill_handle = None
        self.stage0_handle = None
        self.stream = None
        self.started = False
        self.closed = False
        self.history = []
        try:
            self.inputs = NativeInputs(assets, requests, device="cuda:0")
            self.batch = self.inputs.batch
            self.cache = cache_plan(
                self.batch,
                prompt_positions=self.inputs.prompt_positions,
                max_generations=max_generations,
            )
            self.owned_weights = assets.materialize_weights(device="cuda:0")
            (self.pw, self.bw) = weight_bindings(self.owned_weights)
            self.final_hidden = torch.empty_like(self.inputs.prompt)
            self.noise = CohortNoise(
                device="cuda:0", base_seed=base_seed, epoch=epoch, batch=self.batch
            )
            self.codes = torch.empty((self.batch, 16), dtype=torch.int64, device="cuda:0")
            self.phonemes = torch.empty(self.batch, dtype=torch.int32, device="cuda:0")
            self.config = PrefillConfig(
                self.batch, self.inputs.prompt_positions, self.cache["max_sequence_tokens"], 32
            )
            self.prefill_handle = self.prefill_lib.mk2_native_prefill_create(
                C.byref(self.pw),
                C.byref(self.bw),
                C.byref(self.config),
                self.inputs.prompt.data_ptr(),
                self.final_hidden.data_ptr(),
            )
            self._require_handle(self.prefill_handle, "prefill create", self.prefill_lib)
            self.bindings = Stage0Bindings()
            self.info = PrefillInfo()
            self._check(
                self.prefill_lib.mk2_native_prefill_get_state(
                    self.prefill_handle, C.byref(self.bindings), C.byref(self.info)
                ),
                "prefill state",
                self.prefill_lib,
            )
            if (
                self.info.version,
                self.info.batch,
                self.info.prompt_positions,
                self.info.conv_slot_elements,
                self.info.ssm_slot_elements,
                self.info.blocks_per_sequence,
                self.info.kv_pool_bytes_per_layer,
            ) != (
                1,
                self.batch,
                self.inputs.prompt_positions,
                10752,
                196608,
                self.cache["blocks_per_sequence"],
                self.cache["kv_bytes_per_layer"],
            ):
                raise RuntimeError(
                    "native prefill allocation geometry disagrees with source contract"
                )
            self.stage0_handle = self.lib.mk2_create_stage0(
                C.byref(self.pw), C.byref(self.bw), self.batch
            )
            self._require_handle(self.stage0_handle, "Stage0 create")
            if self.lib.mk2_batch(self.stage0_handle) != self.batch:
                raise RuntimeError("native Stage0 library batch mismatch")
            self.bindings.embedding = self.inputs.embedding.data_ptr()
            self.bindings.gumbel = self.noise.buffer.data_ptr()
            self.bindings.temperature = 0.7
            self.bindings.moe_w_inject_debug = None
            self.bindings.codes_out = self.codes.data_ptr()
            self.bindings.phoneme_out = self.phonemes.data_ptr()
            self.stream = torch.cuda.Stream(device=0)
            self.stream.wait_stream(torch.cuda.current_stream(0))
        except BaseException:
            self.close()
            raise

    def _require_handle(self, handle, where, library=None):
        if not handle:
            raise RuntimeError(
                where
                + ": "
                + ((library or self.lib).mk2_last_error() or b"unknown native error").decode()
            )

    def _check(self, rc, where, library=None):
        if rc:
            raise RuntimeError(
                where
                + ": "
                + ((library or self.lib).mk2_last_error() or b"unknown native error").decode()
            )

    def start(self):
        if self.closed or self.started:
            raise RuntimeError("native runner start requires a fresh open instance")
        with self.torch.cuda.stream(self.stream):
            self.noise.context()
            (a, b) = (
                self.torch.cuda.Event(enable_timing=True),
                self.torch.cuda.Event(enable_timing=True),
            )
            a.record(self.stream)
            self._check(
                self.prefill_lib.mk2_native_prefill_run(
                    self.prefill_handle, self.stream.cuda_stream
                ),
                "native prefill",
                self.prefill_lib,
            )
            b.record(self.stream)
        self.prefill_events = (a, b)
        self.started = True

    def step(self):
        if (
            self.closed
            or not self.started
            or self.inputs.generation >= self.cache["max_generations"]
        ):
            raise RuntimeError("native step requires a started instance with remaining capacity")
        with self.torch.cuda.stream(self.stream):
            self.inputs.prepare_decode()
            self.noise.generation(self.batch)
            (a, b) = (
                self.torch.cuda.Event(enable_timing=True),
                self.torch.cuda.Event(enable_timing=True),
            )
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

    def receipt(self):
        self.stream.synchronize()
        return {
            "schema": "native-three-mk.neural-diagnostic.v4",
            "status": "COMPLETED_NATIVE_PREFILL_STAGE0_DIAGNOSTIC",
            "batch": self.batch,
            "library": str(self.library),
            "library_sha256": self.library_sha256,
            "components": {
                "prefill": {
                    "library": str(self.prefill_library),
                    "library_sha256": self.prefill_library_sha256,
                    "entrypoint": "mk2_native_prefill_run",
                    "qualification_claim": False,
                },
                "stage0": {
                    "library": str(self.library),
                    "library_sha256": self.library_sha256,
                    "entrypoint": "mk2_run_native_stage0",
                    "qualification_claim": False,
                },
            },
            "cache_owner": "prefill handle; Stage0 borrows stable views and advances cache lengths",
            "cache": self.cache,
            "rng": self.noise.provenance(),
            "generation_steps": self.inputs.generation,
            "prefill_cuda_event_ms": self.prefill_events[0].elapsed_time(self.prefill_events[1]),
            "stage0_cuda_event_ms": [a.elapsed_time(b) for (a, b) in self.history],
            "scope": "Neural integration only; no codec/PCM/HTTP or independent arithmetic parity "
            "claim. CUDA-event intervals include launch scheduling and first descriptor "
            "setup.",
            "extra_gpu_work": [
                "BF16 text/audio/phoneme embedding composition",
                "full64-row exponential/log/neg noise generation",
                "output stop/phoneme/code/offset control updates",
            ],
            "termination_policy": "fixed slots continue until all lanes EOS or cap; post-EOS "
            "per-lane output "
            "ignored, computation continues; slot reuse unsupported",
            "trt_imported_modules": [
                k
                for k in __import__("sys").modules
                if k == "tensorrt" or k.startswith(("tensorrt.", "tensorrt_llm"))
            ],
        }

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        if self.stream is not None:
            self.stream.synchronize()
        elif self.prefill_handle or self.stage0_handle:
            self.torch.cuda.synchronize(0)
        if self.stage0_handle:
            self.lib.mk2_destroy(self.stage0_handle)
            self.stage0_handle = None
        if self.prefill_handle:
            self.prefill_lib.mk2_native_prefill_destroy(self.prefill_handle)
            self.prefill_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
