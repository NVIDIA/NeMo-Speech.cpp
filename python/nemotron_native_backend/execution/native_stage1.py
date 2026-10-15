from __future__ import annotations

import ctypes as C
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .source_build import source_library_receipt

HERE = Path(__file__).resolve().parent


class NativeStage1Error(RuntimeError):
    pass


@dataclass(frozen=True)
class OpSpec:
    prefix: str
    state: str
    cin: int
    cout: int
    kernel: int
    stride: int = 1
    activation: str | None = None


def operation_specs():
    ops = [OpSpec("pre_conv", "pre_conv", 40, 768, 3)]

    def residual(prefix, state, channels, k):
        ops.extend(
            (
                OpSpec(
                    prefix + ".input_conv",
                    state + "_input",
                    channels,
                    2 * channels,
                    k,
                    activation=prefix + ".input_conv.activation.alpha",
                ),
                OpSpec(
                    prefix + ".skip_conv",
                    state + "_skip",
                    2 * channels,
                    channels,
                    k,
                    activation=prefix + ".output_activation.alpha",
                ),
            )
        )

    residual("pre_resblocks.0", "pre_resblock_0", 768, 3)
    ops.append(
        OpSpec(
            "pre_up_sample_layers.0",
            "pre_upsample_0",
            768,
            768,
            4,
            2,
            "pre_up_sample_layers.0.activation.alpha",
        )
    )
    for i in range(6):
        residual(f"conv_layers.{i}", f"hidden_resblock_{i}", 768, 3)
    cin = 768
    for i, (cout, stride) in enumerate(((384, 9), (128, 7), (32, 7))):
        prefix = f"resblock_up_sample_layers.{i}"
        ops.append(
            OpSpec(
                prefix, f"upsample_{i}", cin, cout, 2 * stride, stride, prefix + ".activation.alpha"
            )
        )
        residual(f"resblocks.{i}", f"upsample_resblock_{i}", cout, 7)
        cin = cout
    ops.append(OpSpec("post_conv", "post_conv", 32, 1, 7))
    return tuple(ops)


OPS = operation_specs()
STATE_SPECS = tuple((o.state, o.cin, 1 if o.stride > 1 else o.kernel - 1) for o in OPS)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


class WeightPointers(C.Structure):
    _fields_ = [("weight", C.c_void_p), ("bias", C.c_void_p), ("alpha", C.c_void_p)]


class NativeCodecWeights:
    def __init__(self, checkpoint, device="cuda:0", expected_sha256=None):
        import torch
        from safetensors.torch import load_file

        self.path = Path(checkpoint).resolve()
        self.device = torch.device(device)
        self.sha256 = file_sha(self.path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise NativeStage1Error("checkpoint hash mismatch")
        weights = load_file(str(self.path), device="cpu")
        codec = {
            k[len("audio_decoder.") :]: v
            for (k, v) in weights.items()
            if k.startswith("audio_decoder.")
        }
        if len(codec) != 76:
            raise NativeStage1Error(f"expected76 codec tensors, got{len(codec)}")
        self.tensors = []
        self.pointers = (WeightPointers * 26)()
        for p, o in enumerate(OPS):
            w = codec[o.prefix + ".conv.weight"]
            b = codec[o.prefix + ".conv.bias"]
            shape = (o.cin, 1, o.kernel) if o.stride > 1 else (o.cout, o.cin, o.kernel)
            if tuple(w.shape) != shape or tuple(b.shape) != (o.cout,):
                raise NativeStage1Error(f"weight shape mismatch at{o.prefix}")
            w = w.to(device=self.device, dtype=torch.float16).contiguous()
            b = b.to(device=self.device, dtype=torch.float32).contiguous()
            alpha = None
            if o.activation:
                alpha = codec[o.activation].reshape(-1)
                if alpha.numel() != o.cout // 2:
                    raise NativeStage1Error("HalfSnake alpha shape mismatch")
                alpha = alpha.to(device=self.device, dtype=torch.float32).contiguous()
            self.tensors.extend(x for x in (w, b, alpha) if x is not None)
            self.pointers[p] = WeightPointers(
                w.data_ptr(), b.data_ptr(), alpha.data_ptr() if alpha is not None else None
            )
        torch.cuda.current_stream(self.device).synchronize()

    def receipt(self):
        return {
            "checkpoint": str(self.path),
            "checkpoint_sha256": self.sha256,
            "tensor_count": len(self.tensors),
            "bytes": sum(x.numel() * x.element_size() for x in self.tensors),
            "weight_dtype": "float16",
            "bias_alpha_dtype": "float32",
            "immutable": True,
        }


def load_library(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.native_stage1_create.argtypes = [C.c_int] * 4 + [C.POINTER(WeightPointers)]
    lib.native_stage1_create.restype = C.c_void_p
    lib.native_stage1_run.argtypes = (
        [C.c_void_p] * 4 + [C.POINTER(C.c_void_p)] * 2 + [C.c_void_p] * 3 + [C.c_int, C.c_void_p]
    )
    lib.native_stage1_run.restype = C.c_int
    lib.native_stage1_destroy.argtypes = [C.c_void_p]
    lib.native_stage1_destroy.restype = C.c_int
    lib.native_stage1_last_error.restype = C.c_char_p
    lib.native_stage1_scratch_bytes.argtypes = [C.c_void_p]
    lib.native_stage1_scratch_bytes.restype = C.c_size_t
    lib.native_stage1_resident_blocks.argtypes = [C.c_void_p]
    lib.native_stage1_resident_blocks.restype = C.c_int
    return lib


@dataclass
class Ticket:
    owner: Any
    event: Any
    codes: Any
    lengths: Any
    slots: Any
    states: tuple
    next_states: tuple
    pcm: Any
    samples: Any
    status: Any
    finite: Any = None


class NativeStage1:
    def __init__(
        self,
        weights: NativeCodecWeights,
        batch=1,
        max_frames=16,
        state_slots=None,
        resident_blocks=0,
        library=None,
    ):
        import torch

        self.weights = weights
        self.device = weights.device
        self.batch = int(batch)
        self.max_frames = int(max_frames)
        self.state_slots = int(batch if state_slots is None else state_slots)
        if library is None:
            raise ValueError("An explicit qualified codec library is required")
        self.path = Path(library).resolve()
        self.lib = load_library(self.path)
        self._lock = threading.RLock()
        self._ticket = None
        self._poisoned = False
        self.device_finite = False
        with torch.cuda.device(self.device):
            self.handle = self.lib.native_stage1_create(
                self.batch,
                self.max_frames,
                self.state_slots,
                int(resident_blocks),
                weights.pointers,
            )
        if not self.handle:
            raise NativeStage1Error(self._error())

    def _error(self):
        return (self.lib.native_stage1_last_error() or b"unknown native Stage1 error").decode()

    def zero_states(self):
        import torch

        return tuple(
            torch.zeros((self.state_slots, c, h), dtype=torch.float16, device=self.device)
            for (_, c, h) in STATE_SPECS
        )

    def bind_buffers(
        self,
        codes,
        lengths,
        slots,
        states,
        next_states,
        pcm,
        samples,
        status,
        stream=None,
        completion_event=None,
    ):
        return BoundStage1(
            self,
            codes,
            lengths,
            slots,
            states,
            next_states,
            pcm,
            samples,
            status,
            stream,
            completion_event,
        )

    def run_async(self, codes, lengths, slots, states, stream=None):
        import torch

        with self._lock:
            if not self.handle or self._poisoned:
                raise NativeStage1Error("closed/poisoned handle")
            if self._ticket is not None:
                raise NativeStage1Error("finish outstanding ticket before reuse")
            frames = int(codes.shape[1]) if codes.ndim == 3 else 0
            if not 1 <= frames <= self.max_frames:
                raise NativeStage1Error("invalid chunk frames")
            tensors = [
                (codes, torch.int64, (self.batch, frames, 16)),
                (lengths, torch.int32, (self.batch,)),
                (slots, torch.int32, (self.batch,)),
            ]
            if len(states) != 26:
                raise NativeStage1Error("expected26 input states")
            tensors.extend(
                (x, torch.float16, (self.state_slots, c, h))
                for (x, (_, c, h)) in zip(states, STATE_SPECS, strict=False)
            )
            for x, dtype, shape in tensors:
                if (
                    x.device != self.device
                    or x.dtype != dtype
                    or tuple(x.shape) != shape
                    or (not x.is_contiguous())
                ):
                    raise NativeStage1Error(
                        f"input ABI mismatch: expected{dtype}/{shape}/{self.device}"
                    )
            stream = stream or torch.cuda.current_stream(self.device)
            with torch.cuda.device(self.device), torch.cuda.stream(stream):
                ns = tuple(torch.empty_like(x) for x in states)
                pcm = torch.empty(
                    (self.batch, frames * 1764), dtype=torch.float32, device=self.device
                )
                samples = torch.empty((self.batch,), dtype=torch.int32, device=self.device)
                status = torch.empty((4,), dtype=torch.int32, device=self.device)
                ptrs = (C.c_void_p * 26)(*(x.data_ptr() for x in states))
                outs = (C.c_void_p * 26)(*(x.data_ptr() for x in ns))
                code = self.lib.native_stage1_run(
                    self.handle,
                    codes.data_ptr(),
                    lengths.data_ptr(),
                    slots.data_ptr(),
                    ptrs,
                    outs,
                    pcm.data_ptr(),
                    samples.data_ptr(),
                    status.data_ptr(),
                    frames,
                    int(stream.cuda_stream),
                )
                if code:
                    self._poisoned = True
                    raise NativeStage1Error(self._error())
                event = torch.cuda.Event(blocking=False)
                event.record(stream)
            self._ticket = Ticket(
                self, event, codes, lengths, slots, tuple(states), ns, pcm, samples, status
            )
            return self._ticket

    def finish(self, ticket, copy_pcm=False):
        with self._lock:
            if ticket is not self._ticket or ticket.owner is not self:
                raise NativeStage1Error("ticket ownership/reuse mismatch")
            try:
                ticket.event.synchronize()
                status = ticket.status.cpu().tolist()
            except Exception:
                self._poisoned = True
                self._ticket = None
                raise
            self._ticket = None
            if status[0]:
                raise NativeStage1Error(
                    f"device input validation failed: code,row,frame,detail={status}"
                )
            return {
                "pcm": ticket.pcm.cpu() if copy_pcm else ticket.pcm,
                "next_states": ticket.next_states,
                "samples": ticket.samples,
                "status": status,
                "sample_rate_hz": 22050,
            }

    def receipt(self):
        source_build = source_library_receipt(self.path, self.batch)
        if source_build is not None:
            return {
                "kind": "native_stage1_single_cooperative_kernel_prototype",
                "qualified": False,
                "library": str(self.path),
                "library_sha256": file_sha(self.path),
                "batch": self.batch,
                **source_build,
                "state_slots": self.state_slots,
                "max_frames": self.max_frames,
                "scratch_bytes": self.lib.native_stage1_scratch_bytes(self.handle),
                "resident_blocks": self.lib.native_stage1_resident_blocks(self.handle),
                "weights": self.weights.receipt(),
                "tensorrt_dependency": False,
            }
        built_source = self.path.parent / "native_stage1.cu"
        if not built_source.is_file():
            raise NativeStage1Error("library has no adjacent frozen source receipt")
        build_path = self.path.parent / "build-receipt.json"
        build = json.loads(build_path.read_text())
        library_hash = file_sha(self.path)
        if build.get("library_sha256") != library_hash:
            raise NativeStage1Error("library does not match its build receipt")
        closure = []
        for entry in build["sources"]:
            frozen = self.path.parent / Path(entry["path"]).name
            actual = file_sha(frozen)
            if actual != entry["sha256"]:
                raise NativeStage1Error(f"frozen build source hash mismatch: {frozen}")
            closure.append({"path": str(frozen), "sha256": actual})
        return {
            "kind": "native_stage1_single_cooperative_kernel_prototype",
            "qualified": False,
            "library": str(self.path),
            "library_sha256": library_hash,
            "source": str(built_source),
            "source_sha256": file_sha(built_source),
            "batch": self.batch,
            "build_receipt": str(build_path),
            "build_receipt_sha256": file_sha(build_path),
            "source_closure": closure,
            "state_slots": self.state_slots,
            "max_frames": self.max_frames,
            "scratch_bytes": self.lib.native_stage1_scratch_bytes(self.handle),
            "resident_blocks": self.lib.native_stage1_resident_blocks(self.handle),
            "weights": self.weights.receipt(),
            "tensorrt_dependency": False,
        }

    def close(self):
        with self._lock:
            if self._ticket is not None:
                self.finish(self._ticket)
            if self.handle:
                result = self.lib.native_stage1_destroy(self.handle)
                self.handle = None
                if result:
                    raise NativeStage1Error(self._error())


class BoundStage1:
    def __init__(
        self,
        engine,
        codes,
        lengths,
        slots,
        states,
        next_states,
        pcm,
        samples,
        status,
        stream=None,
        completion_event=None,
    ):
        import torch

        self.engine = engine
        self.stream = stream or torch.cuda.current_stream(engine.device)
        self.event = completion_event or torch.cuda.Event(blocking=False)
        self.codes = codes
        self.lengths = lengths
        self.slots = slots
        self.states = tuple(states)
        self.next_states = tuple(next_states)
        self.pcm = pcm
        self.samples = samples
        self.status = status
        f = int(codes.shape[1]) if codes.ndim == 3 else 0
        if not 1 <= f <= engine.max_frames or len(states) != 26 or len(next_states) != 26:
            raise NativeStage1Error("invalid fixed-buffer frame/state count")
        specs = [
            (codes, torch.int64, (engine.batch, f, 16)),
            (lengths, torch.int32, (engine.batch,)),
            (slots, torch.int32, (engine.batch,)),
            (pcm, torch.float32, (engine.batch, 1764 * f)),
            (samples, torch.int32, (engine.batch,)),
            (status, torch.int32, (4,)),
        ]
        for tensors in (states, next_states):
            specs.extend(
                (x, torch.float16, (engine.state_slots, c, h))
                for (x, (_, c, h)) in zip(tensors, STATE_SPECS, strict=False)
            )
        for x, dtype, shape in specs:
            if (
                x.dtype != dtype
                or x.device != engine.device
                or tuple(x.shape) != shape
                or (not x.is_contiguous())
            ):
                raise NativeStage1Error(f"fixed-buffer ABI mismatch: expected{dtype}/{shape}")

        def interval(x):
            return (x.data_ptr(), x.data_ptr() + x.numel() * x.element_size())

        read = [interval(x) for x in (codes, lengths, slots, *states)]
        write = [interval(x) for x in (pcm, samples, status, *next_states)]

        def overlaps(a, b):
            return a[0] < b[1] and b[0] < a[1]

        if any(overlaps(a, b) for a in read for b in write) or any(
            overlaps(a, b) for (i, a) in enumerate(write) for b in write[:i]
        ):
            raise NativeStage1Error("fixed-buffer inputs/outputs must not overlap")
        self.inputs = (C.c_void_p * 26)(*(x.data_ptr() for x in states))
        self.outputs = (C.c_void_p * 26)(*(x.data_ptr() for x in next_states))
        self.args = (
            engine.handle,
            codes.data_ptr(),
            lengths.data_ptr(),
            slots.data_ptr(),
            self.inputs,
            self.outputs,
            pcm.data_ptr(),
            samples.data_ptr(),
            status.data_ptr(),
            f,
            int(self.stream.cuda_stream),
        )

    def enqueue(self):
        import torch

        e = self.engine
        with e._lock:
            if not e.handle or e._poisoned:
                raise NativeStage1Error("closed/poisoned handle")
            if e._ticket is not None:
                raise NativeStage1Error("finish outstanding ticket before reuse")
            with torch.cuda.device(e.device), torch.cuda.stream(self.stream):
                if e.lib.native_stage1_run(*self.args):
                    e._poisoned = True
                    raise NativeStage1Error(e._error())
                finite = torch.isfinite(self.pcm).all().view(1) if e.device_finite else None
                self.event.record(self.stream)
            e._ticket = Ticket(
                e,
                self.event,
                self.codes,
                self.lengths,
                self.slots,
                self.states,
                self.next_states,
                self.pcm,
                self.samples,
                self.status,
                finite,
            )
            return e._ticket


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "operations": len(OPS),
                "states": STATE_SPECS,
                "elements_per_request": sum(c * h for (_, c, h) in STATE_SPECS),
                "status": "UNQUALIFIED_BUILD_READY_PROTOTYPE",
            },
            indent=2,
        )
    )
