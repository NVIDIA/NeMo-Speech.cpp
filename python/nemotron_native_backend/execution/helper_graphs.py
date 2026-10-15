from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import MethodType

INPUT_SOURCE_SHA = "237f2261ded2f78ac9f71e574cb17dccc6871261c68548029c7ee3b077801ff8"
MAX_GENERATIONS = 511
_ABSENT = object()
_SEMANTIC_STATE = (
    "offsets",
    "last_codes",
    "last_phoneme",
    "phoneme_ended",
    "audio_valid",
    "stop",
    "control_logits",
    "embedding",
    "combined_model",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HelperGraphs:
    def __init__(self, neural):
        self.neural = neural
        self.inputs = inputs = neural.inputs
        self.torch = neural.torch
        self.stream = neural.stream
        self.closed = False
        self.installed = False
        self.prepare_graph = None
        self.commit_graph = None
        self.prepare_replays = 0
        self.commit_replays = 0
        if getattr(neural, "closed", False) or inputs.prepared:
            raise ValueError("install requires an open runner between generation steps")
        if type(inputs).__name__ != "ReusableInputs":
            raise ValueError("only the selected ReusableInputs implementation is supported")
        self.input_source = inspect.getsourcefile(type(inputs))
        if not self.input_source or _digest(self.input_source) != INPUT_SOURCE_SHA:
            raise ValueError("ReusableInputs differs from the selected source")
        self.original_prepare = inputs.prepare_decode
        self.original_commit = inputs.commit_outputs
        if self.original_prepare.__func__.__globals__["MAX_GENERATIONS"] != MAX_GENERATIONS:
            raise ValueError("selected generation limit changed")
        self.original_close = neural.close
        self._old_slots = {
            (id(obj), name): obj.__dict__.get(name, _ABSENT)
            for (obj, name) in (
                (inputs, "prepare_decode"),
                (inputs, "commit_outputs"),
                (neural, "close"),
                (neural, "_helper_graphs"),
            )
        }
        self._bound = {
            name: value
            for (name, value) in vars(inputs).items()
            if isinstance(value, self.torch.Tensor)
        }
        self._bound.update({"weights." + name: value for (name, value) in inputs.weights.items()})
        self._bound["neural.codes"] = neural.codes
        self._bound["neural.phonemes"] = neural.phonemes
        self._signatures = {name: self._signature(value) for (name, value) in self._bound.items()}
        self._codes_pointer = neural.codes.data_ptr()
        self._phonemes_pointer = neural.phonemes.data_ptr()
        self.validate_bindings()
        try:
            self._capture()
            self.prepare_hook = MethodType(lambda _inputs: self.prepare_decode(), inputs)
            self.commit_hook = MethodType(
                lambda _inputs, codes, phonemes: self.commit_outputs(codes, phonemes), inputs
            )
            self.close_hook = MethodType(lambda _neural: self._close_runner(), neural)
            inputs.prepare_decode = self.prepare_hook
            inputs.commit_outputs = self.commit_hook
            neural.close = self.close_hook
            neural._helper_graphs = self
            self.installed = True
        except BaseException:
            self._retire_graphs()
            self.closed = True
            raise

    @staticmethod
    def _signature(tensor):
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def _resolve(self, name):
        if name.startswith("weights."):
            return self.inputs.weights[name[len("weights.") :]]
        if name.startswith("neural."):
            return getattr(self.neural, name[len("neural.") :])
        return getattr(self.inputs, name)

    def validate_bindings(self):
        if (
            self.closed
            or self.neural.inputs is not self.inputs
            or self.neural.stream is not self.stream
        ):
            raise RuntimeError("helper graph owner changed or was closed")
        for name, expected in self._signatures.items():
            actual = self._resolve(name)
            if actual is not self._bound[name] or self._signature(actual) != expected:
                raise RuntimeError("captured tensor storage changed: " + name)

    def _capture(self):
        torch = self.torch
        inputs = self.inputs
        (generation, prepared) = (inputs.generation, inputs.prepared)
        self.stream.wait_stream(torch.cuda.current_stream(inputs.device))
        with torch.cuda.stream(self.stream), torch.no_grad():
            snapshots = {name: getattr(inputs, name).clone() for name in _SEMANTIC_STATE}
        self.stream.synchronize()

        def restore():
            with torch.cuda.stream(self.stream), torch.no_grad():
                for name, value in snapshots.items():
                    getattr(inputs, name).copy_(value)
            inputs.generation = generation
            inputs.prepared = prepared

        try:
            for _ in range(3):
                with torch.cuda.stream(self.stream), torch.no_grad():
                    self.original_prepare()
                    self.original_commit(self.neural.codes, self.neural.phonemes)
                restore()
            self.stream.synchronize()
            self.prepare_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.prepare_graph, stream=self.stream), torch.no_grad():
                self.original_prepare()
            self.commit_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.commit_graph, stream=self.stream), torch.no_grad():
                self.original_commit(self.neural.codes, self.neural.phonemes)
        finally:
            restore()
            self.stream.synchronize()
        self.validate_bindings()

    def _check_stream(self):
        if self.closed or not self.installed:
            raise RuntimeError("helper graph adapter is not installed")
        if (
            self.torch.cuda.current_stream(self.inputs.device).cuda_stream
            != self.stream.cuda_stream
        ):
            raise RuntimeError("helper replay requires the owning neural CUDA stream")

    def prepare_decode(self):
        self._check_stream()
        inputs = self.inputs
        if inputs.prepared or inputs.generation >= MAX_GENERATIONS:
            raise RuntimeError(
                "prepare requires uncommitted-free state and remaining generation capacity"
            )
        if inputs.generation == 0:
            self.validate_bindings()
        self.prepare_graph.replay()
        inputs.prepared = True
        self.prepare_replays += 1
        return inputs.embedding

    def commit_outputs(self, codes, phonemes):
        self._check_stream()
        inputs = self.inputs
        if not inputs.prepared:
            raise RuntimeError("commit requires prepared inputs")
        if (
            codes.dtype != self.torch.int64
            or tuple(codes.shape) != (inputs.batch, 16)
            or codes.device != inputs.device
            or (codes.data_ptr() != self._codes_pointer)
            or (tuple(codes.stride()) != self._signatures["neural.codes"][2])
        ):
            raise ValueError("codes must use the captured stable output pointer and ABI")
        if (
            phonemes.dtype != self.torch.int32
            or tuple(phonemes.shape) != (inputs.batch,)
            or phonemes.device != inputs.device
            or (phonemes.data_ptr() != self._phonemes_pointer)
            or (tuple(phonemes.stride()) != self._signatures["neural.phonemes"][2])
        ):
            raise ValueError("phonemes must use the captured stable output pointer and ABI")
        self.commit_graph.replay()
        inputs.prepared = False
        inputs.generation += 1
        self.commit_replays += 1
        return inputs.stop

    def receipt(self):
        return dict(
            schema="native.helper-graphs.v1",
            installed=self.installed,
            closed=self.closed,
            batch=self.inputs.batch,
            source_sha256=_digest(__file__),
            input_source=str(self.input_source),
            input_source_sha256=INPUT_SOURCE_SHA,
            prepare_replays=self.prepare_replays,
            commit_replays=self.commit_replays,
            captured_regions=["ReusableInputs.prepare_decode", "ReusableInputs.commit_outputs"],
            neural_kernels_captured=False,
            rng_captured=False,
            warmup_pairs_rolled_back=3,
            graph_lifetime="Retired after stream synchronization and before original neural.close "
            "releases buffers",
            storage_contract="Owned tensor addresses/shapes/strides remain fixed; full "
            "validation at each "
            "generation0; output-pointer validation on every commit",
        )

    def _retire_graphs(self):
        self.stream.synchronize()
        for name in ("commit_graph", "prepare_graph"):
            graph = getattr(self, name, None)
            if graph is not None:
                graph.reset()
                setattr(self, name, None)

    def close(self):
        if self.closed:
            return
        if self.installed:
            for obj, name, hook in (
                (self.inputs, "prepare_decode", self.prepare_hook),
                (self.inputs, "commit_outputs", self.commit_hook),
                (self.neural, "close", self.close_hook),
            ):
                if obj.__dict__.get(name) is not hook:
                    raise RuntimeError("helper hook ownership changed before uninstall: " + name)
        self._retire_graphs()
        if self.installed:
            for obj, name in (
                (self.inputs, "prepare_decode"),
                (self.inputs, "commit_outputs"),
                (self.neural, "close"),
                (self.neural, "_helper_graphs"),
            ):
                old = self._old_slots[id(obj), name]
                if old is _ABSENT:
                    delattr(obj, name)
                else:
                    setattr(obj, name, old)
        self.installed = False
        self.closed = True
        self._bound.clear()

    def _close_runner(self):
        original = self.original_close
        self.close()
        return original()


def install(neural):
    existing = getattr(neural, "_helper_graphs", None)
    if existing is not None:
        if not isinstance(existing, HelperGraphs) or existing.closed:
            raise RuntimeError("unexpected helper graph owner")
        existing.validate_bindings()
        return existing
    return HelperGraphs(neural)


def uninstall(neural):
    adapter = getattr(neural, "_helper_graphs", None)
    if adapter is not None:
        adapter.close()
