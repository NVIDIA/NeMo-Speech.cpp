# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes as C
import json
import re
import struct
import threading
from pathlib import Path


class _Span(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_uint8)), ("len", C.c_size_t)]


class _Status(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("code", C.c_int32),
        ("message", _Span),
        ("context", C.c_void_p),
    ]


class _Record(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("name", _Span),
        ("kind", C.c_uint32),
        ("element_type", C.c_uint32),
        ("data", _Span),
        ("shape", C.POINTER(C.c_size_t)),
        ("rank", C.c_size_t),
        ("sample_rate_hz", C.c_uint32),
        ("channels", C.c_uint16),
        ("audio_encoding", C.c_uint16),
        ("sequence", C.c_uint64),
        ("has_sequence", C.c_uint8),
        ("is_final", C.c_uint8),
        ("reserved", C.c_uint8 * 6),
    ]


class _Envelope(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("schema_version", C.c_uint32),
        ("request_id", _Span),
        ("session_id", _Span),
        ("records", C.POINTER(_Record)),
        ("record_count", C.c_size_t),
        ("sequence", C.c_uint64),
        ("has_sequence", C.c_uint8),
        ("is_final", C.c_uint8),
        ("reserved", C.c_uint8 * 6),
    ]


class _Output(C.Structure):
    _fields_ = _Envelope._fields_ + [("context", C.c_void_p)]


class _Capabilities(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("processor_id", _Span),
        ("batch_input", C.c_uint8),
        ("streaming_input", C.c_uint8),
        ("thread_safe", C.c_uint8),
        ("stateful", C.c_uint8),
        ("reserved", C.c_uint8 * 4),
    ]


class _Api(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("abi_version", C.c_uint32),
        ("capabilities", C.CFUNCTYPE(_Status, C.POINTER(_Capabilities))),
        ("create", C.CFUNCTYPE(_Status, C.POINTER(_Envelope), C.POINTER(C.c_void_p))),
        ("destroy", C.CFUNCTYPE(None, C.c_void_p)),
        ("process", C.CFUNCTYPE(_Status, C.c_void_p, C.POINTER(_Envelope), C.POINTER(_Output))),
        ("release_output", C.CFUNCTYPE(None, C.c_void_p, C.POINTER(_Output))),
        ("release_status", C.CFUNCTYPE(None, C.POINTER(_Status))),
    ]


def _bytes(span, limit=1024 * 1024):
    if span.len > limit or (span.len and not span.data):
        raise RuntimeError("native normalizer returned an invalid byte span")
    return C.string_at(span.data, span.len) if span.len else b""


def _envelope(values, identity=b"nemotron-cpp-normalization"):
    storage = []

    def span(value):
        buffer = C.create_string_buffer(value)
        storage.append(buffer)
        return _Span(C.cast(buffer, C.POINTER(C.c_uint8)), len(value))

    records = (_Record * len(values))()
    for record, (name, value) in zip(records, values.items()):
        record.struct_size = C.sizeof(_Record)
        record.name = span(name.encode())
        if isinstance(value, bool):
            record.kind, record.element_type, data = 8, 1, bytes([value])
        elif isinstance(value, int):
            record.kind, record.element_type, data = 6, 3, struct.pack("=q", value)
        else:
            record.kind, record.element_type, data = 1, 1, value.encode("utf-8")
        record.data = span(data)
    envelope = _Envelope(
        struct_size=C.sizeof(_Envelope),
        schema_version=1,
        request_id=span(identity),
        records=records,
        record_count=len(records),
        is_final=1,
    )
    storage.append(records)
    return envelope, storage


class TextPreparer:
    def __init__(self, config_path=None):
        self.enabled = False
        self._condition = threading.Condition()
        self._handles = []
        self._active = 0
        self._closed = False
        self._api = None
        self._max_input = 65536
        self._max_output = 262144
        self._max_segments = 256
        if config_path is None:
            return
        config = json.loads(Path(config_path).read_text())
        allowed = {
            "enabled",
            "native_library_root",
            "native_processor",
            "normalizer_root",
            "normalizer_path",
            "language_code",
            "normalizer_config",
            "direct_grammar",
            "abbreviations_path",
            "append_punctuation",
            "pool_size",
            "bounded_preparation",
            "max_input_bytes",
            "max_normalized_bytes",
            "max_segments",
        }
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("invalid normalization configuration fields")
        for key in ("enabled", "direct_grammar", "append_punctuation", "bounded_preparation"):
            if key in config and type(config[key]) is not bool:
                raise ValueError(f"{key} must be Boolean")
        if config.get("language_code", "en-US") != "en-US":
            raise ValueError("normalization language_code must be en-US")
        self.enabled = config.get("enabled", True)
        limits = {
            "max_input_bytes": 65536,
            "max_normalized_bytes": 262144,
            "max_segments": 256,
            "pool_size": 2,
        }
        for key, default in limits.items():
            value = config.get(key, default)
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError(f"{key} must fit a positive ABI integer")
            limits[key] = value
        self._max_input = limits["max_input_bytes"]
        self._max_output = limits["max_normalized_bytes"]
        self._max_segments = limits["max_segments"]
        library_root = Path(config["native_library_root"]).resolve(strict=True)
        asset_root = Path(config["normalizer_root"]).resolve(strict=True)
        library = Path(config["native_processor"]).resolve(strict=True)
        language_path = (asset_root / config.get("normalizer_path", "en-US")).resolve(strict=True)
        if not library_root.is_dir() or not asset_root.is_dir() or not language_path.is_dir():
            raise ValueError("normalizer roots and language path must be directories")
        if library_root.is_relative_to(asset_root) or asset_root.is_relative_to(library_root):
            raise ValueError("normalizer libraries and assets must have separate roots")
        if not library.is_file() or not library.is_relative_to(library_root):
            raise ValueError("native_processor must be a file inside native_library_root")
        if not language_path.is_relative_to(asset_root):
            raise ValueError("normalizer_path is outside normalizer_root")
        normalizer_config = config.get("normalizer_config", "sparrowhawk_configuration.ascii_proto")
        if (
            not isinstance(normalizer_config, str)
            or Path(normalizer_config).name != normalizer_config
        ):
            raise ValueError("normalizer_config must name a file in normalizer_path")
        required = (
            ("tokenize_and_classify.far", "verbalize.far")
            if config.get("direct_grammar", False)
            else (normalizer_config,)
        )
        for name in required:
            path = (language_path / name).resolve(strict=True)
            if not path.is_file() or not path.is_relative_to(language_path):
                raise ValueError("normalizer input escapes language directory")
        if config.get("direct_grammar", False):
            for name in ("pre_process.far", "post_process.far", "speech_class.far"):
                path = language_path / name
                if path.exists() or path.is_symlink():
                    path = path.resolve(strict=True)
                    if not path.is_file() or not path.is_relative_to(language_path):
                        raise ValueError("normalizer input escapes language directory")
        values = {
            "language_code": "en-US",
            "normalizer_path": str(language_path),
            "normalizer_config": normalizer_config,
            "direct_grammar": config.get("direct_grammar", False),
            "append_punctuation": config.get("append_punctuation", True),
            **{key: value for key, value in limits.items() if key != "pool_size"},
        }
        if config.get("abbreviations_path") is not None:
            path = (language_path / config["abbreviations_path"]).resolve(strict=True)
            if not path.is_file() or not path.is_relative_to(asset_root):
                raise ValueError("abbreviations_path must be inside normalizer_root")
            values["abbreviations_path"] = str(path)
        self._library = C.CDLL(str(library))
        getter = self._library.nemotron_speech_processor_get_api
        getter.argtypes = [C.c_uint32]
        getter.restype = C.POINTER(_Api)
        api = getter(1)
        if not api or api.contents.struct_size < C.sizeof(_Api) or api.contents.abi_version != 1:
            raise RuntimeError("unsupported native normalizer ABI")
        self._api = api.contents
        if not all(
            (
                self._api.capabilities,
                self._api.create,
                self._api.destroy,
                self._api.process,
                self._api.release_output,
                self._api.release_status,
            )
        ):
            raise RuntimeError("native normalizer ABI has missing functions")
        capabilities = _Capabilities(struct_size=C.sizeof(_Capabilities))
        self._check(self._api.capabilities(C.byref(capabilities)))
        if (
            _bytes(capabilities.processor_id) != b"nemotron_speech.tts.text_preprocessor"
            or capabilities.stateful
        ):
            raise RuntimeError("native processor is not the stateless TTS normalizer")
        envelope, _storage = _envelope(values)
        try:
            for _ in range(limits["pool_size"]):
                handle = C.c_void_p()
                self._check(self._api.create(C.byref(envelope), C.byref(handle)))
                if not handle.value:
                    raise RuntimeError("native normalizer returned an empty handle")
                self._handles.append(handle)
        except BaseException:
            self.close()
            raise

    def _check(self, status):
        try:
            if status.code:
                message = _bytes(status.message).decode("utf-8", errors="replace")
                error = ValueError if 100 <= status.code <= 102 else RuntimeError
                raise error(f"native normalizer error {status.code}: {message}")
        finally:
            self._api.release_status(C.byref(status))

    def prepare(self, text, language, apply_normalize=True):
        if not isinstance(text, str) or not isinstance(language, str):
            raise ValueError("text and language must be strings")
        if language.lower() != "en-us":
            raise ValueError("native backend supports only en-US")
        if type(apply_normalize) is not bool:
            raise ValueError("apply_normalize must be Boolean")
        text = text.strip()
        if not text or len(text.encode("utf-8")) > self._max_input:
            raise ValueError("text is empty or exceeds max_input_bytes")
        with self._condition:
            if self._closed:
                raise RuntimeError("text preparer is closed")
            if not self.enabled or not apply_normalize:
                return text
            self._condition.wait_for(lambda: self._handles or self._closed)
            if self._closed:
                raise RuntimeError("text preparer is closed")
            handle = self._handles.pop()
            self._active += 1
        try:
            return self._process(handle, text)
        finally:
            with self._condition:
                self._handles.append(handle)
                self._active -= 1
                self._condition.notify_all()

    def _process(self, handle, text):
        request, _storage = _envelope({"text": text, "language_code": "en-US"})
        output = _Output(struct_size=C.sizeof(_Output))
        try:
            self._check(self._api.process(handle, C.byref(request), C.byref(output)))
            if (
                output.struct_size < C.sizeof(_Output)
                or output.schema_version != 1
                or _bytes(output.request_id) != _bytes(request.request_id)
                or _bytes(output.session_id) != _bytes(request.session_id)
                or output.sequence != request.sequence
                or output.has_sequence != request.has_sequence
                or output.is_final != request.is_final
            ):
                raise RuntimeError("native normalizer changed envelope identity")
            if not output.records or not 0 < output.record_count <= self._max_segments:
                raise RuntimeError("native normalizer returned invalid segment count")
            segments = []
            size = output.record_count - 1
            for index in range(output.record_count):
                record = output.records[index]
                if (
                    record.struct_size < C.sizeof(_Record)
                    or record.kind != 1
                    or _bytes(record.name) != f"normalized_text.{index}".encode()
                ):
                    raise RuntimeError("native normalizer returned invalid segment order/type")
                data = _bytes(record.data, self._max_output)
                size += len(data)
                segment = data.decode("utf-8")
                if not segment.strip() or size > self._max_output:
                    raise RuntimeError("native normalizer returned empty or oversized text")
                segments.append(segment)
            return re.sub(r" +([,.?!।])", r"\1", " ".join(segments))
        finally:
            self._api.release_output(handle, C.byref(output))

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._active == 0)
            handles, self._handles = self._handles, []
            for handle in handles:
                self._api.destroy(handle)
