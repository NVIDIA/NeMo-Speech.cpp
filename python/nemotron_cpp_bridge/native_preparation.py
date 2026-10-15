# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes as C
import json
import threading
from pathlib import Path


class FrontendPreparation:
    def __init__(self, library, tokenizer_path, config_path=None):
        config = json.loads(Path(config_path).read_text()) if config_path is not None else None
        if config_path is not None and not isinstance(config, dict):
            raise ValueError("normalization configuration must be an object")
        self.enabled = config is not None and config.get("enabled", True)
        if type(self.enabled) is not bool:
            raise ValueError("normalization enabled must be Boolean")
        self.metadata = {
            "library": str(Path(library).resolve(strict=True)),
            "tokenizer": str(tokenizer_path),
            "normalization_config": json.dumps(config),
        }

    def close(self):
        pass


class NativePreparation:
    def __init__(self, library, tokenizer_path, config_path=None):
        config = json.loads(Path(config_path).read_text()) if config_path is not None else None
        if config_path is not None and not isinstance(config, dict):
            raise ValueError("normalization configuration must be an object")
        self.enabled = config is not None and config.get("enabled", True)
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False
        self._library = C.CDLL(str(Path(library).resolve(strict=True)))
        self._create = self._library.nemotron_text_create
        self._create.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
        ]
        self._create.restype = C.c_int
        self._prepare = self._library.nemotron_text_prepare
        self._prepare.argtypes = [
            C.c_void_p,
            C.c_char_p,
            C.c_size_t,
            C.c_uint8,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
        ]
        self._prepare.restype = C.c_int
        self._free = self._library.nemotron_text_free
        self._free.argtypes = [C.c_void_p]
        self._free.restype = None
        self._destroy = self._library.nemotron_text_destroy
        self._destroy.argtypes = [C.c_void_p]
        self._destroy.restype = None
        self._handle = C.c_void_p()
        error = C.c_void_p()
        status = self._create(
            json.dumps(config).encode(),
            str(tokenizer_path).encode(),
            C.byref(self._handle),
            C.byref(error),
        )
        self._check(status, error)
        if not self._handle.value:
            raise RuntimeError("native preparation returned an empty handle")

    def _check(self, status, error):
        try:
            if status:
                message = (
                    C.string_at(error).decode("utf-8", errors="replace")
                    if error.value
                    else "native preparation failed"
                )
                raise (ValueError if status == 1 else RuntimeError)(message)
        finally:
            if error.value:
                self._free(error)

    def prepare_tokens(self, text, apply_normalize=True):
        if not isinstance(text, str) or type(apply_normalize) is not bool:
            raise ValueError("invalid text preparation input")
        data = text.strip().encode("utf-8")
        with self._condition:
            if self._closed:
                raise RuntimeError("text preparer is closed")
            self._active += 1
        output = C.c_void_p()
        error = C.c_void_p()
        try:
            status = self._prepare(
                self._handle, data, len(data), apply_normalize, C.byref(output), C.byref(error)
            )
            self._check(status, error)
            if not output.value:
                raise RuntimeError("native preparation returned an empty result")
            result = json.loads(C.string_at(output))
            return result["text"], result["tokens"]
        finally:
            if output.value:
                self._free(output)
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close(self):
        with self._condition:
            if self._closed:
                self._condition.wait_for(lambda: not self._handle.value)
                return
            self._closed = True
            self._condition.wait_for(lambda: self._active == 0)
            self._destroy(self._handle)
            self._handle = C.c_void_p()
            self._condition.notify_all()
