# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared GGML quantization support for checkpoint converters."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType

try:
    from gguf.quants import quantize as _gguf_quantize
except Exception:  # pragma: no cover - depends on the installed gguf release
    _gguf_quantize = None


QK_K = 256
K_QUANTS = (
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
)

# (linear weight type, default non-linear type, general.file_type)
DEPLOYMENT_WEIGHT_TYPES = {
    "bf16": (GGMLQuantizationType.BF16, GGMLQuantizationType.F32, 32),
    "fp16": (GGMLQuantizationType.F16, GGMLQuantizationType.F32, 1),
    "q8_0": (GGMLQuantizationType.Q8_0, GGMLQuantizationType.F32, 7),
    "q6_k": (GGMLQuantizationType.Q6_K, GGMLQuantizationType.F32, 18),
    "q5_k": (GGMLQuantizationType.Q5_K, GGMLQuantizationType.F32, 17),
    "q4_k": (GGMLQuantizationType.Q4_K, GGMLQuantizationType.F32, 15),
    "nvfp4": (GGMLQuantizationType.NVFP4, GGMLQuantizationType.F32, 39),
    "mxfp4": (GGMLQuantizationType.MXFP4, GGMLQuantizationType.F32, 38),
}

_GGML_LIB: ctypes.CDLL | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def row_size_bytes(qtype: GGMLQuantizationType, elements: int) -> int:
    if qtype == GGMLQuantizationType.F32:
        return elements * 4
    if qtype in (GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
        return elements * 2
    if qtype == GGMLQuantizationType.Q8_0:
        return (elements // 32) * 34
    if qtype == GGMLQuantizationType.NVFP4:
        return (elements // 64) * 36
    if qtype == GGMLQuantizationType.MXFP4:
        return (elements // 32) * 17
    if qtype == GGMLQuantizationType.Q4_K:
        return (elements // 256) * 144
    if qtype == GGMLQuantizationType.Q5_K:
        return (elements // 256) * 176
    if qtype == GGMLQuantizationType.Q6_K:
        return (elements // 256) * 210
    raise ValueError(f"unsupported row-size lookup for {qtype}")


def _find_ggml_base() -> ctypes.CDLL | None:
    global _GGML_LIB
    if _GGML_LIB is not None:
        return _GGML_LIB

    candidates: list[Path] = []
    override = os.environ.get("NEMO_SPEECH_GGML_BASE")
    if override:
        candidates.append(Path(override))
    for build in repo_root().glob("build*"):
        candidates.extend(build.glob("**/libggml-base.*"))
    candidates.extend(repo_root().joinpath("lib").glob("libggml-base.*"))

    library = None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            library = ctypes.CDLL(str(path))
            break
        except OSError:
            continue
    if library is None:
        soname = ctypes.util.find_library("ggml-base")
        if soname:
            try:
                library = ctypes.CDLL(soname)
            except OSError:
                pass
    if library is None:
        return None

    library.ggml_quantize_chunk.restype = ctypes.c_size_t
    library.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.ggml_quantize_init.argtypes = [ctypes.c_int]
    _GGML_LIB = library
    return library


def quantize(array: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    values = np.ascontiguousarray(array, dtype=np.float32)
    if _gguf_quantize is not None:
        try:
            return _gguf_quantize(values, qtype)
        except NotImplementedError:
            pass

    library = _find_ggml_base()
    if library is None:
        raise RuntimeError(
            f"libggml-base was not found and the gguf package cannot emit {qtype.name}; "
            "build the project or set NEMO_SPEECH_GGML_BASE"
        )
    columns = int(values.shape[-1])
    rows = int(values.size // columns)
    expected = row_size_bytes(qtype, columns) * rows
    output = np.zeros((rows, row_size_bytes(qtype, columns)), dtype=np.uint8)
    library.ggml_quantize_init(int(qtype))
    written = library.ggml_quantize_chunk(
        int(qtype),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.c_void_p),
        0,
        rows,
        columns,
        None,
    )
    if written != expected:
        raise RuntimeError(f"ggml wrote {written} quantized bytes; expected {expected}")
    return output
