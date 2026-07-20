#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unpack a reference-dump .npz into raw .bin files C++ parity tests can read.

Each array `key` becomes `<outdir>/<key with '/'→'__'>.bin` with the layout:
    int64 n_dims, int64 dims[n_dims], then the data as float32 (or int64 for
    integer arrays, flagged by dtype code).

Header: magic 'NERB', uint32 dtype (0=f32, 1=i64), int64 n_dims, int64 dims[].

Usage: python export_ref_bins.py <ref.npz> <outdir>
"""

import struct
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    npz_path, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    for key in data.files:
        arr = data[key]
        if arr.dtype.kind == "f":
            arr, code = arr.astype(np.float32), 0
        else:
            arr, code = arr.astype(np.int64), 1
        fname = outdir / (key.replace("/", "__") + ".bin")
        with open(fname, "wb") as f:
            f.write(b"NERB")
            f.write(struct.pack("<I", code))
            f.write(struct.pack("<q", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<q", d))
            f.write(np.ascontiguousarray(arr).tobytes())
    print(f"[export] {len(data.files)} arrays -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
