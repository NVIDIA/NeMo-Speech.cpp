#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Silero-VAD → GGUF converter for nemo-speech.

Produces the sidecar VAD GGUF that `src/asr/vad/silero_vad.cpp` loads
through its own `ggml_runtime::GGUFLoader` instance (separate from the ASR
GGUF). VAD is an orthogonal, opt-in pipeline stage, so it ships as its own
file rather than being bundled into the ASR model — swapping VAD versions
must not force an ASR re-conversion.

Output convention (matches what `SileroVad` expects):

    * `general.architecture = "vad"`
    * Metadata namespace `vad.*` (hparams the C++ port reads at load time)
    * Tensors renamed from the silero PyTorch state_dict to the names the
      runtime modules load against:
        _model.encoder.{i}.reparam_conv.{weight,bias}  → encoder.{i}.conv.{weight,bias}
        _model.decoder.rnn.weight_ih / bias_ih         → lstm.ih.{weight,bias}
        _model.decoder.rnn.weight_hh / bias_hh         → lstm.hh.{weight,bias}
        _model.decoder.decoder.2.{weight,bias}         → final_conv.{weight,bias}
        _model.stft.forward_basis_buffer               → stft.basis
    * Conv weights pass through in PyTorch-native (out, in, k) layout — the
      gguf reader reverses dims to [k, in, out], which is exactly what the
      runtime's Conv1D registers. The final pointwise conv (k=1) is squeezed
      to 2D (out, in) so the runtime's Conv1D dispatches it through
      `ggml_mul_mat` (see src/runtime/ggml/nn.h Conv1D::is_pointwise_2d).
    * Everything is emitted F32. The model is tiny (~1.8 MB) so there's no
      quantization win, and F32 keeps the per-window probability parity test
      against whisper.cpp comfortably under the 1e-4 tolerance.

This mirrors whisper.cpp's `models/convert-silero-vad-to-ggml.py` (same source
checkpoint, same hparams) but writes GGUF instead of whisper's custom GGML
binary, which our GGUFLoader can't read.

Two input sources:
    1. The silero-vad pip package (production path, full-precision checkpoint):
           pip install "silero-vad==6.2.0"
           python3 convert_model.py silero --outfile out/silero-v6.2.0.gguf
    2. whisper.cpp's checked-in GGML binary (offline / no-network path; conv
       weights are F16 there, fine for the <=1e-4 parity test):
           python3 convert_model.py silero --outfile out/silero-v6.2.0.gguf \\
               --from-whisper-ggml ../whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin

Both sources funnel through one emit routine keyed by the original silero
state_dict tensor names.
"""

import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter

ARCH = "vad"

# Fixed architecture of the silero-v6 16 kHz model (matches whisper.cpp's
# converter). Encoder is 4 reparam Conv1D layers; the decoder is a single
# LSTMCell + a pointwise conv → sigmoid producing one speech probability per
# 512-sample (32 ms @ 16 kHz) window.
N_ENCODER_LAYERS = 4
ENCODER_IN_CHANNELS = [129, 128, 64, 64]
ENCODER_OUT_CHANNELS = [128, 64, 64, 128]
ENCODER_KERNEL_SIZE = 3
# Per-layer stride. Layer 0 and 3 are stride-1, the two middle layers stride-2
# (see whisper_vad_build_encoder_layer in whisper.cpp/src/whisper.cpp:4542).
ENCODER_STRIDES = [1, 2, 2, 1]
LSTM_INPUT_SIZE = 128
LSTM_HIDDEN_SIZE = 128
FINAL_CONV_IN = 128
FINAL_CONV_OUT = 1
# STFT front-end: 512-sample window reflect-padded by 64 on each side, then a
# fixed Fourier-basis conv (kernel 256, stride 128) → 129 freq bins.
WINDOW_SIZE = 512
CONTEXT_SIZE = 64
STFT_N_FREQS = 129


def _np(t) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def load_from_silero_package(silero_version: str) -> "dict[str, np.ndarray]":
    """Production path: load the bundled 16 kHz checkpoint via the pip package.

    Returns a dict keyed by the original `_model.*` state_dict names, with
    numpy arrays in PyTorch-native shapes (the 8 kHz sub-model dropped).
    """
    try:
        from silero_vad import __version__ as pkg_version
        from silero_vad import load_silero_vad
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "silero-vad not importable. Install the pinned version:\n"
            '    pip install "silero-vad==6.2.0"'
        ) from e

    if pkg_version != silero_version:
        print(
            f"WARN: silero-vad package is v{pkg_version} but --silero-version "
            f"is {silero_version}. The converter is validated with 6.2.0; "
            f"mismatched versions may shift "
            f"per-window probabilities.",
            file=sys.stderr,
        )

    sd = {}
    for key, value in load_silero_vad().state_dict().items():
        if "_8k" in key:
            continue
        clean = key if key.startswith("_model.") else "_model." + key
        sd[clean] = _np(value)
    return sd


def load_from_whisper_ggml(path: Path) -> "tuple[dict[str, np.ndarray], str]":
    """Offline path: parse whisper.cpp's custom GGML binary.

    Mirrors the on-disk layout written by
    whisper.cpp/models/convert-silero-vad-to-ggml.py: a fixed header, then a
    flat list of tensors whose dimensions are stored GGML-order (numpy-
    reversed). Returns ({original_name: numpy-native-shape array}, version).
    """
    with open(path, "rb") as f:

        def i32() -> int:
            return struct.unpack("i", f.read(4))[0]

        magic = i32()
        if magic != 0x67676D6C:
            raise RuntimeError(f"{path}: bad GGML magic 0x{magic:08x}")
        model_type = f.read(i32()).decode("utf-8")
        major, minor, patch = i32(), i32(), i32()
        version = f"{major}.{minor}.{patch}"
        _window, _context = i32(), i32()
        n_layers = i32()
        for _ in range(n_layers):
            i32(), i32(), i32()  # in_ch, out_ch, kernel — recomputed from tensors
        i32(), i32()  # lstm input/hidden
        i32(), i32()  # final conv in/out
        print(f"[from-ggml] type={model_type} version={version}")

        tensors: "dict[str, np.ndarray]" = {}
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            n_dims, name_len, ftype = struct.unpack("iii", hdr)
            dims = struct.unpack("i" * n_dims, f.read(4 * n_dims))
            name = f.read(name_len).decode("utf-8")
            count = int(np.prod(dims)) if n_dims else 1
            if ftype == 1:
                data = np.frombuffer(f.read(count * 2), dtype=np.float16).astype(np.float32)
            else:
                data = np.frombuffer(f.read(count * 4), dtype=np.float32)
            # GGML dims are numpy-reversed; restore PyTorch-native shape.
            tensors[name] = data.reshape(tuple(reversed(dims)))
    return tensors, version


def emit_gguf(out_path: Path, silero_version: str, sd: "dict[str, np.ndarray]") -> None:
    def get(src: str) -> np.ndarray:
        if src not in sd:
            raise RuntimeError(
                f"missing tensor {src!r} in silero state_dict; keys present: "
                f"{sorted(sd.keys())[:20]} ..."
            )
        return sd[src]

    gw = GGUFWriter(str(out_path), arch=ARCH)
    gw.add_architecture()
    gw.add_string("general.name", f"silero-vad-v{silero_version}")
    gw.add_uint32("general.file_type", 0)  # ALL_F32

    # ---- hparams under vad.* ----
    gw.add_string("vad.version", silero_version)
    gw.add_uint32("vad.sample_rate", 16000)
    gw.add_uint32("vad.window_size", WINDOW_SIZE)
    gw.add_uint32("vad.context_size", CONTEXT_SIZE)
    gw.add_uint32("vad.n_encoder_layers", N_ENCODER_LAYERS)
    gw.add_array("vad.encoder.in_channels", ENCODER_IN_CHANNELS)
    gw.add_array("vad.encoder.out_channels", ENCODER_OUT_CHANNELS)
    gw.add_array("vad.encoder.strides", ENCODER_STRIDES)
    gw.add_uint32("vad.encoder.kernel_size", ENCODER_KERNEL_SIZE)
    gw.add_uint32("vad.lstm.input_size", LSTM_INPUT_SIZE)
    gw.add_uint32("vad.lstm.hidden_size", LSTM_HIDDEN_SIZE)
    gw.add_uint32("vad.final_conv.in_channels", FINAL_CONV_IN)
    gw.add_uint32("vad.final_conv.out_channels", FINAL_CONV_OUT)
    gw.add_uint32("vad.stft.n_freqs", STFT_N_FREQS)
    # STFT Fourier-basis conv shape, so the C++ define_tensors declares
    # `stft.basis` without guessing (GGUFLoader exposes n_dims, not sizes).
    # PyTorch native (out=2*n_freqs, in=1, k=filter_length).
    _basis = get("_model.stft.forward_basis_buffer")
    gw.add_uint32("vad.stft.n_basis", int(_basis.shape[0]))  # 2 * n_freqs (258)
    gw.add_uint32("vad.stft.filter_length", int(_basis.shape[-1]))  # 256

    def emit(name: str, arr: np.ndarray) -> None:
        gw.add_tensor(
            name, np.ascontiguousarray(arr.astype(np.float32)), raw_dtype=GGMLQuantizationType.F32
        )
        print(f"  {name:28s} {tuple(arr.shape)}")

    print(f"Writing tensors to {out_path}:")

    # ---- Encoder: 4 reparam Conv1D layers ----
    # PyTorch weight shape (out, in, k); passed through native — the reader
    # reverses to [k, in, out] which is what Conv1D::define_tensors registers.
    for i in range(N_ENCODER_LAYERS):
        emit(
            f"encoder.{i}.conv.weight", get(f"_model.encoder.{i}.reparam_conv.weight")
        )  # (out,in,k)
        emit(f"encoder.{i}.conv.bias", get(f"_model.encoder.{i}.reparam_conv.bias"))  # (out,)

    # ---- LSTM cell: ih / hh weight + bias (each gate-stacked 4*hidden) ----
    # PyTorch Linear-style (out=4*hidden, in); native passthrough → reader
    # [in, 4*hidden], which the runtime's Linear module owns directly.
    emit("lstm.ih.weight", get("_model.decoder.rnn.weight_ih"))  # (512, 128)
    emit("lstm.ih.bias", get("_model.decoder.rnn.bias_ih"))  # (512,)
    emit("lstm.hh.weight", get("_model.decoder.rnn.weight_hh"))  # (512, 128)
    emit("lstm.hh.bias", get("_model.decoder.rnn.bias_hh"))  # (512,)

    # ---- Final pointwise conv (k=1) → store 2D (out, in) for mul_mat path ----
    # Package source gives (1, 128, 1); whisper-ggml source already squeezed
    # to (128,). Normalise both to (out=1, in=128).
    fc_w = get("_model.decoder.decoder.2.weight")
    if fc_w.ndim == 3 and fc_w.shape[-1] == 1:
        fc_w = fc_w.squeeze(axis=-1)  # (1, 128, 1) → (1, 128)
    elif fc_w.ndim == 1:
        fc_w = fc_w.reshape(1, -1)  # (128,) → (1, 128)
    emit("final_conv.weight", fc_w)
    emit("final_conv.bias", get("_model.decoder.decoder.2.bias").reshape(-1))  # (1,)

    # ---- STFT Fourier basis (conv-weight layout, native passthrough) ----
    # PyTorch (out=2*n_freqs, in=1, k=window) → reader [window, 1, 2*n_freqs];
    # the first n_freqs output channels are the real part, the second n_freqs
    # the imaginary part (see whisper_vad_build_stft_layer).
    emit("stft.basis", get("_model.stft.forward_basis_buffer"))

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"\nDone. Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


def convert(
    output: Path,
    silero_version: str = "6.2.0",
    whisper_ggml: Path | None = None,
) -> None:
    if whisper_ggml:
        state_dict, version = load_from_whisper_ggml(whisper_ggml)
        if version != silero_version:
            print(f"[from-ggml] using version {version} from the GGML header")
    else:
        version = silero_version
        state_dict = load_from_silero_package(version)
    emit_gguf(output, version, state_dict)
