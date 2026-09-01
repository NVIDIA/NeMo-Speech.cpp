# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture detection and dispatch for the unified converter."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .source import list_hugging_face_files, read_nemo_config, resolve_nemo_source

ARCHITECTURES = ("asr", "diarization", "pnc", "vad", "tts", "codec", "nmt", "s2s")


@dataclass
class ConversionRequest:
    source: str
    outfile: Path
    architecture: str = "auto"
    outtype: str = "auto"
    revision: str | None = None
    cache_dir: Path = Path.home() / ".cache" / "huggingface" / "hub"
    head_type: str | None = None
    q8_layout: str = "block"
    max_seq_length: int = 128
    metadata_json: Path | None = None
    local_transformer_outtype: str | None = None
    silero_version: str = "6.2.0"
    from_whisper_ggml: Path | None = None
    llama_cpp: Path | None = None
    llama_quantize: Path | None = None
    auto_build_quantizer: bool = True
    force: bool = False
    dry_run: bool = False


def _architecture_from_config(config: dict) -> str:
    target = str(config.get("target", config.get("_target_", ""))).lower()
    if "sortformer" in target or "sortformer_modules" in config:
        return "diarization"
    if "punctuationcapitalization" in target or ("punct_head" in config and "capit_head" in config):
        return "pnc"
    if "magpietts" in target:
        return "tts"
    if "audiocodecmodel" in target or "vector_quantizer" in config:
        return "codec"
    if "encoder" in config and ("decoder" in config or "joint" in config or "labels" in config):
        return "asr"
    raise RuntimeError(
        "cannot determine checkpoint architecture; pass --architecture " + "|".join(ARCHITECTURES)
    )


def _resolve_nemo_for_detection(request: ConversionRequest) -> Path | None:
    local = Path(request.source).expanduser()
    if local.exists():
        if local.is_dir():
            from . import s2s

            if s2s.is_s2s_source(local):
                return None
        if local.is_dir() and not any(local.rglob("model_config.yaml")):
            return None
        if local.is_file() and local.suffix.lower() != ".nemo":
            return None
        return local

    if request.source.lower() in ("silero", "silero-vad", "vad"):
        return None
    files = list_hugging_face_files(request.source, request.revision)
    if any(name.endswith(".nemo") for name in files):
        return resolve_nemo_source(request.source, request.cache_dir, request.revision)
    from . import s2s

    if s2s.is_s2s_file_list(files):
        return None
    if "config.json" in files or any(name.endswith(".safetensors") for name in files):
        return None
    raise RuntimeError(f"no supported checkpoint found in {request.source}")


def detect_architecture(request: ConversionRequest) -> tuple[str, Path | None]:
    if request.architecture != "auto":
        if request.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture: {request.architecture}")
        if request.architecture in ("vad", "nmt"):
            return request.architecture, None
        if request.architecture == "s2s":
            local = Path(request.source).expanduser()
            return request.architecture, local.resolve() if local.exists() else None
        return (
            request.architecture,
            resolve_nemo_source(request.source, request.cache_dir, request.revision),
        )

    if request.source.lower() in ("silero", "silero-vad", "vad"):
        return "vad", None
    local = Path(request.source).expanduser()
    if local.is_dir():
        from . import s2s

        if s2s.is_s2s_source(local):
            return "s2s", local.resolve()
    checkpoint = _resolve_nemo_for_detection(request)
    if checkpoint is None:
        if local.is_dir():
            from . import s2s

            if s2s.is_s2s_source(local):
                return "s2s", local.resolve()
        elif not local.exists():
            files = list_hugging_face_files(request.source, request.revision)
            from . import s2s

            if s2s.is_s2s_file_list(files):
                return "s2s", None
        return "nmt", None
    return _architecture_from_config(read_nemo_config(checkpoint)), checkpoint


def _normalized_outtype(architecture: str, outtype: str) -> str:
    defaults = {
        "asr": "q8_0",
        "diarization": "f32",
        "pnc": "q8_0",
        "vad": "f32",
        "tts": "f16",
        "codec": "f16",
        "nmt": "f16",
        "s2s": "q4_k_m",
    }
    value = defaults[architecture] if outtype == "auto" else outtype.lower()
    if value == "fp16":
        value = "f16"
    if architecture == "s2s":
        from .s2s import normalize_profile

        value = normalize_profile(value)
    supported = {
        "asr": {"f16", "bf16", "q8_0", "q4_k", "q5_k", "q6_k", "nvfp4", "mxfp4"},
        "diarization": {
            "f32",
            "f16",
            "bf16",
            "q8_0",
            "q4_k",
            "q5_k",
            "q6_k",
            "nvfp4",
            "mxfp4",
        },
        "pnc": {"f16", "bf16", "q8_0"},
        "vad": {"f32"},
        "tts": {"f16", "f32"},
        "codec": {"f16", "f32"},
        "nmt": {"f32", "f16", "bf16", "q8_0", "auto"},
        "s2s": {"bf16", "q4_k_m", "nvfp4"},
    }
    if value not in supported[architecture]:
        choices = ", ".join(sorted(supported[architecture]))
        raise ValueError(f"{architecture} does not support --outtype {value}; choose {choices}")
    return value


def _convert_nmt(request: ConversionRequest, outtype: str) -> None:
    repository = Path(__file__).resolve().parent.parent
    from .s2s_components.voicechat_source import ensure_llama_checkout

    llama_cpp = ensure_llama_checkout(request.llama_cpp or repository / "llama.cpp", repository)
    script = llama_cpp / "convert_hf_to_gguf.py"
    source = Path(request.source).expanduser()
    if not source.exists():
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                repo_id=request.source,
                revision=request.revision,
                cache_dir=str(request.cache_dir),
            )
        )
    command = [
        sys.executable,
        str(script),
        str(source),
        "--outfile",
        str(request.outfile),
        "--outtype",
        outtype,
    ]
    subprocess.run(command, check=True)


def convert_model(request: ConversionRequest) -> str:
    architecture, checkpoint = detect_architecture(request)
    outtype = _normalized_outtype(architecture, request.outtype)
    request.outfile.parent.mkdir(parents=True, exist_ok=True)

    if architecture == "asr":
        from . import asr

        assert checkpoint is not None
        asr.convert(
            checkpoint,
            request.outfile,
            request.head_type,
            "fp16" if outtype == "f16" else outtype,
            request.q8_layout,
        )
    elif architecture == "diarization":
        from . import diarization

        assert checkpoint is not None
        diarization.convert(checkpoint, request.outfile, "fp16" if outtype == "f16" else outtype)
    elif architecture == "pnc":
        from . import pnc

        assert checkpoint is not None
        pnc.convert(
            checkpoint,
            request.outfile,
            "fp16" if outtype == "f16" else outtype,
            request.max_seq_length,
        )
    elif architecture == "vad":
        from . import vad

        vad.convert(request.outfile, request.silero_version, request.from_whisper_ggml)
    elif architecture == "tts":
        from . import tts

        assert checkpoint is not None
        tts.convert(
            checkpoint,
            request.outfile,
            outtype,
            request.metadata_json,
            request.local_transformer_outtype,
        )
    elif architecture == "codec":
        from . import codec

        assert checkpoint is not None
        codec.convert(checkpoint, request.outfile, outtype, request.metadata_json)
    elif architecture == "nmt":
        _convert_nmt(request, outtype)
    else:
        from . import s2s

        s2s.convert(
            request.source,
            request.outfile,
            cache_dir=request.cache_dir,
            outtype=outtype,
            revision=request.revision,
            llama_cpp=request.llama_cpp,
            quantizer=request.llama_quantize,
            auto_build_quantizer=request.auto_build_quantizer,
            force=request.force,
            dry_run=request.dry_run,
            metadata_json=request.metadata_json,
        )
    return architecture
