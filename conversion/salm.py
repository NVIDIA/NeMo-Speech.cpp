# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speech-LLM (SALM) GGUFs from a speechlm2 checkpoint: the decoder, a standard llama.cpp NemotronH GGUF, and next
to it the encoder file (conversion/salm_encoder.py): the speech encoder and the CTC timestamp head. The decoder names
its encoder file (nemo_speech.speech_model) and both carry the checkpoint's id (nemo_speech.checkpoint_id), so the
runtime finds the pair from the decoder alone and refuses parts of different checkpoints. One encoder file serves
every decoder profile of a checkpoint.

The checkpoint's `llm.*` tensors (Nemotron-H MoE backbone with its MTP layer) go through llama.cpp's NemotronH
converter to BF16 (salm_decoder_hf.py), then the patched llama-quantize rewrites the routed experts and the dense
linears (attention, Mamba in/out, shared experts) by `profile`:

    nvfp4   routed experts NVFP4, dense linears BF16 (default; Blackwell runs the experts natively)
    iq4_nl  routed experts IQ4_NL, dense linears Q8_0 (other GPUs, CPU, Vulkan, Metal; sized for 24 GB GPUs)

Embeddings, the output head and the norms stay BF16/F32.

The speech encoder (conversion/pee.py) is F16 ("none" writes the decoder alone, for a second profile of a checkpoint
whose encoder file exists) and the CTC timestamp head (conversion/ctc_timestamp.py) comes from the checkpoint when it
carries one (ctc_timestamp.*).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from .s2s_components.voicechat_source import ensure_llama_checkout, ensure_quantizer

DEFAULT_PROFILE = "nvfp4"
PROFILES = ("nvfp4", "iq4_nl")
DENSE_TYPE = {"nvfp4": "bf16", "iq4_nl": "q8_0"}  # the dense linears of each profile
ENCODER_TYPES = ("f16", "none")
DEFAULT_ENCODER = "f16"
ENCODER_FILE_KEY = "nemo_speech.speech_model"  # the encoder file the decoder names
CHECKPOINT_ID_KEY = "nemo_speech.checkpoint_id"
EXPERT_TENSORS = "ffn_(up|down|gate)_exps"
DENSE_TENSORS = ("ssm_(in|out)", "ffn_(up|down)_shexp", "attn_(q|k|v|output)")
AUDIO_LOCATOR = "<|audio|>"


def normalize_profile(profile: str) -> str:
    value = profile.lower()
    if value not in PROFILES:
        raise ValueError("salm --outtype must be " + ", ".join(PROFILES))
    return value


def backbone_config(source: Path) -> dict:
    """The LLM backbone's HF config: llm_backbone/config.json, else the SALM config's llm_config."""
    path = source / "llm_backbone" / "config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if isinstance(config.get("llm_config"), dict):
        return config["llm_config"]
    raise RuntimeError(f"{source}: no llm_backbone/config.json and no llm_config in config.json")


def stage(source: Path, staging: Path, weights: bool = True) -> None:
    """An HF-shaped directory for the converter: backbone config, the tokenizer without the audio locator (it
    has no embedding row) and, with `weights`, links to the checkpoint's safetensors."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    config = backbone_config(source)
    if config.get("architectures") != ["NemotronHForCausalLM"]:
        raise RuntimeError(f"unsupported SALM backbone {config.get('architectures')}")
    (staging / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    tokenizer = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer["added_tokens"] = [
        t for t in tokenizer.get("added_tokens", []) if t.get("content") != AUDIO_LOCATOR
    ]
    (staging / "tokenizer.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8"
    )
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_config.pop("audio_token", None)
    tokenizer_config.pop("model_specific_special_tokens", None)
    extra = tokenizer_config.get("extra_special_tokens")
    if isinstance(extra, dict):
        extra.pop("audio_token", None)
    decoder = tokenizer_config.get("added_tokens_decoder") or {}
    for key in [k for k, v in decoder.items() if v.get("content") == AUDIO_LOCATOR]:
        del decoder[key]
    (staging / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if (source / "generation_config.json").is_file():
        shutil.copyfile(source / "generation_config.json", staging / "generation_config.json")

    if not weights:
        return
    files = sorted(source.glob("*.safetensors")) + sorted(source.glob("*.safetensors.index.json"))
    if not any(w.suffix == ".safetensors" for w in files):
        raise RuntimeError(f"{source}: no safetensors")
    for w in files:
        (staging / w.name).symlink_to(w.resolve())


def convert(
    source: Path,
    outfile: Path,
    profile: str = DEFAULT_PROFILE,
    llama_cpp: Path | None = None,
    quantizer: Path | None = None,
    auto_build_quantizer: bool = True,
    dry_run: bool = False,
    encoder: str | None = None,
    checkpoint_name: str | None = None,
) -> None:
    profile = normalize_profile(profile)
    dense = DENSE_TYPE[profile]
    encoder = encoder_type(profile, encoder)
    if encoder not in ENCODER_TYPES:
        raise ValueError("salm encoder type must be " + " or ".join(ENCODER_TYPES))
    name = checkpoint_name or source.name
    repository = Path(__file__).resolve().parent.parent
    llama = ensure_llama_checkout(llama_cpp or repository / "llama.cpp", repository)
    quantize = None
    if not dry_run:  # before the long conversion, so a missing quantizer fails fast
        quantize = ensure_quantizer(
            llama, quantizer, profile=profile, repo_root=repository, auto_build=auto_build_quantizer
        )

    work = outfile.with_name(outfile.name + ".work")
    speech = outfile.with_name(encoder_file_name(name, encoder)) if encoder != "none" else None
    identity = checkpoint_id(source)
    if speech:  # first: about a minute, and a bad checkpoint fails before the decoder's long run
        ctc_source = ctc_checkpoint(source)
        if dry_run:
            print(
                f"[convert-salm] would write the encoder file {speech.name}: the {encoder.upper()} encoder"
                + (f" and the CTC timestamp head of {ctc_source.name}" if ctc_source else "")
            )
        else:
            from . import pee
            from .salm_encoder import write_encoder_file

            work.mkdir(parents=True, exist_ok=True)
            print(f"[convert-salm] encoder -> {encoder.upper()} GGUF")
            pee.convert(source, work / "encoder.gguf", "fp16" if encoder == "f16" else encoder)
            if ctc_source:
                from . import ctc_timestamp

                print(f"[convert-salm] CTC timestamp head of {ctc_source.name}")
                ctc_timestamp.convert(ctc_source, work / "ctc.gguf", name)
            write_encoder_file(
                work / "encoder.gguf",
                work / "ctc.gguf" if ctc_source else None,
                speech,
                name=f"{name} Encoder {encoder.upper()}",
                checkpoint_id=identity,
            )

    staging = work / "hf"
    print(f"[convert-salm] staging the decoder of {source}")
    stage(source, staging)
    bf16 = work / "decoder-bf16.gguf"
    print(f"[convert-salm] decoder -> BF16 GGUF {bf16}")
    keys = decoder_keys(name, identity)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("salm_decoder_hf.py")),
            str(llama),
            str(staging),
            "--outfile",
            str(bf16),
            "--outtype",
            "bf16",
            "--model-name",
            f"{name} {profile.upper()}",
            *[arg for key in keys for arg in ("--nemo-speech-key", key)],
            *(["--dry-run"] if dry_run else []),
        ],
        check=True,
    )
    if dry_run:  # tensor mapping checked, nothing written
        shutil.rmtree(work, ignore_errors=True)
        return
    quantize_decoder(bf16, outfile, profile, quantize, dense)
    shutil.rmtree(work, ignore_errors=True)
    if speech:
        print(f"[convert-salm] {outfile.name} and its encoder file {speech.name}")


def encoder_file_name(name: str, encoder: str = DEFAULT_ENCODER) -> str:
    """The encoder file's name: the checkpoint's name and the encoder type, the same for every decoder."""
    return f"{name}-Encoder-{encoder.upper()}.gguf"


def decoder_keys(name: str, identity: str) -> list[str]:
    """The decoder's nemo_speech keys: the checkpoint id and the encoder file it pairs with, also when this run
    writes no encoder file (--salm-encoder none reuses the checkpoint's existing one)."""
    return [f"{CHECKPOINT_ID_KEY}={identity}", f"{ENCODER_FILE_KEY}={encoder_file_name(name)}"]


def checkpoint_id(source: Path) -> str:
    """An id of the checkpoint's weights: the names, types and shapes of the tensors of its safetensors and the first
    and last 4 KiB of each tensor's data (a few MB read, against the whole file)."""
    digest = hashlib.sha256()
    for path in sorted(source.glob("*.safetensors")):
        with open(path, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(size))
            for tensor in sorted(k for k in header if k != "__metadata__"):
                info = header[tensor]
                begin, end = info["data_offsets"]
                digest.update(f"{tensor}|{info['dtype']}|{info['shape']}|".encode())
                f.seek(8 + size + begin)
                digest.update(f.read(min(4096, end - begin)))
                f.seek(8 + size + max(begin, end - 4096))
                digest.update(f.read(min(4096, end - begin)))
    return digest.hexdigest()[:32]


def ctc_checkpoint(source: Path) -> Path | None:
    """The checkpoint safetensors that carries the CTC timestamp head (ctc_timestamp.*), if any."""
    from safetensors import safe_open

    for path in sorted(source.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            if (f.metadata() or {}).get("ctc_timestamp_format"):
                return path
    return None


def encoder_type(profile: str, encoder: str | None = None) -> str:
    """The encoder's type: F16 (every profile shares one encoder file), or "none"."""
    normalize_profile(profile)
    return (encoder or DEFAULT_ENCODER).lower()


def tensor_rules(profile: str, dense: str = "bf16") -> list[str]:
    """llama-quantize --tensor-type rules: they apply over a quantized base type and the first match wins, so the
    expert and dense rules come first and a catch-all keeps everything else BF16 (the base type is never used).
    """
    rules = [f"{EXPERT_TENSORS}={profile}"]
    if dense != "bf16":
        rules.extend(f"{pattern}={dense}" for pattern in DENSE_TENSORS)
    return rules + [".=bf16"]


def quantize_decoder(
    bf16: Path, outfile: Path, profile: str, quantize: Path, dense: str = "bf16"
) -> None:
    """A BF16 decoder GGUF with its routed experts in `profile` and its dense linears in `dense`."""
    threads = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 8
    )
    temporary = outfile.with_name(outfile.name + ".quantizing")
    print(f"[convert-salm] routed experts -> {profile.upper()}, dense linears -> {dense.upper()}")
    command = [str(quantize)]
    for rule in tensor_rules(profile, dense):
        command += ["--tensor-type", rule]
    command += [str(bf16), str(temporary), "Q8_0", str(threads)]
    try:
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"llama-quantize did not produce {temporary}")
        temporary.replace(outfile)
    finally:
        temporary.unlink(missing_ok=True)
