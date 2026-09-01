# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-repository conversion for Nemotron Labs VoiceChat checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .s2s_components.voicechat_source import (
    default_quantizer_path,
    ensure_llama_checkout,
    ensure_quantizer,
    find_quantizer,
)

DEFAULT_PROFILE = "q4_k_m"

_BASE_COMPONENT_FORMATS = {
    "perception": "q8_0",
    "llm_aux": "bf16",
    "eartts_side": "q8_0",
    "codec": "f32",
    "tts_prompt": "f32",
}


def normalize_profile(profile: str) -> str:
    aliases = {"q4": "q4_k_m", "q4_k": "q4_k_m", "repro": "bf16"}
    normalized = aliases.get(profile.lower(), profile.lower())
    if normalized not in {"bf16", "q4_k_m", "nvfp4"}:
        raise ValueError("S2S --outtype must be bf16, q4_k_m, or nvfp4")
    return normalized


def component_formats(profile: str) -> dict[str, str]:
    normalized = normalize_profile(profile)
    formats = dict(_BASE_COMPONENT_FORMATS)
    formats["llm_backbone"] = normalized
    # Use NVFP4 where it provides a meaningful saving: the large conversational
    # backbone. Keep the smaller, speech-critical EarTTS backbone in Q4_K_M;
    # this adds only about 82 MiB and avoids another aggressive quantization
    # boundary in acoustic-token generation.
    formats["eartts_backbone"] = "q4_k_m" if normalized == "nvfp4" else normalized
    return formats


COMPONENT_FORMATS = component_formats(DEFAULT_PROFILE)

_LEGACY_REQUIRED_FILES = (
    "config.json",
    "perception.safetensors",
    "rnnt-asr.safetensors",
    "embeddings.safetensors",
    "codec.safetensors",
    "rnnt_tokenizer/vocab.json",
    "eartts_vllm/config.json",
    "eartts_vllm/model.safetensors",
    "eartts_vllm/tts_model_init_inputs.pt",
    "nano-v2-vllm/config.json",
    "nano-v2-vllm/model.safetensors",
)

_HF_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "rnnt_tokenizer/vocab.json",
)

RUNTIME_ARTIFACTS = (
    "perception.gguf",
    "llm_aux.gguf",
    "codec.gguf",
    "eartts_vllm/eartts_gemma3.gguf",
    "eartts_vllm/eartts_side.gguf",
    "eartts_vllm/tts_prompt.gguf",
    "nano-v2-vllm/nano-v2-llm.gguf",
    "rnnt_tokenizer/vocab.json",
)

_GGUF_CONTRACTS = {
    "perception.gguf": ("asr", ("proj.weight", "decoder.prediction.embed.weight")),
    "llm_aux.gguf": ("s2s_llm_aux", ("function_head.weight", "embed_tokens.weight")),
    "codec.gguf": ("eartts_codec", ("enc.proj_in.weight", "dec.proj_out.weight", "rvq.cb0.mus")),
    "eartts_vllm/eartts_gemma3.gguf": ("gemma3", ("token_embd.weight", "output_norm.weight")),
    "eartts_vllm/eartts_side.gguf": ("s2s_eartts_side", ("bos_emb", "sampler.rvq_embs")),
    "eartts_vllm/tts_prompt.gguf": (
        "s2s_tts_prompt",
        ("tts_prompt.code", "tts_prompt.subword_ids"),
    ),
    "nano-v2-vllm/nano-v2-llm.gguf": ("nemotron_h", ("token_embd.weight", "output.weight")),
}

_GGUF_REQUIRED_METADATA = {
    "llm_aux.gguf": (
        "s2s_llm_aux.user_channel_weight",
        "s2s_llm_aux.text_channel_weight",
        "s2s_llm_aux.function_channel_weight",
    ),
    "eartts_vllm/eartts_gemma3.gguf": ("gemma3.attention.scale",),
    "eartts_vllm/eartts_side.gguf": ("s2s_eartts_side.guidance_scale",),
}


@dataclass(frozen=True)
class ConversionStep:
    name: str
    command: tuple[str, ...]
    output: Path


def _has_files(path: Path, required: tuple[str, ...]) -> bool:
    return path.is_dir() and all((path / relative).is_file() for relative in required)


def _is_legacy_bundle(path: Path) -> bool:
    return _has_files(path, _LEGACY_REQUIRED_FILES)


def _is_hf_checkpoint(path: Path) -> bool:
    return _has_files(path, _HF_REQUIRED_FILES)


def is_s2s_file_list(files: list[str]) -> bool:
    normalized = {str(Path(name)).replace("\\", "/").lstrip("./") for name in files}
    if all(name in normalized for name in _HF_REQUIRED_FILES):
        return True
    suffixes = (
        "perception.safetensors",
        "rnnt-asr.safetensors",
        "eartts_vllm/tts_model_init_inputs.pt",
        "nano-v2-vllm/model.safetensors",
    )
    return all(any(name.endswith(suffix) for name in normalized) for suffix in suffixes)


def discover_bundle_version(source_root: Path) -> Path:
    """Resolve a public checkpoint root or a legacy repository version."""
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"S2S source must be a model repository directory: {source_root}")
    if _is_hf_checkpoint(source_root) or _is_legacy_bundle(source_root):
        return source_root

    candidates = {
        path.parent.resolve()
        for path in source_root.rglob("perception.safetensors")
        if _is_legacy_bundle(path.parent)
    }
    if not candidates:
        expected = ", ".join(_HF_REQUIRED_FILES)
        raise RuntimeError(
            f"no Nemotron Labs VoiceChat checkpoint found under {source_root}; "
            f"the public repository contains: {expected}"
        )
    repositories = {candidate.parent.resolve() for candidate in candidates}
    if len(repositories) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates))
        raise RuntimeError(f"multiple S2S model repositories found under {source_root}: {rendered}")

    def version_key(path: Path) -> tuple[int, int | str]:
        return (1, int(path.name)) if path.name.isdigit() else (0, path.name)

    selected = max(candidates, key=version_key)
    if len(candidates) > 1:
        print(f"[convert-s2s] selected latest model version: {selected}")
    return selected


def is_s2s_source(source_root: Path) -> bool:
    try:
        discover_bundle_version(source_root)
    except RuntimeError:
        return False
    return True


def resolve_source(source: str, cache_dir: Path, revision: str | None) -> Path:
    local = Path(source).expanduser()
    if local.exists():
        return local.resolve()
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] fetching S2S repository {source}")
    return Path(
        snapshot_download(repo_id=source, revision=revision, cache_dir=str(cache_dir))
    ).resolve()


def _base_model_repository(source: Path) -> str:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    try:
        repository = config["model"]["stt"]["model"]["pretrained_llm"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("VoiceChat config does not identify its base language model") from error
    if not isinstance(repository, str) or not repository.strip():
        raise RuntimeError("VoiceChat config does not identify its base language model")
    return repository


def resolve_base_model_assets(source: Path, cache_dir: Path) -> Path:
    legacy = source / "nano-v2-vllm"
    if (legacy / "config.json").is_file() and (legacy / "tokenizer.json").is_file():
        return legacy
    if not _is_hf_checkpoint(source):
        raise RuntimeError(f"VoiceChat source has no Nano base-model assets: {source}")

    from huggingface_hub import snapshot_download

    repository = _base_model_repository(source)
    print(f"[download] fetching tokenizer and config from {repository}")
    return Path(
        snapshot_download(
            repo_id=repository,
            cache_dir=str(cache_dir),
            allow_patterns=(
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ),
        )
    ).resolve()


def output_bundle_dir(source_root: Path, bundle_version: Path, output_root: Path) -> Path:
    source_root = source_root.resolve()
    output_root = output_root.expanduser().resolve()
    if _is_hf_checkpoint(bundle_version):
        return output_root
    try:
        relative = bundle_version.resolve().relative_to(source_root)
    except ValueError:
        relative = Path(bundle_version.parent.name) / bundle_version.name
    return output_root if relative == Path(".") else output_root / relative


def _append_option(command: list[str], option: str, value: Path | str | None) -> None:
    if value is not None:
        command.extend((option, str(value)))


def build_conversion_plan(
    bundle_version: Path,
    output_bundle: Path,
    llama_cpp: Path | None = None,
    *,
    profile: str = DEFAULT_PROFILE,
    base_model_dir: Path | None = None,
    quantizer: Path | None = None,
) -> list[ConversionStep]:
    profile = normalize_profile(profile)
    formats = component_formats(profile)
    repo_root = Path(__file__).resolve().parent.parent
    component_root = Path(__file__).resolve().parent / "s2s_components"
    llama_root = (llama_cpp or repo_root / "llama.cpp").expanduser().resolve()
    converter = llama_root / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise RuntimeError(
            f"llama.cpp converter is missing at {converter}; run "
            "git submodule update --init llama.cpp or pass --llama-cpp"
        )

    public = _is_hf_checkpoint(bundle_version)
    if public and base_model_dir is None:
        raise ValueError("public VoiceChat conversion requires base_model_dir")
    base_model_dir = base_model_dir or bundle_version / "nano-v2-vllm"
    tokenizer_json = base_model_dir / "tokenizer.json"
    py = sys.executable
    eartts_source = bundle_version if public else bundle_version / "eartts_vllm"
    nano_source = bundle_version if public else bundle_version / "nano-v2-vllm"
    codec_source = bundle_version if public else bundle_version / "codec.safetensors"
    prompt_source = bundle_version if public else eartts_source / "tts_model_init_inputs.pt"
    eartts_output = output_bundle / "eartts_vllm"
    nano_output = output_bundle / "nano-v2-vllm"

    side_command = [
        py,
        str(component_root / "eartts_side.py"),
        str(eartts_source),
        str(eartts_output / "eartts_side.gguf"),
        "--weight-type",
        formats["eartts_side"],
    ]
    eartts_command = [
        py,
        str(component_root / "eartts_backbone.py"),
        "--eartts-dir",
        str(eartts_source),
        "--out",
        str(eartts_output / "eartts_gemma3.gguf"),
        "--llama-cpp",
        str(llama_root),
        "--outtype",
        "bf16",
    ]
    llm_command = [
        py,
        str(component_root / "llm_backbone.py"),
        "--ckpt-dir",
        str(nano_source),
        "--out",
        str(nano_output / "nano-v2-llm.gguf"),
        "--llama-cpp",
        str(llama_root),
        "--arch",
        "new",
        "--outtype",
        "bf16",
    ]
    prompt_command = [
        py,
        str(component_root / "tts_prompt.py"),
        str(prompt_source),
        str(eartts_output / "tts_prompt.gguf"),
    ]
    if public:
        _append_option(side_command, "--tokenizer-json", tokenizer_json)
        _append_option(eartts_command, "--tokenizer-json", tokenizer_json)
        _append_option(llm_command, "--base-model-dir", base_model_dir)
        _append_option(prompt_command, "--tokenizer-json", tokenizer_json)
    for command, backbone_format in (
        (eartts_command, formats["eartts_backbone"]),
        (llm_command, formats["llm_backbone"]),
    ):
        if backbone_format != "bf16":
            _append_option(command, "--quantize", backbone_format)
            _append_option(command, "--quantizer", quantizer)

    return [
        ConversionStep(
            "perception",
            (
                py,
                str(component_root / "perception.py"),
                str(bundle_version),
                str(output_bundle / "perception.gguf"),
                "--weight-type",
                formats["perception"],
            ),
            output_bundle / "perception.gguf",
        ),
        ConversionStep(
            "llm_aux",
            (
                py,
                str(component_root / "llm_aux.py"),
                str(bundle_version),
                str(output_bundle / "llm_aux.gguf"),
                "--weight-type",
                formats["llm_aux"],
            ),
            output_bundle / "llm_aux.gguf",
        ),
        ConversionStep("eartts_side", tuple(side_command), eartts_output / "eartts_side.gguf"),
        ConversionStep(
            "eartts_backbone", tuple(eartts_command), eartts_output / "eartts_gemma3.gguf"
        ),
        ConversionStep("llm_backbone", tuple(llm_command), nano_output / "nano-v2-llm.gguf"),
        ConversionStep(
            "codec",
            (
                py,
                str(component_root / "codec.py"),
                "--codec",
                str(codec_source),
                "--out",
                str(output_bundle / "codec.gguf"),
                "--weight-type",
                formats["codec"],
            ),
            output_bundle / "codec.gguf",
        ),
        ConversionStep("tts_prompt", tuple(prompt_command), eartts_output / "tts_prompt.gguf"),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    path: Path,
    output_bundle: Path,
    profile: str,
) -> None:
    artifacts = {}
    for relative in RUNTIME_ARTIFACTS:
        artifact = output_bundle / relative
        artifacts[relative] = {"bytes": artifact.stat().st_size, "sha256": _sha256(artifact)}
    manifest = {
        "schema": 1,
        "architecture": "s2s",
        "profile": profile,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "component_formats": component_formats(profile),
        "artifacts": artifacts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_bundle(output_bundle: Path) -> None:
    """Validate the runtime layout and GGUF interfaces without comparing encodings."""
    from gguf import GGUFReader

    missing = [
        str(output_bundle / relative)
        for relative in RUNTIME_ARTIFACTS
        if not (output_bundle / relative).is_file()
    ]
    if missing:
        raise RuntimeError(f"S2S conversion is incomplete; missing: {', '.join(missing)}")

    for relative, (architecture, required_tensors) in _GGUF_CONTRACTS.items():
        try:
            reader = GGUFReader(output_bundle / relative, mode="r")
            actual_architecture = reader.fields["general.architecture"].contents()
            tensors = {tensor.name for tensor in reader.tensors}
        except Exception as error:
            raise RuntimeError(f"invalid S2S GGUF {relative}: {error}") from error
        if actual_architecture != architecture:
            raise RuntimeError(
                f"invalid architecture in {relative}: expected {architecture}, "
                f"found {actual_architecture}"
            )
        absent = [name for name in required_tensors if name not in tensors]
        if absent:
            raise RuntimeError(f"missing required tensors in {relative}: {', '.join(absent)}")
        absent_metadata = [
            name for name in _GGUF_REQUIRED_METADATA.get(relative, ()) if name not in reader.fields
        ]
        if absent_metadata:
            raise RuntimeError(
                f"missing required metadata in {relative}: {', '.join(absent_metadata)}"
            )
        # GGUFReader memory-maps tensor payloads. Release each mapping before
        # opening the next multi-gigabyte artifact or bundle validation can
        # exhaust virtual mappings and crash inside numpy's mmap teardown.
        del reader
        gc.collect()

    try:
        vocab = json.loads((output_bundle / "rnnt_tokenizer" / "vocab.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid RNNT vocabulary: {error}") from error
    if not isinstance(vocab, list) or not vocab or not all(isinstance(item, str) for item in vocab):
        raise RuntimeError("RNNT vocabulary must be a non-empty JSON string array")


def convert(
    source: str,
    output_root: Path,
    *,
    cache_dir: Path,
    outtype: str = DEFAULT_PROFILE,
    revision: str | None = None,
    llama_cpp: Path | None = None,
    quantizer: Path | None = None,
    auto_build_quantizer: bool = True,
    force: bool = False,
    dry_run: bool = False,
    metadata_json: Path | None = None,
) -> Path:
    profile = normalize_profile(outtype)
    repo_root = Path(__file__).resolve().parent.parent
    llama_root = (llama_cpp or repo_root / "llama.cpp").expanduser().resolve()
    ensure_llama_checkout(llama_root, repo_root)
    source_root = resolve_source(source, cache_dir, revision)
    bundle_version = discover_bundle_version(source_root)
    destination = output_bundle_dir(source_root, bundle_version, output_root)
    if destination == bundle_version or source_root in destination.parents:
        raise RuntimeError("S2S output directory must be separate from the source repository")

    base_model_dir = resolve_base_model_assets(bundle_version, cache_dir)
    plan = build_conversion_plan(
        bundle_version,
        destination,
        llama_root,
        profile=profile,
        base_model_dir=base_model_dir,
        quantizer=quantizer,
    )
    vocab_output = destination / "rnnt_tokenizer" / "vocab.json"
    outputs = [step.output for step in plan] + [vocab_output]
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            f"S2S output artifacts already exist: {rendered}; pass --force to overwrite"
        )

    if profile != "bf16":
        if dry_run:
            if quantizer is not None:
                quantizer = find_quantizer(llama_root, quantizer, profile=profile)
            else:
                try:
                    quantizer = find_quantizer(llama_root, profile=profile)
                except FileNotFoundError:
                    if not auto_build_quantizer:
                        raise FileNotFoundError(
                            "llama-quantize is required and automatic build is disabled; "
                            "pass --llama-quantize PATH"
                        )
                    quantizer = default_quantizer_path(llama_root)
                    print(f"[convert-s2s] dry run: would build llama-quantize at {quantizer}")
        else:
            quantizer = ensure_quantizer(
                llama_root,
                quantizer,
                profile=profile,
                repo_root=repo_root,
                auto_build=auto_build_quantizer,
            )
        plan = build_conversion_plan(
            bundle_version,
            destination,
            llama_root,
            profile=profile,
            base_model_dir=base_model_dir,
            quantizer=quantizer,
        )

    formats = component_formats(profile)
    print(f"[convert-s2s] source checkpoint: {bundle_version}")
    print(f"[convert-s2s] output bundle: {destination}")
    print(f"[convert-s2s] component formats: {json.dumps(formats, sort_keys=True)}")
    for step in plan:
        print(f"[convert-s2s] {step.name}: {' '.join(step.command)}")
        if dry_run:
            continue
        step.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(step.command, cwd=Path(__file__).resolve().parent.parent, check=True)
        if not step.output.is_file() or step.output.stat().st_size == 0:
            raise RuntimeError(f"S2S converter did not produce {step.output}")

    if dry_run:
        return destination

    vocab_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_version / "rnnt_tokenizer" / "vocab.json", vocab_output)
    validate_bundle(destination)
    manifest_path = destination / "manifest.json"
    _write_manifest(manifest_path, destination, profile)
    if metadata_json is not None and metadata_json.expanduser().resolve() != manifest_path:
        _write_manifest(
            metadata_json.expanduser().resolve(),
            destination,
            profile,
        )
    print(f"[convert-s2s] complete: {destination}")
    return destination
