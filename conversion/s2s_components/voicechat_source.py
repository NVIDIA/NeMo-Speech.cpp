# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-layout and post-conversion helpers for VoiceChat components."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

HF_MODEL_FILENAME = "model.safetensors"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_quantizer_path(llama_cpp: Path) -> Path:
    """Return the converter-managed llama-quantize output path."""
    return llama_cpp.expanduser().resolve() / "build-quantize" / "bin" / "llama-quantize"


def ensure_llama_checkout(llama_cpp: Path, repo_root: Path | None = None) -> Path:
    """Initialize the pinned llama.cpp submodule when the default checkout is absent."""
    root = (repo_root or _project_root()).expanduser().resolve()
    checkout = llama_cpp.expanduser().resolve()
    converter = checkout / "convert_hf_to_gguf.py"
    if converter.is_file():
        return checkout

    default_checkout = (root / "llama.cpp").resolve()
    if checkout != default_checkout:
        raise RuntimeError(
            f"llama.cpp converter is missing at {converter}; initialize that checkout or "
            "omit --llama-cpp to use the pinned submodule"
        )

    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "the pinned llama.cpp submodule is not initialized and git is unavailable; "
            "install git or pass --llama-cpp /path/to/llama.cpp"
        )
    print("[convert-model] initializing pinned llama.cpp submodule")
    try:
        subprocess.run([git, "submodule", "update", "--init", "llama.cpp"], cwd=root, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "could not initialize the pinned llama.cpp submodule; check network access, "
            "clone with --recursive, or pass --llama-cpp /path/to/llama.cpp"
        ) from error
    if not converter.is_file():
        raise RuntimeError(f"llama.cpp submodule initialized without {converter}")
    return checkout


def is_hf_voicechat_source(path: Path) -> bool:
    return all(
        (path / relative).is_file()
        for relative in ("config.json", HF_MODEL_FILENAME, "rnnt_tokenizer/vocab.json")
    )


def load_voicechat_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config.get("model"), dict):
        raise ValueError(f"{config_path} is not a VoiceChat model config")
    return config


def load_llm_channel_weights(path: Path) -> dict[str, float]:
    """Read the duplex input-channel weights used by the VoiceChat LLM."""
    root = load_voicechat_config(path)
    try:
        model = root["model"]["stt"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path / 'config.json'} has no VoiceChat STT model config") from error
    return {
        "user": float(model.get("duplex_user_channel_weight", 1.0)),
        "text": float(model.get("duplex_text_channel_weight", 1.0)),
        "function": float(model.get("duplex_function_channel_weight", 1.0)),
    }


def build_character_tables(tokenizer_json: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Recreate the deterministic subword-to-character tables used by EarTTS."""
    tokenizer = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"{tokenizer_json} has no tokenizer model vocabulary")

    single_chars = {token: int(token_id) for token, token_id in vocab.items() if len(token) == 1}
    ordered_chars = sorted(single_chars, key=single_chars.get)
    char_vocab = {char: index for index, char in enumerate(ordered_chars)}
    mappings = {
        int(token_id): tuple(char_vocab[char] for char in token if char in char_vocab)
        for token, token_id in vocab.items()
    }
    mappings = {token_id: chars for token_id, chars in mappings.items() if chars}
    sentinel_id = max(int(token_id) for token_id in vocab.values()) + 1
    mappings[sentinel_id] = (len(char_vocab),)
    max_char_len = max(map(len, mappings.values()))

    ids = np.zeros((sentinel_id + 1, max_char_len), dtype=np.float32)
    mask = np.zeros_like(ids)
    for token_id, chars in mappings.items():
        ids[token_id, : len(chars)] = chars
        mask[token_id, : len(chars)] = 1.0
    return ids, mask, len(char_vocab)


def build_eartts_config(source: Path, tokenizer_json: Path) -> dict[str, Any]:
    """Flatten the public VoiceChat config into the runtime EarTTS schema."""
    if not is_hf_voicechat_source(source):
        config_path = source / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        return json.loads(config_path.read_text(encoding="utf-8"))

    root = load_voicechat_config(source)
    speech = root["model"]["speech_generation"]
    model = speech["model"]
    tts = model["tts_config"]
    backbone = dict(tts["backbone_config"])
    char = dict(tts["cas_config"]["backbone_config"])
    mog = tts["mog_head_config"]
    tokenizer = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"{tokenizer_json} has no tokenizer model vocabulary")
    char_ids, _, char_vocab_size = build_character_tables(tokenizer_json)

    num_layers = int(backbone["num_hidden_layers"])
    layer_types = [
        "full_attention" if (layer + 1) % 6 == 0 else "sliding_attention"
        for layer in range(num_layers)
    ]
    config: dict[str, Any] = {
        "architectures": ["EarTTSForCausalLM"],
        "model_type": "eartts",
        "vocab_size": 1,
        "backbone_type": tts["backbone_type"],
        **backbone,
        "max_position_embeddings": 131072,
        "layer_types": layer_types,
        "latent_size": tts["latent_size"],
        "codebook_size": tts["codebook_size"],
        "num_quantizers": tts["num_quantizers"],
        "exponent": tts["exponent"],
        "mog_num_layers": mog["num_layers"],
        "mog_low_rank": mog["low_rank"],
        "mog_num_predictions": mog["num_predictions"],
        "mog_min_log_std": mog["min_log_std"],
        "mog_eps": mog["eps"],
        "num_iter": 8,
        "noise_scale": model.get("inference_noise_scale", 0.8),
        "top_p_or_k": model.get("inference_top_p_or_k", 0.8),
        "guidance_scale": model.get("inference_guidance_scale", 0.5),
        "emb_backbone_config": char,
        "emb_backbone_type": tts["cas_config"]["backbone_type"],
        "emb_vocab_size": int(char_ids.shape[0]),
        "emb_char_vocab_size": char_vocab_size,
        "max_char_len": char_ids.shape[1],
        "pretrained_tokenizer_name": tts["cas_config"].get("pretrained_tokenizer_name", ""),
        "use_subword_flag_emb": tts.get("use_subword_flag_emb", False),
        "use_bos_eos_emb": tts.get("use_bos_eos_emb", False),
        "use_gated_fusion_for_text_audio": tts.get("use_gated_fusion_for_text_audio", False),
        "use_audio_prompt_frozen_projection": tts.get("use_audio_prompt_frozen_projection", False),
        "enable_guidance": True,
    }
    return config


def _quantizer_supports_profile(quantizer: Path, profile: str | None) -> bool:
    """Check capabilities that are not present in every llama.cpp quantizer."""
    if profile is None or profile.lower() != "nvfp4":
        return True
    try:
        result = subprocess.run(
            [str(quantizer), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    help_text = f"{result.stdout}\n{result.stderr}".upper()
    return "NVFP4" in help_text


def find_quantizer(
    llama_cpp: Path,
    explicit: Path | None = None,
    *,
    profile: str | None = None,
) -> Path:
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise FileNotFoundError(f"llama-quantize is not executable: {resolved}")
        if not _quantizer_supports_profile(resolved, profile):
            raise RuntimeError(
                f"llama-quantize at {resolved} does not advertise NVFP4 support; "
                "use the converter-managed patched llama.cpp build"
            )
        return resolved

    candidates = [
        llama_cpp / "build" / "bin" / "llama-quantize",
        llama_cpp / "build-quantize" / "bin" / "llama-quantize",
    ]
    candidates.extend(sorted(llama_cpp.glob("build*/bin/llama-quantize")))
    installed = shutil.which("llama-quantize")
    if installed:
        candidates.append(Path(installed))
    for candidate in candidates:
        resolved = candidate.resolve()
        if (
            resolved.is_file()
            and os.access(resolved, os.X_OK)
            and _quantizer_supports_profile(resolved, profile)
        ):
            return resolved
    capability = " with NVFP4 support" if profile and profile.lower() == "nvfp4" else ""
    raise FileNotFoundError(
        f"llama-quantize{capability} is required and no compatible binary was found"
    )


def ensure_quantizer(
    llama_cpp: Path,
    explicit: Path | None = None,
    *,
    profile: str | None = None,
    repo_root: Path | None = None,
    auto_build: bool = True,
) -> Path:
    """Find or build the patched llama-quantize tool before conversion starts."""
    checkout = llama_cpp.expanduser().resolve()
    if explicit is not None:
        return find_quantizer(checkout, explicit, profile=profile)
    try:
        return find_quantizer(checkout, profile=profile)
    except FileNotFoundError as missing:
        if not auto_build:
            raise FileNotFoundError(
                f"{missing}; automatic build is disabled, so pass --llama-quantize PATH"
            ) from missing

    root = (repo_root or _project_root()).expanduser().resolve()
    checkout = ensure_llama_checkout(checkout, root)
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError(
            "llama-quantize is not available and its automatic build requires CMake and a "
            "C++ compiler; install the build prerequisites or pass --llama-quantize PATH"
        )

    if checkout == (root / "llama.cpp").resolve():
        patch_script = root / "scripts" / "apply-llama-patches.sh"
        if not patch_script.is_file():
            raise RuntimeError(f"missing llama.cpp patch helper: {patch_script}")
        print("[convert-s2s] applying pinned llama.cpp compatibility patches")
        subprocess.run(["bash", str(patch_script)], cwd=root, check=True)

    build_dir = checkout / "build-quantize"
    configure = [
        cmake,
        "-S",
        str(checkout),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
    ]
    if not (build_dir / "CMakeCache.txt").exists() and shutil.which("ninja"):
        configure[1:1] = ["-G", "Ninja"]

    print(f"[convert-s2s] building llama-quantize once in {build_dir}")
    subprocess.run(configure, check=True)
    subprocess.run(
        [cmake, "--build", str(build_dir), "--target", "llama-quantize", "--parallel"],
        check=True,
    )
    built = default_quantizer_path(checkout)
    try:
        return find_quantizer(checkout, built, profile=profile)
    except FileNotFoundError as error:
        raise RuntimeError(f"automatic llama-quantize build did not produce {built}") from error


def quantize_backbone(
    source: Path,
    destination: Path,
    profile: str,
    llama_cpp: Path,
    quantizer: Path | None,
) -> None:
    quantizer_path = find_quantizer(llama_cpp, quantizer, profile=profile)
    quant_type = {"q4_k_m": "Q4_K_M", "nvfp4": "NVFP4"}[profile]
    temporary = destination.with_name(destination.name + ".quantizing")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run([str(quantizer_path), str(source), str(temporary), quant_type], check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"llama-quantize did not produce {temporary}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
