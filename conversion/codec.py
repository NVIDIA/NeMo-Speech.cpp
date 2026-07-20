#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert NVIDIA NeMo NanoCodec decoder checkpoints to GGUF.

This converter targets:
https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps

The generated GGUF contains the token-to-audio path used by MagpieTTS:
the deterministic Group FSQ quantizer metadata/codebooks and the causal
HiFi-GAN decoder. The audio encoder, discriminators, and loss modules from
the NeMo archive are intentionally omitted.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

from .source import extract_archive, find_checkpoint_files, load_state_dict, read_checkpoint_config

MODEL_URL = "https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
WN_ORIGINAL0 = ".parametrizations.weight.original0"
WN_ORIGINAL1 = ".parametrizations.weight.original1"
GGML_MAX_NAME = 63


def extract_nemo(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None

    tmp = tempfile.TemporaryDirectory(prefix="nano-codec-nemo-")
    out = Path(tmp.name)
    extract_archive(path, out, basenames={"model_config.yaml", "model_weights.ckpt"})

    return out, tmp


def read_config(root: Path) -> dict[str, Any]:
    return read_checkpoint_config(root)


def read_state_dict(root: Path) -> dict[str, torch.Tensor]:
    checkpoint = find_checkpoint_files(root)["weights"]
    if checkpoint is None:
        raise RuntimeError(f"checkpoint contains no model_weights.ckpt: {root}")
    return load_state_dict(checkpoint, weights_only=True)


def cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def add_i32(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    writer.add_int32(key, int(value))


def add_f32(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    writer.add_float32(key, float(value))


def fsq_params(num_levels: list[int]) -> dict[str, list[int]]:
    dim_base_index = np.cumprod([1] + num_levels[:-1], dtype=np.int32).tolist()
    scale = [level // 2 for level in num_levels]
    return {
        "num_levels": [int(x) for x in num_levels],
        "dim_base_index": [int(x) for x in dim_base_index],
        "scale": [int(x) for x in scale],
        "offset": [int(x) for x in scale],
    }


def make_fsq_codebook(num_groups: int, num_levels: list[int]) -> torch.Tensor:
    params = fsq_params(num_levels)
    codebook_size = int(np.prod(num_levels))
    indices = np.arange(codebook_size, dtype=np.int32)[:, None]
    base = np.asarray(params["dim_base_index"], dtype=np.int32)[None, :]
    levels = np.asarray(num_levels, dtype=np.int32)[None, :]
    offset = np.asarray(params["offset"], dtype=np.float32)[None, :]
    scale = np.asarray(params["scale"], dtype=np.float32)[None, :]

    nonnegative = (indices // base) % levels
    codes = (nonnegative.astype(np.float32) - offset) / scale
    grouped = np.repeat(codes[None, :, :], num_groups, axis=0)
    return torch.from_numpy(grouped.astype(np.float32))


def add_metadata(writer: gguf.GGUFWriter, cfg: dict[str, Any]) -> dict[str, Any]:
    decoder = cfg["audio_decoder"]
    quantizer = cfg["vector_quantizer"]

    sample_rate = int(cfg["sample_rate"])
    samples_per_frame = int(cfg["samples_per_frame"])
    up_rates = [int(x) for x in decoder["up_sample_rates"]]
    num_groups = int(quantizer["num_groups"])
    num_levels = [int(x) for x in quantizer["num_levels_per_group"]]
    params = fsq_params(num_levels)
    codebook_size = int(np.prod(num_levels))
    codebook_dim_per_group = len(num_levels)
    latent_dim = num_groups * codebook_dim_per_group

    summary: dict[str, Any] = {
        "architecture": "nemo-nano-codec",
        "source_url": MODEL_URL,
        "nemo_target": cfg.get("target"),
        "nemo_version": cfg.get("nemo_version"),
        "sample_rate": sample_rate,
        "samples_per_frame": samples_per_frame,
        "frame_rate": sample_rate / samples_per_frame,
        "num_codebooks": num_groups,
        "codebook_size": codebook_size,
        "codebook_dim_per_group": codebook_dim_per_group,
        "latent_dim": latent_dim,
        "num_levels_per_group": num_levels,
        "dim_base_index": params["dim_base_index"],
        "decoder_up_sample_rates": up_rates,
        "decoder_base_channels": int(decoder["base_channels"]),
        "decoder_input_dim": int(decoder["input_dim"]),
        "decoder_activation": str(decoder.get("activation", "")),
        "decoder_output_activation": str(decoder.get("output_activation", "")),
        "decoder_pad_mode": str(decoder.get("pad_mode", "")),
        "tensor_layout": "ggml.conv1d/conv_transpose1d short names",
    }

    writer.add_name("NVIDIA NeMo NanoCodec 22kHz 1.89kbps decoder")
    writer.add_description("NeMo NanoCodec token-to-audio decoder converted to GGUF for GGML")
    writer.add_source_url(MODEL_URL)
    writer.add_tensor_data_layout("ggml.conv1d/conv_transpose1d short names")
    writer.add_string("nano_codec.nemo_target", str(cfg.get("target", "")))
    writer.add_string("nano_codec.nemo_version", str(cfg.get("nemo_version", "")))
    writer.add_string("nano_codec.config_json", json.dumps(cfg, ensure_ascii=False, sort_keys=True))
    writer.add_string("nano_codec.decoder.target", str(decoder.get("_target_", "")))
    writer.add_string("nano_codec.decoder.activation", str(decoder.get("activation", "")))
    writer.add_string(
        "nano_codec.decoder.output_activation", str(decoder.get("output_activation", ""))
    )
    writer.add_string("nano_codec.decoder.pad_mode", str(decoder.get("pad_mode", "")))
    writer.add_string("nano_codec.quantizer.target", str(quantizer.get("_target_", "")))

    add_i32(writer, "nano_codec.sample_rate", sample_rate)
    add_i32(writer, "nano_codec.samples_per_frame", samples_per_frame)
    add_f32(writer, "nano_codec.frame_rate", sample_rate / samples_per_frame)
    add_i32(writer, "nano_codec.num_codebooks", num_groups)
    add_i32(writer, "nano_codec.codebook_size", codebook_size)
    add_i32(writer, "nano_codec.codebook_dim_per_group", codebook_dim_per_group)
    add_i32(writer, "nano_codec.latent_dim", latent_dim)
    add_i32(writer, "nano_codec.decoder.input_dim", decoder["input_dim"])
    add_i32(writer, "nano_codec.decoder.base_channels", decoder["base_channels"])
    add_i32(writer, "nano_codec.decoder.in_kernel_size", decoder.get("in_kernel_size", 7))
    add_i32(writer, "nano_codec.decoder.out_kernel_size", decoder.get("out_kernel_size", 3))
    writer.add_bool(
        "nano_codec.decoder.n_groups_equal_to_out_channels",
        bool(decoder.get("n_groups_equal_to_out_channels", True)),
    )
    writer.add_array("nano_codec.quantizer.num_levels_per_group", params["num_levels"])
    writer.add_array("nano_codec.quantizer.dim_base_index", params["dim_base_index"])
    writer.add_array("nano_codec.quantizer.scale", params["scale"])
    writer.add_array("nano_codec.quantizer.offset", params["offset"])
    writer.add_array("nano_codec.decoder.up_sample_rates", up_rates)
    writer.add_array(
        "nano_codec.decoder.resblock_kernel_sizes", decoder.get("resblock_kernel_sizes", [3, 7, 11])
    )
    writer.add_array(
        "nano_codec.decoder.resblock_dilation_sizes",
        decoder.get("resblock_dilation_sizes", [1, 3, 5]),
    )

    return summary


def materialize_weight_norm(sd: dict[str, torch.Tensor], stem: str) -> torch.Tensor:
    g = sd[stem + WN_ORIGINAL0].detach().cpu().float()
    v = sd[stem + WN_ORIGINAL1].detach().cpu().float()
    dims = tuple(i for i in range(v.ndim) if i != 0)
    norm = v.norm(2, dim=dims, keepdim=True).clamp_min(1.0e-12)
    return (g * v / norm).contiguous()


def short_tensor_name(name: str) -> str:
    replacements = {
        "audio_decoder.pre_conv.conv.weight": "dec.pre.w",
        "audio_decoder.pre_conv.conv.bias": "dec.pre.b",
        "audio_decoder.post_conv.conv.weight": "dec.post.w",
        "audio_decoder.post_conv.conv.bias": "dec.post.b",
        "audio_decoder.post_activation.activation.snake_act.alpha": "dec.post_act.alpha",
        "audio_decoder.post_activation.activation.snake_act.alpha_inv": "dec.post_act.alpha_inv",
        "quantizer.fsq.codebook": "q.fsq.codebook",
    }
    if name in replacements:
        return replacements[name]

    match = re.fullmatch(
        r"audio_decoder\.activations\.(\d+)\.activation\.snake_act\.(alpha(?:_inv)?)", name
    )
    if match:
        return f"dec.act.{match.group(1)}.{match.group(2)}"

    match = re.fullmatch(r"audio_decoder\.up_sample_conv_layers\.(\d+)\.conv\.(weight|bias)", name)
    if match:
        suffix = "w" if match.group(2) == "weight" else "b"
        return f"dec.up.{match.group(1)}.{suffix}"

    match = re.fullmatch(
        r"audio_decoder\.res_layers\.(\d+)\.res_blocks\.(\d+)\.res_blocks\.(\d+)\."
        r"(input_activation|skip_activation)\.activation\.snake_act\.(alpha(?:_inv)?)",
        name,
    )
    if match:
        act = "ia" if match.group(4) == "input_activation" else "sa"
        return f"dec.res.{match.group(1)}.{match.group(2)}.{match.group(3)}.{act}.{match.group(5)}"

    match = re.fullmatch(
        r"audio_decoder\.res_layers\.(\d+)\.res_blocks\.(\d+)\.res_blocks\.(\d+)\."
        r"(input_conv|skip_conv)\.conv\.(weight|bias)",
        name,
    )
    if match:
        conv = "ic" if match.group(4) == "input_conv" else "sc"
        suffix = "w" if match.group(5) == "weight" else "b"
        return f"dec.res.{match.group(1)}.{match.group(2)}.{match.group(3)}.{conv}.{suffix}"

    raise ValueError(f"no short tensor name mapping for {name}")


def conv_transpose_to_dense_ggml(weight: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Expand grouped ConvTranspose1d weights to GGML's dense [Cin, Cout, K] logical layout."""

    in_channels, out_per_group, kernel = [int(x) for x in weight.shape]
    if out_channels % out_per_group != 0:
        raise ValueError(
            f"invalid ConvTranspose1d shape {tuple(weight.shape)} for out_channels={out_channels}"
        )
    groups = out_channels // out_per_group
    if in_channels % groups != 0:
        raise ValueError(
            f"invalid grouped ConvTranspose1d shape {tuple(weight.shape)} with groups={groups}"
        )
    in_per_group = in_channels // groups

    dense = torch.zeros((in_channels, out_channels, kernel), dtype=weight.dtype)
    for group in range(groups):
        for icg in range(in_per_group):
            in_idx = group * in_per_group + icg
            for ocg in range(out_per_group):
                out_idx = group * out_per_group + ocg
                dense[in_idx, out_idx, :] = weight[in_idx, ocg, :]
    return dense.contiguous()


def is_f32_tensor(name: str, tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point():
        return False
    return (
        tensor.ndim <= 1
        or name.endswith(".bias")
        or name.endswith(".b")
        or name.endswith(".alpha")
        or name.endswith(".alpha_inv")
        or name.startswith("quantizer.")
        or name.startswith("q.")
    )


def tensor_to_numpy(name: str, tensor: torch.Tensor, outtype: str) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_floating_point():
        if outtype == "f16" and not is_f32_tensor(name, tensor):
            tensor = tensor.to(torch.float16)
        else:
            tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.int64:
        tensor = tensor.to(torch.int32)
    return tensor.numpy()


def add_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor, outtype: str) -> None:
    short_name = short_tensor_name(name)
    if len(short_name) > GGML_MAX_NAME:
        raise ValueError(f"short tensor name is still too long: {short_name}")
    writer.add_tensor(short_name, tensor_to_numpy(short_name, tensor, outtype))


def add_decoder_tensors(
    writer: gguf.GGUFWriter,
    cfg: dict[str, Any],
    sd: dict[str, torch.Tensor],
    outtype: str,
) -> tuple[int, list[str]]:
    n_written = 0
    skipped: list[str] = []
    decoder_prefix = "audio_decoder."
    upsample_prefix = "audio_decoder.up_sample_conv_layers."

    for name in sd:
        if not name.startswith(decoder_prefix):
            continue

        if name.endswith(WN_ORIGINAL0):
            skipped.append(name)
            continue

        if name.endswith(WN_ORIGINAL1):
            stem = name[: -len(WN_ORIGINAL1)]
            out_name = stem + ".weight"
            tensor = materialize_weight_norm(sd, stem)
            if stem.startswith(upsample_prefix):
                bias_name = stem + ".bias"
                if bias_name not in sd:
                    raise KeyError(f"missing bias for {stem}")
                tensor = conv_transpose_to_dense_ggml(tensor, int(sd[bias_name].numel()))
            add_tensor(writer, out_name, tensor, outtype)
            n_written += 1
            continue

        if name.endswith(".bias"):
            tensor = sd[name].detach().cpu().float().reshape(1, -1, 1).contiguous()
            add_tensor(writer, name, tensor, outtype)
            n_written += 1
            continue

        if name.endswith(".alpha"):
            tensor = sd[name].detach().cpu().float().contiguous()
            add_tensor(writer, name, tensor, outtype)
            n_written += 1
            add_tensor(
                writer, name[: -len(".alpha")] + ".alpha_inv", 1.0 / (tensor + 1.0e-9), outtype
            )
            n_written += 1
            continue

        skipped.append(name)

    num_groups = int(cfg_get(cfg, "vector_quantizer.num_groups"))
    num_levels = [int(x) for x in cfg_get(cfg, "vector_quantizer.num_levels_per_group")]
    add_tensor(writer, "quantizer.fsq.codebook", make_fsq_codebook(num_groups, num_levels), outtype)
    n_written += 1

    return n_written, skipped


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("target") != "nemo.collections.tts.models.audio_codec.AudioCodecModel":
        raise RuntimeError(f"unsupported target: {cfg.get('target')}")
    if (
        cfg_get(cfg, "audio_decoder._target_")
        != "nemo.collections.tts.modules.audio_codec_modules.CausalHiFiGANDecoder"
    ):
        raise RuntimeError(f"unsupported audio decoder: {cfg_get(cfg, 'audio_decoder._target_')}")
    if (
        cfg_get(cfg, "vector_quantizer._target_")
        != "nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer"
    ):
        raise RuntimeError(f"unsupported quantizer: {cfg_get(cfg, 'vector_quantizer._target_')}")
    if cfg_get(cfg, "audio_decoder.activation") != "half_snake":
        raise RuntimeError("this converter currently expects half_snake decoder activations")
    if cfg_get(cfg, "audio_decoder.output_activation") != "clamp":
        raise RuntimeError("this converter currently expects clamp decoder output activation")
    if cfg_get(cfg, "audio_decoder.n_groups_equal_to_out_channels", True) is not True:
        raise RuntimeError("this converter currently expects grouped upsample convolutions")


def convert(
    source: Path,
    output: Path,
    outtype: str = "f16",
    metadata_json: Path | None = None,
) -> None:
    root, tmp = extract_nemo(source)

    try:
        cfg = read_config(root)
        sd = read_state_dict(root)
        validate_config(cfg)

        output.parent.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(output, "nemo-nano-codec")
        summary = add_metadata(writer, cfg)
        n_written, skipped = add_decoder_tensors(writer, cfg, sd, outtype)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        summary["tensors_written"] = n_written
        summary["tensors_skipped"] = skipped
        summary["output"] = str(output)
        summary["outtype"] = outtype

        if metadata_json:
            metadata_json.parent.mkdir(parents=True, exist_ok=True)
            metadata_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        print(f"wrote {output}")
        print(f"stored {n_written} decoder/FSQ tensors; skipped {len(skipped)} source tensors")
    finally:
        if tmp is not None:
            tmp.cleanup()
