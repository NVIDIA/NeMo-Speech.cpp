#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the EarTTS RVQ-VAE codec safetensors to a single GGUF file.

The NeMo codec checkpoint (``codec.safetensors``) has three weight groups:

    encoder.*  -- Wav2Latent (spec proj -> 3 stages x (3 ConvNeXt + downsample) -> bottleneck)
    decoder.*  -- Latent2Wav (mirror of encoder with ConvTranspose1d upsamples)
    prvq.*     -- 31-codebook residual VQ (mus + per-codebook EMA variance)

This script extracts every tensor, rewrites the PyTorch state-dict keys into
a compact ggml-friendly naming scheme, and writes a GGUF with architecture
metadata that our codec graph builder needs at load time (n_fft, hop_length,
latent_size, channel_mult, rates, etc).

The architecture parameters are inferred from tensor shapes (which are what's
actually in this specific checkpoint), so the GGUF is self-describing -- the
graph builder reads them back and constructs the cgraph from those values.

Tensor name remapping (PyTorch -> GGUF):
    encoder.layers.0.weight                       -> enc.proj_in.weight
    encoder.layers.{1,2,3}.dwconv.{weight,bias}   -> enc.blk{0..8}.dw.{weight,bias}
    encoder.layers.{1,2,3}.norm.{weight,bias}     -> enc.blk{0..8}.norm.{weight,bias}
    encoder.layers.{1,2,3}.pwconv1.{weight,bias}  -> enc.blk{0..8}.pw1.{weight,bias}
    encoder.layers.{1,2,3}.pwconv2.{weight,bias}  -> enc.blk{0..8}.pw2.{weight,bias}
    encoder.layers.{4,8}.weight                   -> enc.ds.{0,1}.weight  (downsamples)
    encoder.layers.12.weight                      -> enc.bottleneck.weight
    decoder.layers.0.weight                       -> dec.bottleneck.weight  (ConvTranspose1d)
    decoder.layers.{1..11}.*                      -> dec.blk{0..8}.* / dec.us.{0,1,2}.weight
    decoder.layers.12.weight                      -> dec.proj_out.weight
    prvq.mus_list.{i}                             -> rvq.cb{i}.mus
    prvq._variance_list.{i}.variance              -> rvq.cb{i}.variance

Usage:
    python convert_codec_to_gguf.py \\
        --codec /data/models/nemotron-voicechat/1/codec.safetensors \\
        --out   /data/models/nemotron-voicechat/1/codec.gguf
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open

logger = logging.getLogger("convert_codec_to_gguf")


def load_codec_state(path: Path) -> Dict[str, torch.Tensor]:
    """Load only codec tensors from either split or public checkpoints."""
    source = path / "model.safetensors" if path.is_dir() else path
    if not source.is_file():
        raise FileNotFoundError(source)
    with safe_open(str(source), framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        prefix = (
            "tts_model.audio_codec."
            if any(key.startswith("tts_model.audio_codec.") for key in keys)
            else ""
        )
        selected = [key for key in keys if not prefix or key.startswith(prefix)]
        state = {key[len(prefix) :]: tensors.get_tensor(key) for key in selected}
    if not state or "encoder.layers.0.weight" not in state:
        raise RuntimeError(f"no VoiceChat codec tensors found in {source}")
    return state


# ---------------------------------------------------------------------------
# Tensor topology helpers
# ---------------------------------------------------------------------------

# Per-stage layer-index layout the NeMo codec produces (see ear_tts_vae_codec.py
# Wav2Latent.__init__). Three stages of (3 ConvNeXt + 1 downsample), then a
# final bottleneck projection. Encoder: 3*4 = 12 + bottleneck = 13 layers.
# Decoder: bottleneck + 3*(upsample + 3 ConvNeXt) = 1 + 12 = 13 layers, with
# the upsample placed BEFORE its ConvNeXt blocks (the mirror order).

# encoder.layers.{idx}.* mapping:
#   0     -> proj_in (Conv1d 18->384, kernel 1)
#   1,2,3 -> ConvNeXt stage 0 blocks 0,1,2  (channels 384)
#   4     -> downsample 0 (Conv1d 384->768, kernel/stride 7)
#   5,6,7 -> ConvNeXt stage 1 blocks 0,1,2  (channels 768)
#   8     -> downsample 1 (Conv1d 768->1536, kernel/stride 7)
#   9,10,11 -> ConvNeXt stage 2 blocks 0,1,2 (channels 1536)
#   12    -> bottleneck (Conv1d 1536->512,  kernel/stride 9)
ENC_LAYOUT = {
    0: ("proj_in",),
    1: ("blk", 0),
    2: ("blk", 1),
    3: ("blk", 2),
    4: ("ds", 0),
    5: ("blk", 3),
    6: ("blk", 4),
    7: ("blk", 5),
    8: ("ds", 1),
    9: ("blk", 6),
    10: ("blk", 7),
    11: ("blk", 8),
    12: ("bottleneck",),
}

# decoder.layers.{idx}.* mapping:
#   0       -> bottleneck (ConvTranspose1d 512->1536, kernel/stride 9)
#   1,2,3   -> ConvNeXt stage 0 blocks 0,1,2  (channels 1536)
#   4       -> upsample 0 (ConvTranspose1d 1536->768, kernel/stride 7)
#   5,6,7   -> ConvNeXt stage 1 blocks 0,1,2  (channels 768)
#   8       -> upsample 1 (ConvTranspose1d 768->384, kernel/stride 7)
#   9,10,11 -> ConvNeXt stage 2 blocks 0,1,2  (channels 384)
#   12      -> proj_out (Conv1d 384->18, kernel 1)
DEC_LAYOUT = {
    0: ("bottleneck",),
    1: ("blk", 0),
    2: ("blk", 1),
    3: ("blk", 2),
    4: ("us", 0),
    5: ("blk", 3),
    6: ("blk", 4),
    7: ("blk", 5),
    8: ("us", 1),
    9: ("blk", 6),
    10: ("blk", 7),
    11: ("blk", 8),
    12: ("proj_out",),
}


# ---------------------------------------------------------------------------
# Remap a PyTorch tensor key to a GGUF tensor name.
# ---------------------------------------------------------------------------


def _remap_block(prefix: str, blk_idx: int, sub: str) -> str:
    """e.g. 'encoder.layers.1.dwconv.weight' (sub='dwconv.weight')
    becomes ``enc.blk0.dw.weight``."""
    field, _, suffix = sub.partition(".")  # 'dwconv', 'weight'
    short = {
        "dwconv": "dw",
        "norm": "norm",
        "pwconv1": "pw1",
        "pwconv2": "pw2",
    }[field]
    return f"{prefix}.blk{blk_idx}.{short}.{suffix}"


def remap_key(pt_key: str) -> str:
    if pt_key.startswith("encoder.layers."):
        rest = pt_key[len("encoder.layers.") :]
        idx_s, _, sub = rest.partition(".")
        idx = int(idx_s)
        spec = ENC_LAYOUT[idx]
        if spec[0] == "blk":
            return _remap_block("enc", spec[1], sub)
        # Non-block layers carry a single ``weight`` (sometimes ``bias`` too).
        return f"enc.{spec[0]}.{sub}" if spec[0] == "ds" else f"enc.{spec[0]}.{sub}"

    if pt_key.startswith("decoder.layers."):
        rest = pt_key[len("decoder.layers.") :]
        idx_s, _, sub = rest.partition(".")
        idx = int(idx_s)
        spec = DEC_LAYOUT[idx]
        if spec[0] == "blk":
            return _remap_block("dec", spec[1], sub)
        return f"dec.{spec[0]}.{sub}" if spec[0] == "us" else f"dec.{spec[0]}.{sub}"

    if pt_key.startswith("prvq.mus_list."):
        idx_s, _, _ = pt_key[len("prvq.mus_list.") :].partition(".")
        return f"rvq.cb{int(idx_s)}.mus"

    if pt_key.startswith("prvq._variance_list."):
        rest = pt_key[len("prvq._variance_list.") :]
        idx_s, _, sub = rest.partition(".")  # 'variance'
        return f"rvq.cb{int(idx_s)}.variance"

    raise KeyError(f"unrecognized codec tensor key: {pt_key}")


# Multi-element layer indices: encoder.ds.{0,1} need disambiguation back from
# our names (enc.ds.0 vs enc.ds.1). The remap_key above leaves "ds" with no
# index baked in -- fix that for downsample/upsample/bottleneck names that
# come from a single PyTorch layer.
def remap_key_v2(pt_key: str) -> str:
    if pt_key.startswith("encoder.layers."):
        rest = pt_key[len("encoder.layers.") :]
        idx_s, _, sub = rest.partition(".")
        idx = int(idx_s)
        spec = ENC_LAYOUT[idx]
        if spec[0] == "blk":
            return _remap_block("enc", spec[1], sub)
        if spec[0] == "ds":
            return f"enc.ds{spec[1]}.{sub}"
        return f"enc.{spec[0]}.{sub}"

    if pt_key.startswith("decoder.layers."):
        rest = pt_key[len("decoder.layers.") :]
        idx_s, _, sub = rest.partition(".")
        idx = int(idx_s)
        spec = DEC_LAYOUT[idx]
        if spec[0] == "blk":
            return _remap_block("dec", spec[1], sub)
        if spec[0] == "us":
            return f"dec.us{spec[1]}.{sub}"
        return f"dec.{spec[0]}.{sub}"

    if pt_key.startswith("prvq.mus_list."):
        idx_s, _, _ = pt_key[len("prvq.mus_list.") :].partition(".")
        return f"rvq.cb{int(idx_s)}.mus"

    if pt_key.startswith("prvq._variance_list."):
        rest = pt_key[len("prvq._variance_list.") :]
        idx_s, _, _ = rest.partition(".")
        return f"rvq.cb{int(idx_s)}.variance"

    raise KeyError(f"unrecognized codec tensor key: {pt_key}")


# ---------------------------------------------------------------------------
# Architecture inference from tensor shapes
# ---------------------------------------------------------------------------


def infer_arch(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Read off architecture hyperparams from the actual tensor shapes.

    This keeps the GGUF self-describing -- the graph builder doesn't need a
    sibling config.json to know stride values / channel counts.
    """
    # Input spec channels (from proj_in weight: [out, in, kernel])
    proj_in = state["encoder.layers.0.weight"]
    spec_ch = int(proj_in.shape[1])
    base_hidden = int(proj_in.shape[0])  # 384

    # Channels at each stage (downsample output channels)
    ds0 = state["encoder.layers.4.weight"]
    ds1 = state["encoder.layers.8.weight"]
    bottleneck = state["encoder.layers.12.weight"]
    stage_channels = [base_hidden, int(ds0.shape[0]), int(ds1.shape[0])]
    latent = int(bottleneck.shape[0])

    # Downsample rates ARE the kernel sizes (Wav2Latent passes kernel=rate, stride=rate).
    rates = [int(ds0.shape[2]), int(ds1.shape[2]), int(bottleneck.shape[2])]

    # ConvNeXt depthwise kernel (encoder.layers.1.dwconv.weight shape [c, 1, k])
    cn_kernel = int(state["encoder.layers.1.dwconv.weight"].shape[2])

    # ConvNeXt intermediate ratio (4 by convention)
    pw1_out = int(state["encoder.layers.1.pwconv1.weight"].shape[0])
    intermediate_ratio = pw1_out // base_hidden

    # RVQ
    cb_shape = state["prvq.mus_list.0"].shape
    codebook_size = int(cb_shape[0])
    rvq_dim = int(cb_shape[1])
    num_quantizers = sum(1 for k in state if k.startswith("prvq.mus_list."))

    assert rvq_dim == latent, f"RVQ dim {rvq_dim} != latent {latent}"

    # STFT params: spec_ch == 2 * (n_fft//2 + 1)  =>  n_fft = spec_ch - 2
    n_fft = spec_ch - 2

    arch = {
        "spec_channels": spec_ch,
        "n_fft": n_fft,
        "base_hidden_size": base_hidden,
        "stage_channels": stage_channels,
        "latent_size": latent,
        "rates": rates,
        "conv_next_kernel": cn_kernel,
        "intermediate_ratio": intermediate_ratio,
        "num_blocks_per_stage": 3,
        "num_quantizers": num_quantizers,
        "codebook_size": codebook_size,
    }

    # hop_length isn't recoverable from weights alone; it's an inference-time
    # STFT param the runtime can override. We bake the inferred default below.
    return arch


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------


def write_gguf(
    out_path: Path,
    tensors: Dict[str, torch.Tensor],
    arch: Dict[str, Any],
    hop_length: int,
    sample_rate: int,
    weight_type: str = "f32",
) -> None:
    """Use llama.cpp's gguf-py to emit the file."""
    # Locate the llama.cpp checkout for gguf-py
    repo = Path(__file__).resolve().parents[2]
    gguf_dir = repo / "llama.cpp" / "gguf-py"
    if gguf_dir.is_dir() and str(gguf_dir) not in sys.path:
        sys.path.insert(0, str(gguf_dir))
    import gguf

    arch_name = "eartts_codec"
    writer = gguf.GGUFWriter(str(out_path), arch=arch_name)

    # ---- metadata ----
    writer.add_string("general.architecture", arch_name)
    writer.add_string("general.name", "EarTTS RVQ-VAE codec")
    writer.add_uint32(f"{arch_name}.spec_channels", arch["spec_channels"])
    writer.add_uint32(f"{arch_name}.n_fft", arch["n_fft"])
    writer.add_uint32(f"{arch_name}.hop_length", hop_length)
    writer.add_uint32(f"{arch_name}.sample_rate", sample_rate)
    writer.add_uint32(f"{arch_name}.base_hidden_size", arch["base_hidden_size"])
    writer.add_array(f"{arch_name}.stage_channels", arch["stage_channels"])
    writer.add_uint32(f"{arch_name}.latent_size", arch["latent_size"])
    writer.add_array(f"{arch_name}.rates", arch["rates"])
    writer.add_uint32(f"{arch_name}.conv_next_kernel", arch["conv_next_kernel"])
    writer.add_uint32(f"{arch_name}.intermediate_ratio", arch["intermediate_ratio"])
    writer.add_uint32(f"{arch_name}.num_blocks_per_stage", arch["num_blocks_per_stage"])
    writer.add_uint32(f"{arch_name}.num_quantizers", arch["num_quantizers"])
    writer.add_uint32(f"{arch_name}.codebook_size", arch["codebook_size"])

    # ---- tensors ----
    #
    # With f16 output, store matrix weights as f16 while retaining small
    # elementwise tensors and RVQ values at f32:
    #   - norm.weight / norm.bias  (LayerNormNd affine)
    #   - dw.bias / pw1.bias / pw2.bias / proj_*.bias / bottleneck.bias
    #   - rvq codebooks (precision matters for nearest-neighbor argmin)
    #   - rvq offsets / variance scalars
    # i.e. only 2-D+ weight tensors that participate in matmuls get f16.
    # ggml CUDA's conv_transpose_1d kernel asserts src0->type == GGML_TYPE_F32.
    # Three decoder layers hit that path (bottleneck + 2 upsamples); keep them
    # at f32 even when --weight-type=f16 is requested. Everything else (proj,
    # dwconv, pwconv1, pwconv2) goes through ggml_conv_1d → im2col + mul_mat,
    # which supports f16 weights and engages Tensor Cores via cuBLAS hgemm.
    CONV_TRANSPOSE_NAMES = {"dec.bottleneck.weight", "dec.us0.weight", "dec.us1.weight"}

    def _should_f16(name: str, arr) -> bool:
        if weight_type != "f16":
            return False
        if arr.ndim < 2:
            return False
        if name in CONV_TRANSPOSE_NAMES:
            return False
        n = name.lower()
        if "rvq" in n or "codebook" in n or "offset" in n or "variance" in n:
            return False
        if n.endswith(".bias"):
            return False
        if "norm" in n:
            return False
        return True

    for gname, t in sorted(tensors.items()):
        np_arr = t.detach().cpu().contiguous().float().numpy()
        if np_arr.ndim == 0:
            np_arr = np_arr.reshape(1)
        if _should_f16(gname, np_arr):
            np_arr = np_arr.astype("float16")
        writer.add_tensor(gname, np_arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--codec", required=True, help="VoiceChat checkpoint or codec.safetensors")
    ap.add_argument("--out", required=True, help="Destination GGUF path")
    ap.add_argument(
        "--hop-length",
        type=int,
        default=4,
        help="STFT hop_length used at inference (cannot be inferred from weights). Default: 4.",
    )
    ap.add_argument(
        "--sample-rate", type=int, default=22050, help="Target audio sample rate. Default: 22050."
    )
    ap.add_argument(
        "--weight-type",
        choices=["f32", "f16"],
        default="f32",
        help="Storage dtype for the conv stack. f16 enables Tensor Cores on cuBLAS "
        "(Ampere+) for the pwconv1/pwconv2 matmuls in ConvNeXt — ~3-5x decoder "
        "speedup. RVQ codebooks + norm weights + biases stay f32 regardless.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    src = Path(args.codec).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading codec safetensors: %s", src)
    state = load_codec_state(src)

    arch = infer_arch(state)
    logger.info("Inferred architecture: %s", json.dumps(arch, indent=2))
    logger.info(
        "hop_length=%d, sample_rate=%d (override via --hop-length/--sample-rate)",
        args.hop_length,
        args.sample_rate,
    )

    # Sanity: audio samples per code frame should be hop * prod(rates)
    samples_per_frame = args.hop_length * int(np.prod(arch["rates"]))
    frame_ms = samples_per_frame * 1000.0 / args.sample_rate
    logger.info(
        "Derived: %d audio samples per code frame == %.2f ms @ %d Hz",
        samples_per_frame,
        frame_ms,
        args.sample_rate,
    )

    # Remap all tensors.
    gguf_tensors: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for pt_key in sorted(state.keys()):
        try:
            gname = remap_key_v2(pt_key)
        except KeyError as e:
            skipped.append(pt_key)
            logger.warning("skipping unknown key: %s (%s)", pt_key, e)
            continue
        if gname in gguf_tensors:
            raise RuntimeError(f"name collision after remap: {gname} (from {pt_key})")
        gguf_tensors[gname] = state[pt_key]

    logger.info("Remapped %d tensors (%d skipped)", len(gguf_tensors), len(skipped))

    write_gguf(
        out, gguf_tensors, arch, args.hop_length, args.sample_rate, weight_type=args.weight_type
    )
    out_size_mb = out.stat().st_size / 1e6
    logger.info("Wrote GGUF: %s (%.1f MB, %d tensors)", out, out_size_mb, len(gguf_tensors))


if __name__ == "__main__":
    main()
