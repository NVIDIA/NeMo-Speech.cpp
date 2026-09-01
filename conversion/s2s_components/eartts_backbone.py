# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert the Gemma3 backbone of an EarTTS vLLM safetensors checkpoint into a
GGUF model loadable by stock llama.cpp.

The EarTTS checkpoint already laid out by ``convert_eartts_checkpoint.py``
contains three weight groups under prefix ``model.``:
  - ``model.backbone.*``   - Gemma3 decoder (28 layers).  Goes into the GGUF.
  - ``model.total_emb.*``  - input pipeline.  Stays in PyTorch (sidecar).
  - ``model.sampler.*``    - MaskGIT/MoG sampler.  Stays in PyTorch (sidecar).

This script extracts the backbone subset, strips the ``model.backbone.``
prefix, synthesizes a Gemma3-shaped HF directory (``config.json`` +
``model.safetensors``), and runs ``llama.cpp/convert_hf_to_gguf.py``'s
``Gemma3Model`` writer on it.

The HF directory we synthesize has no tokenizer files - the EarTTS backbone
is consumed through embedding passthrough (``batch.embd``) and never tokenizes
text.  We patch ``Gemma3Model.set_vocab`` to emit ``tokenizer.ggml.model="none"``
and ``Gemma3Model.modify_tensors`` to skip the vocab-driven embed_tokens
truncation step.

Usage:
  python convert_eartts_backbone_to_gguf.py \\
      --eartts-dir /path/to/models/.../eartts_vllm \\
      --out /path/to/output/model.gguf \\
      [--llama-cpp /path/to/llama.cpp] \\
      [--outtype bf16|f16|f32|auto]
"""

import argparse
import importlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from voicechat_source import build_eartts_config, is_hf_voicechat_source, quantize_backbone

logger = logging.getLogger("convert_eartts_backbone_to_gguf")

EARTTS_BACKBONE_PREFIX = "model.backbone."
# Keep ``general.name`` deterministic and compatible with the distributed
# artifact layout.
EARTTS_COMPAT_MODEL_NAME = "Eartts_Gemma3__3Noo8Ol"

# Keys forwarded verbatim from the EarTTS config to the synthesized Gemma3 config.
GEMMA3_FORWARDED_HPARAMS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "query_pre_attn_scalar",
    "max_position_embeddings",
    "rope_theta",
    "rope_local_base_freq",
    "sliding_window",
    "layer_types",
)


def _resolve_llama_cpp_root(arg: str | None) -> Path:
    """Locate the llama.cpp checkout (default: <repo-root>/llama.cpp)."""
    if arg is not None:
        root = Path(arg).resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "llama.cpp"
    if not (root / "convert_hf_to_gguf.py").is_file():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found under {root}; " "pass --llama-cpp /path/to/llama.cpp"
        )
    return root


def _import_converter(llama_cpp_root: Path):
    """Import llama.cpp/convert_hf_to_gguf.py as a module."""
    for entry in (str(llama_cpp_root), str(llama_cpp_root / "gguf-py")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    convert_module = importlib.import_module("convert_hf_to_gguf")
    # Recent llama.cpp versions keep model writers in conversion/*.py rather
    # than exporting them from convert_hf_to_gguf.py.  Attach the classes used
    # by this converter while retaining support for older monolithic versions.
    try:
        from conversion.base import TextModel
        from conversion.gemma import Gemma3Model

        convert_module.TextModel = TextModel
        convert_module.Gemma3Model = Gemma3Model
    except ImportError:
        pass
    return convert_module


def _load_eartts_config(eartts_dir: Path, tokenizer_json: Path | None) -> Dict[str, Any]:
    if is_hf_voicechat_source(eartts_dir):
        if tokenizer_json is None or not tokenizer_json.is_file():
            raise FileNotFoundError("the public VoiceChat checkpoint requires --tokenizer-json")
        return build_eartts_config(eartts_dir, tokenizer_json)
    cfg_path = eartts_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _extract_backbone_weights(eartts_dir: Path) -> Dict[str, torch.Tensor]:
    """Filter the EarTTS safetensors tree to backbone-only tensors.

    Supports both single-file and sharded checkpoints.  The returned dict has
    the ``model.backbone.`` prefix stripped and replaced with ``model.`` so it
    matches the HF Gemma3ForCausalLM tensor naming convention.
    """
    public_checkpoint = is_hf_voicechat_source(eartts_dir)
    source_prefix = "tts_model.tts_model.backbone." if public_checkpoint else EARTTS_BACKBONE_PREFIX
    index_path = eartts_dir / "model.safetensors.index.json"
    backbone: Dict[str, torch.Tensor] = {}

    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map: Dict[str, str] = index["weight_map"]
        per_shard: Dict[str, list[str]] = {}
        for k, shard in weight_map.items():
            if k.startswith(source_prefix):
                per_shard.setdefault(shard, []).append(k)
        for shard, keys in per_shard.items():
            shard_path = eartts_dir / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"Shard listed in index not found: {shard_path}")
            with safe_open(str(shard_path), framework="pt", device="cpu") as data:
                for k in keys:
                    backbone[k] = data.get_tensor(k)
    else:
        single = eartts_dir / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(f"Neither index nor single safetensors found in {eartts_dir}")
        with safe_open(str(single), framework="pt", device="cpu") as data:
            backbone = {k: data.get_tensor(k) for k in data.keys() if k.startswith(source_prefix)}

    if not backbone:
        raise RuntimeError(f"No tensors with prefix {source_prefix!r} in {eartts_dir}")

    stripped: Dict[str, torch.Tensor] = {}
    for k, v in backbone.items():
        new_key = "model." + k[len(source_prefix) :]
        stripped[new_key] = v
    if "model.embed_tokens.weight" not in stripped:
        hidden = int(next(iter(stripped.values())).shape[-1])
        stripped["model.embed_tokens.weight"] = torch.zeros((1, hidden), dtype=torch.float32)
    return stripped


def _build_hf_gemma3_dir(
    eartts_dir: Path, work_dir: Path, tokenizer_json: Path | None
) -> Dict[str, Any]:
    """Materialize a Gemma3-shaped HF directory inside ``work_dir``."""
    eartts_cfg = _load_eartts_config(eartts_dir, tokenizer_json)
    weights = _extract_backbone_weights(eartts_dir)

    save_file(weights, str(work_dir / "model.safetensors"))

    gemma_cfg: Dict[str, Any] = {
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "vocab_size": 1,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rms_norm_eps": eartts_cfg.get("rms_norm_eps", 1e-6),
    }
    for k in GEMMA3_FORWARDED_HPARAMS:
        if k in eartts_cfg:
            gemma_cfg[k] = eartts_cfg[k]

    missing = [
        k
        for k in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
        if k not in gemma_cfg
    ]
    if missing:
        raise KeyError(f"EarTTS config is missing required keys for Gemma3: {missing}")

    (work_dir / "config.json").write_text(json.dumps(gemma_cfg, indent=2))
    return gemma_cfg


def _patch_gemma3_for_eartts(convert_module) -> None:
    """Adapt Gemma3 conversion for embedding passthrough and EarTTS scaling.

    We replace ``set_vocab`` with the ``none`` writer, and rewrite
    ``modify_tensors`` to skip the embed_tokens slicing that depends on a
    tokenizer.json being present in the source dir.  RMSNorm shift (the
    ``+1`` baked at conversion time) is preserved. The checkpoint's
    pre-attention scalar is stored as the effective GGUF attention scale.
    """
    Gemma3Model = convert_module.Gemma3Model
    TextModel = convert_module.TextModel
    original_set_gguf_parameters = Gemma3Model.set_gguf_parameters

    def set_gguf_parameters(self):
        original_set_gguf_parameters(self)
        query_pre_attn_scalar = float(self.hparams.get("query_pre_attn_scalar", 256.0))
        if query_pre_attn_scalar <= 0.0:
            raise ValueError("query_pre_attn_scalar must be positive")
        self.gguf_writer.add_attention_scale(query_pre_attn_scalar**-0.5)

    def set_vocab(self):
        # Embedding-passthrough use: we never tokenize text. llama.cpp's Gemma3
        # loader checks ``token_embd.weight`` shape against ``<arch>.vocab_size``
        # (read at llama-vocab.cpp:1773 when tokenizer=="none"). Our backbone
        # has a dummy 1-row embed_tokens, so set vocab_size=1 to make the shape
        # check pass. The tensor itself is never read at inference because we
        # feed ``batch.embd`` directly.
        self.gguf_writer.add_tokenizer_model("none")
        self.gguf_writer.add_vocab_size(1)

    def modify_tensors(self, data_torch, name, bid):
        if "language_model." in name:
            name = name.replace("language_model.", "")
        elif (
            name.startswith("multi_modal_projector.")
            or name.startswith("vision_tower.")
            or name.startswith("multimodal_projector.")
            or name.startswith("vision_model.")
        ):
            return  # vision tensors don't exist in our backbone, but keep filter intact

        # Bake Gemma3RMSNorm "(1 + w)" shift into the GGUF.  Same logic as upstream.
        f_shift = self.norm_shift(name)
        if f_shift != 0.0:
            data_torch = data_torch + f_shift

        # NOTE: upstream's Gemma3Model.modify_tensors slices embed_tokens to
        # match the tokenizer vocab.  We have no tokenizer.json/tokenizer.model
        # and use embedding passthrough for inference, so the slice is a no-op
        # at best and a crash at worst -- skip it.
        yield from TextModel.modify_tensors(self, data_torch, name, bid)

    Gemma3Model.set_vocab = set_vocab
    Gemma3Model.set_gguf_parameters = set_gguf_parameters
    Gemma3Model.modify_tensors = modify_tensors


def _select_ftype(convert_module, outtype: str):
    LFT = convert_module.gguf.LlamaFileType
    return {
        "bf16": LFT.MOSTLY_BF16,
        "f16": LFT.MOSTLY_F16,
        "f32": LFT.ALL_F32,
        "auto": LFT.GUESSED,
    }[outtype]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--eartts-dir",
        required=True,
        help="Path to an EarTTS vllm-format dir (with model.safetensors[.index.json] + config.json).",
    )
    ap.add_argument(
        "--tokenizer-json",
        type=Path,
        help="Nano tokenizer.json used with the public VoiceChat checkpoint",
    )
    ap.add_argument(
        "--quantize",
        choices=["q4_k_m", "nvfp4"],
        help="quantize the converted BF16 backbone",
    )
    ap.add_argument("--quantizer", type=Path, help="path to llama-quantize")
    ap.add_argument("--out", required=True, help="Destination GGUF file path.")
    ap.add_argument(
        "--llama-cpp",
        default=None,
        help="Path to the llama.cpp checkout (default: <repo>/llama.cpp).",
    )
    ap.add_argument(
        "--outtype",
        default="bf16",
        choices=["bf16", "f16", "f32", "auto"],
        help="GGUF tensor encoding (default: bf16).",
    )
    ap.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Don't delete the temporary HF directory (for debugging).",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    eartts_dir = Path(args.eartts_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llama_cpp_root = _resolve_llama_cpp_root(args.llama_cpp)
    convert_module = _import_converter(llama_cpp_root)
    _patch_gemma3_for_eartts(convert_module)

    tmp_ctx = (
        tempfile.TemporaryDirectory(prefix="eartts_gemma3_")
        if not args.keep_workdir
        else _NoopCleanupTempDir(tempfile.mkdtemp(prefix="eartts_gemma3_"))
    )
    with tmp_ctx as tmp:
        work_dir = Path(tmp)
        cfg = _build_hf_gemma3_dir(eartts_dir, work_dir, args.tokenizer_json)
        logger.info(
            "Synthesized HF dir %s: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, sliding=%d",
            work_dir,
            cfg["num_hidden_layers"],
            cfg["hidden_size"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["head_dim"],
            cfg.get("sliding_window", -1),
        )

        Gemma3Model = convert_module.Gemma3Model
        ftype = _select_ftype(convert_module, args.outtype)

        converted_path = work_dir / "eartts-backbone.bf16.gguf" if args.quantize else out_path
        with torch.inference_mode():
            model = Gemma3Model(
                dir_model=work_dir,
                ftype=ftype,
                fname_out=converted_path,
                is_big_endian=False,
                use_temp_file=False,
                eager=False,
                metadata_override=None,
                model_name=EARTTS_COMPAT_MODEL_NAME,
                split_max_tensors=0,
                split_max_size=0,
                dry_run=False,
                small_first_shard=False,
                hparams=None,
            )
            model.write()

        if args.quantize:
            quantize_backbone(
                converted_path,
                out_path,
                args.quantize,
                llama_cpp_root,
                args.quantizer,
            )

        logger.info("Wrote GGUF: %s", out_path)
        if args.keep_workdir:
            logger.info("Kept temporary HF dir at %s", work_dir)


class _NoopCleanupTempDir:
    """Mimic TemporaryDirectory's context-manager API but skip cleanup."""

    def __init__(self, path: str):
        self._path = path

    def __enter__(self) -> str:
        return self._path

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    main()
