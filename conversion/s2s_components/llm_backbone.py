# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert the LLM backbone of a Nemotron Nano v2 voicechat checkpoint into a
GGUF model loadable by stock llama.cpp.

The voicechat checkpoint contains, under prefix ``stt_model.``:
  - ``stt_model.llm.*``           - NemotronH backbone (LLM). Goes into the GGUF.
  - ``stt_model.embed_tokens.*``  - shared text/function input embedding. Into the GGUF.
  - ``stt_model.lm_head.*``       - output projection. Into the GGUF.
  - ``stt_model.function_head.*`` - tool-calling head (NEW arch). Stays in PyTorch
                                    (extracted separately by ``extract_function_head.py``).
  - ``stt_model.asr_head.*``      - ASR head (OLD arch). Stays in PyTorch
                                    (extracted separately by ``extract_asr_head.py``).
  - ``stt_model.embed_asr_tokens.*`` - OLD-arch only. Skipped.

llama.cpp's upstream ``convert_hf_to_gguf.py`` already understands the
``stt_model.*`` prefix and ships the OLD-arch skip list
(``asr_head`` + ``embed_asr_tokens``). For the NEW-arch checkpoint we only need
to swap that skip-list with ``function_head``. We monkey-patch
``NemotronHModel.modify_tensors`` to do exactly that, then invoke the upstream
writer -- no fork of the converter.

Usage:
  python convert_nemotronh_voicechat_to_gguf.py \\
      --ckpt-dir /path/to/nemotron-voicechat/<version>/.../1 \\
      --out /path/to/output/model.gguf \\
      [--llama-cpp /path/to/llama.cpp] \\
      [--arch new|old] \\
      [--outtype bf16|f16|f32|auto]
"""

import argparse
import importlib
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import torch

if __package__:
    from .voicechat_source import is_hf_voicechat_source, quantize_backbone
else:
    from voicechat_source import is_hf_voicechat_source, quantize_backbone

logger = logging.getLogger("convert_nemotronh_voicechat_to_gguf")


# Mapping from --arch to the tensor-name prefixes that should be excluded from
# the GGUF. They live in the safetensors but are consumed by PyTorch side
# networks (function_head / asr_head) and have no llama.cpp tensor mapping.
_SKIP_PREFIXES = {
    "new": ("stt_model.function_head.",),
    "old": ("stt_model.asr_head.", "stt_model.embed_asr_tokens."),
}

_PUBLIC_SKIP_PREFIXES = (
    "stt_model.perception.",
    "stt_model.rnnt_",
    "tts_model.",
)


def _resolve_llama_cpp_root(arg: str | None) -> Path:
    """Locate the llama.cpp checkout (default: <repo-root>/llama.cpp)."""
    if arg is not None:
        root = Path(arg).resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "llama.cpp"
    if not (root / "convert_hf_to_gguf.py").is_file():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found under {root}; pass --llama-cpp /path/to/llama.cpp"
        )
    return root


def _import_converter(llama_cpp_root: Path):
    """Import llama.cpp/convert_hf_to_gguf.py as a module."""
    for entry in (str(llama_cpp_root), str(llama_cpp_root / "gguf-py")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    convert_module = importlib.import_module("convert_hf_to_gguf")
    # llama.cpp split architecture writers out of convert_hf_to_gguf.py into
    # conversion/*.py.  Importing the architecture module both registers it
    # with ModelBase and gives the small skip-list patch below a stable class.
    try:
        from conversion.nemotron import NemotronHModel

        convert_module.NemotronHModel = NemotronHModel
    except ImportError:
        # Compatibility with older monolithic llama.cpp converters.
        pass
    return convert_module


def _patch_nemotronh_skip_list(convert_module, skip_prefixes: tuple[str, ...]) -> None:
    """Teach stock NemotronHModel about the VoiceChat wrapper layout.

    VoiceChat nests a normal NemotronH checkpoint below ``stt_model``.  Stock
    llama.cpp expects ``backbone.*``/``model.embed_tokens``/``lm_head`` names,
    so unwrap those prefixes and exclude the auxiliary head selected by
    ``skip_prefixes``.
    """
    NemotronHModel = convert_module.NemotronHModel
    original_modify = NemotronHModel.modify_tensors

    def patched_modify_tensors(self, data_torch, name, bid):
        if name.startswith(skip_prefixes):
            return iter(())
        if name.startswith("stt_model.llm."):
            name = "backbone." + name.removeprefix("stt_model.llm.")
        elif name.startswith("stt_model.embed_tokens."):
            name = "model.embed_tokens." + name.removeprefix("stt_model.embed_tokens.")
        elif name.startswith("stt_model.lm_head."):
            name = "lm_head." + name.removeprefix("stt_model.lm_head.")
        return original_modify(self, data_torch, name, bid)

    NemotronHModel.modify_tensors = patched_modify_tensors


def _patch_config_loading(convert_module) -> None:
    """Keep the checkpoint's dense/MoE classification authoritative.

    Recent Transformers releases synthesize Nemotron-H MoE defaults while
    loading a dense config.  llama.cpp then sees ``num_experts_per_tok`` and
    writes the checkpoint as NEMOTRON_H_MOE even though it contains only dense
    MLP tensors.  The historical converter consumed config.json directly, so
    retain those semantics for this voicechat conversion.
    """
    original_load_hparams = convert_module.ModelBase.load_hparams

    def load_hparams(dir_model: Path, is_mistral_format: bool):
        if is_mistral_format:
            return original_load_hparams(dir_model, is_mistral_format)
        with (Path(dir_model) / "config.json").open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    convert_module.ModelBase.load_hparams = staticmethod(load_hparams)


def _select_ftype(convert_module, outtype: str):
    LFT = convert_module.gguf.LlamaFileType
    return {
        "bf16": LFT.MOSTLY_BF16,
        "f16": LFT.MOSTLY_F16,
        "f32": LFT.ALL_F32,
        "auto": LFT.GUESSED,
    }[outtype]


def _resolve_arch_class(convert_module, ckpt_dir: Path):
    """Find the right Model subclass for the checkpoint's ``config.json``.

    Replicates ``ModelBase.from_model_architecture`` indirectly by reading the
    config file and looking up the registered class. We do this rather than
    hard-coding ``NemotronHModel`` so a future Nemotron-N or related arch
    automatically picks up its own writer.
    """
    import json

    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    config = json.loads(cfg_path.read_text())
    arches = config.get("architectures") or []
    if not arches:
        raise KeyError(f"{cfg_path} has no 'architectures' field")
    return convert_module.ModelBase.from_model_architecture(arches[0])


def _build_public_adapter(source: Path, base_model_dir: Path, work_dir: Path) -> Path:
    config = base_model_dir / "config.json"
    tokenizer = base_model_dir / "tokenizer.json"
    for required in (config, tokenizer, source / "model.safetensors"):
        if not required.is_file():
            raise FileNotFoundError(required)
    shutil.copy2(config, work_dir / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        candidate = base_model_dir / name
        if candidate.is_file():
            shutil.copy2(candidate, work_dir / name)
    (work_dir / "model.safetensors").symlink_to(source / "model.safetensors")
    return work_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt-dir",
        required=True,
        help="Path to the voicechat checkpoint dir (with config.json + model.safetensors[.index.json]).",
    )
    ap.add_argument(
        "--base-model-dir",
        type=Path,
        help="Nano base-model config and tokenizer directory",
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
        "--arch",
        default="new",
        choices=sorted(_SKIP_PREFIXES.keys()),
        help="'new' (function_head) or 'old' (asr_head + embed_asr_tokens). Default: new.",
    )
    ap.add_argument(
        "--outtype",
        default="bf16",
        choices=["bf16", "f16", "f32", "auto"],
        help="GGUF tensor encoding (default: bf16).",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ckpt_dir = Path(args.ckpt_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llama_cpp_root = _resolve_llama_cpp_root(args.llama_cpp)
    convert_module = _import_converter(llama_cpp_root)
    _patch_config_loading(convert_module)

    public_checkpoint = is_hf_voicechat_source(ckpt_dir)
    skip_prefixes = _SKIP_PREFIXES[args.arch] + (_PUBLIC_SKIP_PREFIXES if public_checkpoint else ())
    logger.info("Patching NemotronHModel to skip prefixes: %s", skip_prefixes)
    _patch_nemotronh_skip_list(convert_module, skip_prefixes)

    with tempfile.TemporaryDirectory(prefix="voicechat_nemotronh_") as temporary:
        work_dir = Path(temporary)
        model_dir = ckpt_dir
        if public_checkpoint:
            if args.base_model_dir is None:
                raise ValueError("the public VoiceChat checkpoint requires --base-model-dir")
            model_dir = _build_public_adapter(
                ckpt_dir, args.base_model_dir.expanduser().resolve(), work_dir
            )

        ModelCls = _resolve_arch_class(convert_module, model_dir)
        if not issubclass(ModelCls, convert_module.NemotronHModel):
            logger.warning(
                "Checkpoint arch resolves to %s, not NemotronHModel — the skip-list "
                "patch will not take effect on this writer.",
                ModelCls.__name__,
            )

        ftype = _select_ftype(convert_module, args.outtype)
        converted_path = work_dir / "llm-backbone.bf16.gguf" if args.quantize else out_path
        with torch.inference_mode():
            model = ModelCls(
                model_dir,
                ftype,
                converted_path,
                is_big_endian=False,
                use_temp_file=False,
                eager=False,
                metadata_override=None,
                model_name="NVIDIA-Nemotron-Nano-9B-v2" if public_checkpoint else None,
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


if __name__ == "__main__":
    main()
