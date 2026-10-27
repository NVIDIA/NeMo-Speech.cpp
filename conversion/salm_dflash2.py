# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 draft model of a speech-LLM (SALM) as a GGUF, for speculative decoding (asr.salm.dflash_path).

The drafter checkpoint (HF safetensors, architecture Qwen3DFlash2DraftModel) goes through the pinned llama.cpp
converter under DFlash2DraftModel, the name its DFlash support registers. A drafter reads its target's token
embeddings and hidden states, so the converter also takes the target's tokenizer and dimensions: they come from the
speech-LLM checkpoint it was trained for, staged as for its decoder (conversion/salm.py) without the weights.

  python convert_model.py <dflash2 checkpoint> --architecture dflash2 --salm-target <speech-LLM checkpoint> \\
      --outfile <checkpoint>-DFlash2-BF16.gguf
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

DRAFTER_ARCHITECTURES = ("Qwen3DFlash2DraftModel", "DFlash2DraftModel")
OUTTYPES = (
    "bf16",
    "f16",
    "f32",
)  # the drafter runs next to the decoder; its weights stay unquantized


def is_dflash2_config(config: dict) -> bool:
    """A DFlash2 drafter's HF config."""
    return any(name in DRAFTER_ARCHITECTURES for name in config.get("architectures") or [])


def is_dflash2_checkpoint(path: Path) -> bool:
    """A local directory holding a DFlash2 drafter (config.json next to safetensors)."""
    config = path / "config.json"
    if not path.is_dir() or not config.is_file():
        return False
    try:
        return is_dflash2_config(json.loads(config.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return False


def convert(
    source: Path,
    target: Path,
    outfile: Path,
    outtype: str = "bf16",
    *,
    llama_cpp: Path,
    name: str | None = None,
    dry_run: bool = False,
) -> None:
    """Write the drafter in `source` as a GGUF for the speech-LLM checkpoint `target`; general.name is
    "<name> DFlash2 <TYPE>" (name: the target's directory name by default)."""
    from .salm import stage

    if not is_dflash2_checkpoint(source):
        raise RuntimeError(f"{source}: not a DFlash2 drafter (config.json architectures)")
    if outtype not in OUTTYPES:
        raise ValueError(
            f"dflash2 does not support --outtype {outtype}; choose {', '.join(OUTTYPES)}"
        )
    work = outfile.with_name(outfile.name + ".work")
    if work.exists():
        shutil.rmtree(work)
    drafter = work / "drafter"
    drafter.mkdir(parents=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["architectures"] = ["DFlash2DraftModel"]
    (drafter / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    weights = sorted(source.glob("*.safetensors")) + sorted(source.glob("*.safetensors.index.json"))
    if not any(w.suffix == ".safetensors" for w in weights):
        raise RuntimeError(f"{source}: no safetensors")
    for w in weights:
        (drafter / w.name).symlink_to(w.resolve())
    staged_target = work / "target"
    stage(target, staged_target, weights=False)
    print(f"[convert-dflash2] drafter {source} for the decoder of {target}")
    try:
        subprocess.run(
            [
                sys.executable,
                str(llama_cpp / "convert_hf_to_gguf.py"),
                str(drafter),
                "--target-model-dir",
                str(staged_target),
                "--outtype",
                outtype,
                "--outfile",
                str(outfile),
                "--model-name",
                f"{name or target.name} DFlash2 {outtype.upper()}",
                *(["--dry-run"] if dry_run else []),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
