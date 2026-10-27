# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The encoder file of a speech-LLM (SALM): its speech encoder (pee) and, optionally, its CTC timestamp head
(ctc_timestamp) in one GGUF, next to the decoder GGUF that names it (nemo_speech.speech_model, conversion/salm.py).

The encoder's general.* keys describe the file, every other key and every tensor is copied unchanged, and
nemo_speech.components lists the models it holds. Names must not collide: the parts use disjoint tensor prefixes
(asr / diar / fusion / preprocessor, ctc) and key prefixes. nemo_speech.checkpoint_id identifies the checkpoint the
parts come from; the runtime refuses a decoder and an encoder file with different ids.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

COMPONENTS_KEY = "nemo_speech.components"
CHECKPOINT_ID_KEY = "nemo_speech.checkpoint_id"


def write_encoder_file(
    encoder: Path,
    ctc: Path | None,
    outfile: Path,
    name: str | None = None,
    checkpoint_id: str | None = None,
) -> None:
    parts = [("pee", encoder)] + ([("ctc_timestamp", ctc)] if ctc else [])
    readers = []
    for component, path in parts:
        reader = gguf.GGUFReader(path)
        field = reader.fields.get(gguf.Keys.General.ARCHITECTURE)
        arch = field.contents() if field else ""
        if arch != component:
            raise ValueError(f"{path}: general.architecture is {arch!r}, expected {component!r}")
        if COMPONENTS_KEY in reader.fields:
            raise ValueError(f"{path} already holds several models")
        readers.append((component, path, reader))

    temporary = outfile.with_name(outfile.name + ".writing")
    writer = gguf.GGUFWriter(temporary, arch="pee")
    keys: set[str] = set()
    for index, (_, path, reader) in enumerate(readers):
        for field in reader.fields.values():
            if field.name.startswith("GGUF.") or field.name == gguf.Keys.General.ARCHITECTURE:
                continue
            if index > 0 and field.name.startswith("general."):
                continue  # the encoder's general.* describe the file
            if field.name == CHECKPOINT_ID_KEY or (name and field.name == gguf.Keys.General.NAME):
                continue
            if field.name in keys:
                raise ValueError(f"{path}: key {field.name} is already in the encoder file")
            keys.add(field.name)
            value_type = field.types[0]
            sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)
    if name:
        writer.add_name(name)
    writer.add_array(COMPONENTS_KEY, [component for component, *_ in readers])
    if checkpoint_id:
        writer.add_string(CHECKPOINT_ID_KEY, checkpoint_id)

    names: set[str] = set()
    for _, path, reader in readers:
        for tensor in reader.tensors:
            if tensor.name in names:
                raise ValueError(f"{path}: tensor {tensor.name} is already in the encoder file")
            names.add(tensor.name)
            writer.add_tensor_info(
                tensor.name,
                tensor.data.shape,
                tensor.data.dtype,
                tensor.data.nbytes,
                tensor.tensor_type,
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, _, reader in readers:
        for tensor in reader.tensors:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    temporary.replace(outfile)
    print(
        f"[salm-encoder] {outfile}: {', '.join(c for c, *_ in readers)}; {len(names)} tensors, "
        f"{outfile.stat().st_size / 2**30:.2f} GiB"
    )
