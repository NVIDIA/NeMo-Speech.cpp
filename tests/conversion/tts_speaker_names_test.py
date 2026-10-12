#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gguf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conversion.tts import convert, read_speaker_names  # noqa: E402


class TtsSpeakerNamesTest(unittest.TestCase):
    def test_checkpoint_ids_determine_order(self) -> None:
        # Deliberately use key orders different from speaker ID order.
        for mapping, expected in (
            (
                {"Aria": 2, "Jason": 3, "John": 0, "Leo": 4, "Sofia": 1},
                ["John", "Sofia", "Aria", "Jason", "Leo"],
            ),
            (
                {"Sofia": 4, "John": 2, "Leo": 3, "Aria": 0, "Jason": 1},
                ["Aria", "Jason", "John", "Leo", "Sofia"],
            ),
        ):
            with self.subTest(mapping=mapping):
                self.assertEqual(read_speaker_names({"speaker_map": mapping}, 5), expected)

    def test_invalid_maps_are_rejected(self) -> None:
        for mapping in (
            None,
            {},
            {"John": 0, "Aria": 0},
            {"John": 0, "Aria": 2},
            {"John": -1, "Aria": 0},
            {"John": True, "Aria": 0},
            {"John": "0", "Aria": 1},
            {"": 0, "Aria": 1},
            {"John": 0, "john": 1},
            {"John": 0},
        ):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                read_speaker_names({"speaker_map": mapping}, 2)

    def test_missing_or_ambiguous_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"speaker_map": "nemo:speakers.json"}
            with self.assertRaisesRegex(ValueError, "found 0"):
                read_speaker_names(config, 1, root)
            for directory in (root / "a", root / "b"):
                directory.mkdir()
                (directory / "speakers.json").write_text('{"John": 0}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "found 2"):
                read_speaker_names(config, 1, root)

    def test_archive_conversion_preserves_speaker_ids_in_gguf_and_summary(self) -> None:
        layouts = {
            "v2602": {"Aria": 2, "Jason": 3, "John": 0, "Leo": 4, "Sofia": 1},
            "v2607": {"Aria": 0, "Jason": 1, "John": 2, "Leo": 3, "Sofia": 4},
        }
        for profile, mapping in layouts.items():
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifact = "nested/hash_magpie.nemo.speakers.json"
                config = {
                    "target": "nemo.collections.tts.models.magpietts.MagpieTTSModel",
                    "model_type": "decoder_ce",
                    "speaker_map": "nemo:" + Path(artifact).name,
                    "encoder": {"n_layers": 1, "sa_n_heads": 1, "kernel_size": 1},
                    "decoder": {
                        "d_model": 1,
                        "d_ffn": 1,
                        "n_layers": 1,
                        "max_length_causal_mask": 8,
                        "sa_n_heads": 1,
                        "xa_n_heads": 1,
                        "xa_d_head": 1,
                        "kernel_size": 1,
                    },
                }
                state = {
                    "audio_embeddings.0.weight": torch.zeros((16, 1)),
                    "text_embedding.weight": torch.zeros((1, 1)),
                    "_baked_embedding_T": torch.tensor(1),
                    "_baked_embedding_D": torch.tensor(1),
                    "baked_context_embedding_len": torch.ones(5, dtype=torch.int32),
                    "baked_context_embedding.weight": torch.zeros((5, 1)),
                    "final_proj.weight": torch.zeros((16, 1)),
                    "local_transformer_out_projections.0.weight": torch.zeros((1, 1)),
                    "local_transformer.position_embeddings.weight": torch.zeros((1, 1)),
                }
                archive = root / "model.nemo"
                with tarfile.open(archive, "w") as tar:
                    for name, payload in (
                        ("model_config.yaml", yaml.safe_dump(config).encode()),
                        ("model_weights.ckpt", b"mocked weights"),
                        (artifact, json.dumps(mapping).encode()),
                    ):
                        member = tarfile.TarInfo(name)
                        member.size = len(payload)
                        tar.addfile(member, io.BytesIO(payload))
                output, summary_path = root / "model.gguf", root / "summary.json"
                with (
                    mock.patch("conversion.tts.read_state_dict", return_value=state),
                    mock.patch("conversion.tts.tokenizer_profile", return_value=profile),
                    mock.patch("conversion.tts.add_tensors", return_value=(0, [])),
                ):
                    convert(archive, output, metadata_json=summary_path)
                expected = sorted(mapping, key=mapping.__getitem__)
                reader = gguf.GGUFReader(output)
                self.assertEqual(reader.get_field("magpietts.speaker_names").contents(), expected)
                self.assertEqual(json.loads(summary_path.read_text())["speaker_names"], expected)


if __name__ == "__main__":
    unittest.main()
