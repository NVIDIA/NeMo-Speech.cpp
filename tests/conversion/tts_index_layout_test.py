#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conversion.tts import (  # noqa: E402
    _indexed_weight_indices,
    _require_contiguous_indices,
    add_metadata,
)


class TtsIndexLayoutTest(unittest.TestCase):
    def test_contiguous_audio_embedding_indexes(self) -> None:
        state = {
            "audio_embeddings.0.weight": object(),
            "audio_embeddings.1.weight": object(),
            "audio_embeddings.2.weight": object(),
        }
        indices = _indexed_weight_indices(state, "audio_embeddings.")
        _require_contiguous_indices("audio embedding", indices, 3)
        self.assertEqual(indices, [0, 1, 2])

    def test_sparse_audio_embedding_indexes_are_rejected(self) -> None:
        state = {
            "audio_embeddings.0.weight": object(),
            "audio_embeddings.2.weight": object(),
        }
        indices = _indexed_weight_indices(state, "audio_embeddings.")
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            _require_contiguous_indices("audio embedding", indices, 2)

    def test_duplicate_lt_head_indexes_are_rejected(self) -> None:
        state = {
            "local_transformer_out_projections.0.weight": object(),
            "local_transformer_out_projections.00.weight": object(),
        }
        indices = _indexed_weight_indices(state, "local_transformer_out_projections.")
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            _require_contiguous_indices("local transformer output projection", indices, 2)

    def test_non_numeric_index_is_rejected(self) -> None:
        state = {"audio_embeddings.first.weight": object()}
        with self.assertRaisesRegex(ValueError, "non-numeric weight index"):
            _indexed_weight_indices(state, "audio_embeddings.")

    def test_sparse_audio_embeddings_fail_before_metadata_is_written(self) -> None:
        writer = mock.Mock()
        config = {"encoder": {}, "decoder": {}, "frame_stacking_factor": 1}
        state = {
            "audio_embeddings.0.weight": object(),
            "audio_embeddings.2.weight": object(),
        }
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            add_metadata(writer, config, state)  # type: ignore[arg-type]
        self.assertFalse(writer.method_calls)

    def test_duplicate_lt_heads_fail_before_metadata_is_written(self) -> None:
        writer = mock.Mock()
        config = {"encoder": {}, "decoder": {}, "frame_stacking_factor": 1}
        state = {
            "audio_embeddings.0.weight": torch.zeros((16, 1)),
            "audio_embeddings.1.weight": torch.zeros((16, 1)),
            "text_embedding.weight": torch.zeros((1, 1)),
            "_baked_embedding_T": torch.tensor(1),
            "_baked_embedding_D": torch.tensor(1),
            "baked_context_embedding_len": torch.tensor([1]),
            "final_proj.weight": torch.zeros((32, 1)),
            "local_transformer_out_projections.0.weight": torch.zeros((1, 1)),
            "local_transformer_out_projections.00.weight": torch.zeros((1, 1)),
        }
        with mock.patch("conversion.tts.tokenizer_profile", return_value="test"):
            with self.assertRaisesRegex(ValueError, "must be contiguous"):
                add_metadata(writer, config, state)
        self.assertFalse(writer.method_calls)


if __name__ == "__main__":
    unittest.main()
