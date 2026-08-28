#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conversion.tts_tokenizer_profiles import (  # noqa: E402
    TOKENIZER_ORDERS,
    TOKENIZER_PROFILE_NEMO_VERSIONS,
    TOKENIZER_TARGETS,
    V2607_LANGUAGE_MAPPING,
    tokenizer_profile,
)


class TtsTokenizerProfilesTest(unittest.TestCase):
    @staticmethod
    def _config(profile: str) -> dict:
        tokenizers = {
            name: {"_target_": TOKENIZER_TARGETS[profile][name]}
            for name in TOKENIZER_ORDERS[profile]
        }
        tokenizers["japanese_phoneme"]["g2p"] = {
            "ascii_letter_case": "upper" if profile == "v2602" else "lower"
        }
        config = {
            "nemo_version": TOKENIZER_PROFILE_NEMO_VERSIONS[profile],
            "text_tokenizers": tokenizers,
        }
        if profile == "v2607":
            tokenizers["hindi_phoneme"]["locale"] = "hi-IN"
            tokenizers["hindi_phoneme"]["g2p"] = {"locale": "hi-IN"}
            tokenizers["portuguese_Brazilian_phoneme"]["locale_specific_punct"] = False
            config["language_to_tokenizer_mapping"] = V2607_LANGUAGE_MAPPING.copy()
        return config

    def test_known_profiles_require_matching_dimensions(self) -> None:
        self.assertEqual(tokenizer_profile(self._config("v2602"), 2362, 1), "v2602")
        self.assertEqual(tokenizer_profile(self._config("v2607"), 3359, 2), "v2607")
        with self.assertRaisesRegex(ValueError, "requires text_vocab_size"):
            tokenizer_profile(self._config("v2602"), 3359, 2)
        with self.assertRaisesRegex(ValueError, "requires text_vocab_size"):
            tokenizer_profile(self._config("v2607"), 2362, 1)

    def test_unknown_or_reordered_layout_is_rejected(self) -> None:
        config = self._config("v2607")
        config["text_tokenizers"]["unexpected_tokenizer"] = {}
        with self.assertRaisesRegex(ValueError, "unsupported Magpie tokenizer layout"):
            tokenizer_profile(config, 3359, 2)

        reordered = list(TOKENIZER_ORDERS["v2607"])
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(ValueError, "unsupported Magpie tokenizer layout"):
            tokenizer_profile({"text_tokenizers": {name: {} for name in reordered}}, 3359, 2)

    def test_critical_tokenizer_config_changes_are_rejected(self) -> None:
        config = self._config("v2607")
        config["text_tokenizers"]["hindi_phoneme"]["_target_"] = "HindiCharsTokenizer"
        with self.assertRaisesRegex(ValueError, "tokenizer target for hindi_phoneme"):
            tokenizer_profile(config, 3359, 2)

        config = self._config("v2607")
        config["text_tokenizers"]["japanese_phoneme"]["g2p"]["ascii_letter_case"] = "upper"
        with self.assertRaisesRegex(ValueError, "Japanese ascii_letter_case"):
            tokenizer_profile(config, 3359, 2)

    def test_null_g2p_mappings_are_rejected(self) -> None:
        config = self._config("v2602")
        config["text_tokenizers"]["japanese_phoneme"]["g2p"] = None
        with self.assertRaisesRegex(ValueError, "Japanese ascii_letter_case"):
            tokenizer_profile(config, 2362, 1)

        config = self._config("v2607")
        config["text_tokenizers"]["hindi_phoneme"]["g2p"] = None
        with self.assertRaisesRegex(ValueError, "Hindi IPA locale"):
            tokenizer_profile(config, 3359, 2)


if __name__ == "__main__":
    unittest.main()
