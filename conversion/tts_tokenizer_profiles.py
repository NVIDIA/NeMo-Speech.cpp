#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Known MagpieTTS tokenizer layouts shared by conversion checks."""

from __future__ import annotations

from typing import Any

TOKENIZER_ORDERS = {
    "v2602": (
        "english_phoneme",
        "spanish_phoneme",
        "german_phoneme",
        "mandarin_phoneme",
        "japanese_phoneme",
        "french_chartokenizer",
        "hindi_chartokenizer",
        "italian_phoneme",
        "vietnamese_phoneme",
        "text_ce_tokenizer",
    ),
    "v2607": (
        "english_phoneme",
        "text_ce_tokenizer",
        "spanish_phoneme",
        "german_phoneme",
        "mandarin_phoneme",
        "japanese_phoneme",
        "portuguese_Brazilian_phoneme",
        "hindi_phoneme",
        "arabic_AE_chartokenizer",
        "arabic_SA_chartokenizer",
        "arabic_MSA_chartokenizer",
        "french_chartokenizer",
        "italian_chartokenizer",
        "vietnamese_chartokenizer",
        "korean_chartokenizer",
    ),
}

TOKENIZER_PROFILE_DIMENSIONS = {
    "v2602": (2362, 1),
    "v2607": (3359, 2),
}

TOKENIZER_PROFILE_NEMO_VERSIONS = {
    "v2602": "2.6.0rc0",
    "v2607": "2.8.0rc0",
}

IPA_TARGET = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer"
BYT5_TARGET = "AutoTokenizer"
TOKENIZER_TARGETS = {
    "v2602": {
        "english_phoneme": IPA_TARGET,
        "spanish_phoneme": IPA_TARGET,
        "german_phoneme": IPA_TARGET,
        "mandarin_phoneme": (
            "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
            "ChinesePhonemesTokenizer"
        ),
        "japanese_phoneme": (
            "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
            "JapanesePhonemeTokenizer"
        ),
        "french_chartokenizer": BYT5_TARGET,
        "hindi_chartokenizer": (
            "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
            "HindiCharsTokenizer"
        ),
        "italian_phoneme": BYT5_TARGET,
        "vietnamese_phoneme": BYT5_TARGET,
        "text_ce_tokenizer": BYT5_TARGET,
    },
    "v2607": {
        "english_phoneme": IPA_TARGET,
        "text_ce_tokenizer": BYT5_TARGET,
        "spanish_phoneme": IPA_TARGET,
        "german_phoneme": IPA_TARGET,
        "mandarin_phoneme": (
            "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
            "ChinesePhonemesTokenizer"
        ),
        "japanese_phoneme": (
            "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
            "JapanesePhonemeTokenizer"
        ),
        "portuguese_Brazilian_phoneme": IPA_TARGET,
        "hindi_phoneme": IPA_TARGET,
        "arabic_AE_chartokenizer": (
            "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
            "ArabicCharsTokenizer"
        ),
        "arabic_SA_chartokenizer": (
            "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
            "ArabicCharsTokenizer"
        ),
        "arabic_MSA_chartokenizer": (
            "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers."
            "ArabicCharsTokenizer"
        ),
        "french_chartokenizer": BYT5_TARGET,
        "italian_chartokenizer": BYT5_TARGET,
        "vietnamese_chartokenizer": BYT5_TARGET,
        "korean_chartokenizer": BYT5_TARGET,
    },
}

V2607_LANGUAGE_MAPPING = {
    "en": ["english_phoneme"],
    "de": ["german_phoneme"],
    "es": ["spanish_phoneme"],
    "fr": ["french_chartokenizer"],
    "it": ["italian_chartokenizer"],
    "vi": ["vietnamese_chartokenizer"],
    "zh": ["mandarin_phoneme"],
    "hi": ["hindi_phoneme"],
    "ja": ["japanese_phoneme"],
    "pt-BR": ["portuguese_Brazilian_phoneme"],
    "ko": ["korean_chartokenizer"],
    "ar-AE": ["arabic_AE_chartokenizer"],
    "ar-SA": ["arabic_SA_chartokenizer"],
    "ar-MSA": ["arabic_MSA_chartokenizer"],
}


def tokenizer_profile(cfg: dict[str, Any], text_vocab: int, frame_stacking: int) -> str:
    tokenizers = cfg.get("text_tokenizers")
    if not isinstance(tokenizers, dict):
        raise ValueError("Magpie config has no text_tokenizers mapping")
    order = tuple(tokenizers)
    matches = [name for name, expected in TOKENIZER_ORDERS.items() if order == expected]
    if len(matches) != 1:
        raise ValueError(f"unsupported Magpie tokenizer layout: {list(order)}")
    profile = matches[0]
    if str(cfg.get("nemo_version", "")) != TOKENIZER_PROFILE_NEMO_VERSIONS[profile]:
        raise ValueError(f"unsupported {profile} nemo_version: {cfg.get('nemo_version')!r}")
    for name, target in TOKENIZER_TARGETS[profile].items():
        tokenizer = tokenizers.get(name)
        actual_target = tokenizer.get("_target_") if isinstance(tokenizer, dict) else None
        if actual_target != target:
            raise ValueError(
                f"unsupported {profile} tokenizer target for {name}: "
                f"{actual_target!r}"
            )
    expected_japanese_case = "upper" if profile == "v2602" else "lower"
    japanese_g2p = tokenizers["japanese_phoneme"].get("g2p")
    japanese_case = (
        japanese_g2p.get("ascii_letter_case") if isinstance(japanese_g2p, dict) else None
    )
    if japanese_case != expected_japanese_case:
        raise ValueError(f"unsupported {profile} Japanese ascii_letter_case: {japanese_case!r}")
    if profile == "v2607":
        hindi = tokenizers["hindi_phoneme"]
        hindi_g2p = hindi.get("g2p")
        hindi_g2p_locale = hindi_g2p.get("locale") if isinstance(hindi_g2p, dict) else None
        if hindi.get("locale") != "hi-IN" or hindi_g2p_locale != "hi-IN":
            raise ValueError("unsupported v2607 Hindi IPA locale")
        if tokenizers["portuguese_Brazilian_phoneme"].get("locale_specific_punct") is not False:
            raise ValueError("unsupported v2607 Portuguese punctuation mode")
        if cfg.get("language_to_tokenizer_mapping") != V2607_LANGUAGE_MAPPING:
            raise ValueError("unsupported v2607 language_to_tokenizer_mapping")
    expected_vocab, expected_stacking = TOKENIZER_PROFILE_DIMENSIONS[profile]
    if (text_vocab, frame_stacking) != (expected_vocab, expected_stacking):
        raise ValueError(
            f"Magpie tokenizer profile {profile} requires text_vocab_size={expected_vocab} "
            f"and frame_stacking_factor={expected_stacking}, got "
            f"text_vocab_size={text_vocab} and frame_stacking_factor={frame_stacking}"
        )
    return profile
