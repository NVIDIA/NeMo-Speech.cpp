#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the data tables used by the native Magpie Mandarin tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from importlib import metadata
from pathlib import Path

import jieba
from jieba.finalseg.prob_emit import P as HMM_EMIT
from jieba.finalseg.prob_start import P as HMM_START
from jieba.finalseg.prob_trans import P as HMM_TRANS
from pypinyin import Style, lazy_pinyin
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
from pypinyin_dict.pinyin_data import cc_cedict

STATES = "BEMS"
MIN_FLOAT = -3.14e100
ZH_PUNCT = list("，。？！；：、‘’“”（）【】「」《》") + list(',.!?-:;/"()[]{}')
MANDARIN_OFFSET = 349
EOS_ID = 2361
EXPECTED_VERSIONS = {
    "jieba": "0.42.1",
    "pypinyin": "0.55.0",
    "pypinyin-dict": "0.9.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phoneme-dict", type=Path)
    parser.add_argument("--utterances", type=Path)
    parser.add_argument("--golden-output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tone3(text: str) -> list[str]:
    return lazy_pinyin(
        text,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors=lambda value: list(value),
    )


def write_hmm_model(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(" ".join(str(HMM_START.get(state, MIN_FLOAT)) for state in STATES) + "\n")
        for source_state in STATES:
            output.write(
                " ".join(str(HMM_TRANS[source_state].get(state, MIN_FLOAT)) for state in STATES)
                + "\n"
            )
        for state in STATES:
            entries = sorted(HMM_EMIT[state].items(), key=lambda item: ord(item[0]))
            output.write(",".join(f"{char}:{prob}" for char, prob in entries) + "\n")


def write_pinyin_tables(output_dir: Path) -> None:
    # NeMo loads this table before creating ChineseG2p. It overrides pypinyin's
    # ordered single-character readings but deliberately leaves phrase data intact.
    cc_cedict.load()

    with (output_dir / "pinyin_chars.tsv").open("w", encoding="utf-8", newline="\n") as output:
        for codepoint in sorted(PINYIN_DICT):
            char = chr(codepoint)
            converted = tone3(char)
            if converted and converted[0] != char:
                output.write(f"{codepoint:X}\t{converted[0]}\n")

    with (output_dir / "pinyin_phrases.tsv").open("w", encoding="utf-8", newline="\n") as output:
        for phrase in sorted(PHRASES_DICT):
            converted = tone3(phrase)
            if converted:
                output.write(f"{phrase}\t{' '.join(converted)}\n")


def load_phoneme_dict(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(";;;"):
            continue
        syllable, pronunciation = line.split("\t", 1)
        result[syllable.lower()] = pronunciation.lower().split()
    return result


def vocabulary(phoneme_dict: dict[str, list[str]]) -> list[str]:
    phonemes = sorted({phone for value in phoneme_dict.values() for phone in value})
    return (
        [" "]
        + phonemes
        + [f"#{tone}" for tone in range(1, 6)]
        + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + ["'"]
        + ZH_PUNCT
        + ["<pad>", "<oov>"]
    )


def reference_trace(text: str, phoneme_dict: dict[str, list[str]]) -> dict[str, object]:
    # ChineseG2p applies the configured ASCII case before word segmentation.
    words = list(jieba.cut(text.upper()))
    pinyin: list[str] = []
    for word in words:
        pinyin.extend(tone3(word))

    symbols: list[str] = []
    for value in pinyin:
        if value[-1:] in "12345":
            pronunciation = phoneme_dict.get(value[:-1])
            if pronunciation:
                symbols.extend(pronunciation)
                symbols.append(f"#{value[-1]}")
        elif value[-1:] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            symbols.append(value[-1])
        else:
            symbols.append(value)

    tokens = vocabulary(phoneme_dict)
    token_to_id = {token: MANDARIN_OFFSET + index for index, token in enumerate(tokens)}
    kept: list[str] = []
    for symbol in symbols:
        if symbol == " ":
            if kept and kept[-1] != " ":
                kept.append(symbol)
        elif symbol in token_to_id:
            kept.append(symbol)
    while kept and kept[-1] == " ":
        kept.pop()
    kept = [" "] + kept + [" "]
    ids = [token_to_id[symbol] for symbol in kept] + [EOS_ID]
    return {"text": text, "words": words, "pinyin": pinyin, "symbols": kept, "tokens": ids}


def write_golden(path: Path, utterances: Path, phoneme_dict_path: Path) -> None:
    phoneme_dict = load_phoneme_dict(phoneme_dict_path)
    rows = [
        reference_trace(line, phoneme_dict)
        for line in utterances.read_text(encoding="utf-8").splitlines()
        if line
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    token_path = path.with_suffix(".tokens")
    token_path.write_text(
        "\n".join(",".join(str(token) for token in row["tokens"]) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_versions = {package: metadata.version(package) for package in EXPECTED_VERSIONS}
    if source_versions != EXPECTED_VERSIONS:
        raise SystemExit(
            f"Mandarin data generator requires {EXPECTED_VERSIONS}, found {source_versions}"
        )

    jieba_dict = Path(jieba.__file__).resolve().parent / "dict.txt"
    shutil.copyfile(jieba_dict, args.output_dir / "jieba.dict.utf8")
    write_hmm_model(args.output_dir / "hmm_model.utf8")
    write_pinyin_tables(args.output_dir)

    generated = [
        args.output_dir / "jieba.dict.utf8",
        args.output_dir / "hmm_model.utf8",
        args.output_dir / "pinyin_chars.tsv",
        args.output_dir / "pinyin_phrases.tsv",
    ]
    manifest = {
        "format": 1,
        "sources": source_versions,
        "files": {path.name: sha256(path) for path in generated},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.golden_output:
        if not args.phoneme_dict or not args.utterances:
            raise SystemExit("--golden-output requires --phoneme-dict and --utterances")
        write_golden(args.golden_output, args.utterances, args.phoneme_dict)


if __name__ == "__main__":
    main()
