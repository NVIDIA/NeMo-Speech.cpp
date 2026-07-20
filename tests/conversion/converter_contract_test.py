#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conversion.registry import (
    ConversionRequest,
    _convert_nmt,
    _normalized_outtype,
    detect_architecture,
)
from conversion.source import extract_archive


class ConverterContractTest(unittest.TestCase):
    def _checkpoint(self, root: Path, name: str, config: dict) -> Path:
        checkpoint = root / f"{name}.nemo"
        payload = yaml.safe_dump(config).encode("utf-8")
        with tarfile.open(checkpoint, "w") as archive:
            member = tarfile.TarInfo("model_config.yaml")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        return checkpoint

    def test_architecture_detection(self) -> None:
        configs = {
            "asr": {
                "target": "nemo.collections.asr.models.EncDecCTCModel",
                "encoder": {},
                "decoder": {},
            },
            "diarization": {"target": "nemo.collections.asr.models.SortformerEncLabelModel"},
            "pnc": {"target": "nemo.collections.nlp.models.PunctuationCapitalizationModel"},
            "tts": {"target": "nemo.collections.tts.models.magpietts.MagpieTTSModel"},
            "codec": {"target": "nemo.collections.tts.models.AudioCodecModel"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for expected, config in configs.items():
                checkpoint = self._checkpoint(root, expected, config)
                request = ConversionRequest(source=str(checkpoint), outfile=root / "out.gguf")
                actual, resolved = detect_architecture(request)
                self.assertEqual(actual, expected)
                self.assertEqual(resolved, checkpoint)

            nmt = root / "nmt"
            nmt.mkdir()
            (nmt / "config.json").write_text("{}\n", encoding="utf-8")
            actual, resolved = detect_architecture(
                ConversionRequest(source=str(nmt), outfile=root / "nmt.gguf")
            )
            self.assertEqual(actual, "nmt")
            self.assertIsNone(resolved)

            actual, resolved = detect_architecture(
                ConversionRequest(source="silero", outfile=root / "vad.gguf")
            )
            self.assertEqual(actual, "vad")
            self.assertIsNone(resolved)

    def test_output_type_defaults_and_validation(self) -> None:
        self.assertEqual(_normalized_outtype("asr", "auto"), "q8_0")
        self.assertEqual(_normalized_outtype("diarization", "auto"), "f32")
        self.assertEqual(_normalized_outtype("tts", "fp16"), "f16")
        with self.assertRaises(ValueError):
            _normalized_outtype("vad", "q8_0")

    def test_archive_traversal_and_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, member in (
                ("traversal", tarfile.TarInfo("../outside")),
                ("link", tarfile.TarInfo("link")),
            ):
                archive_path = root / f"{name}.nemo"
                with tarfile.open(archive_path, "w") as archive:
                    if name == "link":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "target"
                        archive.addfile(member)
                    else:
                        payload = b"invalid"
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                with self.assertRaises(RuntimeError):
                    extract_archive(archive_path, root / f"extract-{name}")

    def test_root_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "convert_model.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--architecture", result.stdout)
        self.assertIn("--outfile", result.stdout)
        self.assertIn("--local-transformer-outtype", result.stdout)

    def test_nmt_adapter_honors_hugging_face_revision_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = ConversionRequest(
                source="org/model",
                outfile=root / "model.gguf",
                architecture="nmt",
                outtype="f16",
                revision="release",
                cache_dir=root / "cache",
            )
            snapshot = root / "snapshot"
            snapshot.mkdir()
            with mock.patch(
                "huggingface_hub.snapshot_download", return_value=str(snapshot)
            ) as download, mock.patch("conversion.registry.subprocess.run") as run:
                _convert_nmt(request, "f16")

            download.assert_called_once_with(
                repo_id="org/model", revision="release", cache_dir=str(root / "cache")
            )
            command = run.call_args.args[0]
            self.assertEqual(Path(command[2]), snapshot)
            self.assertEqual(command[-2:], ["--outtype", "f16"])
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
