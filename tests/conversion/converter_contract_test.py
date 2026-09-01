#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from gguf import GGMLQuantizationType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conversion.registry import (
    ConversionRequest,
    _convert_nmt,
    _normalized_outtype,
    detect_architecture,
)
from conversion.s2s import (
    COMPONENT_FORMATS,
    RUNTIME_ARTIFACTS,
    build_conversion_plan,
    component_formats,
    output_bundle_dir,
)
from conversion.s2s_components.perception import _preserve_subsampling_precision
from conversion.s2s_components.voicechat_source import (
    default_quantizer_path,
    ensure_llama_checkout,
    ensure_quantizer,
    find_quantizer,
    load_llm_channel_weights,
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

    def _s2s_repository(self, root: Path) -> Path:
        version = root / "nemotron-voicechat" / "1"
        for relative in (
            "config.json",
            "perception.safetensors",
            "rnnt-asr.safetensors",
            "embeddings.safetensors",
            "codec.safetensors",
            "rnnt_tokenizer/vocab.json",
            "eartts_vllm/config.json",
            "eartts_vllm/model.safetensors",
            "eartts_vllm/tts_model_init_inputs.pt",
            "nano-v2-vllm/config.json",
            "nano-v2-vllm/model.safetensors",
        ):
            path = version / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return version

    def _hf_s2s_repository(self, root: Path) -> Path:
        for relative in ("config.json", "model.safetensors", "rnnt_tokenizer/vocab.json"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return root

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

            s2s_version = self._s2s_repository(root / "voicechat-source")
            s2s_root = s2s_version.parents[1]
            actual, resolved = detect_architecture(
                ConversionRequest(source=str(s2s_root), outfile=root / "voicechat-gguf")
            )
            self.assertEqual(actual, "s2s")
            self.assertEqual(resolved, s2s_root.resolve())

            hf_s2s = self._hf_s2s_repository(root / "NVIDIA-NemotronLabs-VoiceChat-11B")
            actual, resolved = detect_architecture(
                ConversionRequest(source=str(hf_s2s), outfile=root / "voicechat-hf-gguf")
            )
            self.assertEqual(actual, "s2s")
            self.assertEqual(resolved, hf_s2s.resolve())

            actual, resolved = detect_architecture(
                ConversionRequest(source="silero", outfile=root / "vad.gguf")
            )
            self.assertEqual(actual, "vad")
            self.assertIsNone(resolved)

    def test_output_type_defaults_and_validation(self) -> None:
        self.assertEqual(_normalized_outtype("asr", "auto"), "q8_0")
        self.assertEqual(_normalized_outtype("diarization", "auto"), "f32")
        self.assertEqual(_normalized_outtype("tts", "fp16"), "f16")
        self.assertEqual(_normalized_outtype("s2s", "auto"), "q4_k_m")
        self.assertEqual(_normalized_outtype("s2s", "q4"), "q4_k_m")
        self.assertEqual(_normalized_outtype("s2s", "nvfp4"), "nvfp4")
        with self.assertRaises(ValueError):
            _normalized_outtype("vad", "q8_0")
        with self.assertRaises(ValueError):
            _normalized_outtype("s2s", "q8_0")

    def test_s2s_channel_weights_come_from_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            config = {
                "model": {
                    "stt": {
                        "model": {
                            "duplex_user_channel_weight": 0.75,
                            "duplex_text_channel_weight": 1.25,
                            "duplex_function_channel_weight": 2.5,
                        }
                    }
                }
            }
            (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(
                load_llm_channel_weights(source),
                {"user": 0.75, "text": 1.25, "function": 2.5},
            )

    def test_s2s_quantization_preserves_bf16_perception_stem(self) -> None:
        values = np.array([1.00390625, -0.998046875], dtype=np.float32)
        expected = torch.from_numpy(values).to(torch.bfloat16).float().numpy()

        weight, weight_type = _preserve_subsampling_precision(
            "encoder.pre_encode.conv.0.weight",
            values,
            GGMLQuantizationType.Q8_0,
        )
        bias, bias_type = _preserve_subsampling_precision(
            "encoder.pre_encode.conv.0.bias",
            values,
            GGMLQuantizationType.Q8_0,
        )
        non_stem, non_stem_type = _preserve_subsampling_precision(
            "encoder.layers.0.self_attn.linear_q.weight",
            values,
            GGMLQuantizationType.Q8_0,
        )

        np.testing.assert_array_equal(weight, expected)
        np.testing.assert_array_equal(bias, expected)
        self.assertEqual(weight_type, GGMLQuantizationType.BF16)
        self.assertEqual(bias_type, GGMLQuantizationType.F32)
        self.assertIs(non_stem, values)
        self.assertEqual(non_stem_type, GGMLQuantizationType.Q8_0)

    def test_s2s_bundle_plan_matches_runtime_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._s2s_repository(root / "source")
            llama_cpp = root / "llama.cpp"
            (llama_cpp / "convert_hf_to_gguf.py").parent.mkdir(parents=True)
            (llama_cpp / "convert_hf_to_gguf.py").touch()
            destination = root / "artifacts" / "nemotron-voicechat" / "1"
            plan = build_conversion_plan(source, destination, llama_cpp)

            self.assertEqual(len(plan), 7)
            self.assertEqual(
                {step.output.relative_to(destination).as_posix() for step in plan},
                set(RUNTIME_ARTIFACTS) - {"rnnt_tokenizer/vocab.json"},
            )
            commands = {step.name: step.command for step in plan}
            self.assertEqual(commands["perception"][-1], "q8_0")
            self.assertEqual(commands["eartts_side"][-1], COMPONENT_FORMATS["eartts_side"])
            self.assertEqual(commands["codec"][-1], COMPONENT_FORMATS["codec"])

    def test_hf_s2s_plan_quantizes_backbones_and_uses_flat_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._hf_s2s_repository(root / "source")
            base_model = root / "base-model"
            base_model.mkdir()
            (base_model / "tokenizer.json").touch()
            llama_cpp = root / "llama.cpp"
            (llama_cpp / "convert_hf_to_gguf.py").parent.mkdir(parents=True)
            (llama_cpp / "convert_hf_to_gguf.py").touch()
            quantizer = root / "llama-quantize"
            destination = root / "artifacts"

            plan = build_conversion_plan(
                source,
                destination,
                llama_cpp,
                profile="nvfp4",
                base_model_dir=base_model,
                quantizer=quantizer,
            )
            commands = {step.name: step.command for step in plan}
            self.assertEqual(output_bundle_dir(source, source, destination), destination.resolve())
            self.assertEqual(component_formats("nvfp4")["perception"], "q8_0")
            self.assertEqual(component_formats("nvfp4")["llm_backbone"], "nvfp4")
            self.assertEqual(component_formats("nvfp4")["eartts_backbone"], "q4_k_m")
            self.assertIn("--base-model-dir", commands["llm_backbone"])
            self.assertEqual(
                commands["llm_backbone"][-4:],
                ("--quantize", "nvfp4", "--quantizer", str(quantizer)),
            )
            self.assertEqual(
                commands["eartts_backbone"][-4:],
                ("--quantize", "q4_k_m", "--quantizer", str(quantizer)),
            )
            self.assertIn("--tokenizer-json", commands["eartts_side"])

    def test_explicit_s2s_quantizer_path_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fallback = root / "llama.cpp" / "build" / "bin" / "llama-quantize"
            fallback.parent.mkdir(parents=True)
            fallback.touch(mode=0o755)

            with self.assertRaisesRegex(FileNotFoundError, "is not executable"):
                find_quantizer(root / "llama.cpp", root / "missing-llama-quantize")

    def test_s2s_nvfp4_quantizer_requires_nvfp4_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            llama_cpp = root / "llama.cpp"
            quantizer = llama_cpp / "build" / "bin" / "llama-quantize"
            quantizer.parent.mkdir(parents=True)
            quantizer.touch(mode=0o755)

            unsupported = subprocess.CompletedProcess(
                [str(quantizer), "--help"], 0, stdout="Q4_K_M", stderr=""
            )
            with mock.patch(
                "conversion.s2s_components.voicechat_source.shutil.which",
                return_value=None,
            ), mock.patch(
                "conversion.s2s_components.voicechat_source.subprocess.run",
                return_value=unsupported,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "NVFP4 support"):
                    find_quantizer(llama_cpp, profile="nvfp4")
                with self.assertRaisesRegex(RuntimeError, "does not advertise NVFP4"):
                    find_quantizer(llama_cpp, quantizer, profile="nvfp4")

            supported = subprocess.CompletedProcess(
                [str(quantizer), "--help"], 0, stdout="NVFP4", stderr=""
            )
            with mock.patch(
                "conversion.s2s_components.voicechat_source.subprocess.run",
                return_value=supported,
            ):
                self.assertEqual(
                    find_quantizer(llama_cpp, quantizer, profile="nvfp4"),
                    quantizer,
                )

    def test_pinned_llama_checkout_is_initialized_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            llama_cpp = root / "llama.cpp"
            converter = llama_cpp / "convert_hf_to_gguf.py"

            def fake_run(command, **kwargs):
                self.assertEqual(
                    command,
                    ["/usr/bin/git", "submodule", "update", "--init", "llama.cpp"],
                )
                self.assertEqual(kwargs["cwd"], root)
                converter.parent.mkdir(parents=True)
                converter.touch()
                return subprocess.CompletedProcess(command, 0)

            with mock.patch(
                "conversion.s2s_components.voicechat_source.shutil.which",
                return_value="/usr/bin/git",
            ), mock.patch(
                "conversion.s2s_components.voicechat_source.subprocess.run",
                side_effect=fake_run,
            ):
                actual = ensure_llama_checkout(llama_cpp, root)

            self.assertEqual(actual, llama_cpp)

    def test_s2s_quantizer_is_built_automatically_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            llama_cpp = root / "llama.cpp"
            llama_cpp.mkdir()
            (llama_cpp / "CMakeLists.txt").touch()
            (llama_cpp / "convert_hf_to_gguf.py").touch()
            patch_script = root / "scripts" / "apply-llama-patches.sh"
            patch_script.parent.mkdir()
            patch_script.touch()
            built = default_quantizer_path(llama_cpp)

            def fake_run(command, **_kwargs):
                if "--build" in command:
                    built.parent.mkdir(parents=True)
                    built.touch(mode=0o755)
                if command == [str(built), "--help"]:
                    return subprocess.CompletedProcess(command, 0, stdout="NVFP4", stderr="")
                return subprocess.CompletedProcess(command, 0)

            def fake_which(name):
                return f"/usr/bin/{name}" if name in {"cmake", "ninja"} else None

            with mock.patch(
                "conversion.s2s_components.voicechat_source.shutil.which",
                side_effect=fake_which,
            ), mock.patch(
                "conversion.s2s_components.voicechat_source.subprocess.run",
                side_effect=fake_run,
            ) as run:
                actual = ensure_quantizer(llama_cpp, profile="nvfp4", repo_root=root)

            self.assertEqual(actual, built)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[0], ["bash", str(patch_script)])
            self.assertIn("-DLLAMA_BUILD_TOOLS=ON", commands[1])
            self.assertEqual(commands[2][-3:], ["--target", "llama-quantize", "--parallel"])

    def test_s2s_quantizer_auto_build_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama_cpp = Path(temporary) / "llama.cpp"
            with self.assertRaisesRegex(FileNotFoundError, "automatic build is disabled"):
                ensure_quantizer(llama_cpp, auto_build=False)

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
        self.assertIn("--llama-cpp", result.stdout)
        self.assertIn("--llama-quantize", result.stdout)
        self.assertIn("--no-build-quantizer", result.stdout)
        self.assertIn("s2s", result.stdout)

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
            llama_cpp = root / "llama.cpp"
            llama_cpp.mkdir()
            (llama_cpp / "convert_hf_to_gguf.py").touch()
            with mock.patch(
                "huggingface_hub.snapshot_download", return_value=str(snapshot)
            ) as download, mock.patch(
                "conversion.s2s_components.voicechat_source.ensure_llama_checkout",
                return_value=llama_cpp,
            ) as checkout, mock.patch(
                "conversion.registry.subprocess.run"
            ) as run:
                _convert_nmt(request, "f16")

            checkout.assert_called_once_with(ROOT / "llama.cpp", ROOT)
            download.assert_called_once_with(
                repo_id="org/model", revision="release", cache_dir=str(root / "cache")
            )
            command = run.call_args.args[0]
            self.assertEqual(Path(command[1]), llama_cpp / "convert_hf_to_gguf.py")
            self.assertEqual(Path(command[2]), snapshot)
            self.assertEqual(command[-2:], ["--outtype", "f16"])
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
