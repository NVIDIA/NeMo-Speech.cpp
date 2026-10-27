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
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from gguf import GGMLQuantizationType, GGUFReader
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conversion.asr import _pick_dtype
from conversion.asr import convert as convert_asr
from conversion.asr import reshape_for_emission
from conversion.diarization import is_high_resolution_v3
from conversion.diarization import remap as remap_diarization
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
from conversion.s2s_components.eartts_backbone import _extract_backbone_weights
from conversion.s2s_components.perception import _preserve_subsampling_precision
from conversion.s2s_components.voicechat_source import (
    build_character_tables,
    default_quantizer_path,
    ensure_llama_checkout,
    ensure_quantizer,
    find_quantizer,
    load_llm_channel_weights,
)
from conversion.salm import DENSE_TYPE as SALM_DENSE_TYPE
from conversion.salm import checkpoint_id as salm_checkpoint_id
from conversion.salm import decoder_keys as salm_decoder_keys
from conversion.salm import encoder_file_name as salm_encoder_file_name
from conversion.salm import encoder_type as salm_encoder_type
from conversion.salm import stage as stage_salm
from conversion.salm import tensor_rules as salm_tensor_rules
from conversion.source import extract_archive


class ConverterContractTest(unittest.TestCase):
    def test_streaming_rnnt_q8_preserves_pointwise_precision(self) -> None:
        for layer in (0, 23):
            for conv in (1, 2):
                name = f"encoder.layers.{layer}.conv.pointwise_conv{conv}.weight"
                shape = reshape_for_emission(name, np.zeros((64, 32, 1))).shape
                self.assertEqual(shape, (64, 32))
                dtype, reason = _pick_dtype(
                    name,
                    shape,
                    GGMLQuantizationType.Q8_0,
                    GGMLQuantizationType.F32,
                    cache_aware_rnnt=True,
                )
                self.assertEqual(dtype, GGMLQuantizationType.F16)
                self.assertEqual(reason, "cache-aware RNNT pointwise precision")
                # Offline/CTC and other export precisions retain their policy.
                dtype, _ = _pick_dtype(
                    name, shape, GGMLQuantizationType.Q8_0, GGMLQuantizationType.F32
                )
                self.assertEqual(dtype, GGMLQuantizationType.Q8_0)
                for requested in (GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
                    dtype, _ = _pick_dtype(
                        name,
                        shape,
                        requested,
                        GGMLQuantizationType.F32,
                        cache_aware_rnnt=True,
                    )
                    self.assertEqual(dtype, requested)

    def test_streaming_rnnt_precision_change_is_limited_to_pointwise(self) -> None:
        expected = {
            "encoder.layers.0.self_attn.linear_q.weight": GGMLQuantizationType.Q8_0,
            "encoder.layers.0.feed_forward1.linear1.weight": GGMLQuantizationType.Q8_0,
            "decoder.prediction.dec_rnn.lstm.ih_l0.weight": GGMLQuantizationType.Q8_0,
            "encoder.layers.0.conv.depthwise_conv.weight": GGMLQuantizationType.F16,
            "encoder.pre_encode.conv.0.weight": GGMLQuantizationType.F16,
            "encoder.layers.0.conv.pointwise_conv1.bias": GGMLQuantizationType.F32,
        }
        for name, want in expected.items():
            dtype, reason = _pick_dtype(
                name,
                (64, 32),
                GGMLQuantizationType.Q8_0,
                GGMLQuantizationType.F32,
                cache_aware_rnnt=True,
            )
            self.assertEqual(dtype, want, name)
            self.assertIsNone(reason)

    def test_streaming_rnnt_export_keeps_original_pointwise_values(self) -> None:
        name = "encoder.layers.0.conv.pointwise_conv1.weight"
        linear = "encoder.layers.0.feed_forward1.linear1.weight"
        values = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32, 1) / 337
        config = {
            "encoder": {
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 4,
                "conv_kernel_size": 3,
                "feat_in": 128,
                "pos_emb_max_len": 32,
                "att_context_size": [[70, 3]],
                "att_context_style": "chunked_limited",
            },
            "decoder": {"vocab_size": 32},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save({name: values, linear: values.squeeze(-1)}, root / "model_weights.ckpt")
            for streaming in (True, False):
                config["encoder"]["att_context_style"] = (
                    "chunked_limited" if streaming else "regular"
                )
                (root / "model_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
                output = root / f"model-{streaming}.gguf"
                with redirect_stdout(io.StringIO()):
                    convert_asr(root, output, "rnnt", "q8_0")
                tensors = {tensor.name: tensor for tensor in GGUFReader(output).tensors}
                self.assertEqual(tensors[linear].tensor_type, GGMLQuantizationType.Q8_0)
                want = GGMLQuantizationType.F16 if streaming else GGMLQuantizationType.Q8_0
                self.assertEqual(tensors[name].tensor_type, want)
                if streaming:
                    np.testing.assert_array_equal(
                        tensors[name].data, values.squeeze(-1).half().numpy()
                    )

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
            "pee": {
                "target": "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
                "asr_encoder_cfg": {},
            },
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

            salm = root / "salm"
            salm.mkdir()
            (salm / "config.json").write_text(
                json.dumps(
                    {"pe_encoder_config": {"asr_encoder_type": "transformer"}, "llm_config": {}}
                ),
                encoding="utf-8",
            )
            (salm / "model.safetensors").write_bytes(b"")
            actual, resolved = detect_architecture(
                ConversionRequest(source=str(salm), outfile=root / "salm.gguf")
            )
            self.assertEqual(actual, "pee")
            self.assertEqual(resolved, salm)

            nmt = root / "nmt"
            nmt.mkdir()
            (nmt / "config.json").write_text("{}\n", encoding="utf-8")
            actual, resolved = detect_architecture(
                ConversionRequest(source=str(nmt), outfile=root / "nmt.gguf")
            )
            self.assertEqual(actual, "nmt")
            self.assertIsNone(resolved)

            drafter = root / "dflash2"
            drafter.mkdir()
            (drafter / "config.json").write_text(
                json.dumps({"architectures": ["Qwen3DFlash2DraftModel"]}), encoding="utf-8"
            )
            (drafter / "model.safetensors").write_bytes(b"")
            actual, resolved = detect_architecture(
                ConversionRequest(source=str(drafter), outfile=root / "drafter.gguf")
            )
            self.assertEqual(actual, "dflash2")
            self.assertEqual(resolved, drafter.resolve())

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
        self.assertEqual(_normalized_outtype("dflash2", "auto"), "bf16")
        with self.assertRaises(ValueError):  # the drafter is not quantized
            _normalized_outtype("dflash2", "q8_0")
        self.assertEqual(_normalized_outtype("diarization", "auto"), "f32")
        self.assertEqual(_normalized_outtype("tts", "fp16"), "f16")
        self.assertEqual(_normalized_outtype("tts", "q8_0"), "q8_0")
        self.assertEqual(_normalized_outtype("tts", "auto"), "q8_0")
        self.assertEqual(_normalized_outtype("codec", "auto"), "f16")
        self.assertEqual(_normalized_outtype("s2s", "auto"), "q4_k_m")
        self.assertEqual(_normalized_outtype("s2s", "q4"), "q4_k_m")
        self.assertEqual(_normalized_outtype("s2s", "nvfp4"), "nvfp4")
        with self.assertRaises(ValueError):
            _normalized_outtype("vad", "q8_0")
        with self.assertRaises(ValueError):
            _normalized_outtype("s2s", "q8_0")
        self.assertEqual(_normalized_outtype("salm", "auto"), "nvfp4")
        self.assertEqual(_normalized_outtype("salm", "IQ4_NL"), "iq4_nl")
        for unsupported in ("bf16", "q4_k_m"):
            with self.assertRaises(ValueError):
                _normalized_outtype("salm", unsupported)

    def test_salm_quantization_rules(self) -> None:
        # first match wins: experts, then dense linears, then everything else BF16
        self.assertEqual(salm_tensor_rules("nvfp4"), ["ffn_(up|down|gate)_exps=nvfp4", ".=bf16"])
        self.assertEqual(
            salm_tensor_rules("iq4_nl", "q8_0"),
            [
                "ffn_(up|down|gate)_exps=iq4_nl",
                "ssm_(in|out)=q8_0",
                "ffn_(up|down)_shexp=q8_0",
                "attn_(q|k|v|output)=q8_0",
                ".=bf16",
            ],
        )
        # dense linears: Q8_0 with IQ4_NL experts, BF16 with NVFP4 experts
        self.assertEqual(SALM_DENSE_TYPE, {"nvfp4": "bf16", "iq4_nl": "q8_0"})
        # one F16 encoder file serves every profile; "none" writes the decoder alone
        self.assertEqual(salm_encoder_type("nvfp4"), "f16")
        self.assertEqual(salm_encoder_type("iq4_nl"), "f16")
        self.assertEqual(salm_encoder_type("iq4_nl", "none"), "none")
        self.assertEqual(
            salm_encoder_file_name("example-salm", "f16"),
            "example-salm-Encoder-F16.gguf",
        )
        # every decoder names its encoder file, also one converted with --salm-encoder none
        self.assertEqual(
            salm_decoder_keys("example-salm", "0123"),
            [
                "nemo_speech.checkpoint_id=0123",
                "nemo_speech.speech_model=example-salm-Encoder-F16.gguf",
            ],
        )

    def test_salm_checkpoint_id(self) -> None:
        import numpy as np

        def safetensors(path, tensors):
            header, data = {}, b""
            for name, array in tensors.items():
                raw = array.tobytes()
                header[name] = {
                    "dtype": "F32",
                    "shape": list(array.shape),
                    "data_offsets": [len(data), len(data) + len(raw)],
                }
                data += raw
            text = json.dumps(header).encode()
            path.write_bytes(len(text).to_bytes(8, "little") + text + data)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a", "b", "c"):
                (root / name).mkdir()
            big = np.arange(4096, dtype=np.float32)
            safetensors(root / "a" / "model.safetensors", {"w": big, "v": np.ones(3, np.float32)})
            safetensors(root / "b" / "model.safetensors", {"w": big, "v": np.ones(3, np.float32)})
            changed = big.copy()
            changed[-1] = -1  # in the last 4 KiB
            safetensors(
                root / "c" / "model.safetensors", {"w": changed, "v": np.ones(3, np.float32)}
            )
            ids = [salm_checkpoint_id(root / name) for name in ("a", "b", "c")]
            self.assertEqual(ids[0], ids[1])
            self.assertNotEqual(ids[0], ids[2])
            self.assertEqual(len(ids[0]), 32)

    def test_salm_encoder_file(self) -> None:
        sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
        import gguf
        import numpy as np

        from conversion.salm_encoder import CHECKPOINT_ID_KEY, COMPONENTS_KEY, write_encoder_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def part(name, arch, tensors, keys):
                writer = gguf.GGUFWriter(root / name, arch=arch)
                writer.add_name(name)
                for key, value in keys.items():
                    writer.add_uint32(key, value)
                for tensor in tensors:
                    writer.add_tensor(tensor, np.arange(4, dtype=np.float32))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                return root / name

            decoder = part(
                "d.gguf", "nemotron_h_moe", ["token_embd.weight"], {"nemotron_h_moe.block_count": 1}
            )
            encoder = part("e.gguf", "pee", ["asr.w"], {"pee.asr.n_layers": 2})
            ctc = part("c.gguf", "ctc_timestamp", ["ctc.classifier.weight"], {"ctc.vocab_size": 3})
            out = root / "encoder.gguf"
            write_encoder_file(encoder, ctc, out, checkpoint_id="0123")
            reader = gguf.GGUFReader(out)
            fields = reader.fields
            self.assertEqual(fields["general.architecture"].contents(), "pee")
            self.assertEqual(fields["general.name"].contents(), "e.gguf")
            self.assertEqual(fields[COMPONENTS_KEY].contents(), ["pee", "ctc_timestamp"])
            self.assertEqual(fields[CHECKPOINT_ID_KEY].contents(), "0123")
            self.assertEqual(fields["pee.asr.n_layers"].contents(), 2)
            self.assertEqual(fields["ctc.vocab_size"].contents(), 3)
            self.assertEqual([t.name for t in reader.tensors], ["asr.w", "ctc.classifier.weight"])
            self.assertEqual(reader.tensors[1].data.tolist(), [0.0, 1.0, 2.0, 3.0])
            with self.assertRaises(ValueError):  # an encoder that is not one
                write_encoder_file(ctc, None, root / "bad.gguf")
            with self.assertRaises(ValueError):  # a decoder is not a part of the encoder file
                write_encoder_file(encoder, decoder, root / "bad.gguf")

    def test_salm_decoder_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "salm"
            (source / "llm_backbone").mkdir(parents=True)
            backbone = {"architectures": ["NemotronHForCausalLM"], "vocab_size": 131072}
            (source / "config.json").write_text(
                json.dumps({"pe_encoder_config": {}, "llm_config": {"architectures": ["Other"]}})
            )
            (source / "llm_backbone" / "config.json").write_text(json.dumps(backbone))
            audio = {"id": 131072, "content": "<|audio|>", "special": True}
            (source / "tokenizer.json").write_text(
                json.dumps(
                    {"added_tokens": [{"id": 11, "content": "<|im_end|>"}, audio], "model": {}}
                )
            )
            (source / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "audio_token": "<|audio|>",
                        "extra_special_tokens": {"audio_token": "<|audio|>"},
                        "model_specific_special_tokens": {"audio_token": "<|audio|>"},
                        "added_tokens_decoder": {"131072": audio, "11": {"content": "<|im_end|>"}},
                    }
                )
            )
            (source / "model.safetensors").write_bytes(b"weights")
            staging = root / "staged"
            stage_salm(source, staging)
            self.assertEqual(json.loads((staging / "config.json").read_text()), backbone)
            tokens = json.loads((staging / "tokenizer.json").read_text())["added_tokens"]
            self.assertEqual([t["content"] for t in tokens], ["<|im_end|>"])
            tokenizer_config = json.loads((staging / "tokenizer_config.json").read_text())
            self.assertNotIn("audio_token", tokenizer_config)
            self.assertNotIn("model_specific_special_tokens", tokenizer_config)
            self.assertEqual(tokenizer_config["extra_special_tokens"], {})
            self.assertEqual(list(tokenizer_config["added_tokens_decoder"]), ["11"])
            self.assertTrue((staging / "model.safetensors").is_symlink())
            self.assertEqual((staging / "model.safetensors").read_bytes(), b"weights")
            # a drafter's target: the configs and tokenizer without the weights
            stage_salm(source, root / "target", weights=False)
            self.assertFalse((root / "target" / "model.safetensors").exists())
            self.assertEqual(json.loads((root / "target" / "config.json").read_text()), backbone)
            # without llm_backbone/, the SALM config's llm_config is the backbone config
            (source / "llm_backbone" / "config.json").unlink()
            (source / "llm_backbone").rmdir()
            with self.assertRaises(RuntimeError):  # llm_config names an unsupported backbone
                stage_salm(source, staging)

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

    def test_s2s_character_tables_support_sparse_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = Path(temporary) / "tokenizer.json"
            tokenizer.write_text(
                json.dumps({"model": {"vocab": {"a": 0, "bc": 3}}}),
                encoding="utf-8",
            )

            ids, mask, _ = build_character_tables(tokenizer)

            self.assertEqual(ids.shape[0], 5)
            self.assertEqual(mask.shape, ids.shape)
            self.assertEqual(mask[4, 0], 1.0)

    def test_eartts_dummy_embedding_uses_configured_hidden_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            save_file(
                {"model.backbone.layers.0.weight": torch.zeros((2, 7))},
                str(source / "model.safetensors"),
            )

            weights = _extract_backbone_weights(source, hidden_size=11)

            self.assertEqual(tuple(weights["model.embed_tokens.weight"].shape), (1, 11))

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
            materialize = root / "cmake" / "llama_cpp.cmake"
            materialize.parent.mkdir()
            materialize.touch()
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
            self.assertEqual(commands[0][-2:], ["-P", str(materialize)])
            self.assertIn(f"-DDEST_DIR={root / '.deps' / 'llama.cpp-patched'}", commands[0])
            self.assertEqual(
                commands[1][commands[1].index("-S") + 1], str(root / ".deps" / "llama.cpp-patched")
            )
            self.assertIn("-DLLAMA_BUILD_TOOLS=ON", commands[1])
            self.assertEqual(commands[2][-3:], ["--target", "llama-quantize", "--parallel"])

    def test_s2s_quantizer_auto_build_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama_cpp = Path(temporary) / "llama.cpp"
            with self.assertRaisesRegex(FileNotFoundError, "automatic build is disabled"):
                ensure_quantizer(llama_cpp, auto_build=False)

    def test_sortformer_v2_v3_contract_detection_and_remapping(self) -> None:
        v2 = {
            "encoder": {
                "self_attention_model": "rel_pos",
                "subsampling": "dw_striding",
            }
        }
        v3 = {
            "high_resolution": True,
            "encoder": {
                "self_attention_model": "rope",
                "subsampling": "feature_stacking",
            },
        }
        self.assertFalse(is_high_resolution_v3(v2))
        self.assertTrue(is_high_resolution_v3(v3))
        self.assertFalse(is_high_resolution_v3({**v3, "high_resolution": False}))
        self.assertEqual(
            remap_diarization("sortformer_modules.subpixel_upsample.weight"),
            "subpixel_upsample.weight",
        )
        self.assertEqual(
            remap_diarization("sortformer_modules.learnable_sil_emb"),
            "learnable_sil_emb",
        )
        self.assertIsNone(remap_diarization("sortformer_modules.hidden_to_spks.weight"))

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
