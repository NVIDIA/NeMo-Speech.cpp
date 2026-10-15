from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
PATTERN = "MEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
HIDDEN = 1536
EXPECTED_PREFIXES = {
    "audio_embeddings": 16,
    "decoder": 885,
    "local_transformer": 19,
    "local_transformer_out_projections": 32,
    "phoneme_embeddings": 1,
    "phoneme_final_proj": 2,
    "text_embedding": 1,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_header(path: Path) -> tuple[dict, int, str]:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 32 << 20:
            raise ValueError("unreasonable safetensors header size")
        payload = f.read(length)
    if len(payload) != length:
        raise ValueError("truncated safetensors header")
    header = json.loads(payload)
    size = path.stat().st_size - length - 8
    spans = []
    sizes = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "BOOL": 1}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        shape = spec["shape"]
        if not all(type(v) is int and v >= 0 for v in shape):
            raise ValueError(f"invalid tensor shape: {name}")
        (start, end) = spec["data_offsets"]
        if not 0 <= start <= end <= size or end - start != math.prod(shape) * sizes[spec["dtype"]]:
            raise ValueError(f"invalid tensor extent: {name}")
        spans.append((start, end))
    spans.sort()
    if spans and (
        spans[0][0] != 0
        or spans[-1][1] != size
        or any(a[1] != b[0] for (a, b) in zip(spans, spans[1:], strict=False))
    ):
        raise ValueError("safetensors has a gap or overlapping payloads")
    return (header, length + 8, hashlib.sha256(prefix + payload).hexdigest())


@dataclass(frozen=True)
class WeightSpec:
    family: str
    field: str
    index: int | None
    sources: tuple[str, ...]
    source_shapes: tuple[tuple[int, ...], ...]
    shape: tuple[int, ...]
    dtype: str
    transform: str = "checkpoint_to_bf16"


def weight_specs() -> list[WeightSpec]:
    specs = []

    def add(
        family,
        field,
        index,
        source,
        shape,
        dtype="bfloat16",
        transform="checkpoint_to_bf16",
        source_shape=None,
    ):
        specs.append(
            WeightSpec(
                family,
                field,
                index,
                (source,),
                (tuple(source_shape or shape),),
                tuple(shape),
                dtype,
                transform,
            )
        )

    for i in range(3):
        p = f"local_transformer.layers.{i}."
        for field, suffix, shape, src_shape in (
            ("qkv", "self_attention.qkv_net.weight", (4608, 1536), None),
            ("o", "self_attention.o_net.weight", (1536, 1536), None),
            ("ff_in", "pos_ff.proj.conv.weight", (6144, 1536), (6144, 1536, 1)),
            ("ff_out", "pos_ff.o_net.conv.weight", (1536, 6144), (1536, 6144, 1)),
        ):
            add("predictor", field, i, p + suffix, shape, source_shape=src_shape)
        for field, suffix in (("norm_self", "norm_self.weight"), ("norm_ff", "norm_pos_ff.weight")):
            add("predictor", field, i, p + suffix, (1536,), "float32", "checkpoint_to_fp32")
    add("predictor", "pos_emb", None, "local_transformer.position_embeddings.weight", (18, 1536))
    for i in range(16):
        add("predictor", "audio_emb", i, f"audio_embeddings.{i}.weight", (1032, 1536))
        add("predictor", "head_w", i, f"local_transformer_out_projections.{i}.weight", (1032, 1536))
        add("predictor", "head_b", i, f"local_transformer_out_projections.{i}.bias", (1032,))
    counts = {"M": 0, "E": 0, "*": 0}
    for layer, kind in enumerate(PATTERN):
        p = f"decoder.layers.{layer}."
        add("backbone", "norm", layer, p + "norm.weight", (1536,))
        p += "mixer."
        i = counts[kind]
        counts[kind] += 1
        if kind == "M":
            for field, suffix, shape, src_shape in (
                ("mb_in_proj", "in_proj.weight", (5184, 1536), None),
                ("mb_out_proj", "out_proj.weight", (1536, 1536), None),
                ("mb_conv_w", "conv1d.weight", (3584, 4), (3584, 1, 4)),
                ("mb_conv_b", "conv1d.bias", (3584,), None),
                ("mb_gnorm", "norm.weight", (1536,), None),
            ):
                add("backbone", field, i, p + suffix, shape, source_shape=src_shape)
            add("backbone", "mb_A", i, p + "A_log", (64,), "float32", "negative_exp_fp32")
            for field, suffix in (("mb_D", "D"), ("mb_dt_bias", "dt_bias")):
                add("backbone", field, i, p + suffix, (64,), "float32", "checkpoint_to_fp32")
        elif kind == "*":
            specs.append(
                WeightSpec(
                    "backbone",
                    "at_qkv",
                    i,
                    tuple(p + x + "_proj.weight" for x in "qkv"),
                    ((1536, 1536), (512, 1536), (512, 1536)),
                    (2560, 1536),
                    "bfloat16",
                    "concat_qkv_checkpoint_to_bf16",
                )
            )
            add("backbone", "at_o", i, p + "o_proj.weight", (1536, 1536))
        else:
            for field, suffix, shape in (
                ("moe_exp_up", "up_proj.weight", (768, 1536)),
                ("moe_exp_down", "down_proj.weight", (1536, 768)),
            ):
                specs.append(
                    WeightSpec(
                        "backbone",
                        field,
                        i,
                        tuple(p + f"experts.{e}." + suffix for e in range(24)),
                        (shape,) * 24,
                        (24, *shape),
                        "bfloat16",
                        "stack_numeric_expert_order_checkpoint_to_bf16",
                    )
                )
            for field, suffix, shape in (
                ("moe_sh_up", "shared_experts.up_proj.weight", (2048, 1536)),
                ("moe_sh_down", "shared_experts.down_proj.weight", (1536, 2048)),
                ("moe_gate", "gate.weight", (24, 1536)),
                ("moe_gate_bias", "gate.e_score_correction_bias", (24,)),
            ):
                add("backbone", field, i, p + suffix, shape)
    add("backbone", "norm_f", None, "decoder.norm_f.weight", (1536,))
    add("backbone", "phoneme_w", None, "phoneme_final_proj.weight", (2051, 1536))
    add("backbone", "phoneme_b", None, "phoneme_final_proj.bias", (2051,))
    add(
        "input",
        "text_embedding",
        None,
        "text_embedding.weight",
        (131075, 1536),
        "bfloat16",
        "checkpoint_to_bf16",
    )
    add(
        "input",
        "phoneme_embedding",
        None,
        "phoneme_embeddings.0.weight",
        (2051, 1536),
        "bfloat16",
        "checkpoint_to_bf16",
    )
    add("unused", "backbone_token_embedding", None, "decoder.embeddings.weight", (2, 1536))
    return specs


class ModelAssets:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "config.json").read_text())
        expected = {
            "hidden_size": 1536,
            "num_hidden_layers": 31,
            "num_attention_heads": 12,
            "torch_dtype": "bfloat16",
            "num_key_value_heads": 4,
            "head_dim": 128,
            "mamba_num_heads": 64,
            "mamba_head_dim": 24,
            "ssm_state_size": 128,
            "n_groups": 8,
            "conv_kernel": 4,
            "n_routed_experts": 24,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 768,
            "moe_shared_expert_intermediate_size": 2048,
            "hybrid_override_pattern": PATTERN,
            "text_eos_id": 131073,
            "text_vocab_size": 131075,
            "num_task_embeddings": 0,
            "streaming_speech_delay": 5,
            "streaming_phonemes_delay": 3,
            "phoneme_bos_id": 2048,
            "phoneme_eos_id": 2049,
            "forced_audio_bos_id": 1024,
            "forced_audio_eos_id": 1025,
            "local_transformer_n_layers": 3,
            "local_transformer_n_heads": 12,
            "local_transformer_hidden_dim": 1536,
            "audio_embedding_dim": 1536,
            "embedding_dim": 1536,
            "num_audio_codebooks": 8,
            "frame_stacking_factor": 2,
            "codebook_size": 1024,
            "phoneme_vocab_size": 2051,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "n_group": 1,
            "topk_group": 1,
            "layer_norm_epsilon": 1e-05,
        }
        drift = {
            k: (self.config.get(k), v) for (k, v) in expected.items() if self.config.get(k) != v
        }
        if drift:
            raise ValueError(f"unsupported model configuration: {drift}")
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        from collections import Counter

        if dict(Counter(k.split(".")[0] for k in self.weight_map)) != EXPECTED_PREFIXES:
            raise ValueError("checkpoint prefix inventory drift")
        self.headers = {}
        self.shards = {}
        for name in sorted(set(self.weight_map.values())):
            p = self.root / name
            if Path(name).name != name or not p.is_file():
                raise ValueError("checkpoint shard must be a local regular filename")
            (header, offset, digest) = read_header(p)
            self.shards[name] = {
                "path": str(p),
                "bytes": p.stat().st_size,
                "safetensors_header_bytes": offset,
                "safetensors_header_sha256": digest,
            }
            for key, value in header.items():
                if key != "__metadata__":
                    if key in self.headers or self.weight_map.get(key) != name:
                        raise ValueError("checkpoint index/header disagreement")
                    self.headers[key] = value
        if set(self.headers) != set(self.weight_map):
            raise ValueError("checkpoint index/header tensor set drift")
        self.specs = weight_specs()
        used = set()
        for spec in self.specs:
            for key, shape in zip(spec.sources, spec.source_shapes, strict=False):
                actual = self.headers.get(key)
                if actual is None or tuple(actual["shape"]) != shape or actual["dtype"] != "BF16":
                    raise ValueError(f"checkpoint tensor shape/dtype drift: {key}")
                used.add(key)
        if used != set(self.headers):
            raise ValueError(f"unmapped checkpoint tensors: {sorted(set(self.headers) - used)}")

    def inventory(self, *, hash_weights=False):
        from dataclasses import asdict

        shards = [dict(v) for v in self.shards.values()]
        if hash_weights:
            for shard in shards:
                shard["sha256"] = sha256(Path(shard["path"]))
        small = {}
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "speaker_embeddings/english_sample.pt",
            "codec_native/config.json",
        ):
            p = self.root / name
            small[name] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
        p = self.root / "codec_native/model.safetensors"
        (h, offset, digest) = read_header(p)
        codec = {
            "path": str(p),
            "bytes": p.stat().st_size,
            "tensor_count": len(h) - int("__metadata__" in h),
            "safetensors_header_bytes": offset,
            "safetensors_header_sha256": digest,
        }
        if hash_weights:
            codec["sha256"] = sha256(p)
        return {
            "schema": "native-three-mk.assets.v2",
            "status": "PASS_CPU_ASSET_INVENTORY",
            "root": str(self.root),
            "tensor_count": len(self.headers),
            "checkpoint_dtype": "BF16",
            "effective_input_dtype": "BF16",
            "backbone_abi_dtype": "BF16 with FP32 A/D/dt_bias",
            "constructor_lineage": "BF16Max64V1 -> BF16V1 explicitly bypasses older V3 FP16 "
            "constructor; "
            "inherited forward methods run on BF16 tensors",
            "fp16_weight_roundtrip": False,
            "payload_hashes_measured": bool(hash_weights),
            "assets": small,
            "model_shards": shards,
            "codec_weights": codec,
            "weight_mapping": [asdict(s) for s in self.specs],
            "scope": "Local headers/configuration verified; no GPU execution or model-output "
            "qualification.",
        }

    def read_tensor(self, key, *, device="cpu"):
        from safetensors import safe_open

        with safe_open(str(self.root / self.weight_map[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key).to(device=device)

    def inputs(self, *, device="cpu"):
        import torch

        keys = ["text_embedding.weight", "phoneme_embeddings.0.weight"]
        keys += [f"audio_embeddings.{i}.weight" for i in range(16)]
        values = {
            k: self.read_tensor(k).to(device=device, dtype=torch.bfloat16).contiguous()
            for k in keys
        }
        payload = torch.load(
            self.root / "speaker_embeddings/english_sample.pt",
            map_location="cpu",
            weights_only=True,
        )
        speaker = payload.get("speaker_encoding") if isinstance(payload, dict) else payload
        if (
            not isinstance(speaker, torch.Tensor)
            or tuple(speaker.shape) != (64, 1536)
            or speaker.dtype != torch.float32
        ):
            raise ValueError("english_sample must be FP32[64,1536]")
        values["speaker"] = speaker.to(device=device, dtype=torch.bfloat16).contiguous()
        return values

    def materialize_weights(self, *, device="cpu"):
        import torch

        result = {}
        for spec in self.specs:
            if spec.family in {"input", "unused"}:
                continue
            values = [self.read_tensor(k) for k in spec.sources]
            if spec.transform.startswith("stack_"):
                value = torch.stack(values)
            elif spec.transform.startswith("concat_"):
                value = torch.cat(values, dim=0)
            else:
                value = values[0]
            if spec.transform == "negative_exp_fp32":
                value = value.float().exp().neg()
            value = (
                value.reshape(spec.shape)
                .to(dtype=getattr(torch, spec.dtype), device=device)
                .contiguous()
            )
            result[spec.family, spec.field, spec.index] = value
        return result


class PredictorWeights(C.Structure):
    _fields_ = (
        [(x, C.c_void_p * 3) for x in ("qkv", "o", "ff_in", "ff_out", "norm_self", "norm_ff")]
        + [("pos_emb", C.c_void_p)]
        + [(x, C.c_void_p * 16) for x in ("audio_emb", "head_w", "head_b")]
    )


class BackboneWeights(C.Structure):
    _fields_ = (
        [("norm", C.c_void_p * 31), ("norm_f", C.c_void_p)]
        + [
            (x, C.c_void_p * 14)
            for x in (
                "mb_in_proj",
                "mb_out_proj",
                "mb_conv_w",
                "mb_conv_b",
                "mb_A",
                "mb_D",
                "mb_dt_bias",
                "mb_gnorm",
            )
        ]
        + [(x, C.c_void_p * 3) for x in ("at_qkv", "at_o")]
        + [
            (x, C.c_void_p * 14)
            for x in (
                "moe_exp_up",
                "moe_exp_down",
                "moe_sh_up",
                "moe_sh_down",
                "moe_gate",
                "moe_gate_bias",
            )
        ]
        + [("phoneme_w", C.c_void_p), ("phoneme_b", C.c_void_p), ("pattern", C.c_char_p)]
    )


class Stage0Bindings(C.Structure):
    _fields_ = [
        ("conv_state", C.c_void_p * 14),
        ("conv_slot_stride", C.c_int * 14),
        ("ssm_state", C.c_void_p * 14),
        ("ssm_slot_stride", C.c_int * 14),
        ("slot_map", C.c_void_p),
        ("kv_pool_base", C.c_void_p * 3),
        ("kv_block_table", C.c_void_p * 3),
        ("kv_max_blocks_per_seq", C.c_int),
        ("kv_tokens_per_block", C.c_int),
        ("seq_len", C.c_void_p),
        ("embedding", C.c_void_p),
        ("gumbel", C.c_void_p),
        ("temperature", C.c_float),
        ("moe_w_inject_debug", C.c_void_p),
        ("codes_out", C.c_void_p),
        ("phoneme_out", C.c_void_p),
    ]


def weight_bindings(owned_tensors):
    objects = {"predictor": PredictorWeights(), "backbone": BackboneWeights()}
    objects["backbone"].pattern = PATTERN.encode("ascii")
    for spec in weight_specs():
        if spec.family in {"input", "unused"}:
            continue
        tensor = owned_tensors[spec.family, spec.field, spec.index]
        if (
            str(tensor.device) != "cuda:0"
            or tuple(tensor.shape) != spec.shape
            or (not tensor.is_contiguous())
        ):
            raise ValueError(f"invalid device weight binding: {spec.field}")
        if str(tensor.dtype) != "torch." + spec.dtype:
            raise ValueError(f"invalid weight dtype: {spec.field}")
        obj = objects[spec.family]
        if spec.index is None:
            setattr(obj, spec.field, tensor.data_ptr())
        else:
            getattr(obj, spec.field)[spec.index] = tensor.data_ptr()
    return (objects["predictor"], objects["backbone"])
