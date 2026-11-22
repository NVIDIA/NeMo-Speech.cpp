# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""llama.cpp's NemotronH converter fed from a speechlm2 SALM checkpoint (run by conversion/salm.py).

The SALM checkpoint is one safetensors file holding `perception.*` (encoder and projection) and `llm.*`
(NeMo-Automodel naming). The `llm.*` tensors are re-keyed on the fly to the HF NemotronHForCausalLM layout:

    llm.model.layers.N.mixer.*              -> backbone.layers.N.mixer.*
    llm.model.layers.N.mixer._fp32_params.X -> backbone.layers.N.mixer.X   (A_log, D, dt_bias)
    llm.model.embed_tokens.weight           -> backbone.embeddings.weight
    llm.model.norm.weight                   -> backbone.norm_f.weight
    llm.lm_head.weight                      -> lm_head.weight
    llm.mtp.layers.N.*                      -> mtp.layers.N.*

and the grouped expert weights are un-fused: Automodel stores `gate_and_up_projs [E, hidden, inter]` and
`down_projs [E, inter, hidden]` as [in, out]; GGUF wants per-expert [out, in].

A ModelOpt checkpoint (the NVFP4 checkpoint: `backbone.*` and `lm_head.*` already in the HF layout, the MTP layer as
`llm.mtp.*` with per-expert weights) keeps its weights. llama.cpp's converter writes its NVFP4 tensors as they are;
its FP8 tensors (E4M3 weight and a per-tensor `weight_scale`) become their values, exact in BF16, and a `.scale`
tensor that llama.cpp multiplies in after the matmul, instead of the dequantized product. The MTP layer's FP8 tensors
are the exception: they are written as the dequantized product in BF16, without `.scale` tensors, because llama.cpp
does not skip those when it loads the model without the MTP head.

`--nemo-speech-key KEY=VALUE` (repeatable) adds string keys this project's runtime reads (the encoder file the
decoder pairs with and the checkpoint id, conversion/salm.py); llama-quantize carries them to the quantized file.

  python salm_decoder_hf.py <llama.cpp> <staged model dir> --outfile <gguf> --outtype bf16
      [--nemo-speech-key KEY=VALUE ...] [converter flags]
"""
import sys
from pathlib import Path

# llama.cpp's own `conversion` package, not this directory's modules
_here = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _here]
_llama = Path(sys.argv.pop(1)).resolve()
sys.path[:0] = [str(_llama), str(_llama / "gguf-py")]
_KEYS = []
while "--nemo-speech-key" in sys.argv:
    _i = sys.argv.index("--nemo-speech-key")
    _key, _, _value = sys.argv[_i + 1].partition("=")
    _KEYS.append((_key, _value))
    del sys.argv[_i : _i + 2]

import convert_hf_to_gguf  # noqa: E402
import gguf  # noqa: E402
import torch  # noqa: E402

from conversion.base import LazyTorchTensor, ModelBase  # noqa: E402
from conversion.nemotron import NemotronHModel  # noqa: E402

_PREFIX = "llm."
_RENAMES = (
    ("llm.model.embed_tokens.weight", "backbone.embeddings.weight"),
    ("llm.model.norm.weight", "backbone.norm_f.weight"),
    ("llm.lm_head.weight", "lm_head.weight"),
)
_FUSED = {"gate_and_up_projs": "up_proj", "down_projs": "down_proj"}
_HF_LAYOUT = ("backbone.", "lm_head.")  # a checkpoint already in the HF NemotronH layout
_MTP = "mtp."  # the MTP layer's tensors, after the re-keying


@ModelBase.register("NemotronHForCausalLM")
class SalmNemotronHModel(NemotronHModel):
    """NemotronHModel fed from a speechlm2 SALM safetensors (llm.* Automodel naming)."""

    # Required by ModelBase.__init_subclass__; NemotronHModel.__init__ switches it to NEMOTRON_H_MOE when the
    # config carries MoE parameters.
    model_arch = gguf.MODEL_ARCH.NEMOTRON_H

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        for key, value in _KEYS:
            self.gguf_writer.add_string(key, value)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fp8_scales = {}  # FP8 weight name -> its per-tensor scale

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if "_extra_state" in name:
            return None  # TE extra-state blobs
        if name.startswith(_HF_LAYOUT):
            return super().filter_tensors(item)
        if not name.startswith(_PREFIX):
            return None  # perception encoder and projection, CTC timestamp head
        for src, dst in _RENAMES:
            if name == src:
                name = dst
                break
        else:
            if name.startswith("llm.model.layers."):
                name = "backbone.layers." + name[len("llm.model.layers.") :]
            elif name.startswith("llm.mtp.layers."):
                name = "mtp.layers." + name[len("llm.mtp.layers.") :]
            else:
                raise ValueError(f"unexpected llm tensor {name}")
            name = name.replace("._fp32_params.", ".")
        return super().filter_tensors((name, gen))

    def dequant_model(self):
        """ModelOpt FP8 weights: keep their E4M3 values (written as BF16, exact) and their per-tensor scale for a
        `.scale` tensor, instead of llama.cpp's dequantized product. The MTP layer's are dequantized here (the
        product, written as BF16; no `.scale`). NVFP4 tensors were written before this.
        """
        quantization = self.hparams.get("quantization_config") or {}
        if quantization.get("quant_method") == "modelopt":
            for scale_name in [n for n in self.model_tensors if n.endswith(".weight_scale")]:
                weight_name = scale_name.removesuffix("_scale")
                weight = self.model_tensors.get(weight_name)
                if weight is None or weight().dtype != torch.float8_e4m3fn:
                    continue
                scale = LazyTorchTensor.to_eager(self.model_tensors.pop(scale_name)()).float()
                if scale.numel() != 1:
                    raise ValueError(
                        f"{scale_name}: expected one FP8 scale per tensor, got {list(scale.shape)}"
                    )
                if weight_name.startswith(_MTP):
                    self.model_tensors[weight_name] = (
                        lambda w=weight, s=scale.item(): w().float() * s
                    )
                else:
                    self._fp8_scales[weight_name] = scale.reshape(1)
                    self.model_tensors[weight_name] = lambda w=weight: w().float()
                self.model_tensors.pop(weight_name.removesuffix(".weight") + ".input_scale", None)
        super().dequant_model()

    def modify_tensors(self, data_torch, name, bid):
        written = []
        for new_name, data in self._modify_tensors(data_torch, name, bid):
            written.append(new_name)
            yield new_name, data
        scale = self._fp8_scales.pop(name, None)
        if scale is not None:
            if len(written) != 1:  # stacked experts would need one scale per expert
                raise ValueError(
                    f"{name}: an FP8 weight outside the MTP layer that is not written as one tensor"
                )
            self._write_scale_tensor(written[0].removesuffix(".weight") + ".scale", scale)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._fp8_scales:
            raise ValueError(f"FP8 scales without their weights: {list(self._fp8_scales)}")

    def _modify_tensors(self, data_torch, name, bid):
        leaf = name.rsplit(".", 1)[-1]
        if ".mixer.experts." in name and leaf in _FUSED:
            if name.startswith("mtp.layers."):
                assert self._mtp_bid is not None, "MTP expert tensor with MTP disabled"
                bid = self._mtp_bid
            assert bid is not None, name
            n_experts = self.hparams["n_routed_experts"]
            assert data_torch.shape[0] == n_experts, (name, tuple(data_torch.shape))
            # [E, in, out] -> [E, out, in], as torch.stack of per-expert nn.Linear weights
            stacked = data_torch.permute(0, 2, 1).contiguous()
            merged = f"model.layers.{bid}.mlp.experts.{_FUSED[leaf]}.weight"
            yield from ModelBase.modify_tensors(self, stacked, merged, bid)
            return
        yield from super().modify_tensors(data_torch, name, bid)


if __name__ == "__main__":
    convert_hf_to_gguf.main()
