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

from conversion.base import ModelBase  # noqa: E402
from conversion.nemotron import NemotronHModel  # noqa: E402

_PREFIX = "llm."
_RENAMES = (
    ("llm.model.embed_tokens.weight", "backbone.embeddings.weight"),
    ("llm.model.norm.weight", "backbone.norm_f.weight"),
    ("llm.lm_head.weight", "lm_head.weight"),
)
_FUSED = {"gate_and_up_projs": "up_proj", "down_projs": "down_proj"}


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

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith(_PREFIX) or "_extra_state" in name:
            return None  # perception encoder and projection, TE extra-state blobs
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

    def modify_tensors(self, data_torch, name, bid):
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
