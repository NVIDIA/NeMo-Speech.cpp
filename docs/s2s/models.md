# VoiceChat model conversion

The VoiceChat runtime uses
[nvidia/NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B).
The unified Python converter produces the complete GGUF directory expected by
the realtime server.

## Convert the public checkpoint

Run conversion from a NeMo-Speech.cpp source checkout. A quantized conversion
requires Git, CMake, Ninja or Make, and a C++ compiler in addition to Python.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python convert_model.py \
  nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --outfile models/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF
```

If the model repository requires authentication, accept its terms and run
`hf auth login` before conversion. Downloads use the standard Hugging Face
cache, so rerunning conversion does not download unchanged files again. A
downloaded model directory can be supplied instead of the repository ID.

Architecture detection selects S2S automatically. The converter initializes
the pinned llama.cpp submodule when needed, applies the repository's patches,
and builds or reuses `llama-quantize` for quantized profiles.

## Precision profiles

`--outtype` selects the large conversational LLM format. Perception remains
Q8_0, and the smaller components use fixed formats in every profile. NVFP4
deliberately keeps EarTTS in Q4_K_M because the small speech-critical backbone
benefits little from the additional size reduction.

| `--outtype` | Perception | LLM backbone | EarTTS backbone | Use |
|---|---|---|---|---|
| `q4_k_m` (default) | Q8_0 | Q4_K_M | Q4_K_M | Recommended portable profile |
| `nvfp4` | Q8_0 | NVFP4 | Q4_K_M | Smaller conversational LLM on supported NVIDIA GPUs |
| `bf16` | Q8_0 | BF16 | BF16 | Backbone reference profile |

For example, create the NVFP4 bundle with:

```bash
python convert_model.py \
  nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --outfile models/NVIDIA-NemotronLabs-VoiceChat-11B-NVFP4-GGUF \
  --outtype nvfp4
```

Q4_K_M and NVFP4 require `llama-quantize`. Use `--llama-quantize PATH` to
select an existing compatible binary, or `--no-build-quantizer` to disable
automatic provisioning.

## Output layout

```text
NVIDIA-NemotronLabs-VoiceChat-11B-GGUF/
├── perception.gguf
├── llm_aux.gguf
├── codec.gguf
├── manifest.json
├── eartts_vllm/
│   ├── eartts_gemma3.gguf
│   ├── eartts_side.gguf
│   └── tts_prompt.gguf
├── nano-v2-vllm/
│   └── nano-v2-llm.gguf
└── rnnt_tokenizer/
    └── vocab.json
```

The source checkpoint is never modified. Existing output artifacts are
protected unless `--force` is supplied. `--dry-run` validates the source and
prints the conversion plan without writing GGUF files:

```bash
python convert_model.py \
  nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --outfile models/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF \
  --dry-run
```

Use `--revision` to pin a Hugging Face branch, tag, or commit, and `--cache-dir`
to override the download cache. Run `python convert_model.py --help` for every
available option.
