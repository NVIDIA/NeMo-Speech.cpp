# Model conversion

`convert_model.py` is the unified entry point for every supported model family.
Published NeMo-Speech.cpp models often provide ready-to-run GGUF files; use
conversion for compatible custom checkpoints, a different supported precision,
or supporting models that do not publish a GGUF.

The converters are Python source tools and are not included in the native
release archives; the C++ runtime itself does not require Python.

## Quick start

Run the converter from a source checkout in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python convert_model.py SOURCE --outfile OUTPUT
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

`SOURCE` may be a local `.nemo` archive, an extracted NeMo checkpoint, a local
Hugging Face model directory, or a repository ID. Architecture detection and
the model-family default precision are automatic. For NeMo model repositories,
the converter downloads only the `.nemo` checkpoint through the standard
Hugging Face cache. Use `--revision` to pin a remote branch, tag, or commit.

## Supported model families

| Family | Default | Output | Model-specific guide |
|---|---|---|---|
| ASR | Q8_0 | One GGUF | [ASR models](asr/models.md) |
| Diarization | F32 | One GGUF | [ASR models](asr/models.md) |
| Punctuation and capitalization | Q8_0 | One GGUF | [ASR models](asr/models.md) |
| VAD | F32 | One GGUF | [ASR models](asr/models.md) |
| TTS | F16 | One GGUF | [TTS models](tts/models.md) |
| Audio codec | F16 | One GGUF | [TTS models](tts/models.md) |
| NMT | F16 | One GGUF | [NMT models](nmt/models.md) |
| S2S VoiceChat | Q4_K_M profile | Runtime bundle directory | [VoiceChat conversion](s2s/models.md) |
| Speech-LLM (SALM) | NVFP4 experts | Decoder GGUF and encoder GGUF | [Speech-LLM (SALM)](#speech-llm-salm) |

Typical one-file conversions look the same across families:

```bash
python3 convert_model.py nvidia/nemotron-speech-streaming-en-0.6b \
  --outfile models/asr.gguf
python3 convert_model.py nvidia/diar_streaming_sortformer_4spk-v2 \
  --outfile models/diarization.gguf --outtype q8_0
python3 convert_model.py nvidia/Nemotron-3-Diarization \
  --outfile models/diarization-v3.gguf --outtype f32
python3 convert_model.py silero --outfile models/vad.gguf
```

## Speech-LLM (SALM)

A speechlm2 SALM checkpoint converts with one command to two GGUF files: the LLM
decoder, a standard llama.cpp NemotronH GGUF, and next to it the encoder file: the speech encoder (with its speaker
branch) and, when the checkpoint carries one, its CTC timestamp head (`ctc_timestamp.*`, for word timestamps and
speaker labels). The decoder names its encoder file, so the runtime needs only the decoder's path; both files carry
an id of the checkpoint, and a decoder and encoder file of different checkpoints are refused.

```bash
python3 convert_model.py checkpoints/salm --architecture salm --outfile models/salm.gguf
# also writes models/salm-Encoder-F16.gguf
nemo-speech serve --asr-model models/salm.gguf
```

`--outtype` selects the decoder's routed experts and, with them, its dense linears (attention, Mamba in/out, shared
experts); embeddings, the output head and the norms stay BF16/F32:

| `--outtype` | Routed experts | Dense linears | Use |
|---|---|---|---|
| `nvfp4` (default) | NVFP4 | BF16 | Blackwell GPUs run the experts natively |
| `iq4_nl` | IQ4_NL | Q8_0 | GPUs without NVFP4 (Ampere, Ada), CPU, Vulkan, Metal; the smaller file |

The encoder file is the same for every `--outtype` (F16 encoder and F32 CTC head), named after the
checkpoint, so decoders of several types share one. `--salm-encoder none` writes only the decoder, for a second
`--outtype` of a checkpoint whose encoder file exists; `--architecture pee` converts only the speech encoder.

The conversion stages the checkpoint's decoder for llama.cpp's NemotronH converter (the pinned `llama.cpp`
submodule), writes a BF16 intermediate next to the output (the size of the BF16 decoder) and then quantizes it
with the patched `llama-quantize`, built once on first use like the S2S converter's. `--dry-run` checks the tensor
mapping without writing.

### DFlash2 draft model

A DFlash2 drafter drafts tokens for speculative decoding instead of the model's MTP head (`asr.salm.dflash_path`,
batched CUDA path). It converts on its own, with the speech-LLM checkpoint it drafts for: the drafter reads that
model's embeddings and hidden states, so its GGUF carries the target's tokenizer and dimensions (only the target's
configs and tokenizer are read).

```bash
python3 convert_model.py checkpoints/salm-dflash2 --architecture dflash2 \
  --salm-target checkpoints/salm --outfile models/salm-DFlash2-BF16.gguf
```

The drafter stays BF16 (`--outtype f16` or `f32` also work); a local drafter directory is detected without
`--architecture`.

## Common options

Pass `--architecture` only when auto-detection is ambiguous. `--outtype`
overrides the family default, and `--dry-run` validates and prints a conversion
plan where supported. Run `python3 convert_model.py --help` for the complete
option list.

The converters do not import `nemo_toolkit`, and source checkpoints are not
modified.

### Cache-aware RNNT Q8 precision

Cache-aware RNNT Q8_0 exports keep the Conformer pointwise convolution weights
in F16. Quantizing these kernels to Q8 can cause speech to be decoded as long
blank runs; attention, feed-forward and decoder matrix quantization is unchanged.
Existing GGUFs are not repaired by updating the runtime: regenerate them from
the original `.nemo` checkpoint with `--outtype q8_0`. Converting already-quantized
weights to F16 cannot recover the original values. Other model/head types and
requested weight types keep their previous conversion policy.
