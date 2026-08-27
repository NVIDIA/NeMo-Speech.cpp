# ASR models

The runtime loads one **GGUF** per ASR model. Ready-to-run Q8 GGUFs are
published alongside the original checkpoints on Hugging Face and indexed by
the CLI:

```bash
nemo-speech model list
nemo-speech pull nemotron-3.5
```

`nemotron-3.5` is the default when `--model` is omitted. A short name, full
repository ID, or existing local GGUF path can be passed to `--model`.

## Nemotron 3.5 (0.6B, multilingual, prompt-conditioned RNNT)

A cache-aware FastConformer-RNNT with **language-ID prompt conditioning** across
40+ language-locales. Hugging Face:
[nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)

```bash
nemo-speech pull nemotron-3.5
```

Select a language such as `en-US` or `es-ES`, or use `auto` for model-based
detection. Structured results include the selected or detected language:

```bash
nemo-speech transcribe audio.wav \
  --model nemotron-3.5 \
  --language auto \
  --json
```

When ITN is configured with a parent grammar directory (`en/`, `es/`, ...),
the same explicit or auto-detected language code selects the grammar used for
the final transcript. Unsupported languages remain unchanged.

The CLI uses this model by default. It supports whole-file recognition,
recorded streaming with `--stream`, and live microphone transcription.

## Nemotron-Speech Streaming (0.6B, cache-aware RNNT)

English FastConformer-RNNT for whole-file or cache-aware streaming inference.
Hugging Face:
[nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

```bash
nemo-speech pull nemotron-en
```

## Parakeet TDT (0.6B v3, multilingual, offline transducer)

Token-and-Duration Transducer: the joint predicts each token together with its
frame span. 25 European languages, self-punctuating. Hugging Face:
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

```bash
nemo-speech pull parakeet-tdt
```

The model does not support cache-aware streaming; inference is full-utterance
only. Streaming requests are rejected with an error; use offline recognition
(`nemo-speech transcribe`, `POST /v1/audio/transcriptions`, or gRPC
`Recognize`).

## Parakeet CTC (1.1B, offline / buffered streaming)

English FastConformer-CTC for whole-file recognition or overlapping buffered
streaming. Hugging Face:
[nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b)

```bash
nemo-speech pull parakeet-ctc
```

## Converting custom ASR checkpoints

Use the root [`convert_model.py`](../../convert_model.py) converter for a custom
checkpoint or alternate quantization. Follow the [model conversion
guide](../model-conversion.md) to set up its isolated Python environment and
choose a supported source. ASR head type (CTC, RNNT, or TDT) is auto-detected;
use `--head-type` only when an override is needed.

## Quantization (`--outtype`)

```bash
python3 convert_model.py model.nemo --outfile model.gguf --outtype q8_0
```

Quantization applies to linear and pointwise-convolution weights. Other tensors
retain their supported floating-point formats.

| `--outtype` | format | bytes/elem | use case |
| --- | --- | --- | --- |
| `q8_0` (default) | Q8_0 | 1.062 | compact, high-quality default |
| `bf16` | BF16 | 2.000 | modern NVIDIA / ARM v9 |
| `fp16` | F16 | 2.000 | Apple Silicon, older GPUs |
| `q6_k` | Q6_K | 0.820 | smaller artifact, more quantization |
| `q5_k` | Q5_K | 0.688 | smaller artifact, more quantization |
| `q4_k` | Q4_K | 0.562 | compact K-quant |
| `nvfp4` | NVFP4 | 0.562 | FP4; native acceleration on supported Blackwell GPUs |
| `mxfp4` | MXFP4 | 0.531 | compact FP4; acceleration depends on the backend |

`q8_0` is the portable default; pass `--outtype` to choose a different
size/precision tradeoff. K-quants
(`q4_k`/`q5_k`/`q6_k`) require inner dim divisible by 256; any tensor that fails
alignment falls back to F16 and is reported by the converter. NVFP4 and MXFP4
require inner dim divisible by 64 and use the same fallback. Validate FP4
accuracy and performance on the target model and backend before deployment.

### CUDA batching: planar Q8 layout

The default Q8 layout is portable. For high-concurrency CUDA inference, the
converter can instead produce a planar Q8 layout:

```bash
python3 convert_model.py model.nemo --outfile model.planar.q8_0.gguf \
    --outtype q8_0 --q8-layout planar
```

Planar Q8 is CUDA-only. Keep a default-layout artifact for other backends.

## Companion models (optional)

These optional GGUFs are loaded alongside the ASR model and can be changed
without reconverting it. Enable them with their runtime options; see
[configuration](configuration.md).

### Silero VAD

Used for [VAD feature masking](configuration.md#vad-feature-masking) and
VAD-driven [endpointing](configuration.md#endpointing). Convert from the public
Silero VAD package:

```bash
pip install "silero-vad==6.2.0"
python3 convert_model.py silero --outfile models/silero-v6.2.0.gguf
# offline alternative, using an existing whisper.cpp Silero checkpoint:
#   python3 convert_model.py silero --outfile models/silero-v6.2.0.gguf \
#       --from-whisper-ggml /path/to/silero-v6.2.0-ggml.bin
```

Source: [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (the pip
package), or whisper.cpp's bundled checkpoint for the offline path.

### Sortformer speaker diarization

Used for ASR speaker tags and standalone `nemo-speech diarize`. Sortformer v2
supports up to four speakers, with stateful streaming for long recordings and
full-attention inference for short recordings. Convert it with:

```bash
python3 convert_model.py nvidia/diar_streaming_sortformer_4spk-v2 \
    --outfile models/sortformer-v2-f32.gguf
# --outtype f32 is the default; f16 and q8_0 produce smaller artifacts.
```

Enable with `--diar-model models/sortformer-v2-f32.gguf`; streaming geometry
comes from `--diar-preset` (see [configuration](configuration.md)). Segment
postprocessing defaults follow the checkpoint and may need tuning for your
audio.

Source: [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2).

### PnC (punctuation + capitalization)

Used for [automatic punctuation](configuration.md#postprocessing-profanity-itn-pnc)
- restores casing and `. , ?` for models that emit lowercase unpunctuated text
(e.g. Parakeet CTC). Use a compatible PnC GGUF, or convert a local NeMo BERT
punctuation-and-capitalization `.nemo` checkpoint directly
with `convert_model.py`:

```bash
python3 convert_model.py pnc.nemo --outfile pnc-bert.q8_0.gguf --outtype q8_0
```
