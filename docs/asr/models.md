# ASR models

The runtime loads one **GGUF** per ASR model. Ready-to-run Q8 GGUFs are
published alongside the original checkpoints on Hugging Face. Install the
Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
```

## Parakeet CTC (1.1B, offline / buffered streaming)

Hugging Face: [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b)

```bash
hf download nvidia/parakeet-ctc-1.1b \
    parakeet-ctc-1.1b.q8_0.gguf --local-dir models
```

## Parakeet TDT (0.6B v3, multilingual, offline transducer)

Token-and-Duration Transducer: the joint predicts each token together with its
frame span. 25 European languages, self-punctuating. Hugging Face:
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

```bash
hf download nvidia/parakeet-tdt-0.6b-v3 \
    parakeet-tdt-0.6b-v3.q8_0.gguf --local-dir models
```

The model is not cache-aware trained: inference is full-utterance only.
Streaming requests are rejected with an error; use offline recognition
(`nemo-speech transcribe`, `POST /v1/audio/transcriptions`, or gRPC
`Recognize`).

## Nemotron-Speech Streaming (0.6B, cache-aware RNNT)

Hugging Face: [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

```bash
hf download nvidia/nemotron-speech-streaming-en-0.6b \
    nemotron-speech-streaming-en-0.6b.q8_0.gguf --local-dir models
```

## Nemotron 3.5 (0.6B, multilingual, prompt-conditioned RNNT)

The same cache-aware FastConformer-RNNT plus **language-ID prompt conditioning**
across 40+ language-locales (`EncDecRNNTBPEModelWithPrompt`). Hugging Face:
[nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)

```bash
hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
    nemotron-3.5-asr-streaming-0.6b.q8_0.gguf --local-dir models
```

The GGUF contains the prompt metadata (`asr.rnnt.num_prompts`,
`asr.rnnt.prompt_dictionary`), and the runtime applies the model's
`prompt_kernel` language fusion ahead of the RNNT joint. Select the language via
the request's `language_code` (`en-US`, `es-ES`, ...) or `auto`; the `<lang>` tag
is stripped from the transcript and the detected language is returned on
`SpeechRecognitionAlternative.language_code` and per-word `WordInfo.language_code`:

```bash
riva_server \
    --asr.model.path models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
    --bind 0.0.0.0:50051

# In another shell:
riva_streaming_asr_client --riva_uri=localhost:50051 \
    --audio_file=audio.wav --language_code=auto \
    --interim_results=false --word_time_offsets=true
```

When ITN is configured with a parent grammar directory (`en/`, `es/`, ...),
the same explicit or auto-detected language code selects the grammar used for
the final transcript. Unsupported languages remain unchanged.

## Converting custom ASR checkpoints

The root [`convert_model.py`](../../convert_model.py) converter accepts a local
`.nemo` archive, an extracted NeMo checkpoint, a local Hugging Face model
directory, or a Hugging Face repository ID. It emits the unified `asr.*`
metadata the runtime expects. The head type (CTC, RNNT, or TDT) is auto-detected
from `model_config.yaml`; override it with `--head-type {ctc,rnnt,tdt}`.

Install the conversion dependencies in a virtual environment:

```bash
pip install -r requirements.txt
```

The converter reads `.nemo` archives directly and does not require
`nemo_toolkit`. Remote checkpoints use the standard Hugging Face cache.

## Quantization (`--outtype`)

```bash
python3 convert_model.py model.nemo --outfile model.gguf --outtype q8_0
```

Applied to Linear weights (encoder MHA + FFN, RNNT LSTM predictor, joint
projections) and the ConformerConv pointwise convs. Conv weights and embeddings
default to F16; norms / biases / positional encodings stay F32.

| `--outtype` | format | bytes/elem | use case |
| --- | --- | --- | --- |
| `q8_0` (default) | Q8_0 | 1.062 | compact, high-quality default |
| `bf16` | BF16 | 2.000 | modern NVIDIA / ARM v9 |
| `fp16` | F16 | 2.000 | Apple Silicon, older GPUs |
| `q6_k` | Q6_K | 0.820 | smaller artifact, more quantization |
| `q5_k` | Q5_K | 0.688 | smaller artifact, more quantization |
| `q4_k` | Q4_K | 0.562 | smallest listed artifact, most quantization |

`q8_0` is the portable default; pass `--outtype` to choose a different
size/precision tradeoff. K-quants
(`q4_k`/`q5_k`/`q6_k`) require inner dim divisible by 256; any tensor that fails
alignment falls back to F16 and is reported by the converter.

### CUDA batching: planar Q8 layout

The converter's default Q8 layout is the portable block-interleaved format. The
CUDA backend in this project can instead store all encoder Q8 values and scales
in tensor-wide planes so high-concurrency FastConformer projections enter the
batched skinny-Q8 tensor-core path without a runtime repack:

```bash
python3 convert_model.py model.nemo --outfile model.planar.q8_0.gguf \
    --outtype q8_0 --q8-layout planar
```

The layout flag covers ordinary encoder projections and fused attention QKV.
Planar Q8 is CUDA-only. Keep a block-layout artifact for other backends.

## Companion models (optional)

These are separate GGUFs the server loads alongside the ASR model - each is its
own file (own `general.architecture`), not bundled into the ASR GGUF, so they can
be swapped without re-converting the ASR model. Enable them at runtime via their
server flags; see [configuration](configuration.md).

### Silero VAD

Used for [VAD feature masking](configuration.md#vad-feature-masking) and
VAD-driven [endpointing](configuration.md#endpointing). Converted from the public
Silero-VAD package (`general.architecture="vad"`):

```bash
pip install "silero-vad==6.2.0"
python3 convert_model.py silero --outfile models/silero-v6.2.0.gguf
# offline alternative, using an existing whisper.cpp Silero checkpoint:
#   python3 convert_model.py silero --outfile models/silero-v6.2.0.gguf \
#       --from-whisper-ggml /path/to/for-tests-silero-v6.2.0-ggml.bin
```

Source: [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (the pip
package), or whisper.cpp's bundled checkpoint for the offline path.

### Sortformer speaker diarization

Used for word-level speaker tags (`WordInfo.speaker_tag`, requested via
`diarization_config.enable_speaker_diarization`) and standalone diarization
(`examples/diarize_file` over the `nemo_speech_diar_*` C ABI, streaming or `--offline`).
Converted
from the public streaming Sortformer v2 checkpoint
(`general.architecture="sortformer"`):

```bash
python3 convert_model.py nvidia/diar_streaming_sortformer_4spk-v2 \
    --outfile models/sortformer-v2-f32.gguf
# --outtype f32 is the default; f16 and q8_0 produce smaller artifacts.
```

Enable with `--diar-model models/sortformer-v2-f32.gguf`; streaming geometry
comes from `--diar-preset` (see [configuration](configuration.md)). Segment
postprocessing defaults are NeMo's callhome-tuned values and are
dataset-sensitive.

Source: [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2).

### PnC (punctuation + capitalization)

Used for [automatic punctuation](configuration.md#postprocessing-profanity-itn-pnc)
- restores casing and `. , ?` for models that emit lowercase unpunctuated text
(e.g. Parakeet CTC). Use a compatible PnC GGUF, or convert a local NeMo BERT
punctuation-and-capitalization `.nemo` checkpoint directly
(`general.architecture="pnc"`):

```bash
python3 convert_model.py pnc.nemo --outfile pnc-bert.q8_0.gguf --outtype q8_0
```
