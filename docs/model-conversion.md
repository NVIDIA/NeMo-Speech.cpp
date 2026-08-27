# Model conversion

Published NeMo-Speech.cpp models provide ready-to-run GGUF files on their
Hugging Face pages; see the [ASR](asr/models.md) and [TTS](tts/models.md) model
guides. Use the converter for compatible custom checkpoints, alternate
quantization choices, and supporting models that do not publish a GGUF.

The converters are Python source tools and are not included in the native
release archives. Run them from a source checkout in a virtual environment;
the C++ runtime itself does not require Python. All model families use the same
root entry point:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python convert_model.py SOURCE --outfile MODEL.gguf
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

`SOURCE` may be a local `.nemo` archive, an extracted NeMo checkpoint, a local
Hugging Face model directory, or a Hugging Face repository ID. For NeMo model
repositories, the converter downloads only the `.nemo` checkpoint through the
standard Hugging Face cache. Use `--revision` to pin a remote branch, tag, or
commit.

| Architecture | Default output type | Example source |
| --- | --- | --- |
| ASR | `q8_0` | `nvidia/nemotron-speech-streaming-en-0.6b` |
| diarization | `f32` | `nvidia/diar_streaming_sortformer_4spk-v2` |
| PnC | `q8_0` | local NeMo checkpoint |
| VAD | `f32` | `silero` |
| TTS | `f16` | `nvidia/magpie_tts_multilingual_357m` |
| codec | `f16` | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` |
| NMT | `f16` | `nvidia/Riva-Translate-4B-Instruct-v2` |

The full per-domain model catalogs live in [ASR models](asr/models.md),
[TTS models](tts/models.md), and [NMT models](nmt/models.md).

For example:

```bash
python3 convert_model.py nvidia/nemotron-speech-streaming-en-0.6b \
    --outfile models/asr.gguf
python3 convert_model.py nvidia/diar_streaming_sortformer_4spk-v2 \
    --outfile models/diarization.gguf --outtype q8_0
python3 convert_model.py silero --outfile models/vad.gguf
```

NMT conversion additionally uses the pinned llama.cpp converter. Initialize and
install it only when converting an NMT model:

```bash
git submodule update --init llama.cpp
python -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python3 convert_model.py nvidia/Riva-Translate-4B-Instruct-v2 \
    --outfile models/translate.q8_0.gguf --outtype q8_0
```

Pass `--architecture` only when auto-detection is not possible or when checking
an integration path explicitly. Run `python3 convert_model.py --help` for
architecture-specific options such as ASR's planar Q8 layout, PnC sequence
length, and the offline Silero input.

The converters do not import `nemo_toolkit`; `convert_model.py` is the single
supported conversion entry point.

See the model-family guides for runtime assets and model-specific behavior:

- [ASR, VAD, diarization, and PnC](asr/models.md)
- [TTS and NanoCodec](tts/models.md)
- [NMT](nmt/models.md)
