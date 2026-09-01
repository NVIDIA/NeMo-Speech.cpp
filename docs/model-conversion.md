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

Typical one-file conversions look the same across families:

```bash
python3 convert_model.py nvidia/nemotron-speech-streaming-en-0.6b \
  --outfile models/asr.gguf
python3 convert_model.py nvidia/diar_streaming_sortformer_4spk-v2 \
  --outfile models/diarization.gguf --outtype q8_0
python3 convert_model.py silero --outfile models/vad.gguf
```

## Common options

Pass `--architecture` only when auto-detection is ambiguous. `--outtype`
overrides the family default, and `--dry-run` validates and prints a conversion
plan where supported. Run `python3 convert_model.py --help` for the complete
option list.

The converters do not import `nemo_toolkit`, and source checkpoints are not
modified.
