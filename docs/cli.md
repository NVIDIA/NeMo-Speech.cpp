# Command-line guide

`nemo-speech` is the primary local interface. Inference commands accept local
model paths, indexed Hugging Face repository IDs, or short model names. ASR,
diarization, and TTS also have ready-to-run defaults.

Run `nemo-speech --help` for the command inventory and
`nemo-speech help <command>` for the options compiled into the installed
build.

## Models and cache

List the models built into this CLI release:

```bash
nemo-speech model list
nemo-speech --json model list
```

The human output is grouped by ASR, diarization, and TTS. `*` marks models used
by default. The JSON form also exposes every alias, artifact role, applicable
command, companion, pinned revision, and license.

Inference downloads a missing indexed model automatically. You can download it
ahead of time with either a short name or the full repository ID:

```bash
nemo-speech pull nemotron-3.5
nemo-speech pull nvidia/nemotron-3.5-asr-streaming-0.6b
nemo-speech pull magpie
```

Pulling `magpie` also installs its tokenizer assets and the required NanoCodec
companion. Downloads use the system `curl` executable, follow HTTPS only,
resume interrupted regular-file downloads, and are accepted only after the
pinned size and SHA-256 match. Concurrent processes share per-artifact cache
locks. Run `nemo-speech doctor` to confirm that `curl` is available.

The cache location is platform-specific:

| Platform | Default cache |
|---|---|
| macOS | `~/Library/Caches/NeMoSpeech/models` |
| Linux | `${XDG_CACHE_HOME:-~/.cache}/nemo-speech/models` |
| Windows | `%LOCALAPPDATA%\NeMoSpeech\models` |

Set `NEMO_SPEECH_MODEL_DIR` to use another location. Passing an existing local
GGUF path or tokenizer directory always takes precedence and does not use the
network.

## Transcribe audio

Transcribe one WAV file:

```bash
nemo-speech transcribe recording.wav
nemo-speech transcribe recording.wav --model nemotron-en
nemo-speech transcribe recording.wav --model ./models/asr.q8_0.gguf
```

The file CLI accepts mono or stereo PCM16 and float32 WAV input from 8-96 kHz.
It downmixes and resamples to the model rate. Unsupported containers or codecs
produce an error with a conversion command.

### Transcribe a microphone live

```bash
nemo-speech transcribe --live \
  --backend auto
```

The command captures the system's default microphone and prints interim and
endpointed transcripts to stderr while you speak. Press Ctrl-C once to stop;
the stream is flushed and the complete final transcript is written to stdout.
Use `--output transcript.txt` to write it to a file, or select `json`, `srt`,
or `vtt` with `--format`.

Live capture is compiled directly into the CLI through miniaudio and uses the
native host audio API: CoreAudio on macOS, WASAPI on Windows, and ALSA or
PulseAudio on Linux. No PortAudio runtime is required. The operating system may
ask for microphone permission the first time; grant access to the terminal or
shell running `nemo-speech`.

### Subtitles and structured output

```bash
nemo-speech transcribe recording.wav --format srt --output recording.srt
nemo-speech transcribe recording.wav --format vtt --output recording.vtt
nemo-speech transcribe recording.wav --json --word-times
```

SRT and WebVTT cues prefer sentence, clause, and pause boundaries. Cues use up
to two lines, target 37 characters per line, and allow small whole-word
overflow within the common 42-character subtitle limit.

Plain results are written to stdout. Progress and diagnostics are written to
stderr so output can be redirected safely. Global `--json`, `--quiet`, and
`--verbose` options work across commands. The default output keeps command
lifecycle, effective inference configuration, results, warnings, and errors.
Model-loader, backend, ggml, and llama.cpp diagnostics require `--verbose`.

### Transcribe a directory

```bash
nemo-speech transcribe recordings/ \
  --recursive \
  --output-dir transcripts \
  --concurrency 4
```

The recognizer is loaded once. Concurrent utterances share that recognizer and
compatible inference work is dynamically batched on the GPU. Relative directory
paths are preserved and existing outputs require `--force`.

### Compose speech features

A transcription can use VAD, speaker diarization, punctuation, ITN, and NMT in
one pass:

```bash
nemo-speech transcribe meeting.wav \
  --vad-model silero.gguf \
  --diar-model sortformer.gguf \
  --pnc-model punctuation.gguf \
  --itn-model-dir grammars/en-US \
  --nmt-model translate.q8_0.gguf \
  --translate-to es \
  --json
```

Use only the companion models needed by the workflow.

## Diarize audio

Standalone diarization does not require an ASR model:

```bash
nemo-speech diarize meeting.wav
nemo-speech diarize meeting.wav --format rttm --output meeting.rttm
nemo-speech diarize recordings/ \
  --format rttm --output-dir rttms --concurrency 4
```

Directory inputs load one shared model and dynamically batch compatible steps.
Relative paths are preserved. Streaming geometry is the default; use
`--offline` for full-attention processing of short recordings. Segmentation
thresholds are dataset-dependent; use `--onset`, `--offset`, `--pad-onset`,
`--pad-offset`, `--min-duration-on`, and `--min-duration-off` when applying a
checkpoint's published postprocessing configuration.

## Translate text

```bash
nemo-speech translate --model translate.q8_0.gguf --from en --to de "Speech runs locally."
printf '%s\n' "First line" "Second line" | \
  nemo-speech translate --model translate.q8_0.gguf --from en --to es
```

Use `--input` for a line-oriented text file and `--output` to write translations
to a file.

## Synthesize speech

```bash
nemo-speech synthesize "Hello" --output hello.wav
```

## Select a backend

Backend selection is automatic. Override it with `--device` or its `--backend`
alias:

```bash
nemo-speech transcribe recording.wav --device cuda:0
nemo-speech transcribe recording.wav --device cpu
nemo-speech transcribe recording.wav --device metal
nemo-speech transcribe recording.wav --device vulkan:0
```

Run `nemo-speech doctor` to see the compiled backends and detected devices.

## Convert and inspect models

The built-in index covers the published ready-to-run GGUFs. Use the converter
when working with a custom checkpoint or producing a different quantization:

```bash
python convert_model.py custom-model.nemo --outfile custom-model.q8_0.gguf
nemo-speech model info custom-model.q8_0.gguf
```

The converter can also resolve Hugging Face repository IDs through the standard
cache. See [model conversion](model-conversion.md) for the isolated Python
environment and supported model families. Custom files remain local; pass
their paths explicitly or record a reusable multi-model setup in a [YAML
configuration file](server.md#engine-and-listener-configuration).

## Benchmark

Benchmark end-to-end ASR concurrency with one shared recognizer:

```bash
nemo-speech bench asr recordings/ \
  --model asr.q8_0.gguf \
  --concurrency 1,2,4 \
  --json
```
