# Command-line guide

`nemo-speech` is the primary local interface. Inference commands accept
explicit local model paths.

Run `nemo-speech --help` for the command inventory and
`nemo-speech help <command>` for the options compiled into the installed
build.

## Transcribe audio

Transcribe one WAV file:

```bash
nemo-speech transcribe recording.wav --model ./models/asr.q8_0.gguf
```

The file CLI accepts mono or stereo PCM16 and float32 WAV input from 8–96 kHz.
It downmixes and resamples to the model rate. Unsupported containers or codecs
produce an error with a conversion command.

### Subtitles and structured output

```bash
nemo-speech transcribe recording.wav --format srt --output recording.srt
nemo-speech transcribe recording.wav --format vtt --output recording.vtt
nemo-speech transcribe recording.wav --json --word-times
```

Plain results are written to stdout. Progress and diagnostics are written to
stderr so output can be redirected safely. Global `--json`, `--quiet`, and
`--verbose` options work across commands.

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
nemo-speech diarize meeting.wav --model sortformer.gguf
nemo-speech diarize meeting.wav --format rttm --output meeting.rttm
```

Streaming geometry is the default. Use `--offline` for full-attention
processing of short recordings.

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
nemo-speech synthesize "Hello" \
  --magpie-model magpie.f16.gguf \
  --codec-model nanocodec.f16.gguf \
  --tokenizer-dir magpie-tokenizer \
  --output hello.wav
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

```bash
python convert_model.py nvidia/nemotron-speech-streaming-en-0.6b \
  --outfile nemotron-speech-streaming-en-0.6b.q8_0.gguf
nemo-speech model info nemotron-speech-streaming-en-0.6b.q8_0.gguf
```

The converter downloads the published `.nemo` checkpoint through the standard
Hugging Face cache and writes the GGUF consumed by the runtime. See
[model conversion](model-conversion.md) for the isolated Python environment and
other model families. Model files remain local; pass their paths explicitly or
record a reusable multi-model setup in a YAML configuration file.

## Benchmark

Benchmark end-to-end ASR concurrency with one shared recognizer:

```bash
nemo-speech bench asr recordings/ \
  --model asr.q8_0.gguf \
  --concurrency 1,2,4 \
  --json
```
