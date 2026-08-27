# MagpieTTS Runtime

This directory contains the GGML/GGUF MagpieTTS runtime used by the public
surfaces:

- `nemo-speech synthesize`: local text-to-speech CLI.
- `nemo-speech serve`: HTTP API and browser playground.
- `synthesize_text`: optional stable-C-ABI example with text and token-ID input.
- `riva_server`: optional Riva-compatible gRPC server.

MagpieTTS generates codec tokens autoregressively. The public multilingual
357M checkpoint uses the separate NeMo NanoCodec decoder
(`nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`) to turn those codec tokens
into 22050 Hz mono PCM audio.

## Model Files

The examples below assume these files are available:

```text
models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf
models/magpie-tts/extracted
models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf
```

`magpie_tts_multilingual_357m.v2602.f16.gguf` is the MagpieTTS autoregressive
model. The `extracted` directory is the unpacked MagpieTTS `.nemo` checkpoint
and is needed by the tokenizer. The NanoCodec GGUF is the token-to-audio
decoder.

## Build

Use a supported preset. The configure helper validates dependencies and applies
the pinned ggml patches for CUDA builds:

```bash
scripts/configure.sh cuda-tts
cmake --build --preset cuda-tts
```

Use `cpu-tts` for a CPU build. To build the optional gRPC server, use
`cuda-full` after installing the optional dependencies described in the
[`build guide`](../../../docs/build.md). To build `synthesize_text`, add
`-DNEMO_SPEECH_BUILD_EXAMPLES=ON` while configuring.

## Convert To GGUF

Convert the MagpieTTS `.nemo` checkpoint or extracted checkpoint directory:

```bash
python convert_model.py models/magpie-tts/extracted \
  --outfile models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf \
  --outtype f16 \
  --metadata-json models/magpie-tts/magpie_tts_multilingual_357m.gguf.json
```

Convert the NanoCodec decoder separately:

```bash
python convert_model.py models/nano-codec/extracted \
  --outfile models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --outtype f16
```

The converters are source-tree Python tools; see
[`docs/model-conversion.md`](../../../docs/model-conversion.md) for setup.

## Unified CLI

The default command downloads the pinned MagpieTTS, tokenizer, and NanoCodec
artifacts when needed:

```bash
build/cuda-tts/bin/nemo-speech synthesize "Hello world." --output magpie.wav
```

## Standalone Example

`synthesize_text` uses the same stable C ABI available to external applications.
It accepts text by default and also supports pre-tokenized IDs for diagnostics:

```bash
build/cuda-tts/bin/synthesize_text \
  --tts.magpie-model models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf \
  --tts.codec-model models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --tts.tokenizer-model-dir models/magpie-tts/extracted \
  --tts.text "Hello world." \
  --tts.speaker 0 \
  --tts.steps 64 \
  --tts.wav-out /tmp/magpietts.wav
```

Run `build/cuda-tts/bin/synthesize_text --help` for text-file, token-ID,
resampling, and generation options.

## Riva TTS Server

`riva_server` exposes MagpieTTS through the Riva
`RivaSpeechSynthesis` API. It accepts real text in `SynthesizeSpeechRequest.text`
and tokenizes it internally with the native C++ tokenizer.

Launch the server:

```bash
build/cuda-full/bin/riva_server \
  --tts.magpie-model models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf \
  --tts.codec-model models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --tts.tokenizer-model-dir models/magpie-tts/extracted \
  --bind 0.0.0.0:50051 \
  --tts.language-code en-US \
  --tts.voice-name John \
  --benchmark
```

Send requests with a Riva-compatible client; see
[`docs/clients.md`](../../../docs/clients.md).

The server implements `Synthesize`, `SynthesizeOnline`, and
`GetRivaSynthesisConfig`. It returns raw `LINEAR_PCM` s16le audio at the
NanoCodec sample rate. Native tokenization supports `en`, `es`, `de`, `fr`,
`it`, `vi`, and `hi` by default; `zh` and `ja` require builds with
`NEMO_SPEECH_TTS_WITH_ZH=ON` and `NEMO_SPEECH_TTS_WITH_JA=ON`, respectively.
Tokenizers are loaded once and cached by language.

`GetRivaSynthesisConfig` advertises every compiled-in TTS language and the
dotted voice names accepted by synthesis requests. Bare local voice names are
also accepted.

By default, the server runs a short discarded startup warmup request. Use:

- `--tts.no-warmup`: skip startup warmup.
- `--tts.warmup-text TEXT`: choose the warmup text.
- `--tts.warmup-steps N`: choose the warmup decoder frame count.

Useful server options:

- `--benchmark`: print tokenizer, encoder, decoder, codec, and E2E metrics per
  request.
- `--verbose`: print detailed MagpieTTS and NanoCodec logs. Warmup metrics are
  only printed when verbose logging is enabled.
- `--tts.voice-name NAME` or `--tts.speaker N`: choose the default baked speaker.
  Available names are `John`, `Sofia`, `Aria`, `Jason`, and `Leo`.
- `--tts.codec-cpu`, `--tts.codec-threads`, `--tts.chunk-frames`,
  `--tts.lt-backend`, `--tts.lt-fp32`, `--tts.sampling-backend`, `--tts.uma-mode`,
  `--tts.no-cfg`, `--tts.no-local-transformer`, `--tts.no-kv-cache`, and
  `--tts.no-stateful-codec`: use the same runtime controls as the standalone
  path.
- `--tts.seed`, `--tts.steps`, `--tts.temperature`, `--tts.top-k`, and
  `--tts.cfg-scale`: set default generation controls. The Python client can
  override these per request through `custom_configuration`.

## Troubleshooting

- If CUDA TTS targets fail to compile after a ggml update, rerun
  `scripts/configure.sh cuda-tts` before rebuilding.
- If `nemo-speech synthesize` rejects text input, check that `--tokenizer-dir`
  points at the extracted MagpieTTS `.nemo` directory and that `--language` is
  supported. The `synthesize_text` equivalents are
  `--tts.tokenizer-model-dir` and `--tts.language-code`. For tokenizer
  debugging, generate token IDs with `scripts/tts/tokenize-magpietts.py` and
  pass them with `--tts.tokens` or `--tts.tokens-file`.
- If the server rejects a request, check `language_code`, `voice_name`, and
  `sample_rate_hz`. The client may omit `sample_rate_hz` for native 22050 Hz
  output or request downsampling to any integer rate from 8000 through 22050
  Hz, including 11025 and 16000 Hz.
- If first-request latency matters, keep startup warmup enabled. Disable it with
  `--no-warmup` in the unified CLI or `--tts.no-warmup` in `riva_server`.
