# MagpieTTS Runtime

This directory contains the GGML/GGUF MagpieTTS runtime used by two public
surfaces:

- `synthesize_text`: standalone C-ABI text or token-ID to WAV example.
- `riva_server`: Riva-compatible gRPC server that accepts real text for TTS.

MagpieTTS generates codec tokens autoregressively. The public multilingual
357M checkpoint uses the separate NeMo NanoCodec decoder
(`nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`) to turn those codec tokens
into 22050 Hz mono PCM audio.

## Model Files

The examples below assume these files are available:

```text
models/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.f16.gguf
models/magpie_tts_multilingual_357m/extracted
models/nemo_nano_codec_22khz_1.89kbps_21.5fps/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf
```

`magpie_tts_multilingual_357m.f16.gguf` is the MagpieTTS autoregressive model.
The `extracted` directory is the unpacked MagpieTTS `.nemo` checkpoint and is
needed by the tokenizer. The NanoCodec GGUF is the token-to-audio decoder.

## Build

For CUDA builds, apply the local ggml patch before compiling MagpieTTS or
NanoCodec targets:

```bash
scripts/apply-ggml-patches.sh
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DNEMO_SPEECH_BUILD_GRPC=ON
cmake --build build --target synthesize_text riva_server -j$(nproc)
```

`GGML_CUDA=ON` enables the CUDA backend. `GGML_CUDA_GRAPHS=ON` reduces CUDA
graph-launch overhead. `NEMO_SPEECH_BUILD_GRPC=ON` is required for
`riva_server`. For a CPU-only standalone build, omit the CUDA flags; for a
standalone-only build, the gRPC flag is not required.

At runtime, set `GGML_CUDA_DISABLE_GRAPHS=1` to compare CUDA behavior without
CUDA Graphs. The ggml patch also exposes `GGML_CUDA_GRAPH_EVICT_AFTER_MS` and
`GGML_CUDA_GRAPH_SWEEP_MS`; the default eviction policy is 10 seconds with a
5 second sweep interval, and `GGML_CUDA_GRAPH_EVICT_AFTER_MS=0` disables
eviction.

## Convert To GGUF

Convert the MagpieTTS `.nemo` checkpoint or extracted checkpoint directory:

```bash
python convert_model.py models/magpie_tts_multilingual_357m/extracted \
  --outfile models/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.f16.gguf \
  --outtype f16 \
  --metadata-json models/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.gguf.json
```

Convert the NanoCodec decoder separately:

```bash
python convert_model.py models/nemo_nano_codec_22khz_1.89kbps_21.5fps/extracted \
  --outfile models/nemo_nano_codec_22khz_1.89kbps_21.5fps/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --outtype f16
```

## Standalone Example

`synthesize_text` uses the same stable C ABI available to external applications.
It accepts text by default and also supports pre-tokenized IDs for diagnostics:

```bash
build/bin/synthesize_text \
  --tts.magpie-model models/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.f16.gguf \
  --tts.codec-model models/nemo_nano_codec_22khz_1.89kbps_21.5fps/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --tts.tokenizer-model-dir models/magpie_tts_multilingual_357m/extracted \
  --tts.text "Hello world." \
  --tts.speaker 0 \
  --tts.steps 64 \
  --tts.wav-out /tmp/magpietts.wav
```

Run `build/bin/synthesize_text --help` for text-file, token-ID, resampling, and
generation options.

## Riva TTS Server

`riva_server` exposes MagpieTTS through the Riva
`RivaSpeechSynthesis` API. It accepts real text in `SynthesizeSpeechRequest.text`
and tokenizes it internally with the native C++ tokenizer.

Launch the server:

```bash
build/bin/riva_server \
  --tts.magpie-model models/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.f16.gguf \
  --tts.codec-model models/nemo_nano_codec_22khz_1.89kbps_21.5fps/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --tts.tokenizer-model-dir models/magpie_tts_multilingual_357m/extracted \
  --bind 0.0.0.0:50051 \
  --tts.language-code en-US \
  --tts.voice-name John \
  --benchmark
```

Send requests with a Riva-compatible client; see
[`docs/clients.md`](../../../docs/clients.md).

The server implements `Synthesize`, `SynthesizeOnline`, and
`GetRivaSynthesisConfig`. It returns raw `LINEAR_PCM` s16le audio at the
NanoCodec sample rate. Native tokenization currently supports `en`, `es`, `de`,
`fr`, `it`, `vi`, `zh`, `hi`, and `ja`; tokenizers are loaded once and cached
by language.

`GetRivaSynthesisConfig` exposes speakers using Riva's `voice_name` and
`subvoices` parameters. The resulting dotted names, such as `magpietts.John`,
and the original short names, such as `John`, are both accepted in synthesis
requests. The config response advertises every compiled-in TTS language in
`language_code` and includes a `voices_by_language` JSON parameter with the
dotted Magpie voice names for each language.

By default, the server runs a short discarded startup warmup request to
initialize tokenizer, runtime state, and local-transformer graph capture.
`riva_server` also defaults `GGML_CUDA_GRAPH_EVICT_AFTER_MS=0` before
model load unless the environment variable is already set, so CUDA Graphs stay
resident across idle request gaps. Use:

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
- `--seed`, `--steps`, `--temperature`, `--top-k`, and `--cfg-scale`: set
  default generation controls. The Python client can override these per
  request through `custom_configuration`.

## Troubleshooting

- If CUDA TTS targets fail to compile, run
  `scripts/apply_ggml_magpietts_patch.sh` before rebuilding.
- If the standalone runner rejects text input, check that
  `--tokenizer-model-dir` points at the extracted MagpieTTS `.nemo` directory
  and that `--language-code` is supported. For tokenizer debugging, generate
  token IDs with `scripts/tts/tokenize-magpietts.py` and pass them with
  `--tokens` or `--tokens-file`.
- If the server rejects a request, check `language_code`, `voice_name`, and
  `sample_rate_hz`. The client may omit `sample_rate_hz` for native 22050 Hz
  output or request downsampling to any integer rate from 8000 through 22050
  Hz, including 11025 and 16000 Hz.
- If first-request latency matters, keep startup warmup enabled. For controlled
  cold-start experiments, use `--no-warmup`.
