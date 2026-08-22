# TTS configuration

Configuration for MagpieTTS + NanoCodec. `nemo-speech serve` hosts HTTP; the
separate `riva_server` hosts Riva-compatible gRPC.
Either process can load ASR, TTS, and NMT together. For how keys are set (YAML,
environment, and CLI precedence), see
[Server configuration](../server.md#engine-and-listener-configuration). To
download the models and extract the tokenizer, see [TTS models](models.md).

## Serving

```bash
nemo-speech serve --config config/tts.example.yaml
```

or with flags for HTTP:

```bash
nemo-speech serve \
    --tts.magpie-model models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf \
    --tts.codec-model models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
    --tts.tokenizer-model-dir models/magpie-tts/extracted \
    --host 127.0.0.1 --port 8080 \
    --tts.language-code en-US --tts.voice-name John
```

For Riva-compatible gRPC, use the same engine options with `riva_server` and
`--bind 0.0.0.0:50051`.

`--tts.tokenizer-model-dir` is the extracted MagpieTTS `.nemo` directory - see
[TTS models](models.md) for how to obtain it.

`--tts.tn-model-dir` optionally enables Sparrowhawk text normalization before
tokenization (written form to spoken form, for example `I have 2 apples` to
`I have two apples`). It requires a `-DNEMO_SPEECH_WITH_NORM=ON` build and a TN
grammar directory containing `tokenize_and_classify.far` and `verbalize.far`.
For multiple languages, pass a parent containing language-named children such
as `en/`, `fr/`, and `vi/`, each with those two FARs; `post_process.far` is used
when present. The older split layout (`classify/tokenize_and_classify.far`,
`verbalize/verbalize.far`) remains supported.

The optional `riva_server` supports `RivaSpeechSynthesis.Synthesize`,
`SynthesizeOnline`, and `GetRivaSynthesisConfig`. It takes plain text in
`SynthesizeSpeechRequest.text`, supports native Magpie tokenizers for `en`,
`es`, `de`, `fr`, `it`, `vi`, `zh`, `hi`, and `ja`, and returns `LINEAR_PCM`
s16le at the NanoCodec sample rate. Japanese and Mandarin are included when
`NEMO_SPEECH_TTS_WITH_JA` and `NEMO_SPEECH_TTS_WITH_ZH`, respectively, are
enabled at build time; both default to `OFF`. Mandarin uses bundled Jieba and
pypinyin-compatible data together with the model's pinyin-to-phoneme
dictionary. Set `MAGPIE_MANDARIN_G2P_DIR` only to override the bundled Mandarin
data directory.

`GetRivaSynthesisConfig` advertises the compiled-in TTS languages and the
dotted voice names accepted by synthesis requests.

TTS auto-enables when `tts.magpie-model`, `tts.codec-model`, and
`tts.tokenizer-model-dir` are all set; force with `tts.enabled`.

## Voices

Voice names are case-insensitive. The runtime accepts a local speaker name, a
zero-based speaker index, or a model-qualified name such as
`magpietts.John`. `tts.voice-name` selects the default; otherwise
`tts.speaker` is used.

The HTTP model inventory lists the available local names. On
`/v1/audio/speech`, `default` and supported OpenAI voice aliases such as
`alloy` select that configured local default; they are not additional voices. See the
[HTTP API reference](../api.md#post-v1audiospeech).

## Text normalization

Install the shared Sparrowhawk/OpenFST normalizer and enable it in the build:

```bash
scripts/build_itn_deps.sh
scripts/configure.sh cpu-tts -DNEMO_SPEECH_WITH_NORM=ON
cmake --build --preset cpu-tts
```

Pass the grammar directory to the CLI or server:

```bash
nemo-speech synthesize "I have 2 apples." \
    --magpie-model models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf \
    --codec-model models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
    --tokenizer-dir models/magpie-tts/extracted \
    --tn-model-dir models/tn_configs \
    --output normalized.wav
```

The equivalent YAML setting is:

```yaml
tts:
  magpie-model: /models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf
  codec-model: /models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf
  tokenizer-model-dir: /models/magpie-tts/extracted
  tn-model-dir: /models/tn_configs
  language-code: en-US
```

The grammar directory may be a single-language Sparrowhawk TN directory or a
multilingual root such as `models/tn_configs`, with immediate language-named
children (`en/`, `fr/`, `vi/`, and so on). Each language directory must contain
`tokenize_and_classify.far` and `verbalize.far`; `post_process.far` is optional.

If a TN model directory is set in a build without
`NEMO_SPEECH_WITH_NORM=ON`, startup logs a warning and text passes through
unchanged.

## Local synthesis

Use the unified CLI for synthesis without a server:

```bash
nemo-speech synthesize "Hello from Magpie." \
    --magpie-model models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf \
    --codec-model models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
    --tokenizer-dir models/magpie-tts/extracted \
    --speaker 0 --output magpie.wav
```

See the [CLI guide](../cli.md#synthesize-speech) for the common workflow. The
optional `synthesize_text` example additionally exposes the stable C ABI and
pre-tokenized input path when built with `NEMO_SPEECH_BUILD_EXAMPLES=ON`.

## Key reference

All keys nest under `tts.`. Defaults shown; CLI alias listed where one exists.

### Models and identity

| key | CLI alias | default | meaning |
|---|---|---|---|
| `tts.enabled` | - | `auto` | `true` / `false` / `auto` |
| `tts.magpie-model` | - | - | MagpieTTS GGUF token generator (required) |
| `tts.codec-model` | - | - | NanoCodec decoder GGUF (required) |
| `tts.tokenizer-model-dir` | - | - | extracted Magpie `.nemo` dir (required) |
| `tts.tokenizer.sentence-limit.<lang>` | - | per language (`en` 45 ... `ja` 40) | sentence-chunking threshold in words (characters for `zh`/`ja`); subkeys `en`, `es`, `fr`, `vi`, `it`, `de`, `zh`, `hi`, `ja` |
| `tts.tn-model-dir` | - | - | enables Sparrowhawk TN with this grammar dir; requires `NEMO_SPEECH_WITH_NORM=ON` |
| `tts.language-code` | - | `en-US` | default text language code |
| `tts.voice-name` | - | - | default voice name or speaker index |
| `tts.speaker` | - | `0` | default baked speaker index |

### Sampling and decoding

| key | CLI alias | default | meaning |
|---|---|---|---|
| `tts.seed` | - | `-1` | RNG seed; `-1` = current time |
| `tts.steps` | - | `-1` | max decoder frames; `-1` = model default |
| `tts.top-k` | - | `-1` | top-k sampling; `-1` = model default |
| `tts.temperature` | - | model default | sampling temperature |
| `tts.cfg-scale` | - | model default | classifier-free guidance scale |
| `tts.use-cfg` / `tts.no-cfg` | - | on | enable / disable CFG |
| `tts.use-local-transformer` / `tts.no-local-transformer` | - | on | local transformer |
| `tts.use-kv-cache` / `tts.no-kv-cache` | - | on | decoder KV cache |

### Codec streaming

| key | CLI alias | default | meaning |
|---|---|---|---|
| `tts.chunk-frames` | - | `3` | codec frames per streamed audio chunk |
| `tts.codec-queue-depth` | - | `4` | codec worker queue depth |
| `tts.codec-history-frames` | - | `-1` | rolling codec history frames |
| `tts.codec-future-frames` | - | `1` | rolling codec future frames |
| `tts.window-ms` | - | `0` | overlap-add window (ms) |
| `tts.flush-partial-chunk` | - | `true` | emit a final partial codec chunk |
| `tts.use-stateful-codec` / `tts.no-stateful-codec` | - | on | fast layer-state codec |
| `tts.codec-cpu` | - | `false` | force NanoCodec decoder onto CPU |

### Execution

| key | CLI alias | default | meaning |
|---|---|---|---|
| `tts.threads` | `--threads` | `4` | CPU threads for Magpie + codec; use the dotted key with HTTP, where `--threads` controls request workers |
| `tts.codec-threads` | - | `0` | codec CPU threads; `0` = use `threads` |
| `tts.lt-backend` | - | `auto` | local-transformer backend: `auto`/`cpu`/`cuda` |
| `tts.lt-fp32` | `--tts.local-transformer-fp32` | `false` | run the local transformer in FP32 |
| `tts.sampling-backend` | - | `auto` | sampling backend: `auto`/`cpu`/`cuda` |
| `tts.uma-mode` | - | `auto` | CUDA managed memory: `auto`/`off`/`on` |
| `tts.longform` | - | `auto` | sentence-chunk longform mode: `auto`/`off`/`on` |

### Diagnostics and warmup

| key | CLI alias | default | meaning |
|---|---|---|---|
| `tts.benchmark` | `--benchmark` | `false` | emit per-request metrics from `riva_server` |
| `tts.verbose` | `--verbose` | `false` | detailed Magpie/NanoCodec logs; use global `--verbose` with `nemo-speech` |
| `tts.warmup-enabled` / `tts.no-warmup` | - | on | startup tokenizer/runtime warmup |
| `tts.warmup-text` | - | (built-in) | text used for startup warmup |
| `tts.warmup-steps` | - | `8` | decoder frames used for startup warmup |
