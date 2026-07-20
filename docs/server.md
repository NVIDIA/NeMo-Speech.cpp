# HTTP, realtime, and Riva-compatible gRPC serving

The project provides two server executables over the same core C++ engines:

- `nemo-speech serve` hosts HTTP, realtime WebSocket, and the browser
  playground. It loads configured models once into an `EngineRegistry`.
- `riva_server` hosts the Riva-compatible gRPC services.

They are separate processes and do not share loaded model instances. The
source installer and `*-server` presets build the HTTP executable for ASR and
diarization plus TTS without the gRPC dependency chain. `cuda-full`, `developer`,
or explicit component options add NMT, optional language frontends, and
`riva_server`.

```bash
nemo-speech serve \
  --asr-model models/asr.q8_0.gguf \
  --tts-model models/magpie.f16.gguf \
  --codec-model models/nano-codec.decoder.f16.gguf \
  --tokenizer-dir models/magpie-tokenizer
# HTTP API and playground: http://127.0.0.1:8080/

# Open the playground after the listener is ready.
nemo-speech serve --asr-model models/asr.q8_0.gguf --open

# Start the separate Riva-compatible gRPC server.
riva_server --asr.model.path models/asr.q8_0.gguf --bind 0.0.0.0:50051
```

## Engine and listener configuration

Both servers accept the same dotted engine keys, such as
`asr.vad.masker.onset`, through YAML, environment variables, or CLI options.
Settings are applied in this order:

```text
built-in defaults < --config FILE.yaml < NEMO_SPEECH_<KEY> env < CLI option
```

Nested YAML maps mirror the dotted keys. Start from a checked-in example:

| file | use |
|---|---|
| [`config/asr.example.yaml`](../config/asr.example.yaml) | ASR-only server |
| [`config/diar.example.yaml`](../config/diar.example.yaml) | standalone diarization |
| [`config/tts.example.yaml`](../config/tts.example.yaml) | TTS-only server |
| [`config/nmt.example.yaml`](../config/nmt.example.yaml) | NMT-only server |
| [`config/server.example.yaml`](../config/server.example.yaml) | combined HTTP speech server |

```bash
nemo-speech serve --config config/asr.example.yaml
```

```yaml
asr:
  backend:
    gpu: 0
  model:
    path: /models/nemotron-speech-streaming-en-0.6b.q8_0.gguf
  streaming:
    rnnt_right_context: 1
```

Unknown keys are errors. For environment variables, uppercase the dotted key
and replace `.` and `-` with `_`; for example,
`asr.model.path` becomes `NEMO_SPEECH_ASR_MODEL_PATH`. CLI options accept the
canonical dotted form (`--asr.model.path /models/asr.gguf`) and common aliases
such as `--asr-model`. Boolean keys can be negated with an explicit value, such
as `--asr.endpointing.enable=false`.

HTTP listener settings use the same precedence:

| flag / key | default | meaning |
|---|---|---|
| `--host` / `http.host` | `127.0.0.1` | HTTP bind address |
| `--port` / `http.port` | `8080` | HTTP port |
| `--threads` / `http.threads` | `4` | bounded worker pool |
| `--max-upload-mb` / `http.max-upload-mb` | `512` | request-body limit |
| `--read-timeout` / `http.read-timeout` | `30` | socket read timeout in seconds |
| `--write-timeout` / `http.write-timeout` | `30` | socket write timeout in seconds |
| `--access-log` / `http.access-log` | false | log completed requests |
| `--log-format` / `http.log-format` | `text` | text or JSON access records |

Capabilities auto-enable when their required model paths are present. An
explicit `asr.enabled`, `tts.enabled`, or `nmt.enabled` value can be `true`,
`false`, or `auto` (the default). Starting with no enabled capability is an
error. `riva_server` ignores HTTP listener settings and instead accepts
`--bind HOST:PORT` (default `0.0.0.0:50051`); use an engine-only YAML file with
that binary.

The complete engine key references are in [ASR configuration](asr/configuration.md),
[TTS configuration](tts/configuration.md), and
[NMT configuration](nmt/configuration.md).

The default loopback binding is intentional. Use `--host 0.0.0.0`, TLS, and an
API key explicitly for remote access. Prefer the
`NEMO_SPEECH_HTTP_API_KEY` environment variable over placing a secret in
command-line arguments. API keys require `Authorization: Bearer <key>`;
browser WebSockets may supply `?api_key=<key>`. NVIDIA NIM is the supported
production deployment path; this server is intended for local use and direct
integration.

Cross-origin browser access is disabled by default. `--cors-origin ORIGIN`
allows one explicit origin; use `*` only for an intentionally public API.
HTTP listener options can also be set under `http:` in YAML, through `--http.*`
dotted overrides, or with environment variables such as
`NEMO_SPEECH_HTTP_PORT`. The gRPC binary accepts `--bind HOST:PORT`; both
binaries accept the same `asr.*`, `tts.*`, and `nmt.*` engine settings.

Use `--access-log` to log completed requests without headers or query strings.
`--log-format json` emits one JSON object per request; the global `--json`
option also makes listener-ready events and enabled access logs
machine-readable:

```bash
nemo-speech --json serve --access-log --asr-model models/asr.q8_0.gguf
```

## Health and readiness

`GET /health` returns the engine status and runtime version. `GET /ready`
returns readiness, selected device, and loaded capabilities. Both return HTTP
503 until the configured engines are ready. The CLI can check either endpoint;
it uses `/ready` by default:

```bash
nemo-speech health --url http://127.0.0.1:8080/ready
```

## HTTP endpoints

- `GET /`, `GET /health`, `GET /ready`, and `GET /version`
- `GET /v1/models`
- `POST /v1/audio/transcriptions` (OpenAI-compatible multipart subset)
- `POST /v1/audio/speech` (OpenAI-compatible JSON subset)
- `POST /v1/audio/translations` when ASR and NMT are loaded
- `POST /v1/audio/speech/translations` for ASR → NMT → TTS audio (extension)
- `POST /v1/translations` for text NMT
- `POST /v1/audio/diarizations` for speaker segments (`/v1/diarizations` alias;
  optional multipart `mode=streaming|offline`)
- WebSocket `/v1/realtime` for live PCM16 transcription

The realtime socket accepts binary little-endian PCM16 frames. Send a
`session.update` JSON event before audio to set `sample_rate`, `language`,
`automatic_punctuation`, `verbatim`, or `word_timestamps`; then send
`input_audio_buffer.commit` to finish. Base64 audio in
`input_audio_buffer.append` is also accepted. The server emits
`conversation.item.input_audio_transcription.delta` and `.completed` events.

The bundled playground uses this protocol directly for microphone input and
has no Node.js, Python, CDN, analytics, or external runtime assets. It reports
server readiness, selected device, loaded model capabilities, accepts dropped
WAV files, and disables panels whose required model is not loaded.

`/v1/audio/transcriptions/realtime` is an alias for clients that prefer the
audio namespace. Browser clients may authenticate the socket with the
`api_key` query parameter; other requests should use the bearer header. The
query credential is not accepted on non-realtime routes.

## Limits and lifecycle

Uploads are capped at 512 MiB by default (`--max-upload-mb`). Socket reads and
writes time out after 30 seconds by default (`--read-timeout` and
`--write-timeout`); inference work runs on a bounded worker pool (`--threads`).
The NMT context pool remains an engine setting (`nmt.pool.contexts`) and is not
silently expanded to match HTTP workers. SIGINT/SIGTERM stops HTTP admission and
releases its loaded models. `--no-warmup` is available for HTTP diagnostics but
is not recommended when startup readiness matters.

The separate `riva_server` accepts messages up to gRPC's signed 32-bit limit and
drains active RPCs for up to 10 seconds on SIGINT/SIGTERM. It currently uses
plaintext server credentials; terminate TLS in a trusted proxy when exposing it
outside a controlled network.
