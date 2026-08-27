# HTTP, realtime, and Riva-compatible gRPC serving

The project provides two server executables over the same core C++ engines:

- `nemo-speech serve` hosts HTTP, realtime WebSocket, and the browser
  playground. It loads each configured model once.
- `riva_server` hosts the Riva-compatible gRPC services.

They are separate processes and do not share loaded models. The `*-server`
presets include HTTP support for ASR, diarization, NMT, and TTS without gRPC.
`cuda-full` adds gRPC, text normalization, and optional TTS language frontends;
`developer` also adds examples, tests, and tools. Individual features can be
selected explicitly (presets: [build guide](build.md)).

```bash
nemo-speech serve \
  --asr-model models/asr.q8_0.gguf \
  --tts-model models/magpie-tts/magpie_tts_multilingual_357m.v2602.f16.gguf \
  --codec-model models/nano-codec/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --tokenizer-dir models/magpie-tts/extracted
# HTTP API and playground: http://127.0.0.1:8080/

# Open the playground after the listener is ready.
nemo-speech serve --asr-model models/asr.q8_0.gguf --open

# Serve standalone speaker diarization (downloads the indexed model if needed).
nemo-speech serve --diar-model sortformer

# Start the separate Riva-compatible gRPC server.
riva_server --asr.model.path models/asr.q8_0.gguf --bind 0.0.0.0:50051
```

## Engine and listener configuration

Both servers accept the same dotted engine keys, such as
`asr.vad.masker.onset`, through YAML, environment variables, or CLI options.
The HTTP server additionally accepts top-level `diar.*` keys for standalone
diarization; use `asr.diar.*` to add speaker labels to ASR results.
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
| [`config/server.example.yaml`](../config/server.example.yaml) | combined ASR, diarization, NMT, and TTS server |

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
canonical dotted form (`--asr.model.path /models/asr.gguf`); `nemo-speech serve`
also accepts common aliases such as `--asr-model` (`riva_server` does not).
Boolean keys can be negated with an explicit value, such
as `--asr.endpointing.enable=false`.

HTTP listener settings use the same precedence:

| flag / key | default | meaning |
|---|---|---|
| `--host` / `http.host` | `127.0.0.1` | HTTP bind address |
| `--port` / `http.port` | `8080` | HTTP port |
| `--api-key` / `http.api-key` | none | require `Authorization: Bearer <key>` on API routes |
| `--tls-cert` / `http.tls-cert` | none | TLS certificate path (with `--tls-key`; requires a build with `NEMO_SPEECH_HTTP_TLS=ON`) |
| `--tls-key` / `http.tls-key` | none | TLS private-key path |
| `--cors-origin` / `http.cors-origin` | none | allowed browser origin |
| `--no-ui` / `http.playground` | playground on | disable the embedded playground |
| `--threads` / `http.threads` | `4` | bounded worker pool |
| `--max-upload-mb` / `http.max-upload-mb` | `512` | request-body limit |
| `--read-timeout` / `http.read-timeout` | `30` | socket read timeout in seconds |
| `--write-timeout` / `http.write-timeout` | `30` | socket write timeout in seconds |
| `--access-log` / `http.access-log` | false | log completed requests |
| `--log-format` / `http.log-format` | `text` | text or JSON access records |

Capabilities auto-enable when their required model paths are present. An
explicit `asr.enabled`, `tts.enabled`, or `nmt.enabled` value can be `true`,
`false`, or `auto` (the default). Starting with no enabled capability is an
error. `riva_server` does not accept HTTP listener settings; it takes
`--bind HOST:PORT` (default `0.0.0.0:50051`); use an engine-only YAML file with
that binary.

The complete engine key references are in [ASR configuration](asr/configuration.md),
[TTS configuration](tts/configuration.md), and
[NMT configuration](nmt/configuration.md).

The default loopback binding is intentional. For remote access, set
`--host 0.0.0.0`, TLS (`--tls-cert` + `--tls-key`), and an API key explicitly.
TLS must be enabled at source-build time with `NEMO_SPEECH_HTTP_TLS=ON` and is
not part of the default server presets. Prefer the `NEMO_SPEECH_HTTP_API_KEY`
environment variable over placing a secret in command-line arguments. API
routes require `Authorization: Bearer <key>`; browser WebSockets may supply
`?api_key=<key>`. The playground, health, readiness, and version routes remain
unauthenticated. NVIDIA NIM is the supported production deployment path; this
server is intended for local use and direct integration.

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

`GET /health` returns a compact engine status and runtime version. `GET /ready`
returns readiness, selected device, and loaded capabilities. Both return HTTP
503 when no engine is ready. The CLI can check either endpoint; it uses
`/ready` by default:

```bash
nemo-speech health --url http://127.0.0.1:8080/ready
```

## HTTP endpoints

Full request/response field reference: **[HTTP API reference](api.md)**.

- `GET /`, `GET /health`, `GET /ready`, and `GET /version`
- `GET /v1/models` - loaded model inventory (OpenAI SDK-compatible, with
  capability metadata)
- `POST /v1/audio/transcriptions` - speech-to-text (OpenAI-compatible subset)
- `POST /v1/audio/speech` - text-to-speech (OpenAI-compatible subset)
- `POST /v1/translations` - text translation
- `POST /v1/audio/translations` - speech translation (ASR -> NMT)
- `POST /v1/audio/speech/translations` - speech-to-speech translation (extension)
- `POST /v1/audio/diarizations` - speaker segments (`/v1/diarizations` alias)
- WebSocket `/v1/realtime` - live PCM16 transcription

OpenAI SDK compatibility is limited to model listing and the documented
transcription and speech subsets. The translation and diarization routes are
project extensions, and the realtime socket is not the OpenAI Realtime API.

The realtime socket accepts binary little-endian PCM16 frames; an optional
`session.update` JSON event before the first frame sets session options. See
the [API reference](api.md#websocket-v1realtime) for session fields and the
event protocol.

The bundled playground uses this protocol directly for microphone input and
has no Node.js, Python, CDN, analytics, or external runtime assets. It reports
server readiness, selected device, loaded model capabilities, accepts dropped
WAV files, and disables panels whose required model is not loaded.

`/v1/audio/transcriptions/realtime` is an alias for clients that prefer the
audio namespace. Browser clients may authenticate the socket with the
`api_key` query parameter; other requests should use the bearer header. The
query credential is not accepted on non-realtime routes.

## Limits and lifecycle

Uploads are capped at 512 MiB by default (`--max-upload-mb`); the same limit
applies to cumulative audio on a realtime WebSocket stream. Socket reads and
writes time out after 30 seconds by default (`--read-timeout` and
`--write-timeout`); inference work runs on a bounded worker pool (`--threads`).
`--threads` does not change the NMT context pool (`nmt.pool.contexts`).
SIGINT/SIGTERM stops HTTP admission and releases loaded models. `--no-warmup`
is available for diagnostics but is not recommended when startup readiness
matters.

The separate `riva_server` accepts messages up to gRPC's signed 32-bit limit and
drains active RPCs for up to 10 seconds on SIGINT/SIGTERM. It currently uses
plaintext server credentials; terminate TLS in a trusted proxy when exposing it
outside a controlled network.
