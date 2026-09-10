# VoiceChat configuration

The realtime server loads one process-level VoiceChat runtime from a converted
model-version directory. `s2s.model_dir` enables the runtime automatically;
`s2s.enabled` can override automatic detection.

## Common settings

| Key | Default | Description |
|---|---:|---|
| `s2s.enabled` | `auto` | Enablement policy: `auto`, `true`, or `false` |
| `s2s.model_dir` | empty | Converted model-version directory |
| `s2s.max_streams` | `32` | Maximum number of resident conversation states |
| `s2s.verbose` | `false` | Emit detailed GGML and llama.cpp diagnostics |

`nemo-speech serve` also accepts `--s2s-model-dir` and
`--s2s-max-streams` as aliases.

```yaml
s2s:
  enabled: auto
  model_dir: /models/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF
  max_streams: 32
  verbose: false
```

```bash
nemo-speech serve --config config/voicechat.yaml
```

`s2s.max_streams` is a state-reservation ceiling, not a throughput target.
Set it to `1` for a single-conversation deployment. Incoming requests are
batched dynamically; conversation, sampler, and generated-audio state remain
isolated per stream.

`s2s.max_streams` is also the server's only concurrent-session admission
control, and the two roles share one number: it caps how many resident
conversation states fit in memory, and it is the hard limit past which a new
session is refused. The WebSocket handshake itself always succeeds — the
server sends `session.created` unconditionally — but the moment a session
past the ceiling sends its first `session.update` or audio chunk, the server
emits an `error` event (`code: inference_error`, message `"maximum
concurrent streams reached"`) and force-closes that socket. Sessions already
admitted are unaffected; the rejection happens before the new session
touches inference.

Because both roles share one number, sizing for memory headroom alone is not
enough: `s2s.max_streams` is also the ceiling past which generation for
*every* admitted session slows down, since the dynamic batcher schedules all
admitted streams together without ever shedding load once they're in (see
`S2S_BATCH_QUEUE_DELAY_US` below). Measure the largest concurrency that still
generates audio at real-time pace for your GPU and model profile, and set
`s2s.max_streams` to that number rather than to the largest count that merely
fits in GPU memory — sessions above your compute-safe number are then
refused outright instead of silently degrading the sessions already running.

When `nemo-speech serve` chooses its default HTTP worker count, it reserves
enough workers for the configured VoiceChat stream ceiling. An explicitly set
`http.threads` value remains authoritative; it should be at least the intended
number of simultaneous WebSocket sessions.

## Realtime WebSocket settings

These settings apply to `nemo-speech serve`:

| Key | Default | Description |
|---|---:|---|
| `s2s.max_session_seconds` | `300` | Maximum cumulative input-audio duration per session |
| `s2s.max_pending_function_responses` | `64` | Maximum queued tool responses per session |
| `s2s.output_text_events` | `false` | Also emit legacy `response.output_text.*` events |

The listener's `http.api-key`, upload limit, read timeout, write timeout, and
TLS settings apply to VoiceChat sockets. Never put the configured, long-lived
API key in a URL or the `api_key` query parameter, where logs and browser
history can expose it. Header-capable clients should send
`Authorization: Bearer <key>`. Browser clients should obtain a short-lived,
scope-limited token from a trusted backend instead of receiving the configured
server key.

## Runtime environment variables

| Variable | Default | Description |
|---|---:|---|
| `S2S_BATCH_QUEUE_DELAY_US` | `1000` | Maximum internal stage batching delay |
| `S2S_INGRESS_COHORT_DELAY_US` | `2000` | Initial stream-cohort collection delay |
| `S2S_TTS_SEED` | `0` | Acoustic-token sampling seed; nonzero values enable reproducible sampling |
| `S2S_MAX_STREAMS` | configured value | Override the resident-stream ceiling |
| `S2S_DEBUG_TIMING` | unset | Emit per-stage timing diagnostics |

Environment variables are optional. Start with defaults and change batching
delays only when tuning for a measured workload.
