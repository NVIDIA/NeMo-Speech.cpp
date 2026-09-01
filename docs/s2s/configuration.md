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

When `nemo-speech serve` chooses its default HTTP worker count, it reserves
enough workers for the configured VoiceChat stream ceiling. An explicitly set
`http.threads` value remains authoritative; it should be at least the intended
number of simultaneous WebSocket sessions.

## Realtime WebSocket settings

These settings apply to `nemo-speech serve`:

| Key | Default | Description |
|---|---:|---|
| `s2s.max_session_seconds` | `300` | Maximum cumulative input-audio duration per session |
| `s2s.output_text_events` | `false` | Also emit legacy `response.output_text.*` events |

The listener's `http.api-key`, upload limit, read timeout, write timeout, and
TLS settings apply to VoiceChat sockets. WebSocket clients may supply the API
key as a bearer token or as the `api_key` query parameter.

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
