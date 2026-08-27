# HTTP API reference

Complete field reference for the `nemo-speech serve` HTTP API. Every endpoint
follows the same conventions:

- **Auth**: if the server was started with an API key, send
  `Authorization: Bearer <key>` to `/v1` routes (WebSocket clients may use
  `?api_key=<key>`). The playground, health, readiness, and version routes are
  intentionally unauthenticated.
- **Errors**: non-2xx responses carry
  `{"error": {"message": "...", "type": "invalid_request_error" | "server_error"}}`.
- **Audio uploads**: multipart form with the uncompressed RIFF/WAVE file in
  `file`. Mono or stereo PCM16 and float32 WAVs at 8-96 kHz are accepted;
  stereo is downmixed to mono.

Endpoints return 501 when their capability is not included in the build and a
`server_error` when its model is not loaded. `GET /v1/models` lists the active
capabilities.

OpenAI SDK compatibility covers model listing and the documented subsets of
`/v1/audio/transcriptions` and `/v1/audio/speech`; it does not extend to other
OpenAI APIs. Realtime transcription uses the WebSocket contract below, not the
OpenAI Realtime API.

## Service

| Method + path | Purpose |
|---|---|
| `GET /` | bundled playground UI |
| `GET /health`, `GET /ready` | compact health status (`{"status": "ok", ...}`) / detailed readiness (`{"ready": true, ...}`) |
| `GET /version` | build version |
| `GET /v1/models` | loaded models and capabilities |

## POST /v1/audio/transcriptions

Speech-to-text (OpenAI-compatible multipart subset). Alias:
`/v1/audio/transcriptions/realtime` upgrades to the WebSocket protocol below.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | loaded ASR model | accepted for client compatibility; this server uses its one loaded ASR model |
| `file` | WAV upload | required | audio to transcribe |
| `language` | string | model default | language code; prompt-conditioned models select their language prompt |
| `response_format` | string | `json` | `json`, `verbose_json` (adds words/timestamps), `text`, `srt`, or `vtt` |
| `automatic_punctuation` | bool | `true` | punctuation + capitalization |
| `verbatim` | bool | `false` | skip inverse text normalization |
| `profanity_filter` | bool | `false` | mask words from the configured list |
| `diarization` | bool | `false` | tag words with speakers (requires `verbose_json` and a diarizer model) |
| `max_speaker_count` | int | ignored | deprecated compatibility field; Sortformer v2 supports up to four speakers |
| `speech_contexts` | JSON array | none | word boosting, `[{"phrases": ["..."], "boost": N}]` - same shape as gRPC; scoring: [word boosting](asr/configuration.md#word-boosting) |
| `prompt` | string | none | OpenAI-compat: one boosted phrase at boost 10 |

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
    -F "file=@audio.wav" -F "response_format=verbose_json" \
    -F 'speech_contexts=[{"phrases": ["Kowalczyk"], "boost": 3.0}]'
```

Response (`json`): `{"text": "..."}`. `verbose_json` adds `task`, `language`,
`duration`, and `words[]` (`word`, `start`, `end`, `confidence`, and `speaker`
when diarization is on). SRT and WebVTT responses use the same readable,
punctuation-aware cue grouping as the file CLI.

## WebSocket /v1/realtime

Live PCM16 transcription using a project-specific event protocol. The server
sends `session.created` on connect. Optionally send one `session.update` JSON
event (rejected once audio has started); then binary little-endian PCM16 frames
(or base64 chunks in `input_audio_buffer.append`); finish with
`input_audio_buffer.commit`. `input_audio_buffer.clear` or `response.cancel`
discards buffered audio (`input_audio_buffer.cleared`).

`session.update` -> `{"type": "session.update", "session": {...}}` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `sample_rate` | int | model input rate (16000 for the shipped models) | PCM sample rate, 8000-96000 |
| `language` | string | model default | language code |
| `automatic_punctuation` | bool | `true` | punctuation + capitalization |
| `verbatim` | bool | `false` | skip inverse text normalization |
| `profanity_filter` | bool | `false` | mask words from the configured list |
| `word_timestamps` | bool | `false` | word timings on final events |
| `speaker_diarization` | bool | `false` | tag words with speakers; requires a loaded diarizer |
| `max_speaker_count` | int | ignored | deprecated compatibility field; Sortformer v2 supports up to four speakers |
| `endpointing_ms` | number | server default | end-of-utterance silence threshold |
| `speech_contexts` | array | none | word boosting, as in `/v1/audio/transcriptions` |
| `prompt` | string | none | OpenAI-compat: one boosted phrase at boost 10 |

Server events: `session.created`, `session.updated`,
`conversation.item.input_audio_transcription.delta` (partials), `.completed`
(finals, with `words` when requested), `input_audio_buffer.committed`,
`input_audio_buffer.cleared`, and `error`.

## POST /v1/audio/speech

Text-to-speech (OpenAI-compatible JSON subset).

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | loaded TTS model | accepted for client compatibility; this server uses its one loaded TTS model |
| `input` | string | required | text to synthesize |
| `voice` | string | model default | local voice name, model-qualified voice name, or zero-based speaker index |
| `language` | string | model default | language code |
| `speed` | number | `1.0` | only `1.0` is accepted; other values return 400 |
| `sample_rate` | int | model default | output sample rate, from 8000 Hz through the model rate (22050 Hz for the supported NanoCodec model) |
| `response_format` | string | `wav` | `wav` or `pcm` |

Response: mono signed PCM16, either in a WAV container or raw little-endian
bytes with the matching content type. The complete audio is buffered before the
HTTP response; streaming synthesis is not part of this compatibility subset.

Local voice names are case-insensitive and are listed in the `voices` field of
the speech entry returned by `GET /v1/models`. `<model-id>.<voice>` is also
accepted. `default` and supported OpenAI voice aliases such as `alloy` select
the server's configured default local speaker; they do not provide the
corresponding hosted OpenAI voices. An unrecognized local name returns 400.

## POST /v1/translations

Text translation (JSON body).

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string or array | required | text(s) to translate |
| `source_language` | string | required | source language code |
| `target_language` | string | required | target language code |

Response: `{"translations": [{"text": "..."}]}`.

## POST /v1/audio/translations

Speech translation (ASR -> NMT, multipart). Accepts the ASR common fields:
`automatic_punctuation`, `verbatim`, `profanity_filter`, `speech_contexts`,
`prompt` - identical semantics to `/v1/audio/transcriptions` - plus:

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | WAV upload | required | source speech |
| `language` | string | auto | source language code |
| `target_language` | string | `en-US` | target language code |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |

Response (`json`): `{"text": "..."}` (the translation); `verbose_json` adds
`task`, `language`, and `duration`.

## POST /v1/audio/speech/translations

Speech-to-speech translation (ASR -> NMT -> TTS, multipart; extension). Same
fields as `/v1/audio/translations`, except `target_language` is required and
`response_format` is audio:

| Field | Type | Default | Description |
|---|---|---|---|
| `target_language` | string | required | target language code |
| `response_format` | string | `wav` | `wav` or `pcm` |
| `voice` | string | model default | TTS voice for the translated audio; follows `/v1/audio/speech` voice rules |
| `sample_rate` | int | model default | output rate, from 8000 Hz through the loaded TTS model rate |

Response: translated mono signed PCM16 audio in the requested container.

## POST /v1/audio/diarizations

Speaker segmentation without transcription. Alias: `/v1/diarizations`.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | WAV upload | required | audio to segment |
| `mode` | string | `streaming` | `streaming` for long-form audio, or full-attention `offline` for recordings up to about 6.6 minutes |

Response: `{"segments": [{"start": s, "end": s, "speaker": n}]}` (1-based
speaker ids).

Request `mode=offline` uses full attention. It is distinct from
`diar.preset: offline`, which still uses the streaming path.
