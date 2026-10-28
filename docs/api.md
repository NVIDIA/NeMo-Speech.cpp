# HTTP API reference

Complete field reference for the `nemo-speech serve` HTTP API. Every endpoint
follows the same conventions:

- **Auth**: if the server was started with an API key, send
  `Authorization: Bearer <key>` to `/v1` routes (WebSocket clients may use
  `?api_key=<key>`). The playground, health, readiness, and version routes are
  intentionally unauthenticated.
- **Errors**: non-2xx responses carry
  `{"error": {"message": "...", "type": "invalid_request_error" | "rate_limit_error" | "server_error"}}`.
  Invalid requests return 400 (`invalid_request_error`), a full request queue 429
  (`rate_limit_error`), a shutting-down engine 503 and other failures 500 (`server_error`); the
  Riva gRPC server and the CLI use the same classes.
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
| `GET /v1/realtime/health` | VoiceChat WebSocket readiness |

## POST /v1/audio/transcriptions

Speech-to-text (OpenAI-compatible multipart subset). Realtime transcription
uses the `/v1/audio/transcriptions/realtime` WebSocket described below.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | loaded ASR model | omitted or empty: the loaded ASR model; otherwise its id from `GET /v1/models` (or the chat API's served id), and any other id is a 404 `unknown_model` |
| `file` | WAV upload | required | audio to transcribe |
| `language` | string | model default | language code; prompt-conditioned models select their language prompt |
| `response_format` | string | `json` | `json`, `verbose_json` (adds words/timestamps), `text`, `srt`, or `vtt` |
| `automatic_punctuation` | bool | `true` | punctuation + capitalization |
| `verbatim` | bool | `false` | skip inverse text normalization |
| `profanity_filter` | bool | `false` | mask words from the configured list |
| `timestamp_granularities[]` | string, repeatable | none | `word` or `segment` (requires `verbose_json`); accepted for OpenAI clients, both return `words` (there are no segment-level times) |
| `diarization` | bool | `false` | tag words with speakers (requires `verbose_json` and a diarizer model) |
| `max_speaker_count` | int | ignored | deprecated compatibility field; capacity comes from the loaded Sortformer model |
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

Word timestamps cover every language. As in NeMo's aligner, a Chinese or Japanese transcript is timed per
character: each letter or digit is one entry of `words` (punctuation is not). A Thai transcript is timed per
character too, each with its combining marks (NeMo uses a dictionary word segmenter there). SRT, WebVTT and the
transcript keep the text as written. `confidence` is 0 for SALM models.

## POST /v1/chat/completions

Speech-LLM (SALM) transcription in the OpenAI Chat Completions shape (JSON body). Other ASR models transcribe
through it too, greedily: they refuse an instruction and a `temperature` above 0 with a 400. The audio and an
optional instruction travel as one user message:

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | the id listed by `GET /v1/models` |
| `messages` | array | required | exactly one `user` message whose `content` holds one `input_audio` part and at most one `text` part (the instruction) |
| `language` | string | none | language hint, added to the default instruction as "The speech is in <language>." for the codes the model knows (`en`, `fr`, `zh`, `tl`, ...); a `text` instruction is used as written. Echoed in the response (`und` when omitted) |
| `timestamp_granularities` | array | none | `"word"` and/or `"segment"`: word timestamps, subtitle cues and speakers in the response, in any language (Chinese, Japanese and Thai per character, see transcriptions); needs the CTC timestamp adapter |
| `word_timestamps` | bool | `false` | `true` asks for the same as `timestamp_granularities: ["word"]`; a request sends one of the two |
| `subtitle_format` | string | none | `srt` or `vtt`, with word timestamps and without `stream`: the subtitle cues also as subtitle text in `transcription.subtitles`; a 422 `timestamp_unavailable` when the alignment is unavailable |
| `temperature` | number | greedy | 0 to 1; below 1e-5 decodes greedily |
| `seed` | integer | random | non-negative; the same request and seed give the same transcript |
| `stream` | bool | `false` | send the transcript as server-sent events while it is generated (below) |
| `stream_options` | object | none | with `stream`: `{"include_usage": true}` adds a usage chunk |
| `n` | integer | 1 | must be 1 |
| `response_format` | object | `{"type": "text"}` | only `{"type": "text"}` |
| `modalities` | array | `["text"]` | only `["text"]` |
| `tool_choice` | string | | only `"none"` |
| `user`, `metadata` | any | | accepted and ignored |

`tools`, `max_tokens`, `max_completion_tokens` (the output budget follows the audio duration) and `audio` are refused
with a 400 that names the field, as is any field not listed here.

An `input_audio` part is `{"data": <base64>, "format": <container>}`: standard base64 with padding, and a lowercase
container name such as `wav`, `flac` or `mp3`. The server identifies the container from the bytes (WAV including
streamed WAVs with unknown sizes, FLAC, MP3), so `format` is required but not checked against the content. Audio
runs up to one hour (`http.chat.max-audio-seconds`) and needs at least 320 samples at 16 kHz; the request body is
limited to 256 MiB. An instruction steers the model ("Translate the speech into English.", a speaker-tag format, ...);
it is a prompt, not a guarantee of the output format. It must not contain the tokenizer's special tokens, such as the
chat template's `<|im_end|>` (a 400); the speaker, event and style tags the model writes (`<spk:0>`, ...) are allowed.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d @- <<EOF
{"model": "salm", "language": "en", "timestamp_granularities": ["word"],
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "Transcribe the audio."},
   {"type": "input_audio", "input_audio": {"data": "$(base64 -w0 audio.wav)", "format": "wav"}}]}]}
EOF
```

The response is a `chat.completion`: `choices[0].message.content` is the transcript and `finish_reason` is `stop`.
`usage` counts the prompt (with its `audio_tokens`) and the transcript; it is absent when no decode ran (near-silent
audio). The top-level `transcription` object carries the structured result:

```json
"transcription": {
  "schema_version": "1", "task": "transcribe", "transcript_origin": "llm",
  "language": "en", "duration": 3.17, "text": "Hello John, how are you?", "segments": [],
  "words": [
    {"word": "Hello", "word_index": 0, "start": 0.555, "end": 0.713,
     "speaker": "spk:0", "speaker_slot": 0, "speaker_confidence": 0.992}
  ],
  "subtitle_cues": [{"first_word": 0, "last_word": 4, "start": 0.555, "end": 2.378, "text": "Hello John, how are you?"}],
  "word_timestamp_alignment": {"status": "aligned", "model": "salm",
    "canonical_word_version": "nemo-ctc-sot-v1", "frame_stride_seconds": 0.039625, "failure_reason": null},
  "diarization_timestamps": [{"speaker": 0, "start": 0.281, "end": 2.299}],
  "diarization_frame_seconds": 0.01,
  "speaker_word_timestamps": {"spk:0": [{"word": "Hello", "...": "..."}]}
}
```

| Field | Description |
|---|---|
| `language`, `duration`, `text` | the requested language (`und` when omitted), the audio length in seconds, the transcript |
| `segments` | always empty in this release; `subtitle_cues` group the words |
| `words` | with `timestamp_granularities`: `word`, `word_index`, `start`, `end` (seconds, to the millisecond) and the speaker (`spk:N`, numbered from 0) with its diarization slot and the slot's mean activity over the word; the words of all speakers ordered by start time, as NeMo's CTC timestamp aligner gives them (overlapping speakers interleave; a speaker the alignment cannot place is left out) |
| `subtitle_cues` | word ranges and times grouped as SRT/WebVTT cues |
| `subtitles` | with `subtitle_format`: `{"format": "srt" or "vtt", "text": ...}`, the cues with times rounded to the millisecond; the cues of overlapping speakers can overlap |
| `word_timestamp_alignment` | `status` (`aligned` or `unavailable`), the alignment model and, when unavailable, `failure_reason` (for example `empty_transcript`); the transcript is returned either way |
| `diarization_timestamps`, `speaker_word_timestamps` | the diarizer's 10 ms speaker activity (each run above 0.5, per slot) and the words per speaker, when the alignment succeeded |

With `stream: true` the response is `text/event-stream`: `data: <chat.completion.chunk>` events, as the OpenAI SDKs
read them. The first chunk carries `delta.role`, the next ones the transcript as it is generated in `delta.content`
(the pieces add up to the non-streamed `content`), and the last chunk has `finish_reason: "stop"` and the
`transcription` object above, word timestamps and speakers included (they are aligned once the transcript is
complete). With `stream_options.include_usage` a final chunk with empty `choices` carries `usage`; `data: [DONE]` ends
the stream. The audio is still sent whole: only the output streams. A request refused before any text keeps its
HTTP status; a failure after the stream started arrives as an `{"error": ...}` event followed by `[DONE]`.

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}
data: {...,"choices":[{"index":0,"delta":{"content":"Hello John,"},"logprobs":null,"finish_reason":null}]}
data: {...,"choices":[{"index":0,"delta":{"content":" how are you?"},"logprobs":null,"finish_reason":null}]}
data: {...,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}],"transcription":{...}}
data: [DONE]
```

Errors use the OpenAI envelope `{"error": {"message", "type", "param", "code"}}`; `param` names the offending field
where there is one:

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | a field's value or shape is invalid, or the audio header is malformed |
| 404 | `unknown_model` | `model` is not the served id |
| 413 | `payload_too_large` | the request body exceeds the limit |
| 415 | `unsupported_media_type` | the body is not `application/json` |
| 422 | `invalid_request` | the audio cannot be decoded (`invalid or unsupported audio file`) |
| 422 | `timestamp_unavailable` | timestamps were requested and no CTC timestamp adapter is loaded |
| 429 | `overloaded` | the request queue is full; retry later |
| 500 | `server_error` | an unexpected server failure |
| 503 | `engine_error` | the backend failed |

## WebSocket /v1/audio/transcriptions/realtime

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
| `max_speaker_count` | int | ignored | deprecated compatibility field; capacity comes from the loaded Sortformer model |
| `endpointing_ms` | number | server default | end-of-utterance silence threshold |
| `speech_contexts` | array | none | word boosting, as in `/v1/audio/transcriptions` |
| `prompt` | string | none | OpenAI-compat: one boosted phrase at boost 10 |

Server events: `session.created`, `session.updated`,
`conversation.item.input_audio_transcription.delta` (partials), `.completed`
(finals, with `words` when requested), `input_audio_buffer.committed`,
`input_audio_buffer.cleared`, and `error`. An `error` event carries `message`,
`type` (as on the REST routes) and `code` (`invalid_argument`,
`resource_exhausted`, `unavailable` or `internal_error`). Models that transcribe
complete audio only (SALM; `/v1/models` reports `"streaming": false`) send one
`error` with code `streaming_unsupported` on connect and close the socket with
status 1003; use `POST /v1/audio/transcriptions` instead.

For backward compatibility, `/v1/realtime` serves this transcription protocol
when VoiceChat is not loaded. Use the explicit audio-namespaced path for new
transcription clients.

## VoiceChat WebSocket /v1/realtime and /realtime

When a VoiceChat model is loaded, both paths expose the full-duplex Riva
VoiceChat session protocol used by the reference client and compatible realtime
integrations. The server sends `session.created` on connection. Send
`session.update` before audio begins; session configuration is immutable after
the first audio frame.

`session.audio.input.format` accepts `"pcm16"` or an object such as
`{"type":"audio/pcm","rate":24000}`. Input must be mono little-endian
PCM16 at 16-48 kHz. `session.audio.output.format` must select PCM16 at 24 kHz.
Audio may be sent as binary frames or as base64 in
`input_audio_buffer.append`. Output is base64 PCM16 in 80 ms
`response.output_audio.delta` packets.

| `session` field | Type | Default | Description |
|---|---|---|---|
| `audio.input.format` | string or object | PCM16 at 24 kHz | client input encoding and rate |
| `audio.output.format` | string or object | PCM16 at 24 kHz | output encoding; only 24 kHz PCM16 is supported |
| `instructions` | string | server VoiceChat prompt | system instructions, including conversational policy |
| `tools` | array or JSON string | `[]` | OpenAI-format tool definitions; `ack_messages` is supported |

Response events include `response.created`,
`response.output_audio.delta`, `response.output_audio.done`,
`response.output_audio_transcript.delta`,
`response.output_audio_transcript.done`, and `response.done`. User turns emit
`input_audio_buffer.speech_started`, transcription delta/completed events, and
`input_audio_buffer.speech_stopped`. Every event includes an `event_id`.

Tool requests arrive as `response.function_call_arguments.done`, with
`call_id`, `name`, and JSON-encoded `arguments`. Return the result on the same
socket:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "function_call_output",
    "call_id": "call_123",
    "output": "{\"status\":\"ok\"}"
  }
}
```

Send `session.close` to finish cleanly. The server drains residual input,
emits response completion events, then sends `session.end` with received,
sent, dropped, inference, and audio-duration counters. See the
[VoiceChat client guide](s2s/clients.md) for the complete event flow.

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

Speech translation (multipart): ASR then NMT when an NMT model is loaded; otherwise a speech-LLM model
(SALM) translates directly into any language it knows (`target_language` codes such as `en`, `de`,
`fr`, `ja`, `hi`; an unknown code is a 400). Accepts the ASR common fields:
`automatic_punctuation`, `verbatim`, `profanity_filter`, `speech_contexts`,
`prompt` - identical semantics to `/v1/audio/transcriptions` - plus:

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | loaded model | as on `/v1/audio/transcriptions`; the `speech-translation` id from `GET /v1/models` is also accepted |
| `file` | WAV upload | required | source speech |
| `language` | string | auto | source language code |
| `target_language` | string | `en-US` | target language code |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |

Response (`json`): `{"text": "..."}` (the translation); `verbose_json` adds
`task`, `language` (the target), and `duration`.

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

Without a diarization model, a speech-LLM (SALM) model answers
with its encoder's speaker branch and no transcript: the 10 ms speaker turns its
word timestamps carry (activity above 0.5), whatever the `mode`. It needs the
CTC timestamp head, like word timestamps.
