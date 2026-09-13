# Live speaker-change event for the realtime transcription WebSocket

## Problem

`NeMoSpeech` (the companion project driving this server for live captions)
finalizes transcripts on a client-side silence timer
(`SilenceCommitTrigger`: 700ms of detected silence, or a 15s hard cap as a
backstop). In continuous conversational audio — which frequently lacks a
genuine 700ms pause — this means captions can go up to 15 real seconds
without finalizing, arriving as one large multi-sentence, multi-speaker
block instead of a steady stream of short, per-speaker turns. Verified
against a real session: a 15+ second rambling monologue stayed entirely in
the `partial` field, un-attributed to any speaker, before the hard cap
forced one giant commit.

Root cause: the realtime WebSocket handler
(`server/http/http_server.cpp`, `/v1/audio/transcriptions/realtime`) only
reports diarization results as word-level `speaker` tags attached to a
`conversation.item.input_audio_transcription.completed` event — i.e. only
once the client asks for a transcript via `input_audio_buffer.commit`.
There is currently no way for a client to know a speaker change happened
*before* deciding to commit — the information doesn't reach the wire until
then, even though the diarizer has effectively already detected it
internally.

## What's already there

`RecognitionStream` (`src/asr/recognizer.h`/`.cpp`) already owns a
`std::unique_ptr<DiarStream> diar_`, fed incrementally on every `push()`
call as a sidecar alongside ASR. `DiarStream::segments()`
(`src/asr/diar/diar_pipeline.h`) is documented as "valid on a live stream
at any point" and returns hysteresis-cleaned, confirmed speaker segments
(onset/offset probability hysteresis + `min_duration_on`/`min_duration_off`
gap-filling) — not raw per-frame probabilities. This is the existing
noise-filtering the fix should build on, not duplicate: a segment boundary
`segments()` reports is already a confirmed turn, not frame-level flicker.

This capability is simply never queried by the realtime WS handler today.

## Approach

Poll `diar_->segments()` after each audio push inside `RecognitionStream`,
and surface a new, additive WebSocket event the moment the most recent
segment's speaker differs from the last one reported. The client
(`NemoSpeech`'s `live_bridge.py`) listens for this event and immediately
sends `input_audio_buffer.commit` itself, instead of waiting on
`SilenceCommitTrigger`.

**Why this shape, and not alternatives considered:**

- *Server auto-commits on speaker change* (server decides to finalize,
  not just announce) was rejected: `/v1/audio/transcriptions/realtime` is
  shared with the built-in browser playground
  (`server/http/http_server.cpp`'s embedded JS), and changing when/whether
  a transcript is delivered would change behavior for every existing
  client of this endpoint, not just ours. An additive event that existing
  clients simply don't recognize keeps their behavior untouched.
- *Raw per-frame probability streaming* (exposing `frame_probs()` directly
  to the client) was rejected: it would move the hysteresis/noise-filtering
  logic that already exists in `segments()` into every consumer, duplicating
  work `DiarStream` already does correctly.
- Keeping `SilenceCommitTrigger` as a backstop (unchanged) rather than
  removing it: a single continuous speaker never triggers a "change," and
  non-diarized sessions (`--diarize` off, or `--diar-model` not loaded)
  have no speaker-change signal at all. Both cases still need silence-based
  finalization to happen eventually.

## Protocol change

New event type, emitted only when a session has diarization enabled
(`speaker_diarization: true`, the existing session-update flag) and a
diarizer model is loaded:

```json
{
  "type": "conversation.item.speaker_diarization.changed",
  "event_id": "event_42",
  "speaker": 2,
  "start_time": 4.32
}
```

- `speaker`: 1-based speaker id, matching `WordInfo.speaker_tag` /
  `nemo_speech_diar_segment.speaker` elsewhere in the API.
- `start_time`: seconds from stream start, the new segment's start —
  matches the existing `words[].start`/`end` units already used in
  `completed` events.

This is purely additive: existing event types (`delta`, `completed`,
`session.*`, `input_audio_buffer.*`, `error`) are unchanged, and a client
that doesn't recognize the new type (e.g. the built-in playground) simply
ignores it, exactly as it already ignores unrecognized fields today.

## Server implementation

1. **`RecognitionStream` (`src/asr/recognizer.h`/`.cpp`):** add

   ```cpp
   struct SpeakerChange { int speaker; double start_time; };
   std::optional<SpeakerChange> poll_speaker_change();
   ```

   Implementation: if `diar_` is null, always return `nullopt`. Otherwise
   call `diar_->segments()`, look at `.back()` (the most recent segment).
   Track the last-reported speaker as a new private member
   (`std::optional<int> last_reported_speaker_`). If the list is non-empty
   and `segments().back().speaker != last_reported_speaker_`, update
   `last_reported_speaker_` and return the change; otherwise `nullopt`.
   Called once per `push()`, not per `next()` poll loop iteration, since
   `segments()` re-derives its result from the whole retained timeline each
   call and only needs to be checked once per new audio chunk.

2. **`http_server.cpp`'s realtime handler (`append_audio`, around line
   1079):** after the existing `stream->push(...)` /
   `while (auto result = stream->next())` block, call
   `stream->poll_speaker_change()`. If it returns a value, `send()` the new
   event type with its `speaker`/`start_time` fields. No separate
   `options.enable_speaker_diarization` check is needed at this call site —
   `poll_speaker_change()` already returns `nullopt` unconditionally when
   `diar_` is null, which is exactly the non-diarized case.

No change to `session.update` handling, `emit()`, or the `completed`/
`delta` event construction — this is a new, independent branch in
`append_audio`, not a modification of the existing ASR result path.

## Expected latency floor

The diarizer's "streaming" geometry preset processes audio in ~1.6s chunks
(`chunk_len = 20` frames × 80ms, `DiarGeometry::riva_streaming()`), and
`segments()`'s hysteresis requires roughly `min_duration_on` (~0.5s) of a
new speaker before confirming a segment. So a real speaker change should
surface as a `speaker_diarization.changed` event, and thus a commit, within
roughly 1-2 seconds of the change happening — a large improvement over the
current up-to-15-second worst case, though this should be measured against
real audio during implementation rather than assumed.

## Testing

- **New C++ unit test** (`tests/cpp/asr/`, alongside the existing
  `test_diar_frame_lookup.cpp` pattern from PR #1): feed a `RecognitionStream`
  synthetic multi-speaker audio (or mocked `DiarStream` segments, if the
  existing test harness supports substituting one) and assert
  `poll_speaker_change()` fires exactly once per genuine speaker transition,
  never on the first segment (there's no "previous" speaker to differ
  from), and never fires when `diar_` is null.
- **Realtime WS integration test**: extend the existing realtime handler
  test coverage (check `tests/` for the current `/v1/audio/transcriptions/realtime`
  coverage pattern before adding a new one) to assert the new event type
  appears in the stream when `speaker_diarization: true` and a real
  multi-speaker fixture is pushed, and does not appear when diarization is
  off.
- **Manual verification**: run `nemo-speech serve --diar-model sortformer`
  locally and confirm the new event type appears on the wire (e.g. via the
  existing browser playground's console, or a small throwaway WS client)
  when speaking, pausing, and having a second voice/file interject.

## Non-goals / explicitly out of scope

- Changing `SilenceCommitTrigger`'s defaults or removing it — it remains
  the fallback for non-diarized sessions and continuous single-speaker
  audio.
- Any change to `live_bridge.py`/`live.py` (the `NemoSpeech` client) —
  tracked as separate follow-on work in that repo once this event exists
  to consume.
- Any change to the offline/batch diarization path (`transcribe --diarize`,
  `run.py`'s two-pass workaround) — unrelated to the realtime WS handler.
- The `NemoSpeech` overlay/debug-view UI redesign — separate, downstream
  work in that repo.
