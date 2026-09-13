# Fork changes

This fork (`bsips/NeMo-Speech.cpp`) tracks `NVIDIA/NeMo-Speech.cpp`. This file
logs every place our `main` has diverged from upstream, in order, so a future
rebase/merge from upstream knows what to reconcile.

Companion project: [`NeMoSpeech`](https://github.com/bsips/NeMoSpeech) drives
this project's `nemo-speech` CLI and is where this fork's fixes were
discovered from real-world use. See its `docs/superpowers/` plans/specs and
the linked Obsidian vault notes for the user-facing side of each story.

## Divergences

### 2026-09-10 — Combined-mode diarization collapse on long audio

- **Upstream base:** `a5b6953` (`main`, matches the `v0.1.0` release tag's
  code for the affected files — no drift between them at time of fix).
- **PR:** [bsips/NeMo-Speech.cpp#1](https://github.com/bsips/NeMo-Speech.cpp/pull/1),
  merged as `202c8e6`.
- **Files:** `src/asr/diar/diar_pipeline.{h,cpp}`,
  `tests/cpp/asr/test_diar_frame_lookup.cpp` (new),
  `tests/cpp/asr/CMakeLists.txt`.
- **What:** `DiarStream::speaker_for_frames()` clamped any word-timestamp
  query predating the retained live probability window (`probs_base_`) to
  the window's first frame — correct only when every query lands "near the
  frontier" (soon after the audio is processed), as in streaming/interactive
  use. `Recognizer::recognize()` (the whole-file path behind `transcribe
  --diarize`) tags every word in a single pass at the very end instead,
  silently violating that assumption for any file long enough to trigger
  timeline compaction (~20 min default). Result: combined-mode diarization
  collapsed onto one dominant speaker for most of a long recording, while
  the standalone `diarize` command (whose `segments()` already folds
  `frozen_segs_` back in) was unaffected.
- **Fix:** extracted the frame-range resolution into
  `speaker_for_frame_range()`; a query starting before `probs_base_` now
  resolves against `frozen_segs_` (nearest segment by time if it falls in a
  gap) instead of clamping into the live window.
- **Verified against:** a real 2h39m two-speaker interview that previously
  collapsed onto one speaker for ~140 of 160 minutes; after the fix, both
  speakers alternate proportionally across every 10-minute window of the
  full file. Full detail and the original bug report live in
  `NeMoSpeech`'s project notes (see companion link above).
- **Upstreaming status:** not yet proposed to `NVIDIA/NeMo-Speech.cpp`. This
  fork stays ahead of upstream on this fix until/unless that happens.

### 2026-09-13 — Live speaker-change event on the realtime WebSocket

- **Upstream base:** `23a06766d248c4d208f17b4ca3bf50ce2d01b131` (`main` at
  the commit this branch forked from).
- **PR:** (pending — opened by the controller after this task)
- **Files:** `src/asr/diar/diar_pipeline.{h,cpp}`, `src/asr/recognizer.{h,cpp}`,
  `server/http/http_server.cpp`, `tests/cpp/asr/test_diar_speaker_change.cpp` (new),
  `tests/cpp/asr/test_diar_recognizer.cpp`, `tests/cpp/asr/CMakeLists.txt`,
  `tests/integration/http_conformance_test.py`.
- **What:** `/v1/audio/transcriptions/realtime` only reported diarization
  results as word-level speaker tags on `.completed` events -- i.e. only once
  a client sent `input_audio_buffer.commit`. `NemoSpeech`'s client-side
  silence-based segmentation could therefore go up to 15s without
  finalizing during continuous speech, since there was no way to know a
  speaker change had happened sooner. Added a new, additive
  `conversation.item.speaker_diarization.changed` event, built on
  `DiarStream::segments()`'s existing incremental, hysteresis-cleaned
  segment tracking, so a client can commit immediately on a real speaker
  change instead.
- **Fix:** see `docs/superpowers/specs/2026-09-13-live-speaker-change-event-design.md`
  for the full design.
- **Verified against:** `tests/cpp/asr/test_diar_speaker_change.cpp` (pure
  logic, synthetic segments), an extended `test_diar_recognizer.cpp` run
  against a real multi-speaker fixture, and the extended
  `tests/integration/http_conformance_test.py` WebSocket assertions.
- **Upstreaming status:** not yet proposed to `NVIDIA/NeMo-Speech.cpp`.
