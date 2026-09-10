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
