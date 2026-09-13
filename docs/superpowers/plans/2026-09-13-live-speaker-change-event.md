# Live Speaker-Change Event Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a new, additive WebSocket event the instant the realtime diarizer confirms a speaker change, so a client can finalize immediately instead of waiting on a silence timer.

**Architecture:** `DiarStream::segments()` already computes hysteresis-cleaned, confirmed speaker segments incrementally on a live stream. Add a pure comparison function (`detect_speaker_change`) that flags when the most recent segment's speaker differs from the last one reported; wrap it in a new `RecognitionStream::poll_speaker_change()` method; call that from the realtime WS handler (`http_server.cpp`) after every audio push and emit a new `conversation.item.speaker_diarization.changed` event when it fires.

**Tech Stack:** C++17, the existing `nemo_speech::asr` library (`src/asr/`), `json.h`'s `Value` type for WS event construction, CTest for unit tests, the existing Python `websockets`-based integration test (`tests/integration/http_conformance_test.py`).

**Spec:** `docs/superpowers/specs/2026-09-13-live-speaker-change-event-design.md`

## Global Constraints

- Purely additive protocol change: no existing WS event type, field, or behavior changes for any client of `/v1/audio/transcriptions/realtime` (including the built-in browser playground).
- `speaker` fields exposed over the wire are 1-based, matching the existing `word.speaker_tag`/`item["speaker"] = segment.speaker + 1` convention already used elsewhere in `http_server.cpp`. Internally, `DiarSegment::speaker` stays 0-based, matching its existing meaning everywhere else in `diar_pipeline.{h,cpp}`.
- `SilenceCommitTrigger`-equivalent behavior (this repo has no such class; this is about not changing existing commit/finalization behavior) must be unaffected: this change only ever *adds* a new event type, never suppresses or alters `delta`/`completed`/`session.*`/`input_audio_buffer.*`/`error` events.
- Every new/modified `.cpp`/`.h` file keeps the existing SPDX header (`SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` / `SPDX-License-Identifier: Apache-2.0`) — copy verbatim from a neighboring file in the same directory.
- Follow `CONTRIBUTING.md`: build with `scripts/configure.sh cpu-asr -DNEMO_SPEECH_BUILD_TESTS=ON` + `cmake --build --preset cpu-asr`, run new unit tests with `ctest --test-dir build/cpu-asr --output-on-failure`.

---

## Task 1: Pure speaker-change detection logic (`diar_pipeline.{h,cpp}`)

**Files:**
- Modify: `src/asr/diar/diar_pipeline.h` (add after `speaker_for_frame_range`'s declaration, ~line 129, before `class DiarStream`)
- Modify: `src/asr/diar/diar_pipeline.cpp` (add implementation near `diar_segments_from_probs`)
- Create: `tests/cpp/asr/test_diar_speaker_change.cpp`
- Modify: `tests/cpp/asr/CMakeLists.txt` (register the new test)

**Interfaces:**
- Produces: `struct DiarSpeakerChange { int speaker; double start_time; };` and `std::optional<DiarSpeakerChange> detect_speaker_change(const std::vector<DiarSegment>& segments, std::optional<int> last_reported);` — both in `namespace nemo_speech::asr`. `segments` is assumed sorted ascending by `t0` (guaranteed by `DiarStream::segments()`/`diar_segments_from_probs`, which already sort this way). `last_reported` and `DiarSpeakerChange::speaker` are **0-based**, matching `DiarSegment::speaker` — callers convert to the wire's 1-based convention at the JSON boundary (Task 3), not here.

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/asr/test_diar_speaker_change.cpp`:

```cpp
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Unit tests for detect_speaker_change(): the pure comparison logic behind
// the realtime WebSocket's speaker_diarization.changed event. Runs entirely
// on synthetic DiarSegment lists, no model required -- mirrors the
// speaker_for_frame_range test pattern in test_diar_frame_lookup.cpp.
#include <cstdio>
#include <optional>
#include <vector>

#include "diar_pipeline.h"

using namespace nemo_speech::asr;

namespace {

bool
test_first_segment_ever_is_reported() {
    // Nothing reported yet (nullopt): the very first segment always counts
    // as a change, since there's no prior speaker to compare against.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}};
    const auto got = detect_speaker_change(segs, std::nullopt);
    if (!got || got->speaker != 0 || got->start_time != 0.0) {
        std::fprintf(stderr, "[FAIL] expected first segment reported as speaker 0 at t=0.0\n");
        return false;
    }
    return true;
}

bool
test_same_speaker_continuing_is_not_reported() {
    // The most recent segment's speaker matches what was already reported:
    // this is the same speaker's turn continuing, not a change.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {2.0, 5.0, 0}};
    const auto got = detect_speaker_change(segs, /*last_reported=*/0);
    if (got) {
        std::fprintf(stderr, "[FAIL] expected no change when speaker is unchanged\n");
        return false;
    }
    return true;
}

bool
test_real_change_is_reported_with_new_segments_start_time() {
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {2.0, 5.0, 1}};
    const auto got = detect_speaker_change(segs, /*last_reported=*/0);
    if (!got || got->speaker != 1 || got->start_time != 2.0) {
        std::fprintf(stderr, "[FAIL] expected speaker 1 reported at t=2.0\n");
        return false;
    }
    return true;
}

bool
test_empty_segments_is_not_reported() {
    // No confirmed segments yet at all (e.g. right at stream start, before
    // onset hysteresis confirms anything).
    const std::vector<DiarSegment> segs = {};
    const auto got = detect_speaker_change(segs, std::nullopt);
    if (got) {
        std::fprintf(stderr, "[FAIL] expected no change when there are no segments\n");
        return false;
    }
    return true;
}

bool
test_third_speaker_after_two_reported_changes() {
    // Guards against an implementation that only ever compares against the
    // *first* reported speaker instead of the *last* one.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {2.0, 4.0, 1}, {4.0, 6.0, 2}};
    const auto got = detect_speaker_change(segs, /*last_reported=*/1);
    if (!got || got->speaker != 2 || got->start_time != 4.0) {
        std::fprintf(stderr, "[FAIL] expected speaker 2 reported at t=4.0\n");
        return false;
    }
    return true;
}

}  // namespace

int
main() {
    bool ok = true;
    ok &= test_first_segment_ever_is_reported();
    ok &= test_same_speaker_continuing_is_not_reported();
    ok &= test_real_change_is_reported_with_new_segments_start_time();
    ok &= test_empty_segments_is_not_reported();
    ok &= test_third_speaker_after_two_reported_changes();
    if (!ok) {
        std::fprintf(stderr, "[FAIL] detect_speaker_change\n");
        return 1;
    }
    std::printf("[PASS] detect_speaker_change reports real speaker transitions only\n");
    return 0;
}
```

Register it in `tests/cpp/asr/CMakeLists.txt`, right after the existing `test_diar_frame_lookup` block:

```cmake
add_executable(test_diar_speaker_change test_diar_speaker_change.cpp)
target_link_libraries(test_diar_speaker_change PRIVATE nemo_speech_asr)
add_test(NAME diar_speaker_change COMMAND test_diar_speaker_change)
```

- [ ] **Step 2: Configure the test build and verify the new test fails to build (the declarations don't exist yet)**

```bash
git submodule update --init ggml llama.cpp
scripts/configure.sh cpu-asr -DNEMO_SPEECH_BUILD_TESTS=ON
cmake --build --preset cpu-asr --target test_diar_speaker_change
```

Expected: build FAILS with an error that `detect_speaker_change`/`DiarSpeakerChange` are undeclared.

- [ ] **Step 3: Add the declaration to `diar_pipeline.h`**

Insert immediately after `speaker_for_frame_range`'s declaration (the block ending `int64_t start_frame, int64_t end_frame);` just before `// Per-stream streaming state + timeline.\nclass DiarStream {`):

```cpp
// Speaker-change detection for the realtime WebSocket handler
// (conversation.item.speaker_diarization.changed): compares the most
// recent confirmed segment's speaker against `last_reported` (nullopt if
// nothing has been reported yet for this stream) and returns the new
// speaker + that segment's start time iff they differ. `segments` must be
// sorted ascending by t0 (DiarStream::segments() already guarantees this).
// Both `last_reported` and the returned `speaker` are 0-based, matching
// DiarSegment::speaker -- callers convert to the wire's 1-based convention
// themselves (see http_server.cpp's existing `segment.speaker + 1`).
struct DiarSpeakerChange {
    int speaker;
    double start_time;
};
std::optional<DiarSpeakerChange> detect_speaker_change(
    const std::vector<DiarSegment>& segments, std::optional<int> last_reported);
```

Add `#include <optional>` to `diar_pipeline.h`'s includes if not already present.

- [ ] **Step 4: Implement it in `diar_pipeline.cpp`**

Add near `diar_segments_from_probs` (e.g. directly above it):

```cpp
std::optional<DiarSpeakerChange>
nemo_speech::asr::detect_speaker_change(
    const std::vector<DiarSegment>& segments, std::optional<int> last_reported) {
    if (segments.empty())
        return std::nullopt;
    const DiarSegment& latest = segments.back();
    if (last_reported.has_value() && *last_reported == latest.speaker)
        return std::nullopt;
    return DiarSpeakerChange{latest.speaker, latest.t0};
}
```

- [ ] **Step 5: Build and run the test**

```bash
cmake --build --preset cpu-asr --target test_diar_speaker_change
ctest --test-dir build/cpu-asr -R diar_speaker_change --output-on-failure
```

Expected: PASS, printing `[PASS] detect_speaker_change reports real speaker transitions only`.

- [ ] **Step 6: Commit**

```bash
git add src/asr/diar/diar_pipeline.h src/asr/diar/diar_pipeline.cpp \
        tests/cpp/asr/test_diar_speaker_change.cpp tests/cpp/asr/CMakeLists.txt
git commit -m "feat(diar): add detect_speaker_change() for live speaker-change detection"
```

---

## Task 2: `RecognitionStream::poll_speaker_change()`

**Files:**
- Modify: `src/asr/recognizer.h` (public method after `finish()` ~line 85; private member after `diar_` ~line 97)
- Modify: `src/asr/recognizer.cpp` (implementation, near `force_endpoint()`)
- Modify: `tests/cpp/asr/test_diar_recognizer.cpp` (exercise the new method end-to-end with a real model + wav, matching this file's existing argv-driven convention)

**Interfaces:**
- Consumes: `DiarSpeakerChange`, `detect_speaker_change` (Task 1).
- Produces: `std::optional<DiarSpeakerChange> RecognitionStream::poll_speaker_change();` — public method on the existing `RecognitionStream` class. Safe to call any time after construction; returns `std::nullopt` whenever `diar_` is null (diarization not enabled for this stream).

- [ ] **Step 1: Add the declaration to `recognizer.h`**

In the public section, immediately after:
```cpp
    Result finish();
```
add:
```cpp
    // Poll for a confirmed speaker change since the last call (or since
    // stream start, for the first call). Always returns nullopt if
    // diarization isn't enabled for this stream. See detect_speaker_change()
    // in diar_pipeline.h for the comparison semantics.
    std::optional<DiarSpeakerChange> poll_speaker_change();
```

In the private section, immediately after:
```cpp
    std::unique_ptr<DiarStream> diar_;
```
add:
```cpp
    // 0-based; matches DiarSegment::speaker. nullopt = nothing reported yet.
    std::optional<int> last_reported_speaker_;
```

- [ ] **Step 2: Implement it in `recognizer.cpp`**

Add right after `RecognitionStream::force_endpoint()`'s closing brace:

```cpp
std::optional<DiarSpeakerChange>
RecognitionStream::poll_speaker_change() {
    if (!diar_)
        return std::nullopt;
    auto change = detect_speaker_change(diar_->segments(), last_reported_speaker_);
    if (change)
        last_reported_speaker_ = change->speaker;
    return change;
}
```

- [ ] **Step 3: Build**

```bash
cmake --build --preset cpu-asr --target nemo_speech_asr
```

Expected: builds cleanly (no test to run yet for this glue method in isolation — see Step 4 for end-to-end verification, following this repo's existing convention of testing `RecognitionStream`-level diarization behavior with a real model + wav rather than mocking `Recognizer`, e.g. `test_diar_recognizer.cpp`).

- [ ] **Step 4: Extend `test_diar_recognizer.cpp` to exercise `poll_speaker_change()` end-to-end**

In the existing streaming loop:
```cpp
    const size_t push = 160 * 16;  // 160 ms
    for (size_t off = 0; off < audio.size(); off += push) {
        stream->push(audio.data() + off, std::min(push, audio.size() - off));
        while (auto r = stream->next()) {
            take(*r);
            if (!r->is_final)
                break;
        }
    }
```
add a `speaker_changes` counter and poll after each push:
```cpp
    int speaker_changes = 0;
    const size_t push = 160 * 16;  // 160 ms
    for (size_t off = 0; off < audio.size(); off += push) {
        stream->push(audio.data() + off, std::min(push, audio.size() - off));
        while (auto r = stream->next()) {
            take(*r);
            if (!r->is_final)
                break;
        }
        if (auto change = stream->poll_speaker_change()) {
            speaker_changes++;
            std::printf(
                "[diar-rec] speaker change -> speaker %d at %.2fs\n", change->speaker + 1,
                change->start_time);
        }
    }
```
and extend the final pass/fail check (currently `if (words.empty() || untagged > 0 || static_cast<int>(tag_counts.size()) < min_speakers)`) to also require at least one detected change when more than one speaker is expected:
```cpp
    std::printf("[diar-rec] %d speaker changes detected\n", speaker_changes);
    if (words.empty() || untagged > 0 || static_cast<int>(tag_counts.size()) < min_speakers ||
        (min_speakers > 1 && speaker_changes < 1)) {
        std::printf("[diar-rec] FAIL\n");
        return 1;
    }
```

- [ ] **Step 5: Build and run manually against a real fixture**

This test takes real model/audio arguments and is not part of default `ctest` (matching its existing convention — it's a manual/local developer tool, not wired into any CI workflow; confirmed no `.github/workflows/*.yml` references it). It's also not run against this repo's only committed wav (`test_files/asr/wav/test/jfk.wav`, a solo JFK speech recording) — its own default `min_speakers=2` already requires a genuinely multi-speaker file to pass at all, with or without this change. Supply your own short multi-speaker recording (two people talking, or two single-speaker clips concatenated) and real ASR/diarizer GGUF paths (see `NemoSpeech`'s README "Verified environment" section for where `nemo-speech pull` caches them, typically under `~/.cache/nemo-speech/models/`):

```bash
cmake --build --preset cpu-asr --target test_diar_recognizer
./build/cpu-asr/tests/cpp/asr/test_diar_recognizer <asr.gguf> <diar.gguf> <multi-speaker.wav>
```

Expected: output includes at least one `speaker change ->` line and ends with `[diar-rec] OK`.

- [ ] **Step 6: Commit**

```bash
git add src/asr/recognizer.h src/asr/recognizer.cpp tests/cpp/asr/test_diar_recognizer.cpp
git commit -m "feat(asr): add RecognitionStream::poll_speaker_change()"
```

---

## Task 3: Wire the event into the realtime WS handler

**Files:**
- Modify: `server/http/http_server.cpp` (`append_audio` lambda, ~line 1079-1097)
- Modify: `tests/integration/http_conformance_test.py` (extend the existing WebSocket section, ~line 300-357)

**Interfaces:**
- Consumes: `RecognitionStream::poll_speaker_change()` (Task 2).
- Produces: the wire event itself — `{"type": "conversation.item.speaker_diarization.changed", "speaker": <1-based int>, "start_time": <seconds float>}`, sent via the handler's existing `send(Value)` lambda.

- [ ] **Step 1: Write the failing integration assertion**

In `tests/integration/http_conformance_test.py`, in the WebSocket section (the `with connect(...) as websocket:` block that streams `pcm` and waits for `input_audio_buffer.committed`), after the existing:
```python
        completed = [
            event for event in events if event.get("type", "").endswith("transcription.completed")
        ]
```
add:
```python
        speaker_changes = [
            event for event in events
            if event.get("type") == "conversation.item.speaker_diarization.changed"
        ]
```
and after the existing `if args.diar_model:` block that checks `"WebSocket speaker tags"`, add:
```python
        if args.diar_model:
            require(speaker_changes, "WebSocket speaker-change event")
            for change in speaker_changes:
                require(isinstance(change.get("speaker"), int) and change["speaker"] > 0,
                        "speaker-change event has a 1-based speaker id")
                require(isinstance(change.get("start_time"), (int, float)),
                        "speaker-change event has a start_time")
```

- [ ] **Step 2: Run the conformance test to verify the new assertion fails**

This script is a standalone local tool (not wired into any CI workflow, confirmed by grepping `.github/workflows/*.yml`) that starts the server itself — it isn't run against a fixed fixture. `--audio` must point at a genuinely multi-speaker recording for the new assertion to mean anything (the repo's only committed wav, `test_files/asr/wav/test/jfk.wav`, is a solo speech and would make this new assertion fail even on a correct implementation, same caveat as Task 2 Step 5 — supply your own):

```bash
python3 tests/integration/http_conformance_test.py \
  --binary build/cpu-asr/bin/nemo-speech \
  --asr-model <asr.gguf> \
  --diar-model <diar.gguf> \
  --audio <multi-speaker.wav>
```

Expected: FAILS on `"WebSocket speaker-change event"` (no such event exists yet).

- [ ] **Step 3: Implement the server-side emission in `http_server.cpp`**

Modify the `append_audio` lambda (currently ending):
```cpp
            auto append_audio = [&](const std::string& pcm) {
                if (pcm.size() > this->config.max_upload_bytes -
                                     std::min(audio_bytes, this->config.max_upload_bytes))
                    throw std::invalid_argument(
                        "realtime audio exceeds the configured upload limit");
                audio_bytes += pcm.size();
                const auto samples = pcm16_samples(pcm);
                if (samples.empty())
                    return true;
                ensure_stream();
                stream->push(samples.data(), samples.size(), sample_rate);
                while (auto result = stream->next()) {
                    if (!emit(*result))
                        return false;
                    if (!result->is_final)
                        break;
                }
                return true;
            };
```
to:
```cpp
            auto append_audio = [&](const std::string& pcm) {
                if (pcm.size() > this->config.max_upload_bytes -
                                     std::min(audio_bytes, this->config.max_upload_bytes))
                    throw std::invalid_argument(
                        "realtime audio exceeds the configured upload limit");
                audio_bytes += pcm.size();
                const auto samples = pcm16_samples(pcm);
                if (samples.empty())
                    return true;
                ensure_stream();
                stream->push(samples.data(), samples.size(), sample_rate);
                while (auto result = stream->next()) {
                    if (!emit(*result))
                        return false;
                    if (!result->is_final)
                        break;
                }
                if (auto change = stream->poll_speaker_change()) {
                    Value changed(Value::Object{});
                    changed["type"] = "conversation.item.speaker_diarization.changed";
                    changed["speaker"] = change->speaker + 1;  // wire convention is 1-based
                    changed["start_time"] = change->start_time;
                    if (!send(std::move(changed)))
                        return false;
                }
                return true;
            };
```

- [ ] **Step 4: Build the server**

```bash
cmake --build --preset cpu-asr --target nemo_speech_server
```

- [ ] **Step 5: Re-run the conformance test**

Same invocation as Step 2. Expected: PASSES, including the new `"WebSocket speaker-change event"` check.

- [ ] **Step 6: Manual end-to-end sanity check**

Start the real server used by `NemoSpeech` and confirm the new event actually appears on the wire during natural conversation (not just the fixture wav):

```bash
build/cuda-asr/bin/nemo-speech serve --asr-model nemotron-3.5 --diar-model sortformer --port 8080
```

Connect with any WS client (e.g. `websocat`, or the browser playground's dev console at `http://127.0.0.1:8080/`) with `speaker_diarization: true`, speak with a genuine pause and a change of voice/file, and confirm `conversation.item.speaker_diarization.changed` events appear before any `input_audio_buffer.commit` is sent.

- [ ] **Step 7: Commit**

```bash
git add server/http/http_server.cpp tests/integration/http_conformance_test.py
git commit -m "feat(server): emit speaker_diarization.changed on the realtime WebSocket"
```

---

## Task 4: Fork divergence log + PR

**Files:**
- Modify: `FORK_CHANGES.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by later tasks — this is the last task.

- [ ] **Step 1: Add a new divergence entry**

Append to `FORK_CHANGES.md`, following the exact structure of the existing 2026-09-10 entry (upstream base commit, PR link placeholder to fill in after Step 2, files touched, what/why, verification, upstreaming status):

```markdown
### 2026-09-13 — Live speaker-change event on the realtime WebSocket

- **Upstream base:** `main` at the commit this branch forked from (run
  `git merge-base feat/live-speaker-change-event main` to confirm before
  filling this in).
- **PR:** (fill in after opening it in Step 2)
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
```

- [ ] **Step 2: Commit, push, open a PR, and merge**

```bash
git add FORK_CHANGES.md
git commit -m "docs: log the live speaker-change event divergence"
git push -u origin feat/live-speaker-change-event
gh pr create --title "feat: live speaker-change event on the realtime WebSocket" --body "$(cat <<'EOF'
## Summary
- Adds a new, additive `conversation.item.speaker_diarization.changed` WebSocket event, built on `DiarStream::segments()`'s existing incremental segment tracking.
- Lets a realtime client finalize immediately on a confirmed speaker change instead of waiting on a silence timer.
- No existing event types, fields, or client behavior change.

## Test plan
- [x] `ctest --test-dir build/cpu-asr -R diar_speaker_change --output-on-failure`
- [x] `test_diar_recognizer` run against a real multi-speaker fixture, confirms `speaker change ->` lines and `[diar-rec] OK`
- [x] `tests/integration/http_conformance_test.py` WebSocket section, including the new speaker-change assertions

See `docs/superpowers/specs/2026-09-13-live-speaker-change-event-design.md` for the full design.
EOF
)"
gh pr merge --merge
```

Then go back and fill in the PR link + confirmed upstream-base commit in `FORK_CHANGES.md`'s new entry from Step 1 (a small follow-up commit on `main` after merge, matching how the existing 2026-09-10 entry records its actual merge commit).
