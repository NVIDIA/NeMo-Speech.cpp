// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "subtitles.h"
#include "types.h"

namespace nemo_speech::cli {

using asr::Alternative;
using asr::Result;

struct LivePartialPresentation {
    std::string text;
    // Replaceable snapshot, not committed text.
    std::vector<subtitle::SpeakerTurn> turns;
};

// Groups live results into speaker turns. Only finals update committed state;
// adjacent same-speaker finals merge into one turn.
class LiveTranscriptPresenter {
   public:
    explicit LiveTranscriptPresenter(bool diarize) : diarize_(diarize) {}

    LivePartialPresentation present_partial(const Alternative& alternative) const;
    std::vector<subtitle::SpeakerTurn> present_final(const Result& result);
    // Release the current speaker turn at end-of-stream.
    std::vector<subtitle::SpeakerTurn> finish();
    // Append late punctuation to the open turn; false if none is open.
    bool attach_late_punctuation(const std::string& punctuation);

    const subtitle::SpeakerTurn* current_turn() const {
        return pending_turn_valid_ ? &pending_turn_ : nullptr;
    }

   private:
    std::vector<subtitle::SpeakerTurn> settle_pending();

    bool diarize_ = false;
    bool pending_turn_valid_ = false;
    subtitle::SpeakerTurn pending_turn_;
};

// Holds ASR finals until their speaker labels are stable; each update re-tags
// every buffered final. stable_before_seconds is the diarizer's frontier.
class LiveDiarizationBuffer {
   public:
    void push(Result result);
    std::vector<Result> update(
        double stable_before_seconds, const std::function<void(Result&)>& retag);
    LivePartialPresentation preview(
        const LiveTranscriptPresenter& committed, const Alternative* partial = nullptr,
        double stable_before_seconds = std::numeric_limits<double>::infinity()) const;
    // Append late punctuation to the newest buffered final; false if empty.
    bool attach_late_punctuation(const std::string& punctuation);

   private:
    std::deque<Result> pending_;
};

}  // namespace nemo_speech::cli
