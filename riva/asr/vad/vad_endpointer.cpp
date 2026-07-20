// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "vad_endpointer.h"

namespace nemo_speech::asr {

bool
VadEndpointer::poll(double now_ms, double last_speech_ms) {
    // Speech advanced since the last poll -> (re)arm so the next silence run
    // can fire again. This is the latch that limits a long pause to one EOU.
    if (last_speech_ms > last_speech_ms_) {
        last_speech_ms_ = last_speech_ms;
        armed_ = true;
    }
    if (forced_) {
        forced_ = false;
        armed_ = false;
        return true;
    }
    if (!cfg_.enable || !armed_)
        return false;
    if (now_ms - last_speech_ms_ >= stop_history_eou_ms()) {
        armed_ = false;  // latch until speech resumes
        return true;
    }
    return false;
}

void
VadEndpointer::reset() {
    last_speech_ms_ = 0.0;
    armed_ = false;
    forced_ = false;
}

}  // namespace nemo_speech::asr
