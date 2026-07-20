// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Trailing-silence EOU policy over a decoded millisecond timeline. It fires
// once at the threshold and re-arms only after speech advances.
#pragma once

#include <string>

#include "parameter_parser.h"

namespace nemo_speech::asr {

struct VadEndpointerCfg {
    bool enable = false;
    // EOU silence signal. Default token-silence (decoder token timeline), so
    // endpointing works with no VAD model loaded. Set vad_based=true (and load
    // a VAD model) for riva's vad_based_eou=true behaviour off the Silero
    // speech/silence timeline.
    bool vad_based = false;
    float stop_history_eou_ms = 800.0f;  // matches riva RNNTPostprocessor::Config

    void Register(common::ParameterParser& p) {
        // bool fields take bare-flag CLI semantics (--endpointing, --vad-based-eou).
        p.Register("enable", &enable, "Enable mid-stream endpointing (EOU)", {"--endpointing"});
        p.Register(
            "vad_based", &vad_based, "EOU rides the VAD timeline (else decoder token-silence)",
            {"--vad-based-eou"});
        p.Register(
            "stop_history_eou_ms", &stop_history_eou_ms, "Trailing-silence EOU threshold (ms)",
            {"--stop-history-eou-ms"});
    }
};

class VadEndpointer {
   public:
    explicit VadEndpointer(VadEndpointerCfg cfg) : cfg_(cfg) {}

    // Advance the timeline. `now_ms` is the decoded audio time so far (the
    // decode clock, NOT the raw-audio frontier - the two can differ by the
    // runner's lookahead); `last_speech_ms` is the end time of the most recent
    // speech at or before now_ms (VAD segment end, or last-token frame).
    // Returns true exactly once when the trailing silence (now_ms -
    // last_speech_ms) first reaches the threshold; re-arms only after
    // `last_speech_ms` advances (speech resumed). Threshold fires require
    // cfg.enable; force() does not.
    bool poll(double now_ms, double last_speech_ms);

    // Client force_eou (riva runtime_config["force_eou"]): the next poll()
    // fires regardless of silence/threshold (and regardless of `enable`),
    // then re-arms on the next speech.
    void force() { forced_ = true; }

    // New stream: clear all latched state. The per-request threshold override
    // survives (like the runner's opts_); a later set_stop_history_eou_ms(<=0)
    // clears it.
    void reset();

    // Per-request override of the EOU silence threshold (riva
    // custom_configuration["stop_history_eou"], ms). <= 0 clears the override,
    // restoring the configured threshold.
    void set_stop_history_eou_ms(float ms) { override_ms_ = ms; }

    // Effective threshold: per-request override when set, else the configured value.
    float stop_history_eou_ms() const {
        return override_ms_ > 0.0f ? override_ms_ : cfg_.stop_history_eou_ms;
    }

    const VadEndpointerCfg& config() const { return cfg_; }

   private:
    VadEndpointerCfg cfg_;
    float override_ms_ = -1.0f;    // <= 0 = no per-request override
    double last_speech_ms_ = 0.0;  // max last_speech_ms observed
    bool armed_ = false;           // speech seen since the last fire
    bool forced_ = false;          // force() pending
};

}  // namespace nemo_speech::asr
