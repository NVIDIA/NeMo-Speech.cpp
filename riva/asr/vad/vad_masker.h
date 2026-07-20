// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// VadMasker - per-stream feature masker over a SileroVad's speech/silence
// timeline.
//
// VAD masking follows four steps:
//   1. binarize  - onset/offset hysteresis -> per-frame raw speech bit.
//                  (computed by SileroVad; shared with endpointing.)
//   2. pad       - extend each speech segment's edges by pad_onset/pad_offset.
//   3. merge     - silence gaps shorter than min_duration_off are not masked.
//   4. mask      - frames outside the resulting speech segments are set to
//                  mask_value in raw log-mel space before normalization.
//
// The Silero inference + binarized timeline live in SileroVad so a single
// inference feeds both masking and endpointing. The masker borrows the
// SileroVad (the runner owns it) and only applies the pad/merge/mask policy.
// Short-speech deletion is intentionally not applied; overlapping segments
// are handled by frame membership.
//
#pragma once

#include <cstdint>

#include "parameter_parser.h"
#include "silero_vad.h"

namespace nemo_speech::asr {

struct VadMaskerCfg {
    // Feature masking is independent of endpointing and disabled by default.
    bool mask_enable = false;
    float onset = 0.5f;                  // prob > onset  -> enter speech
    float offset = 0.3f;                 // prob < offset -> leave speech
    float pad_onset_ms = 200.0f;         // extend each segment start earlier
    float pad_offset_ms = 200.0f;        // extend each segment end later
    float min_duration_off_ms = 500.0f;  // silence gaps shorter than this are kept (not masked)
    float mask_value = -16.635f;         // raw log-mel fill for masked frames (pre-normalization)
    // Keep normalization consistent between masked and unmasked paths.
    float stddev_floor = 1e-5f;

    void Register(common::ParameterParser& p) {
        p.Register("onset", &onset, "VAD onset threshold", {"--vad-onset"});
        p.Register("offset", &offset, "VAD offset threshold", {"--vad-offset"});
        p.Register(
            "min_duration_off_ms", &min_duration_off_ms, "VAD min silence to merge (ms)",
            {"--vad-min-duration-off-ms"});
        p.Register("pad_onset_ms", &pad_onset_ms, "VAD speech-onset pad (ms)");
        p.Register("pad_offset_ms", &pad_offset_ms, "VAD speech-offset pad (ms)");
        p.Register(
            "pad_ms",
            [this](const std::string& v) {
                pad_onset_ms = pad_offset_ms = common::detail::to_float("vad.masker.pad_ms", v);
            },
            "VAD onset+offset pad (ms; sets both)", {"--vad-pad-ms"});
        p.Register(
            "stddev_floor", &stddev_floor, "Per-feature normalization stddev floor",
            {"--vad-stddev-floor"});
        p.Register(
            "mask_value", &mask_value, "Mel value written to masked frames", {"--vad-mask-value"});
        p.Register("mask_enable", &mask_enable, "Enable VAD feature masking", {"--vad-masking"});
    }
};

struct VadConfig {
    std::string model_path;  // empty = no VAD loaded
    VadMaskerCfg masker;
    void Register(common::ParameterParser& p) {
        p.Register("model_path", &model_path, "Silero VAD GGUF path", {"--vad-model"});
        p.Register("masker", masker);
    }
};

class VadMasker {
   public:
    // Borrows a SileroVad (owned by the runner, shared with the endpointer).
    // The vad's binarizer must be configured first (set_binarizer): its
    // hop_length drives the ms->mel-frame conversion of the pad/merge
    // thresholds. `n_mels` comes from the model's FE config (mask stride).
    VadMasker(SileroVad* vad, int n_mels, VadMaskerCfg cfg);

    // Mask `n_mel_frames` mel frames at `mel` (frame-major, n_mels floats per
    // frame) in place, using the SileroVad's recorded speech bits. The first
    // frame is global mel frame `first_global_frame` (= audio offset /
    // hop_length). Idempotent and order-independent: each frame's decision is a
    // pure function of the recorded speech bits in its neighbourhood, so this
    // is safe on overlapping windows (CTC) or monotonic appends (RNNT).
    void apply(float* mel, int n_mel_frames, int64_t first_global_frame);

    const VadMaskerCfg& config() const { return cfg_; }
    // Farthest decide_masked looks for neighbouring speech (mel frames). The
    // runner keeps at least this much committed timeline below the decode
    // cursor when trimming, so scans never cross the discarded base.
    int max_scan() const { return max_scan_; }

   private:
    // riva's segment rule: is silence frame g masked? (speech frames never are.)
    bool decide_masked(int64_t g) const;

    SileroVad* vad_;  // non-owning; the runner owns it (shared with endpointer)
    int n_mels_;
    VadMaskerCfg cfg_;
    // Thresholds converted from ms to mel frames.
    int pad_onset_frames_;
    int pad_offset_frames_;
    int merge_off_frames_;
    int max_scan_;  // how far decide_masked scans for neighbouring speech
};

}  // namespace nemo_speech::asr
