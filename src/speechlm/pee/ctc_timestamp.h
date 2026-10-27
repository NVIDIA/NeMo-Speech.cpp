// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime.h"
#include "timestamp_alignment.h"

namespace nemo_speech::asr {
struct SpeechLmDetails;
}

namespace nemo_speech::speechlm {

class CtcEvidence;

// One timestamp request's encoder evidence for the whole file (frame-major).
struct TimestampEvidence {
    // The head's transcript-independent output for the joined window cores, on the device
    // (CtcTimestampModel::reserve / stage / finish at encode time), one row per CTC frame.
    std::shared_ptr<const CtcEvidence> head;
    // Encoder frames on the 80 ms grid, ceil(T_mel / 8) (= head->encoder_frames()).
    int encoder_frames = 0;
    // Frames the alignment runs on: the head's CTC frames, upsampling() x encoder_frames (40 ms
    // with the released head).
    int ctc_frames() const;
    // Speaker activity on the 80 ms grid (activity_frames x speakers): 8:1 means of the 10 ms
    // activity. The alignment fits it to encoder_frames rows as NeMo's encoder does (the last row
    // repeated, or cut), then resamples it to ctc_frames() rows (NeMo's _resample_speaker_probs).
    const float* activity = nullptr;
    int activity_frames = 0;
    // 10 ms speaker activity of the valid mel frames (activity10_frames x speakers).
    const float* activity10 = nullptr;
    int activity10_frames = 0;
    int speakers = 0;
    // Audio duration in seconds (16 kHz samples / 16000).
    double duration_seconds = 0.0;
};

struct TimestampOptions {
    // Special tokens of the decoding tokenizer that the transcript may hold literally: removed
    // before alignment except speaker, event and style tags (NeMo's clean_sot_transcript).
    std::vector<std::string> special_tokens;
    // CTC head windows in encoder frames: core length and left / right context.
    int head_window = 500;
    int head_left_context = 50;
    int head_right_context = 50;
    // Round the head's log-probabilities to bfloat16 before alignment, as a bf16 head would
    // produce.
    bool bf16_emissions = true;
};

// The head's transcript-independent output for one request: per CTC frame the adapter output h
// (dimension() floats, kept on the device) and the log-sum-exp and maximum of the full-vocabulary
// logits (host). Emissions of any classes are then one small projection, W[c].h[t] + b[c] - lse[t].
// Two clocks: the request's joined ASR states are encoder frames (80 ms); the head upsamples each
// into upsampling() CTC frames (1 or 2: 80 or 40 ms), and h, lse(), max_logit() and emissions()
// are on that grid.
class CtcEvidence {
   public:
    ~CtcEvidence();
    CtcEvidence(const CtcEvidence&) = delete;
    CtcEvidence& operator=(const CtcEvidence&) = delete;
    // CTC frames: upsampling() x encoder_frames().
    int frames() const { return encoder_frames_ * upsampling_; }
    // Encoder frames: the joined ASR state rows reserve() / stage() take.
    int encoder_frames() const { return encoder_frames_; }
    int upsampling() const { return upsampling_; }
    // Waits for the head; rethrows its failure.
    void wait() const {
        if (!ready_.valid())
            throw std::runtime_error("CTC evidence was not finished");
        ready_.get();
    }
    // Per CTC frame log-sum-exp and maximum of the logits (valid after wait()).
    const std::vector<float>& lse() const { return lse_; }
    const std::vector<float>& max_logit() const { return max_; }

   private:
    friend class CtcTimestampModel;
    struct Rows;
    CtcEvidence() = default;
    int encoder_frames_ = 0, upsampling_ = 1;
    std::unique_ptr<Rows> staging_, h_;  // device rows: joined ASR states (encoder frames), then h
    std::vector<float> lse_, max_;
    std::shared_future<void> ready_;
};

class CtcTimestampModel {
   public:
    CtcTimestampModel(ggml_runtime::BackendManager& bm, const std::string& path);
    ~CtcTimestampModel();
    int dimension() const;
    int blank_id() const;
    // CTC frames per encoder frame (ctc.upsampling_factor: 2 for the released 40 ms head, 1 for an
    // adapter without upsampling).
    int upsampling() const;
    // Alignment model name reported in the response: the GGUF's ctc.model_name (the checkpoint's),
    // else a name derived from the vocabulary size.
    std::string model_name() const;

    // ---- evidence at encode time ----
    // Reserve device rows for one request's joined ASR states (`encoder_frames` rows of
    // dimension()).
    std::shared_ptr<CtcEvidence> reserve(int encoder_frames);
    // Copy `rows` rows (dimension() floats each) of `src` from row `src_row` into the evidence's
    // joined states at `dst_row`, queued on `backend` (the producer's stream). Synchronize that
    // backend before finish().
    void stage(
        ggml_backend_t backend, const ggml_runtime::DeviceTensor& src, size_t src_row,
        CtcEvidence& evidence, int dst_row, int rows);
    // Run the head over staged evidence (windows batched across requests) in the background: h
    // stays on the device, the joined states are released, and each evidence is ready when it is
    // done. Windows (TimestampOptions::head_*) are in encoder frames; each window's core becomes
    // upsampling() times as many CTC frames.
    void finish(
        const std::vector<std::shared_ptr<CtcEvidence>>& evidence,
        const TimestampOptions& options = {});
    // Log-probabilities of each request's `columns` (vocabulary ids) for every CTC frame of its
    // evidence (frames() x columns.size(), frame-major), requests batched together.
    std::vector<std::vector<float>> emissions(
        const std::vector<const CtcEvidence*>& evidence,
        const std::vector<std::vector<int>>& columns, const TimestampOptions& options = {});

    // Word timestamps, speaker attribution and diarization segments of one request, as NeMo's
    // MultiSpeakerSOTWordTimestampAligner and the vLLM plugin's public result give them
    // (timestamp_alignment.h): words of every speaker stream ordered by (start, end) with times in
    // milliseconds, speaker_slot the stream's Sortformer column, and the 10 ms diarization
    // segments. Fills the alignment fields of `details` and leaves the rest alone. A speaker stream
    // without a legal path is left out (a warning names it) and the rest stay aligned, as in NeMo.
    // An alignment failure does not throw: the status becomes "unavailable" with failure_reason
    // "empty_transcript", "ctc_evidence_unavailable" or "ctc_alignment_failed" (NeMo rejects the
    // transcript; the diarization segments are still filled), and the returned string carries the
    // reason for the log ("" when aligned). Thread-safe.
    std::string word_timestamps(
        const TimestampEvidence& evidence, const std::string& transcript,
        asr::SpeechLmDetails& details, const TimestampOptions& options = {});
    // The same for several requests (one head pass over all of them): one reason per request.
    std::vector<std::string> word_timestamps(
        const std::vector<TimestampEvidence>& evidence, const std::vector<std::string>& transcripts,
        const std::vector<asr::SpeechLmDetails*>& details, const TimestampOptions& options = {});

    // The head's log-probabilities of `columns` (vocabulary ids) for every CTC frame of host joined
    // states (upsampling() x frames x columns.size(), frame-major; `frames` encoder frames): the
    // head over the states, then emissions().
    std::vector<float> head_log_probs(
        const float* states, int frames, const std::vector<int>& columns,
        const TimestampOptions& options = {});
    // The same for several requests, their windows batched together.
    struct HeadRequest {
        const float* states;
        int frames;  // encoder frames
        std::vector<int> columns;
    };
    std::vector<std::vector<float>> head_log_probs(
        const std::vector<HeadRequest>& requests, const TimestampOptions& options = {});

    // The speaker streams and CTC targets of a transcript with this head's tokenizer
    // (timestamps::alignment_targets, every stream checked against `ctc_frames`); throws
    // std::runtime_error when NeMo would reject the transcript.
    ::nemo_speech::speechlm::timestamps::AlignmentTargets targets(
        const std::string& transcript, int ctc_frames = std::numeric_limits<int>::max()) const;

    // ---- per-window interface (used by tests) ----
    // The adapter output h of a batch of windows (D x upsampling() * frames x B; `frames` and
    // `lengths` in encoder frames, padded keys masked).
    std::vector<float> bridge(const float* states, int frames, const std::vector<int>& lengths);
    // Logits of `token_ids` for `frames` rows of h (CTC frames).
    std::vector<float> project(
        const std::vector<float>& states, int frames, const std::vector<int>& token_ids);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace nemo_speech::speechlm
