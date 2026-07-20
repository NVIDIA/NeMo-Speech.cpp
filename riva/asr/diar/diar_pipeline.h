// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Standalone streaming diarization pipeline: audio in -> per-frame speaker
// probabilities (80 ms frames) + word/segment-level speaker attribution.
//
// The diarizer owns a model-specific mel frontend, slides an
// [lc | chunk | rc] window, and carries AOSC state between chunks. It can run
// independently or as an ASR sidecar.
//
// One SortformerModel (Session) can back many DiarStream instances; compute
// serializes on the BackendManager mutex.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "aosc_state.h"
#include "fe.h"
#include "parameter_parser.h"
#include "sortformer_model.h"

namespace nemo_speech::asr {

// Recognizer-level diarization config.
struct DiarConfig {
    std::string model_path;  // empty = diarization not available
    // Named geometry preset ("streaming" | "offline" - see DiarGeometry).
    // When set it REPLACES the individual geometry keys below
    // (the parser cannot tell explicit values from defaults, so mixing a
    // preset with individual overrides is not supported).
    std::string preset;
    DiarGeometry geometry;

    DiarGeometry resolved_geometry() const {
        return preset.empty() ? geometry : DiarGeometry::preset(preset);
    }

    void Register(common::ParameterParser& p) {
        p.Register("model_path", &model_path, "Sortformer diarizer GGUF path", {"--diar-model"});
        p.Register(
            "preset", &preset,
            "diarizer geometry preset (streaming | offline); "
            "overrides the individual geometry keys",
            {"--diar-preset"});
        p.Register(
            "chunk", &geometry.chunk_len, "diarizer chunk length (80ms frames)", {"--diar-chunk"});
        p.Register(
            "right_context", &geometry.chunk_right_context,
            "diarizer chunk right context (80ms frames)", {"--diar-rc"});
        p.Register(
            "left_context", &geometry.chunk_left_context,
            "diarizer chunk left context (80ms frames)", {"--diar-lc"});
        p.Register("fifo", &geometry.fifo_len, "diarizer FIFO length (frames)", {"--diar-fifo"});
        p.Register(
            "spkcache", &geometry.spkcache_len, "diarizer speaker cache length (frames)",
            {"--diar-spkcache"});
        p.Register(
            "update_period", &geometry.spkcache_update_period,
            "diarizer speaker cache update period (frames)", {"--diar-update-period"});
    }
};

// Shared, stream-independent resources: the model Session + its FE config.
class DiarModel {
   public:
    DiarModel(ggml_runtime::BackendManager& bm, const std::string& gguf_path);

    SortformerModel& model() { return model_; }
    MelSpectrogramExtractor& fe() { return fe_; }
    const SortformerModelConfig& cfg() const { return model_.cfg(); }

    // Full offline diarization: one forward pass over the whole file with
    // full self-attention and NO streaming state (NeMo streaming_mode=False;
    // the per-chunk graph with empty spkcache/fifo is exactly that forward).
    // Returns per-frame speaker probabilities, (n_frames x n_spk) frame-major,
    // one frame per 80 ms. Bounded by the rel-pos table: audio longer than
    // pos_emb_max_len encoder frames (5000 = ~6.6 min) throws - use
    // DiarStream for long-form.
    std::vector<float> diarize_offline(const float* audio, size_t n_samples, int64_t* n_frames);

   private:
    SortformerModel model_;
    MelSpecConfig fe_cfg_;
    MelSpectrogramExtractor fe_;
};

// Per-speaker segmentation parameters, NeMo ts_vad_post_processing semantics
// (pyannote binarize): onset/offset hysteresis -> pad segment edges -> fill
// gaps < min_duration_off -> drop segments < min_duration_on. Defaults are
// NeMo's tuned values for this checkpoint
// (post_processing/diar_streaming_sortformer_4spk-v2_callhome-part1.yaml).
// NOTE: dataset-sensitive - callhome-like conversational audio scores well
// with these; clean read speech wants lower onset + larger pads.
struct DiarSegmentationCfg {
    float onset = 0.641f;
    float offset = 0.561f;
    double pad_onset = 0.229;
    double pad_offset = 0.079;
    double min_duration_on = 0.511;
    double min_duration_off = 0.296;
};

// Per-speaker segmentation of a frame-probability timeline; shared by the
// streaming (DiarStream::segments) and offline (diarize_offline) paths.
struct DiarSegment {
    double t0, t1;
    int speaker;
};
std::vector<DiarSegment> diar_segments_from_probs(
    const float* probs, int64_t n_frames, int n_spk, double sec_per_frame,
    const DiarSegmentationCfg& cfg);

// Per-stream streaming state + timeline.
class DiarStream {
   public:
    DiarStream(DiarModel& model, const DiarGeometry& geometry);

    // Feed 16 kHz mono samples; runs any chunks whose right context is
    // covered. Call finish() once at end-of-stream to flush the tail.
    void feed_audio(const float* samples, size_t n_samples);
    void finish();
    void reset();

    // On-demand diarization (riva ProcessOnDemandDiarization): label
    // already-arrived audio early - with truncated right context - until the
    // timeline covers `target_frame` or no whole encoder frame of new mel
    // remains. Called by the recognizer when a final's words end past the
    // diarized frontier, so word tags there come from real predictions
    // instead of last-frame extrapolation. Chunks stay on the 80 ms encoder
    // frame grid; the stream continues normally afterwards.
    void flush_available(int64_t target_frame);

    // Total emitted 80 ms frames (monotonic; includes frames whose raw
    // probabilities were compacted away, see below).
    int64_t n_frames() const { return probs_base_ + static_cast<int64_t>(probs_.size()) / n_spk_; }
    // Retained per-frame speaker probabilities, frame-major, covering frames
    // [frame_probs_base(), n_frames()). For streams below the compaction
    // horizon frame_probs_base() is 0 and this is the whole timeline.
    const std::vector<float>& frame_probs() const { return probs_; }
    int64_t frame_probs_base() const { return probs_base_; }
    double seconds_per_frame() const { return sec_per_frame_; }

    // Long-stream memory bound. The raw probability timeline would otherwise
    // grow forever (n_spk floats / 80 ms) and every segments() call would
    // re-segment all of it. Once more than `trigger_frames` are retained, the
    // prefix up to an all-speaker-silent gap (leaving at least `retain_frames`)
    // is converted to frozen segments (using the library-default segmentation
    // config) and its raw probabilities are dropped. Cutting only inside a
    // silent gap longer than the postprocessing's temporal reach makes
    // frozen + recomputed-tail segments EXACTLY equal to a full recompute
    // under the same config; a per-call segments() config only affects the
    // retained tail. If no such gap exists yet, compaction retries later.
    // Defaults retain roughly the latest 10 minutes after a 20-minute trigger.
    void set_compaction(int64_t trigger_frames, int64_t retain_frames);

    // Speaker id (0-based) for a [start, end) frame range: mean probability
    // over the range, argmax. Frames beyond the emitted timeline use the last
    // emitted frame (riva's extrapolation for words past the diarized
    // frontier). Returns -1 when nothing has been emitted yet.
    int speaker_for_frames(int64_t start_frame, int64_t end_frame) const;
    // Same, for a time range in seconds (used for word timestamps).
    int speaker_for_time(double t0, double t1) const;

    using Segment = DiarSegment;
    std::vector<Segment> segments(const DiarSegmentationCfg& cfg = DiarSegmentationCfg()) const;

   private:
    // Run the next chunk if possible. force = accept a partial chunk (fewer
    // than chunk_len new frames, truncated right context); final_flush
    // additionally accepts the sub-frame tail remainder at end-of-stream.
    bool run_one_chunk(bool force, bool final_flush);
    void run_ready_chunks(bool end_of_stream);
    void ensure_mel();
    void maybe_compact();

    DiarModel& m_;
    DiarGeometry geo_;
    int n_spk_;
    int sub_;  // mel frames per encoder frame
    double sec_per_frame_;

    AoscState state_;
    std::vector<float> audio_buf_;
    size_t audio_base_ = 0;  // global sample index of audio_buf_[0]
    std::vector<float> mel_buf_;
    int64_t mel_base_ = 0;  // global mel frame index of mel_buf_[0]
    // Global mel frames produced so far - derived, so it cannot desync from
    // the buffer: mel_buf_ always holds frames [mel_base_, mel_produced()).
    int64_t mel_produced() const {
        return mel_base_ + static_cast<int64_t>(mel_buf_.size()) / n_mels_;
    }
    int n_mels_ = 0;
    int64_t mel_consumed_ = 0;  // mel frames consumed = start of the next chunk
    bool finished_ = false;

    std::vector<float> probs_;              // retained timeline tail (see frame_probs_base)
    int64_t probs_base_ = 0;                // emitted frames compacted off the front
    std::vector<DiarSegment> frozen_segs_;  // finalized segments before probs_base_
    // ~20 min trigger / ~10 min retained at 80 ms frames.
    int64_t compact_trigger_frames_ = 15000;
    int64_t compact_retain_frames_ = 7500;
};

}  // namespace nemo_speech::asr
