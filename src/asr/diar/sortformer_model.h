// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Sortformer v2/v3 streaming diarization models on the ggml runtime.
//
// One Session builds the whole per-chunk graph at NeMo's inference boundary.
// V2 uses:
//
//   mel window (n_mels, T_mel)
//     -> NEST pre_encode (8x dw-striding conv stem)     -> chunk embs (512, T3)
//   concat over time [ compact state (spkcache + fifo) | chunk embs ]
//     -> xscale + rel-pos + 17 conformer layers (FastConformerEncoder reuse)
//     -> encoder_proj 512->192
//     -> 18-layer post-LN transformer
//     -> head: relu -> Linear(192,192) -> relu -> Linear(192,4) -> sigmoid
//
// V3 replaces both encoder stacks with 8x feature stacking followed by a
// 31-layer pre-LN RoPE Transformer, then applies an 8x subpixel Conv1D before
// the sigmoid head. It returns native 10 ms probabilities while AOSC continues
// to consume probabilities averaged onto the 80 ms embedding grid.
//
// Outputs: native predictions and chunk embeddings (512, T3). Host code also
// derives coarse predictions for AOSC when the model is V3.
//
// V2 batches right-aligned state prefixes with leading-padding masks. V3
// coalesces equal state lengths so RoPE positions exactly match scalar runs.
//
// Host code in aosc_state.h updates the speaker cache and FIFO between chunks.
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "batching.h"
#include "fastconformer.h"
#include "rope_transformer.h"
#include "runtime.h"
#include "transformer_encoder.h"

namespace nemo_speech::asr {

// AOSC compression constants (model-tied, from GGUF `sortformer.scoring.*`).
struct DiarScoringConfig {
    int sil_frames_per_spk = 3;
    float pred_score_threshold = 0.25f;
    float scores_boost_latest = 0.05f;
    float sil_threshold = 0.2f;
    float strong_boost_rate = 0.75f;
    float weak_boost_rate = 1.5f;
    float min_pos_scores_rate = 0.5f;
};

enum class SortformerVersion { V2, V3 };

struct SortformerModelConfig {
    SortformerVersion version = SortformerVersion::V2;
    EncoderConfig encoder;  // NEST Fast-Conformer (offline / full attention)
    RopeTransformerConfig rope_encoder;
    TransformerConfig transformer;
    int num_speakers = 4;
    DiarScoringConfig scoring;
    bool high_resolution = false;
    bool learnable_silence = false;
    int output_subsampling_factor = 8;  // output frames relative to 10 ms mel frames
    int upsample_factor = 1;            // output frames per coarse encoder frame

    // FE (from GGUF sortformer.preprocessor.*)
    int sample_rate = 16000;
    float window_size = 0.025f;
    float window_stride = 0.01f;
    int n_fft = 512;
    int n_mels = 128;
    float preemph = 0.97f;
    // NeMo FilterbankFeatures "add"-type guard; Sortformer trains with the
    // 2^-24 default (silence bins land on the log floor, so this matters).
    float log_zero_guard = 5.9604645e-8f;

    bool is_v3() const { return version == SortformerVersion::V3; }
    double seconds_per_output_frame() const { return window_stride * output_subsampling_factor; }
    int word_anchor_frames() const {
        // Count on the frontend's integer sample grid. Dividing by the F32
        // period and applying ceil turns nominal 160/80 ms into three frames.
        const int64_t hop = std::llround(window_stride * sample_rate) * output_subsampling_factor;
        if (hop <= 0)
            throw std::invalid_argument("Sortformer output hop must be positive");
        const int64_t anchor = std::llround(0.16 * sample_rate);
        return std::max(1, static_cast<int>((anchor + hop - 1) / hop));
    }
};

// Root Module for the per-chunk graph. Per-call inputs (by name):
//   input.mel      (n_mels, T_mel)  - required
//   input.state    (512, Lmax, B)   - optional compact state
//   input.positions                 - V3 RoPE positions
//   input.attention_mask            - V2 optional leading-padding key mask
//   input.valid_mask                - V2 optional convolution mask
// Output bag: [0] native preds, [1] chunk embeddings (512, T3, B).
class SortformerGraph : public ggml_runtime::Module {
   public:
    explicit SortformerGraph(const SortformerModelConfig& cfg);
    ~SortformerGraph();

    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* session) override;

   private:
    SortformerModelConfig cfg_;
    FastConformerEncoder* encoder_;
    ggml_runtime::Linear* feature_stack_proj_;
    RopeTransformerEncoder* rope_encoder_;
    ggml_runtime::Linear* encoder_proj_;
    TransformerEncoderModule* transformer_;
    ggml_runtime::Conv1D* subpixel_upsample_;
    ggml_runtime::Linear* head_hidden_;
    ggml_runtime::Linear* head_spks_;
};

// Owns loader + graph + Session; runs one streaming chunk at a time.
// Thread-safety comes from Session::run (BackendManager compute mutex); one
// SortformerModel can serve many streams, each supplying its own state
// buffers per call.
class SortformerModel {
   public:
    SortformerModel(
        ggml_runtime::BackendManager& bm, const std::string& gguf_path,
        const BatchingConfig& batching = {});
    ~SortformerModel();

    const SortformerModelConfig& cfg() const { return cfg_; }

    // Trained mel filterbank (n_mels x (n_fft/2+1)) from the GGUF, for the FE.
    const std::vector<float>& mel_basis() const { return mel_basis_; }
    const std::vector<float>& learnable_silence_embedding() const {
        return learnable_silence_embedding_;
    }

    ggml_runtime::Session* session() const { return session_.get(); }

    // Encoder-frame count the pre_encode stem produces for t_mel input frames
    // (symmetric dw-striding: ceil-div by 2 per stage).
    int subsampled_len(int t_mel) const;

    struct ChunkOutput {
        // Coarse predictions used exclusively by AOSC state management.
        std::vector<float> preds;  // (L1+L2+T3) x n_spk, frame-major
        int total_frames = 0;
        std::vector<float> chunk_embs;  // T3 x 512, frame-major
        int chunk_frames = 0;
        // Native-cadence full-sequence predictions. Empty for v2, where
        // `preds` already has native cadence. V3 emits 8 rows per coarse row.
        std::vector<float> native_preds;
        int native_total_frames = 0;
    };

    // mel: (n_mels, t_mel) frame-major (each frame's n_mels contiguous).
    // spkcache/fifo: frames x 512, frame-major; pass nullptr/0 when empty.
    ChunkOutput run_chunk(
        const float* mel, int t_mel, const float* spkcache, int spkcache_frames, const float* fifo,
        int fifo_frames);
    BatchMetrics batch_metrics() const;

   private:
    class SortformerBatcher;

    SortformerModelConfig cfg_;
    std::vector<float> mel_basis_;
    std::vector<float> learnable_silence_embedding_;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<SortformerGraph> graph_;
    std::unique_ptr<ggml_runtime::Session> session_;
    std::unique_ptr<SortformerBatcher> batcher_;
};

}  // namespace nemo_speech::asr
