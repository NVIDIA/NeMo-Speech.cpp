// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Sortformer v2 streaming diarization model (nvidia/diar_streaming_sortformer_4spk-v2)
// on the ggml runtime.
//
// One Session builds the whole per-chunk graph at NeMo's ONNX-export boundary
// (`SortformerEncLabelModel.forward_for_export`):
//
//   mel window (n_mels, T_mel)
//     -> NEST pre_encode (8x dw-striding conv stem)     -> chunk embs (512, T3)
//   concat over time [ spkcache (512,L1) | fifo (512,L2) | chunk embs ]
//     -> xscale + rel-pos + 17 conformer layers (FastConformerEncoder reuse)
//     -> encoder_proj 512->192
//     -> 18-layer post-LN transformer
//     -> head: relu -> Linear(192,192) -> relu -> Linear(192,4) -> sigmoid
//   outputs: preds (n_spk, L1+L2+T3), chunk embs (512, T3)
//
// B=1, exact shapes, no masks: sequence length equals the true
// spkcache+fifo+chunk lengths, matching NeMo's sync streaming loop where no
// batch padding exists. Distinct (T_mel, L1, L2) combinations key the
// Session's run cache; the steady-state FIFO cycle re-uses ~10 cached graphs.
//
// Host code in aosc_state.h updates the speaker cache and FIFO between chunks.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "batching.h"
#include "fastconformer.h"
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

struct SortformerModelConfig {
    EncoderConfig encoder;  // NEST Fast-Conformer (offline / full attention)
    TransformerConfig transformer;
    int num_speakers = 4;
    DiarScoringConfig scoring;

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
};

// Root Module for the per-chunk graph. Per-call inputs (by name):
//   input.mel      (n_mels, T_mel)  - required
//   input.spkcache (512, L1)        - optional (omit when L1 == 0)
//   input.fifo     (512, L2)        - optional (omit when L2 == 0)
// Output bag: [0] preds (n_spk, L1+L2+T3), [1] chunk embs (512, T3).
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
    ggml_runtime::Linear* encoder_proj_;
    TransformerEncoderModule* transformer_;
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

    ggml_runtime::Session* session() const { return session_.get(); }

    // Encoder-frame count the pre_encode stem produces for t_mel input frames
    // (symmetric dw-striding: ceil-div by 2 per stage).
    int subsampled_len(int t_mel) const;

    struct ChunkOutput {
        std::vector<float> preds;  // (L1+L2+T3) x n_spk, frame-major
        int total_frames = 0;
        std::vector<float> chunk_embs;  // T3 x 512, frame-major
        int chunk_frames = 0;
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
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<SortformerGraph> graph_;
    std::unique_ptr<ggml_runtime::Session> session_;
    std::unique_ptr<SortformerBatcher> batcher_;
};

}  // namespace nemo_speech::asr
