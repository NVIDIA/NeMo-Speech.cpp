// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// S2S perception on the native batched RNNT model. One FastConformer graph
// fans out to both the voicechat projection (LLM hidden) and RNNT joint
// encoder projection; the S2S frontend deliberately disables normalization.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace nemo_speech::s2s {

class S2SPerception {
   public:
    S2SPerception(
        ggml_runtime::BackendManager& bm, const std::string& gguf_path,
        const asr::BatchingConfig& batching);
    ~S2SPerception();

    int d_model() const { return d_model_; }
    int d_proj() const { return d_proj_; }
    int rnnt_dim() const { return model_->rnnt_config().joint_dim; }
    int sample_rate() const { return sample_rate_; }
    int subsampling() const { return subsampling_; }
    int feat_in() const { return feat_in_; }

    // One stateless window encode. audio: float32 mono @16k.
    // out_proj:    [T_out, d_proj]  row-major (LLM-side embeddings)
    // out_rnnt_emb (optional, nullable): [T_out, joint_dim] (RNNT encoder
    //              projection, feeding the native staged decoder)
    // Returns T_out.
    int step(
        const float* audio, int n_samples, int valid_samples, std::vector<float>& out_proj,
        std::vector<float>* out_rnnt_emb);

    const asr::RnntConfig& rnnt_config() const { return model_->rnnt_config(); }
    const std::vector<std::string>& vocab() const { return model_->vocab(); }
    asr::RnntEngine& rnnt_engine() { return *model_; }
    asr::BatchMetrics encoder_batch_metrics() const {
        return model_->offline_encoder_batch_metrics();
    }
    asr::BatchMetrics frontend_batch_metrics() const { return model_->fe().batch_metrics(); }

   private:
    int d_model_ = 0, d_proj_ = 0, sample_rate_ = 0, subsampling_ = 0, feat_in_ = 0;
    std::unique_ptr<asr::RnntModel> model_;
};

}  // namespace nemo_speech::s2s
