// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "perception.h"

#include <stdexcept>

namespace nemo_speech::s2s {

S2SPerception::S2SPerception(
    ggml_runtime::BackendManager& bm, const std::string& gguf_path,
    const asr::BatchingConfig& batching) {
    auto loaded = asr::AsrModel::load(bm, gguf_path, batching);
    auto* rnnt = dynamic_cast<asr::RnntModel*>(loaded.get());
    if (!rnnt)
        throw std::runtime_error("S2S perception model must use an RNNT head");
    loaded.release();
    model_.reset(rnnt);
    if (!model_->has_s2s_projection())
        throw std::runtime_error("perception GGUF is missing the S2S proj.* tensors");
    d_model_ = model_->encoder_config().d_model;
    d_proj_ = model_->s2s_projection_dim();
    sample_rate_ = model_->sample_rate();
    subsampling_ = model_->subsampling_factor();
    feat_in_ = model_->encoder_config().feat_in;
}

S2SPerception::~S2SPerception() = default;

int
S2SPerception::step(
    const float* audio, int n_samples, int valid_samples, std::vector<float>& out_proj,
    std::vector<float>* out_rnnt_emb) {
    // FE with normalize disabled (S2S-trained behavior).
    std::vector<float> feats;
    int n_frames = 0;
    model_->fe().compute_padded(
        audio, static_cast<size_t>(n_samples), static_cast<size_t>(valid_samples), feats, n_frames,
        /*reflect_left=*/true, /*normalize=*/false);
    if (n_frames == 0) {
        out_proj.clear();
        if (out_rnnt_emb)
            out_rnnt_emb->clear();
        return 0;
    }

    std::vector<float> rnnt;
    int T = 0;
    model_->infer_s2s_from_mel(feats.data(), n_frames, out_proj, rnnt, T);
    if (out_rnnt_emb)
        *out_rnnt_emb = std::move(rnnt);
    return T;
}

}  // namespace nemo_speech::s2s
