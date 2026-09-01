// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// EarTTS input embedder: builds the (cond, uncond) input-embedding pair the
// Gemma3 TTS backbone consumes each step — audio RVQ depth-sum + bos +
// prompt projection, char-aware subword text encoder, gated fusion. This is
// the input half of the EarTTS side network; EarTTSSampler is the output
// half. Both share one GGUFLoader (the side-network weights, ~500 MB).
#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <vector>

#include "config.h"
#include "runtime.h"

namespace nemo_speech::s2s {

class EarTTSEmbedModule : public ggml_runtime::Module {
   public:
    explicit EarTTSEmbedModule(const EarTTSConfig& cfg) : cfg_(cfg) {}

    // Inputs (order): [0] acoustic_tokens (BT, Q) i32 — ne[0]=BT, ne[1]=Q,
    //   i.e. q-major memory: codebook q's BT values contiguous (the facade
    //   transposes the caller's row-major [BT, Q] buffer before upload);
    //   [1] text_tokens (BT,) i32; [2] text_mask (BT,) f32;
    //   [3] bos_mask (BT,) f32; [4] pre-baked audio prompt latent
    //   (hidden, BT) f32; [5] use_prebaked (1,) f32.
    // Outputs: [0] cond (hidden, BT) f32; [1] uncond (hidden, BT) f32.
    void define_tensors(ggml_runtime::Session* s) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag in,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* s) override;

   private:
    EarTTSConfig cfg_;
};

class EarTTSEmbedder {
   public:
    EarTTSEmbedder(
        ggml_runtime::BackendManager& bm, std::shared_ptr<ggml_runtime::GGUFLoader> loader);
    ~EarTTSEmbedder();

    EarTTSEmbedder(const EarTTSEmbedder&) = delete;
    EarTTSEmbedder& operator=(const EarTTSEmbedder&) = delete;

    const EarTTSConfig& config() const { return cfg_; }

    // acoustic: int32 [BT, Q] row-major (codebook_size == "masked");
    // text: int32 [BT]; text_mask / bos_mask: f32 [BT].
    // out_cond / out_uncond: f32 [BT, hidden] row-major.
    void embed_pair(
        const int32_t* acoustic, const int32_t* text, const float* text_mask, const float* bos_mask,
        int BT, float* out_cond, float* out_uncond, const float* audio_prompt_latent = nullptr);

    // Per-node backend assignment of the most recent run (diagnostics).
    void dump_schedules(std::ostream& os) const;

   private:
    EarTTSConfig cfg_;
    std::shared_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<EarTTSEmbedModule> module_;
    std::unique_ptr<ggml_runtime::Session> session_;
    std::mutex mu_;
};

}  // namespace nemo_speech::s2s
