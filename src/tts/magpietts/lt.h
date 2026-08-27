// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <random>

#include "decoder.h"

namespace nemo_speech::tts {

class LocalTransformerGraphBank;

bool magpietts_model_init_local_transformer_cpu(const magpietts_model& src, magpietts_model& dst);
bool magpietts_model_init_local_transformer_fp32(
    const magpietts_model& src, magpietts_model& dst, bool use_cuda);

struct LocalCodebookLogitDump {
    const char* path = nullptr;
    const char* label = nullptr;
    int chunk_index = 0;
    int step = 0;
    int frame_index = 0;
};

bool magpietts_stack_forced_code_frames(
    const std::vector<std::vector<int32_t>>& raw_frames, size_t first_frame,
    const magpietts_hparams& h, std::vector<int32_t>& stacked_codes);

class LocalCodebookSampler {
   public:
    LocalCodebookSampler(const magpietts_model& model, int threads);
    ~LocalCodebookSampler();

    LocalCodebookSampler(const LocalCodebookSampler&) = delete;
    LocalCodebookSampler& operator=(const LocalCodebookSampler&) = delete;

    void setThreads(int threads);

    bool prewarm(bool use_cfg, int passes = 2);

    bool sample(
        const std::vector<float>& cond_hidden, const std::vector<float>& uncond_hidden,
        bool use_cfg, float cfg_scale, float temperature, int top_k, bool forbid_audio_eos,
        std::mt19937& rng, std::vector<int32_t>& codes, std::vector<int32_t>& argmax_codes,
        const LocalCodebookLogitDump* logit_dump = nullptr,
        const std::vector<int32_t>* forced_codes = nullptr);
#if defined(MAGPIETTS_CUDA_SAMPLING)
    bool sampleCuda(
        const magpietts_backend_tensor& cond_hidden, const magpietts_backend_tensor& uncond_hidden,
        bool use_cfg, float cfg_scale, float temperature, int top_k, bool forbid_audio_eos,
        magpietts_cuda_sampler* cuda_sampler, uint64_t seed, int frame_index,
        std::vector<int32_t>& codes, std::vector<int32_t>& argmax_codes);
#endif

   private:
    const magpietts_model& model_;
    int threads_ = 1;
    std::unique_ptr<LocalTransformerGraphBank> graph_bank_;
    MagpiePinnedHostScratch transfer_staging_;
};

}  // namespace nemo_speech::tts
