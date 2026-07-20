// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "model.h"

namespace nemo_speech::tts {

class DecoderKvCache {
   public:
    DecoderKvCache() = default;
    ~DecoderKvCache();

    DecoderKvCache(const DecoderKvCache&) = delete;
    DecoderKvCache& operator=(const DecoderKvCache&) = delete;
    DecoderKvCache(DecoderKvCache&& other) noexcept;
    DecoderKvCache& operator=(DecoderKvCache&& other) noexcept;

    void reset();
    void clear();
    bool init(const magpietts_model& model);
    bool init(
        ggml_backend_t backend, int n_layers, int context_length, int embedding_dim,
        const char* label);
    bool initialized() const { return ctx != nullptr; }

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor* memory_k = nullptr;
    ggml_tensor* memory_v = nullptr;
    int n_ctx = 0;
    int n_layers = 0;
    int n_embd = 0;
    int n_tokens = 0;
};

class DecoderCrossKvCache {
   public:
    DecoderCrossKvCache() = default;
    ~DecoderCrossKvCache();

    DecoderCrossKvCache(const DecoderCrossKvCache&) = delete;
    DecoderCrossKvCache& operator=(const DecoderCrossKvCache&) = delete;
    DecoderCrossKvCache(DecoderCrossKvCache&& other) noexcept;
    DecoderCrossKvCache& operator=(DecoderCrossKvCache&& other) noexcept;

    void reset();
    void clear();
    bool init(const magpietts_model& model, int text_len);
    bool initialized() const { return ctx != nullptr; }
    bool validFor(const magpietts_model& model, int text_len) const;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor* memory_k = nullptr;
    ggml_tensor* memory_v = nullptr;
    int text_len = 0;
    int n_layers = 0;
    int n_cross_dim = 0;
    bool valid = false;
};

struct decoder_result {
    std::vector<float> logits_last;
    std::vector<float> hidden_last;
    std::vector<float> cross_attn_last;
};

struct magpietts_cuda_sample_request {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    magpietts_cuda_sampler* sampler = nullptr;
#else
    void* sampler = nullptr;
#endif
    bool use_cfg = false;
    float cfg_scale = 1.0f;
    float temperature = 0.0f;
    int top_k = 1;
    bool forbid_audio_eos = false;
    uint64_t seed = 0;
    int frame_index = 0;
    std::vector<int32_t> codes;
    std::vector<int32_t> argmax_codes;
};

class MagpieDecoder {
   public:
    explicit MagpieDecoder(const magpietts_model& model) : model_(model) {}

    bool eval(
        const std::vector<float>& text_cond, int text_len,
        const std::vector<std::vector<int32_t>>& audio_codes, int speaker, bool conditional,
        int threads, decoder_result& result, magpietts_cuda_sample_request* cuda_sample = nullptr,
        const magpietts_backend_tensor* text_cond_device = nullptr,
        magpietts_backend_tensor* hidden_out = nullptr,
        const magpietts_decoder_attention* attention = nullptr) const;
    bool evalPair(
        const std::vector<float>& text_cond, int text_len,
        const std::vector<std::vector<int32_t>>& audio_codes, int speaker, int threads,
        decoder_result& cond_result, decoder_result& uncond_result,
        magpietts_cuda_sample_request* cuda_sample = nullptr,
        const magpietts_backend_tensor* text_cond_device = nullptr,
        magpietts_backend_tensor* cond_hidden_out = nullptr,
        magpietts_backend_tensor* uncond_hidden_out = nullptr,
        const magpietts_decoder_attention* attention = nullptr) const;
    bool evalCached(
        const std::vector<float>& text_cond, int text_len,
        const std::vector<std::vector<int32_t>>& audio_codes, int speaker, bool conditional,
        int threads, DecoderKvCache& kv_state, decoder_result& result,
        magpietts_cuda_sample_request* cuda_sample = nullptr,
        const magpietts_backend_tensor* text_cond_device = nullptr,
        magpietts_backend_tensor* hidden_out = nullptr, DecoderCrossKvCache* cross_kv = nullptr,
        const magpietts_decoder_attention* attention = nullptr) const;
    bool evalCachedPair(
        const std::vector<float>& text_cond, int text_len,
        const std::vector<std::vector<int32_t>>& audio_codes, int speaker, int threads,
        DecoderKvCache& cond_kv, DecoderKvCache& uncond_kv, decoder_result& cond_result,
        decoder_result& uncond_result, magpietts_cuda_sample_request* cuda_sample = nullptr,
        const magpietts_backend_tensor* text_cond_device = nullptr,
        magpietts_backend_tensor* cond_hidden_out = nullptr,
        magpietts_backend_tensor* uncond_hidden_out = nullptr,
        DecoderCrossKvCache* cond_cross_kv = nullptr,
        const magpietts_decoder_attention* attention = nullptr) const;

   private:
    const magpietts_model& model_;
    mutable MagpiePinnedHostScratch output_staging_;
};

class MagpieCodebookSampler {
   public:
    static bool runCuda(
        ggml_backend_t backend, const magpietts_hparams& h, magpietts_cuda_sample_request* request,
        const ggml_tensor* logits_cond, const ggml_tensor* logits_uncond, size_t logits_off_floats,
        int codebooks, int codebook_offset);
    static int sampleFromLogits(
        std::vector<float> logits, const magpietts_hparams& h, float temperature, int top_k,
        std::mt19937& rng, bool forbid_audio_eos);
    static int argmaxFromLogits(
        std::vector<float> logits, const magpietts_hparams& h, bool forbid_audio_eos);
    static std::vector<int32_t> sampleParallel(
        const std::vector<float>& cond_logits, const std::vector<float>& uncond_logits,
        const magpietts_hparams& h, bool use_cfg, float cfg_scale, float temperature, int top_k,
        bool forbid_audio_eos, std::mt19937& rng, std::vector<int32_t>* argmax_codes);
    static bool hasEos(const std::vector<int32_t>& a, const std::vector<int32_t>& b, int eos_id);
};

}  // namespace nemo_speech::tts
