// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#if defined(MAGPIETTS_CUDA_SAMPLING)
#include "ggml-cuda.h"
#include "magpietts_cuda_sampling.h"
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace nemo_speech::tts {

inline constexpr int MAGPIETTS_MAX_NODES = 32768;
inline constexpr float MAGPIETTS_LN_EPS = 1.0e-5f;
inline constexpr double MAGPIETTS_NANO_CODEC_FPS = 22050.0 / 1024.0;

enum magpietts_backend_preference {
    MAGPIETTS_BACKEND_AUTO,
    MAGPIETTS_BACKEND_CPU,
    MAGPIETTS_BACKEND_CUDA,
};

enum magpietts_uma_mode {
    MAGPIETTS_UMA_AUTO,
    MAGPIETTS_UMA_OFF,
    MAGPIETTS_UMA_ON,
};

struct magpietts_hparams {
    int32_t text_vocab_size = 0;
    int32_t audio_codebooks = 8;
    int32_t audio_codebook_size = 2016;
    int32_t audio_vocab_size = 2024;
    int32_t audio_bos_id = 2016;
    int32_t audio_eos_id = 2017;
    int32_t mask_token_id = 2020;
    int32_t frame_stacking_factor = 1;

    // Number of model prediction slots in one decoder position.  v2602 has
    // one slot per codec codebook, while v2607 predicts two consecutive codec
    // frames at once.
    int32_t stacked_audio_codebooks() const { return audio_codebooks * frame_stacking_factor; }

    int32_t n_embd = 768;
    int32_t n_ffn = 3072;
    int32_t n_ctx = 2048;
    int32_t n_enc_layer = 6;
    int32_t n_enc_head = 12;
    int32_t enc_kernel = 3;
    int32_t n_dec_layer = 12;
    int32_t n_dec_head = 12;
    int32_t n_cross_head = 1;
    int32_t n_cross_dhead = 128;
    int32_t dec_kernel = 1;

    int32_t lt_layers = 1;
    int32_t lt_heads = 1;
    int32_t lt_hidden = 256;
    int32_t lt_ctx = 10;

    int32_t baked_context_length = 110;
    int32_t baked_context_dim = 768;
    int32_t baked_speakers = 5;

    int32_t max_decoder_steps = 500;
    int32_t top_k = 80;
    int32_t min_generated_frames = 4;
    float temperature = 0.6f;
    float cfg_scale = 2.5f;

    bool apply_attention_prior = false;
    float attention_prior_epsilon = 0.1f;
    int32_t attention_prior_lookahead_window = 5;
    int32_t start_prior_after_n_audio_steps = 0;
    int32_t attention_prior_advance_threshold = 8;
    int32_t attention_prior_decay_threshold = 10;
    std::vector<int32_t> estimate_alignment_from_layers;
    std::vector<int32_t> apply_prior_to_layers;
};

inline std::string
magpietts_infer_tokenizer_profile(const magpietts_hparams& h) {
    if (h.text_vocab_size == 2362 && h.frame_stacking_factor == 1) {
        return "v2602";
    }
    if (h.text_vocab_size == 3359 && h.frame_stacking_factor == 2) {
        return "v2607";
    }
    return {};
}

inline bool
magpietts_tokenizer_profile_matches(
    const std::string& profile, const magpietts_hparams& h) {
    return profile == magpietts_infer_tokenizer_profile(h) && !profile.empty();
}

inline bool
magpietts_unstack_codes(
    const std::vector<int32_t>& stacked_codes, const magpietts_hparams& h,
    std::vector<std::vector<int32_t>>& frames) {
    if ((int)stacked_codes.size() != h.stacked_audio_codebooks()) {
        return false;
    }
    frames.assign((size_t)h.frame_stacking_factor, std::vector<int32_t>(h.audio_codebooks));
    for (int lane = 0; lane < h.frame_stacking_factor; ++lane) {
        for (int codebook = 0; codebook < h.audio_codebooks; ++codebook) {
            frames[(size_t)lane][(size_t)codebook] =
                stacked_codes[(size_t)(codebook + lane * h.audio_codebooks)];
        }
    }
    return true;
}

inline int
magpietts_first_eos_lane(
    const std::vector<int32_t>& sampled, const std::vector<int32_t>& greedy,
    const magpietts_hparams& h) {
    if ((int)sampled.size() != h.stacked_audio_codebooks() ||
        (int)greedy.size() != h.stacked_audio_codebooks()) {
        return -1;
    }
    for (int lane = 0; lane < h.frame_stacking_factor; ++lane) {
        for (int codebook = 0; codebook < h.audio_codebooks; ++codebook) {
            const size_t index = (size_t)(codebook + lane * h.audio_codebooks);
            if (sampled[index] == h.audio_eos_id || greedy[index] == h.audio_eos_id) {
                return lane;
            }
        }
    }
    return -1;
}

struct magpietts_layer {
    ggml_tensor* norm_self = nullptr;
    ggml_tensor* self_qkv = nullptr;
    ggml_tensor* self_o = nullptr;

    ggml_tensor* norm_xattn_query = nullptr;
    ggml_tensor* cross_q = nullptr;
    ggml_tensor* cross_kv = nullptr;
    ggml_tensor* cross_o = nullptr;
    ggml_tensor* norm_xattn_memory = nullptr;

    ggml_tensor* norm_ff = nullptr;
    std::vector<ggml_tensor*> ff_proj;
    std::vector<ggml_tensor*> ff_out;

    bool has_cross = false;
    int kernel = 1;
};

struct magpietts_transformer {
    ggml_tensor* norm_out = nullptr;
    ggml_tensor* pos_emb = nullptr;
    std::vector<magpietts_layer> layers;

    int n_embd = 0;
    int n_head = 0;
    int n_cross_head = 0;
    int n_cross_dhead = 0;
    int kernel = 1;
    bool causal = true;
    bool portable_causal_mask = false;
    bool has_cross = false;
    bool apply_attention_prior = false;
    std::vector<int32_t> estimate_alignment_from_layers;
    std::vector<int32_t> apply_prior_to_layers;
};

struct magpietts_decoder_attention {
    const std::vector<float>* prior = nullptr;
    std::vector<float>* alignment_scores = nullptr;
};

class MagpieAttentionPriorState {
   public:
    void reset();
    bool enabled(const magpietts_hparams& h, int text_len) const;
    bool shouldCollect(const magpietts_hparams& h, int step, int text_len) const;
    const std::vector<float>* priorForStep(const magpietts_hparams& h, int text_len) const;
    void update(
        const magpietts_hparams& h, int step, int text_len,
        const std::vector<float>& alignment_scores);

    const std::vector<float>& prior() const { return prior_; }
    int lastAttended() const { return last_attended_; }

   private:
    void ensureTextLen(int text_len);

    int text_len_ = 0;
    int last_attended_ = 1;
    std::vector<int> attended_counts_;
    std::vector<float> prior_;
};

class MagpieLongformAttentionPriorState {
   public:
    void reset();
    bool enabled(const magpietts_hparams& h, int text_len) const;
    bool shouldCollect(const magpietts_hparams& h, int step, int text_len) const;
    const std::vector<float>* priorForStep(const magpietts_hparams& h, int text_len) const;
    void beginChunk(
        const magpietts_hparams& h, int left_offset, int text_len, int current_chunk_len,
        bool first_chunk);
    void update(
        const magpietts_hparams& h, int step, int text_len,
        const std::vector<float>& alignment_scores);

    const std::vector<float>& prior() const { return prior_; }
    int lastAttendedAbsolute() const { return last_attended_absolute_; }
    int lastAttendedRelative() const;
    int leftOffset() const { return left_offset_; }
    int currentChunkLen() const { return current_chunk_len_; }
    bool initialized() const { return initialized_; }

   private:
    void ensureAbsoluteCapacity(int absolute_len);
    void buildInitialChunkPrior(const magpietts_hparams& h);
    void buildPrior(const magpietts_hparams& h);

    bool initialized_ = false;
    bool first_chunk_ = true;
    int left_offset_ = 0;
    int text_len_ = 0;
    int current_chunk_len_ = 0;
    int last_attended_absolute_ = 1;
    std::vector<int> attended_counts_;
    std::vector<float> prior_;
};

class MagpieModel {
   public:
    MagpieModel() = default;
    ~MagpieModel();

    MagpieModel(const MagpieModel&) = delete;
    MagpieModel& operator=(const MagpieModel&) = delete;
    MagpieModel(MagpieModel&& other) noexcept;
    MagpieModel& operator=(MagpieModel&& other) noexcept;

    bool load(
        const std::string& fname, magpietts_uma_mode uma_mode = MAGPIETTS_UMA_AUTO,
        bool force_cpu = false);
    void reset();
    bool loaded() const { return gguf != nullptr && ctx != nullptr && backend != nullptr; }

    magpietts_hparams hparams;
    std::string tokenizer_profile;
    std::string nemo_version;

    gguf_context* gguf = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    bool cuda_unified_memory = false;

    ggml_tensor* text_embedding = nullptr;
    std::vector<ggml_tensor*> audio_embeddings;
    ggml_tensor* baked_context = nullptr;
    ggml_tensor* final_proj_w = nullptr;
    ggml_tensor* final_proj_b = nullptr;
    ggml_tensor* lt_in_w = nullptr;
    ggml_tensor* lt_in_b = nullptr;
    std::vector<ggml_tensor*> lt_out_w;
    std::vector<ggml_tensor*> lt_out_b;

    magpietts_transformer encoder;
    magpietts_transformer decoder;
    magpietts_transformer local;
};

using magpietts_model = MagpieModel;

class MagpiePinnedHostScratch {
   public:
    MagpiePinnedHostScratch() = default;
    ~MagpiePinnedHostScratch();

    MagpiePinnedHostScratch(const MagpiePinnedHostScratch&) = delete;
    MagpiePinnedHostScratch& operator=(const MagpiePinnedHostScratch&) = delete;
    MagpiePinnedHostScratch(MagpiePinnedHostScratch&& other) noexcept;
    MagpiePinnedHostScratch& operator=(MagpiePinnedHostScratch&& other) noexcept;

    void reset();
    bool reserve(const magpietts_model& model, size_t size);
    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t capacity() const { return capacity_; }
    bool pinned() const { return pinned_; }

   private:
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<uint8_t> fallback_;
    uint8_t* data_ = nullptr;
    size_t capacity_ = 0;
    bool pinned_ = false;
};

bool magpietts_should_stage_pinned(const magpietts_model& model, size_t nbytes);
void magpietts_backend_tensor_set_staged(
    const magpietts_model& model, MagpiePinnedHostScratch& scratch, ggml_tensor* tensor,
    const void* data, size_t offset, size_t nbytes);
void magpietts_backend_tensor_get_staged(
    const magpietts_model& model, MagpiePinnedHostScratch& scratch, const ggml_tensor* tensor,
    void* data, size_t offset, size_t nbytes);

struct magpietts_params {
    std::string model;
    std::vector<int32_t> tokens;
    std::string tokens_file;
    std::vector<int32_t> warmup_tokens;
    std::string warmup_tokens_file;
    std::string codes_out;
    int speaker = 0;
    int threads = 4;
    int seed = -1;
    int steps = -1;
    int top_k = -1;
    double codec_fps = MAGPIETTS_NANO_CODEC_FPS;
    float temperature = NAN;
    float cfg_scale = NAN;
    bool use_cfg = true;
    bool use_local_transformer = true;
    bool lt_fp32 = false;
    bool use_kv_cache = true;
    magpietts_backend_preference lt_backend = MAGPIETTS_BACKEND_AUTO;
    magpietts_backend_preference sampling_backend = MAGPIETTS_BACKEND_AUTO;
    magpietts_uma_mode uma_mode = MAGPIETTS_UMA_AUTO;
};

class BackendTensor {
   public:
    BackendTensor() = default;
    ~BackendTensor();

    BackendTensor(const BackendTensor&) = delete;
    BackendTensor& operator=(const BackendTensor&) = delete;
    BackendTensor(BackendTensor&& other) noexcept;
    BackendTensor& operator=(BackendTensor&& other) noexcept;

    void reset();
    bool alloc2d(
        const magpietts_model& model, enum ggml_type type, int64_t ne0, int64_t ne1,
        const char* name);
    bool allocated() const { return tensor != nullptr; }

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor* tensor = nullptr;
};

using magpietts_backend_tensor = BackendTensor;

struct magpietts_run_metrics {
    int64_t start_us = 0;
    int64_t first_frame_us = 0;
    int64_t last_frame_us = 0;
    int frames = 0;
    double inter_frame_sum_ms = 0.0;
    double inter_frame_min_ms = 0.0;
    double inter_frame_max_ms = 0.0;
    double ttff_ms = 0.0;
    double e2e_elapsed_s = 0.0;
    double audio_s = 0.0;
    double e2e_rtf = 0.0;
    double e2e_rtfx = 0.0;

    void begin();
    double record_frame(int64_t now_us, bool& first_frame);
    void finish(size_t frames_generated, double codec_fps);
    double inter_frame_avg_ms() const;
    double inter_frame_min_value_ms() const;
};

class MagpieCodeGenerator {
   public:
    explicit MagpieCodeGenerator(magpietts_model& model) : model_(model) {}

    bool generate(
        const magpietts_params& params, const std::vector<int32_t>& tokens,
        std::vector<std::vector<int32_t>>* generated_frames_out, magpietts_run_metrics& metrics,
        const char* run_label, const magpietts_model* local_transformer_cpu_model = nullptr);

   private:
    magpietts_model& model_;
};

const char* magpietts_backend_preference_name(magpietts_backend_preference backend);
bool parse_backend_preference(const std::string& value, magpietts_backend_preference& backend);
const char* magpietts_uma_mode_name(magpietts_uma_mode mode);
bool parse_uma_mode(const std::string& value, magpietts_uma_mode& mode);

bool magpietts_backend_is_cuda(ggml_backend_t backend);

std::vector<int32_t> parse_token_list(const std::string& text);
std::string read_file(const std::string& path);

bool generate_magpietts_codes(
    magpietts_model& model, const magpietts_params& params, const std::vector<int32_t>& tokens,
    std::vector<std::vector<int32_t>>* generated_frames_out, magpietts_run_metrics& metrics,
    const char* run_label, const magpietts_model* local_transformer_cpu_model = nullptr);

bool magpietts_resolve_lt_backend(
    const magpietts_model& model, magpietts_backend_preference requested, bool& use_cuda_lt);

bool magpietts_resolve_sampling_backend(
    const magpietts_model& model, magpietts_backend_preference requested, bool& use_cuda_sampling);

}  // namespace nemo_speech::tts
