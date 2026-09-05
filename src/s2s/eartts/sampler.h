// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// EarTTS code sampler: turns backbone hidden states into acoustic codes via
// the MoG head — the output half of the EarTTS side network (EarTTSEmbedder
// is the input half; both share one GGUFLoader).
//
// Scalar and batched sampling both run the unrolled device graph in one
// Session::run; each batch row retains its own deterministic noise stream.
#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <vector>

#include "config.h"
#include "runtime.h"

namespace nemo_speech::s2s {

class EarTTSMogStepModule : public ggml_runtime::Module {
   public:
    explicit EarTTSMogStepModule(const EarTTSConfig& cfg) : cfg_(cfg) {}

    // Inputs: [0] depth_sum (L, BT) f32; [1] h_cond (H, BT) f32;
    //   [2] h_uncond (H, BT) f32 (zeros when CFG disabled);
    //   [3] guidance_scale (1,) f32 (0 disables CFG mixing bit-exactly).
    // Outputs: [0] logits (N, BT); [1] mus_all (LR, N, BT);
    //   [2] logs (1, BT) clamped; [3] proj_else (L, BT).
    void define_tensors(ggml_runtime::Session* s) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag in,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* s) override;

   private:
    EarTTSConfig cfg_;
};

class EarTTSSampleModule : public ggml_runtime::Module {
   public:
    explicit EarTTSSampleModule(const EarTTSConfig& cfg) : cfg_(cfg) {}

    // NS = cfg.num_sampling_iter().
    // Inputs: [0] h_cond (H, BT) f32; [1] h_uncond (H, BT) f32 (zeros when CFG
    //   disabled); [2] gumbel (N, BT, NS) f32; [3] gauss (L, BT, NS) f32;
    //   [4] guidance_scale (1,) f32.
    // Outputs: [0] codes (BT, Q) i32 in GGML's q-major physical storage.
    void define_tensors(ggml_runtime::Session* s) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag in,
        ggml_runtime::TensorContainer* tc) override;
    void set_data(ggml_runtime::Session* s) override;

   private:
    EarTTSConfig cfg_;
};

class EarTTSSampler {
   public:
    EarTTSSampler(
        ggml_runtime::BackendManager& bm, std::shared_ptr<ggml_runtime::GGUFLoader> loader);
    ~EarTTSSampler();

    EarTTSSampler(const EarTTSSampler&) = delete;
    EarTTSSampler& operator=(const EarTTSSampler&) = delete;

    const EarTTSConfig& config() const { return cfg_; }
    int num_sampling_iter() const { return cfg_.num_sampling_iter(); }
    float default_guidance() const { return cfg_.guidance_default; }
    bool has_default_guidance() const { return cfg_.has_guidance_default; }

    // hidden_cond / hidden_uncond: f32 [BT, hidden] row-major (uncond may be
    // null when guidance_scale <= 0). seed 0 = nondeterministic.
    // out_codes: int32 [BT, Q] row-major.
    void sample(
        const float* hidden_cond, const float* hidden_uncond, float guidance_scale, uint64_t seed,
        int BT, int32_t* out_codes);
    // Per-row seeds preserve scalar sampling exactly when independently
    // evolving streams are coalesced into one GPU graph.
    void sample_batch(
        const float* hidden_cond, const float* hidden_uncond, float guidance_scale,
        const uint64_t* seeds, int BT, int32_t* out_codes);

    // Deterministic variant for parity testing — caller supplies all
    // randomness. Both buffers follow the contiguous physical layout implied
    // by GGML shapes (feature, BT, n_iter): gumbel is conventional row-major
    // [n_iter][BT][num_predictions], and gauss is
    // [n_iter][BT][latent]. n_iter must equal num_sampling_iter().
    void sample_with_noise(
        const float* hidden_cond, const float* hidden_uncond, float guidance_scale,
        const float* gumbel, const float* gauss, int n_iter, int BT, int32_t* out_codes);

    // Per-node backend assignment of the most recent run on each Session
    // (CPU-fallback diagnostics).
    void dump_schedules(std::ostream& os) const;

   private:
    void run_maskgit(
        const float* hidden_cond, const float* hidden_uncond, float guidance_scale,
        const float* gumbel, const float* gauss, int BT, int32_t* out_codes);
    void ensure_mog_session();
    void ensure_host_tables();

    EarTTSConfig cfg_;
    std::shared_ptr<ggml_runtime::GGUFLoader> loader_;
    ggml_runtime::BackendManager* bm_ = nullptr;
    std::unique_ptr<EarTTSSampleModule> sample_module_;
    std::unique_ptr<ggml_runtime::Session> sample_session_;
    // Lazy: only the BT>1 host-loop path needs these.
    std::unique_ptr<EarTTSMogStepModule> mog_module_;
    std::unique_ptr<ggml_runtime::Session> mog_session_;

    // Host F32 copies for the BT>1 path (depth-sum, low_mat matmul, RVQ NN).
    std::vector<float> rvq_host_;      // [Q, CB, L]
    std::vector<float> low_mat_host_;  // [N, L, LR]

    std::mutex mu_;
};

}  // namespace nemo_speech::s2s
