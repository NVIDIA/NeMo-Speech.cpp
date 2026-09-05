// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>

#include "ggml.h"
#include "graph_utils.h"

namespace gr = ggml_runtime;

namespace nemo_speech::s2s {

using namespace eartts_graph;

void
EarTTSMogStepModule::define_tensors(gr::Session* s) {
    declare_mog_head(s, cfg_);
}

void
EarTTSMogStepModule::set_data(gr::Session* s) {
    load_mog_head(s, cfg_);
}

gr::TensorBag
EarTTSMogStepModule::build_graph(gr::Session* s, gr::TensorBag in, gr::TensorContainer* tc) {
    const int N = cfg_.mog_num_predictions;
    const int LR = cfg_.mog_low_rank;

    auto depth = in.get_tensor(0);  // (L, BT) f32
    ggml_context* g = tc->get_ctx_of_buffer_type(depth.buft).ctx;
    const int BT = static_cast<int>(depth.tensor->ne[1]);
    ggml_tensor* h_cond = in.get_tensor(1).tensor;    // (H, BT)
    ggml_tensor* h_uncond = in.get_tensor(2).tensor;  // (H, BT)
    ggml_tensor* gs = in.get_tensor(3).tensor;        // (1,)

    ggml_tensor* embed = linear(g, W(s, "sampler.embed_code.weight"), depth.tensor);
    ggml_tensor* x_cond = mog_mlp_stack(g, s, cfg_, ggml_add(g, embed, h_cond));
    ggml_tensor* x_uncond = mog_mlp_stack(g, s, cfg_, ggml_add(g, embed, h_uncond));
    ggml_tensor* x = cfg_combine(g, x_cond, x_uncond, gs);

    ggml_tensor* logits = linear(g, W(s, "sampler.mog_head.proj_logits.weight"), x);  // (N, BT)
    ggml_tensor* mus_all = linear(g, W(s, "sampler.mog_head.proj_mus.weight"), x);
    mus_all = ggml_reshape_3d(g, mus_all, LR, N, BT);
    ggml_tensor* logs = linear(g, W(s, "sampler.mog_head.proj_logs.weight"), x);  // (1, BT)
    logs = ggml_clamp(g, logs, cfg_.mog_min_log_std, 1e30f);
    ggml_tensor* proj_else = linear(g, W(s, "sampler.mog_head.proj_else.weight"), x);  // (L, BT)

    ggml_set_name(logits, "logits");
    ggml_set_name(mus_all, "mus_all");
    ggml_set_name(logs, "logs");
    ggml_set_name(proj_else, "proj_else_x");
    for (ggml_tensor* t : {logits, mus_all, logs, proj_else}) ggml_set_output(t);

    gr::TensorBag out;
    out.add_tensor(gr::ggml_bf_tensor(logits, depth.buft));
    out.add_tensor(gr::ggml_bf_tensor(mus_all, depth.buft));
    out.add_tensor(gr::ggml_bf_tensor(logs, depth.buft));
    out.add_tensor(gr::ggml_bf_tensor(proj_else, depth.buft));
    return out;
}

// ===========================================================================
// EarTTSSampleModule — single-graph MaskGIT sampler (all batch widths).
// ===========================================================================

void
EarTTSSampleModule::define_tensors(gr::Session* s) {
    declare_mog_head(s, cfg_);
    declare_like(s, "sampler.rvq_embs");
    declare_like(s, "sampler.mog_head.low_mat");
    // RVQ row-norm² lookup table (codebook, Q) F32 — computed in set_data so
    // the in-graph nearest-neighbour search does `dot - 0.5 * norm2`.
    s->model_tensor_container->create_tensor_2d(
        "sampler.rvq_norm2", GGML_TYPE_F32, cfg_.codebook_size, cfg_.num_quantizers);
}

void
EarTTSSampleModule::set_data(gr::Session* s) {
    load_mog_head(s, cfg_);
    s->load_weight("sampler.rvq_embs");
    s->load_weight("sampler.mog_head.low_mat");

    // Compute the norm² table from the raw gguf bytes (double accumulation,
    // matching tts_ggml.cpp's load-time precompute).
    const int64_t L = cfg_.latent;
    const int64_t CB = cfg_.codebook_size;
    const int64_t Q = cfg_.num_quantizers;
    const ggml_type rvq_type = s->gguf_loader->get_tensor_type("sampler.rvq_embs");
    const size_t row_bytes = ggml_row_size(rvq_type, L);
    const char* base = s->gguf_loader->get_tensor_file_data("sampler.rvq_embs", row_bytes * CB * Q);
    const ggml_type_traits* tt = ggml_get_type_traits(rvq_type);
    if (!tt || !tt->to_float)
        throw std::runtime_error("eartts: sampler.rvq_embs dtype lacks to_float");
    std::vector<float> norm2_host(static_cast<size_t>(CB) * Q);
    std::vector<float> row(static_cast<size_t>(L));
    for (int64_t q = 0; q < Q; q++) {
        for (int64_t v = 0; v < CB; v++) {
            tt->to_float(base + (static_cast<size_t>(q) * CB + v) * row_bytes, row.data(), L);
            double acc = 0.0;
            for (int64_t l = 0; l < L; l++)
                acc += static_cast<double>(row[l]) * static_cast<double>(row[l]);
            norm2_host[static_cast<size_t>(q) * CB + v] = static_cast<float>(acc);
        }
    }
    auto t = s->model_tensor_container->get_tensor_by_name("sampler.rvq_norm2");
    ggml_backend_tensor_set(t.tensor, norm2_host.data(), 0, norm2_host.size() * sizeof(float));
}

gr::TensorBag
EarTTSSampleModule::build_graph(gr::Session* s, gr::TensorBag in, gr::TensorContainer* tc) {
    const int L = cfg_.latent;
    const int N = cfg_.mog_num_predictions;
    const int LR = cfg_.mog_low_rank;
    const int CB = cfg_.codebook_size;
    const int Q = cfg_.num_quantizers;
    const int NS = cfg_.num_sampling_iter();
    const float top_p = cfg_.top_p_or_k;
    const bool use_top_p = (top_p > 0.f && top_p < 1.f);
    const float ns = cfg_.noise_scale;

    auto hc_bf = in.get_tensor(0);  // (H, 1)
    ggml_context* g = tc->get_ctx_of_buffer_type(hc_bf.buft).ctx;
    const int BT = static_cast<int>(hc_bf.tensor->ne[1]);
    ggml_tensor* in_hc = hc_bf.tensor;
    ggml_tensor* in_hu = in.get_tensor(1).tensor;      // (H, 1)
    ggml_tensor* in_gumbel = in.get_tensor(2).tensor;  // (N, 1, NS)
    ggml_tensor* in_gauss = in.get_tensor(3).tensor;   // (L, 1, NS)
    ggml_tensor* gs = in.get_tensor(4).tensor;         // (1,)

    ggml_tensor* rvq_embs = W(s, "sampler.rvq_embs");         // ne = (L, CB, Q)
    ggml_tensor* norm2_tbl = W(s, "sampler.rvq_norm2");       // (CB, Q) f32
    ggml_tensor* low_mat = W(s, "sampler.mog_head.low_mat");  // ne = (LR, L, N)

    auto rvq_view_for_q = [&](int q) -> ggml_tensor* {
        return ggml_view_2d(
            g, rvq_embs, L, CB, rvq_embs->nb[1], static_cast<size_t>(q) * rvq_embs->nb[2]);
    };
    auto norm2_view_for_q = [&](int q) -> ggml_tensor* {
        return ggml_view_1d(g, norm2_tbl, CB, static_cast<size_t>(q) * CB * sizeof(float));
    };

    std::vector<ggml_tensor*> code_per_q;
    code_per_q.reserve(Q);
    std::vector<ggml_tensor*> emb_per_q;
    emb_per_q.reserve(Q);

    int cnt = 0;
    for (int iter = 0; iter < NS; iter++) {
        const int k = cfg_.sampling_per_step[iter];
        if (k == 0)
            continue;

        // ---- 1. MoG input: embed(depth_sum) + h_cond / h_uncond ----
        ggml_tensor* mog_in_cond;
        ggml_tensor* mog_in_uncond;
        if (cnt == 0) {
            mog_in_cond = in_hc;
            mog_in_uncond = in_hu;
        } else {
            ggml_tensor* depth_sum = emb_per_q[0];
            for (int q = 1; q < cnt; q++) depth_sum = ggml_add(g, depth_sum, emb_per_q[q]);
            ggml_tensor* embed = ggml_mul_mat(g, W(s, "sampler.embed_code.weight"), depth_sum);
            mog_in_cond = ggml_add(g, embed, in_hc);
            mog_in_uncond = ggml_add(g, embed, in_hu);
        }

        // ---- 2. MoG MLP stack (cond + uncond + CFG combine) ----
        ggml_tensor* x_cond = mog_mlp_stack(g, s, cfg_, mog_in_cond);
        ggml_tensor* x_uncond = mog_mlp_stack(g, s, cfg_, mog_in_uncond);
        ggml_tensor* x = cfg_combine(g, x_cond, x_uncond, gs);

        // ---- 3. Final projections ----
        ggml_tensor* logits = ggml_mul_mat(g, W(s, "sampler.mog_head.proj_logits.weight"), x);
        ggml_tensor* mus_all = ggml_mul_mat(g, W(s, "sampler.mog_head.proj_mus.weight"), x);
        mus_all = ggml_reshape_3d(g, mus_all, LR, N, BT);
        ggml_tensor* logs = ggml_mul_mat(g, W(s, "sampler.mog_head.proj_logs.weight"), x);
        logs = ggml_clamp(g, logs, cfg_.mog_min_log_std, 1e30f);
        ggml_tensor* proj_else = ggml_mul_mat(g, W(s, "sampler.mog_head.proj_else.weight"), x);

        // ---- 4. TopP filter + Gumbel-max argmax ----
        ggml_tensor* gumbel_iter = ggml_view_2d(
            g, in_gumbel, N, BT, in_gumbel->nb[1], static_cast<size_t>(iter) * in_gumbel->nb[2]);

        ggml_tensor* mix_idx;  // (BT,) i32 in original logit space
        if (use_top_p) {
            ggml_tensor* probs = ggml_soft_max(g, logits);  // (N, BT)
            ggml_tensor* sorted_idx_desc = ggml_argsort(g, probs, GGML_SORT_ORDER_DESC);
            // get_rows gathers along ne1. Top-p sorts along ne0 independently
            // for each batch row, so perform the small index plumbing per row
            // while retaining all expensive neural/RVQ work as one batch.
            ggml_tensor* mix_f32 = nullptr;
            ggml_tensor* gumbel_cont = ggml_cont(g, gumbel_iter);
            for (int b = 0; b < BT; ++b) {
                auto view_row = [&](ggml_tensor* src) {
                    return ggml_view_2d(
                        g, src, N, 1, src->nb[1], static_cast<size_t>(b) * src->nb[1]);
                };
                ggml_tensor* probs_b = view_row(probs);
                ggml_tensor* logits_b = view_row(logits);
                ggml_tensor* gumbel_b = view_row(gumbel_cont);
                ggml_tensor* idx_b = ggml_view_1d(
                    g, sorted_idx_desc, N, static_cast<size_t>(b) * sorted_idx_desc->nb[1]);
                auto gather_sorted = [&](ggml_tensor* src_b) {
                    auto src_1_N = ggml_reshape_2d(g, src_b, 1, N);
                    return ggml_reshape_1d(g, ggml_get_rows(g, src_1_N, idx_b), N);
                };
                ggml_tensor* sorted_probs = gather_sorted(probs_b);
                ggml_tensor* sorted_logits = gather_sorted(logits_b);
                ggml_tensor* sorted_gumbel = gather_sorted(gumbel_b);
                ggml_tensor* cum_desc = ggml_cumsum(g, sorted_probs);
                ggml_tensor* excl_cum = ggml_sub(g, cum_desc, sorted_probs);
                ggml_tensor* gap = ggml_scale_bias(g, excl_cum, -1.f, top_p);
                ggml_tensor* keep = ggml_step(g, gap);
                ggml_tensor* penalty = ggml_scale(g, ggml_scale_bias(g, keep, -1.f, 1.f), -1e30f);
                ggml_tensor* scored =
                    ggml_add(g, ggml_add(g, sorted_logits, penalty), sorted_gumbel);
                ggml_tensor* sorted_choice = ggml_argmax(g, scored);
                ggml_tensor* sorted_idx_1_N = ggml_reshape_2d(g, idx_b, 1, N);
                ggml_tensor* original_choice = ggml_get_rows(g, sorted_idx_1_N, sorted_choice);
                ggml_tensor* choice_f32 = ggml_cast(g, original_choice, GGML_TYPE_F32);
                mix_f32 = mix_f32 ? ggml_concat(g, mix_f32, choice_f32, 0) : choice_f32;
            }
            mix_idx = ggml_cast(g, ggml_reshape_1d(g, mix_f32, BT), GGML_TYPE_I32);
        } else {
            ggml_tensor* scored = ggml_add(g, logits, gumbel_iter);
            mix_idx = ggml_argmax(g, scored);
        }

        // ---- 5. Gather mu_low = mus_all[:, mix_idx, :] (LR, BT) ----
        ggml_tensor* mu_low = nullptr;
        for (int b = 0; b < BT; ++b) {
            ggml_tensor* mus_b = ggml_view_2d(
                g, mus_all, LR, N, mus_all->nb[1], static_cast<size_t>(b) * mus_all->nb[2]);
            ggml_tensor* idx_b =
                ggml_view_1d(g, mix_idx, 1, static_cast<size_t>(b) * mix_idx->nb[0]);
            ggml_tensor* selected = ggml_get_rows(g, mus_b, idx_b);  // (LR, 1)
            mu_low = mu_low ? ggml_concat(g, mu_low, selected, 1) : selected;
        }

        // ---- 6. Gather low_mat row -> (LR, L, BT) ----
        ggml_tensor* low_mat_2d = ggml_reshape_2d(g, low_mat, static_cast<int64_t>(LR) * L, N);
        ggml_tensor* low_mat_row_flat = ggml_get_rows(g, low_mat_2d, mix_idx);  // (LR*L, BT)
        ggml_tensor* low_mat_row = ggml_reshape_3d(g, low_mat_row_flat, LR, L, BT);

        // ---- 7. mu = low_mat_row @ mu_low ----
        ggml_tensor* mu_low_3d = ggml_reshape_3d(g, mu_low, LR, 1, BT);
        ggml_tensor* mu = ggml_mul_mat(g, low_mat_row, mu_low_3d);  // (L, 1, BT)
        mu = ggml_reshape_2d(g, mu, L, BT);

        // ---- 8. mu = mu * exp(logs) + proj_else ----
        ggml_tensor* scale_mu = ggml_exp(g, logs);  // (1, BT)
        mu = ggml_mul(g, mu, scale_mu);
        mu = ggml_add(g, mu, proj_else);

        // ---- 9. z = mu + noise_scale * exp(logs) * gauss_iter ----
        ggml_tensor* gauss_iter = ggml_view_2d(
            g, in_gauss, L, BT, in_gauss->nb[1], static_cast<size_t>(iter) * in_gauss->nb[2]);
        ggml_tensor* noise_term = ggml_mul(g, gauss_iter, scale_mu);
        ggml_tensor* z = ggml_add(g, mu, ggml_scale(g, noise_term, ns));

        // ---- 10. Per-position RVQ nearest-neighbour search ----
        for (int j = 0; j < k; j++) {
            const int q = cnt + j;
            ggml_tensor* rvq_q = rvq_view_for_q(q);        // (L, CB)
            ggml_tensor* dot = ggml_mul_mat(g, rvq_q, z);  // (CB, BT)
            ggml_tensor* norm2_q_2d = ggml_reshape_2d(g, norm2_view_for_q(q), CB, 1);
            ggml_tensor* score = ggml_sub(g, dot, ggml_scale(g, norm2_q_2d, 0.5f));
            ggml_tensor* idx = ggml_argmax(g, score);  // (BT,) i32
            // Cast to f32 for stacking — CUDA ggml_concat only supports F32.
            code_per_q.push_back(ggml_cast(g, idx, GGML_TYPE_F32));
            ggml_tensor* emb_idx = ggml_get_rows(g, rvq_q, idx);  // (L, BT)
            if (emb_idx->type != GGML_TYPE_F32)
                emb_idx = ggml_cast(g, emb_idx, GGML_TYPE_F32);
            z = ggml_sub(g, z, emb_idx);
            emb_per_q.push_back(emb_idx);
        }

        cnt += k;
    }

    // ---- Final: stack code_per_q into GGML shape (BT, Q) i32 ----
    // ne[0] is contiguous, so the resulting storage is q-major:
    // codes_qb[q * BT + b]. The public sampler API deinterleaves this into
    // conventional stream-major [BT, Q] storage after Session::run.
    ggml_tensor* code_out_f32 = nullptr;
    for (int q = 0; q < static_cast<int>(code_per_q.size()); q++) {
        ggml_tensor* row = ggml_reshape_2d(g, code_per_q[q], BT, 1);
        code_out_f32 = (q == 0) ? row : ggml_concat(g, code_out_f32, row, 1);
    }
    ggml_tensor* codes = ggml_cast(g, code_out_f32, GGML_TYPE_I32);  // (BT, Q)
    ggml_set_name(codes, "codes_out");
    ggml_set_output(codes);
    tc->cache_tensor("codes_out", gr::ggml_bf_tensor(codes, hc_bf.buft));

    gr::TensorBag out;
    out.add_tensor(gr::ggml_bf_tensor(codes, hc_bf.buft));
    return out;
}

EarTTSSampler::EarTTSSampler(gr::BackendManager& bm, std::shared_ptr<gr::GGUFLoader> loader)
    : loader_(std::move(loader)), bm_(&bm) {
    cfg_ = EarTTSConfig::from_gguf(*loader_);
    sample_module_ = std::make_unique<EarTTSSampleModule>(cfg_);
    sample_session_ = std::make_unique<gr::Session>(bm, sample_module_.get(), loader_.get());
    sample_session_->set_run_cache_capacity(16);
    sample_session_->setup();
}

EarTTSSampler::~EarTTSSampler() = default;

void
EarTTSSampler::dump_schedules(std::ostream& os) const {
    if (sample_session_)
        sample_session_->dump_schedule(os, "eartts.sample");
    if (mog_session_)
        mog_session_->dump_schedule(os, "eartts.mog_step");
}

void
EarTTSSampler::ensure_mog_session() {
    if (mog_session_)
        return;
    mog_module_ = std::make_unique<EarTTSMogStepModule>(cfg_);
    mog_session_ = std::make_unique<gr::Session>(*bm_, mog_module_.get(), loader_.get());
    mog_session_->setup();
}


// Dequantize sampler.rvq_embs and sampler.mog_head.low_mat from the GGUF
// bytes into host F32 (mirrors tts_ggml's dequant_to_host_f32 caches).
void
EarTTSSampler::ensure_host_tables() {
    if (!rvq_host_.empty())
        return;
    auto dequant = [&](const std::string& name, std::vector<float>& dst) {
        auto ne = loader_->get_tensor_ne(name);
        if (ne.empty() || std::any_of(ne.begin(), ne.end(), [](int64_t d) { return d <= 0; }))
            throw std::runtime_error("eartts: " + name + " has invalid dimensions");
        int64_t n = 1;
        for (int64_t d : ne) n *= d;
        const ggml_type t = loader_->get_tensor_type(name);
        const size_t nbytes = ggml_row_size(t, ne[0]) * (n / ne[0]);
        const char* data = loader_->get_tensor_file_data(name, nbytes);
        dst.resize(static_cast<size_t>(n));
        if (t == GGML_TYPE_F32) {
            std::memcpy(dst.data(), data, static_cast<size_t>(n) * sizeof(float));
            return;
        }
        const ggml_type_traits* tt = ggml_get_type_traits(t);
        if (!tt || !tt->to_float)
            throw std::runtime_error("eartts: " + name + " dtype lacks to_float");
        tt->to_float(data, dst.data(), n);
    };
    dequant("sampler.rvq_embs", rvq_host_);
    dequant("sampler.mog_head.low_mat", low_mat_host_);
}


void
EarTTSSampler::sample(
    const float* hidden_cond, const float* hidden_uncond, float guidance_scale, uint64_t seed,
    int BT, int32_t* out_codes) {
    std::vector<uint64_t> seeds(static_cast<size_t>(BT), seed);
    sample_batch(hidden_cond, hidden_uncond, guidance_scale, seeds.data(), BT, out_codes);
}

void
EarTTSSampler::sample_batch(
    const float* hidden_cond, const float* hidden_uncond, float guidance_scale,
    const uint64_t* seeds, int BT, int32_t* out_codes) {
    if (BT <= 0)
        throw std::invalid_argument("eartts: batch size must be positive");
    const int N = cfg_.mog_num_predictions;
    const int L = cfg_.latent;
    const int NS = cfg_.num_sampling_iter();

    // Each row gets the exact scalar RNG stream: all of its Gumbels first,
    // followed by all Gaussians. GGML tensors use ne = (feature, BT, NS),
    // therefore their contiguous host layout is [iter][batch][feature].
    // Keeping that ordering is essential: [iter][feature][batch] happens to
    // be equivalent at BT == 1 but shuffles random values between features
    // and streams for every real batch.
    std::vector<float> gumbel(static_cast<size_t>(NS) * N * BT);
    std::vector<float> gauss(static_cast<size_t>(NS) * L * BT);
    for (int b = 0; b < BT; ++b) {
        std::mt19937_64 rng(seeds[b] == 0 ? std::random_device{}() : seeds[b]);
        std::uniform_real_distribution<float> uni(1e-8f, 1.0f);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (int iter = 0; iter < NS; ++iter)
            for (int n = 0; n < N; ++n) {
                const float u = uni(rng);
                gumbel[(static_cast<size_t>(iter) * BT + b) * N + n] =
                    -std::log(-std::log(u) + 1e-8f);
            }
        for (int iter = 0; iter < NS; ++iter)
            for (int l = 0; l < L; ++l)
                gauss[(static_cast<size_t>(iter) * BT + b) * L + l] = normal(rng);
    }

    sample_with_noise(
        hidden_cond, hidden_uncond, guidance_scale, gumbel.data(), gauss.data(), NS, BT, out_codes);
}


void
EarTTSSampler::sample_with_noise(
    const float* hidden_cond, const float* hidden_uncond, float guidance_scale, const float* gumbel,
    const float* gauss, int n_iter, int BT, int32_t* out_codes) {
    if (n_iter != cfg_.num_sampling_iter())
        throw std::runtime_error("eartts: n_iter must equal num_sampling_iter()");
    if (guidance_scale > 0.f && !hidden_uncond)
        throw std::runtime_error("eartts: guidance_scale > 0 requires hidden_uncond");

    const int H = cfg_.hidden;
    const int N = cfg_.mog_num_predictions;
    const int L = cfg_.latent;
    const int Q = cfg_.num_quantizers;
    const int NS = cfg_.num_sampling_iter();
    std::lock_guard<std::mutex> lk(mu_);

    // CFG is disabled bit-exactly by gs = 0 (x = x_cond + 0 * delta); the
    // uncond input then carries zeros.
    const bool has_uncond = (guidance_scale > 0.f && hidden_uncond != nullptr);
    const float gs_val = has_uncond ? guidance_scale : 0.f;
    std::vector<float> hu_zeros;
    const float* hu = hidden_uncond;
    if (!hu) {
        hu_zeros.assign(static_cast<size_t>(H) * BT, 0.f);
        hu = hu_zeros.data();
    }

    std::vector<gr::Session::Input> inputs = {
        {"input.h_cond", GGML_TYPE_F32, hidden_cond, {H, BT}},
        {"input.h_uncond", GGML_TYPE_F32, hu, {H, BT}},
        {"input.gumbel", GGML_TYPE_F32, gumbel, {N, BT, NS}},
        {"input.gauss", GGML_TYPE_F32, gauss, {L, BT, NS}},
        {"input.gs", GGML_TYPE_F32, &gs_val, {1}},
    };
    // Graph output ne (BT, Q) is q-major in GGML storage because ne[0]
    // (BT) is contiguous. Deinterleave it into the API's [BT, Q] layout.
    std::vector<int32_t> codes_qb(static_cast<size_t>(BT) * Q);
    std::vector<gr::Session::Output> outputs = {
        {0, "", codes_qb.data(), codes_qb.size() * sizeof(int32_t)},
    };
    sample_session_->run(inputs, outputs);
    for (int b = 0; b < BT; ++b)
        for (int q = 0; q < Q; ++q)
            out_codes[static_cast<size_t>(b) * Q + q] = codes_qb[static_cast<size_t>(q) * BT + b];
}


// Host-orchestrated MaskGIT loop (any BT). Faithful port of tts_ggml's
// run_maskgit + run_mog_step host-side parts: depth-sum, TopPLogitsWarper +
// Gumbel-Max, low_mat matmul, z noise, per-quantizer RVQ nearest-neighbour.
void
EarTTSSampler::run_maskgit(
    const float* h_cond, const float* h_uncond, float guidance_scale, const float* ext_gumbel,
    const float* ext_gauss, int BT, int32_t* out_codes) {
    if (BT <= 0)
        throw std::invalid_argument("eartts: batch size must be positive");
    const int H = cfg_.hidden;
    const int L = cfg_.latent;
    const int N = cfg_.mog_num_predictions;
    const int LR = cfg_.mog_low_rank;
    const int CB = cfg_.codebook_size;
    const int Q = cfg_.num_quantizers;
    const float top_p = cfg_.top_p_or_k;

    std::lock_guard<std::mutex> lk(mu_);
    ensure_mog_session();
    ensure_host_tables();

    const bool has_uncond = (guidance_scale > 0.f && h_uncond != nullptr);
    const float gs_val = has_uncond ? guidance_scale : 0.f;
    std::vector<float> hu_zeros;
    const float* hu = h_uncond;
    if (!has_uncond) {
        hu_zeros.assign(static_cast<size_t>(H) * BT, 0.f);
        hu = hu_zeros.data();
    }

    // code (Q, BT) — initialised to codebook_size (== "masked").
    std::vector<int32_t> code(static_cast<size_t>(Q) * BT, CB);

    std::vector<float> depth_sum(static_cast<size_t>(L) * BT);
    std::vector<float> logits_h(static_cast<size_t>(N) * BT);
    std::vector<float> mus_h(static_cast<size_t>(LR) * N * BT);
    std::vector<float> logs_h(static_cast<size_t>(BT));
    std::vector<float> else_h(static_cast<size_t>(L) * BT);
    std::vector<float> probs(N);
    std::vector<int> sorted_idx(N);
    std::vector<float> z(static_cast<size_t>(L) * BT);
    std::vector<float> norm2(static_cast<size_t>(CB));

    int iter_idx = 0;
    int cnt = 0;
    for (int k : cfg_.sampling_per_step) {
        if (k == 0) {
            iter_idx++;
            continue;
        }
        const float* gum = ext_gumbel + static_cast<size_t>(iter_idx) * N * BT;
        const float* gau = ext_gauss + static_cast<size_t>(iter_idx) * L * BT;

        // ---- Depth-sum host-side (masked index CB contributes a zero row).
        std::fill(depth_sum.begin(), depth_sum.end(), 0.f);
        for (int i = 0; i < Q; i++) {
            const float* tbl_i = rvq_host_.data() + static_cast<size_t>(i) * CB * L;
            for (int bt = 0; bt < BT; bt++) {
                const int idx = code[static_cast<size_t>(i) * BT + bt];
                if (idx >= CB)
                    continue;
                const float* row = tbl_i + static_cast<size_t>(idx) * L;
                float* dst = depth_sum.data() + static_cast<size_t>(bt) * L;
                for (int l = 0; l < L; l++) dst[l] += row[l];
            }
        }

        // ---- MoG-head graph evaluation.
        std::vector<gr::Session::Input> inputs = {
            {"input.depth_sum", GGML_TYPE_F32, depth_sum.data(), {L, BT}},
            {"input.h_cond", GGML_TYPE_F32, h_cond, {H, BT}},
            {"input.h_uncond", GGML_TYPE_F32, hu, {H, BT}},
            {"input.gs", GGML_TYPE_F32, &gs_val, {1}},
        };
        std::vector<gr::Session::Output> outputs = {
            {0, "", logits_h.data(), logits_h.size() * sizeof(float)},
            {1, "", mus_h.data(), mus_h.size() * sizeof(float)},
            {2, "", logs_h.data(), logs_h.size() * sizeof(float)},
            {3, "", else_h.data(), else_h.size() * sizeof(float)},
        };
        mog_session_->run(inputs, outputs);

        // ---- Host TopPLogitsWarper(top_p) + Gumbel-Max per batch row.
        for (int bt = 0; bt < BT; bt++) {
            float* L_row = logits_h.data() + static_cast<size_t>(bt) * N;
            const float* G_row = gum + static_cast<size_t>(bt) * N;

            if (top_p > 0.0f && top_p < 1.0f) {
                float maxv = L_row[0];
                for (int i = 1; i < N; i++)
                    if (L_row[i] > maxv)
                        maxv = L_row[i];
                float sum = 0.0f;
                for (int i = 0; i < N; i++) {
                    probs[i] = std::exp(L_row[i] - maxv);
                    sum += probs[i];
                }
                const float inv_sum = 1.0f / sum;
                for (int i = 0; i < N; i++) probs[i] *= inv_sum;

                for (int i = 0; i < N; i++) sorted_idx[i] = i;
                std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
                    return probs[a] < probs[b];
                });

                // Mask the cumulative bottom (1 - top_p) mass; the top-1
                // token is unconditionally kept (min_tokens_to_keep=1).
                const float threshold = 1.0f - top_p;
                float cum = 0.0f;
                for (int s_i = 0; s_i < N - 1; s_i++) {
                    const int i = sorted_idx[s_i];
                    cum += probs[i];
                    if (cum <= threshold)
                        L_row[i] = -INFINITY;
                }
            }

            int best = 0;
            float best_v = -INFINITY;
            for (int i = 0; i < N; i++) {
                const float v = L_row[i] + G_row[i];
                if (v > best_v) {
                    best_v = v;
                    best = i;
                }
            }

            // mu = (low_mat[best] @ mu_low) * exp(logs) + proj_else, then
            // z = mu + exp(logs) * noise_scale * gauss.
            const float* mus_bt = mus_h.data() + static_cast<size_t>(bt) * LR * N;
            const float* mu_low = mus_bt + static_cast<size_t>(best) * LR;
            const float* low_mat_idx = low_mat_host_.data() + static_cast<size_t>(best) * L * LR;
            const float scale_mu = std::exp(logs_h[bt]);
            const float noise_sc = scale_mu * cfg_.noise_scale;
            const float* else_b = else_h.data() + static_cast<size_t>(bt) * L;
            const float* gau_b = gau + static_cast<size_t>(bt) * L;
            float* z_b = z.data() + static_cast<size_t>(bt) * L;
            for (int l = 0; l < L; l++) {
                float acc = 0.0f;
                const float* row = low_mat_idx + static_cast<size_t>(l) * LR;
                for (int kk = 0; kk < LR; kk++) acc += row[kk] * mu_low[kk];
                const float mu_l = acc * scale_mu + else_b[l];
                z_b[l] = mu_l + noise_sc * gau_b[l];
            }
        }

        // ---- _depthsum_encoding_step: RVQ nearest-neighbour per position.
        for (int i = cnt; i < cnt + k && i < Q; i++) {
            const float* tbl = rvq_host_.data() + static_cast<size_t>(i) * CB * L;
            for (int v = 0; v < CB; v++) {
                const float* row = tbl + static_cast<size_t>(v) * L;
                float n2 = 0.f;
                for (int l = 0; l < L; l++) n2 += row[l] * row[l];
                norm2[v] = n2;
            }
            for (int bt = 0; bt < BT; bt++) {
                float* r_bt = z.data() + static_cast<size_t>(bt) * L;
                float best = -std::numeric_limits<float>::infinity();
                int best_v = 0;
                for (int v = 0; v < CB; v++) {
                    const float* row = tbl + static_cast<size_t>(v) * L;
                    float dot = 0.f;
                    for (int l = 0; l < L; l++) dot += row[l] * r_bt[l];
                    const float score = dot - 0.5f * norm2[v];
                    if (score > best) {
                        best = score;
                        best_v = v;
                    }
                }
                code[static_cast<size_t>(i) * BT + bt] = best_v;
                const float* row = tbl + static_cast<size_t>(best_v) * L;
                for (int l = 0; l < L; l++) r_bt[l] -= row[l];
            }
        }
        cnt += k;
        iter_idx++;
    }

    // Output codes — caller layout: [BT, Q] row-major. Transpose from (Q, BT).
    for (int bt = 0; bt < BT; bt++)
        for (int q = 0; q < Q; q++)
            out_codes[static_cast<size_t>(bt) * Q + q] = code[static_cast<size_t>(q) * BT + bt];
}


}  // namespace nemo_speech::s2s
