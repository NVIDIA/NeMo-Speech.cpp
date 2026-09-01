// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "embedder.h"

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

#include "ggml.h"
#include "graph_utils.h"

namespace gr = ggml_runtime;

namespace nemo_speech::s2s {

using namespace eartts_graph;

// ===========================================================================
// EarTTSEmbedModule
// ===========================================================================

void
EarTTSEmbedModule::define_tensors(gr::Session* s) {
    declare_like(s, "bos_emb");
    declare_like(s, "null_emb");
    if (cfg_.use_audio_prompt_proj)
        declare_like(s, "audio_prompt_projection_W");
    for (int i = 0; i < cfg_.num_quantizers; i++)
        declare_like(s, "rvq_embs." + std::to_string(i) + ".weight");
    declare_like(s, "embed_code.weight");

    declare_like(s, "char.embed_subwords.weight");
    declare_like(s, "char.embed_subwords_mask.weight");
    declare_like(s, "char.embed_tokens.weight");
    declare_like(s, "char.proj_embedding.weight");
    declare_like(s, "char.encoder.norm.weight");
    for (int L = 0; L < cfg_.char_cfg.n_layers; L++) {
        const std::string p = "char.encoder.layers." + std::to_string(L) + ".";
        for (const char* part :
             {"pre_self_attn_layernorm.weight", "post_self_attn_layernorm.weight",
              "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
              "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
              "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
              "mlp.down_proj.weight"})
            declare_like(s, p + part);
    }

    if (cfg_.use_subword_flag) {
        declare_like(s, "subword_flag_emb.cont_emb.weight");
        declare_like(s, "subword_flag_emb.is_continuation");
    }
    if (cfg_.use_bos_eos) {
        declare_like(s, "bos_eos_emb.special_emb.weight");
        declare_like(s, "bos_eos_emb.special_flags");
    }
    if (cfg_.use_gated_fusion) {
        declare_like(s, "gated_fusion.audio_proj.weight");
        declare_like(s, "gated_fusion.audio_proj.bias");
        declare_like(s, "gated_fusion.text_proj.weight");
        declare_like(s, "gated_fusion.text_proj.bias");
        declare_like(s, "gated_fusion.gate_sigmoid");
        declare_like(s, "gated_fusion.residual_scale_sigmoid");
        declare_like(s, "gated_fusion.final_norm.weight");
    }
}

void
EarTTSEmbedModule::set_data(gr::Session* s) {
    s->load_weight("bos_emb");
    s->load_weight("null_emb");
    if (cfg_.use_audio_prompt_proj)
        s->load_weight("audio_prompt_projection_W");
    for (int i = 0; i < cfg_.num_quantizers; i++)
        s->load_weight("rvq_embs." + std::to_string(i) + ".weight");
    s->load_weight("embed_code.weight");

    s->load_weight("char.embed_subwords.weight");
    s->load_weight("char.embed_subwords_mask.weight");
    s->load_weight("char.embed_tokens.weight");
    s->load_weight("char.proj_embedding.weight");
    s->load_weight("char.encoder.norm.weight");
    for (int L = 0; L < cfg_.char_cfg.n_layers; L++) {
        const std::string p = "char.encoder.layers." + std::to_string(L) + ".";
        for (const char* part :
             {"pre_self_attn_layernorm.weight", "post_self_attn_layernorm.weight",
              "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
              "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
              "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
              "mlp.down_proj.weight"})
            s->load_weight(p + part);
    }

    if (cfg_.use_subword_flag) {
        s->load_weight("subword_flag_emb.cont_emb.weight");
        s->load_weight("subword_flag_emb.is_continuation");
    }
    if (cfg_.use_bos_eos) {
        s->load_weight("bos_eos_emb.special_emb.weight");
        s->load_weight("bos_eos_emb.special_flags");
    }
    if (cfg_.use_gated_fusion) {
        s->load_weight("gated_fusion.audio_proj.weight");
        s->load_weight("gated_fusion.audio_proj.bias");
        s->load_weight("gated_fusion.text_proj.weight");
        s->load_weight("gated_fusion.text_proj.bias");
        s->load_weight("gated_fusion.gate_sigmoid");
        s->load_weight("gated_fusion.residual_scale_sigmoid");
        s->load_weight("gated_fusion.final_norm.weight");
    }
}

gr::TensorBag
EarTTSEmbedModule::build_graph(gr::Session* s, gr::TensorBag in, gr::TensorContainer* tc) {
    const int Q = cfg_.num_quantizers;
    const int H = cfg_.hidden;

    auto tok = in.get_tensor(0);  // (BT, Q) i32, q-major memory
    ggml_context* g = tc->get_ctx_of_buffer_type(tok.buft).ctx;
    const int BT = static_cast<int>(tok.tensor->ne[0]);
    ggml_tensor* tok_codes = tok.tensor;
    ggml_tensor* text_ids = in.get_tensor(1).tensor;      // (BT,) i32
    ggml_tensor* tmask = in.get_tensor(2).tensor;         // (BT,) f32
    ggml_tensor* bmask = in.get_tensor(3).tensor;         // (BT,) f32
    ggml_tensor* baked_prompt = in.get_tensor(4).tensor;  // (H, BT) f32
    ggml_tensor* use_baked = in.get_tensor(5).tensor;     // (1,) f32

    // ---------- Audio embedding ----------
    // RVQ depth-sum: audio = sum_i embedding(codes[i], rvq_embs[i]) -> (L, BT)
    ggml_tensor* audio = nullptr;
    for (int i = 0; i < Q; i++) {
        ggml_tensor* idx_i =
            ggml_view_1d(g, tok_codes, BT, static_cast<size_t>(i) * BT * sizeof(int32_t));
        ggml_tensor* emb_i =
            ggml_get_rows(g, W(s, "rvq_embs." + std::to_string(i) + ".weight"), idx_i);
        audio = (i == 0) ? emb_i : ggml_add(g, audio, emb_i);
    }
    audio = linear(g, W(s, "embed_code.weight"), audio);  // (H, BT)

    // Optional audio prompt projection on pre-bos rows.
    if (cfg_.use_audio_prompt_proj) {
        // PT: prompt_latent = audio @ W -> ggml mul_mat with W^T contiguous.
        ggml_tensor* Wp = W(s, "audio_prompt_projection_W");
        ggml_tensor* Wf = (Wp->type == GGML_TYPE_F32) ? Wp : ggml_cast(g, Wp, GGML_TYPE_F32);
        ggml_tensor* W_T = ggml_cont(g, ggml_transpose(g, Wf));
        ggml_tensor* projected_prompt = linear(g, W_T, audio);  // (H, BT)
        // Current checkpoints contain the frozen projection result in the
        // prompt file.  Blend it in when present; older checkpoints retain
        // the live projection path.
        ggml_tensor* prompt_latent = ggml_add(
            g, projected_prompt,
            ggml_mul(g, ggml_sub(g, baked_prompt, projected_prompt), use_baked));
        // Match torch exactly: pre_bos_mask = (bos_mask == 0).  Decode uses
        // bos_mask=1e-20, so 1-bos_mask is not a valid substitute here.
        ggml_tensor* pre_bos =
            ggml_scale_bias(g, ggml_step(g, bmask), -1.0f, 1.0f);  // 1 - step(bos)
        ggml_tensor* delta = ggml_sub(g, prompt_latent, audio);
        ggml_tensor* mask_bcast = ggml_reshape_2d(g, pre_bos, 1, BT);
        audio = ggml_add(g, audio, ggml_mul(g, delta, mask_bcast));
    }

    // bos addition: audio += bos_mask * bos_emb.
    {
        ggml_tensor* bos_2d = ggml_reshape_2d(g, W(s, "bos_emb"), H, 1);
        ggml_tensor* bos_rep = ggml_repeat(g, bos_2d, audio);  // (H, BT)
        ggml_tensor* bm_2d = ggml_reshape_2d(g, bmask, 1, BT);
        audio = ggml_add(g, audio, ggml_mul(g, bos_rep, bm_2d));
    }

    // ---------- Text embedding (cond branch) ----------
    ggml_tensor* text_emb = build_char_encoder(g, s, cfg_, text_ids, BT);  // (H, BT)
    {
        ggml_tensor* tm_2d = ggml_reshape_2d(g, tmask, 1, BT);
        text_emb = ggml_mul(g, text_emb, tm_2d);
    }
    if (cfg_.use_subword_flag) {
        ggml_tensor* tbl_src = W(s, "subword_flag_emb.is_continuation");
        const int64_t Nv = tbl_src->ne[0];
        ggml_tensor* tbl = ggml_reshape_2d(g, tbl_src, 1, Nv);
        ggml_tensor* flag = ggml_get_rows(g, tbl, text_ids);  // (1, BT) i32
        ggml_tensor* flag_1d = ggml_reshape_1d(g, flag, BT);
        ggml_tensor* cont_emb = ggml_get_rows(g, W(s, "subword_flag_emb.cont_emb.weight"), flag_1d);
        text_emb = ggml_add(g, text_emb, cont_emb);
    }
    if (cfg_.use_bos_eos) {
        ggml_tensor* tbl_src = W(s, "bos_eos_emb.special_flags");
        const int64_t Nv = tbl_src->ne[0];
        ggml_tensor* tbl = ggml_reshape_2d(g, tbl_src, 1, Nv);
        ggml_tensor* flag = ggml_get_rows(g, tbl, text_ids);
        ggml_tensor* flag_1d = ggml_reshape_1d(g, flag, BT);
        ggml_tensor* sp_emb = ggml_get_rows(g, W(s, "bos_eos_emb.special_emb.weight"), flag_1d);
        text_emb = ggml_add(g, text_emb, sp_emb);
    }

    // ---------- Uncond text branch = null_emb broadcast ----------
    ggml_tensor* null_text;
    {
        ggml_tensor* zeros = ggml_scale(g, audio, 0.0f);
        ggml_tensor* n2d = ggml_reshape_2d(g, W(s, "null_emb"), H, 1);
        null_text = ggml_add(g, zeros, n2d);
    }

    // ---------- Combine (gated fusion or plain add) ----------
    auto build_combine = [&](ggml_tensor* a, ggml_tensor* t) -> ggml_tensor* {
        if (!cfg_.use_gated_fusion)
            return ggml_add(g, a, t);
        ggml_tensor* a_scaled = ggml_scale(g, a, 1.0f / static_cast<float>(cfg_.num_quantizers));
        ggml_tensor* ah = linear(g, W(s, "gated_fusion.audio_proj.weight"), a_scaled);
        ah = ggml_add(g, ah, ggml_reshape_2d(g, W(s, "gated_fusion.audio_proj.bias"), H, 1));
        ggml_tensor* th = linear(g, W(s, "gated_fusion.text_proj.weight"), t);
        th = ggml_add(g, th, ggml_reshape_2d(g, W(s, "gated_fusion.text_proj.bias"), H, 1));
        // gate * ah + (1 - gate) * th = th + gate * (ah - th).
        ggml_tensor* g_sig = ggml_reshape_2d(g, W(s, "gated_fusion.gate_sigmoid"), H, 1);
        ggml_tensor* delta = ggml_sub(g, ah, th);
        ggml_tensor* h = ggml_add(g, th, ggml_mul(g, delta, g_sig));
        h = ggml_mul(g, h, W(s, "gated_fusion.residual_scale_sigmoid"));  // (1,) broadcast
        h = gemma3_rms_norm(g, h, W(s, "gated_fusion.final_norm.weight"), cfg_.char_cfg.rms_eps);
        return h;
    };

    ggml_tensor* out_c = to_f32(g, build_combine(audio, text_emb));
    ggml_tensor* out_u = to_f32(g, build_combine(audio, null_text));
    ggml_set_name(out_c, "out_cond");
    ggml_set_name(out_u, "out_uncond");
    ggml_set_output(out_c);
    ggml_set_output(out_u);
    tc->cache_tensor("out_cond", gr::ggml_bf_tensor(out_c, tok.buft));
    tc->cache_tensor("out_uncond", gr::ggml_bf_tensor(out_u, tok.buft));

    gr::TensorBag out;
    out.add_tensor(gr::ggml_bf_tensor(out_c, tok.buft));
    out.add_tensor(gr::ggml_bf_tensor(out_u, tok.buft));
    return out;
}

// ===========================================================================
// EarTTSMogStepModule
// ===========================================================================

EarTTSEmbedder::EarTTSEmbedder(gr::BackendManager& bm, std::shared_ptr<gr::GGUFLoader> loader)
    : loader_(std::move(loader)) {
    cfg_ = EarTTSConfig::from_gguf(*loader_);
    module_ = std::make_unique<EarTTSEmbedModule>(cfg_);
    session_ = std::make_unique<gr::Session>(bm, module_.get(), loader_.get());
    session_->setup();
}

EarTTSEmbedder::~EarTTSEmbedder() = default;

void
EarTTSEmbedder::dump_schedules(std::ostream& os) const {
    if (session_)
        session_->dump_schedule(os, "eartts.embed");
}

void
EarTTSEmbedder::embed_pair(
    const int32_t* acoustic, const int32_t* text, const float* text_mask, const float* bos_mask,
    int BT, float* out_cond, float* out_uncond, const float* audio_prompt_latent) {
    const int Q = cfg_.num_quantizers;
    const int H = cfg_.hidden;

    if (BT <= 0)
        throw std::invalid_argument("eartts: embed batch size must be positive");

    // Keep the char-aware side-network on its scalar-parity CUDA kernel
    // regime. At widths above four, its BF16 GEMMs produce a one-ULP change
    // at row 4 even when every input row is identical. Acoustic sampling is
    // discrete, so that otherwise tiny difference recursively changes the
    // waveform. Chunking only this inexpensive embed graph preserves exact
    // per-stream results; the LLM and TTS backbone still use the full batch.
    constexpr int kDeterministicBatch = 4;
    if (BT > kDeterministicBatch) {
        for (int begin = 0; begin < BT; begin += kDeterministicBatch) {
            const int count = std::min(kDeterministicBatch, BT - begin);
            embed_pair(
                acoustic + static_cast<size_t>(begin) * Q, text + begin, text_mask + begin,
                bos_mask + begin, count, out_cond + static_cast<size_t>(begin) * H,
                out_uncond + static_cast<size_t>(begin) * H,
                audio_prompt_latent ? audio_prompt_latent + static_cast<size_t>(begin) * H
                                    : nullptr);
        }
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);

    // acoustic is row-major [BT, Q]; the graph reads per-q slices, so upload
    // q-major: codes_t[q*BT + bt].
    std::vector<int32_t> codes_t(static_cast<size_t>(BT) * Q);
    for (int bt = 0; bt < BT; bt++)
        for (int q = 0; q < Q; q++)
            codes_t[static_cast<size_t>(q) * BT + bt] = acoustic[static_cast<size_t>(bt) * Q + q];

    std::vector<float> zero_prompt;
    if (audio_prompt_latent == nullptr) {
        zero_prompt.assign(static_cast<size_t>(BT) * H, 0.0f);
        audio_prompt_latent = zero_prompt.data();
    }
    const float use_baked = zero_prompt.empty() ? 1.0f : 0.0f;

    std::vector<gr::Session::Input> inputs = {
        {"input.acoustic", GGML_TYPE_I32, codes_t.data(), {BT, Q}},
        {"input.text", GGML_TYPE_I32, text, {BT}},
        {"input.text_mask", GGML_TYPE_F32, text_mask, {BT}},
        {"input.bos_mask", GGML_TYPE_F32, bos_mask, {BT}},
        {"input.audio_prompt_latent", GGML_TYPE_F32, audio_prompt_latent, {H, BT}},
        {"input.use_prebaked_prompt", GGML_TYPE_F32, &use_baked, {1}},
    };
    // Graph outputs are ggml (H, BT): ne[0]=H contiguous == row-major [BT, H].
    std::vector<gr::Session::Output> outputs = {
        {0, "", out_cond, static_cast<size_t>(BT) * H * sizeof(float)},
        {1, "", out_uncond, static_cast<size_t>(BT) * H * sizeof(float)},
    };
    session_->run(inputs, outputs);
}

}  // namespace nemo_speech::s2s
