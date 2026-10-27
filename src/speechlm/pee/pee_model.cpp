// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pee_model.h"

#if defined(NEMO_SPEECH_GGML_PATCHED) && defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "nvtx_utils.h"

using namespace nemo_speech::speechlm;

namespace {

// Streaming Sortformer parameters: NeMo's PEE online-inference defaults (FIFO, cache length, update
// period) and the checkpoint's sortformer_modules config (silence frames, score threshold, boosts).
constexpr int kFifoLen = 40, kCacheLen = 188, kUpdatePeriod = 300, kSilFramesPerSpk = 1;
constexpr float kPredScoreThreshold = 0.25f, kScoresBoostLatest = 0.05f;
constexpr float kStrongBoostRate = 0.75f, kWeakBoostRate = 1.5f, kMinPosScoresRate = 0.5f;
// Finite stand-in for the +inf (silence frame) scores on the graph (product-domain scores are below
// 20), so differences stay finite.
constexpr float kPosScore = 1e30f;

ggml_runtime::TensorBag
bag1(ggml_tensor* t, ggml_backend_buffer_type_t buft) {
    ggml_runtime::TensorBag b;
    b.add_tensor(ggml_runtime::ggml_bf_tensor(t, buft));
    return b;
}

// Contiguous F32 (ne0, ne1, ne2) tensor of value v.
ggml_tensor*
constant_3d(ggml_context* ctx, float v, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor* t = ggml_arange(ctx, 0.0f, static_cast<float>(ne0 * ne1 * ne2), 1.0f);
    return ggml_reshape_3d(ctx, ggml_scale_bias(ctx, t, 0.0f, v), ne0, ne1, ne2);
}

// 1/0 mask of the k largest values in each row of key (N, R, B), ties to the lower index, as the
// host's topk: the k-th largest value is the threshold; every value above it is in, and of the
// values equal to it the first (k - #above) by index.
ggml_tensor*
select_top(ggml_context* ctx, ggml_tensor* key, int k) {
    const int64_t N = key->ne[0], R = key->ne[1], B = key->ne[2];
    ggml_tensor* order = ggml_argsort(ctx, key, GGML_SORT_ORDER_DESC);  // (N, R, B) i32
    ggml_tensor* kth = ggml_cont(
        ctx, ggml_view_3d(
                 ctx, order, 1, R, B, order->nb[1], order->nb[2],
                 static_cast<size_t>(k - 1) * ggml_element_size(order)));
    ggml_tensor* thr = ggml_get_rows(
        ctx, ggml_reshape_3d(ctx, key, 1, N, R * B), ggml_reshape_2d(ctx, kth, 1, R * B));
    ggml_tensor* diff = ggml_sub(ctx, key, ggml_reshape_3d(ctx, thr, 1, R, B));
    ggml_tensor* above = ggml_step(ctx, diff);
    ggml_tensor* below = ggml_step(ctx, ggml_scale(ctx, diff, -1.0f));
    ggml_tensor* tie = ggml_scale_bias(ctx, ggml_add(ctx, above, below), -1.0f, 1.0f);
    // ties still needed: k - #above (>= 1); a tie is in while its running count is at most that
    ggml_tensor* need =
        ggml_scale_bias(ctx, ggml_sum_rows(ctx, above), -1.0f, static_cast<float>(k));
    ggml_tensor* rank = ggml_cumsum(ctx, tie);
    ggml_tensor* take =
        ggml_step(ctx, ggml_scale_bias(ctx, ggml_sub(ctx, rank, need), -1.0f, 0.5f));
    return ggml_add(ctx, above, ggml_mul(ctx, tie, take));
}

nemo_speech::speechlm::RopeTransformerConfig
parse_branch(const ggml_runtime::GGUFLoader& loader, const std::string& p) {
    nemo_speech::speechlm::RopeTransformerConfig c;
    c.feat_in = loader.get_u32(p + ".feat_in", c.feat_in);
    c.d_model = loader.get_u32(p + ".d_model", c.d_model);
    c.n_heads = loader.get_u32(p + ".n_heads", c.n_heads);
    c.n_layers = loader.get_u32(p + ".n_layers", c.n_layers);
    c.ff_hidden = loader.get_u32(p + ".ff_hidden", 4 * c.d_model);
    c.subsampling_factor = loader.get_u32(p + ".subsampling_factor", c.subsampling_factor);
    c.rotary_dims = loader.get_u32(p + ".rotary_dims", c.d_k());
    c.qk_norm = loader.get_bool(p + ".qk_norm", false);
    c.qkv_bias = loader.get_bool(p + ".qkv_bias", false);
    c.pre_block_norm = loader.get_bool(p + ".pre_block_norm", true);
    return c;
}

PeeConfig
parse_config(const ggml_runtime::GGUFLoader& loader) {
    if (!ggml_runtime::gguf_holds(loader, "pee"))
        throw std::runtime_error("not a pee GGUF (general.architecture or nemo_speech.components)");
    PeeConfig cfg;
    cfg.asr = parse_branch(loader, "pee.asr");
    cfg.diar = parse_branch(loader, "pee.diar");
    cfg.num_speakers = loader.get_u32("pee.diar.num_speakers", cfg.num_speakers);
    cfg.tf_d_model = loader.get_u32("pee.diar.tf_d_model", cfg.tf_d_model);
    cfg.upsample_factor = loader.get_u32("pee.diar.upsample_factor", cfg.diar.subsampling_factor);
    cfg.speaker_activity_threshold =
        loader.get_f32("pee.fusion.speaker_activity_threshold", cfg.speaker_activity_threshold);
    cfg.asr_normalize_type = loader.get_str("pee.asr.normalize_type", cfg.asr_normalize_type);
    cfg.online_inference_length =
        loader.get_u32("pee.online.inference_length", cfg.online_inference_length);
    cfg.chunk_left_context =
        loader.get_u32("pee.online.chunk_left_context", cfg.chunk_left_context);
    cfg.chunk_right_context =
        loader.get_u32("pee.online.chunk_right_context", cfg.chunk_right_context);
    cfg.sample_rate = loader.get_u32("pee.preprocessor.sample_rate", cfg.sample_rate);
    cfg.window_size = loader.get_f32("pee.preprocessor.window_size", cfg.window_size);
    cfg.window_stride = loader.get_f32("pee.preprocessor.window_stride", cfg.window_stride);
    cfg.n_fft = loader.get_u32("pee.preprocessor.n_fft", cfg.n_fft);
    cfg.n_mels = loader.get_u32("pee.preprocessor.features", cfg.n_mels);
    cfg.proj_dim = loader.get_u32("pee.fusion.proj_dim", 0);
    if (cfg.proj_dim > 0 && !loader.has_tensor("fusion.proj.weight"))
        throw std::runtime_error("pee: pee.fusion.proj_dim set but fusion.proj.weight missing");
    if (cfg.asr.subsampling_factor != cfg.diar.subsampling_factor)
        throw std::runtime_error("pee: asr/diar subsampling factors differ");
    if (cfg.upsample_factor != cfg.diar.subsampling_factor)
        throw std::runtime_error("pee: upsample_factor must equal the diar subsampling factor");
    return cfg;
}

}  // namespace

void
nemo_speech::speechlm::per_feature_normalize(const float* mel, int t, int n_mels, float* out) {
    for (int f = 0; f < n_mels; f++) {
        double mean = 0;
        for (int i = 0; i < t; i++) mean += mel[static_cast<size_t>(i) * n_mels + f];
        mean /= t;
        double ss = 0;
        for (int i = 0; i < t; i++) {
            const double d = mel[static_cast<size_t>(i) * n_mels + f] - mean;
            ss += d * d;
        }
        // unbiased std (N-1), + CONSTANT
        float std = t > 1 ? static_cast<float>(std::sqrt(ss / (t - 1))) : 0.0f;
        std += 1e-5f;
        for (int i = 0; i < t; i++)
            out[static_cast<size_t>(i) * n_mels + f] =
                static_cast<float>((mel[static_cast<size_t>(i) * n_mels + f] - mean) / std);
    }
}

// ------------------------------------------------------------------ graph --

PeeGraph::PeeGraph(
    const PeeConfig& cfg, bool want_asr_states, bool want_native_probs, bool has_sil_emb)
    : cfg_(cfg), want_asr_states_(want_asr_states), want_native_probs_(want_native_probs),
      has_sil_emb_(has_sil_emb) {
    const int64_t ln_d[4] = {cfg.asr.d_model, 1, 1, 1};
    const int64_t ln_s[4] = {cfg.num_speakers, 1, 1, 1};
    asr_norm_ = new ggml_runtime::LayerNorm("fusion.asr_norm", ln_d);
    diar_norm_ = new ggml_runtime::LayerNorm("fusion.diar_norm", ln_s);
    asr_ = new nemo_speech::speechlm::RopeTransformerEncoder("asr", cfg.asr);
    diar_ = new nemo_speech::speechlm::RopeTransformerEncoder("diar", cfg.diar);
    encoder_proj_ =
        new ggml_runtime::Linear("diar.head.encoder_proj", cfg.diar.d_model, cfg.tf_d_model);
    upsample_ = new ggml_runtime::Conv1D(
        "diar.head.subpixel_upsample", cfg.tf_d_model, cfg.tf_d_model * cfg.upsample_factor,
        /*kernel_size=*/3, /*stride=*/1, /*padding=*/1);
    head_hidden_ = new ggml_runtime::Linear(
        "diar.head.first_hidden_to_hidden", cfg.tf_d_model, cfg.tf_d_model);
    head_spks_ = new ggml_runtime::Linear(
        "diar.head.single_hidden_to_spks", cfg.tf_d_model, cfg.num_speakers);
    if (cfg.proj_dim > 0)
        proj_ = new ggml_runtime::Linear("fusion.proj", cfg.asr.d_model, cfg.proj_dim);
}

PeeGraph::~PeeGraph() {
    delete proj_;
    delete asr_;
    delete diar_;
    delete encoder_proj_;
    delete upsample_;
    delete head_hidden_;
    delete head_spks_;
    delete asr_norm_;
    delete diar_norm_;
}

void
PeeGraph::define_tensors(ggml_runtime::Session* session) {
    asr_->define_tensors(session);
    diar_->define_tensors(session);
    encoder_proj_->define_tensors(session);
    upsample_->define_tensors(session);
    head_hidden_->define_tensors(session);
    head_spks_->define_tensors(session);
    asr_norm_->define_tensors(session);
    diar_norm_->define_tensors(session);
    if (proj_)
        proj_->define_tensors(session);
    session->model_tensor_container->create_tensor_4d(
        "fusion.diar_kernel", GGML_TYPE_F32, cfg_.asr.d_model, cfg_.num_speakers, 1, 1);
    if (has_sil_emb_)
        session->model_tensor_container->create_tensor_1d(
            "diar.head.learnable_sil_emb", GGML_TYPE_F32, cfg_.diar.d_model);
}

void
PeeGraph::set_data(ggml_runtime::Session* session) {
    asr_->set_data(session);
    diar_->set_data(session);
    encoder_proj_->set_data(session);
    upsample_->set_data(session);
    head_hidden_->set_data(session);
    head_spks_->set_data(session);
    asr_norm_->set_data(session);
    diar_norm_->set_data(session);
    if (proj_)
        proj_->set_data(session);
    session->load_weight("fusion.diar_kernel");
    if (has_sil_emb_)
        session->load_weight("diar.head.learnable_sil_emb");
}

ggml_tensor*
PeeGraph::speaker_probs(
    ggml_runtime::Session* session, const ggml_runtime::ggml_bf_tensor& emb, ggml_tensor* kmask,
    ggml_runtime::TensorContainer* tc) {
    ggml_context* ctx = tc->get_ctx_of_buffer_type(emb.buft).ctx;
    auto proj = encoder_proj_->build_graph(session, bag1(emb.tensor, emb.buft), tc)
                    .get_tensor(0);  // (192, T, B)
    const int64_t T = proj.tensor->ne[1];
    const int64_t B = proj.tensor->ne[2];
    ggml_tensor* x = proj.tensor;
    if (kmask) {
        // padded tokens are zero before the upsample conv (NeMo's mask_padding), so the conv's
        // taps at a row's last valid token read zeros, as when the row runs alone (B = 1)
        ggml_tensor* keep = ggml_step(ctx, ggml_scale_bias(ctx, kmask, 1.0f, 1.0f));  // 1 / 0
        x = ggml_mul(ctx, x, ggml_reshape_3d(ctx, keep, 1, T, B));
    }
    // subpixel upsample: Conv1d(192 -> 192*8, k=3, p=1) over time (Conv1D wants (T, C, B)),
    // then each (1536) frame vector is 8 consecutive 192-d 10 ms frames.
    auto proj_t = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    auto up =
        upsample_->build_graph(session, bag1(proj_t, proj.buft), tc).get_tensor(0);  // (T, 1536, B)
    auto up_c = ggml_cont(ctx, ggml_permute(ctx, up.tensor, 1, 0, 2, 3));            // (1536, T, B)
    auto hi =
        ggml_reshape_3d(ctx, up_c, cfg_.tf_d_model, T * cfg_.upsample_factor, B);  // (192, 8T, B)
    auto h1 =
        head_hidden_->build_graph(session, bag1(ggml_relu(ctx, hi), up.buft), tc).get_tensor(0);
    auto logits = head_spks_->build_graph(session, bag1(ggml_relu(ctx, h1.tensor), h1.buft), tc)
                      .get_tensor(0);
    return ggml_sigmoid(ctx, logits.tensor);  // (n_spk, 8T, B)
}

ggml_runtime::TensorBag
PeeGraph::build_speaker_step(ggml_runtime::Session* session, ggml_runtime::TensorContainer* tc) {
    auto mel =
        tc->get_tensor_by_name("input.spk_mel");  // (n_mels, T_mel, B) raw log-mel of the window
    auto pos =
        tc->get_tensor_by_name("input.spk_pos");  // (P + T') positions over the whole concatenation
    ggml_context* ctx = tc->get_ctx_of_buffer_type(mel.buft).ctx;
    const int sub = cfg_.diar.subsampling_factor;
    auto chunk = diar_->pre_encode(session, mel, tc);  // (d_diar, T', B)
    ggml_tensor* x = chunk.tensor;
    if (tc->has_tensor_by_name("input.spk_prev"))  // [speaker cache | FIFO | window]
        x = ggml_concat(ctx, tc->get_tensor_by_name("input.spk_prev").tensor, x, 1);
    std::optional<ggml_runtime::ggml_bf_tensor> km;
    if (tc->has_tensor_by_name("input.spk_kmask"))
        km.emplace(tc->get_tensor_by_name("input.spk_kmask"));
    auto h =
        diar_
            ->encode_rows(
                session, ggml_runtime::ggml_bf_tensor(x, chunk.buft), pos, km ? &*km : nullptr, tc)
            .get_tensor(0);
    ggml_tensor* probs =
        speaker_probs(session, h, km ? km->tensor : nullptr, tc);  // (n_spk, 8L, B)
    // 80 ms: plain mean of each 8 high-resolution frames (downsample_preds)
    ggml_tensor* p_t = ggml_cont(ctx, ggml_permute(ctx, probs, 1, 0, 2, 3));  // (8L, n_spk, B)
    ggml_tensor* p80_t =
        ggml_pool_2d(ctx, p_t, GGML_OP_POOL_AVG, sub, 1, sub, 1, 0, 0);       // (L, n_spk, B)
    ggml_tensor* p80 = ggml_cont(ctx, ggml_permute(ctx, p80_t, 1, 0, 2, 3));  // (n_spk, L, B)
    ggml_runtime::TensorBag out;
    out.add_tensor(chunk);
    out.add_tensor(ggml_runtime::ggml_bf_tensor(p80, chunk.buft));
    out.add_tensor(ggml_runtime::ggml_bf_tensor(probs, chunk.buft));
    if (!tc->has_tensor_by_name("input.spk_out_rows"))
        return out;  // host state: the caller downloads [0]-[2]

    // Device state. Every row list indexes this step's rows (per batch item); a list entry one past
    // the probabilities selects an appended zero row.
    const int n_spk = cfg_.num_speakers;
    const int64_t L = p80->ne[1], B = p80->ne[2];
    auto rows = [&](const char* name) { return tc->get_tensor_by_name(name).tensor; };
    auto add = [&](ggml_tensor* t) { out.add_tensor(ggml_runtime::ggml_bf_tensor(t, chunk.buft)); };
    // [3] the window's 80 ms probabilities on its own frames (core rows, context rows zero)
    add(ggml_get_rows(
        ctx, ggml_concat(ctx, p80, constant_3d(ctx, 0.0f, n_spk, 1, B), 1),
        rows("input.spk_out_rows")));
    // [4] the core's native 10 ms probabilities: 8 sub-frames x n_spk per 80 ms row
    ggml_tensor* p10 = ggml_reshape_3d(ctx, probs, n_spk * sub, L, B);
    add(ggml_get_rows(
        ctx, ggml_concat(ctx, p10, constant_3d(ctx, 0.0f, n_spk * sub, 1, B), 1),
        rows("input.spk_core_rows")));
    if (!tc->has_tensor_by_name("input.spk_cache_rows"))
        return out;  // last window: the state is not needed again (NeMo's update_state=False)

    // SortformerModules.streaming_update (sync): the cache plus the rows popped from [FIFO | core]
    // (rows of x = [cache | FIFO | window]); their probabilities are the state's cache
    // probabilities after a compression, else this pass's, and this pass's for the popped rows.
    ggml_tensor* cache = ggml_get_rows(ctx, x, rows("input.spk_cache_rows"));  // (d_diar, n, B)
    ggml_tensor* pred_src =
        tc->has_tensor_by_name("input.spk_cpred")
            ? ggml_concat(ctx, rows("input.spk_cpred"), p80, 1)  // [state cache probs | p80]
            : p80;
    ggml_tensor* cache_preds = ggml_get_rows(ctx, pred_src, rows("input.spk_cpred_rows"));
    if (cache->ne[1] > kCacheLen)
        std::tie(cache, cache_preds) = compress_cache(session, ctx, cache, cache_preds);
    // [5] the next step's [cache | FIFO] rows, [6] the cache's probabilities
    add(tc->has_tensor_by_name("input.spk_fifo_rows")
            ? ggml_concat(ctx, cache, ggml_get_rows(ctx, x, rows("input.spk_fifo_rows")), 1)
            : cache);
    add(cache_preds);
    return out;
}

std::pair<ggml_tensor*, ggml_tensor*>
PeeGraph::compress_cache(
    ggml_runtime::Session* session, ggml_context* ctx, ggml_tensor* emb, ggml_tensor* preds) {
    // The host compress_cache below, op for op: the same product-domain f32 scores in the same
    // order, 0 / kPosScore for -inf / +inf, and top-k with ties to the lower index (select_top).
    const int64_t d = emb->ne[0], n = emb->ne[1], B = emb->ne[2];
    const int n_spk = cfg_.num_speakers;
    const int per_spk = kCacheLen / n_spk - kSilFramesPerSpk;
    const int strong = static_cast<int>(std::floor(per_spk * kStrongBoostRate));
    const int weak = static_cast<int>(std::floor(per_spk * kWeakBoostRate));
    const int min_pos = static_cast<int>(std::floor(per_spk * kMinPosScoresRate));
    const float big = std::numeric_limits<float>::max();
    ggml_tensor* p = ggml_cont(ctx, ggml_permute(ctx, preds, 1, 0, 2, 3));  // (n, n_spk, B)
    ggml_tensor* p_s = ggml_clamp(ctx, p, kPredScoreThreshold, big);
    ggml_tensor* q =
        ggml_clamp(ctx, ggml_scale_bias(ctx, p, -1.0f, 1.0f), kPredScoreThreshold, big);
    // score[t, s] = 2 p_s[t, s] prod_j (j == s ? 1 : q[t, j]), multiplied in index order:
    // factors (n, s, j, B) = q[t, j] off the diagonal, 1 on it
    ggml_tensor* spk = ggml_arange(ctx, 0.0f, static_cast<float>(n_spk), 1.0f);
    ggml_tensor* eye = ggml_step(
        ctx,
        ggml_scale_bias(
            ctx,
            ggml_abs(
                ctx, ggml_sub(
                         ctx,
                         ggml_repeat_4d(
                             ctx, ggml_reshape_4d(ctx, spk, 1, n_spk, 1, 1), 1, n_spk, n_spk, 1),
                         ggml_reshape_4d(ctx, spk, 1, 1, n_spk, 1))),
            -1.0f, 0.5f));  // (1, s, j, 1)
    ggml_tensor* factors = ggml_add(
        ctx,
        ggml_mul(
            ctx, ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, q, n, 1, n_spk, B), n, n_spk, n_spk, B),
            ggml_scale_bias(ctx, eye, -1.0f, 1.0f)),
        eye);
    ggml_tensor* score = nullptr;
    for (int j = 0; j < n_spk; ++j) {
        ggml_tensor* f = ggml_view_4d(
            ctx, factors, n, n_spk, 1, B, factors->nb[1], factors->nb[2], factors->nb[3],
            static_cast<size_t>(j) * factors->nb[2]);
        score = score ? ggml_mul(ctx, score, f) : ggml_cont(ctx, f);
    }
    score = ggml_scale(ctx, ggml_mul(ctx, ggml_reshape_3d(ctx, score, n, n_spk, B), p_s), 2.0f);
    // valid (not -inf): speech (p > 0.5), and above 1 when the speaker has min_pos such frames
    ggml_tensor* speech = ggml_step(ctx, ggml_scale_bias(ctx, p, 1.0f, -0.5f));
    ggml_tensor* positive = ggml_step(ctx, ggml_scale_bias(ctx, score, 1.0f, -1.0f));
    ggml_tensor* n_pos = ggml_sum_rows(ctx, ggml_mul(ctx, speech, positive));  // (1, n_spk, B)
    ggml_tensor* strict =
        ggml_step(ctx, ggml_scale_bias(ctx, n_pos, 1.0f, 0.5f - static_cast<float>(min_pos)));
    ggml_tensor* drop = ggml_mul(ctx, ggml_scale_bias(ctx, positive, -1.0f, 1.0f), strict);
    ggml_tensor* valid = ggml_mul(ctx, speech, ggml_scale_bias(ctx, drop, -1.0f, 1.0f));
    // newly added frames (index >= the cache length): x exp(0.05)
    ggml_tensor* newer = ggml_step(
        ctx, ggml_scale_bias(
                 ctx, ggml_arange(ctx, 0.0f, static_cast<float>(n), 1.0f), 1.0f,
                 0.5f - static_cast<float>(kCacheLen)));
    score = ggml_mul(
        ctx, score,
        ggml_add(
            ctx, ggml_scale(ctx, newer, std::exp(kScoresBoostLatest)),
            ggml_scale_bias(ctx, newer, -1.0f, 1.0f)));
    // at least `strong` frames per speaker (x 4), then no single speaker dominates (x 2)
    score = ggml_mul(
        ctx, score,
        ggml_scale_bias(ctx, select_top(ctx, ggml_mul(ctx, score, valid), strong), 3.0f, 1.0f));
    score = ggml_mul(
        ctx, score,
        ggml_scale_bias(ctx, select_top(ctx, ggml_mul(ctx, score, valid), weak), 1.0f, 1.0f));
    // speaker-major flattening with kSilFramesPerSpk +inf frames per speaker; keep the kCacheLen
    // largest that are not -inf, in index order, then disabled rows
    const int64_t nt = n + kSilFramesPerSpk, nf = nt * n_spk;
    ggml_tensor* flat = ggml_reshape_3d(
        ctx,
        ggml_concat(
            ctx, ggml_mul(ctx, score, valid),
            constant_3d(ctx, kPosScore, kSilFramesPerSpk, n_spk, B), 0),
        nf, 1, B);
    ggml_tensor* keep = ggml_mul(ctx, select_top(ctx, flat, kCacheLen), ggml_step(ctx, flat));
    ggml_tensor* n_keep = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, keep), 1, B);
    ggml_tensor* order = ggml_argsort(
        ctx,
        ggml_add(
            ctx, ggml_scale_bias(ctx, keep, -static_cast<float>(nf), static_cast<float>(nf)),
            ggml_arange(ctx, 0.0f, static_cast<float>(nf), 1.0f)),
        GGML_SORT_ORDER_ASC);  // kept indices ascending first
    order = ggml_cont(ctx, ggml_view_2d(ctx, order, kCacheLen, B, order->nb[2], 0));  // (K, B)
    // frame of each pick (silence frames and disabled rows: row n, the silence embedding)
    ggml_tensor* frame = ggml_repeat_4d(
        ctx,
        ggml_clamp(
            ctx, ggml_arange(ctx, 0.0f, static_cast<float>(nt), 1.0f), 0.0f, static_cast<float>(n)),
        nt, n_spk, B, 1);
    frame = ggml_reshape_2d(
        ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, frame, 1, nf, B), order), kCacheLen, B);
    ggml_tensor* disabled = ggml_step(
        ctx, ggml_scale_bias(
                 ctx,
                 ggml_sub(
                     ctx,
                     ggml_repeat_4d(
                         ctx, ggml_arange(ctx, 0.0f, static_cast<float>(kCacheLen), 1.0f),
                         kCacheLen, B, 1, 1),
                     n_keep),
                 1.0f, 0.5f));  // pick index >= kept count
    frame = ggml_add(
        ctx, ggml_mul(ctx, frame, ggml_scale_bias(ctx, disabled, -1.0f, 1.0f)),
        ggml_scale(ctx, disabled, static_cast<float>(n)));
    ggml_tensor* pick = ggml_cast(ctx, frame, GGML_TYPE_I32);  // (K, B)
    ggml_tensor* sil =
        session->model_tensor_container->get_tensor_by_name("diar.head.learnable_sil_emb").tensor;
    ggml_tensor* emb_ext = ggml_concat(ctx, emb, ggml_repeat_4d(ctx, sil, d, 1, B, 1), 1);
    ggml_tensor* preds_ext = ggml_concat(ctx, preds, constant_3d(ctx, 0.0f, n_spk, 1, B), 1);
    return {ggml_get_rows(ctx, emb_ext, pick), ggml_get_rows(ctx, preds_ext, pick)};
}

ggml_runtime::TensorBag
PeeGraph::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    (void)input_tensors;  // fetched by name
    if (tc->has_tensor_by_name("input.spk_mel"))
        return build_speaker_step(session, tc);
    auto mel = tc->get_tensor_by_name("input.mel");
    auto mean = tc->get_tensor_by_name("input.mean");
    auto inv_std = tc->get_tensor_by_name("input.inv_std");
    auto valid = tc->get_tensor_by_name("input.valid");  // (sub*T', 1, B)
    auto pos = tc->get_tensor_by_name("input.pos");
    const bool masked = tc->has_tensor_by_name("input.kmask");  // (T', 1, 1, B), padded batch
    auto bf_ctx = tc->get_ctx_of_buffer_type(mel.buft);
    ggml_context* ctx = bf_ctx.ctx;
    const int sub = cfg_.asr.subsampling_factor;

    // ---- ASR expert: per-utterance per-feature normalization on the graph ----
    ggml_tensor* mel_asr = ggml_mul(ctx, ggml_sub(ctx, mel.tensor, mean.tensor), inv_std.tensor);
    if (masked) {
        // Padded mel frames must be zero AFTER normalization (NeMo pads the
        // normalized features); (t_pad, 1, B) validity view of input.valid.
        const int64_t t_pad = mel.tensor->ne[1];
        ggml_tensor* fv = ggml_view_3d(
            ctx, valid.tensor, t_pad, 1, valid.tensor->ne[2], valid.tensor->nb[1],
            valid.tensor->nb[2], 0);
        fv = ggml_reshape_3d(ctx, ggml_cont(ctx, fv), 1, t_pad, valid.tensor->ne[2]);
        mel_asr = ggml_mul(ctx, mel_asr, fv);
    }
    ggml_runtime::TensorBag asr_in;
    asr_in.add_tensor(ggml_runtime::ggml_bf_tensor(mel_asr, mel.buft));
    asr_in.add_tensor(pos);
    if (masked)
        asr_in.add_tensor(tc->get_tensor_by_name("input.kmask"));
    auto asr_out = asr_->build_graph(session, asr_in, tc).get_tensor(0);  // (d_model, T', B)

    // ---- Speaker expert (or the stateful speaker steps' probabilities) ----
    const bool external = tc->has_tensor_by_name("input.p80");
    ggml_tensor* probs = nullptr;
    ggml_tensor* p80 = nullptr;
    if (external) {
        p80 = ggml_scale(ctx, tc->get_tensor_by_name("input.p80").tensor, 1.0f);  // (n_spk, T', B)
    } else {
        ggml_runtime::TensorBag diar_in;
        diar_in.add_tensor(mel);
        diar_in.add_tensor(pos);
        if (masked)
            diar_in.add_tensor(tc->get_tensor_by_name("input.kmask"));
        auto emb = diar_->build_graph(session, diar_in, tc).get_tensor(0);  // (512, T', B)
        probs = speaker_probs(
            session, emb, masked ? tc->get_tensor_by_name("input.kmask").tensor : nullptr,
            tc);  // (n_spk, 8T', B)

        // ---- 10 ms -> 80 ms: mask-weighted average over valid frames ----
        // pool over time needs time in ne0: (8T, n_spk, B); valid is (8T, 1, B).
        // ggml_pool_2d with a (sub x 1) window: the CUDA backend has no POOL_1D, and
        // a CPU-side split inside a cached (re-executed) session graph returns stale
        // pooled values from the previous run.
        ggml_tensor* p_t = ggml_cont(ctx, ggml_permute(ctx, probs, 1, 0, 2, 3));  // (8T, n_spk, B)
        ggml_tensor* pm = ggml_mul(ctx, p_t, valid.tensor);
        ggml_tensor* num =
            ggml_pool_2d(ctx, pm, GGML_OP_POOL_AVG, sub, 1, sub, 1, 0, 0);  // (T, n_spk, B)
        ggml_tensor* den =
            ggml_pool_2d(ctx, valid.tensor, GGML_OP_POOL_AVG, sub, 1, sub, 1, 0, 0);  // (T, 1, B)
        // fully padded 80 ms frames have den == 0: clamp so they yield 0, not NaN
        ggml_tensor* p80_t = ggml_div(ctx, num, ggml_clamp(ctx, den, 1e-6f, 1.0f));
        p80 = ggml_cont(ctx, ggml_permute(ctx, p80_t, 1, 0, 2, 3));  // (n_spk, T, B)
    }

    // ---- fusion: step(p - thr) -> LayerNorm(n_spk) -> @ kernel -> + LayerNorm(d)(asr) ----
    ggml_tensor* bin =
        ggml_step(ctx, ggml_scale_bias(ctx, p80, 1.0f, -cfg_.speaker_activity_threshold));
    auto ln_spk =
        diar_norm_->build_graph(session, bag1(bin, mel.buft), tc).get_tensor(0);  // (n_spk, T, B)
    auto kernel =
        session->model_tensor_container->get_tensor_by_name("fusion.diar_kernel");  // (d, n_spk)
    ggml_tensor* k_t = ggml_cont(ctx, ggml_transpose(ctx, kernel.tensor));          // (n_spk, d)
    ggml_tensor* infusion = ggml_mul_mat(ctx, k_t, ln_spk.tensor);                  // (d, T, B)
    auto ln_asr =
        asr_norm_->build_graph(session, bag1(asr_out.tensor, asr_out.buft), tc).get_tensor(0);
    ggml_tensor* fused = ggml_add(ctx, ln_asr.tensor, infusion);

    ggml_runtime::TensorBag out;
    out.add_tensor(ggml_runtime::ggml_bf_tensor(fused, asr_out.buft));
    out.add_tensor(ggml_runtime::ggml_bf_tensor(p80, asr_out.buft));
    if (want_asr_states_)
        out.add_tensor(asr_out);
    if (proj_) {
        // SALM modality projection: (d, T, B) -> (proj_dim, T, B), the LLM's input embeddings.
        auto pj = proj_->build_graph(session, bag1(fused, asr_out.buft), tc).get_tensor(0);
        out.add_tensor(pj);
    }
    if (want_native_probs_ && probs)
        out.add_tensor({probs, asr_out.buft});
    return out;
}

// ------------------------------------------------------------- host buffers --

// Pinned host buffers are pooled process-wide: cudaMallocHost/cudaFreeHost synchronize the device,
// so allocating them per encoder batch stalled the decoder running on another stream. Buffers
// return to the pool on release and are reused for any request of up to their size (within 2x).
// Never freed at exit (CUDA may already be torn down).
namespace {
struct PinnedPool {
    std::mutex mu;
    std::multimap<size_t, ggml_backend_buffer_t> free_bufs;
    size_t cached = 0;
    static constexpr size_t kCap = size_t(6) << 30;

    ggml_backend_buffer_t take(ggml_backend_buffer_type_t buft, size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = free_bufs.lower_bound(bytes);
            if (it != free_bufs.end() && it->first <= 2 * bytes + (size_t(16) << 20)) {
                ggml_backend_buffer_t b = it->second;
                cached -= it->first;
                free_bufs.erase(it);
                return b;
            }
        }
        // round up so later, slightly larger batches reuse the buffer
        const size_t mib = size_t(1) << 20;
        const size_t want = ((bytes + bytes / 4) + mib - 1) / mib * mib;
        return ggml_backend_buft_alloc_buffer(buft, want);
    }
    void give(ggml_backend_buffer_t b) {
        const size_t sz = ggml_backend_buffer_get_size(b);
        std::lock_guard<std::mutex> lock(mu);
        while (cached + sz > kCap &&
               !free_bufs.empty()) {  // drop the smallest cached buffers first
            auto it = free_bufs.begin();
            cached -= it->first;
            ggml_backend_buffer_free(it->second);
            free_bufs.erase(it);
        }
        free_bufs.emplace(sz, b);
        cached += sz;
    }
};
PinnedPool&
pinned_pool() {
    static PinnedPool* pool = new PinnedPool();
    return *pool;
}
}  // namespace

PeeModel::HostBuffer::HostBuffer(ggml_backend_buffer_type_t host_buft, size_t bytes) {
    reserve(host_buft, bytes);
}
PeeModel::HostBuffer::~HostBuffer() {
    if (buf_)
        pinned_pool().give(buf_);
}
void
PeeModel::HostBuffer::reserve(ggml_backend_buffer_type_t host_buft, size_t bytes) {
    if (base_ && (buf_ ? ggml_backend_buffer_get_size(buf_) >= bytes : bytes <= bytes_)) {
        bytes_ = std::max(bytes_, bytes);
        return;
    }
    if (buf_) {
        pinned_pool().give(buf_);
        buf_ = nullptr;
    }
    base_ = nullptr;
    if (host_buft) {
        buf_ = pinned_pool().take(host_buft, bytes);
        if (buf_)
            base_ = ggml_backend_buffer_get_base(buf_);
    }
    if (!base_) {  // no GPU host buffer type available: plain memory
        fallback_.resize(bytes);
        base_ = fallback_.data();
    }
    bytes_ = bytes;
}

// ------------------------------------------------------------ device blocks --

// Free device buffers by size. Taking reuses one of up to twice the size; a returned buffer is
// freed only beyond kCap bytes of idle buffers (cudaFree synchronizes the device).
struct PeeModel::DevicePool {
    std::mutex mu;
    ggml_backend_buffer_type_t buft = nullptr;
    std::multimap<size_t, ggml_backend_buffer_t> free_bufs;
    size_t cached = 0;
    static constexpr size_t kCap = size_t(8) << 30;

    ~DevicePool() {
        for (auto& [size, b] : free_bufs) ggml_backend_buffer_free(b);
    }
    ggml_backend_buffer_t take(size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = free_bufs.lower_bound(bytes);
            if (it != free_bufs.end() && it->first <= 2 * bytes + (size_t(16) << 20)) {
                ggml_backend_buffer_t b = it->second;
                cached -= it->first;
                free_bufs.erase(it);
                return b;
            }
        }
        // round up so later, slightly larger batches reuse the buffer
        const size_t mib = size_t(1) << 20;
        const size_t want = ((bytes + bytes / 4) + mib - 1) / mib * mib;
        return ggml_backend_buft_alloc_buffer(buft, want);
    }
    void give(ggml_backend_buffer_t b) {
        const size_t sz = ggml_backend_buffer_get_size(b);
        std::lock_guard<std::mutex> lock(mu);
        while (cached + sz > kCap && !free_bufs.empty()) {  // drop the smallest idle buffers first
            auto it = free_bufs.begin();
            cached -= it->first;
            ggml_backend_buffer_free(it->second);
            free_bufs.erase(it);
        }
        free_bufs.emplace(sz, b);
        cached += sz;
    }
};

PeeModel::DeviceBlock::~DeviceBlock() {
    if (ctx_)
        ggml_free(ctx_);
    if (buf_)
        pool_->give(buf_);
}

std::shared_ptr<const PeeModel::DeviceBlock>
PeeModel::keep_on_device(const ggml_runtime::DeviceTensor& out, int64_t items) {
    if (!out.valid() || !ggml_is_contiguous(out.tensor) || out.tensor->ne[3] != 1 || items < 1 ||
        items > out.tensor->ne[2])
        throw std::runtime_error("pee: projection output is not a contiguous device tensor");
    // the first `items` batch items (padding items past them are not kept)
    int64_t ne[GGML_MAX_DIMS] = {out.tensor->ne[0], out.tensor->ne[1], items, 1};
    const size_t bytes = out.tensor->nb[2] * static_cast<size_t>(items);
    auto block = std::make_shared<DeviceBlock>();
    block->pool_ = device_pool_;
    block->buf_ = device_pool_->take(bytes);
    if (!block->buf_)
        throw std::runtime_error("pee: out of device memory for the encoder output");
    ggml_init_params ip = {ggml_tensor_overhead(), nullptr, true};
    block->ctx_ = ggml_init(ip);
    block->tensor_ = ggml_new_tensor(block->ctx_, GGML_TYPE_F32, GGML_MAX_DIMS, ne);
    if (ggml_backend_tensor_alloc(
            block->buf_, block->tensor_, ggml_backend_buffer_get_base(block->buf_)) !=
        GGML_STATUS_SUCCESS)
        throw std::runtime_error("pee: device block allocation");
    // the graph has finished (its other outputs were downloaded); the copy is ordered on the same
    // stream, and the block is complete when this returns
    // (byte views: the source may be larger than the block)
    ggml_tensor from = *out.tensor, to = *block->tensor_;
    from.data = static_cast<char*>(out.tensor->data) + out.byte_offset;
    from.ne[2] = items;
    from.nb[3] = to.nb[3] = bytes;
    std::lock_guard<std::mutex> lock(bm_->compute_mutex());
    ggml_backend_tensor_copy_async(gpu_, gpu_, &from, &to);
    ggml_backend_synchronize(gpu_);
    return block;
}

std::shared_ptr<PeeModel::DeviceBlock>
PeeModel::device_block(size_t floats) {
    if (!device_pool_ || floats == 0)
        throw std::runtime_error("pee: device block without a GPU backend");
    auto block = std::make_shared<DeviceBlock>();
    block->pool_ = device_pool_;
    block->buf_ = device_pool_->take(floats * sizeof(float));
    if (!block->buf_)
        throw std::runtime_error("pee: out of device memory for the speaker state");
    ggml_init_params ip = {ggml_tensor_overhead(), nullptr, true};
    block->ctx_ = ggml_init(ip);
    block->tensor_ = ggml_new_tensor_1d(block->ctx_, GGML_TYPE_F32, static_cast<int64_t>(floats));
    if (ggml_backend_tensor_alloc(
            block->buf_, block->tensor_, ggml_backend_buffer_get_base(block->buf_)) !=
        GGML_STATUS_SUCCESS)
        throw std::runtime_error("pee: device block allocation");
    return block;
}

// ------------------------------------------------------------------ model --

PeeModel::PeeModel(
    ggml_runtime::BackendManager& bm, const std::string& gguf_path, bool want_asr_states,
    const std::string& ctc_path)
    : want_asr_states_(want_asr_states), bm_(&bm), gpu_(bm.gpu_backend_handle()) {
    loader_ = std::make_unique<ggml_runtime::GGUFLoader>(gguf_path);
    cfg_ = parse_config(*loader_);

    const int n_freq = cfg_.n_fft / 2 + 1;
    read_host_tensor("preprocessor.fb", mel_basis_, static_cast<size_t>(cfg_.n_mels) * n_freq);
    if (loader_->has_tensor("diar.head.learnable_sil_emb")) {
        read_host_tensor(
            "diar.head.learnable_sil_emb", sil_emb_, static_cast<size_t>(cfg_.diar.d_model));
    } else {
        std::fprintf(
            stderr,
            "[pee] %s has no diar.head.learnable_sil_emb: speaker windows run without state "
            "(re-export with conversion/pee.py)\n",
            gguf_path.c_str());
    }

    backend_ = bm.gpu_backend_handle() ? bm.gpu_backend_handle() : bm.get_backends().front();
    if (ggml_backend_t gpu = bm.gpu_backend_handle()) {
        // Pinned host memory for uploads/downloads (DMA path, no staging).
        host_buft_ = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu));
        device_pool_ = std::make_shared<DevicePool>();
        device_pool_->buft = ggml_backend_get_default_buffer_type(gpu);
#if defined(NEMO_SPEECH_GGML_PATCHED) && defined(GGML_USE_CUDA)
        // The patched CUDA RoPE stores F16 directly (RopeTransformerConfig::kv_f16).
        if (ggml_backend_is_cuda(gpu)) {
            cfg_.asr.kv_f16 = cfg_.diar.kv_f16 = true;
        }
#endif
    }
    if (!ctc_path.empty()) {
        ctc_ = std::make_unique<CtcTimestampModel>(bm, ctc_path);
        if (ctc_->dimension() != cfg_.asr.d_model)
            throw std::runtime_error("CTC adapter feature dimension does not match PEE");
    }
    graph_ = std::make_unique<PeeGraph>(
        cfg_, want_asr_states_ || bool(ctc_), bool(ctc_), stateful_speakers());
    speaker_on_device_ = gpu_ != nullptr && stateful_speakers();
    session_ = std::make_unique<ggml_runtime::Session>(bm, graph_.get(), loader_.get());
    // (bucket length, batch) shapes recur; keep their graphs resident.
    session_->set_run_cache_capacity(64);
    // Runs are serialized (batch_mu_) and nothing reads a run's device outputs after the next run
    // starts (projection blocks and timestamp evidence are copied out before a run returns): every
    // shape shares one grow-only arena for its inputs and outputs instead of allocating (and on
    // eviction freeing, a device-wide sync) a buffer per shape.
    session_->set_shared_named_arena(true);
    session_->setup();
}

PeeModel::~PeeModel() = default;

PeeModel::MelStore::~MelStore() {
    if (buf_)
        ggml_backend_buffer_free(buf_);
    if (ctx_)
        ggml_free(ctx_);
}

std::shared_ptr<PeeModel::MelStore>
PeeModel::mel_store(size_t frames) {
    const size_t need = frames + zero_frames();
    std::unique_ptr<MelStore> store;
    {
        std::lock_guard<std::mutex> lock(store_mu_);
        auto fit = free_stores_.end();
        for (auto it = free_stores_.begin(); it != free_stores_.end(); ++it)
            if ((*it)->frames_ >= need &&
                (fit == free_stores_.end() || (*it)->frames_ < (*fit)->frames_))
                fit = it;
        if (fit != free_stores_.end()) {
            store = std::move(*fit);
            free_stores_.erase(fit);
        } else if (!free_stores_.empty()) {
            // too small for this job: replace the largest free one rather than keep both
            auto big = std::max_element(
                free_stores_.begin(), free_stores_.end(),
                [](const auto& a, const auto& b) { return a->frames_ < b->frames_; });
            free_stores_.erase(big);
        }
    }
    if (!store) {
        constexpr size_t kGrain = size_t(1) << 16;  // frames; grow with headroom, in coarse steps
        const size_t cap = (need + need / 4 + kGrain - 1) / kGrain * kGrain;
        store.reset(new MelStore());
        ggml_init_params ip = {ggml_tensor_overhead(), nullptr, true};
        store->ctx_ = ggml_init(ip);
        ggml_tensor* t =
            ggml_new_tensor_2d(store->ctx_, GGML_TYPE_F32, cfg_.n_mels, static_cast<int64_t>(cap));
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
        store->buf_ = ggml_backend_alloc_ctx_tensors_from_buft(store->ctx_, buft);
        if (!store->buf_)
            throw std::runtime_error("pee: mel store allocation failed (out of device memory?)");
        ggml_backend_buffer_clear(store->buf_, 0);  // the zero frames
        store->dev_ = {t, buft, 0};
        store->frames_ = cap;
        store->frame_bytes_ = static_cast<size_t>(cfg_.n_mels) * sizeof(float);
    }
    return std::shared_ptr<MelStore>(store.release(), [this](MelStore* m) {
        std::lock_guard<std::mutex> lock(store_mu_);
        free_stores_.emplace_back(m);
    });
}

void
PeeModel::upload_mel(MelStore& store, size_t frame, const float* mel, size_t frames) {
    if (frame < zero_frames() || frame + frames > store.frames_)
        throw std::runtime_error("pee: mel upload outside the store");
    ggml_backend_tensor_set(
        store.dev_.tensor, mel, store.offset(frame), frames * store.frame_bytes_);
}

std::vector<ggml_runtime::Session::Input::DeviceRange>
PeeModel::window_gather(
    const MelStore& store, const std::vector<size_t>& mel_frame, const std::vector<int>& lens,
    int t_pad, int Bg) const {
    const size_t fb = store.frame_bytes_;
    const size_t B = lens.size();
    if (mel_frame.size() != B || static_cast<size_t>(t_pad) > zero_frames())
        throw std::runtime_error("pee: bad window gather");
    std::vector<ggml_runtime::Session::Input::DeviceRange> ranges;
    ranges.reserve(2 * static_cast<size_t>(Bg));
    for (int b = 0; b < Bg; b++) {
        const size_t src = static_cast<size_t>(b) < B ? b : 0;
        const size_t len = static_cast<size_t>(lens[src]);
        if (mel_frame[src] < zero_frames() || mel_frame[src] + len > store.frames_)
            throw std::runtime_error("pee: window outside the mel store");
        const size_t dst = static_cast<size_t>(b) * t_pad * fb;
        ranges.push_back({&store.dev_, store.offset(mel_frame[src]), dst, len * fb});
        if (len < static_cast<size_t>(t_pad))
            ranges.push_back({&store.dev_, 0, dst + len * fb, (t_pad - len) * fb});
    }
    return ranges;
}

void
PeeModel::read_host_tensor(const std::string& name, std::vector<float>& out, size_t n) {
    if (!loader_->has_tensor(name))
        throw std::runtime_error("pee GGUF missing tensor " + name);
    if (loader_->get_tensor_type(name) != GGML_TYPE_F32)
        throw std::runtime_error("pee GGUF tensor " + name + " must be f32");
    out.resize(n);
    const char* data = loader_->get_tensor_file_data(name, n * sizeof(float));
    std::memcpy(out.data(), data, n * sizeof(float));
}

std::vector<PeeModel::Window>
PeeModel::plan_windows(int t_mel) const {
    // ParallelExpertEncoder._forward_online geometry, in mel frames.
    const int sub = cfg_.asr.subsampling_factor;
    const int chunk_feat = cfg_.online_inference_length * sub;
    const int lc = cfg_.chunk_left_context * sub;
    const int rc = cfg_.chunk_right_context * sub;
    std::vector<Window> plan;
    if (t_mel <= chunk_feat) {
        plan.push_back({0, t_mel, 0, out_frames(t_mel)});
        return plan;
    }
    const int n = (t_mel + chunk_feat - 1) / chunk_feat;
    for (int ci = 0; ci < n; ci++) {
        Window w;
        const int stt = ci * chunk_feat;
        const int end = std::min(stt + chunk_feat, t_mel);
        w.enc_stt = std::max(stt - lc, 0);
        w.enc_end = std::min(end + rc, t_mel);
        w.left_drop = (stt - w.enc_stt) / sub;
        w.right = w.enc_end - end;
        // _asr_output_frame_boundary: ceil for the transformer encoder
        int core = (end + sub - 1) / sub - (stt + sub - 1) / sub;
        core = std::max(0, std::min(core, out_frames(w.len()) - w.left_drop));
        w.core = core;
        plan.push_back(w);
    }
    return plan;
}

PeeModel::NormStats
PeeModel::norm_stats(const float* mel, int t_mel) const {
    // NeMo normalize_batch("per_feature"): per-bin mean, unbiased std, + 1e-5.
    const int n = cfg_.n_mels;
    std::vector<double> sum(n, 0.0), sq(n, 0.0);
    for (int i = 0; i < t_mel; i++) {  // frame-major single pass
        const float* f = mel + static_cast<size_t>(i) * n;
        for (int j = 0; j < n; j++) {
            sum[j] += f[j];
            sq[j] += static_cast<double>(f[j]) * f[j];
        }
    }
    NormStats st;
    st.mean.resize(n);
    st.inv_std.resize(n);
    for (int j = 0; j < n; j++) {
        const double mean = sum[j] / t_mel;
        double var = t_mel > 1 ? (sq[j] - t_mel * mean * mean) / (t_mel - 1) : 0.0;
        if (var < 0)
            var = 0;
        st.mean[j] = static_cast<float>(mean);
        st.inv_std[j] = 1.0f / (static_cast<float>(std::sqrt(var)) + 1e-5f);
    }
    return st;
}

// ------------------------------------------------------------ stateful speakers --

namespace {

// Indices of the k largest values of v (ties: lower index first).
std::vector<int>
topk(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    k = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return v[a] > v[b] || (v[a] == v[b] && a < b);
    });
    idx.resize(k);
    return idx;
}

// SortformerModules._compress_spkcache (inference: no speaker permutation): keep kCacheLen rows of
// the n-row cache, grouped by speaker (time order within a speaker), each speaker block ending with
// its silence row(s).
//
// Scores in the product domain: exp of NeMo's log score, 2 max(p_s, thr) prod_{j != s}
// max(1 - p_j, thr), with its additive boosts as factors (exp(0.05) for newly added frames, 4 and 2
// for the strong and weak top-k). The order is NeMo's up to rounding, and the f32 products in a
// fixed order are exact to reproduce: the device graph (PeeGraph::compress_cache) computes the same
// values, so both paths keep the same frames (device logarithms are approximate, so log-domain
// scores differed between them at near ties). 0 marks a disabled (-inf) frame, +inf a silence
// frame.
void
compress_cache(
    std::vector<float>& emb, std::vector<float>& preds, int n, int d, int n_spk,
    const std::vector<float>& sil_emb) {
    const int per_spk = kCacheLen / n_spk - kSilFramesPerSpk;
    const int strong = static_cast<int>(std::floor(per_spk * kStrongBoostRate));
    const int weak = static_cast<int>(std::floor(per_spk * kWeakBoostRate));
    const int min_pos = static_cast<int>(std::floor(per_spk * kMinPosScoresRate));
    const float disabled = 0.0f;  // -inf in the log domain
    const float latest = std::exp(kScoresBoostLatest);
    // scores[s][t] (speaker-major, as the flattened top-k sees them), plus kSilFramesPerSpk +inf
    // frames per speaker
    const int nt = n + kSilFramesPerSpk;
    std::vector<std::vector<float>> scores(
        n_spk, std::vector<float>(nt, std::numeric_limits<float>::infinity()));
    for (int t = 0; t < n; ++t) {
        const float* p = preds.data() + static_cast<size_t>(t) * n_spk;
        for (int s = 0; s < n_spk; ++s) {
            float prod = 1.0f;
            for (int j = 0; j < n_spk; ++j)
                prod *= j == s ? 1.0f : std::max(1.0f - p[j], kPredScoreThreshold);
            prod *= std::max(p[s], kPredScoreThreshold);
            scores[s][t] = prod * 2.0f;
        }
    }
    for (int s = 0; s < n_spk; ++s) {
        // non-speech -> disabled; with enough positive (> 1, log > 0) frames, the other speech
        // frames (overlap) too
        int n_pos = 0;
        for (int t = 0; t < n; ++t) {
            if (!(preds[static_cast<size_t>(t) * n_spk + s] > 0.5f))
                scores[s][t] = disabled;
            n_pos += scores[s][t] > 1.0f;
        }
        if (n_pos >= min_pos)
            for (int t = 0; t < n; ++t)
                if (!(scores[s][t] > 1.0f))
                    scores[s][t] = disabled;
        for (int t = kCacheLen; t < n; ++t) scores[s][t] *= latest;  // newly added frames
        std::vector<float> row(scores[s].begin(), scores[s].begin() + n);
        for (int t : topk(row, strong)) scores[s][t] *= 4.0f;  // at least `strong` per speaker
        row.assign(scores[s].begin(), scores[s].begin() + n);
        for (int t : topk(row, weak)) scores[s][t] *= 2.0f;  // no single speaker dominates
    }
    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(n_spk) * nt);
    for (int s = 0; s < n_spk; ++s) flat.insert(flat.end(), scores[s].begin(), scores[s].end());
    std::vector<int> pick = topk(flat, kCacheLen);
    const int placeholder = static_cast<int>(flat.size());
    for (int& i : pick)
        if (flat[i] == disabled)
            i = placeholder;
    std::sort(pick.begin(), pick.end());
    std::vector<float> out_emb(static_cast<size_t>(kCacheLen) * d),
        out_preds(static_cast<size_t>(kCacheLen) * n_spk, 0.f);
    for (int i = 0; i < kCacheLen; ++i) {
        const int t = pick[i] == placeholder ? -1 : pick[i] % nt;
        float* e = out_emb.data() + static_cast<size_t>(i) * d;
        if (t < 0 || t >= n) {
            std::copy(sil_emb.begin(), sil_emb.end(), e);
        } else {
            std::copy_n(emb.data() + static_cast<size_t>(t) * d, d, e);
            std::copy_n(
                preds.data() + static_cast<size_t>(t) * n_spk, n_spk,
                out_preds.data() + static_cast<size_t>(i) * n_spk);
        }
    }
    emb.swap(out_emb);
    preds.swap(out_preds);
}

}  // namespace

void
PeeModel::speaker_steps(const MelStore& store, const std::vector<SpeakerUtt*>& utts) {
    for (SpeakerUtt* u : utts) {
        if (!u->plan || u->plan->empty())
            throw std::runtime_error("pee: speaker steps without windows");
        const size_t n = u->plan->size();
        u->p80.assign(n, {});
        u->p10.assign(n, {});
        u->p80_host.clear();
    }
    if (utts.empty())
        return;
    if (speaker_on_device_)
        speaker_steps_device(store, utts);
    else
        speaker_steps_host(store, utts);
}

namespace {
// One utterance's streaming state on the host: pre-encoded rows (d_diar each) of the speaker
// cache and FIFO, with their 80 ms speaker probabilities.
struct HostSpeakerState {
    std::vector<float> cache, cache_preds, fifo, fifo_preds;
    bool compressed = false;
};
}  // namespace

void
PeeModel::speaker_steps_host(const MelStore& store, const std::vector<SpeakerUtt*>& utts) {
    const int n_mels = cfg_.n_mels;
    const int d = cfg_.diar.d_model;
    const int n_spk = cfg_.num_speakers;
    const int sub = cfg_.diar.subsampling_factor;
    std::vector<HostSpeakerState> states(utts.size());
    size_t max_windows = 0;
    for (SpeakerUtt* u : utts) {
        u->p80_host.assign(u->plan->size(), {});
        max_windows = std::max(max_windows, u->plan->size());
    }
    for (size_t w = 0; w < max_windows; ++w) {
        // batches: same prefix length (cache + FIFO rows) and length bucket
        std::map<std::pair<int, int>, std::vector<size_t>> groups;
        for (size_t u = 0; u < utts.size(); ++u) {
            if (w >= utts[u]->plan->size())
                continue;
            const int prev = static_cast<int>((states[u].cache.size() + states[u].fifo.size()) / d);
            groups[{prev, bucket_len((*utts[u]->plan)[w].len())}].push_back(u);
        }
        for (auto& [key, all] : groups) {
            const int P = key.first, t_pad = key.second, T = out_frames(t_pad), L = P + T;
            for (size_t a = 0; a < all.size(); a += static_cast<size_t>(max_batch_)) {
                const std::vector<size_t> group(
                    all.begin() + a, all.begin() + std::min(all.size(), a + max_batch_));
                const int B = static_cast<int>(group.size());
                std::vector<float> prev(static_cast<size_t>(P) * d * B);
                std::vector<size_t> mel_frame(B);
                std::vector<int> lens(B);
                std::vector<int32_t> pos(L);
                std::vector<float> kmask(static_cast<size_t>(L) * B, 0.f);
                bool padded = false;
                for (int b = 0; b < B; ++b) {
                    const Window& win = (*utts[group[b]]->plan)[w];
                    const HostSpeakerState& ss = states[group[b]];
                    mel_frame[b] = utts[group[b]]->mel_frame + static_cast<size_t>(win.enc_stt);
                    lens[b] = win.len();
                    float* pv = prev.data() + static_cast<size_t>(b) * P * d;
                    pv = std::copy(ss.cache.begin(), ss.cache.end(), pv);
                    std::copy(ss.fifo.begin(), ss.fifo.end(), pv);
                    const int valid = P + out_frames(lens[b]);
                    std::fill(
                        kmask.begin() + static_cast<size_t>(b) * L + valid,
                        kmask.begin() + static_cast<size_t>(b + 1) * L, -1e9f);
                    padded = padded || valid < L;
                }
                for (int i = 0; i < L; ++i) pos[i] = i;
                std::vector<ggml_runtime::Session::Input> inputs;
                ggml_runtime::Session::Input mel{
                    "input.spk_mel", GGML_TYPE_F32, nullptr, {n_mels, t_pad, B}};
                mel.gather = window_gather(store, mel_frame, lens, t_pad, B);
                inputs.push_back(std::move(mel));
                if (P > 0)
                    inputs.push_back({"input.spk_prev", GGML_TYPE_F32, prev.data(), {d, P, B}});
                inputs.push_back({"input.spk_pos", GGML_TYPE_I32, pos.data(), {L}});
                if (padded)
                    inputs.push_back(
                        {"input.spk_kmask", GGML_TYPE_F32, kmask.data(), {L, 1, 1, B}});
                std::vector<float> chunk(static_cast<size_t>(d) * T * B),
                    p80(static_cast<size_t>(n_spk) * L * B),
                    p10(static_cast<size_t>(n_spk) * L * sub * B);
                std::vector<ggml_runtime::Session::Output> outputs(3);
                outputs[0].index = 0;
                outputs[0].host_buffer = chunk.data();
                outputs[0].nbytes = chunk.size() * sizeof(float);
                outputs[1].index = 1;
                outputs[1].host_buffer = p80.data();
                outputs[1].nbytes = p80.size() * sizeof(float);
                outputs[2].index = 2;
                outputs[2].host_buffer = p10.data();
                outputs[2].nbytes = p10.size() * sizeof(float);
                {
                    std::lock_guard<std::mutex> batch_lock(batch_mu_);
                    ggml_nvtx::range nvtx_run("pee.speaker");
                    session_->run(inputs, outputs);
                }
                for (int b = 0; b < B; ++b) {
                    SpeakerUtt& utt = *utts[group[b]];
                    const Window& win = (*utt.plan)[w];
                    HostSpeakerState& ss = states[group[b]];
                    const int S = static_cast<int>(ss.cache.size() / d),
                              F = static_cast<int>(ss.fifo.size() / d);
                    const int Tb = out_frames(lens[b]);
                    // the speaker core: chunk rows minus left (round) and right (ceil) context,
                    // as forward_streaming_step
                    const int lc = win.left_drop, rc = (win.right + sub - 1) / sub,
                              c = std::max(0, Tb - lc - rc);
                    const float* P80 = p80.data() + static_cast<size_t>(b) * L * n_spk;
                    const float* P10 = p10.data() + static_cast<size_t>(b) * L * sub * n_spk;
                    const float* emb = chunk.data() + static_cast<size_t>(b) * T * d;
                    // outputs, aligned to the ASR core (_align_diar_frames: repeat the last frame
                    // / truncate)
                    std::vector<float>& out80 = utt.p80_host[w];
                    out80.assign(static_cast<size_t>(Tb) * n_spk, 0.f);
                    if (utt.want_p10)
                        utt.p10[w].assign(static_cast<size_t>(win.core) * sub * n_spk, 0.f);
                    for (int t = 0; t < win.core && c > 0; ++t) {
                        const int src = S + F + lc + std::min(t, c - 1);
                        std::copy_n(
                            P80 + static_cast<size_t>(src) * n_spk, n_spk,
                            out80.data() + static_cast<size_t>(lc + t) * n_spk);
                        if (utt.want_p10)
                            std::copy_n(
                                P10 + static_cast<size_t>(src) * sub * n_spk, sub * n_spk,
                                utt.p10[w].data() + static_cast<size_t>(t) * sub * n_spk);
                    }
                    if (w + 1 == utt.plan->size())
                        continue;  // the state after the last window is not used
                    // SortformerModules.streaming_update (sync)
                    ss.fifo_preds.assign(
                        P80 + static_cast<size_t>(S) * n_spk,
                        P80 + static_cast<size_t>(S + F) * n_spk);
                    ss.fifo.insert(
                        ss.fifo.end(), emb + static_cast<size_t>(lc) * d,
                        emb + static_cast<size_t>(lc + c) * d);
                    ss.fifo_preds.insert(
                        ss.fifo_preds.end(), P80 + static_cast<size_t>(S + F + lc) * n_spk,
                        P80 + static_cast<size_t>(S + F + lc + c) * n_spk);
                    if (F + c > kFifoLen) {
                        const int pop = std::min(std::max(kUpdatePeriod, c - kFifoLen + F), F + c);
                        if (!ss.compressed)
                            ss.cache_preds.assign(P80, P80 + static_cast<size_t>(S) * n_spk);
                        ss.cache.insert(
                            ss.cache.end(), ss.fifo.begin(),
                            ss.fifo.begin() + static_cast<size_t>(pop) * d);
                        ss.cache_preds.insert(
                            ss.cache_preds.end(), ss.fifo_preds.begin(),
                            ss.fifo_preds.begin() + static_cast<size_t>(pop) * n_spk);
                        ss.fifo.erase(
                            ss.fifo.begin(), ss.fifo.begin() + static_cast<size_t>(pop) * d);
                        ss.fifo_preds.erase(
                            ss.fifo_preds.begin(),
                            ss.fifo_preds.begin() + static_cast<size_t>(pop) * n_spk);
                        const int n = static_cast<int>(ss.cache.size() / d);
                        if (n > kCacheLen) {
                            compress_cache(ss.cache, ss.cache_preds, n, d, n_spk, sil_emb_);
                            ss.compressed = true;
                        }
                    }
                }
            }
        }
    }
    for (SpeakerUtt* u : utts)
        for (size_t w = 0; w < u->p80.size(); ++w) u->p80[w].host = u->p80_host[w].data();
}

namespace {
// Byte copy between device tensors on `backend`, queued on its stream.
void
copy_on_device(
    ggml_backend_t backend, const ggml_tensor* src, size_t src_offset, const ggml_tensor* dst,
    size_t dst_offset, size_t bytes) {
    if (bytes == 0)
        return;
    if (src_offset + bytes > ggml_nbytes(src) || dst_offset + bytes > ggml_nbytes(dst))
        throw std::runtime_error("pee: device copy outside its tensors");
    ggml_tensor from = {}, to = {};
    from.type = to.type = GGML_TYPE_I8;
    from.ne[0] = to.ne[0] = static_cast<int64_t>(bytes);
    from.ne[1] = from.ne[2] = from.ne[3] = to.ne[1] = to.ne[2] = to.ne[3] = 1;
    from.nb[0] = to.nb[0] = 1;
    from.nb[1] = from.nb[2] = from.nb[3] = to.nb[1] = to.nb[2] = to.nb[3] = bytes;
    from.buffer = src->buffer;
    from.data = static_cast<char*>(src->data) + src_offset;
    to.buffer = dst->buffer;
    to.data = static_cast<char*>(dst->data) + dst_offset;
    ggml_backend_tensor_copy_async(backend, backend, &from, &to);
}
}  // namespace

void
PeeModel::speaker_steps_device(const MelStore& store, const std::vector<SpeakerUtt*>& utts) {
    using Range = ggml_runtime::Session::Input::DeviceRange;
    const int n_mels = cfg_.n_mels;
    const int d = cfg_.diar.d_model;
    const int n_spk = cfg_.num_speakers;
    const int sub = cfg_.diar.subsampling_factor;
    const int f10 = n_spk * sub;  // floats of 10 ms probabilities per 80 ms frame

    // The state geometry of every step is known from the window plans (the cache and FIFO row
    // counts do not depend on the data), so the row lists of all steps are built here and uploaded
    // once; the steps then run without the host.
    struct Step {
        size_t u = 0, w = 0;
        int len = 0, t_pad = 0, T = 0, Tb = 0;  // mel frames, padded; output frames, padded / own
        int lc = 0, c = 0, core = 0;            // speaker left context and core rows; the ASR core
        int S = 0, F = 0;                       // cache and FIFO rows before the step
        bool compressed = false;                // the cache probabilities are the state's
        bool update = false;                    // not the utterance's last window
        int pop = 0, n = 0, F2 = 0, S2 = 0;     // popped rows; cache rows before compression; after
        size_t p80_at = 0, p10_at = 0;          // output slots (floats past the outputs' start)
        // row of x = [cache | FIFO | window] (and of the 80 ms probs) of FIFO+core queue row q
        int queue_row(int q) const { return q < F ? S + q : S + F + lc + (q - F); }
    };
    struct UttMem {
        size_t prev_at = 0, cpred_at = 0;  // [cache | FIFO] rows, cache probabilities (floats)
        int prev_rows = 0, cache_rows = 0;
    };
    std::vector<Step> steps;
    std::vector<UttMem> mem(utts.size());
    size_t out_floats = 0;
    for (size_t u = 0; u < utts.size(); ++u) {
        const std::vector<Window>& plan = *utts[u]->plan;
        int S = 0, F = 0;
        bool compressed = false;
        for (size_t w = 0; w < plan.size(); ++w) {
            const Window& win = plan[w];
            Step s;
            s.u = u;
            s.w = w;
            s.len = win.len();
            s.t_pad = bucket_len(s.len);
            s.T = out_frames(s.t_pad);
            s.Tb = out_frames(s.len);
            s.lc = win.left_drop;
            s.core = win.core;
            s.c = std::max(0, s.Tb - s.lc - (win.right + sub - 1) / sub);
            s.S = S;
            s.F = F;
            s.compressed = compressed;
            s.update = w + 1 < plan.size();
            if (s.update) {
                s.pop = F + s.c > kFifoLen
                            ? std::min(std::max(kUpdatePeriod, s.c - kFifoLen + F), F + s.c)
                            : 0;
                s.n = S + s.pop;
                s.F2 = F + s.c - s.pop;
                s.S2 = std::min(s.n, kCacheLen);
                compressed = compressed || s.n > kCacheLen;
                S = s.S2;
                F = s.F2;
                mem[u].prev_rows = std::max(mem[u].prev_rows, S + F);
                mem[u].cache_rows = std::max(mem[u].cache_rows, S);
            }
            s.p80_at = out_floats;
            out_floats += static_cast<size_t>(s.Tb) * n_spk;
            if (utts[u]->want_p10) {
                s.p10_at = out_floats;
                out_floats += static_cast<size_t>(s.core) * f10;
            }
            steps.push_back(s);
        }
    }

    // Graph runs: by window index, then shape; up to max_batch_ steps, padded to the batch ladder
    // with copies of the first (their outputs are not kept).
    struct Run {
        std::vector<size_t> items;  // steps
        int Bg = 0;
        size_t out_rows = 0, core_rows = 0, cache_rows = 0, cpred_rows = 0, fifo_rows = 0;
    };
    std::map<std::tuple<size_t, int, int, bool, int, int, int>, std::vector<size_t>> shapes;
    for (size_t i = 0; i < steps.size(); ++i) {
        const Step& s = steps[i];
        shapes[{s.w, s.S + s.F, s.t_pad, s.update, s.update ? s.S : 0, s.n, s.F2}].push_back(i);
    }
    std::vector<int32_t> lists;  // every run's row lists, then the constants (as 32-bit words)
    std::vector<Run> runs;
    int L_max = 0;
    for (const auto& [key, all] : shapes) {
        for (size_t a = 0; a < all.size(); a += static_cast<size_t>(max_batch_)) {
            Run r;
            r.items.assign(all.begin() + a, all.begin() + std::min(all.size(), a + max_batch_));
            r.Bg = batch_pad(static_cast<int>(r.items.size()));
            const Step& s0 = steps[r.items[0]];
            const int P = s0.S + s0.F, T = s0.T, L = P + T;
            L_max = std::max(L_max, L);
            auto item = [&](int b) -> const Step& {
                return steps[r.items[static_cast<size_t>(b) < r.items.size() ? b : 0]];
            };
            r.out_rows = lists.size();  // window frame t <- core row t - lc, repeating the last
            for (int b = 0; b < r.Bg; ++b) {
                const Step& s = item(b);
                for (int t = 0; t < T; ++t)
                    lists.push_back(
                        t >= s.lc && t < s.lc + s.core && s.c > 0
                            ? P + s.lc + std::min(t - s.lc, s.c - 1)
                            : L);
            }
            r.core_rows = lists.size();
            for (int b = 0; b < r.Bg; ++b) {
                const Step& s = item(b);
                for (int t = 0; t < T; ++t)
                    lists.push_back(t < s.core && s.c > 0 ? P + s.lc + std::min(t, s.c - 1) : L);
            }
            if (!s0.update) {
                runs.push_back(std::move(r));
                continue;
            }
            r.cache_rows = lists.size();
            for (int b = 0; b < r.Bg; ++b) {
                const Step& s = item(b);
                for (int i = 0; i < s.n; ++i) lists.push_back(i < s.S ? i : s.queue_row(i - s.S));
            }
            r.cpred_rows = lists.size();  // rows of [state cache probs (S) | 80 ms probs (L)]
            for (int b = 0; b < r.Bg; ++b) {
                const Step& s = item(b);
                for (int i = 0; i < s.n; ++i)
                    lists.push_back(
                        i < s.S ? (s.compressed ? i : s.S + i) : s.S + s.queue_row(i - s.S));
            }
            r.fifo_rows = lists.size();
            for (int b = 0; b < r.Bg; ++b) {
                const Step& s = item(b);
                for (int q = s.pop; q < s.F + s.c; ++q) lists.push_back(s.queue_row(q));
            }
            runs.push_back(std::move(r));
        }
    }
    const size_t pos_at = lists.size();  // 0 .. L_max-1
    for (int i = 0; i < L_max; ++i) lists.push_back(i);
    const size_t zero_at = lists.size(), masked_at = zero_at + L_max;  // key mask values
    lists.resize(masked_at + L_max, 0);
    const float masked = -1e9f;
    for (int i = 0; i < L_max; ++i) std::memcpy(&lists[masked_at + i], &masked, sizeof(float));

    // One device block: the lists, each utterance's state, every step's outputs.
    size_t at = lists.size();
    for (UttMem& m : mem) {
        m.prev_at = at;
        at += static_cast<size_t>(m.prev_rows) * d;
        m.cpred_at = at;
        at += static_cast<size_t>(m.cache_rows) * n_spk;
    }
    const size_t out_at = at;
    at += out_floats;
    std::shared_ptr<DeviceBlock> block = device_block(at);
    ggml_tensor* blk = block->tensor();
    const ggml_runtime::DeviceTensor dev{blk, device_pool_->buft, 0};
    ggml_backend_tensor_set(blk, lists.data(), 0, lists.size() * sizeof(int32_t));
    auto words = [](size_t n) { return n * sizeof(float); };  // 32-bit words -> bytes

    {
        std::lock_guard<std::mutex> batch_lock(batch_mu_);
        for (const Run& r : runs) {
            ggml_nvtx::range nvtx_run("pee.speaker");
            const Step& s0 = steps[r.items[0]];
            const int B = static_cast<int>(r.items.size()), Bg = r.Bg;
            const int P = s0.S + s0.F, T = s0.T, L = P + T, t_pad = s0.t_pad;
            auto item = [&](int b) -> const Step& { return steps[r.items[b < B ? b : 0]]; };
            std::vector<size_t> mel_frame(B);
            std::vector<int> lens(B);
            bool padded = false;
            for (int b = 0; b < B; ++b) {
                const Step& s = item(b);
                mel_frame[b] =
                    utts[s.u]->mel_frame + static_cast<size_t>((*utts[s.u]->plan)[s.w].enc_stt);
                lens[b] = s.len;
                padded = padded || P + s.Tb < L;
            }
            std::vector<ggml_runtime::Session::Input> inputs;
            auto input = [&](const char* name, ggml_type type, std::vector<int64_t> shape,
                             std::vector<Range> ranges) {
                ggml_runtime::Session::Input in{name, type, nullptr, std::move(shape)};
                in.upload = false;
                in.gather = std::move(ranges);
                inputs.push_back(std::move(in));
            };
            auto list = [&](size_t list_at, size_t rows) {
                return std::vector<Range>{{&dev, words(list_at), 0, words(rows * Bg)}};
            };
            input(
                "input.spk_mel", GGML_TYPE_F32, {n_mels, t_pad, Bg},
                window_gather(store, mel_frame, lens, t_pad, Bg));
            if (P > 0) {
                std::vector<Range> prev;
                for (int b = 0; b < Bg; ++b)
                    prev.push_back(
                        {&dev, words(mem[item(b).u].prev_at), words(static_cast<size_t>(b) * P * d),
                         words(static_cast<size_t>(P) * d)});
                input("input.spk_prev", GGML_TYPE_F32, {d, P, Bg}, std::move(prev));
            }
            input("input.spk_pos", GGML_TYPE_I32, {L}, {{&dev, words(pos_at), 0, words(L)}});
            if (padded) {
                std::vector<Range> km;
                for (int b = 0; b < Bg; ++b) {
                    const size_t valid = static_cast<size_t>(P + item(b).Tb);
                    km.push_back(
                        {&dev, words(zero_at), words(static_cast<size_t>(b) * L), words(valid)});
                    if (valid < static_cast<size_t>(L))
                        km.push_back(
                            {&dev, words(masked_at), words(static_cast<size_t>(b) * L + valid),
                             words(L - valid)});
                }
                input("input.spk_kmask", GGML_TYPE_F32, {L, 1, 1, Bg}, std::move(km));
            }
            input("input.spk_out_rows", GGML_TYPE_I32, {T, Bg}, list(r.out_rows, T));
            input("input.spk_core_rows", GGML_TYPE_I32, {T, Bg}, list(r.core_rows, T));
            if (s0.update) {
                input("input.spk_cache_rows", GGML_TYPE_I32, {s0.n, Bg}, list(r.cache_rows, s0.n));
                input("input.spk_cpred_rows", GGML_TYPE_I32, {s0.n, Bg}, list(r.cpred_rows, s0.n));
                if (s0.F2 > 0)
                    input(
                        "input.spk_fifo_rows", GGML_TYPE_I32, {s0.F2, Bg},
                        list(r.fifo_rows, s0.F2));
                if (s0.S > 0) {
                    std::vector<Range> cpred;
                    for (int b = 0; b < Bg; ++b)
                        cpred.push_back(
                            {&dev, words(mem[item(b).u].cpred_at),
                             words(static_cast<size_t>(b) * s0.S * n_spk),
                             words(static_cast<size_t>(s0.S) * n_spk)});
                    input("input.spk_cpred", GGML_TYPE_F32, {n_spk, s0.S, Bg}, std::move(cpred));
                }
            }
            // outputs stay on the device (no synchronization); copied out before the next run
            std::vector<ggml_runtime::DeviceTensor> out(s0.update ? 4 : 2);
            std::vector<ggml_runtime::Session::Output> outputs(out.size());
            for (size_t i = 0; i < out.size(); ++i) {
                outputs[i].index = static_cast<int>(3 + i);
                outputs[i].device_tensor = &out[i];
            }
            session_->run(inputs, outputs);
            const int rows_next = s0.S2 + s0.F2;
            auto expect = [&](size_t i, int64_t ne0, int64_t ne1) {
                if (outputs[i].out_shape[0] != ne0 || outputs[i].out_shape[1] != ne1 ||
                    outputs[i].out_shape[2] != Bg || !ggml_is_contiguous(out[i].tensor))
                    throw std::runtime_error("pee: unexpected speaker step output");
            };
            expect(0, n_spk, T);
            expect(1, f10, T);
            if (s0.update) {
                expect(2, d, rows_next);
                expect(3, n_spk, s0.S2);
            }
            std::lock_guard<std::mutex> lock(bm_->compute_mutex());
            for (int b = 0; b < B; ++b) {
                const Step& s = item(b);
                const UttMem& m = mem[s.u];
                copy_on_device(
                    gpu_, out[0].tensor, words(static_cast<size_t>(b) * n_spk * T), blk,
                    words(out_at + s.p80_at), words(static_cast<size_t>(s.Tb) * n_spk));
                if (utts[s.u]->want_p10)
                    copy_on_device(
                        gpu_, out[1].tensor, words(static_cast<size_t>(b) * f10 * T), blk,
                        words(out_at + s.p10_at), words(static_cast<size_t>(s.core) * f10));
                if (!s.update)
                    continue;
                copy_on_device(
                    gpu_, out[2].tensor, words(static_cast<size_t>(b) * d * rows_next), blk,
                    words(m.prev_at), words(static_cast<size_t>(d) * rows_next));
                copy_on_device(
                    gpu_, out[3].tensor, words(static_cast<size_t>(b) * n_spk * s0.S2), blk,
                    words(m.cpred_at), words(static_cast<size_t>(n_spk) * s0.S2));
            }
        }
    }

    bool want_p10 = false;
    for (const Step& s : steps) {
        utts[s.u]->p80[s.w] = SpeakerProbs{nullptr, block, words(out_at + s.p80_at)};
        want_p10 = want_p10 || utts[s.u]->want_p10;
    }
    if (!want_p10)
        return;  // the steps run on; encode_batch reads the p80 blocks after them
    std::vector<float> outs(out_floats);
    {
        std::lock_guard<std::mutex> lock(bm_->compute_mutex());
        ggml_backend_tensor_get_async(gpu_, blk, outs.data(), words(out_at), words(out_floats));
        ggml_backend_synchronize(gpu_);
    }
    for (const Step& s : steps) {
        if (!utts[s.u]->want_p10)
            continue;
        const float* p = outs.data() + s.p10_at;
        utts[s.u]->p10[s.w].assign(p, p + static_cast<size_t>(s.core) * f10);
    }
}

const float*
PeeModel::WindowOut::fused() const {
    return batch->fused.data() + static_cast<size_t>(index) * batch->T * batch->d;
}
const float*
PeeModel::WindowOut::proj() const {
    return batch->proj.data() + static_cast<size_t>(index) * batch->T * batch->dp;
}
const float*
PeeModel::WindowOut::probs80() const {
    return batch->probs80.data() + static_cast<size_t>(index) * batch->T * batch->n_spk;
}
const float*
PeeModel::WindowOut::asr() const {
    return batch->asr.data() ? batch->asr.data() + static_cast<size_t>(index) * batch->T * batch->d
                             : nullptr;
}

int
PeeModel::batch_pad(int n) const {
    // powers of two and 1.5x powers of two: 1, 2, 3, 4, 6, 8, 12, 16, ...
    int b = 1;
    while (b < n) b = b == 1 ? 2 : (b & (b - 1)) == 0 ? b + b / 2 : b / 3 * 4;
    return std::min(b, max_batch_);
}

std::vector<std::vector<size_t>>
PeeModel::plan_batches(const std::vector<int>& lens) const {
    const size_t n = lens.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    std::stable_sort(
        order.begin(), order.end(), [&](size_t a, size_t b) { return lens[a] > lens[b]; });
    // best[i]: cost of the first i windows (longest first); a group [j, i) runs at the bucket of
    // its longest window, order[j]
    std::vector<int64_t> best(n + 1, std::numeric_limits<int64_t>::max());
    std::vector<size_t> cut(n + 1, 0);
    best[0] = 0;
    for (size_t i = 1; i <= n; i++) {
        const size_t lo = i > static_cast<size_t>(max_batch_) ? i - max_batch_ : 0;
        for (size_t j = lo; j < i; j++) {
            const int64_t c =
                best[j] + kRunCostFrames +
                int64_t(batch_pad(static_cast<int>(i - j))) * bucket_len(lens[order[j]]);
            if (c < best[i]) {
                best[i] = c;
                cut[i] = j;
            }
        }
    }
    std::vector<std::vector<size_t>> groups;
    for (size_t i = n; i > 0; i = cut[i])
        groups.emplace_back(order.begin() + cut[i], order.begin() + i);
    std::reverse(groups.begin(), groups.end());
    return groups;
}

std::vector<PeeModel::WindowOut>
PeeModel::encode_batch(
    const MelStore& store, const std::vector<size_t>& mel_frame,
    const std::vector<const NormStats*>& stats, const std::vector<int>& lens, int t_pad,
    bool evidence, const std::vector<SpeakerProbs>* p80_in, const std::vector<EvidenceSink>* sinks,
    int b_pad) {
    // The input arena and session are shared: concurrent encoder jobs take turns here (their
    // host-side feature extraction and window fusion still overlap).
    std::lock_guard<std::mutex> batch_lock(batch_mu_);
    ggml_nvtx::range nvtx_r("pee.batch");
    const int B = static_cast<int>(lens.size());
    if (B < 1 || stats.size() != lens.size() || mel_frame.size() != lens.size())
        throw std::runtime_error("pee: bad batch");
    // graph batch: padding items repeat window 0 (real values, outputs never downloaded)
    const int Bg = std::max(B, b_pad);
    const int n_mels = cfg_.n_mels;
    const int d = cfg_.asr.d_model;
    const int n_spk = cfg_.num_speakers;
    const int sub = cfg_.asr.subsampling_factor;
    const int T = out_frames(t_pad);
    bool padded = false;
    for (int l : lens) {
        if (l < 1 || l > t_pad)
            throw std::runtime_error("pee: window length exceeds pad length");
        padded = padded || l != t_pad;
    }

    // The mel windows are copied from the store on the device; one pinned arena sliced into the
    // other inputs.
    const size_t n_stat = static_cast<size_t>(n_mels) * Bg,
                 n_valid = static_cast<size_t>(T) * sub * Bg, n_km = static_cast<size_t>(T) * Bg,
                 n_pos = static_cast<size_t>(T);
    in_arena_.reserve(host_buft_, (2 * n_stat + n_valid + n_km + n_pos) * sizeof(float));
    float* in_mean = in_arena_.data();
    float* in_inv = in_mean + n_stat;
    float* in_valid = in_inv + n_stat;
    float* in_kmask = in_valid + n_valid;
    int32_t* in_pos = reinterpret_cast<int32_t*>(in_kmask + n_km);
    for (int b = 0; b < Bg; b++) {
        const int src = b < B ? b : 0;
        std::memcpy(in_mean + b * n_mels, stats[src]->mean.data(), n_mels * sizeof(float));
        std::memcpy(in_inv + b * n_mels, stats[src]->inv_std.data(), n_mels * sizeof(float));
        float* v = in_valid + static_cast<size_t>(b) * T * sub;
        std::fill(v, v + lens[src], 1.0f);
        std::fill(v + lens[src], v + static_cast<size_t>(T) * sub, 0.0f);
        const int t_valid = out_frames(lens[src]);
        float* km = in_kmask + static_cast<size_t>(b) * T;
        std::fill(km, km + t_valid, 0.0f);
        std::fill(km + t_valid, km + T, -1e9f);
    }
    for (int i = 0; i < T; i++) in_pos[i] = i;
    // stateful windows: fuse the speaker steps' probabilities (rows past each window's frames stay
    // zero), from the host or device to device (zeros from the store's leading zero frames)
    std::vector<float> in_p80;
    std::vector<ggml_runtime::Session::Input::DeviceRange> p80_gather;
    std::vector<ggml_runtime::DeviceTensor> p80_dev;
    if (p80_in) {
        if (p80_in->size() != lens.size())
            throw std::runtime_error("pee: p80 per window required");
        const bool on_device = (*p80_in)[0].block != nullptr;
        const size_t row = static_cast<size_t>(n_spk) * sizeof(float);
        if (on_device) {
            p80_dev.reserve(B);
            for (const SpeakerProbs& sp : *p80_in) {
                if (!sp.block)
                    throw std::runtime_error("pee: p80 windows mix host and device memory");
                p80_dev.push_back({sp.block->tensor(), device_pool_->buft, sp.offset});
            }
        } else {
            in_p80.assign(static_cast<size_t>(n_spk) * T * Bg, 0.0f);
        }
        for (int b = 0; b < Bg; b++) {
            const int src = b < B ? b : 0;
            const size_t rows = static_cast<size_t>(out_frames(lens[src]));
            if (!on_device) {
                if (!(*p80_in)[src].host)
                    throw std::runtime_error("pee: p80 windows mix host and device memory");
                std::copy_n(
                    (*p80_in)[src].host, rows * n_spk,
                    in_p80.data() + static_cast<size_t>(b) * T * n_spk);
                continue;
            }
            const size_t dst = static_cast<size_t>(b) * T * row;
            p80_gather.push_back({&p80_dev[src], 0, dst, rows * row});
            if (rows < static_cast<size_t>(T))
                p80_gather.push_back({&store.dev_, 0, dst + rows * row, (T - rows) * row});
        }
    }

    std::vector<ggml_runtime::Session::Input> inputs;
    ggml_runtime::Session::Input mel_in{"input.mel", GGML_TYPE_F32, nullptr, {n_mels, t_pad, Bg}};
    mel_in.gather = window_gather(store, mel_frame, lens, t_pad, Bg);
    inputs.push_back(std::move(mel_in));
    inputs.push_back({"input.mean", GGML_TYPE_F32, in_mean, {n_mels, 1, Bg}});
    inputs.push_back({"input.inv_std", GGML_TYPE_F32, in_inv, {n_mels, 1, Bg}});
    inputs.push_back({"input.valid", GGML_TYPE_F32, in_valid, {T * sub, 1, Bg}});
    inputs.push_back({"input.pos", GGML_TYPE_I32, in_pos, {T}});
    if (padded)
        inputs.push_back({"input.kmask", GGML_TYPE_F32, in_kmask, {T, 1, 1, Bg}});
    if (p80_in) {
        ggml_runtime::Session::Input p80{
            "input.p80",
            GGML_TYPE_F32,
            p80_gather.empty() ? in_p80.data() : nullptr,
            {n_spk, T, Bg}};
        p80.gather = std::move(p80_gather);
        inputs.push_back(std::move(p80));
    }

    auto bo = std::make_shared<BatchOut>();
    bo->T = T;
    bo->B = B;
    bo->d = d;
    bo->n_spk = n_spk;
    bo->dp = static_cast<size_t>(cfg_.proj_dim);
    const size_t d_item = static_cast<size_t>(T) * d;
    const size_t p_item = static_cast<size_t>(T) * n_spk;
    if (emit_fused_)
        bo->fused.reserve(host_buft_, d_item * B * sizeof(float));
    bo->probs80.reserve(host_buft_, p_item * B * sizeof(float));
    const bool has_proj = cfg_.proj_dim > 0;
    const bool emit_asr = want_asr_states_ || bool(ctc_);  // graph outputs (fixed per model)
    evidence = evidence && ctc_;
    if (sinks && (!ctc_ || sinks->size() != lens.size()))
        throw std::runtime_error("pee: one evidence sink per window, with a CTC adapter");
    ggml_runtime::DeviceTensor raw_asr, raw_probs10;  // on the device only
    const bool native =
        ctc_ && !p80_in;  // the graph's 10 ms probabilities (not with external speaker probs)
    std::vector<ggml_runtime::Session::Output> outputs(
        2 + (emit_asr ? 1 : 0) + (has_proj ? 1 : 0) + (native ? 1 : 0));
    outputs[0].index = 0;
    ggml_runtime::DeviceTensor fused_dev;  // !emit_fused_: shape only, no download
    if (emit_fused_) {
        outputs[0].host_buffer = bo->fused.data();
        outputs[0].nbytes = bo->fused.bytes();
    } else {
        outputs[0].device_tensor = &fused_dev;
    }
    outputs[1].index = 1;
    outputs[1].host_buffer = bo->probs80.data();
    outputs[1].nbytes = bo->probs80.bytes();
    if (emit_asr) {
        outputs[2].index = 2;
        outputs[2].device_tensor = &raw_asr;
    }
    if (want_asr_states_) {
        bo->asr.reserve(host_buft_, d_item * B * sizeof(float));
        outputs[2].host_buffer = bo->asr.data();
        outputs[2].nbytes = bo->asr.bytes();
    }
    const int proj_slot = emit_asr ? 3 : 2;
    ggml_runtime::DeviceTensor proj_dev;  // proj_on_device_: copied into a block after the run
    if (has_proj) {
        outputs[proj_slot].index = proj_slot;
        if (proj_on_device_) {
            outputs[proj_slot].device_tensor = &proj_dev;
        } else {
            bo->proj.reserve(host_buft_, static_cast<size_t>(T) * bo->dp * B * sizeof(float));
            outputs[proj_slot].host_buffer = bo->proj.data();
            outputs[proj_slot].nbytes = bo->proj.bytes();
        }
    }
    if (native) {
        auto& output = outputs.back();
        output.index = int(outputs.size()) - 1;
        if (evidence) {
            bo->probs10.reserve(host_buft_, p_item * sub * B * sizeof(float));
            output.host_buffer = bo->probs10.data();
            output.nbytes = bo->probs10.bytes();
        } else {
            output.device_tensor = &raw_probs10;
        }
    }
    // host outputs are sized for the real windows: padding items are not downloaded
    for (auto& output : outputs)
        if (output.host_buffer)
            output.download_bytes = output.nbytes;
    {
        ggml_nvtx::range nvtx_run("pee.run");
        session_->run(inputs, outputs);  // upload + graph + download
    }
    if (outputs[0].out_shape[0] != d || outputs[0].out_shape[1] != T ||
        outputs[0].out_shape[2] != Bg || outputs[1].out_shape[0] != n_spk ||
        outputs[1].out_shape[1] != T || outputs[1].out_shape[2] != Bg) {
        throw std::runtime_error("pee: unexpected output shapes");
    }
    if (has_proj &&
        (outputs[proj_slot].out_shape[0] != static_cast<int64_t>(bo->dp) ||
         outputs[proj_slot].out_shape[1] != T || outputs[proj_slot].out_shape[2] != Bg)) {
        throw std::runtime_error("pee: unexpected proj output shape");
    }
    if (has_proj && proj_on_device_)
        bo->proj_block = keep_on_device(proj_dev, B);

    // Word-timestamp windows: their ASR state cores go to the CTC evidence device to device (no
    // download), complete before the head reads them on its own lane.
    bool staged = false;
    for (int b = 0; sinks && b < B; b++) {
        const EvidenceSink& sink = (*sinks)[b];
        if (!sink.evidence)
            continue;
        if (sink.left_drop + sink.core > out_frames(lens[b]))
            throw std::runtime_error("pee: evidence core outside the window");
        ctc_->stage(
            backend_, raw_asr, static_cast<size_t>(b) * T + sink.left_drop, *sink.evidence,
            sink.row, sink.core);
        staged = true;
    }
    if (staged)
        ggml_backend_synchronize(backend_);

    std::vector<WindowOut> outs(B);
    for (int b = 0; b < B; b++) outs[b] = WindowOut{bo, b, out_frames(lens[b])};
    return outs;
}

PeeModel::Output
PeeModel::fuse_windows(
    const std::vector<Window>& plan, const std::vector<WindowOut>& outs, bool evidence,
    const std::vector<const std::vector<float>*>* probs10) const {
    if (plan.size() != outs.size())
        throw std::runtime_error("pee: plan/outs mismatch");
    const int d = cfg_.asr.d_model;
    const int n_spk = cfg_.num_speakers;
    Output o;
    for (const auto& w : plan) o.n_frames += w.core;
    const int T = o.n_frames;
    const size_t dp = static_cast<size_t>(cfg_.proj_dim);
    if (emit_fused_)
        o.fused.resize(static_cast<size_t>(T) * d);
    o.spk_probs.resize(static_cast<size_t>(T) * n_spk);
    evidence = evidence && ctc_;
    const bool asr = want_asr_states_;
    if (asr)
        o.asr_states.resize(static_cast<size_t>(T) * d);
    if (evidence)
        o.spk_probs10.resize(static_cast<size_t>(T) * cfg_.upsample_factor * n_spk);
    if (dp > 0 && !proj_on_device_)
        o.proj.resize(static_cast<size_t>(T) * dp);
    int t = 0;
    for (size_t i = 0; i < plan.size(); i++) {
        const auto& w = plan[i];
        const auto& r = outs[i];
        if (w.left_drop + w.core > r.t_out)
            throw std::runtime_error("pee: window shorter than its core");
        if (emit_fused_)
            std::memcpy(
                &o.fused[static_cast<size_t>(t) * d],
                r.fused() + static_cast<size_t>(w.left_drop) * d,
                static_cast<size_t>(w.core) * d * sizeof(float));
        std::memcpy(
            &o.spk_probs[static_cast<size_t>(t) * n_spk],
            r.probs80() + static_cast<size_t>(w.left_drop) * n_spk,
            static_cast<size_t>(w.core) * n_spk * sizeof(float));
        if (asr) {
            if (!r.asr())
                throw std::runtime_error("pee: window encoded without ASR states");
            std::memcpy(
                &o.asr_states[static_cast<size_t>(t) * d],
                r.asr() + static_cast<size_t>(w.left_drop) * d,
                static_cast<size_t>(w.core) * d * sizeof(float));
        }
        if (dp > 0 && proj_on_device_) {
            if (!r.batch->proj_block)
                throw std::runtime_error("pee: window encoded without its device projection");
            o.proj_spans.push_back(
                {r.batch->proj_block, static_cast<size_t>(r.index) * r.batch->T + w.left_drop,
                 w.core});
        } else if (dp > 0) {
            std::memcpy(
                &o.proj[static_cast<size_t>(t) * dp],
                r.proj() + static_cast<size_t>(w.left_drop) * dp,
                static_cast<size_t>(w.core) * dp * sizeof(float));
        }
        if (evidence && probs10) {
            const size_t factor = static_cast<size_t>(cfg_.upsample_factor) * n_spk;
            const auto& p = *(*probs10)[i];
            if (p.size() != static_cast<size_t>(w.core) * factor)
                throw std::runtime_error("pee: window 10 ms speaker core size");
            std::copy(p.begin(), p.end(), o.spk_probs10.data() + static_cast<size_t>(t) * factor);
        } else if (evidence) {
            if (!r.batch->probs10.data())
                throw std::runtime_error("pee: window encoded without speaker activity");
            const size_t offset = static_cast<size_t>(r.index) * r.batch->T + w.left_drop;
            const size_t factor = static_cast<size_t>(cfg_.upsample_factor) * n_spk;
            std::memcpy(
                o.spk_probs10.data() + static_cast<size_t>(t) * factor,
                r.batch->probs10.data() + offset * factor,
                static_cast<size_t>(w.core) * factor * sizeof(float));
        }
        t += w.core;
    }
    o.spk_binary.resize(o.spk_probs.size());
    for (size_t i = 0; i < o.spk_probs.size(); i++)
        o.spk_binary[i] = o.spk_probs[i] > cfg_.speaker_activity_threshold ? 1.0f : 0.0f;
    return o;
}

PeeModel::Output
PeeModel::encode(const float* mel, int t_mel) {
    const int n_mels = cfg_.n_mels;
    NormStats st;
    if (cfg_.asr_normalize_type == "per_feature") {
        st = norm_stats(mel, t_mel);  // whole utterance, once
    } else {
        st.mean.assign(n_mels, 0.0f);
        st.inv_std.assign(n_mels, 1.0f);
    }
    const auto plan = plan_windows(t_mel);
    std::vector<WindowOut> outs(plan.size());
    const auto store = mel_store(static_cast<size_t>(t_mel));
    const size_t base = zero_frames();
    upload_mel(*store, base, mel, static_cast<size_t>(t_mel));

    // Multi-window utterances: the speaker branch runs window by window with its streaming state
    // first.
    const bool stateful = stateful_speakers() && plan.size() > 1;
    SpeakerUtt spk;
    if (stateful) {
        spk.mel_frame = base;
        spk.plan = &plan;
        spk.want_p10 = bool(ctc_);
        speaker_steps(*store, {&spk});
    }

    // Batch same-length windows (middle windows share a shape; the first and
    // last usually differ), up to max_batch_ per graph run.
    std::vector<size_t> order(plan.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return bucket_len(plan[a].len()) < bucket_len(plan[b].len());
    });
    size_t i = 0;
    while (i < order.size()) {
        const int blen = bucket_len(plan[order[i]].len());
        std::vector<size_t> group;
        while (i < order.size() && bucket_len(plan[order[i]].len()) == blen &&
               static_cast<int>(group.size()) < max_batch_)
            group.push_back(order[i++]);
        std::vector<SpeakerProbs> p80;
        std::vector<size_t> frames;
        std::vector<const NormStats*> stats;
        std::vector<int> lens;
        for (size_t g : group) {
            frames.push_back(base + static_cast<size_t>(plan[g].enc_stt));
            stats.push_back(&st);
            lens.push_back(plan[g].len());
            if (stateful)
                p80.push_back(spk.p80[g]);
        }
        auto res =
            encode_batch(*store, frames, stats, lens, blen, bool(ctc_), stateful ? &p80 : nullptr);
        for (size_t k = 0; k < group.size(); k++) outs[group[k]] = std::move(res[k]);
    }
    std::vector<const std::vector<float>*> p10;
    for (const auto& p : spk.p10) p10.push_back(&p);
    return fuse_windows(plan, outs, bool(ctc_), stateful ? &p10 : nullptr);
}
