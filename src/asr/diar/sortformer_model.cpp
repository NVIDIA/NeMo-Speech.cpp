// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sortformer_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace nemo_speech::asr;

static SortformerModelConfig
parse_config(const ggml_runtime::GGUFLoader& loader) {
    SortformerModelConfig cfg;

    const std::string version = loader.get_str("sortformer.version", "v2");
    if (version == "v3")
        cfg.version = SortformerVersion::V3;
    else if (version != "v2")
        throw std::runtime_error("sortformer: unsupported model version '" + version + "'");

    EncoderConfig& e = cfg.encoder;
    e.d_model = loader.get_u32("sortformer.encoder.d_model", 512);
    e.n_layers = loader.get_u32("sortformer.encoder.n_layers", 17);
    e.n_heads = loader.get_u32("sortformer.encoder.n_heads", 8);
    e.d_ff = loader.get_u32("sortformer.encoder.d_ff", 2048);
    e.conv_kernel_size = loader.get_u32("sortformer.encoder.conv_kernel_size", 9);
    e.subsampling_factor = loader.get_u32("sortformer.encoder.subsampling_factor", 8);
    e.subsampling_conv_channels =
        loader.get_u32("sortformer.encoder.subsampling_conv_channels", 256);
    e.feat_in = loader.get_u32("sortformer.encoder.feat_in", 128);
    e.xscaling = loader.get_bool("sortformer.encoder.xscaling", true);
    e.use_bias = loader.get_bool("sortformer.encoder.use_bias", true);
    e.pos_emb_max_len = loader.get_u32("sortformer.encoder.pos_emb_max_len", 5000);
    e.cache_mode = CacheMode::Disabled;
    e.conv_context = ConvContext::Symmetric;
    const std::string conv_norm = loader.get_str("sortformer.encoder.conv_norm", "batch_norm");
    e.conv_norm = (conv_norm == "layer_norm") ? ConvNorm::LayerNorm : ConvNorm::BatchNorm;

    RopeTransformerConfig& r = cfg.rope_encoder;
    r.d_model = e.d_model;
    r.n_layers = e.n_layers;
    r.n_heads = e.n_heads;
    r.d_ff = e.d_ff;
    r.qkv_bias = loader.get_bool("sortformer.encoder.qkv_bias", false);
    r.pre_block_norm = loader.get_bool("sortformer.encoder.pre_block_norm", true);
    r.rope_base = loader.get_f32("sortformer.encoder.rope_base", 10000.0f);
    r.rotary_fraction = loader.get_f32("sortformer.encoder.rotary_fraction", 1.0f);
    r.pos_emb_max_len = e.pos_emb_max_len;
    if (loader.get_bool("sortformer.encoder.qk_norm", false))
        throw std::runtime_error("sortformer: qk_norm RoPE encoders are not supported");

    TransformerConfig& t = cfg.transformer;
    t.n_layers = loader.get_u32("sortformer.transformer.n_layers", 18);
    t.hidden_size = loader.get_u32("sortformer.transformer.hidden_size", 192);
    t.inner_size = loader.get_u32("sortformer.transformer.inner_size", 768);
    t.n_heads = loader.get_u32("sortformer.transformer.n_heads", 8);
    if (!cfg.is_v3() && loader.get_bool("sortformer.transformer.pre_ln", false)) {
        throw std::runtime_error("sortformer: pre_ln transformer variant is not supported");
    }

    cfg.num_speakers = loader.get_u32("sortformer.num_speakers", 4);

    DiarScoringConfig& s = cfg.scoring;
    s.sil_frames_per_spk = loader.get_u32("sortformer.scoring.spkcache_sil_frames_per_spk", 3);
    s.pred_score_threshold = loader.get_f32("sortformer.scoring.pred_score_threshold", 0.25f);
    s.scores_boost_latest = loader.get_f32("sortformer.scoring.scores_boost_latest", 0.05f);
    s.sil_threshold = loader.get_f32("sortformer.scoring.sil_threshold", 0.2f);
    s.strong_boost_rate = loader.get_f32("sortformer.scoring.strong_boost_rate", 0.75f);
    s.weak_boost_rate = loader.get_f32("sortformer.scoring.weak_boost_rate", 1.5f);
    s.min_pos_scores_rate = loader.get_f32("sortformer.scoring.min_pos_scores_rate", 0.5f);

    cfg.high_resolution = loader.get_bool("sortformer.high_resolution", false);
    cfg.output_subsampling_factor =
        loader.get_u32("sortformer.output_subsampling_factor", cfg.is_v3() ? 1 : 8);
    cfg.upsample_factor = loader.get_u32("sortformer.upsample_factor", cfg.is_v3() ? 8 : 1);
    cfg.learnable_silence = loader.get_bool("sortformer.learnable_silence", false);
    if (cfg.is_v3() && (!cfg.high_resolution || cfg.output_subsampling_factor != 1 ||
                        cfg.upsample_factor != e.subsampling_factor || !cfg.learnable_silence)) {
        throw std::runtime_error("sortformer: unsupported v3 resolution or silence contract");
    }

    cfg.sample_rate = loader.get_u32("sortformer.preprocessor.sample_rate", 16000);
    cfg.window_size = loader.get_f32("sortformer.preprocessor.window_size", 0.025f);
    cfg.window_stride = loader.get_f32("sortformer.preprocessor.window_stride", 0.01f);
    cfg.n_fft = loader.get_u32("sortformer.preprocessor.n_fft", 512);
    cfg.n_mels = loader.get_u32("sortformer.preprocessor.features", e.feat_in);
    cfg.preemph = loader.get_f32("sortformer.preprocessor.preemph", 0.97f);
    cfg.log_zero_guard =
        loader.get_f32("sortformer.preprocessor.log_zero_guard", cfg.log_zero_guard);

    // NOTE: the GGUF also carries sortformer.streaming.* (the checkpoint's
    // training-time geometry, e.g. 188/0/188) as provenance metadata; runtime
    // geometry deliberately comes from DiarGeometry/DiarConfig instead.
    return cfg;
}

SortformerGraph::SortformerGraph(const SortformerModelConfig& cfg)
    : cfg_(cfg), encoder_(nullptr), feature_stack_proj_(nullptr), rope_encoder_(nullptr),
      transformer_(nullptr), subpixel_upsample_(nullptr) {
    if (cfg.is_v3()) {
        feature_stack_proj_ = new ggml_runtime::Linear(
            "encoder.pre_encode.proj", cfg.n_mels * cfg.encoder.subsampling_factor,
            cfg.encoder.d_model, false);
        rope_encoder_ = new RopeTransformerEncoder("encoder", cfg.rope_encoder);
    } else {
        encoder_ = new FastConformerEncoder("encoder", cfg.encoder);
    }
    encoder_proj_ =
        new ggml_runtime::Linear("encoder_proj", cfg.encoder.d_model, cfg.transformer.hidden_size);
    if (cfg.is_v3()) {
        subpixel_upsample_ = new ggml_runtime::Conv1D(
            "subpixel_upsample", cfg.transformer.hidden_size,
            cfg.transformer.hidden_size * cfg.upsample_factor, 3, 1, 1);
    } else {
        transformer_ = new TransformerEncoderModule("transformer", cfg.transformer);
    }
    head_hidden_ = new ggml_runtime::Linear(
        "head.first_hidden_to_hidden", cfg.transformer.hidden_size, cfg.transformer.hidden_size);
    head_spks_ = new ggml_runtime::Linear(
        "head.single_hidden_to_spks", cfg.transformer.hidden_size, cfg.num_speakers);
}

SortformerGraph::~SortformerGraph() {
    delete encoder_;
    delete feature_stack_proj_;
    delete rope_encoder_;
    delete encoder_proj_;
    delete transformer_;
    delete subpixel_upsample_;
    delete head_hidden_;
    delete head_spks_;
}

void
SortformerGraph::define_tensors(ggml_runtime::Session* session) {
    if (encoder_ != nullptr)
        encoder_->define_tensors(session);
    if (feature_stack_proj_ != nullptr)
        feature_stack_proj_->define_tensors(session);
    if (rope_encoder_ != nullptr)
        rope_encoder_->define_tensors(session);
    encoder_proj_->define_tensors(session);
    if (transformer_ != nullptr)
        transformer_->define_tensors(session);
    if (subpixel_upsample_ != nullptr)
        subpixel_upsample_->define_tensors(session);
    head_hidden_->define_tensors(session);
    head_spks_->define_tensors(session);
}

void
SortformerGraph::set_data(ggml_runtime::Session* session) {
    if (encoder_ != nullptr)
        encoder_->set_data(session);
    if (feature_stack_proj_ != nullptr)
        feature_stack_proj_->set_data(session);
    if (rope_encoder_ != nullptr)
        rope_encoder_->set_data(session);
    encoder_proj_->set_data(session);
    if (transformer_ != nullptr)
        transformer_->set_data(session);
    if (subpixel_upsample_ != nullptr)
        subpixel_upsample_->set_data(session);
    head_hidden_->set_data(session);
    head_spks_->set_data(session);
}

ggml_runtime::TensorBag
SortformerGraph::build_graph(
    ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
    ggml_runtime::TensorContainer* tc) {
    (void)input_tensors;  // inputs are fetched by name (optional set varies)
    // 1. Pre-encode the mel window: (n_mels, T_mel) -> (512, T3). NO xscale
    //    here - AOSC caches store raw pre-encode embeddings (NeMo parity).
    auto mel = tc->get_tensor_by_name("input.mel");
    if (cfg_.is_v3()) {
        auto bf_ctx = tc->get_ctx_of_buffer_type(mel.buft);
        ggml_context* ctx = bf_ctx.ctx;
        const int64_t t_mel = mel.tensor->ne[1];
        const int64_t batch = mel.tensor->ne[3];
        const int factor = cfg_.encoder.subsampling_factor;
        const int64_t pad = (factor - t_mel % factor) % factor;
        auto padded = pad == 0 ? mel.tensor : ggml_pad(ctx, mel.tensor, 0, pad, 0, 0);
        const int64_t coarse = (t_mel + pad) / factor;
        auto stacked =
            ggml_reshape_3d(ctx, padded, static_cast<int64_t>(cfg_.n_mels) * factor, coarse, batch);
        ggml_runtime::TensorBag stack_in;
        stack_in.add_tensor(ggml_runtime::ggml_bf_tensor(stacked, mel.buft));
        auto chunk_embs = feature_stack_proj_->build_graph(session, stack_in, tc).get_tensor(0);

        ggml_tensor* x = chunk_embs.tensor;
        if (tc->has_tensor_by_name("input.state")) {
            x = ggml_concat(ctx, tc->get_tensor_by_name("input.state").tensor, x, 1);
        }
        ggml_runtime::TensorBag enc_in;
        enc_in.add_tensor(ggml_runtime::ggml_bf_tensor(x, chunk_embs.buft));
        enc_in.add_tensor(tc->get_tensor_by_name("input.positions"));
        auto encoded = rope_encoder_->build_graph(session, enc_in, tc);
        auto proj = encoder_proj_->build_graph(session, encoded, tc).get_tensor(0);

        // Conv1D consumes (time, channels, batch). Convert back to a
        // channel-major contiguous tensor before PixelShuffle-style reshape.
        auto conv_input = ggml_cont(ctx, ggml_permute(ctx, proj.tensor, 1, 0, 2, 3));
        std::vector<ggml_tensor*> conv_items;
        conv_items.reserve(static_cast<size_t>(batch));
        for (int64_t b = 0; b < batch; ++b) {
            auto item = ggml_view_3d(
                ctx, conv_input, conv_input->ne[0], conv_input->ne[1], 1, conv_input->nb[1],
                conv_input->nb[2], static_cast<size_t>(b) * conv_input->nb[2]);
            ggml_runtime::TensorBag conv_bag;
            conv_bag.add_tensor(ggml_runtime::ggml_bf_tensor(ggml_cont(ctx, item), proj.buft));
            auto item_out =
                subpixel_upsample_->build_graph(session, conv_bag, tc).get_tensor(0).tensor;
            conv_items.push_back(item_out);
        }
        // ggml's portable Conv1D path requires one item at a time. Join those
        // items as a balanced tree so every output is copied O(log B) times;
        // left-folding the concatenation copies the growing prefix O(B^2).
        while (conv_items.size() > 1) {
            std::vector<ggml_tensor*> joined;
            joined.reserve((conv_items.size() + 1) / 2);
            for (size_t i = 0; i < conv_items.size(); i += 2) {
                joined.push_back(
                    i + 1 < conv_items.size()
                        ? ggml_concat(ctx, conv_items[i], conv_items[i + 1], 2)
                        : conv_items[i]);
            }
            conv_items = std::move(joined);
        }
        ggml_tensor* conv_tensor = conv_items.front();
        auto channels_first = ggml_cont(ctx, ggml_permute(ctx, conv_tensor, 1, 0, 2, 3));
        auto high_res = ggml_reshape_3d(
            ctx, channels_first, cfg_.transformer.hidden_size, x->ne[1] * cfg_.upsample_factor,
            batch);

        ggml_runtime::TensorBag head_in;
        head_in.add_tensor(ggml_runtime::ggml_bf_tensor(ggml_relu(ctx, high_res), proj.buft));
        auto h2 = head_hidden_->build_graph(session, head_in, tc).get_tensor(0);
        ggml_runtime::TensorBag head_in2;
        head_in2.add_tensor(ggml_runtime::ggml_bf_tensor(ggml_relu(ctx, h2.tensor), h2.buft));
        auto logits = head_spks_->build_graph(session, head_in2, tc).get_tensor(0);

        ggml_runtime::TensorBag out;
        out.add_tensor(ggml_runtime::ggml_bf_tensor(ggml_sigmoid(ctx, logits.tensor), logits.buft));
        out.add_tensor(chunk_embs);
        return out;
    }

    ggml_runtime::TensorBag mel_bag;
    mel_bag.add_tensor(mel);
    auto pre = encoder_->build_pre_encode(session, mel_bag, tc);
    auto chunk_embs = pre.get_tensor(0);
    auto bf_ctx = tc->get_ctx_of_buffer_type(chunk_embs.buft);

    // 2. Concat the compact state prefix and current chunk along time.
    ggml_tensor* x = chunk_embs.tensor;
    if (tc->has_tensor_by_name("input.state")) {
        x = ggml_concat(bf_ctx.ctx, tc->get_tensor_by_name("input.state").tensor, x, 1);
    }

    // 3. xscale + rel-pos + conformer stack over the concatenation.
    ggml_runtime::TensorBag enc_in;
    enc_in.add_tensor(ggml_runtime::ggml_bf_tensor(x, chunk_embs.buft));
    if (tc->has_tensor_by_name("input.attention_mask")) {
        enc_in.add_tensor(tc->get_tensor_by_name("input.attention_mask"));
        enc_in.add_tensor(tc->get_tensor_by_name("input.valid_mask"));
    }
    auto enc_out = encoder_->build_graph_from_embeddings(session, enc_in, tc);

    // 4. Projection 512->192, transformer, sigmoid head
    //    (NeMo forward_speaker_sigmoids: relu -> linear -> relu -> linear -> sigmoid).
    auto proj = encoder_proj_->build_graph(session, enc_out, tc);
    ggml_runtime::TensorBag trans_in;
    trans_in.add_tensor(proj.get_tensor(0));
    if (tc->has_tensor_by_name("input.attention_mask")) {
        trans_in.add_tensor(tc->get_tensor_by_name("input.attention_mask"));
    }
    auto trans = transformer_->build_graph(session, trans_in, tc);

    auto h = trans.get_tensor(0);
    ggml_runtime::TensorBag head_in;
    head_in.add_tensor(ggml_runtime::ggml_bf_tensor(ggml_relu(bf_ctx.ctx, h.tensor), h.buft));
    auto h2 = head_hidden_->build_graph(session, head_in, tc).get_tensor(0);
    ggml_runtime::TensorBag head_in2;
    head_in2.add_tensor(ggml_runtime::ggml_bf_tensor(ggml_relu(bf_ctx.ctx, h2.tensor), h2.buft));
    auto logits = head_spks_->build_graph(session, head_in2, tc).get_tensor(0);
    auto preds = ggml_sigmoid(bf_ctx.ctx, logits.tensor);

    ggml_runtime::TensorBag out;
    out.add_tensor(ggml_runtime::ggml_bf_tensor(preds, logits.buft));
    out.add_tensor(chunk_embs);
    return out;
}

class SortformerModel::SortformerBatcher {
   public:
    struct Key {
        int t_mel = 0;
        int state_frames = -1;

        bool operator==(const Key& other) const {
            return t_mel == other.t_mel && state_frames == other.state_frames;
        }
    };

    struct Request {
        std::vector<float> mel;
        std::vector<float> spkcache;
        std::vector<float> fifo;
        int spkcache_frames = 0;
        int fifo_frames = 0;
    };

    SortformerBatcher(SortformerModel* model, const BatchingConfig& batching)
        : model_(model), queue_(batching, [this](const Key& key, std::vector<Request>&& requests) {
              return execute(key, std::move(requests));
          }) {}

    ChunkOutput run(
        const float* mel, int t_mel, const float* spkcache, int spkcache_frames, const float* fifo,
        int fifo_frames) {
        const int d = model_->cfg_.encoder.d_model;
        Request request;
        request.mel.assign(
            mel, mel + static_cast<size_t>(model_->cfg_.n_mels) * static_cast<size_t>(t_mel));
        if (spkcache_frames > 0) {
            request.spkcache.assign(spkcache, spkcache + static_cast<size_t>(d) * spkcache_frames);
        }
        if (fifo_frames > 0)
            request.fifo.assign(fifo, fifo + static_cast<size_t>(d) * fifo_frames);
        request.spkcache_frames = spkcache_frames;
        request.fifo_frames = fifo_frames;
        const int state_key = model_->cfg_.is_v3() ? spkcache_frames + fifo_frames : -1;
        return queue_.run({t_mel, state_key}, std::move(request));
    }

    BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    std::vector<ChunkOutput> execute(const Key& key, std::vector<Request>&& requests) {
        const int B = static_cast<int>(requests.size());
        const int d = model_->cfg_.encoder.d_model;
        const int n_mels = model_->cfg_.n_mels;
        const int n_spk = model_->cfg_.num_speakers;
        const int t3 = model_->subsampled_len(key.t_mel);
        int max_state_frames = 0;
        for (const auto& request : requests) {
            max_state_frames =
                std::max(max_state_frames, request.spkcache_frames + request.fifo_frames);
        }
        const int total = max_state_frames + t3;
        const size_t mel_item = static_cast<size_t>(n_mels) * key.t_mel;
        const size_t state_item = static_cast<size_t>(d) * max_state_frames;

        std::vector<float> mel(mel_item * B);
        std::vector<float> state(state_item * B, 0.0f);
        bool needs_padding_mask = false;
        for (int b = 0; b < B; ++b) {
            const auto& request = requests[static_cast<size_t>(b)];
            const size_t spkcache_size = static_cast<size_t>(d) * request.spkcache_frames;
            const size_t fifo_size = static_cast<size_t>(d) * request.fifo_frames;
            if (request.mel.size() != mel_item || request.spkcache.size() != spkcache_size ||
                request.fifo.size() != fifo_size) {
                throw std::runtime_error("sortformer batch contains incompatible inputs");
            }
            std::copy(
                request.mel.begin(), request.mel.end(),
                mel.begin() + static_cast<size_t>(b) * mel_item);
            const int state_frames = request.spkcache_frames + request.fifo_frames;
            const int state_offset = max_state_frames - state_frames;
            needs_padding_mask = needs_padding_mask || state_offset != 0;
            auto state_out = state.begin() + static_cast<size_t>(b) * state_item +
                             static_cast<size_t>(state_offset) * d;
            std::copy(request.spkcache.begin(), request.spkcache.end(), state_out);
            std::copy(request.fifo.begin(), request.fifo.end(), state_out + spkcache_size);
        }

        std::vector<ggml_runtime::Session::Input> inputs;
        inputs.push_back({"input.mel", GGML_TYPE_F32, mel.data(), {n_mels, key.t_mel, 1, B}});
        if (max_state_frames > 0) {
            inputs.push_back(
                {"input.state", GGML_TYPE_F32, state.data(), {d, max_state_frames, B}});
        }
        std::vector<int32_t> positions;
        if (model_->cfg_.is_v3()) {
            if (key.state_frames != max_state_frames)
                throw std::runtime_error("sortformer v3 batch mixed state lengths");
            positions.resize(total);
            for (int i = 0; i < total; ++i) positions[static_cast<size_t>(i)] = i;
            inputs.push_back({"input.positions", GGML_TYPE_I32, positions.data(), {total}});
        }
        std::vector<float> attention_mask;
        std::vector<float> valid_mask;
        if (needs_padding_mask) {
            attention_mask.assign(static_cast<size_t>(total) * B, 0.0f);
            valid_mask.assign(static_cast<size_t>(total) * B, 1.0f);
            for (int b = 0; b < B; ++b) {
                const auto& request = requests[static_cast<size_t>(b)];
                const int state_frames = request.spkcache_frames + request.fifo_frames;
                const int state_offset = max_state_frames - state_frames;
                auto mask_base = static_cast<size_t>(b) * total;
                std::fill_n(attention_mask.begin() + mask_base, state_offset, -1e9f);
                std::fill_n(valid_mask.begin() + mask_base, state_offset, 0.0f);
            }
            inputs.push_back(
                {"input.attention_mask", GGML_TYPE_F32, attention_mask.data(), {total, 1, 1, B}});
            inputs.push_back({"input.valid_mask", GGML_TYPE_F32, valid_mask.data(), {1, total, B}});
        }

        const int output_factor = model_->cfg_.is_v3() ? model_->cfg_.upsample_factor : 1;
        const size_t preds_item = static_cast<size_t>(total) * output_factor * n_spk;
        const size_t embs_item = static_cast<size_t>(t3) * d;
        std::vector<float> preds(preds_item * B);
        std::vector<float> embs(embs_item * B);
        std::vector<ggml_runtime::Session::Output> outputs(2);
        outputs[0].index = 0;
        outputs[0].host_buffer = preds.data();
        outputs[0].nbytes = preds.size() * sizeof(float);
        outputs[1].index = 1;
        outputs[1].host_buffer = embs.data();
        outputs[1].nbytes = embs.size() * sizeof(float);
        model_->session_->run(inputs, outputs);

        const int out_native_total = static_cast<int>(outputs[0].out_shape[1]);
        const int out_pred_batch = static_cast<int>(outputs[0].out_shape[2]);
        const int out_t3 = static_cast<int>(outputs[1].out_shape[1]);
        const int out_emb_batch = static_cast<int>(outputs[1].out_shape[2]);
        if (out_native_total != total * output_factor || out_t3 != t3 || out_pred_batch != B ||
            out_emb_batch != B) {
            throw std::runtime_error(
                "sortformer: unexpected batched output shape (got total=" +
                std::to_string(out_native_total) + " chunk=" + std::to_string(out_t3) +
                " batch=" + std::to_string(out_pred_batch) + "/" + std::to_string(out_emb_batch) +
                ", expected total=" + std::to_string(total) + " chunk=" + std::to_string(t3) +
                " batch=" + std::to_string(B) + ")");
        }

        std::vector<ChunkOutput> results(static_cast<size_t>(B));
        for (int b = 0; b < B; ++b) {
            auto& result = results[static_cast<size_t>(b)];
            const auto& request = requests[static_cast<size_t>(b)];
            const int state_frames = request.spkcache_frames + request.fifo_frames;
            const int state_offset = max_state_frames - state_frames;
            result.total_frames = state_frames + t3;
            result.chunk_frames = t3;
            const auto pred_begin = preds.begin() + static_cast<size_t>(b) * preds_item +
                                    static_cast<size_t>(state_offset) * output_factor * n_spk;
            if (model_->cfg_.is_v3()) {
                result.native_total_frames = result.total_frames * output_factor;
                result.native_preds.assign(
                    pred_begin,
                    pred_begin + static_cast<size_t>(result.native_total_frames) * n_spk);
                result.preds.resize(static_cast<size_t>(result.total_frames) * n_spk);
                for (int f = 0; f < result.total_frames; ++f) {
                    for (int s = 0; s < n_spk; ++s) {
                        float sum = 0.0f;
                        for (int u = 0; u < output_factor; ++u) {
                            sum += result.native_preds
                                       [(static_cast<size_t>(f) * output_factor + u) * n_spk + s];
                        }
                        result.preds[static_cast<size_t>(f) * n_spk + s] = sum / output_factor;
                    }
                }
            } else {
                result.preds.assign(
                    pred_begin, pred_begin + static_cast<size_t>(result.total_frames) * n_spk);
            }
            result.chunk_embs.assign(
                embs.begin() + static_cast<size_t>(b) * embs_item,
                embs.begin() + static_cast<size_t>(b + 1) * embs_item);
        }
        return results;
    }

    SortformerModel* model_;
    MicroBatcher<Key, Request, ChunkOutput> queue_;
};

SortformerModel::SortformerModel(
    ggml_runtime::BackendManager& bm, const std::string& gguf_path,
    const BatchingConfig& batching) {
    loader_ = std::make_unique<ggml_runtime::GGUFLoader>(gguf_path);
    cfg_ = parse_config(*loader_);

    // Trained mel filterbank for the FE (kept on host; the FE consumes it).
    const int n_freq = cfg_.n_fft / 2 + 1;
    mel_basis_.resize(static_cast<size_t>(cfg_.n_mels) * n_freq);
    const char* fb = loader_->get_tensor_file_data("preprocessor.fb", mel_basis_.size() * 4);
    std::memcpy(mel_basis_.data(), fb, mel_basis_.size() * 4);
    if (cfg_.learnable_silence) {
        learnable_silence_embedding_.resize(cfg_.encoder.d_model);
        const char* sil = loader_->get_tensor_file_data(
            "learnable_sil_emb", learnable_silence_embedding_.size() * sizeof(float));
        std::memcpy(
            learnable_silence_embedding_.data(), sil,
            learnable_silence_embedding_.size() * sizeof(float));
    }

    graph_ = std::make_unique<SortformerGraph>(cfg_);
    session_ = std::make_unique<ggml_runtime::Session>(bm, graph_.get(), loader_.get());
    // Keep the streaming state-length cycle and occasional tail shapes cached.
    session_->set_run_cache_capacity(48);
    session_->setup();
    batcher_ = std::make_unique<SortformerBatcher>(this, batching);
}

SortformerModel::~SortformerModel() = default;

int
SortformerModel::subsampled_len(int t_mel) const {
    if (cfg_.is_v3())
        return (t_mel + cfg_.encoder.subsampling_factor - 1) / cfg_.encoder.subsampling_factor;
    int len = t_mel;
    const int n_stages = static_cast<int>(std::log2(cfg_.encoder.subsampling_factor));
    for (int i = 0; i < n_stages; i++) len = (len + 1) / 2;
    return len;
}

SortformerModel::ChunkOutput
SortformerModel::run_chunk(
    const float* mel, int t_mel, const float* spkcache, int spkcache_frames, const float* fifo,
    int fifo_frames) {
    return batcher_->run(mel, t_mel, spkcache, spkcache_frames, fifo, fifo_frames);
}

BatchMetrics
SortformerModel::batch_metrics() const {
    return batcher_->metrics();
}
