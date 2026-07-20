// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sortformer_model.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace nemo_speech::asr;

static SortformerModelConfig
parse_config(const ggml_runtime::GGUFLoader& loader) {
    SortformerModelConfig cfg;

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

    TransformerConfig& t = cfg.transformer;
    t.n_layers = loader.get_u32("sortformer.transformer.n_layers", 18);
    t.hidden_size = loader.get_u32("sortformer.transformer.hidden_size", 192);
    t.inner_size = loader.get_u32("sortformer.transformer.inner_size", 768);
    t.n_heads = loader.get_u32("sortformer.transformer.n_heads", 8);
    if (loader.get_bool("sortformer.transformer.pre_ln", false)) {
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

SortformerGraph::SortformerGraph(const SortformerModelConfig& cfg) : cfg_(cfg) {
    encoder_ = new FastConformerEncoder("encoder", cfg.encoder);
    encoder_proj_ =
        new ggml_runtime::Linear("encoder_proj", cfg.encoder.d_model, cfg.transformer.hidden_size);
    transformer_ = new TransformerEncoderModule("transformer", cfg.transformer);
    head_hidden_ = new ggml_runtime::Linear(
        "head.first_hidden_to_hidden", cfg.transformer.hidden_size, cfg.transformer.hidden_size);
    head_spks_ = new ggml_runtime::Linear(
        "head.single_hidden_to_spks", cfg.transformer.hidden_size, cfg.num_speakers);
}

SortformerGraph::~SortformerGraph() {
    delete encoder_;
    delete encoder_proj_;
    delete transformer_;
    delete head_hidden_;
    delete head_spks_;
}

void
SortformerGraph::define_tensors(ggml_runtime::Session* session) {
    encoder_->define_tensors(session);
    encoder_proj_->define_tensors(session);
    transformer_->define_tensors(session);
    head_hidden_->define_tensors(session);
    head_spks_->define_tensors(session);
}

void
SortformerGraph::set_data(ggml_runtime::Session* session) {
    encoder_->set_data(session);
    encoder_proj_->set_data(session);
    transformer_->set_data(session);
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
    ggml_runtime::TensorBag mel_bag;
    mel_bag.add_tensor(mel);
    auto pre = encoder_->build_pre_encode(session, mel_bag, tc);
    auto chunk_embs = pre.get_tensor(0);
    auto bf_ctx = tc->get_ctx_of_buffer_type(chunk_embs.buft);

    // 2. Concat [spkcache | fifo | chunk] along time. The optional parts are
    //    per-call inputs; when a length is 0 the pipeline omits the input
    //    entirely (a 0-length ggml tensor is not representable), which also
    //    keys a distinct graph in the run cache.
    ggml_tensor* x = chunk_embs.tensor;
    if (tc->has_tensor_by_name("input.fifo")) {
        x = ggml_concat(bf_ctx.ctx, tc->get_tensor_by_name("input.fifo").tensor, x, 1);
    }
    if (tc->has_tensor_by_name("input.spkcache")) {
        x = ggml_concat(bf_ctx.ctx, tc->get_tensor_by_name("input.spkcache").tensor, x, 1);
    }

    // 3. xscale + rel-pos + conformer stack over the concatenation.
    ggml_runtime::TensorBag enc_in;
    enc_in.add_tensor(ggml_runtime::ggml_bf_tensor(x, chunk_embs.buft));
    auto enc_out = encoder_->build_graph_from_embeddings(session, enc_in, tc);

    // 4. Projection 512->192, transformer, sigmoid head
    //    (NeMo forward_speaker_sigmoids: relu -> linear -> relu -> linear -> sigmoid).
    auto proj = encoder_proj_->build_graph(session, enc_out, tc);
    auto trans = transformer_->build_graph(session, proj, tc);

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

SortformerModel::SortformerModel(ggml_runtime::BackendManager& bm, const std::string& gguf_path) {
    loader_ = std::make_unique<ggml_runtime::GGUFLoader>(gguf_path);
    cfg_ = parse_config(*loader_);

    // Trained mel filterbank for the FE (kept on host; the FE consumes it).
    const int n_freq = cfg_.n_fft / 2 + 1;
    mel_basis_.resize(static_cast<size_t>(cfg_.n_mels) * n_freq);
    const char* fb = loader_->get_tensor_file_data("preprocessor.fb", mel_basis_.size() * 4);
    std::memcpy(mel_basis_.data(), fb, mel_basis_.size() * 4);

    graph_ = std::make_unique<SortformerGraph>(cfg_);
    session_ = std::make_unique<ggml_runtime::Session>(bm, graph_.get(), loader_.get());
    // Warmup + the steady-state FIFO cycle touch a few dozen distinct
    // (T_mel, L1, L2) shapes; keep them all cached.
    session_->set_run_cache_capacity(48);
    session_->setup();
}

SortformerModel::~SortformerModel() = default;

int
SortformerModel::subsampled_len(int t_mel) const {
    int len = t_mel;
    const int n_stages = static_cast<int>(std::log2(cfg_.encoder.subsampling_factor));
    for (int i = 0; i < n_stages; i++) len = (len + 1) / 2;
    return len;
}

SortformerModel::ChunkOutput
SortformerModel::run_chunk(
    const float* mel, int t_mel, const float* spkcache, int spkcache_frames, const float* fifo,
    int fifo_frames) {
    const int d = cfg_.encoder.d_model;
    const int n_spk = cfg_.num_speakers;
    const int t3 = subsampled_len(t_mel);
    const int total = spkcache_frames + fifo_frames + t3;

    std::vector<ggml_runtime::Session::Input> inputs;
    inputs.push_back({"input.mel", GGML_TYPE_F32, mel, {cfg_.n_mels, t_mel}});
    if (spkcache_frames > 0) {
        inputs.push_back({"input.spkcache", GGML_TYPE_F32, spkcache, {d, spkcache_frames}});
    }
    if (fifo_frames > 0) {
        inputs.push_back({"input.fifo", GGML_TYPE_F32, fifo, {d, fifo_frames}});
    }

    ChunkOutput res;
    res.preds.resize(static_cast<size_t>(total) * n_spk);
    res.chunk_embs.resize(static_cast<size_t>(t3) * d);
    std::vector<ggml_runtime::Session::Output> outputs(2);
    outputs[0].index = 0;
    outputs[0].host_buffer = res.preds.data();
    outputs[0].nbytes = res.preds.size() * sizeof(float);
    outputs[1].index = 1;
    outputs[1].host_buffer = res.chunk_embs.data();
    outputs[1].nbytes = res.chunk_embs.size() * sizeof(float);
    session_->run(inputs, outputs);

    res.total_frames = static_cast<int>(outputs[0].out_shape[1]);
    res.chunk_frames = static_cast<int>(outputs[1].out_shape[1]);
    if (res.total_frames != total || res.chunk_frames != t3) {
        throw std::runtime_error(
            "sortformer: unexpected output shape (got total=" + std::to_string(res.total_frames) +
            " chunk=" + std::to_string(res.chunk_frames) +
            ", expected total=" + std::to_string(total) + " chunk=" + std::to_string(t3) + ")");
    }
    return res;
}
