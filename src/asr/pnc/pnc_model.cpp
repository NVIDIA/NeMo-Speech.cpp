// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pnc_model.h"

#include <ggml.h>

#include <cmath>

#include "nn.h"

namespace nemo_speech::asr::pnc {
namespace {

namespace rt = ggml_runtime;

// BERT encoder (embeddings + post-LN blocks) + punct/capit FC heads. Per-call
// inputs by name on the per-call container:
//   pnc.in.ids / pnc.in.pos / pnc.in.typ  - I32 [n] token / position / segment ids.
// Outputs: [0] punct ids [n], [1] capit ids [n].  Keeping the argmax in the
// graph avoids copying both token-classification matrices back to the host.
class PncBertModule : public rt::Module {
   public:
    explicit PncBertModule(const PncConfig& cfg) : cfg_(cfg) {
        const int64_t norm_shape[GGML_MAX_DIMS] = {cfg_.hidden, 1, 1, 1};
        embd_norm_ = std::make_unique<rt::LayerNorm>("pnc.embd_norm", norm_shape);
        for (int i = 0; i < cfg_.n_layers; ++i) {
            const std::string p = "pnc.blk." + std::to_string(i) + ".";
            Layer l;
            l.q = std::make_unique<rt::Linear>(p + "attn_q", cfg_.hidden, cfg_.hidden);
            l.k = std::make_unique<rt::Linear>(p + "attn_k", cfg_.hidden, cfg_.hidden);
            l.v = std::make_unique<rt::Linear>(p + "attn_v", cfg_.hidden, cfg_.hidden);
            l.attn_out = std::make_unique<rt::Linear>(p + "attn_out", cfg_.hidden, cfg_.hidden);
            l.attn_norm = std::make_unique<rt::LayerNorm>(p + "attn_norm", norm_shape);
            l.ffn_up = std::make_unique<rt::Linear>(p + "ffn_up", cfg_.hidden, cfg_.intermediate);
            l.ffn_down =
                std::make_unique<rt::Linear>(p + "ffn_down", cfg_.intermediate, cfg_.hidden);
            l.ffn_norm = std::make_unique<rt::LayerNorm>(p + "ffn_norm", norm_shape);
            layers_.push_back(std::move(l));
        }
        punct_head_ = std::make_unique<rt::Linear>(
            "pnc.punct_head", cfg_.hidden, static_cast<int>(cfg_.punct_labels.size()));
        capit_head_ = std::make_unique<rt::Linear>(
            "pnc.capit_head", cfg_.hidden, static_cast<int>(cfg_.capit_labels.size()));
    }

    void define_tensors(rt::Session* session) override {
        auto* tc = session->model_tensor_container.get();
        auto* g = session->gguf_loader;
        tc->create_tensor_2d(
            "pnc.token_embd.weight", g->get_tensor_type("pnc.token_embd.weight"), cfg_.hidden,
            static_cast<int64_t>(cfg_.vocab.size()));
        tc->create_tensor_2d(
            "pnc.pos_embd.weight", g->get_tensor_type("pnc.pos_embd.weight"), cfg_.hidden,
            cfg_.max_position);
        tc->create_tensor_2d(
            "pnc.type_embd.weight", g->get_tensor_type("pnc.type_embd.weight"), cfg_.hidden,
            cfg_.type_vocab_size);
        embd_norm_->define_tensors(session);
        for (auto& l : layers_) {
            l.q->define_tensors(session);
            l.k->define_tensors(session);
            l.v->define_tensors(session);
            l.attn_out->define_tensors(session);
            l.attn_norm->define_tensors(session);
            l.ffn_up->define_tensors(session);
            l.ffn_down->define_tensors(session);
            l.ffn_norm->define_tensors(session);
        }
        punct_head_->define_tensors(session);
        capit_head_->define_tensors(session);
    }

    void set_data(rt::Session* session) override {
        session->load_weight("pnc.token_embd.weight");
        session->load_weight("pnc.pos_embd.weight");
        session->load_weight("pnc.type_embd.weight");
        embd_norm_->set_data(session);
        for (auto& l : layers_) {
            l.q->set_data(session);
            l.k->set_data(session);
            l.v->set_data(session);
            l.attn_out->set_data(session);
            l.attn_norm->set_data(session);
            l.ffn_up->set_data(session);
            l.ffn_down->set_data(session);
            l.ffn_norm->set_data(session);
        }
        punct_head_->set_data(session);
        capit_head_->set_data(session);
    }

    rt::TensorBag build_graph(
        rt::Session* session, rt::TensorBag, rt::TensorContainer* tc) override {
        auto* mtc = session->model_tensor_container.get();
        auto tok_w = mtc->get_tensor_by_name("pnc.token_embd.weight");
        auto pos_w = mtc->get_tensor_by_name("pnc.pos_embd.weight");
        auto typ_w = mtc->get_tensor_by_name("pnc.type_embd.weight");
        ggml_backend_buffer_type_t buft = tok_w.buft;
        ggml_context* ctx = tc->get_ctx_of_buffer_type(buft).ctx;

        ggml_tensor* ids = tc->get_tensor_by_name("pnc.in.ids").tensor;
        ggml_tensor* pos = tc->get_tensor_by_name("pnc.in.pos").tensor;
        ggml_tensor* typ = tc->get_tensor_by_name("pnc.in.typ").tensor;

        // CUDA GET_ROWS indexes a matrix, so gather each item's embedding
        // columns independently and concatenate them into [H,L,B] in the one
        // graph.  The BERT layers below then run as real batched GEMMs.
        const int64_t L_in = ids->ne[0];
        const int64_t B_in = ids->ne[1];
        ggml_tensor* x = nullptr;
        for (int64_t b = 0; b < B_in; ++b) {
            auto ids_b = ggml_view_1d(ctx, ids, L_in, static_cast<size_t>(b) * ids->nb[1]);
            auto pos_b = ggml_view_1d(ctx, pos, L_in, static_cast<size_t>(b) * pos->nb[1]);
            auto typ_b = ggml_view_1d(ctx, typ, L_in, static_cast<size_t>(b) * typ->nb[1]);
            auto xb = ggml_add(
                ctx,
                ggml_add(
                    ctx, ggml_get_rows(ctx, tok_w.tensor, ids_b),
                    ggml_get_rows(ctx, pos_w.tensor, pos_b)),
                ggml_get_rows(ctx, typ_w.tensor, typ_b));
            xb = ggml_reshape_3d(ctx, xb, cfg_.hidden, L_in, 1);
            x = x == nullptr ? xb : ggml_concat(ctx, x, xb, 2);
        }
        x = sub(embd_norm_.get(), session, tc, x);

        const int64_t d_k = cfg_.hidden / cfg_.n_heads;
        const int64_t n_head = cfg_.n_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

        for (auto& l : layers_) {
            const int64_t L = x->ne[1];
            const int64_t B = x->ne[2];
            ggml_tensor* q = sub(l.q.get(), session, tc, x);
            ggml_tensor* k = sub(l.k.get(), session, tc, x);
            ggml_tensor* v = sub(l.v.get(), session, tc, x);

            ggml_tensor* qh = ggml_cont(
                ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, d_k, n_head, L, B), 0, 2, 1, 3));
            ggml_tensor* kh = ggml_cont(
                ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, d_k, n_head, L, B), 0, 2, 1, 3));
            ggml_tensor* vh = ggml_cont(
                ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, d_k, n_head, L, B), 0, 2, 1, 3));

            ggml_tensor* scores = ggml_scale_inplace(ctx, ggml_mul_mat(ctx, kh, qh), scale);
            if (tc->has_tensor_by_name("pnc.in.mask")) {
                scores = ggml_add(ctx, scores, tc->get_tensor_by_name("pnc.in.mask").tensor);
            }
            ggml_tensor* attn = ggml_soft_max_inplace(ctx, scores);
            auto vtk = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3));
            auto ctxh = ggml_mul_mat(ctx, vtk, attn);
            ggml_tensor* o = ggml_cont(ctx, ggml_permute(ctx, ctxh, 0, 2, 1, 3));
            o = ggml_reshape_3d(ctx, o, n_head * d_k, L, B);
            o = sub(l.attn_out.get(), session, tc, o);
            x = sub(l.attn_norm.get(), session, tc, ggml_add(ctx, x, o));

            ggml_tensor* h = sub(l.ffn_up.get(), session, tc, x);
            h = ggml_gelu_inplace(ctx, h);
            h = sub(l.ffn_down.get(), session, tc, h);
            x = sub(l.ffn_norm.get(), session, tc, ggml_add(ctx, x, h));
        }

        auto compact_argmax = [&](ggml_tensor* logits) {
            const int64_t C = logits->ne[0];
            const int64_t L = logits->ne[1];
            const int64_t B = logits->ne[2];
            std::vector<ggml_tensor*> items;
            items.reserve(static_cast<size_t>(B));
            for (int64_t b = 0; b < B; ++b) {
                auto item = ggml_view_3d(
                    ctx, logits, C, L, 1, logits->nb[1], logits->nb[2],
                    static_cast<size_t>(b) * logits->nb[2]);
                item = ggml_reshape_2d(ctx, ggml_cont(ctx, item), C, L);
                auto ids_b = ggml_reshape_2d(ctx, ggml_argmax(ctx, item), L, 1);
                items.push_back(ids_b);
            }
            return items;
        };
        // Keep compact I32 reductions per item because this view/argmax layout
        // is not safe to concatenate on every backend. The transformer work
        // remains batched in the shared graph.
        const auto punct = compact_argmax(sub(punct_head_.get(), session, tc, x));
        const auto capit = compact_argmax(sub(capit_head_.get(), session, tc, x));
        rt::TensorBag out;
        for (auto* t : punct) out.add_tensor(rt::ggml_bf_tensor(t, buft));
        for (auto* t : capit) out.add_tensor(rt::ggml_bf_tensor(t, buft));
        return out;
    }

   private:
    struct Layer {
        std::unique_ptr<rt::Linear> q, k, v, attn_out, ffn_up, ffn_down;
        std::unique_ptr<rt::LayerNorm> attn_norm, ffn_norm;
    };

    // Run a single-in/single-out submodule (Linear/LayerNorm). Those modules
    // take their compute context from their own weight's buffer type, so the
    // input bag's buft is unused (passed null).
    ggml_tensor* sub(rt::Module* m, rt::Session* s, rt::TensorContainer* tc, ggml_tensor* in) {
        rt::TensorBag bag;
        bag.add_tensor(rt::ggml_bf_tensor(in, nullptr));
        return m->build_graph(s, bag, tc).get_tensor(0).tensor;
    }

    PncConfig cfg_;
    std::unique_ptr<rt::LayerNorm> embd_norm_;
    std::vector<Layer> layers_;
    std::unique_ptr<rt::Linear> punct_head_, capit_head_;
};

}  // namespace

class PncModel::PncBatcher {
   public:
    struct Result {
        std::vector<int> punct, capit;
    };
    PncBatcher(PncModel* model, const BatchingConfig& cfg)
        : model_(model), queue_(cfg, [this](const int&, std::vector<std::vector<int32_t>>&& req) {
              const int B = static_cast<int>(req.size());
              int n = 0;
              for (const auto& item : req) n = std::max(n, static_cast<int>(item.size()));
              std::vector<int32_t> ids(static_cast<size_t>(n) * B, model_->cfg_.pad_id);
              std::vector<int32_t> pos(static_cast<size_t>(n) * B);
              std::vector<int32_t> typ(static_cast<size_t>(n) * B, 0);
              std::vector<float> mask(static_cast<size_t>(n) * B, 0.0f);
              for (int b = 0; b < B; ++b) {
                  std::copy(req[b].begin(), req[b].end(), ids.begin() + static_cast<size_t>(b) * n);
                  for (int i = 0; i < n; ++i) {
                      pos[static_cast<size_t>(b) * n + i] = i;
                      if (i >= static_cast<int>(req[b].size()))
                          mask[static_cast<size_t>(b) * n + i] = -1e9f;
                  }
              }
              std::vector<std::vector<int32_t>> pbuf(
                  static_cast<size_t>(B), std::vector<int32_t>(static_cast<size_t>(n)));
              std::vector<std::vector<int32_t>> cbuf(
                  static_cast<size_t>(B), std::vector<int32_t>(static_cast<size_t>(n)));
              std::vector<rt::Session::Output> outputs;
              outputs.reserve(static_cast<size_t>(2 * B));
              for (int b = 0; b < B; ++b)
                  outputs.push_back({b, "", pbuf[b].data(), pbuf[b].size() * sizeof(int32_t)});
              for (int b = 0; b < B; ++b)
                  outputs.push_back({B + b, "", cbuf[b].data(), cbuf[b].size() * sizeof(int32_t)});
              std::vector<rt::Session::Input> inputs = {
                  {"pnc.in.ids", GGML_TYPE_I32, ids.data(), {n, B}},
                  {"pnc.in.pos", GGML_TYPE_I32, pos.data(), {n, B}},
                  {"pnc.in.typ", GGML_TYPE_I32, typ.data(), {n, B}}};
              if (B > 1)
                  inputs.push_back({"pnc.in.mask", GGML_TYPE_F32, mask.data(), {n, 1, 1, B}});
              model_->session_->run(inputs, outputs);
              std::vector<Result> result(static_cast<size_t>(B));
              for (int b = 0; b < B; ++b) {
                  const size_t item_size = req[b].size();
                  result[b].punct.assign(pbuf[b].begin(), pbuf[b].begin() + item_size);
                  result[b].capit.assign(cbuf[b].begin(), cbuf[b].begin() + item_size);
              }
              return result;
          }) {}
    Result run(const int32_t* ids, int n) {
        constexpr int bucket_size = 16;
        const int bucket = ((n + bucket_size - 1) / bucket_size) * bucket_size;
        return queue_.run(bucket, std::vector<int32_t>(ids, ids + n));
    }
    BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    PncModel* model_;
    MicroBatcher<int, std::vector<int32_t>, Result> queue_;
};

PncModel::PncModel(
    rt::BackendManager& bm, const std::string& gguf_path, const BatchingConfig& batching)
    : bm_(&bm) {
    loader_ = std::make_unique<rt::GGUFLoader>(gguf_path);
    cfg_.hidden = loader_->get_u32("pnc.hidden_size", 768);
    cfg_.n_layers = loader_->get_u32("pnc.n_layers", 12);
    cfg_.n_heads = loader_->get_u32("pnc.n_heads", 12);
    cfg_.intermediate = loader_->get_u32("pnc.intermediate_size", 3072);
    cfg_.max_position = loader_->get_u32("pnc.max_position_embeddings", 512);
    cfg_.max_seq_length = loader_->get_u32("pnc.max_seq_length", 128);
    cfg_.type_vocab_size = loader_->get_u32("pnc.type_vocab_size", 2);
    cfg_.cls_id = loader_->get_u32("pnc.tokenizer.cls_id", 101);
    cfg_.sep_id = loader_->get_u32("pnc.tokenizer.sep_id", 102);
    cfg_.pad_id = loader_->get_u32("pnc.tokenizer.pad_id", 0);
    cfg_.unk_id = loader_->get_u32("pnc.tokenizer.unk_id", 100);
    cfg_.punct_labels = loader_->get_str_array("pnc.punct.labels");
    cfg_.capit_labels = loader_->get_str_array("pnc.capit.labels");
    cfg_.vocab = loader_->get_str_array("pnc.tokenizer.vocab");
    module_ = std::make_unique<PncBertModule>(cfg_);
    session_ = std::make_unique<rt::Session>(*bm_, module_.get(), loader_.get());
    // Sequence length and batch size both contribute shape variants. Retain the
    // common B<=32 working set but do not scale cache residency without bound.
    session_->set_run_cache_capacity(
        static_cast<size_t>(std::max(16, std::min(32, batching.max_batch_size * 2))));
    session_->setup();
    batcher_ = std::make_unique<PncBatcher>(this, batching);
}

PncModel::~PncModel() = default;

void
PncModel::infer(const int32_t* ids, int n, std::vector<int>& punct, std::vector<int>& capit) {
    auto result = batcher_->run(ids, n);
    punct = std::move(result.punct);
    capit = std::move(result.capit);
}

BatchMetrics
PncModel::batch_metrics() const {
    return batcher_->metrics();
}

}  // namespace nemo_speech::asr::pnc
