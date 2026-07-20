// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "cache_aware_encoder.h"  // CacheAwareEncoder (cache-aware streaming subsystem)
#include "nvtx_utils.h"
#include "rnnt_modules.h"  // RnntPredictorModule / RnntJointModule / PromptFusionModule
#include "runtime.h"

namespace nemo_speech::asr {

// CTCEncoderClassifier: ggml Module wrapping (encoder + head) so a single Session
// can build/run one graph through both.
class CtcModel::CTCEncoderClassifier : public ggml_runtime::Module {
   public:
    CTCEncoderClassifier(FastConformerEncoder* enc, CtcHeadModule* head)
        : encoder_(enc), head_(head) {}

    void define_tensors(ggml_runtime::Session* s) override {
        encoder_->define_tensors(s);
        head_->define_tensors(s);
    }
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag input,
        ggml_runtime::TensorContainer* tc) override {
        const std::string input_name = input.get_tensor(0).tensor->name;
        auto enc_out = encoder_->build_graph(s, input, tc);
        if (input_name == "input.features.greedy")
            return head_->build_greedy_graph(s, enc_out, tc);
        return head_->build_graph(s, enc_out, tc);
    }
    void set_data(ggml_runtime::Session* s) override {
        encoder_->set_data(s);
        head_->set_data(s);
    }

   private:
    FastConformerEncoder* encoder_;
    CtcHeadModule* head_;
};

namespace {

std::string
rnnt_h_state_name(int bank, int layer) {
    return "rnnt.state.h" + std::to_string(bank) + "." + std::to_string(layer);
}
std::string
rnnt_c_state_name(int bank, int layer) {
    return "rnnt.state.c" + std::to_string(bank) + "." + std::to_string(layer);
}
constexpr const char* kRnntPredProjectionState = "rnnt.state.pred_projection";

}  // namespace

// The RNNT encoder tail lives in the same Session as the cache-aware encoder:
// raw encoder activations flow directly through optional prompt fusion and
// joint.enc without being staged through host memory.
class RnntModel::RnntEncoderTail : public ggml_runtime::Module {
   public:
    RnntEncoderTail(RnntJointModule* joint, PromptFusionModule* prompt)
        : joint_(joint), prompt_(prompt) {}

    void define_tensors(ggml_runtime::Session* s) override {
        joint_->define_encoder_tensors(s);
        if (prompt_)
            prompt_->define_tensors(s);
    }
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag input,
        ggml_runtime::TensorContainer* tc) override {
        ggml_runtime::ggml_bf_tensor enc = input.get_tensor(0);
        if (prompt_ && input.tensor_count() == 2) {
            ggml_runtime::TensorBag prompt_in;
            prompt_in.add_tensor(enc);
            prompt_in.add_tensor(input.get_tensor(1));
            enc = prompt_->build_graph(s, prompt_in, tc).get_tensor(0);
        }
        return joint_->build_encoder_projection(s, enc, tc);
    }
    void set_data(ggml_runtime::Session* s) override {
        joint_->set_encoder_data(s);
        if (prompt_)
            prompt_->set_data(s);
    }

   private:
    RnntJointModule* joint_;
    PromptFusionModule* prompt_;
};

// Full-utterance encoder root: non-cache-aware FastConformer followed by the
// same prompt fusion + joint encoder projection used by streaming. It is kept
// in a separate lazy Session because cache-aware and offline graphs have
// different persistent-state and attention semantics.
class RnntModel::OfflineEncoderRoot : public ggml_runtime::Module {
   public:
    OfflineEncoderRoot(FastConformerEncoder* encoder, RnntEncoderTail* tail)
        : encoder_(encoder), tail_(tail) {}

    void define_tensors(ggml_runtime::Session* s) override {
        encoder_->define_tensors(s);
        tail_->define_tensors(s);
    }
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag input,
        ggml_runtime::TensorContainer* tc) override {
        auto enc = encoder_->build_graph(s, input, tc);
        ggml_runtime::TensorBag tail_input;
        tail_input.add_tensor(enc.get_tensor(0));
        if (input.tensor_count() > 1)
            tail_input.add_tensor(input.get_tensor(1));
        return tail_->build_graph(s, tail_input, tc);
    }
    void set_data(ggml_runtime::Session* s) override {
        encoder_->set_data(s);
        tail_->set_data(s);
    }

   private:
    FastConformerEncoder* encoder_;
    RnntEncoderTail* tail_;
};

class RnntModel::OfflineEncoderBatcher {
   public:
    struct Request {
        std::vector<float> features;
        int prompt_index = -1;
    };
    struct Result {
        std::vector<float> enc;
        int T = 0;
    };

    OfflineEncoderBatcher(RnntModel* model, const BatchingConfig& cfg)
        : model_(model), queue_(cfg, [this](const int& frames, std::vector<Request>&& requests) {
              const int B = static_cast<int>(requests.size());
              const int F = model_->enc_cfg_.feat_in;
              const int J = model_->rnnt_cfg_.joint_dim;
              const int Tout = model_->enc_cfg_.subsample_time_length(frames);
              const size_t feature_item = static_cast<size_t>(F) * frames;
              std::vector<float> packed(feature_item * B);
              for (int b = 0; b < B; ++b) {
                  if (requests[b].features.size() != feature_item)
                      throw std::runtime_error("offline transducer feature shape mismatch");
                  std::copy(
                      requests[b].features.begin(), requests[b].features.end(),
                      packed.begin() + static_cast<size_t>(b) * feature_item);
              }

              std::vector<float> prompts;
              std::vector<ggml_runtime::Session::Input> inputs = {
                  {"input.features", GGML_TYPE_F32, packed.data(), {F, frames, 1, B}}};
              if (model_->prompt_fusion_) {
                  const int P = model_->num_prompts_;
                  prompts.assign(static_cast<size_t>(P) * Tout * B, 0.0f);
                  for (int b = 0; b < B; ++b) {
                      const int p = requests[b].prompt_index;
                      if (p < 0 || p >= P)
                          continue;
                      for (int t = 0; t < Tout; ++t) {
                          prompts[(static_cast<size_t>(b) * Tout + t) * P + p] = 1.0f;
                      }
                  }
                  inputs.push_back(
                      {"encoder.offline.prompt", GGML_TYPE_F32, prompts.data(), {P, Tout, B}});
              }

              std::vector<float> output(static_cast<size_t>(J) * Tout * B);
              std::vector<ggml_runtime::Session::Output> outputs(1);
              outputs[0].index = 0;
              outputs[0].host_buffer = output.data();
              outputs[0].nbytes = output.size() * sizeof(float);
              model_->offline_encoder_session_->run(inputs, outputs);
              const int out_j = static_cast<int>(outputs[0].out_shape[0]);
              const int out_t = static_cast<int>(outputs[0].out_shape[1]);
              const int out_b = static_cast<int>(outputs[0].out_shape[2]);
              if (out_j != J || out_b != B)
                  throw std::runtime_error("offline transducer graph lost its batch dimension");
              const size_t output_item = static_cast<size_t>(J) * out_t;
              std::vector<Result> results(static_cast<size_t>(B));
              for (int b = 0; b < B; ++b) {
                  results[b].T = out_t;
                  results[b].enc.assign(
                      output.begin() + static_cast<size_t>(b) * output_item,
                      output.begin() + static_cast<size_t>(b + 1) * output_item);
              }
              return results;
          }) {}

    Result run(const float* features, int frames, int prompt_index) {
        Request request;
        request.features.assign(
            features, features + static_cast<size_t>(model_->enc_cfg_.feat_in) * frames);
        request.prompt_index = prompt_index;
        return queue_.run(frames, std::move(request));
    }
    BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    RnntModel* model_;
    MicroBatcher<int, Request, Result> queue_;
};

// One decoder weight-owning root with two independently cached graph stages:
//
//   rnnt.predict.{0,1} -> prediction LSTM -> joint.pred -> device state
//   rnnt.joint.enc     + device predictor projection -> joint tail -> argmax
//
// Session::run keys its graph cache by input names/shapes, so dispatching here
// avoids separate Sessions (and duplicate weights) while still making every
// hot shape a steady-state cache hit.
class RnntModel::RnntDecoderStages : public ggml_runtime::Module {
   public:
    RnntDecoderStages(RnntPredictorModule* pred, RnntJointModule* joint, int slots)
        : pred_(pred), joint_(joint), slots_(slots) {}

    void define_tensors(ggml_runtime::Session* s) override {
        pred_->define_tensors(s);
        joint_->define_decoder_tensors(s);
        const auto& cfg = pred_->cfg();
        for (int bank = 0; bank < 2; ++bank) {
            for (int l = 0; l < cfg.pred_num_layers; ++l) {
                s->model_tensor_container->create_tensor_2d(
                    rnnt_h_state_name(bank, l), GGML_TYPE_F32, cfg.pred_hidden, slots_);
                s->model_tensor_container->create_tensor_2d(
                    rnnt_c_state_name(bank, l), GGML_TYPE_F32, cfg.pred_hidden, slots_);
            }
        }
        s->model_tensor_container->create_tensor_2d(
            kRnntPredProjectionState, GGML_TYPE_F32, cfg.joint_dim, slots_);
    }
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* s, ggml_runtime::TensorBag input,
        ggml_runtime::TensorContainer* tc) override {
        const auto first = input.get_tensor(0);
        const std::string name = first.tensor->name;

        const bool fused_tdt = name == "rnnt.predict_joint.0" || name == "rnnt.predict_joint.1";
        if (name == "rnnt.predict.0" || name == "rnnt.predict.1" || fused_tdt) {
            const int active_bank = name.back() - '0';
            const int candidate_bank = active_bank ^ 1;
            auto slot_ids = input.get_tensor(1);
            ggml_runtime::TensorBag pred_in;
            const auto& cfg = pred_->cfg();
            pred_in.add_tensor(first);
            for (int i = 0; i < cfg.pred_num_layers; i++) {
                auto h = s->model_tensor_container->get_tensor_by_name(
                    rnnt_h_state_name(active_bank, i));
                auto c = s->model_tensor_container->get_tensor_by_name(
                    rnnt_c_state_name(active_bank, i));
                auto bf = tc->get_ctx_of_buffer_type(h.buft);
                pred_in.add_tensor(ggml_runtime::ggml_bf_tensor(
                    ggml_get_rows(bf.ctx, h.tensor, slot_ids.tensor), h.buft));
                pred_in.add_tensor(ggml_runtime::ggml_bf_tensor(
                    ggml_get_rows(bf.ctx, c.tensor, slot_ids.tensor), c.buft));
            }
            auto pred_out = pred_->build_graph(s, pred_in, tc);
            auto pred_proj =
                joint_->build_predictor_projection(s, pred_out.get_tensor(0), tc).get_tensor(0);

            auto bf = tc->get_ctx_of_buffer_type(pred_proj.buft);
            ggml_runtime::TensorBag state_out;
            auto pred_state =
                s->model_tensor_container->get_tensor_by_name(kRnntPredProjectionState);
            state_out.add_tensor(ggml_runtime::ggml_bf_tensor(
                ggml_set_rows(
                    bf.ctx, pred_state.tensor, ggml_cont(bf.ctx, pred_proj.tensor),
                    slot_ids.tensor),
                pred_proj.buft));
            for (int i = 0; i < cfg.pred_num_layers; i++) {
                auto h_dst = s->model_tensor_container->get_tensor_by_name(
                    rnnt_h_state_name(candidate_bank, i));
                auto c_dst = s->model_tensor_container->get_tensor_by_name(
                    rnnt_c_state_name(candidate_bank, i));
                state_out.add_tensor(ggml_runtime::ggml_bf_tensor(
                    ggml_set_rows(
                        bf.ctx, h_dst.tensor,
                        ggml_cont(bf.ctx, pred_out.get_tensor(1 + 2 * i).tensor), slot_ids.tensor),
                    pred_out.get_tensor(1 + 2 * i).buft));
                state_out.add_tensor(ggml_runtime::ggml_bf_tensor(
                    ggml_set_rows(
                        bf.ctx, c_dst.tensor,
                        ggml_cont(bf.ctx, pred_out.get_tensor(1 + 2 * i + 1).tensor),
                        slot_ids.tensor),
                    pred_out.get_tensor(1 + 2 * i + 1).buft));
            }
            if (!fused_tdt)
                return state_out;

            const int64_t B = slot_ids.tensor->ne[0];
            auto pred_for_joint = ggml_runtime::ggml_bf_tensor(
                ggml_reshape_3d(bf.ctx, pred_proj.tensor, pred_proj.tensor->ne[0], 1, B),
                pred_proj.buft);
            auto joint_out = joint_->build_joint_tail(
                s, input.get_tensor(2), pred_for_joint, tc, /*argmax_only=*/true);
            ggml_runtime::TensorBag out;
            for (size_t i = 0; i < joint_out.tensor_count(); ++i)
                out.add_tensor(joint_out.get_tensor(i));
            // Keep the state writes as graph outputs even though the caller
            // only copies the compact token/duration outputs.
            for (size_t i = 0; i < state_out.tensor_count(); ++i)
                out.add_tensor(state_out.get_tensor(i));
            return out;
        }

        if (name == "rnnt.joint.enc") {
            auto pred_state =
                s->model_tensor_container->get_tensor_by_name(kRnntPredProjectionState);
            auto slot_ids = input.get_tensor(1);
            auto bf = tc->get_ctx_of_buffer_type(pred_state.buft);
            auto pred = ggml_get_rows(bf.ctx, pred_state.tensor, slot_ids.tensor);
            pred =
                ggml_reshape_3d(bf.ctx, pred, pred_state.tensor->ne[0], 1, slot_ids.tensor->ne[0]);
            ggml_runtime::ggml_bf_tensor pred_bf(pred, pred_state.buft);
            if (input.tensor_count() > 2) {
                auto bias = input.get_tensor(2);
                return joint_->build_joint_tail(s, first, pred_bf, tc, /*argmax_only=*/true, &bias);
            }
            return joint_->build_joint_tail(s, first, pred_bf, tc, /*argmax_only=*/true);
        }

        throw std::runtime_error("RnntDecoderStages: unknown input signature '" + name + "'");
    }
    void set_data(ggml_runtime::Session* s) override {
        pred_->set_data(s);
        joint_->set_decoder_data(s);
        const auto& cfg = pred_->cfg();
        for (int bank = 0; bank < 2; ++bank)
            for (int l = 0; l < cfg.pred_num_layers; ++l)
                for (const auto& name : {rnnt_h_state_name(bank, l), rnnt_c_state_name(bank, l)}) {
                    auto t = s->model_tensor_container->get_tensor_by_name(name);
                    ggml_backend_tensor_memset(t.tensor, 0, 0, ggml_nbytes(t.tensor));
                }
        auto p = s->model_tensor_container->get_tensor_by_name(kRnntPredProjectionState);
        ggml_backend_tensor_memset(p.tensor, 0, 0, ggml_nbytes(p.tensor));
    }

   private:
    RnntPredictorModule* pred_;
    RnntJointModule* joint_;
    int slots_;
};

class RnntModel::RnntDecoderState : public RnntStreamState {
   public:
    RnntDecoderState(RnntModel* owner, int slot) : owner(owner), slot(slot) {}
    ~RnntDecoderState() override {
        if (owner && slot >= 0)
            owner->release_decoder_slot(slot);
    }
    RnntModel* owner;
    int slot;
};

class RnntModel::DecoderBatchers {
   public:
    struct PredictRequest {
        int slot;
        int32_t token;
    };
    struct JointKey {
        int T;
        bool has_bias;
        bool operator==(const JointKey& o) const { return T == o.T && has_bias == o.has_bias; }
    };
    struct JointRequest {
        int slot;
        std::vector<float> enc;
        std::vector<float> bias;
    };
    struct TdtFusedRequest {
        int slot;
        int32_t token;
        std::vector<float> enc;
    };
    struct JointResult {
        std::vector<int32_t> tokens;
        std::vector<int32_t> durations;
    };

    DecoderBatchers(RnntModel* owner, const BatchingConfig& cfg)
        : owner_(owner),
          predictor_(
              cfg,
              [this](const int& bank, std::vector<PredictRequest>&& req) {
                  const ggml_nvtx::range nvtx("asr.predictor.batch");
                  const int B = static_cast<int>(req.size());
                  std::vector<int32_t> tokens(static_cast<size_t>(B));
                  std::vector<int32_t> slots(static_cast<size_t>(B));
                  for (int b = 0; b < B; ++b) {
                      tokens[b] = req[b].token;
                      slots[b] = req[b].slot;
                  }
                  std::vector<ggml_runtime::Session::Input> inputs = {
                      {"rnnt.predict." + std::to_string(bank), GGML_TYPE_I32, tokens.data(), {B}},
                      {"rnnt.slot_ids", GGML_TYPE_I32, slots.data(), {B}}};
                  std::vector<ggml_runtime::Session::Output> outputs;
                  owner_->decoder_session_->run(inputs, outputs);
                  return std::vector<uint8_t>(static_cast<size_t>(B), 1);
              }),
          joint_(
              cfg,
              [this](const JointKey& key, std::vector<JointRequest>&& req) {
                  const ggml_nvtx::range nvtx("asr.joint.batch");
                  const int B = static_cast<int>(req.size());
                  const int J = owner_->rnnt_cfg_.joint_dim;
                  const int V = owner_->rnnt_cfg_.vocab_size;
                  const size_t enc_item = static_cast<size_t>(J) * key.T;
                  std::vector<float> enc(enc_item * B);
                  std::vector<int32_t> slots(static_cast<size_t>(B));
                  std::vector<float> bias;
                  if (key.has_bias)
                      bias.resize(static_cast<size_t>(V) * B);
                  for (int b = 0; b < B; ++b) {
                      std::copy(req[b].enc.begin(), req[b].enc.end(), enc.begin() + b * enc_item);
                      slots[b] = req[b].slot;
                      if (key.has_bias)
                          std::copy(
                              req[b].bias.begin(), req[b].bias.end(),
                              bias.begin() + static_cast<size_t>(b) * V);
                  }
                  std::vector<ggml_runtime::Session::Input> inputs = {
                      {"rnnt.joint.enc", GGML_TYPE_F32, enc.data(), {J, key.T * B}},
                      {"rnnt.slot_ids", GGML_TYPE_I32, slots.data(), {B}}};
                  if (key.has_bias)
                      inputs.push_back({"rnnt.joint.bias", GGML_TYPE_F32, bias.data(), {V, 1, B}});
                  std::vector<int32_t> packed_tokens(static_cast<size_t>(key.T) * B);
                  std::vector<int32_t> packed_durations;
                  std::vector<ggml_runtime::Session::Output> outputs(1);
                  outputs[0].index = 0;
                  outputs[0].host_buffer = packed_tokens.data();
                  outputs[0].nbytes = packed_tokens.size() * sizeof(int32_t);
                  if (owner_->rnnt_cfg_.is_tdt()) {
                      packed_durations.resize(static_cast<size_t>(key.T) * B);
                      outputs.resize(2);
                      outputs[1].index = 1;
                      outputs[1].host_buffer = packed_durations.data();
                      outputs[1].nbytes = packed_durations.size() * sizeof(int32_t);
                  }
                  owner_->decoder_session_->run(inputs, outputs);
                  std::vector<JointResult> result(static_cast<size_t>(B));
                  for (int b = 0; b < B; ++b) {
                      const auto token_begin =
                          packed_tokens.begin() + static_cast<size_t>(b) * key.T;
                      result[b].tokens.assign(token_begin, token_begin + key.T);
                      if (owner_->rnnt_cfg_.is_tdt()) {
                          const auto duration_begin =
                              packed_durations.begin() + static_cast<size_t>(b) * key.T;
                          result[b].durations.assign(duration_begin, duration_begin + key.T);
                      }
                  }
                  return result;
              }),
          fused_tdt_(cfg, [this](const int& bank, std::vector<TdtFusedRequest>&& req) {
              const int B = static_cast<int>(req.size());
              const int J = owner_->rnnt_cfg_.joint_dim;
              std::vector<int32_t> tokens(static_cast<size_t>(B));
              std::vector<int32_t> slots(static_cast<size_t>(B));
              std::vector<float> enc(static_cast<size_t>(J) * B);
              for (int b = 0; b < B; ++b) {
                  tokens[b] = req[b].token;
                  slots[b] = req[b].slot;
                  std::copy(
                      req[b].enc.begin(), req[b].enc.end(),
                      enc.begin() + static_cast<size_t>(b) * J);
              }
              std::vector<ggml_runtime::Session::Input> inputs = {
                  {"rnnt.predict_joint." + std::to_string(bank), GGML_TYPE_I32, tokens.data(), {B}},
                  {"rnnt.slot_ids", GGML_TYPE_I32, slots.data(), {B}},
                  {"rnnt.joint.enc", GGML_TYPE_F32, enc.data(), {J, B}}};
              std::vector<int32_t> packed_tokens(static_cast<size_t>(B));
              std::vector<int32_t> packed_durations(static_cast<size_t>(B));
              std::vector<ggml_runtime::Session::Output> outputs(2);
              outputs[0].index = 0;
              outputs[0].host_buffer = packed_tokens.data();
              outputs[0].nbytes = packed_tokens.size() * sizeof(int32_t);
              outputs[1].index = 1;
              outputs[1].host_buffer = packed_durations.data();
              outputs[1].nbytes = packed_durations.size() * sizeof(int32_t);
              owner_->decoder_session_->run(inputs, outputs);
              std::vector<JointResult> result(static_cast<size_t>(B));
              for (int b = 0; b < B; ++b) {
                  result[b].tokens = {packed_tokens[static_cast<size_t>(b)]};
                  result[b].durations = {packed_durations[static_cast<size_t>(b)]};
              }
              return result;
          }) {}

    void predict(int slot, int32_t token, int bank, int active_decodes) {
        if (active_decodes == 1)
            (void)predictor_.run_inline(bank, {slot, token});
        else
            (void)predictor_.run(bank, {slot, token}, active_decodes);
    }
    JointResult joint(
        int slot, const float* enc, int T, const float* bias, int vocab_size, int active_decodes) {
        JointRequest req;
        req.slot = slot;
        req.enc.assign(enc, enc + static_cast<size_t>(owner_->rnnt_cfg_.joint_dim) * T);
        if (bias)
            req.bias.assign(bias, bias + vocab_size);
        if (active_decodes == 1)
            return joint_.run_inline({T, bias != nullptr}, std::move(req));
        return joint_.run({T, bias != nullptr}, std::move(req), active_decodes);
    }
    JointResult predict_joint_tdt(
        int slot, int32_t token, int bank, const float* enc, int active_decodes) {
        TdtFusedRequest req;
        req.slot = slot;
        req.token = token;
        req.enc.assign(enc, enc + owner_->rnnt_cfg_.joint_dim);
        if (active_decodes == 1)
            return fused_tdt_.run_inline(bank, std::move(req));
        return fused_tdt_.run(bank, std::move(req), active_decodes);
    }
    static BatchMetrics add(BatchMetrics a, const BatchMetrics& b) {
        a.batches += b.batches;
        a.items += b.items;
        a.singleton_batches += b.singleton_batches;
        a.max_observed_batch = std::max(a.max_observed_batch, b.max_observed_batch);
        return a;
    }
    BatchMetrics predictor_metrics() const {
        return add(predictor_.metrics(), fused_tdt_.metrics());
    }
    BatchMetrics joint_metrics() const { return add(joint_.metrics(), fused_tdt_.metrics()); }

   private:
    RnntModel* owner_;
    MicroBatcher<int, PredictRequest, uint8_t> predictor_;
    MicroBatcher<JointKey, JointRequest, JointResult> joint_;
    MicroBatcher<int, TdtFusedRequest, JointResult> fused_tdt_;
};

namespace {

EncoderConfig
load_encoder_cfg(const ggml_runtime::GGUFLoader& loader, const std::string& A) {
    EncoderConfig cfg;
    cfg.d_model = loader.get_u32(A + ".encoder.d_model", cfg.d_model);
    cfg.n_layers = loader.get_u32(A + ".encoder.n_layers", cfg.n_layers);
    cfg.n_heads = loader.get_u32(A + ".encoder.n_heads", cfg.n_heads);
    cfg.d_ff = loader.get_u32(A + ".encoder.d_ff", cfg.d_ff);
    cfg.conv_kernel_size = loader.get_u32(A + ".encoder.conv_kernel_size", cfg.conv_kernel_size);
    cfg.subsampling_factor =
        loader.get_u32(A + ".encoder.subsampling_factor", cfg.subsampling_factor);
    cfg.subsampling_conv_channels =
        loader.get_u32(A + ".encoder.subsampling_conv_channels", cfg.subsampling_conv_channels);
    cfg.feat_in = loader.get_u32(A + ".encoder.feat_in", cfg.feat_in);
    cfg.pos_emb_max_len = loader.get_u32(A + ".encoder.pos_emb_max_len", cfg.pos_emb_max_len);
    cfg.xscaling = loader.get_bool(A + ".encoder.xscaling", cfg.xscaling);
    cfg.use_bias = loader.get_bool(A + ".encoder.use_bias", cfg.use_bias);

    // cache_supported is a capability, not a run mode. Consumers select
    // cache_mode because offline execution has no SessionState to bind.
    const std::string conv_norm = loader.get_str(A + ".encoder.conv_norm", "");
    if (conv_norm == "layer_norm")
        cfg.conv_norm = ConvNorm::LayerNorm;
    else if (conv_norm == "batch_norm")
        cfg.conv_norm = ConvNorm::BatchNorm;
    const std::string conv_ctx = loader.get_str(A + ".encoder.conv_context", "");
    if (conv_ctx == "causal")
        cfg.conv_context = ConvContext::Causal;
    else if (conv_ctx == "symmetric")
        cfg.conv_context = ConvContext::Symmetric;
    cfg.chunked_limited_attention =
        loader.get_str(A + ".encoder.att_context_style", "regular") == "chunked_limited";
    cfg.cache_supported = loader.get_bool(A + ".encoder.cache_supported", false);
    cfg.cache_left_ctx = loader.get_u32(A + ".encoder.train_left_ctx", cfg.cache_left_ctx);
    cfg.cache_right_ctx = loader.get_u32(A + ".encoder.train_right_ctx", cfg.cache_right_ctx);

    // Preserve the model's trained offline attention window. -1 is unlimited
    // and also supports GGUFs that predate these fields.
    cfg.offline_left_ctx = loader.get_i32(A + ".encoder.offline_left_ctx", cfg.offline_left_ctx);
    cfg.offline_right_ctx = loader.get_i32(A + ".encoder.offline_right_ctx", cfg.offline_right_ctx);
    return cfg;
}

CtcConfig
load_ctc_cfg(
    const ggml_runtime::GGUFLoader& loader, const std::string& A, const EncoderConfig& enc) {
    CtcConfig cfg;
    cfg.d_model = enc.d_model;
    // Our converter emits `asr.ctc.*`; legacy CTC GGUFs use
    // `parakeet-ctc.decoder.*` - accept both.
    const std::string head_ns = (A == "asr") ? "ctc" : "decoder";
    cfg.num_classes = loader.get_u32(A + "." + head_ns + ".num_classes", cfg.num_classes);
    cfg.blank_id = loader.get_u32(A + "." + head_ns + ".blank_id", cfg.blank_id);
    return cfg;
}

RnntConfig
load_rnnt_cfg(const ggml_runtime::GGUFLoader& loader, const EncoderConfig& enc) {
    RnntConfig cfg;
    cfg.d_model = enc.d_model;
    cfg.vocab_size = loader.get_u32("asr.rnnt.vocab_size", cfg.vocab_size);
    cfg.blank_id = loader.get_u32("asr.rnnt.blank_id", cfg.blank_id);
    cfg.pred_embed_dim = loader.get_u32("asr.rnnt.pred_embed_dim", cfg.pred_embed_dim);
    cfg.pred_hidden = loader.get_u32("asr.rnnt.pred_hidden", cfg.pred_hidden);
    cfg.pred_num_layers = loader.get_u32("asr.rnnt.pred_num_layers", cfg.pred_num_layers);
    cfg.joint_dim = loader.get_u32("asr.rnnt.joint_dim", cfg.joint_dim);
    cfg.max_symbols_per_step =
        loader.get_u32("asr.rnnt.max_symbols_per_step", cfg.max_symbols_per_step);
    cfg.durations = loader.get_i32_array("asr.tdt.durations");
    return cfg;
}

MelSpecConfig
load_fe_cfg(const ggml_runtime::GGUFLoader& loader, const std::string& A) {
    MelSpecConfig fe;
    fe.sample_rate = loader.get_u32(A + ".preprocessor.sample_rate", fe.sample_rate);
    fe.window_size = loader.get_f32(A + ".preprocessor.window_size", fe.window_size);
    fe.window_stride = loader.get_f32(A + ".preprocessor.window_stride", fe.window_stride);
    fe.n_fft = loader.get_u32(A + ".preprocessor.n_fft", fe.n_fft);
    fe.n_mels = loader.get_u32(A + ".preprocessor.features", fe.n_mels);
    fe.preemph = loader.get_f32(A + ".preprocessor.preemph", fe.preemph);
    fe.normalize_per_feature =
        loader.get_str(A + ".preprocessor.normalize", "per_feature") == "per_feature";
    return fe;
}

}  // namespace

AsrModel::AsrModel(ggml_runtime::BackendManager& bm, Common&& c, const BatchingConfig& batching)
    : bm_(&bm), loader_(std::move(c.loader)), ns_(std::move(c.ns)),
      model_name_(std::move(c.model_name)), enc_cfg_(std::move(c.enc_cfg)),
      fe_cfg_(std::move(c.fe_cfg)), vocab_(std::move(c.vocab)) {
    // Use the GPU frontend by default; NEMO_SPEECH_STREAM_GPU_FE=0 disables it.
    static const bool stream_gpu_fe = [] {
        const char* e = std::getenv("NEMO_SPEECH_STREAM_GPU_FE");
        return e == nullptr || e[0] != '0';
    }();
    const bool gpu_fe = (batching.enabled && batching.max_batch_size > 1) || stream_gpu_fe;
    fe_ = std::make_unique<MelSpectrogramExtractor>(fe_cfg_, gpu_fe ? &bm : nullptr, batching);

    apply_model_mel_basis(*fe_);
}

void
AsrModel::apply_model_mel_basis(MelSpectrogramExtractor& extractor) {
    // Prefer the serialized Slaney-normalized filterbank used during training.
    // Older GGUFs fall back to the generated unit-peak filterbank.
    const std::string fb_name = "preprocessor.fb";
    if (loader_->has_tensor(fb_name)) {
        const int n_mels = fe_cfg_.n_mels;
        const int n_bins = fe_cfg_.n_fft / 2 + 1;
        const size_t want = static_cast<size_t>(n_mels) * n_bins * sizeof(float);
        const char* data = loader_->get_tensor_file_data(fb_name, want);
        extractor.set_mel_basis(reinterpret_cast<const float*>(data), n_mels, n_bins);
    }
}

AsrModel::~AsrModel() = default;

std::unique_ptr<AsrModel>
AsrModel::load(
    ggml_runtime::BackendManager& bm, const std::string& model_path,
    const BatchingConfig& batching) {
    Common c;
    c.loader = std::make_unique<ggml_runtime::GGUFLoader>(model_path);
    const ggml_runtime::GGUFLoader& loader = *c.loader;

    // "asr" is the current namespace; legacy parakeet-ctc GGUFs remain
    // readable, while legacy RNNT layouts must be re-exported.
    const std::string arch = loader.get_str("general.architecture", "");
    HeadKind head = HeadKind::Ctc;
    if (arch == "asr") {
        c.ns = "asr";
        const std::string h = loader.get_str("asr.head_type", "ctc");
        if (h == "ctc") {
            head = HeadKind::Ctc;
        } else if (h == "rnnt") {
            head = HeadKind::Rnnt;
        } else if (h == "tdt") {
            head = HeadKind::Tdt;
        } else {
            throw std::runtime_error("Unknown asr.head_type='" + h + "'");
        }
        c.model_name = loader.get_str("general.name", "asr");
    } else if (arch == "parakeet-ctc" || arch.empty()) {
        head = HeadKind::Ctc;
        c.ns = "parakeet-ctc";
        c.model_name = c.ns + "-1.1b";
    } else if (arch == "nemo") {
        throw std::runtime_error(
            "Legacy RNNT GGUF detected. Re-export via "
            "convert_model.py to produce the supported "
            "conv-weight layout + asr.* metadata namespace.");
    } else {
        throw std::runtime_error(
            "Unknown general.architecture='" + arch +
            "'. Supported: 'asr' (preferred), 'parakeet-ctc'.");
    }

    c.enc_cfg = load_encoder_cfg(loader, c.ns);
    c.fe_cfg = load_fe_cfg(loader, c.ns);
    c.vocab = loader.get_str_array(c.ns + ".tokenizer.vocab");

    const char* head_name =
        head == HeadKind::Ctc ? "ctc" : (head == HeadKind::Tdt ? "tdt" : "rnnt");
    std::cerr << "[asr_model] arch=" << arch << " head=" << head_name
              << " d_model=" << c.enc_cfg.d_model << " n_layers=" << c.enc_cfg.n_layers
              << " n_heads=" << c.enc_cfg.n_heads << " d_ff=" << c.enc_cfg.d_ff
              << " k=" << c.enc_cfg.conv_kernel_size << " feat_in=" << c.enc_cfg.feat_in
              << " sr=" << c.fe_cfg.sample_rate << " vocab=" << c.vocab.size() << "\n";

    // `new` rather than make_unique: Common is a protected nested type, which
    // std::make_unique (in namespace std) cannot name for template deduction;
    // here in a member of AsrModel we have access.
    if (head == HeadKind::Ctc) {
        return std::unique_ptr<AsrModel>(new CtcModel(bm, std::move(c), batching));
    }
    return std::unique_ptr<AsrModel>(new RnntModel(bm, std::move(c), batching));
}

class CtcModel::CtcBatcher {
   public:
    struct Key {
        int n_frames = 0;
        bool greedy = false;
        bool operator==(const Key& other) const {
            return n_frames == other.n_frames && greedy == other.greedy;
        }
    };
    struct Request {
        std::vector<float> features;
    };
    struct Result {
        std::vector<float> log_probs;
        std::vector<int32_t> ids;
        std::vector<float> probs;
        int T = 0;
        int C = 0;
    };

    CtcBatcher(CtcModel* model, const BatchingConfig& cfg)
        : model_(model), queue_(cfg, [this](const Key& key, std::vector<Request>&& requests) {
              return execute(key, std::move(requests));
          }) {}

    Result run(bool greedy, const float* features, int n_frames) {
        Request request;
        request.features.assign(
            features, features + static_cast<size_t>(model_->enc_cfg_.feat_in) * n_frames);
        return queue_.run({n_frames, greedy}, std::move(request), current_batch_cohort_target());
    }

    BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    std::vector<Result> execute(const Key& key, std::vector<Request>&& requests) {
        const int B = static_cast<int>(requests.size());
        const int F = model_->enc_cfg_.feat_in;
        const int T = key.n_frames;
        std::vector<float> packed(static_cast<size_t>(F) * T * B);
        const size_t item_size = static_cast<size_t>(F) * T;
        for (int b = 0; b < B; ++b) {
            if (requests[b].features.size() != item_size)
                throw std::runtime_error("CTC batch item has an incompatible feature shape");
            std::copy(
                requests[b].features.begin(), requests[b].features.end(),
                packed.begin() + static_cast<size_t>(b) * item_size);
        }

        std::vector<Result> results(static_cast<size_t>(B));
        std::vector<ggml_runtime::Session::Output> outputs;
        if (key.greedy) {
            std::vector<int32_t> ids(static_cast<size_t>(T) * B);
            std::vector<float> probs(static_cast<size_t>(T) * B);
            outputs.push_back({0, "", ids.data(), ids.size() * sizeof(int32_t)});
            outputs.push_back({1, "", probs.data(), probs.size() * sizeof(float)});
            model_->session_->run(
                // Conv2D consumes [width, height, channels, batch].  The
                // subsampler converts its output to the encoder's [D,T,B].
                {{"input.features.greedy", GGML_TYPE_F32, packed.data(), {F, T, 1, B}}}, outputs);
            const int Tout = static_cast<int>(outputs[0].out_shape[0]);
            const int Bout = static_cast<int>(outputs[0].out_shape[1]);
            if (Bout != B)
                throw std::runtime_error("CTC greedy graph lost its batch dimension");
            for (int b = 0; b < B; ++b) {
                auto& r = results[static_cast<size_t>(b)];
                r.T = Tout;
                r.ids.assign(
                    ids.begin() + static_cast<size_t>(b) * Tout,
                    ids.begin() + static_cast<size_t>(b + 1) * Tout);
                r.probs.assign(
                    probs.begin() + static_cast<size_t>(b) * Tout,
                    probs.begin() + static_cast<size_t>(b + 1) * Tout);
            }
        } else {
            const int Ccap = model_->ctc_cfg_.num_classes + 1;
            std::vector<float> log_probs(static_cast<size_t>(Ccap) * T * B);
            outputs.push_back({0, "", log_probs.data(), log_probs.size() * sizeof(float)});
            model_->session_->run(
                {{"input.features", GGML_TYPE_F32, packed.data(), {F, T, 1, B}}}, outputs);
            const int C = static_cast<int>(outputs[0].out_shape[0]);
            const int Tout = static_cast<int>(outputs[0].out_shape[1]);
            const int Bout = static_cast<int>(outputs[0].out_shape[2]);
            if (Bout != B)
                throw std::runtime_error("CTC graph lost its batch dimension");
            const size_t out_item = static_cast<size_t>(C) * Tout;
            for (int b = 0; b < B; ++b) {
                auto& r = results[static_cast<size_t>(b)];
                r.T = Tout;
                r.C = C;
                r.log_probs.assign(
                    log_probs.begin() + static_cast<size_t>(b) * out_item,
                    log_probs.begin() + static_cast<size_t>(b + 1) * out_item);
            }
        }
        return results;
    }

    CtcModel* model_;
    MicroBatcher<Key, Request, Result> queue_;
};

CtcModel::CtcModel(ggml_runtime::BackendManager& bm, Common&& c, const BatchingConfig& batching)
    : AsrModel(bm, std::move(c), batching) {
    ctc_cfg_ = load_ctc_cfg(*loader(), ns_, enc_cfg_);
    std::cerr << "[asr_model] ctc classes=" << ctc_cfg_.num_classes << "\n";

    // Full utterances amortize the GPU frontend's launch overhead.
    offline_fe_ = std::make_unique<MelSpectrogramExtractor>(fe_cfg_, &bm, batching);
    apply_model_mel_basis(*offline_fe_);

    // CTC uses the full-context encoder even when the GGUF supports caching.
    EncoderConfig ec = enc_cfg_;
    ec.cache_mode = CacheMode::Disabled;
    encoder_ = std::make_unique<FastConformerEncoder>("encoder", ec);

    ctc_head_ = std::make_unique<CtcHeadModule>("ctc_head", ctc_cfg_);
    ctc_enc_classifier_ = std::make_unique<CTCEncoderClassifier>(encoder_.get(), ctc_head_.get());
    session_ = std::make_unique<ggml_runtime::Session>(
        backend_manager(), ctc_enc_classifier_.get(), loader());
    session_->set_weight_load_hook(planar_q8_weight_load_hook());
    // Buffered streaming visits multiple startup/tail lengths and batch sizes.
    // Keep the steady-state and edge-shape graphs resident instead of rebuilding
    // them as a wide-concurrency cohort moves through the window sequence.
    session_->set_run_cache_capacity(128);
    session_->setup();
    batcher_ = std::make_unique<CtcBatcher>(this, batching);
}

CtcModel::~CtcModel() = default;

void
CtcModel::infer_ctc(
    const float* audio, size_t n_samples, std::vector<float>& out_log_probs, int& T_out,
    int& n_classes) {
    std::vector<float> feats;
    int n_frames = 0;
    offline_fe_->compute(audio, n_samples, feats, n_frames);
    infer_ctc_from_mel(feats.data(), n_frames, out_log_probs, T_out, n_classes);
}

void
CtcModel::infer_ctc_from_mel(
    const float* feats, int n_frames, std::vector<float>& out_log_probs, int& T_out,
    int& n_classes) {
    if (n_frames == 0) {
        T_out = 0;
        n_classes = 0;
        out_log_probs.clear();
        return;
    }

    auto result = batcher_->run(/*greedy=*/false, feats, n_frames);
    out_log_probs = std::move(result.log_probs);
    T_out = result.T;
    n_classes = result.C;
}

void
CtcModel::infer_ctc_greedy_from_mel(
    const float* feats, int n_frames, std::vector<int32_t>& best_ids,
    std::vector<float>& best_probs, int& T_out) {
    if (n_frames == 0) {
        T_out = 0;
        best_ids.clear();
        best_probs.clear();
        return;
    }

    auto result = batcher_->run(/*greedy=*/true, feats, n_frames);
    best_ids = std::move(result.ids);
    best_probs = std::move(result.probs);
    T_out = result.T;
}

BatchMetrics
CtcModel::batch_metrics() const {
    return batcher_->metrics();
}

BatchMetrics
CtcModel::offline_frontend_batch_metrics() const {
    return offline_fe_->batch_metrics();
}

bool
CtcModel::offline_frontend_uses_gpu() const {
    return offline_fe_->uses_gpu();
}

std::vector<AsrModel::DiagSession>
CtcModel::diagnostic_sessions() const {
    std::vector<DiagSession> out;
    if (fe().diagnostic_session())
        out.push_back({"streaming frontend", fe().diagnostic_session()});
    if (offline_fe_->diagnostic_session())
        out.push_back({"offline frontend", offline_fe_->diagnostic_session()});
    out.push_back({"encoder+CTC", session_.get()});
    return out;
}

RnntModel::RnntModel(ggml_runtime::BackendManager& bm, Common&& c, const BatchingConfig& batching)
    : AsrModel(bm, std::move(c), batching), batching_cfg_(batching) {
    rnnt_cfg_ = load_rnnt_cfg(*loader(), enc_cfg_);
    std::cerr << "[asr_model] rnnt vocab=" << rnnt_cfg_.vocab_size
              << " blank=" << rnnt_cfg_.blank_id << " pred_hidden=" << rnnt_cfg_.pred_hidden
              << " joint_dim=" << rnnt_cfg_.joint_dim;
    if (rnnt_cfg_.is_tdt()) {
        std::cerr << " durations=";
        for (int duration : rnnt_cfg_.durations) std::cerr << duration << ',';
    }
    std::cerr << "\n";

    // Prompt fusion and joint.enc execute together once per encoder chunk.
    num_prompts_ = static_cast<int>(loader()->get_u32("asr.rnnt.num_prompts", 0));
    if (loader()->has_key("asr.rnnt.prompt_dictionary")) {
        for (const auto& entry : loader()->get_str_array("asr.rnnt.prompt_dictionary")) {
            const auto colon = entry.rfind(':');
            if (colon != std::string::npos) {
                prompt_dictionary_[entry.substr(0, colon)] =
                    std::atoi(entry.substr(colon + 1).c_str());
            }
        }
    }
    if (num_prompts_ > 0 && loader()->has_tensor("prompt_kernel.0.weight")) {
        std::cerr << "[asr_model] prompt fusion enabled: num_prompts=" << num_prompts_
                  << " languages=" << prompt_dictionary_.size() << "\n";
        prompt_fusion_ = std::make_unique<PromptFusionModule>(rnnt_cfg_.d_model, num_prompts_);
    } else {
        num_prompts_ = 0;  // no prompt_kernel in this GGUF -> disable fusion
    }

    // One Session owns predictor projection + joint-tail weights and the
    // per-stream predictor state. joint.enc and optional prompt fusion are
    // owned by the cache-aware encoder Session below so raw encoder activations
    // never cross the host boundary.
    rnnt_predictor_ = std::make_unique<RnntPredictorModule>("rnnt_predictor", rnnt_cfg_);
    rnnt_joint_ = std::make_unique<RnntJointModule>("rnnt_joint", rnnt_cfg_);
    decoder_arena_slots_ = std::max(1, batching.state_arena_slots);
    rnnt_encoder_tail_ = std::make_unique<RnntEncoderTail>(rnnt_joint_.get(), prompt_fusion_.get());
    rnnt_decoder_stages_ = std::make_unique<RnntDecoderStages>(
        rnnt_predictor_.get(), rnnt_joint_.get(), decoder_arena_slots_);
    decoder_session_ = std::make_unique<ggml_runtime::Session>(
        backend_manager(), rnnt_decoder_stages_.get(), loader());
    // Bound graph variants across predictor batch and joint time shapes.
    decoder_session_->set_run_cache_capacity(64);
    decoder_session_->setup();
    decoder_slots_used_.assign(static_cast<size_t>(decoder_arena_slots_), false);
    decoder_batchers_ = std::make_unique<DecoderBatchers>(this, batching);

    // The cache-aware streaming encoder Session is built lazily after the
    // runner selects right-context geometry.
    if (enc_cfg_.cache_supported) {
        cache_encoder_ = std::make_unique<CacheAwareEncoder>(
            backend_manager(), loader(), enc_cfg_, fe_cfg_.n_mels, rnnt_encoder_tail_.get(),
            rnnt_cfg_.joint_dim, batching);
    }
}

RnntModel::~RnntModel() = default;

void
RnntModel::ensure_offline_path() {
    std::lock_guard<std::mutex> lock(offline_init_mu_);
    if (offline_encoder_session_)
        return;

    offline_fe_ =
        std::make_unique<MelSpectrogramExtractor>(fe_cfg_, &backend_manager(), batching_cfg_);
    apply_model_mel_basis(*offline_fe_);

    EncoderConfig offline_cfg = enc_cfg_;
    offline_cfg.cache_mode = CacheMode::Disabled;
    // Older streaming GGUFs predate explicit offline context metadata. A
    // full-utterance pass should still respect the finite attention window the
    // cache-aware model was trained with, rather than silently becoming global.
    if (offline_cfg.cache_supported) {
        if (offline_cfg.offline_left_ctx < 0)
            offline_cfg.offline_left_ctx = offline_cfg.cache_left_ctx;
        if (offline_cfg.offline_right_ctx < 0)
            offline_cfg.offline_right_ctx = offline_cfg.cache_right_ctx;
    }
    offline_encoder_ = std::make_unique<FastConformerEncoder>("encoder", offline_cfg);
    offline_encoder_root_ =
        std::make_unique<OfflineEncoderRoot>(offline_encoder_.get(), rnnt_encoder_tail_.get());
    offline_encoder_session_ = std::make_unique<ggml_runtime::Session>(
        backend_manager(), offline_encoder_root_.get(), loader());
    offline_encoder_session_->set_weight_load_hook(planar_q8_weight_load_hook());
    offline_encoder_session_->set_run_cache_capacity(16);
    offline_encoder_session_->setup();
    offline_encoder_batcher_ = std::make_unique<OfflineEncoderBatcher>(this, batching_cfg_);
}

void
RnntModel::infer_offline(
    const float* audio, size_t n_samples, std::vector<float>& enc_out, int& T_enc,
    int prompt_index) {
    ensure_offline_path();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> features;
    int frames = 0;
    offline_fe_->compute(audio, n_samples, features, frames);
    const auto t1 = std::chrono::steady_clock::now();
    infer_offline_from_mel(features.data(), frames, enc_out, T_enc, prompt_index);
    if (std::getenv("NEMO_SPEECH_TIMING")) {
        const auto t2 = std::chrono::steady_clock::now();
        const auto elapsed_ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(
            stderr,
            "[timing] offline-transducer frames=%d enc_frames=%d fe=%.2f "
            "encoder+encproj=%.2f ms\n",
            frames, T_enc, elapsed_ms(t0, t1), elapsed_ms(t1, t2));
    }
}

void
RnntModel::infer_offline_from_mel(
    const float* feats, int n_frames, std::vector<float>& enc_out, int& T_enc, int prompt_index) {
    if (n_frames <= 0) {
        enc_out.clear();
        T_enc = 0;
        return;
    }
    ensure_offline_path();
    auto result = offline_encoder_batcher_->run(feats, n_frames, prompt_index);
    enc_out = std::move(result.enc);
    T_enc = result.T;
}

BatchMetrics
RnntModel::offline_encoder_batch_metrics() const {
    return offline_encoder_batcher_ ? offline_encoder_batcher_->metrics() : BatchMetrics{};
}

BatchMetrics
RnntModel::offline_frontend_batch_metrics() const {
    return offline_fe_ ? offline_fe_->batch_metrics() : BatchMetrics{};
}

bool
RnntModel::offline_frontend_uses_gpu() const {
    return offline_fe_ && offline_fe_->uses_gpu();
}

int
RnntModel::prompt_index_for_lang(const std::string& lang) const {
    if (num_prompts_ <= 0)
        return -1;
    auto it = prompt_dictionary_.find(lang);
    if (it != prompt_dictionary_.end())
        return it->second;
    // Unknown/empty tag: fall back to "auto" (language detection) if present.
    auto a = prompt_dictionary_.find("auto");
    return (a != prompt_dictionary_.end()) ? a->second : -1;
}

std::unique_ptr<RnntStreamState>
RnntModel::make_rnnt_stream_state() {
    return std::make_unique<RnntDecoderState>(this, acquire_decoder_slot());
}

void
RnntModel::begin_decode_step() {
    active_decode_steps_.fetch_add(1, std::memory_order_acq_rel);
    if (!batching_cfg_.enabled || batching_cfg_.max_batch_size <= 1 ||
        batching_cfg_.max_queue_delay_us <= 0)
        return;

    std::unique_lock<std::mutex> lock(decode_admission_mu_);
    const uint64_t generation = decode_admission_generation_;
    if (decode_admission_waiting_++ == 0) {
        decode_admission_deadline_ = std::chrono::steady_clock::now() +
                                     std::chrono::microseconds(batching_cfg_.max_queue_delay_us);
    }
    if (decode_admission_waiting_ >= batching_cfg_.max_batch_size) {
        decode_admission_waiting_ = 0;
        ++decode_admission_generation_;
        decode_admission_cv_.notify_all();
        return;
    }
    if (!decode_admission_cv_.wait_until(lock, decode_admission_deadline_, [&] {
            return decode_admission_generation_ != generation;
        })) {
        // The oldest waiter closes this wave at the shared deadline. Every
        // other waiter observes the generation change and starts together.
        if (decode_admission_generation_ == generation) {
            decode_admission_waiting_ = 0;
            ++decode_admission_generation_;
            decode_admission_cv_.notify_all();
        }
    }
}

void
RnntModel::end_decode_step() {
    active_decode_steps_.fetch_sub(1, std::memory_order_acq_rel);
}

void
RnntModel::predict_rnnt(RnntStreamState& stream_state, int prev_token, int active_bank) {
    auto* state = dynamic_cast<RnntDecoderState*>(&stream_state);
    if (!state || state->owner != this || state->slot < 0)
        throw std::invalid_argument("predict_rnnt: stream state belongs to another engine");
    if (active_bank != 0 && active_bank != 1)
        throw std::invalid_argument("predict_rnnt: active bank must be 0 or 1");
    decoder_batchers_->predict(
        state->slot, static_cast<int32_t>(prev_token), active_bank,
        active_decode_steps_.load(std::memory_order_acquire));
}

void
RnntModel::joint_argmax(
    RnntStreamState& stream_state, const float* enc_proj, int joint_dim, int T, int32_t* token_ids,
    const float* logit_bias) {
    auto* state = dynamic_cast<RnntDecoderState*>(&stream_state);
    if (!state || state->owner != this || state->slot < 0)
        throw std::invalid_argument("joint_argmax: stream state belongs to another engine");
    if (joint_dim != rnnt_cfg_.joint_dim || T <= 0) {
        throw std::runtime_error("joint_argmax: invalid projected encoder shape");
    }
    auto result = decoder_batchers_->joint(
        state->slot, enc_proj, T, logit_bias, rnnt_cfg_.vocab_size,
        active_decode_steps_.load(std::memory_order_acquire));
    std::copy(result.tokens.begin(), result.tokens.end(), token_ids);
}

void
RnntModel::joint_tdt_argmax(
    RnntStreamState& stream_state, const float* enc_proj, int joint_dim, int T, int32_t* token_ids,
    int32_t* duration_ids) {
    if (!rnnt_cfg_.is_tdt())
        throw std::runtime_error("joint_tdt_argmax called for a non-TDT model");
    auto* state = dynamic_cast<RnntDecoderState*>(&stream_state);
    if (!state || state->owner != this || state->slot < 0)
        throw std::invalid_argument("joint_tdt_argmax: stream state belongs to another engine");
    if (joint_dim != rnnt_cfg_.joint_dim || T <= 0)
        throw std::runtime_error("joint_tdt_argmax: invalid projected encoder shape");
    auto result = decoder_batchers_->joint(
        state->slot, enc_proj, T, nullptr, rnnt_cfg_.vocab_size,
        active_decode_steps_.load(std::memory_order_acquire));
    if (result.durations.size() != result.tokens.size())
        throw std::runtime_error("TDT joint did not return duration argmax values");
    std::copy(result.tokens.begin(), result.tokens.end(), token_ids);
    std::copy(result.durations.begin(), result.durations.end(), duration_ids);
}

void
RnntModel::predict_and_joint_tdt_argmax(
    RnntStreamState& stream_state, int prev_token, int active_bank, const float* enc_proj,
    int joint_dim, int32_t* token_id, int32_t* duration_id) {
    if (!rnnt_cfg_.is_tdt())
        throw std::runtime_error("fused TDT decoder stage called for a non-TDT model");
    auto* state = dynamic_cast<RnntDecoderState*>(&stream_state);
    if (!state || state->owner != this || state->slot < 0)
        throw std::invalid_argument("fused TDT stream state belongs to another engine");
    if (joint_dim != rnnt_cfg_.joint_dim)
        throw std::runtime_error("fused TDT joint received an invalid encoder width");
    if (active_bank != 0 && active_bank != 1)
        throw std::invalid_argument("fused TDT active bank must be 0 or 1");
    auto result = decoder_batchers_->predict_joint_tdt(
        state->slot, static_cast<int32_t>(prev_token), active_bank, enc_proj,
        active_decode_steps_.load(std::memory_order_acquire));
    *token_id = result.tokens.front();
    *duration_id = result.durations.front();
}

std::unique_ptr<Decoder>
RnntModel::make_transducer_decoder() {
    if (rnnt_cfg_.is_tdt())
        return std::make_unique<TdtGreedyDecoder>(this);
    return std::make_unique<RnntGreedyDecoder>(this);
}

int
RnntModel::acquire_decoder_slot() {
    std::lock_guard<std::mutex> lock(decoder_slots_mu_);
    for (int slot = 0; slot < decoder_arena_slots_; ++slot) {
        if (!decoder_slots_used_[static_cast<size_t>(slot)]) {
            decoder_slots_used_[static_cast<size_t>(slot)] = true;
            return slot;
        }
    }
    throw std::runtime_error("RnntModel: predictor-state arena is full");
}

void
RnntModel::zero_decoder_slot(int slot) {
    std::lock_guard<std::mutex> compute_lock(backend_manager().compute_mutex());
    for (int bank = 0; bank < 2; ++bank) {
        for (int l = 0; l < rnnt_cfg_.pred_num_layers; ++l) {
            for (const auto& name : {rnnt_h_state_name(bank, l), rnnt_c_state_name(bank, l)}) {
                auto t = decoder_session_->model_tensor_container->get_tensor_by_name(name).tensor;
                ggml_backend_tensor_memset(t, 0, static_cast<size_t>(slot) * t->nb[1], t->nb[1]);
            }
        }
    }
    auto p = decoder_session_->model_tensor_container->get_tensor_by_name(kRnntPredProjectionState)
                 .tensor;
    ggml_backend_tensor_memset(p, 0, static_cast<size_t>(slot) * p->nb[1], p->nb[1]);
}

void
RnntModel::release_decoder_slot(int slot) {
    zero_decoder_slot(slot);
    std::lock_guard<std::mutex> lock(decoder_slots_mu_);
    decoder_slots_used_[static_cast<size_t>(slot)] = false;
}

BatchMetrics
RnntModel::encoder_batch_metrics() const {
    return cache_encoder_ ? cache_encoder_->batch_metrics() : BatchMetrics{};
}
BatchMetrics
RnntModel::predictor_batch_metrics() const {
    return decoder_batchers_->predictor_metrics();
}
BatchMetrics
RnntModel::joint_batch_metrics() const {
    return decoder_batchers_->joint_metrics();
}

void
RnntModel::set_cache_right_ctx(int R) {
    if (!cache_encoder_)
        throw std::runtime_error("model encoder does not support cache-aware streaming");
    cache_encoder_->set_right_ctx(R);
}

CacheAwareEncoder::State
RnntModel::make_cache_state() {
    if (!cache_encoder_)
        throw std::runtime_error("model encoder does not support cache-aware streaming");
    return cache_encoder_->make_state();
}

void
RnntModel::reset_cache_state(CacheAwareEncoder::State& state) {
    if (!cache_encoder_)
        throw std::runtime_error("model encoder does not support cache-aware streaming");
    cache_encoder_->reset_state(state);
}

std::vector<AsrModel::DiagSession>
RnntModel::diagnostic_sessions() const {
    // cache_encoder_->session() is null until the first encode builds it.
    std::vector<DiagSession> out;
    if (fe().diagnostic_session())
        out.push_back({"streaming frontend", fe().diagnostic_session()});
    out.push_back(
        {rnnt_cfg_.is_tdt() ? "TDT decoder stages" : "RNNT decoder stages",
         decoder_session_.get()});
    if (cache_encoder_)
        out.push_back({"cache-aware encoder", cache_encoder_->session()});
    if (offline_fe_ && offline_fe_->diagnostic_session())
        out.push_back({"offline frontend", offline_fe_->diagnostic_session()});
    if (offline_encoder_session_)
        out.push_back({"offline transducer encoder", offline_encoder_session_.get()});
    return out;
}

void
RnntModel::encode_cache_aware(
    CacheAwareEncoder::State& state, const float* mel, int n_mel_frames, const float* attn_mask,
    int attn_mask_len, std::vector<float>& enc_out, int& T_enc, int prompt_index) {
    if (!cache_encoder_)
        throw std::runtime_error("model encoder does not support cache-aware streaming");
    std::vector<float> onehot;
    const float* tail_input = nullptr;
    if (prompt_fusion_ && prompt_index >= 0 && prompt_index < num_prompts_) {
        const int T = cache_encoder_->chunk_frames();
        onehot.assign(static_cast<size_t>(num_prompts_) * T, 0.0f);
        for (int t = 0; t < T; ++t)
            onehot[static_cast<size_t>(t) * num_prompts_ + prompt_index] = 1.0f;
        tail_input = onehot.data();
    }
    cache_encoder_->encode(
        state, mel, n_mel_frames, attn_mask, attn_mask_len, enc_out, T_enc, tail_input,
        tail_input ? num_prompts_ : 0);
}

}  // namespace nemo_speech::asr
