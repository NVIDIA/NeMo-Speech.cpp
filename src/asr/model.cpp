// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <sentencepiece_processor.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "cache_aware_encoder.h"  // CacheAwareEncoder (cache-aware streaming subsystem)
#include "nvtx_utils.h"
#include "rnnt_modules.h"  // RnntPredictorModule / RnntJointModule / PromptFusionModule
#include "runtime.h"

namespace nemo_speech::asr {

namespace {
// Decode standard base64 (the SPM tokenizer proto is stored base64 in GGUF
// metadata so it survives the UTF-8 string value type). Ignores whitespace;
// returns "" on malformed input.
std::string
base64_decode(const std::string& in) {
    static constexpr char kPad = '=';
    std::array<int8_t, 256> rev;
    rev.fill(-1);
    const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) rev[static_cast<uint8_t>(tbl[i])] = static_cast<int8_t>(i);
    std::string out;
    out.reserve(in.size() * 3 / 4);
    int val = 0, bits = 0;
    for (char ch : in) {
        if (ch == kPad)
            break;
        const int8_t d = rev[static_cast<uint8_t>(ch)];
        if (d < 0)
            continue;  // skip whitespace / newlines
        val = (val << 6) | d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((val >> bits) & 0xFF));
        }
    }
    return out;
}
}  // namespace

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
        const bool fused_rnnt =
            name == "rnnt.predict_joint_rnnt.0" || name == "rnnt.predict_joint_rnnt.1";
        if (name == "rnnt.predict.0" || name == "rnnt.predict.1" || fused_tdt || fused_rnnt) {
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
                pred_in.add_tensor(
                    ggml_runtime::ggml_bf_tensor(
                        ggml_get_rows(bf.ctx, h.tensor, slot_ids.tensor), h.buft));
                pred_in.add_tensor(
                    ggml_runtime::ggml_bf_tensor(
                        ggml_get_rows(bf.ctx, c.tensor, slot_ids.tensor), c.buft));
            }
            auto pred_out = pred_->build_graph(s, pred_in, tc);
            auto pred_proj =
                joint_->build_predictor_projection(s, pred_out.get_tensor(0), tc).get_tensor(0);

            auto bf = tc->get_ctx_of_buffer_type(pred_proj.buft);
            ggml_runtime::TensorBag state_out;
            auto pred_state =
                s->model_tensor_container->get_tensor_by_name(kRnntPredProjectionState);
            state_out.add_tensor(
                ggml_runtime::ggml_bf_tensor(
                    ggml_set_rows(
                        bf.ctx, pred_state.tensor, ggml_cont(bf.ctx, pred_proj.tensor),
                        slot_ids.tensor),
                    pred_proj.buft));
            for (int i = 0; i < cfg.pred_num_layers; i++) {
                auto h_dst = s->model_tensor_container->get_tensor_by_name(
                    rnnt_h_state_name(candidate_bank, i));
                auto c_dst = s->model_tensor_container->get_tensor_by_name(
                    rnnt_c_state_name(candidate_bank, i));
                state_out.add_tensor(
                    ggml_runtime::ggml_bf_tensor(
                        ggml_set_rows(
                            bf.ctx, h_dst.tensor,
                            ggml_cont(bf.ctx, pred_out.get_tensor(1 + 2 * i).tensor),
                            slot_ids.tensor),
                        pred_out.get_tensor(1 + 2 * i).buft));
                state_out.add_tensor(
                    ggml_runtime::ggml_bf_tensor(
                        ggml_set_rows(
                            bf.ctx, c_dst.tensor,
                            ggml_cont(bf.ctx, pred_out.get_tensor(1 + 2 * i + 1).tensor),
                            slot_ids.tensor),
                        pred_out.get_tensor(1 + 2 * i + 1).buft));
            }
            if (!fused_tdt && !fused_rnnt)
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
    RnntDecoderState(RnntModel* owner, int slot, bool needs_reset)
        : owner(owner), slot(slot), needs_reset(needs_reset) {}
    ~RnntDecoderState() override {
        if (owner && slot >= 0)
            owner->release_decoder_slot(slot);
    }
    RnntModel* owner;
    int slot;
    bool needs_reset;
};

class RnntModel::DecoderBatchers {
   public:
    struct PredictRequest {
        int slot;
        int32_t token;
        bool reset;
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
        bool reset;
        std::vector<float> enc;
    };
    struct JointResult {
        std::vector<int32_t> tokens;
        std::vector<int32_t> durations;
    };

    DecoderBatchers(RnntModel* owner, const BatchingConfig& cfg)
        : owner_(owner), max_batch_size_(std::max(1, cfg.max_batch_size)),
          max_queue_delay_us_(std::max(0, cfg.max_queue_delay_us)),
          max_queue_depth_(std::max(max_batch_size_, cfg.max_queue_depth)) {
        if (cfg.enabled && max_batch_size_ > 1) {
            for (int i = 0; i < kExecutorCount; ++i)
                executors_.emplace_back([this] { executor_loop(); });
            scheduler_ = std::thread([this] { scheduler_loop(); });
        }
    }

    ~DecoderBatchers() { shutdown(); }

    DecoderBatchers(const DecoderBatchers&) = delete;
    DecoderBatchers& operator=(const DecoderBatchers&) = delete;

    void predict(int slot, int32_t token, int bank, int admitted_decodes, bool reset) {
        std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mu_);
        if (!scheduler_.joinable() || admitted_decodes == 1) {
            reject_if_stopping();
            (void)run_predictor(bank, {{slot, token, reset}});
            predictor_metrics_.record_inline();
            return;
        }
        lifecycle_lock.unlock();
        auto job = std::make_shared<PredictJob>(PredictRequest{slot, token, reset});
        auto result = job->promise.get_future();
        enqueue(predictor_queues_[static_cast<size_t>(bank)], job);
        (void)result.get();
    }

    JointResult joint(
        int slot, const float* enc, int T, const float* bias, int vocab_size,
        int admitted_decodes) {
        JointRequest req;
        req.slot = slot;
        req.enc.assign(enc, enc + static_cast<size_t>(owner_->rnnt_cfg_.joint_dim) * T);
        if (bias)
            req.bias.assign(bias, bias + vocab_size);
        const JointKey key{T, bias != nullptr};
        std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mu_);
        if (!scheduler_.joinable() || admitted_decodes == 1) {
            reject_if_stopping();
            auto result = run_joint(key, {std::move(req)});
            joint_metrics_.record_inline();
            return std::move(result.front());
        }
        lifecycle_lock.unlock();
        auto job = std::make_shared<JointJob>(std::move(req));
        auto result = job->promise.get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            check_enqueue_locked();
            joint_queues_[key].push_back(job);
        }
        cv_.notify_one();
        return result.get();
    }

    JointResult joint_device(
        int slot, const ggml_runtime::DeviceTensor& enc, int frame_offset, int T, const float* bias,
        int vocab_size) {
        return run_joint_device(slot, enc, frame_offset, T, bias, vocab_size);
    }

    JointResult predict_joint_rnnt_device(
        int slot, int32_t token, int bank, const ggml_runtime::DeviceTensor& enc, int frame_offset,
        int T, const float* bias, int vocab_size, bool reset) {
        return run_fused_rnnt_device(
            slot, token, bank, enc, frame_offset, T, bias, vocab_size, reset);
    }

    JointResult predict_joint_tdt(
        int slot, int32_t token, int bank, const float* enc, int admitted_decodes, bool reset) {
        TdtFusedRequest req;
        req.slot = slot;
        req.token = token;
        req.reset = reset;
        req.enc.assign(enc, enc + owner_->rnnt_cfg_.joint_dim);
        std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mu_);
        if (!scheduler_.joinable() || admitted_decodes == 1) {
            reject_if_stopping();
            auto result = run_fused_tdt(bank, {std::move(req)});
            fused_metrics_.record_inline();
            return std::move(result.front());
        }
        lifecycle_lock.unlock();
        auto job = std::make_shared<FusedJob>(std::move(req));
        auto result = job->promise.get_future();
        enqueue(fused_queues_[static_cast<size_t>(bank)], job);
        return result.get();
    }

    // A decode step can finish while all remaining steps are blocked in ready
    // queues. Wake the scheduler so it can re-evaluate that exact ready set
    // instead of waiting for a queue timer.
    void notify_active_change() { cv_.notify_one(); }

    static BatchMetrics add(BatchMetrics a, const BatchMetrics& b) {
        a.batches += b.batches;
        a.items += b.items;
        a.singleton_batches += b.singleton_batches;
        a.max_observed_batch = std::max(a.max_observed_batch, b.max_observed_batch);
        a.target_reached_batches += b.target_reached_batches;
        a.deadline_batches += b.deadline_batches;
        a.requested_items += b.requested_items;
        a.queue_wait_ns += b.queue_wait_ns;
        a.capacity_batches += b.capacity_batches;
        a.ready_set_batches += b.ready_set_batches;
        a.ready_items += b.ready_items;
        a.compatible_items += b.compatible_items;
        a.execution_ns += b.execution_ns;
        return a;
    }
    BatchMetrics predictor_metrics() const {
        return add(predictor_metrics_.snapshot(), fused_metrics_.snapshot());
    }
    BatchMetrics joint_metrics() const {
        return add(joint_metrics_.snapshot(), fused_metrics_.snapshot());
    }

   private:
    using Clock = std::chrono::steady_clock;
    enum class Op { Predictor, Joint, FusedTdt };
    enum class DispatchReason { Capacity, ReadySet, Deadline, Shutdown };

    struct JointKeyLess {
        bool operator()(const JointKey& a, const JointKey& b) const {
            if (a.T != b.T)
                return a.T < b.T;
            return a.has_bias < b.has_bias;
        }
    };

    template <class Request, class Result>
    struct Job {
        explicit Job(Request r) : request(std::move(r)), queued_at(Clock::now()) {}
        Request request;
        Clock::time_point queued_at;
        std::promise<Result> promise;
    };
    using PredictJob = Job<PredictRequest, uint8_t>;
    using JointJob = Job<JointRequest, JointResult>;
    using FusedJob = Job<TdtFusedRequest, JointResult>;

    struct Candidate {
        Op op = Op::Predictor;
        int bank = 0;
        JointKey joint_key{0, false};
        size_t count = 0;
        Clock::time_point oldest{};
    };

    struct BatchTask {
        Candidate selected;
        DispatchReason reason = DispatchReason::Deadline;
        size_t admitted = 0;
        size_t ready = 0;
        size_t compatible = 0;
        uint64_t wait_ns = 0;
        std::vector<std::shared_ptr<PredictJob>> predict_jobs;
        std::vector<std::shared_ptr<JointJob>> joint_jobs;
        std::vector<std::shared_ptr<FusedJob>> fused_jobs;

        size_t size() const { return predict_jobs.size() + joint_jobs.size() + fused_jobs.size(); }
    };

    struct AtomicMetrics {
        std::atomic<uint64_t> batches{0};
        std::atomic<uint64_t> items{0};
        std::atomic<uint64_t> singleton_batches{0};
        std::atomic<uint64_t> max_observed_batch{0};
        std::atomic<uint64_t> capacity_batches{0};
        std::atomic<uint64_t> ready_set_batches{0};
        std::atomic<uint64_t> deadline_batches{0};
        std::atomic<uint64_t> requested_items{0};
        std::atomic<uint64_t> ready_items{0};
        std::atomic<uint64_t> compatible_items{0};
        std::atomic<uint64_t> queue_wait_ns{0};
        std::atomic<uint64_t> execution_ns{0};

        void record_inline() {
            batches.fetch_add(1, std::memory_order_relaxed);
            items.fetch_add(1, std::memory_order_relaxed);
            singleton_batches.fetch_add(1, std::memory_order_relaxed);
            uint64_t old = max_observed_batch.load(std::memory_order_relaxed);
            while (old < 1 && !max_observed_batch.compare_exchange_weak(
                                  old, 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
            }
        }

        void record(
            size_t n, DispatchReason reason, size_t admitted, size_t ready, size_t compatible,
            uint64_t wait_ns, uint64_t run_ns) {
            batches.fetch_add(1, std::memory_order_relaxed);
            items.fetch_add(n, std::memory_order_relaxed);
            if (n == 1)
                singleton_batches.fetch_add(1, std::memory_order_relaxed);
            uint64_t old = max_observed_batch.load(std::memory_order_relaxed);
            while (old < n && !max_observed_batch.compare_exchange_weak(
                                  old, n, std::memory_order_relaxed, std::memory_order_relaxed)) {
            }
            if (reason == DispatchReason::Capacity)
                capacity_batches.fetch_add(1, std::memory_order_relaxed);
            else if (reason == DispatchReason::ReadySet)
                ready_set_batches.fetch_add(1, std::memory_order_relaxed);
            else if (reason == DispatchReason::Deadline)
                deadline_batches.fetch_add(1, std::memory_order_relaxed);
            requested_items.fetch_add(admitted, std::memory_order_relaxed);
            ready_items.fetch_add(ready, std::memory_order_relaxed);
            compatible_items.fetch_add(compatible, std::memory_order_relaxed);
            queue_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
            execution_ns.fetch_add(run_ns, std::memory_order_relaxed);
        }

        BatchMetrics snapshot() const {
            BatchMetrics out;
            out.batches = batches.load(std::memory_order_relaxed);
            out.items = items.load(std::memory_order_relaxed);
            out.singleton_batches = singleton_batches.load(std::memory_order_relaxed);
            out.max_observed_batch = max_observed_batch.load(std::memory_order_relaxed);
            out.target_reached_batches = capacity_batches.load(std::memory_order_relaxed);
            out.deadline_batches = deadline_batches.load(std::memory_order_relaxed);
            out.requested_items = requested_items.load(std::memory_order_relaxed);
            out.queue_wait_ns = queue_wait_ns.load(std::memory_order_relaxed);
            out.capacity_batches = capacity_batches.load(std::memory_order_relaxed);
            out.ready_set_batches = ready_set_batches.load(std::memory_order_relaxed);
            out.ready_items = ready_items.load(std::memory_order_relaxed);
            out.compatible_items = compatible_items.load(std::memory_order_relaxed);
            out.execution_ns = execution_ns.load(std::memory_order_relaxed);
            return out;
        }
    };

    template <class JobType>
    void enqueue(std::deque<std::shared_ptr<JobType>>& queue, std::shared_ptr<JobType> job) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            check_enqueue_locked();
            queue.push_back(std::move(job));
        }
        cv_.notify_one();
    }

    void reject_if_stopping() const {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_)
            throw std::runtime_error("RNNT decoder scheduler: submit after shutdown");
    }

    void check_enqueue_locked() const {
        if (stopping_)
            throw std::runtime_error("RNNT decoder scheduler: submit after shutdown");
        if (pending_locked() >= static_cast<size_t>(max_queue_depth_))
            throw std::runtime_error("RNNT decoder scheduler: queue is full");
    }

    size_t pending_locked() const {
        size_t total = 0;
        for (const auto& queue : predictor_queues_) total += queue.size();
        for (const auto& entry : joint_queues_) total += entry.second.size();
        for (const auto& queue : fused_queues_) total += queue.size();
        return total;
    }

    std::vector<Candidate> candidates_locked() const {
        std::vector<Candidate> out;
        out.reserve(4 + joint_queues_.size());
        for (int bank = 0; bank < 2; ++bank) {
            const auto& predictor = predictor_queues_[static_cast<size_t>(bank)];
            if (!predictor.empty())
                out.push_back(
                    {Op::Predictor, bank, {}, predictor.size(), predictor.front()->queued_at});
            const auto& fused = fused_queues_[static_cast<size_t>(bank)];
            if (!fused.empty())
                out.push_back({Op::FusedTdt, bank, {}, fused.size(), fused.front()->queued_at});
        }
        for (const auto& [key, queue] : joint_queues_)
            if (!queue.empty())
                out.push_back({Op::Joint, 0, key, queue.size(), queue.front()->queued_at});
        return out;
    }

    static Candidate oldest_candidate(const std::vector<Candidate>& candidates) {
        return *std::min_element(
            candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.oldest < b.oldest; });
    }

    static Candidate largest_candidate(const std::vector<Candidate>& candidates) {
        return *std::max_element(
            candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                if (a.count != b.count)
                    return a.count < b.count;
                return a.oldest > b.oldest;
            });
    }

    std::optional<Candidate> capacity_candidate(const std::vector<Candidate>& candidates) const {
        std::optional<Candidate> selected;
        for (const auto& candidate : candidates) {
            if (candidate.count < static_cast<size_t>(max_batch_size_))
                continue;
            if (!selected || candidate.oldest < selected->oldest)
                selected = candidate;
        }
        return selected;
    }

    std::vector<uint8_t> run_predictor(int bank, std::vector<PredictRequest>&& req) {
        const ggml_nvtx::range nvtx("asr.predictor.batch");
        const int B = static_cast<int>(req.size());
        std::vector<int32_t> tokens(static_cast<size_t>(B));
        std::vector<int32_t> slots(static_cast<size_t>(B));
        std::vector<int> reset_slots;
        for (int b = 0; b < B; ++b) {
            tokens[b] = req[b].token;
            slots[b] = req[b].slot;
            if (req[b].reset)
                reset_slots.push_back(req[b].slot);
        }
        if (!reset_slots.empty())
            owner_->zero_decoder_slots(std::move(reset_slots));
        std::vector<ggml_runtime::Session::Input> inputs = {
            {"rnnt.predict." + std::to_string(bank), GGML_TYPE_I32, tokens.data(), {B}},
            {"rnnt.slot_ids", GGML_TYPE_I32, slots.data(), {B}}};
        std::vector<ggml_runtime::Session::Output> outputs;
        owner_->decoder_session_->run(inputs, outputs);
        return std::vector<uint8_t>(static_cast<size_t>(B), 1);
    }

    std::vector<JointResult> run_joint(const JointKey& key, std::vector<JointRequest>&& req) {
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
            const auto token_begin = packed_tokens.begin() + static_cast<size_t>(b) * key.T;
            result[b].tokens.assign(token_begin, token_begin + key.T);
            if (owner_->rnnt_cfg_.is_tdt()) {
                const auto duration_begin =
                    packed_durations.begin() + static_cast<size_t>(b) * key.T;
                result[b].durations.assign(duration_begin, duration_begin + key.T);
            }
        }
        return result;
    }

    JointResult run_joint_device(
        int slot, const ggml_runtime::DeviceTensor& enc, int frame_offset, int T, const float* bias,
        int vocab_size) {
        const ggml_nvtx::range nvtx("asr.joint.device");
        const int J = owner_->rnnt_cfg_.joint_dim;
        ggml_runtime::DeviceTensor enc_slice = enc;
        enc_slice.byte_offset += static_cast<size_t>(frame_offset) * J * sizeof(float);

        ggml_runtime::Session::Input enc_input{"rnnt.joint.enc", GGML_TYPE_F32, nullptr, {J, T}};
        enc_input.upload = false;
        enc_input.device_tensor = &enc_slice;
        int32_t slot_id = slot;
        std::vector<ggml_runtime::Session::Input> inputs;
        inputs.push_back(enc_input);
        inputs.push_back({"rnnt.slot_ids", GGML_TYPE_I32, &slot_id, {1}});
        if (bias != nullptr)
            inputs.push_back({"rnnt.joint.bias", GGML_TYPE_F32, bias, {vocab_size, 1, 1}});

        JointResult result;
        result.tokens.resize(static_cast<size_t>(T));
        std::vector<ggml_runtime::Session::Output> outputs(1);
        outputs[0].index = 0;
        outputs[0].host_buffer = result.tokens.data();
        outputs[0].nbytes = result.tokens.size() * sizeof(int32_t);
        if (owner_->rnnt_cfg_.is_tdt()) {
            result.durations.resize(static_cast<size_t>(T));
            outputs.resize(2);
            outputs[1].index = 1;
            outputs[1].host_buffer = result.durations.data();
            outputs[1].nbytes = result.durations.size() * sizeof(int32_t);
        }
        owner_->decoder_session_->run(inputs, outputs);
        return result;
    }

    JointResult run_fused_rnnt_device(
        int slot, int32_t token, int bank, const ggml_runtime::DeviceTensor& enc, int frame_offset,
        int T, const float* bias, int vocab_size, bool reset) {
        const ggml_nvtx::range nvtx("asr.predict_joint.device");
        if (reset)
            owner_->zero_decoder_slot(slot);
        const int J = owner_->rnnt_cfg_.joint_dim;
        ggml_runtime::DeviceTensor enc_slice = enc;
        enc_slice.byte_offset += static_cast<size_t>(frame_offset) * J * sizeof(float);
        ggml_runtime::Session::Input enc_input{"rnnt.joint.enc", GGML_TYPE_F32, nullptr, {J, T}};
        enc_input.upload = false;
        enc_input.device_tensor = &enc_slice;
        int32_t slot_id = slot;
        std::vector<ggml_runtime::Session::Input> inputs = {
            {"rnnt.predict_joint_rnnt." + std::to_string(bank), GGML_TYPE_I32, &token, {1}},
            {"rnnt.slot_ids", GGML_TYPE_I32, &slot_id, {1}},
            enc_input};
        if (bias != nullptr)
            inputs.push_back({"rnnt.joint.bias", GGML_TYPE_F32, bias, {vocab_size, 1, 1}});

        JointResult result;
        result.tokens.resize(static_cast<size_t>(T));
        std::vector<ggml_runtime::Session::Output> outputs(1);
        outputs[0].index = 0;
        outputs[0].host_buffer = result.tokens.data();
        outputs[0].nbytes = result.tokens.size() * sizeof(int32_t);
        owner_->decoder_session_->run(inputs, outputs);
        return result;
    }

    std::vector<JointResult> run_fused_tdt(int bank, std::vector<TdtFusedRequest>&& req) {
        const ggml_nvtx::range nvtx("asr.predict_joint_tdt.batch");
        const int B = static_cast<int>(req.size());
        const int J = owner_->rnnt_cfg_.joint_dim;
        std::vector<int32_t> tokens(static_cast<size_t>(B));
        std::vector<int32_t> slots(static_cast<size_t>(B));
        std::vector<int> reset_slots;
        std::vector<float> enc(static_cast<size_t>(J) * B);
        for (int b = 0; b < B; ++b) {
            tokens[b] = req[b].token;
            slots[b] = req[b].slot;
            if (req[b].reset)
                reset_slots.push_back(req[b].slot);
            std::copy(
                req[b].enc.begin(), req[b].enc.end(), enc.begin() + static_cast<size_t>(b) * J);
        }
        if (!reset_slots.empty())
            owner_->zero_decoder_slots(std::move(reset_slots));
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
    }

    template <class JobType>
    static std::vector<std::shared_ptr<JobType>> take_jobs(
        std::deque<std::shared_ptr<JobType>>& queue, size_t count) {
        std::vector<std::shared_ptr<JobType>> jobs;
        jobs.reserve(count);
        while (jobs.size() < count) {
            jobs.push_back(std::move(queue.front()));
            queue.pop_front();
        }
        return jobs;
    }

    template <class JobType>
    static void fail_jobs(
        const std::vector<std::shared_ptr<JobType>>& jobs, const std::exception_ptr& error) {
        for (const auto& job : jobs) job->promise.set_exception(error);
    }

    void scheduler_loop() {
        for (;;) {
            BatchTask task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] {
                    return (stopping_ && pending_locked() == 0) ||
                           (pending_locked() > 0 && tasks_in_flight_ < kExecutorCount);
                });
                if (stopping_ && pending_locked() == 0) {
                    scheduling_done_ = true;
                    executor_cv_.notify_all();
                    return;
                }

                for (;;) {
                    const auto candidates = candidates_locked();
                    if (candidates.empty())
                        break;
                    task.ready = pending_locked();
                    task.admitted = static_cast<size_t>(std::max(
                        0, owner_->admitted_decode_steps_.load(std::memory_order_acquire)));
                    if (const auto full = capacity_candidate(candidates)) {
                        task.selected = *full;
                        task.reason = DispatchReason::Capacity;
                        break;
                    }
                    const size_t blocked = task.ready + items_in_flight_;
                    if (task.admitted > 0 && blocked >= task.admitted) {
                        task.selected = largest_candidate(candidates);
                        task.reason = DispatchReason::ReadySet;
                        break;
                    }
                    const Candidate oldest = oldest_candidate(candidates);
                    const auto deadline =
                        oldest.oldest + std::chrono::microseconds(max_queue_delay_us_);
                    if (stopping_) {
                        task.selected = largest_candidate(candidates);
                        task.reason = DispatchReason::Shutdown;
                        break;
                    }
                    if (Clock::now() >= deadline) {
                        task.selected = oldest;
                        task.reason = DispatchReason::Deadline;
                        break;
                    }
                    cv_.wait_until(lock, deadline);
                }

                task.compatible = task.selected.count;
                const size_t count =
                    std::min(task.selected.count, static_cast<size_t>(max_batch_size_));
                task.wait_ns =
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              Clock::now() - task.selected.oldest)
                                              .count());
                if (task.selected.op == Op::Predictor) {
                    task.predict_jobs = take_jobs(
                        predictor_queues_[static_cast<size_t>(task.selected.bank)], count);
                } else if (task.selected.op == Op::FusedTdt) {
                    task.fused_jobs =
                        take_jobs(fused_queues_[static_cast<size_t>(task.selected.bank)], count);
                } else {
                    auto it = joint_queues_.find(task.selected.joint_key);
                    task.joint_jobs = take_jobs(it->second, count);
                    if (it->second.empty())
                        joint_queues_.erase(it);
                }
                items_in_flight_ += task.size();
                ++tasks_in_flight_;
                task_queue_.push_back(std::move(task));
            }
            executor_cv_.notify_one();
        }
    }

    void execute_task(BatchTask task) {
        const size_t task_size = task.size();
        const auto run_started = Clock::now();
        bool released = false;
        try {
            if (!task.predict_jobs.empty()) {
                std::vector<PredictRequest> requests;
                requests.reserve(task.predict_jobs.size());
                for (auto& job : task.predict_jobs) requests.push_back(std::move(job->request));
                auto results = run_predictor(task.selected.bank, std::move(requests));
                if (results.size() != task.predict_jobs.size())
                    throw std::runtime_error("RNNT predictor returned wrong result count");
                const uint64_t run_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - run_started)
                        .count());
                predictor_metrics_.record(
                    task.predict_jobs.size(), task.reason, task.admitted, task.ready,
                    task.compatible, task.wait_ns, run_ns);
                finish_task(task_size);
                released = true;
                for (size_t i = 0; i < task.predict_jobs.size(); ++i)
                    task.predict_jobs[i]->promise.set_value(results[i]);
            } else if (!task.fused_jobs.empty()) {
                std::vector<TdtFusedRequest> requests;
                requests.reserve(task.fused_jobs.size());
                for (auto& job : task.fused_jobs) requests.push_back(std::move(job->request));
                auto results = run_fused_tdt(task.selected.bank, std::move(requests));
                if (results.size() != task.fused_jobs.size())
                    throw std::runtime_error("RNNT fused TDT returned wrong result count");
                const uint64_t run_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - run_started)
                        .count());
                fused_metrics_.record(
                    task.fused_jobs.size(), task.reason, task.admitted, task.ready, task.compatible,
                    task.wait_ns, run_ns);
                finish_task(task_size);
                released = true;
                for (size_t i = 0; i < task.fused_jobs.size(); ++i)
                    task.fused_jobs[i]->promise.set_value(std::move(results[i]));
            } else {
                std::vector<JointRequest> requests;
                requests.reserve(task.joint_jobs.size());
                for (auto& job : task.joint_jobs) requests.push_back(std::move(job->request));
                auto results = run_joint(task.selected.joint_key, std::move(requests));
                if (results.size() != task.joint_jobs.size())
                    throw std::runtime_error("RNNT joint returned wrong result count");
                const uint64_t run_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - run_started)
                        .count());
                joint_metrics_.record(
                    task.joint_jobs.size(), task.reason, task.admitted, task.ready, task.compatible,
                    task.wait_ns, run_ns);
                finish_task(task_size);
                released = true;
                for (size_t i = 0; i < task.joint_jobs.size(); ++i)
                    task.joint_jobs[i]->promise.set_value(std::move(results[i]));
            }
        }
        catch (...) {
            const auto error = std::current_exception();
            if (!released)
                finish_task(task_size);
            fail_jobs(task.predict_jobs, error);
            fail_jobs(task.joint_jobs, error);
            fail_jobs(task.fused_jobs, error);
        }
    }

    void finish_task(size_t items) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            items_in_flight_ -= items;
            --tasks_in_flight_;
        }
        cv_.notify_one();
    }

    void executor_loop() {
        for (;;) {
            BatchTask task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                executor_cv_.wait(
                    lock, [this] { return scheduling_done_ || !task_queue_.empty(); });
                if (task_queue_.empty() && scheduling_done_)
                    return;
                task = std::move(task_queue_.front());
                task_queue_.pop_front();
            }
            execute_task(std::move(task));
        }
    }

    void shutdown() {
        std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mu_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_)
                return;
            stopping_ = true;
        }
        cv_.notify_all();
        if (scheduler_.joinable())
            scheduler_.join();
        executor_cv_.notify_all();
        for (auto& executor : executors_)
            if (executor.joinable())
                executor.join();
    }

    RnntModel* owner_;
    int max_batch_size_;
    int max_queue_delay_us_;
    int max_queue_depth_;
    mutable std::shared_mutex lifecycle_mu_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable executor_cv_;
    std::array<std::deque<std::shared_ptr<PredictJob>>, 2> predictor_queues_;
    std::map<JointKey, std::deque<std::shared_ptr<JointJob>>, JointKeyLess> joint_queues_;
    std::array<std::deque<std::shared_ptr<FusedJob>>, 2> fused_queues_;
    std::deque<BatchTask> task_queue_;
    // The backend remains serialized. Three bounded preparation slots overlap
    // host packing and dependency wakeups without allowing an unbounded queue
    // of already-shaped GPU submissions to accumulate.
    static constexpr int kExecutorCount = 3;
    size_t items_in_flight_ = 0;
    int tasks_in_flight_ = 0;
    bool stopping_ = false;
    bool scheduling_done_ = false;
    std::thread scheduler_;
    std::vector<std::thread> executors_;
    AtomicMetrics predictor_metrics_;
    AtomicMetrics joint_metrics_;
    AtomicMetrics fused_metrics_;
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
    fe.stft_center_window =
        loader.get_bool(A + ".preprocessor.stft_center_window", fe.stft_center_window);
    fe.hann_periodic = loader.get_bool(A + ".preprocessor.hann_periodic", fe.hann_periodic);
    fe.mask_invalid_frames =
        loader.get_bool(A + ".preprocessor.mask_invalid_frames", fe.mask_invalid_frames);
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

    // Load the embedded SentencePiece tokenizer for word-boosting phrase
    // tokenization (flashlight OOV phrases, RNNT context biasing). Stored
    // base64 under asr.tokenizer.spm_model; absent in older GGUFs (boosting
    // paths then fall back or no-op with a warning).
    const std::string spm_b64 = loader()->get_str("asr.tokenizer.spm_model", "");
    if (!spm_b64.empty()) {
        const std::string proto = base64_decode(spm_b64);
        auto sp = std::make_unique<sentencepiece::SentencePieceProcessor>();
        const auto st = sp->LoadFromSerializedProto(proto);
        if (st.ok()) {
            spm_ = std::move(sp);
            GGMLF_LOG_INFO(
                "[asr_model] embedded SentencePiece tokenizer loaded (%d pieces) for word "
                "boosting\n",
                spm_->GetPieceSize());
        } else {
            GGMLF_LOG_WARN(
                "[asr_model] failed to load embedded SentencePiece tokenizer: %s "
                "(word boosting degraded)\n",
                st.ToString().c_str());
        }
    }
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
    // Cache-aware GGUFs produced before the frontend-geometry metadata was
    // introduced still originate from NeMo FilterbankFeatures. Recover its
    // actual centered/symmetric/masked contract for those existing files.
    // New conversions serialize these keys explicitly above, so this fallback
    // is limited to old cache-aware artifacts rather than changing legacy CTC.
    if (c.enc_cfg.cache_supported) {
        const std::string fe_prefix = c.ns + ".preprocessor.";
        if (!loader.has_key(fe_prefix + "stft_center_window"))
            c.fe_cfg.stft_center_window = true;
        if (!loader.has_key(fe_prefix + "hann_periodic"))
            c.fe_cfg.hann_periodic = false;
        if (!loader.has_key(fe_prefix + "mask_invalid_frames"))
            c.fe_cfg.mask_invalid_frames = true;
    }
    c.vocab = loader.get_str_array(c.ns + ".tokenizer.vocab");

    const char* head_name =
        head == HeadKind::Ctc ? "ctc" : (head == HeadKind::Tdt ? "tdt" : "rnnt");
    GGMLF_LOG_INFO(
        "[asr_model] arch=%s head=%s d_model=%d n_layers=%d n_heads=%d d_ff=%d k=%d "
        "feat_in=%d sr=%d vocab=%zu\n",
        arch.c_str(), head_name, c.enc_cfg.d_model, c.enc_cfg.n_layers, c.enc_cfg.n_heads,
        c.enc_cfg.d_ff, c.enc_cfg.conv_kernel_size, c.enc_cfg.feat_in, c.fe_cfg.sample_rate,
        c.vocab.size());

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
        return queue_.run({n_frames, greedy}, std::move(request));
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
    GGMLF_LOG_INFO("[asr_model] ctc classes=%d\n", ctc_cfg_.num_classes);

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
CtcModel::infer_ctc_greedy(
    const float* audio, size_t n_samples, std::vector<int32_t>& best_ids,
    std::vector<float>& best_probs, int& T_out) {
    std::vector<float> feats;
    int n_frames = 0;
    offline_fe_->compute(audio, n_samples, feats, n_frames);
    infer_ctc_greedy_from_mel(feats.data(), n_frames, best_ids, best_probs, T_out);
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
    std::ostringstream model_info;
    model_info << "[asr_model] rnnt vocab=" << rnnt_cfg_.vocab_size
               << " blank=" << rnnt_cfg_.blank_id << " pred_hidden=" << rnnt_cfg_.pred_hidden
               << " joint_dim=" << rnnt_cfg_.joint_dim;
    if (rnnt_cfg_.is_tdt()) {
        model_info << " durations=";
        for (int duration : rnnt_cfg_.durations) model_info << duration << ',';
    }
    GGMLF_LOG_INFO("%s\n", model_info.str().c_str());

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
        GGMLF_LOG_INFO(
            "[asr_model] prompt fusion enabled: num_prompts=%d languages=%zu\n", num_prompts_,
            prompt_dictionary_.size());
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
    decoder_slots_need_reset_.assign(static_cast<size_t>(decoder_arena_slots_), false);
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

std::vector<int>
RnntModel::encode_phrase(const std::string& text) const {
    if (!spm_)
        return {};
    std::vector<int> ids;
    if (!spm_->Encode(text, &ids).ok())
        return {};
    return ids;
}

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
    bool needs_reset = false;
    const int slot = acquire_decoder_slot(needs_reset);
    return std::make_unique<RnntDecoderState>(this, slot, needs_reset);
}

void
RnntModel::begin_decode_step() {
    const int cohort_target = current_batch_cohort_target();
    if (!batching_cfg_.enabled || batching_cfg_.max_batch_size <= 1 ||
        batching_cfg_.max_queue_delay_us <= 0 || cohort_target == 1) {
        admitted_decode_steps_.fetch_add(1, std::memory_order_acq_rel);
        decoder_batchers_->notify_active_change();
        return;
    }

    std::unique_lock<std::mutex> lock(decode_admission_mu_);
    const uint64_t generation = decode_admission_generation_;
    if (decode_admission_waiting_++ == 0) {
        decode_admission_target_ = cohort_target > 1
                                       ? std::min(cohort_target, batching_cfg_.max_batch_size)
                                       : batching_cfg_.max_batch_size;
        decode_admission_deadline_ = std::chrono::steady_clock::now() +
                                     std::chrono::microseconds(batching_cfg_.max_queue_delay_us);
    } else if (cohort_target > 1) {
        decode_admission_target_ = std::min(
            decode_admission_target_, std::min(cohort_target, batching_cfg_.max_batch_size));
    }
    if (decode_admission_waiting_ >= decode_admission_target_) {
        admitted_decode_steps_.fetch_add(decode_admission_waiting_, std::memory_order_acq_rel);
        decode_admission_waiting_ = 0;
        decode_admission_target_ = 0;
        ++decode_admission_generation_;
        decode_admission_cv_.notify_all();
        decoder_batchers_->notify_active_change();
        return;
    }
    if (!decode_admission_cv_.wait_until(lock, decode_admission_deadline_, [&] {
            return decode_admission_generation_ != generation;
        })) {
        // The oldest waiter closes this wave at the shared deadline. Every
        // other waiter observes the generation change and starts together.
        if (decode_admission_generation_ == generation) {
            admitted_decode_steps_.fetch_add(decode_admission_waiting_, std::memory_order_acq_rel);
            decode_admission_waiting_ = 0;
            decode_admission_target_ = 0;
            ++decode_admission_generation_;
            decode_admission_cv_.notify_all();
            decoder_batchers_->notify_active_change();
        }
    }
}

void
RnntModel::end_decode_step() {
    admitted_decode_steps_.fetch_sub(1, std::memory_order_acq_rel);
    decoder_batchers_->notify_active_change();
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
        admitted_decode_steps_.load(std::memory_order_acquire), state->needs_reset);
    state->needs_reset = false;
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
        admitted_decode_steps_.load(std::memory_order_acquire));
    std::copy(result.tokens.begin(), result.tokens.end(), token_ids);
}

void
RnntModel::joint_argmax_device(
    RnntStreamState& stream_state, const ggml_runtime::DeviceTensor& enc_proj, int frame_offset,
    int joint_dim, int T, int32_t* token_ids, const float* logit_bias) {
    auto* state = dynamic_cast<RnntDecoderState*>(&stream_state);
    if (!state || state->owner != this || state->slot < 0)
        throw std::invalid_argument("joint_argmax_device: stream state belongs to another engine");
    if (!enc_proj.valid() || joint_dim != rnnt_cfg_.joint_dim || frame_offset < 0 || T <= 0)
        throw std::runtime_error("joint_argmax_device: invalid projected encoder input");
    auto result = decoder_batchers_->joint_device(
        state->slot, enc_proj, frame_offset, T, logit_bias, rnnt_cfg_.vocab_size);
    std::copy(result.tokens.begin(), result.tokens.end(), token_ids);
}

void
RnntModel::predict_and_joint_rnnt_argmax_device(
    RnntStreamState& stream_state, int prev_token, int active_bank,
    const ggml_runtime::DeviceTensor& enc_proj, int frame_offset, int joint_dim, int T,
    int32_t* token_ids, const float* logit_bias) {
    auto* state = dynamic_cast<RnntDecoderState*>(&stream_state);
    if (!state || state->owner != this || state->slot < 0)
        throw std::invalid_argument(
            "predict_and_joint_rnnt_argmax_device: stream state belongs to another engine");
    if (!enc_proj.valid() || joint_dim != rnnt_cfg_.joint_dim || frame_offset < 0 || T <= 0)
        throw std::runtime_error(
            "predict_and_joint_rnnt_argmax_device: invalid projected encoder input");
    if (active_bank != 0 && active_bank != 1)
        throw std::invalid_argument(
            "predict_and_joint_rnnt_argmax_device: active bank must be 0 or 1");
    auto result = decoder_batchers_->predict_joint_rnnt_device(
        state->slot, static_cast<int32_t>(prev_token), active_bank, enc_proj, frame_offset, T,
        logit_bias, rnnt_cfg_.vocab_size, state->needs_reset);
    state->needs_reset = false;
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
        admitted_decode_steps_.load(std::memory_order_acquire));
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
        admitted_decode_steps_.load(std::memory_order_acquire), state->needs_reset);
    state->needs_reset = false;
    *token_id = result.tokens.front();
    *duration_id = result.durations.front();
}

std::unique_ptr<Decoder>
RnntModel::make_transducer_decoder(const DecoderConfig& cfg) {
    if (rnnt_cfg_.is_tdt())
        return std::make_unique<TdtGreedyDecoder>(this);
    return std::make_unique<RnntGreedyDecoder>(this, cfg);
}

int
RnntModel::acquire_decoder_slot(bool& needs_reset) {
    int acquired = -1;
    needs_reset = false;
    std::unique_lock<std::mutex> lock(decoder_slots_mu_);
    const auto claim_free = [&]() -> bool {
        for (int slot = 0; slot < decoder_arena_slots_; ++slot) {
            if (!decoder_slots_used_[static_cast<size_t>(slot)]) {
                decoder_slots_used_[static_cast<size_t>(slot)] = true;
                needs_reset = decoder_slots_need_reset_[static_cast<size_t>(slot)];
                decoder_slots_need_reset_[static_cast<size_t>(slot)] = false;
                acquired = slot;
                return true;
            }
        }
        return false;
    };
    // Block (bounded) when every slot is transiently held rather than failing:
    // teardown frees slots on the RPC's own thread and lags admission under
    // churn. The timeout only fires on genuine over-subscription.
    decoder_slots_cv_.wait_for(lock, std::chrono::seconds(60), claim_free);
    if (acquired < 0)
        throw std::runtime_error("RnntModel: predictor-state arena is full");
    return acquired;
}

void
RnntModel::zero_decoder_slot(int slot) {
    zero_decoder_slots({slot});
}

void
RnntModel::zero_decoder_slots(std::vector<int> slots) {
    if (slots.empty())
        return;
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    std::lock_guard<std::mutex> compute_lock(backend_manager().compute_mutex());
    for (size_t first = 0; first < slots.size();) {
        size_t last = first + 1;
        while (last < slots.size() && slots[last] == slots[last - 1] + 1) ++last;
        const size_t start = static_cast<size_t>(slots[first]);
        const size_t count = last - first;
        for (int bank = 0; bank < 2; ++bank) {
            for (int l = 0; l < rnnt_cfg_.pred_num_layers; ++l) {
                for (const auto& name : {rnnt_h_state_name(bank, l), rnnt_c_state_name(bank, l)}) {
                    auto t =
                        decoder_session_->model_tensor_container->get_tensor_by_name(name).tensor;
                    ggml_backend_tensor_memset(t, 0, start * t->nb[1], count * t->nb[1]);
                }
            }
        }
        auto p =
            decoder_session_->model_tensor_container->get_tensor_by_name(kRnntPredProjectionState)
                .tensor;
        ggml_backend_tensor_memset(p, 0, start * p->nb[1], count * p->nb[1]);
        first = last;
    }
}

void
RnntModel::release_decoder_slot(int slot) {
    {
        std::lock_guard<std::mutex> lock(decoder_slots_mu_);
        decoder_slots_need_reset_[static_cast<size_t>(slot)] = true;
        decoder_slots_used_[static_cast<size_t>(slot)] = false;
    }
    decoder_slots_cv_.notify_one();
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
    int attn_mask_len, std::vector<float>& enc_out, int& T_enc, int prompt_index,
    ggml_runtime::DeviceTensor* device_output) {
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
        tail_input ? num_prompts_ : 0, device_output);
}

}  // namespace nemo_speech::asr
