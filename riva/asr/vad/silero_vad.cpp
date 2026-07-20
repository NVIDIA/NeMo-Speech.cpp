// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "silero_vad.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "runtime.h"

namespace nemo_speech::asr {

// SileroVadModule - the ggml graph. One probability per window_size frame.
//
// Port of whisper.cpp's four build functions (src/whisper.cpp:4519-4653).
// LSTM state is threaded differently: rather than ggml_cpy back into the
// state tensors mid-graph, the new h/c are exposed as named graph outputs
// and re-uploaded as persistent inputs next call (build_graph must not write
// model_tensor_container).
class SileroVadModule : public ggml_runtime::Module {
   public:
    SileroVadModule(const SileroVadConfig& cfg, int arena_slots)
        : cfg_(cfg), arena_slots_(arena_slots) {}
    ~SileroVadModule() override = default;

    void define_tensors(ggml_runtime::Session* session) override {
        auto* tc = session->model_tensor_container.get();
        const int H = cfg_.lstm_hidden;

        // STFT Fourier basis: conv weight [k=filter_length, in=1, out=n_basis].
        // ggml_conv_1d's im2col requires an F16 kernel (CPU asserts
        // src0->type==F16); load_weight converts the F32 GGUF bytes to F16 on
        // upload. Same convention as the runtime's Conv1D module (nn.cpp).
        tc->create_tensor_3d(
            "stft.basis", GGML_TYPE_F16, cfg_.stft_filter_length, 1, cfg_.stft_n_basis);

        // 4 Conv1D encoder layers: weight [k, in, out] (F16 kernel), bias [out].
        for (int i = 0; i < cfg_.n_encoder_layers; i++) {
            tc->create_tensor_3d(
                enc_w(i), GGML_TYPE_F16, cfg_.enc_kernel, cfg_.enc_in_channels[i],
                cfg_.enc_out_channels[i]);
            tc->create_tensor_1d(enc_b(i), GGML_TYPE_F32, cfg_.enc_out_channels[i]);
        }

        // LSTMCell: ih/hh gate-stacked weight [in=H, 4H], bias [4H].
        tc->create_tensor_2d("lstm.ih.weight", GGML_TYPE_F32, H, 4 * H);
        tc->create_tensor_1d("lstm.ih.bias", GGML_TYPE_F32, 4 * H);
        tc->create_tensor_2d("lstm.hh.weight", GGML_TYPE_F32, H, 4 * H);
        tc->create_tensor_1d("lstm.hh.bias", GGML_TYPE_F32, 4 * H);

        // Final pointwise conv stored 2D (in=H, out=1) → runs as mul_mat.
        tc->create_tensor_2d("final_conv.weight", GGML_TYPE_F32, H, 1);
        tc->create_tensor_1d("final_conv.bias", GGML_TYPE_F32, 1);

        // One persistent recurrent-state row per stream slot, selected by the
        // batched `vad.slot_ids` input.
        tc->create_tensor_2d("lstm.h_arena", GGML_TYPE_F32, H, arena_slots_);
        tc->create_tensor_2d("lstm.c_arena", GGML_TYPE_F32, H, arena_slots_);
    }

    void set_data(ggml_runtime::Session* session) override {
        session->load_weight("stft.basis");
        for (int i = 0; i < cfg_.n_encoder_layers; i++) {
            session->load_weight(enc_w(i));
            session->load_weight(enc_b(i));
        }
        session->load_weight("lstm.ih.weight");
        session->load_weight("lstm.ih.bias");
        session->load_weight("lstm.hh.weight");
        session->load_weight("lstm.hh.bias");
        session->load_weight("final_conv.weight");
        session->load_weight("final_conv.bias");
        for (const char* name : {"lstm.h_arena", "lstm.c_arena"}) {
            auto t = session->model_tensor_container->get_tensor_by_name(name);
            ggml_backend_tensor_memset(t.tensor, 0, 0, ggml_nbytes(t.tensor));
        }
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override {
        auto frame = input_tensors.get_tensor(0);     // (window_size, 1, B)
        auto slot_ids = input_tensors.get_tensor(1);  // i32[B]
        auto buft = frame.buft;
        ggml_context* g = tc->get_ctx_of_buffer_type(buft).ctx;
        auto* mtc = session->model_tensor_container.get();
        const int H = cfg_.lstm_hidden;
        auto conv1d_batch = [&](ggml_tensor* weight, ggml_tensor* x, int stride, int padding) {
            if (x->ne[2] == 1)
                return ggml_conv_1d(g, weight, x, stride, padding, /*d0=*/1);
            ggml_tensor* all = nullptr;
            for (int64_t b = 0; b < x->ne[2]; ++b) {
                auto item = ggml_view_3d(
                    g, x, x->ne[0], x->ne[1], 1, x->nb[1], x->nb[2],
                    static_cast<size_t>(b) * x->nb[2]);
                auto y = ggml_conv_1d(g, weight, ggml_cont(g, item), stride, padding, /*d0=*/1);
                all = all == nullptr ? y : ggml_concat(g, all, y, 2);
            }
            return all;
        };

        // STFT frontend.
        // reflect-pad context_size each side, fixed Fourier-basis conv, then
        // magnitude over the real/imag halves of the output channels.
        ggml_tensor* basis = mtc->get_tensor_by_name("stft.basis").tensor;
        const int stft_hop = cfg_.stft_filter_length / 2;  // = 128
        ggml_tensor* padded =
            ggml_pad_reflect_1d(g, frame.tensor, cfg_.context_size, cfg_.context_size);
        ggml_tensor* stft = conv1d_batch(basis, padded, stft_hop, /*padding=*/0);

        const int w = static_cast<int>(stft->ne[0]);  // STFT frames (=4)
        const int cutoff = cfg_.stft_n_basis / 2;     // n_freqs (=129)
        const int B = static_cast<int>(stft->ne[2]);
        ggml_tensor* re = ggml_view_3d(g, stft, w, cutoff, B, stft->nb[1], stft->nb[2], 0);
        ggml_tensor* im = ggml_view_3d(
            g, stft, w, cutoff, B, stft->nb[1], stft->nb[2],
            static_cast<size_t>(cutoff) * stft->nb[1]);
        // Channel-half views carry the parent STFT stride; CUDA unary kernels
        // require dense rows once B adds another outer dimension.
        re = ggml_cont(g, re);
        im = ggml_cont(g, im);
        ggml_tensor* mag =
            ggml_sqrt(g, ggml_add(g, ggml_sqr(g, re), ggml_sqr(g, im)));  // (w, n_freqs)

        // Conv1D encoder: convolution, bias, then ReLU.
        ggml_tensor* cur = mag;
        for (int i = 0; i < cfg_.n_encoder_layers; i++) {
            ggml_tensor* wt = mtc->get_tensor_by_name(enc_w(i)).tensor;
            ggml_tensor* b = mtc->get_tensor_by_name(enc_b(i)).tensor;
            cur = conv1d_batch(wt, cur, cfg_.enc_strides[i], /*padding=*/1);
            cur = ggml_add(g, cur, ggml_reshape_3d(g, b, 1, cfg_.enc_out_channels[i], 1));
            cur = ggml_relu(g, cur);
        }
        // Encoder collapses the STFT frames to width 1; take that frame's
        // 128 channels as the LSTM input column (whisper's [:, :, 0]).
        cur = ggml_cont(g, ggml_view_3d(g, cur, 1, H, B, cur->nb[1], cur->nb[2], 0));

        // LSTM cell.
        ggml_tensor* h_arena = mtc->get_tensor_by_name("lstm.h_arena").tensor;
        ggml_tensor* c_arena = mtc->get_tensor_by_name("lstm.c_arena").tensor;
        ggml_tensor* h_state = ggml_get_rows(g, h_arena, slot_ids.tensor);  // (H,B)
        ggml_tensor* c_state = ggml_get_rows(g, c_arena, slot_ids.tensor);  // (H,B)
        ggml_tensor* ih_w = mtc->get_tensor_by_name("lstm.ih.weight").tensor;
        ggml_tensor* ih_b = mtc->get_tensor_by_name("lstm.ih.bias").tensor;
        ggml_tensor* hh_w = mtc->get_tensor_by_name("lstm.hh.weight").tensor;
        ggml_tensor* hh_b = mtc->get_tensor_by_name("lstm.hh.bias").tensor;

        ggml_tensor* x_t = ggml_cont(g, ggml_permute(g, cur, 1, 0, 2, 3));
        x_t = ggml_reshape_2d(g, x_t, H, B);
        ggml_tensor* inp = ggml_add(g, ggml_mul_mat(g, ih_w, x_t), ih_b);
        ggml_tensor* hid = ggml_add(g, ggml_mul_mat(g, hh_w, h_state), hh_b);
        ggml_tensor* gates = ggml_add(g, inp, hid);  // (4H,B)

        const size_t hsz = ggml_row_size(gates->type, H);
        auto gate = [&](int index) {
            return ggml_cont(g, ggml_view_2d(g, gates, H, B, gates->nb[1], index * hsz));
        };
        ggml_tensor* i_t = ggml_sigmoid(g, gate(0));
        ggml_tensor* f_t = ggml_sigmoid(g, gate(1));
        ggml_tensor* g_t = ggml_tanh(g, gate(2));
        ggml_tensor* o_t = ggml_sigmoid(g, gate(3));

        ggml_tensor* c_out = ggml_add(g, ggml_mul(g, f_t, c_state), ggml_mul(g, i_t, g_t));  // (H)
        ggml_tensor* h_out = ggml_mul(g, o_t, ggml_tanh(g, c_out));                          // (H)

        // Output head: ReLU, pointwise projection, bias, then sigmoid.
        ggml_tensor* fc_w = mtc->get_tensor_by_name("final_conv.weight").tensor;  // (H, 1)
        ggml_tensor* fc_b = mtc->get_tensor_by_name("final_conv.bias").tensor;    // (1)
        ggml_tensor* prob = ggml_mul_mat(g, fc_w, ggml_relu(g, h_out));           // (1, 1)
        prob = ggml_sigmoid(g, ggml_add(g, prob, fc_b));
        ggml_set_name(prob, "prob");
        ggml_set_output(prob);

        ggml_runtime::TensorBag out;
        out.add_tensor(ggml_runtime::ggml_bf_tensor(prob, buft));
        // Commit the active recurrent-state rows in-graph.
        out.add_tensor(ggml_runtime::ggml_bf_tensor(
            ggml_set_rows(g, h_arena, ggml_cont(g, h_out), slot_ids.tensor), buft));
        out.add_tensor(ggml_runtime::ggml_bf_tensor(
            ggml_set_rows(g, c_arena, ggml_cont(g, c_out), slot_ids.tensor), buft));
        return out;
    }

   private:
    static std::string enc_w(int i) { return "encoder." + std::to_string(i) + ".conv.weight"; }
    static std::string enc_b(int i) { return "encoder." + std::to_string(i) + ".conv.bias"; }

    SileroVadConfig cfg_;
    int arena_slots_;
};

class SileroVadModel::VadBatcher {
   public:
    struct Request {
        int slot;
        std::vector<float> frame;
    };
    VadBatcher(SileroVadModel* model, const BatchingConfig& cfg)
        : model_(model), queue_(cfg, [this](const int&, std::vector<Request>&& requests) {
              const int B = static_cast<int>(requests.size());
              const int W = model_->cfg_.window_size;
              std::vector<float> frames(static_cast<size_t>(W) * B);
              std::vector<int32_t> slots(static_cast<size_t>(B));
              for (int b = 0; b < B; ++b) {
                  std::copy(
                      requests[b].frame.begin(), requests[b].frame.end(),
                      frames.begin() + static_cast<size_t>(b) * W);
                  slots[static_cast<size_t>(b)] = requests[b].slot;
              }
              std::vector<float> probs(static_cast<size_t>(B));
              std::vector<ggml_runtime::Session::Output> outputs;
              outputs.push_back({0, "", probs.data(), probs.size() * sizeof(float)});
              model_->session_->run(
                  {{"input.frame", GGML_TYPE_F32, frames.data(), {W, 1, B}},
                   {"vad.slot_ids", GGML_TYPE_I32, slots.data(), {B}}},
                  outputs);
              return probs;
          }) {}
    float run(int slot, const float* frame) {
        Request request;
        request.slot = slot;
        request.frame.assign(frame, frame + model_->cfg_.window_size);
        return queue_.run(0, std::move(request));
    }
    BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    SileroVadModel* model_;
    MicroBatcher<int, Request, float> queue_;
};

SileroVadModel::State::~State() {
    if (owner_ && slot_ >= 0)
        owner_->release_slot(slot_);
}

SileroVadModel::State::State(State&& other) noexcept : owner_(other.owner_), slot_(other.slot_) {
    other.owner_ = nullptr;
    other.slot_ = -1;
}

SileroVadModel::State&
SileroVadModel::State::operator=(State&& other) noexcept {
    if (this == &other)
        return *this;
    if (owner_ && slot_ >= 0)
        owner_->release_slot(slot_);
    owner_ = other.owner_;
    slot_ = other.slot_;
    other.owner_ = nullptr;
    other.slot_ = -1;
    return *this;
}

SileroVadModel::SileroVadModel(
    ggml_runtime::BackendManager& bm, const std::string& gguf_path, const BatchingConfig& batching)
    : bm_(&bm), arena_slots_(std::max(1, batching.state_arena_slots)) {
    loader_ = std::make_unique<ggml_runtime::GGUFLoader>(gguf_path);

    const std::string arch = loader_->get_str("general.architecture", "");
    if (arch != "vad") {
        throw std::runtime_error(
            "SileroVad: '" + gguf_path + "' has general.architecture='" + arch +
            "', expected 'vad'. Convert via convert_model.py.");
    }

    cfg_.sample_rate = loader_->get_u32("vad.sample_rate", cfg_.sample_rate);
    cfg_.window_size = loader_->get_u32("vad.window_size", cfg_.window_size);
    cfg_.context_size = loader_->get_u32("vad.context_size", cfg_.context_size);
    cfg_.lstm_hidden = loader_->get_u32("vad.lstm.hidden_size", cfg_.lstm_hidden);
    cfg_.stft_filter_length = loader_->get_u32("vad.stft.filter_length", cfg_.stft_filter_length);
    cfg_.stft_n_basis = loader_->get_u32("vad.stft.n_basis", cfg_.stft_n_basis);
    cfg_.n_freqs = loader_->get_u32("vad.stft.n_freqs", cfg_.n_freqs);
    cfg_.n_encoder_layers = loader_->get_u32("vad.n_encoder_layers", cfg_.n_encoder_layers);

    module_ = std::make_unique<SileroVadModule>(cfg_, arena_slots_);
    session_ = std::make_unique<ggml_runtime::Session>(bm, module_.get(), loader_.get());
    session_->set_run_cache_capacity(
        static_cast<size_t>(std::max(8, std::min(32, batching.max_batch_size))));
    session_->setup();
    slots_used_.assign(static_cast<size_t>(arena_slots_), false);
    batcher_ = std::make_unique<VadBatcher>(this, batching);
}

SileroVadModel::~SileroVadModel() = default;

SileroVadModel::State
SileroVadModel::make_state() {
    std::lock_guard<std::mutex> lock(slots_mu_);
    for (int slot = 0; slot < arena_slots_; ++slot) {
        if (!slots_used_[static_cast<size_t>(slot)]) {
            slots_used_[static_cast<size_t>(slot)] = true;
            return State(this, slot);
        }
    }
    throw std::runtime_error("SileroVadModel: recurrent-state arena is full");
}

void
SileroVadModel::zero_slot(int slot) {
    std::lock_guard<std::mutex> compute_lock(bm_->compute_mutex());
    for (const char* name : {"lstm.h_arena", "lstm.c_arena"}) {
        auto t = session_->model_tensor_container->get_tensor_by_name(name).tensor;
        const size_t bytes = static_cast<size_t>(cfg_.lstm_hidden) * sizeof(float);
        ggml_backend_tensor_memset(t, 0, static_cast<size_t>(slot) * t->nb[1], bytes);
    }
}

void
SileroVadModel::reset_state(State& state) {
    if (state.owner_ != this || state.slot_ < 0)
        throw std::invalid_argument("SileroVadModel::reset_state: foreign or invalid state");
    zero_slot(state.slot_);
}

void
SileroVadModel::release_slot(int slot) {
    zero_slot(slot);
    std::lock_guard<std::mutex> lock(slots_mu_);
    slots_used_[static_cast<size_t>(slot)] = false;
}

float
SileroVadModel::infer(State& state, const float* frame) {
    if (state.owner_ != this || state.slot_ < 0)
        throw std::invalid_argument("SileroVadModel::infer: foreign or invalid state");
    return batcher_->run(state.slot_, frame);
}

BatchMetrics
SileroVadModel::batch_metrics() const {
    return batcher_->metrics();
}

SileroVad::SileroVad(ggml_runtime::BackendManager& bm, const std::string& gguf_path)
    : SileroVad(std::make_shared<SileroVadModel>(bm, gguf_path)) {}

SileroVad::SileroVad(std::shared_ptr<SileroVadModel> model) : model_(std::move(model)) {
    if (!model_) {
        throw std::invalid_argument("SileroVad: shared model is null");
    }
    cfg_ = model_->config();
    recurrent_state_ = model_->make_state();

    frame_.assign(cfg_.window_size, 0.0f);
}

SileroVad::~SileroVad() = default;

float
SileroVad::run_window(const float* frame) {
    return model_->infer(recurrent_state_, frame);
}

int
SileroVad::feed_audio(const float* samples, size_t n_samples, std::vector<float>& out_probs) {
    pending_audio_.insert(pending_audio_.end(), samples, samples + n_samples);

    const size_t W = static_cast<size_t>(cfg_.window_size);
    size_t off = 0;
    int consumed = 0;
    while (off + W <= pending_audio_.size()) {
        out_probs.push_back(run_window(pending_audio_.data() + off));
        off += W;
        consumed++;
    }
    if (off > 0) {
        pending_audio_.erase(pending_audio_.begin(), pending_audio_.begin() + off);
    }
    return consumed;
}

int
SileroVad::flush(std::vector<float>& out_probs) {
    if (pending_audio_.empty()) {
        return 0;
    }
    std::fill(frame_.begin(), frame_.end(), 0.0f);
    const size_t n = std::min(pending_audio_.size(), static_cast<size_t>(cfg_.window_size));
    std::copy(pending_audio_.begin(), pending_audio_.begin() + n, frame_.begin());
    out_probs.push_back(run_window(frame_.data()));
    pending_audio_.clear();
    return 1;
}

void
SileroVad::reset() {
    model_->reset_state(recurrent_state_);
    pending_audio_.clear();
    in_speech_ = false;
    windows_binarized_ = 0;
    last_speech_frame_ = -1;
    speech_.clear();
    speech_base_ = 0;
    prob_scratch_.clear();
}

void
SileroVad::set_binarizer(float onset, float offset, int hop_length) {
    onset_ = onset;
    offset_ = offset;
    hop_ = hop_length;
}

void
SileroVad::binarize_window(float p, int64_t k) {
    // Hysteresis (riva BinarizeVADPredictions): enter speech when p > onset,
    // leave when p < offset, hold otherwise.
    if (p > onset_) {
        in_speech_ = true;
    } else if (p < offset_) {
        in_speech_ = false;
    }
    // Record across the mel frames this window covers (contiguous, no gaps).
    // Indices are global; storage is offset by speech_base_ (see
    // discard_timeline_before).
    const int win = cfg_.window_size;
    const int64_t f0 = (k * win) / hop_;
    const int64_t f1 = ((k + 1) * win) / hop_;
    if (speech_base_ + static_cast<int64_t>(speech_.size()) < f1)
        speech_.resize(f1 - speech_base_, 0);
    for (int64_t f = std::max(f0, speech_base_); f < f1; f++)
        speech_[f - speech_base_] = in_speech_ ? 1 : 0;
    if (in_speech_ && f1 > f0)
        last_speech_frame_ = f1 - 1;
}

void
SileroVad::observe_audio(const float* samples, size_t n) {
    prob_scratch_.clear();
    if (n > 0)
        feed_audio(samples, n, prob_scratch_);
    for (float p : prob_scratch_) binarize_window(p, windows_binarized_++);
}

void
SileroVad::flush_timeline() {
    prob_scratch_.clear();
    flush(prob_scratch_);
    for (float p : prob_scratch_) binarize_window(p, windows_binarized_++);
}

bool
SileroVad::frame_speech(int64_t g) const {
    if (g < speech_base_)
        return false;  // before the stream, or discarded
    if (g < speech_base_ + static_cast<int64_t>(speech_.size()))
        return speech_[g - speech_base_] != 0;
    return in_speech_;  // provisional for not-yet-decided frames
}

void
SileroVad::discard_timeline_before(int64_t g) {
    if (g <= speech_base_)
        return;
    const int64_t end = speech_base_ + static_cast<int64_t>(speech_.size());
    const int64_t new_base = std::min(g, end);
    speech_.erase(speech_.begin(), speech_.begin() + (new_base - speech_base_));
    speech_base_ = new_base;
}

}  // namespace nemo_speech::asr
