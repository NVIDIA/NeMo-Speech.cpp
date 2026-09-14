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

// SileroVadModule - the ggml graph. Consecutive windows and streams share one
// graph run, with one probability emitted per window_size frame.
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
        // dim2 is ordered as step + steps * batch_item.
        auto frame = input_tensors.get_tensor(0);     // (window_size, 1, steps*B)
        auto slot_ids = input_tensors.get_tensor(1);  // i32[B]
        auto buft = frame.buft;
        ggml_context* g = tc->get_ctx_of_buffer_type(buft).ctx;
        auto* mtc = session->model_tensor_container.get();
        const int H = cfg_.lstm_hidden;
        const int B = static_cast<int>(slot_ids.tensor->ne[0]);
        GGML_ASSERT(B > 0 && frame.tensor->ne[2] % B == 0);
        const int steps = static_cast<int>(frame.tensor->ne[2] / B);

        // STFT frontend.
        // reflect-pad context_size each side, fixed Fourier-basis conv, then
        // magnitude over the real/imag halves of the output channels.
        ggml_tensor* basis = mtc->get_tensor_by_name("stft.basis").tensor;
        const int stft_hop = cfg_.stft_filter_length / 2;  // = 128
        ggml_tensor* padded =
            ggml_pad_reflect_1d(g, frame.tensor, cfg_.context_size, cfg_.context_size);
        ggml_tensor* stft = ggml_conv_1d(g, basis, padded, stft_hop, /*padding=*/0, /*d0=*/1);

        const int w = static_cast<int>(stft->ne[0]);  // STFT frames (=4)
        const int cutoff = cfg_.stft_n_basis / 2;     // n_freqs (=129)
        const int frame_batch = static_cast<int>(stft->ne[2]);
        ggml_tensor* re =
            ggml_view_3d(g, stft, w, cutoff, frame_batch, stft->nb[1], stft->nb[2], 0);
        ggml_tensor* im = ggml_view_3d(
            g, stft, w, cutoff, frame_batch, stft->nb[1], stft->nb[2],
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
            cur = ggml_conv_1d(g, wt, cur, cfg_.enc_strides[i], /*padding=*/1, /*d0=*/1);
            cur = ggml_add(g, cur, ggml_reshape_3d(g, b, 1, cfg_.enc_out_channels[i], 1));
            cur = ggml_relu(g, cur);
        }
        // Encoder collapses the STFT frames to width 1; take that frame's
        // 128 channels as the LSTM input column (whisper's [:, :, 0]).
        cur = ggml_cont(g, ggml_view_3d(g, cur, 1, H, frame_batch, cur->nb[1], cur->nb[2], 0));

        // LSTM cell.
        ggml_tensor* h_arena = mtc->get_tensor_by_name("lstm.h_arena").tensor;
        ggml_tensor* c_arena = mtc->get_tensor_by_name("lstm.c_arena").tensor;
        ggml_tensor* h_state = ggml_get_rows(g, h_arena, slot_ids.tensor);  // (H,B)
        ggml_tensor* c_state = ggml_get_rows(g, c_arena, slot_ids.tensor);  // (H,B)
        ggml_tensor* ih_w = mtc->get_tensor_by_name("lstm.ih.weight").tensor;
        ggml_tensor* ih_b = mtc->get_tensor_by_name("lstm.ih.bias").tensor;
        ggml_tensor* hh_w = mtc->get_tensor_by_name("lstm.hh.weight").tensor;
        ggml_tensor* hh_b = mtc->get_tensor_by_name("lstm.hh.bias").tensor;

        ggml_tensor* fc_w = mtc->get_tensor_by_name("final_conv.weight").tensor;  // (H, 1)
        ggml_tensor* fc_b = mtc->get_tensor_by_name("final_conv.bias").tensor;    // (1)
        ggml_tensor* probabilities = nullptr;
        for (int step = 0; step < steps; ++step) {
            auto step_input = ggml_view_3d(
                g, cur, 1, H, B, cur->nb[1], static_cast<size_t>(steps) * cur->nb[2],
                static_cast<size_t>(step) * cur->nb[2]);
            ggml_tensor* x_t = ggml_cont(g, ggml_permute(g, step_input, 1, 0, 2, 3));
            x_t = ggml_reshape_2d(g, x_t, H, B);
            ggml_tensor* inp = ggml_add(g, ggml_mul_mat(g, ih_w, x_t), ih_b);
            ggml_tensor* hid = ggml_add(g, ggml_mul_mat(g, hh_w, h_state), hh_b);
            ggml_tensor* gates = ggml_add(g, inp, hid);

            const size_t hsz = ggml_row_size(gates->type, H);
            auto gate = [&](int index) {
                return ggml_cont(g, ggml_view_2d(g, gates, H, B, gates->nb[1], index * hsz));
            };
            ggml_tensor* i_t = ggml_sigmoid(g, gate(0));
            ggml_tensor* f_t = ggml_sigmoid(g, gate(1));
            ggml_tensor* g_t = ggml_tanh(g, gate(2));
            ggml_tensor* o_t = ggml_sigmoid(g, gate(3));

            c_state = ggml_add(g, ggml_mul(g, f_t, c_state), ggml_mul(g, i_t, g_t));
            h_state = ggml_mul(g, o_t, ggml_tanh(g, c_state));

            ggml_tensor* probability = ggml_mul_mat(g, fc_w, ggml_relu(g, h_state));
            probability = ggml_sigmoid(g, ggml_add(g, probability, fc_b));
            probabilities = probabilities == nullptr
                                ? probability
                                : ggml_concat(g, probabilities, probability, 0);
        }
        ggml_set_name(probabilities, "probabilities");
        ggml_set_output(probabilities);

        ggml_runtime::TensorBag out;
        out.add_tensor(ggml_runtime::ggml_bf_tensor(probabilities, buft));
        // Commit the active recurrent-state rows in-graph.
        out.add_tensor(
            ggml_runtime::ggml_bf_tensor(
                ggml_set_rows(g, h_arena, ggml_cont(g, h_state), slot_ids.tensor), buft));
        out.add_tensor(
            ggml_runtime::ggml_bf_tensor(
                ggml_set_rows(g, c_arena, ggml_cont(g, c_state), slot_ids.tensor), buft));
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
        bool reset;
        std::vector<float> frames;
    };
    VadBatcher(SileroVadModel* model, const BatchingConfig& cfg)
        : model_(model), queue_(cfg, [this](const int& steps, std::vector<Request>&& requests) {
              const int B = static_cast<int>(requests.size());
              const int W = model_->cfg_.window_size;
              const size_t item_size = static_cast<size_t>(W) * steps;
              std::vector<float> frames(item_size * B);
              std::vector<int32_t> slots(static_cast<size_t>(B));
              std::vector<int> reset_slots;
              for (int b = 0; b < B; ++b) {
                  if (requests[b].frames.size() != item_size)
                      throw std::runtime_error("VAD batch contains incompatible window counts");
                  std::copy(
                      requests[b].frames.begin(), requests[b].frames.end(),
                      frames.begin() + static_cast<size_t>(b) * item_size);
                  slots[static_cast<size_t>(b)] = requests[b].slot;
                  if (requests[b].reset)
                      reset_slots.push_back(requests[b].slot);
              }
              if (!reset_slots.empty())
                  model_->zero_slots(std::move(reset_slots));
              std::vector<float> probs(static_cast<size_t>(steps) * B);
              std::vector<ggml_runtime::Session::Output> outputs;
              outputs.push_back({0, "", probs.data(), probs.size() * sizeof(float)});
              model_->session_->run(
                  {{"input.frame", GGML_TYPE_F32, frames.data(), {W, 1, steps * B}},
                   {"vad.slot_ids", GGML_TYPE_I32, slots.data(), {B}}},
                  outputs);
              std::vector<std::vector<float>> result(static_cast<size_t>(B));
              for (int b = 0; b < B; ++b) {
                  result[static_cast<size_t>(b)].assign(
                      probs.begin() + static_cast<size_t>(b) * steps,
                      probs.begin() + static_cast<size_t>(b + 1) * steps);
              }
              return result;
          }) {}
    std::vector<float> run(int slot, const float* frames, int steps, bool reset) {
        Request request;
        request.slot = slot;
        request.reset = reset;
        request.frames.assign(
            frames, frames + static_cast<size_t>(model_->cfg_.window_size) * steps);
        return queue_.run(steps, std::move(request));
    }
    BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    SileroVadModel* model_;
    MicroBatcher<int, Request, std::vector<float>> queue_;
};

SileroVadModel::State::~State() {
    if (owner_ && slot_ >= 0)
        owner_->release_slot(slot_);
}

SileroVadModel::State::State(State&& other) noexcept
    : owner_(other.owner_), slot_(other.slot_), needs_reset_(other.needs_reset_) {
    other.owner_ = nullptr;
    other.slot_ = -1;
    other.needs_reset_ = false;
}

SileroVadModel::State&
SileroVadModel::State::operator=(State&& other) noexcept {
    if (this == &other)
        return *this;
    if (owner_ && slot_ >= 0)
        owner_->release_slot(slot_);
    owner_ = other.owner_;
    slot_ = other.slot_;
    needs_reset_ = other.needs_reset_;
    other.owner_ = nullptr;
    other.slot_ = -1;
    other.needs_reset_ = false;
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
    slots_need_reset_.assign(static_cast<size_t>(arena_slots_), false);
    batcher_ = std::make_unique<VadBatcher>(this, batching);
}

SileroVadModel::~SileroVadModel() = default;

SileroVadModel::State
SileroVadModel::make_state() {
    int acquired = -1;
    bool reset = false;
    {
        std::lock_guard<std::mutex> lock(slots_mu_);
        for (int slot = 0; slot < arena_slots_; ++slot) {
            if (!slots_used_[static_cast<size_t>(slot)]) {
                slots_used_[static_cast<size_t>(slot)] = true;
                reset = slots_need_reset_[static_cast<size_t>(slot)];
                slots_need_reset_[static_cast<size_t>(slot)] = false;
                acquired = slot;
                break;
            }
        }
    }
    if (acquired < 0)
        throw std::runtime_error("SileroVadModel: recurrent-state arena is full");
    return State(this, acquired, reset);
}

void
SileroVadModel::zero_slot(int slot) {
    zero_slots({slot});
}

void
SileroVadModel::zero_slots(std::vector<int> slots) {
    if (slots.empty())
        return;
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    std::lock_guard<std::mutex> compute_lock(bm_->compute_mutex());
    for (size_t first = 0; first < slots.size();) {
        size_t last = first + 1;
        while (last < slots.size() && slots[last] == slots[last - 1] + 1) ++last;
        const size_t start = static_cast<size_t>(slots[first]);
        const size_t count = last - first;
        for (const char* name : {"lstm.h_arena", "lstm.c_arena"}) {
            auto t = session_->model_tensor_container->get_tensor_by_name(name).tensor;
            ggml_backend_tensor_memset(t, 0, start * t->nb[1], count * t->nb[1]);
        }
        first = last;
    }
}

void
SileroVadModel::reset_state(State& state) {
    if (state.owner_ != this || state.slot_ < 0)
        throw std::invalid_argument("SileroVadModel::reset_state: foreign or invalid state");
    zero_slot(state.slot_);
    state.needs_reset_ = false;
}

void
SileroVadModel::release_slot(int slot) {
    std::lock_guard<std::mutex> lock(slots_mu_);
    slots_need_reset_[static_cast<size_t>(slot)] = true;
    slots_used_[static_cast<size_t>(slot)] = false;
}

void
SileroVadModel::infer(
    State& state, const float* frames, int n_windows, std::vector<float>& probabilities) {
    if (state.owner_ != this || state.slot_ < 0)
        throw std::invalid_argument("SileroVadModel::infer: foreign or invalid state");
    if (frames == nullptr || n_windows <= 0)
        throw std::invalid_argument("SileroVadModel::infer: empty window batch");
    probabilities = batcher_->run(state.slot_, frames, n_windows, state.needs_reset_);
    state.needs_reset_ = false;
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

void
SileroVad::run_windows(const float* frames, int n_windows, std::vector<float>& probabilities) {
    model_->infer(recurrent_state_, frames, n_windows, probabilities);
}

int
SileroVad::feed_audio(const float* samples, size_t n_samples, std::vector<float>& out_probs) {
    pending_audio_.insert(pending_audio_.end(), samples, samples + n_samples);

    constexpr size_t kMaxWindowsPerRun = 5;
    const size_t W = static_cast<size_t>(cfg_.window_size);
    int consumed = 0;
    while (pending_audio_.size() >= W) {
        const size_t windows = std::min(kMaxWindowsPerRun, pending_audio_.size() / W);
        std::vector<float> probabilities;
        run_windows(pending_audio_.data(), static_cast<int>(windows), probabilities);
        out_probs.insert(out_probs.end(), probabilities.begin(), probabilities.end());
        const size_t samples_consumed = windows * W;
        pending_audio_.erase(pending_audio_.begin(), pending_audio_.begin() + samples_consumed);
        consumed += static_cast<int>(windows);
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
    std::vector<float> probabilities;
    run_windows(frame_.data(), 1, probabilities);
    out_probs.push_back(probabilities.front());
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
