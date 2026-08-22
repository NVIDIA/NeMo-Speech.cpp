// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "cache_aware_encoder.h"

#include <ggml.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "nvtx_utils.h"

namespace nemo_speech::asr {

// Compose the streaming encoder with an optional model-specific tail in one
// Session. The encoder's cache-copy side outputs are retained so state feedback
// still executes even though the user-visible output is the tail result.
class CacheAwareEncoder::EncoderRoot : public ggml_runtime::Module {
   public:
    EncoderRoot(FastConformerEncoder* encoder, ggml_runtime::Module* tail)
        : encoder_(encoder), tail_(tail) {}

    void define_tensors(ggml_runtime::Session* session) override {
        encoder_->define_tensors(session);
        if (tail_)
            tail_->define_tensors(session);
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag input,
        ggml_runtime::TensorContainer* tc) override {
        // One-shot fill of the chunk-invariant tensors (pos projections +
        // repacked dw-conv weights), dispatched by input name (same pattern
        // as RnntDecoderStages). Bypasses the tail: its outputs are the
        // per-layer store nodes, not encoder frames.
        if (std::string(input.get_tensor(0).tensor->name) ==
            FastConformerEncoder::kPrecomputeTrigger) {
            return encoder_->build_precompute_graph(session, tc);
        }
        auto enc = encoder_->build_graph(session, input, tc);
        if (!tail_)
            return enc;

        ggml_runtime::TensorBag tail_in;
        tail_in.add_tensor(enc.get_tensor(0));
        // input[0]=mel, input[1]=mask, input[2]=slot ids, input[3]=K/V ring
        // heads. Optional input[4] carries the prompt one-hot matrix.
        if (input.tensor_count() > 4)
            tail_in.add_tensor(input.get_tensor(4));
        auto tail_out = tail_->build_graph(session, tail_in, tc);

        ggml_runtime::TensorBag out;
        out.add_tensor(tail_out.get_tensor(0));
        for (size_t i = 1; i < enc.tensor_count(); ++i) out.add_tensor(enc.get_tensor(i));
        return out;
    }

    void set_data(ggml_runtime::Session* session) override {
        encoder_->set_data(session);
        if (tail_)
            tail_->set_data(session);
    }

   private:
    FastConformerEncoder* encoder_;
    ggml_runtime::Module* tail_;
};

class CacheAwareEncoder::EncoderBatcher {
   public:
    struct Key {
        int mel_frames, mask_len, tail_dim;
        bool device_output;
        bool operator==(const Key& o) const {
            return mel_frames == o.mel_frames && mask_len == o.mask_len && tail_dim == o.tail_dim &&
                   device_output == o.device_output;
        }
    };
    struct Request {
        int slot;
        int ring_head;
        bool reset;
        std::vector<float> mel, mask, tail;
    };
    struct Result {
        std::vector<float> output;
        ggml_runtime::DeviceTensor device_output;
        int T = 0;
    };
    EncoderBatcher(CacheAwareEncoder* owner, const BatchingConfig& cfg)
        : owner_(owner), queue_(cfg, [this](const Key& key, std::vector<Request>&& req) {
              const ggml_nvtx::range nvtx("asr.encoder.batch");
              const int B = static_cast<int>(req.size());
              const int M = owner_->n_mels_;
              const int Tout = owner_->cfg_.cache_chunk_frames;
              std::vector<float> mel(static_cast<size_t>(M) * key.mel_frames * B);
              std::vector<float> mask(static_cast<size_t>(key.mask_len) * B);
              // K and V cache planes use the same stream slots. Upload the two
              // identical columns once so one 3D GET_ROWS/SET_ROWS operation
              // can move both planes without a device-side repeat kernel.
              std::vector<int32_t> slots(static_cast<size_t>(2 * B));
              std::vector<int32_t> ring_heads(static_cast<size_t>(B));
              std::vector<int> reset_slots;
              std::vector<float> tail;
              if (key.tail_dim > 0)
                  tail.resize(static_cast<size_t>(key.tail_dim) * Tout * B);
              for (int b = 0; b < B; ++b) {
                  std::copy(
                      req[b].mel.begin(), req[b].mel.end(),
                      mel.begin() + static_cast<size_t>(b) * M * key.mel_frames);
                  std::copy(
                      req[b].mask.begin(), req[b].mask.end(),
                      mask.begin() + static_cast<size_t>(b) * key.mask_len);
                  if (key.tail_dim > 0)
                      std::copy(
                          req[b].tail.begin(), req[b].tail.end(),
                          tail.begin() + static_cast<size_t>(b) * key.tail_dim * Tout);
                  slots[b] = req[b].slot;
                  slots[static_cast<size_t>(B + b)] = req[b].slot;
                  ring_heads[b] = req[b].ring_head;
                  if (req[b].reset)
                      reset_slots.push_back(req[b].slot);
              }
              if (!reset_slots.empty())
                  owner_->zero_slots(std::move(reset_slots));
              const bool device_only = key.device_output && B == 1;
              std::vector<float> packed;
              if (!device_only)
                  packed.resize(static_cast<size_t>(owner_->output_dim_) * Tout * B);
              std::vector<ggml_runtime::Session::Input> inputs = {
                  {"input.features", GGML_TYPE_F32, mel.data(), {M, key.mel_frames, 1, B}},
                  {owner_->encoder_->attn_mask_name(),
                   GGML_TYPE_F32,
                   mask.data(),
                   {key.mask_len, B}},
                  {"encoder.slot_ids", GGML_TYPE_I32, slots.data(), {B, 2}},
                  {"encoder.cache.ring_heads", GGML_TYPE_I32, ring_heads.data(), {B}}};
              if (key.tail_dim > 0)
                  inputs.push_back(
                      {"encoder.tail.input", GGML_TYPE_F32, tail.data(), {key.tail_dim, Tout, B}});
              if (device_only) {
                  for (auto& input : inputs) input.synchronous_upload = true;
              }
              ggml_runtime::DeviceTensor device_output;
              std::vector<ggml_runtime::Session::Output> outputs(1);
              outputs[0].index = 0;
              if (device_only) {
                  outputs[0].device_tensor = &device_output;
              } else {
                  outputs[0].host_buffer = packed.data();
                  outputs[0].nbytes = packed.size() * sizeof(float);
              }
              {
                  const ggml_nvtx::range compute_nvtx("asr.encoder.session");
                  owner_->session_->run(inputs, outputs);
              }
              const size_t item = static_cast<size_t>(owner_->output_dim_) * Tout;
              std::vector<Result> result(static_cast<size_t>(B));
              for (int b = 0; b < B; ++b) {
                  result[b].T = Tout;
                  if (device_only) {
                      result[b].device_output = device_output;
                  } else {
                      result[b].output.assign(
                          packed.begin() + static_cast<size_t>(b) * item,
                          packed.begin() + static_cast<size_t>(b + 1) * item);
                  }
              }
              return result;
          }) {}
    Result run(
        int slot, int ring_head, const float* mel, int mel_frames, const float* mask, int mask_len,
        const float* tail, int tail_dim, bool reset, bool device_output) {
        Request r;
        r.slot = slot;
        r.ring_head = ring_head;
        r.reset = reset;
        r.mel.assign(mel, mel + static_cast<size_t>(owner_->n_mels_) * mel_frames);
        r.mask.assign(mask, mask + mask_len);
        if (tail)
            r.tail.assign(
                tail, tail + static_cast<size_t>(tail_dim) * owner_->cfg_.cache_chunk_frames);
        return queue_.run({mel_frames, mask_len, tail ? tail_dim : 0, device_output}, std::move(r));
    }
    BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    CacheAwareEncoder* owner_;
    MicroBatcher<Key, Request, Result> queue_;
};

CacheAwareEncoder::State::~State() {
    if (owner_ && slot_ >= 0)
        owner_->release_slot(slot_);
}
CacheAwareEncoder::State::State(State&& o) noexcept
    : owner_(o.owner_), slot_(o.slot_), ring_head_(o.ring_head_), needs_reset_(o.needs_reset_) {
    o.owner_ = nullptr;
    o.slot_ = -1;
    o.ring_head_ = 0;
    o.needs_reset_ = false;
}
CacheAwareEncoder::State&
CacheAwareEncoder::State::operator=(State&& o) noexcept {
    if (this == &o)
        return *this;
    if (owner_ && slot_ >= 0)
        owner_->release_slot(slot_);
    owner_ = o.owner_;
    slot_ = o.slot_;
    ring_head_ = o.ring_head_;
    needs_reset_ = o.needs_reset_;
    o.owner_ = nullptr;
    o.slot_ = -1;
    o.ring_head_ = 0;
    o.needs_reset_ = false;
    return *this;
}

CacheAwareEncoder::CacheAwareEncoder(
    ggml_runtime::BackendManager& bm, ggml_runtime::GGUFLoader* loader,
    const EncoderConfig& base_cfg, int n_mels, ggml_runtime::Module* output_tail, int output_dim,
    const BatchingConfig& batching)
    : bm_(&bm), loader_(loader), n_mels_(n_mels), output_tail_(output_tail),
      output_dim_(output_dim > 0 ? output_dim : base_cfg.d_model), batching_(batching),
      arena_slots_(std::max(1, batching.state_arena_slots)), cfg_(base_cfg) {
    batcher_ = std::make_unique<EncoderBatcher>(this, batching_);
}

CacheAwareEncoder::~CacheAwareEncoder() = default;

void
CacheAwareEncoder::set_right_ctx(int R) {
    std::lock_guard<std::mutex> lock(mu_);
    if (session_) {
        // Session already built (an earlier stream). Same R is a no-op - all
        // streams share the one Session. A different R post-build would mean
        // swapping topology mid-process; not supported.
        if (right_ctx_ >= 0 && right_ctx_ != R) {
            throw std::runtime_error("set_right_ctx with different R after Session built");
        }
        return;
    }
    right_ctx_ = R;
}

void
CacheAwareEncoder::ensure_session() {
    // Called under mu_.
    if (session_)
        return;
    // Same derivation CacheStreamRunner uses, so the geometry can't drift.
    cfg_ = make_cache_aware_config(cfg_, right_ctx_);
    cfg_.cache_state_slots = arena_slots_;
    encoder_ = std::make_unique<FastConformerEncoder>("encoder", cfg_);
    root_ = std::make_unique<EncoderRoot>(encoder_.get(), output_tail_);
    session_ = std::make_unique<ggml_runtime::Session>(*bm_, root_.get(), loader_);
    // Serialized tensor-planar Q8 weights are flagged at load (and rejected
    // off-CUDA) by this hook; the runtime itself stays model-agnostic.
    session_->set_weight_load_hook(planar_q8_weight_load_hook());
    // Retain common partial-batch graph shapes without letting cache residency
    // grow with an arbitrarily large configured maximum.
    session_->set_run_cache_capacity(
        static_cast<size_t>(std::max(16, std::min(32, batching_.max_batch_size))));
    session_->setup();  // declares the indexed K/V/conv cache arenas
    slots_used_.assign(static_cast<size_t>(arena_slots_), false);
    slots_need_reset_.assign(static_cast<size_t>(arena_slots_), true);
#ifdef NEMO_SPEECH_FUSED_RELPOS_ATTN
    const int d_k = cfg_.d_model / cfg_.n_heads;
    ring_cache_enabled_ = session_->params.use_gpu && cfg_.cache_left_ctx > 0 &&
                          cfg_.cache_chunk_frames > 0 && (d_k & (d_k - 1)) == 0;
#endif

    // Fill the chunk-invariant per-layer tensors once: positional
    // projections (replacing a per-chunk per-layer linear_pos GEMM) and the
    // channel-inner dw-conv weight repacks. Chunk graphs read them as
    // constants (LayerCacheIO::pos_proj / dw_conv_w_ct). Must precede the
    // first encode(); Session::run serializes on the backend compute mutex
    // internally.
    float trigger = 0.0f;
    std::vector<ggml_runtime::Session::Input> inputs = {
        {FastConformerEncoder::kPrecomputeTrigger, GGML_TYPE_F32, &trigger, {1}}};
    std::vector<ggml_runtime::Session::Output> outputs;
    session_->run(inputs, outputs);
}

CacheAwareEncoder::State
CacheAwareEncoder::make_state() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensure_session();
    }
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
        throw std::runtime_error("CacheAwareEncoder: state arena is full");
    return State(this, acquired, reset);
}

void
CacheAwareEncoder::zero_slot(int slot) {
    zero_slots({slot});
}

void
CacheAwareEncoder::zero_slots(std::vector<int> slots) {
    if (slots.empty())
        return;
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensure_session();
    }
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    std::lock_guard<std::mutex> compute_lock(bm_->compute_mutex());
    for (size_t first = 0; first < slots.size();) {
        size_t last = first + 1;
        while (last < slots.size() && slots[last] == slots[last - 1] + 1) ++last;
        const size_t start = static_cast<size_t>(slots[first]);
        const size_t count = last - first;
        for (int l = 0; l < cfg_.n_layers; ++l) {
            auto kv =
                session_->model_tensor_container->get_tensor_by_name(encoder_->kv_cache_name(l))
                    .tensor;
            for (int plane = 0; plane < 2; ++plane)
                ggml_backend_tensor_memset(
                    kv, 0, static_cast<size_t>(plane) * kv->nb[2] + start * kv->nb[1],
                    count * kv->nb[1]);
            auto conv =
                session_->model_tensor_container->get_tensor_by_name(encoder_->conv_cache_name(l))
                    .tensor;
            ggml_backend_tensor_memset(conv, 0, start * conv->nb[1], count * conv->nb[1]);
        }
        first = last;
    }
}

void
CacheAwareEncoder::reset_state(State& state) {
    if (state.owner_ != this || state.slot_ < 0)
        throw std::invalid_argument("invalid cache state");
    zero_slot(state.slot_);
    state.ring_head_ = 0;
    state.needs_reset_ = false;
}

void
CacheAwareEncoder::release_slot(int slot) {
    std::lock_guard<std::mutex> lock(slots_mu_);
    slots_need_reset_[static_cast<size_t>(slot)] = true;
    slots_used_[static_cast<size_t>(slot)] = false;
}

int
CacheAwareEncoder::chunk_frames() {
    std::lock_guard<std::mutex> lock(mu_);
    ensure_session();
    return cfg_.cache_chunk_frames;
}

void
CacheAwareEncoder::encode(
    State& state, const float* mel, int n_mel_frames, const float* attn_mask, int attn_mask_len,
    std::vector<float>& enc_out, int& T_enc, const float* tail_input, int tail_input_dim,
    ggml_runtime::DeviceTensor* device_output) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        ensure_session();
    }
    if (state.owner_ != this || state.slot_ < 0)
        throw std::invalid_argument("invalid cache state");
    if (tail_input && tail_input_dim <= 0)
        throw std::runtime_error("CacheAwareEncoder: invalid tail input width");
    auto result = batcher_->run(
        state.slot_, state.ring_head_, mel, n_mel_frames, attn_mask, attn_mask_len, tail_input,
        tail_input_dim, state.needs_reset_, device_output != nullptr);
    state.needs_reset_ = false;
    if (ring_cache_enabled_ && result.T > 0) {
        const int advance = std::min(cfg_.cache_left_ctx, result.T);
        state.ring_head_ = (state.ring_head_ + advance) % cfg_.cache_left_ctx;
    }
    enc_out = std::move(result.output);
    if (device_output != nullptr)
        *device_output = result.device_output;
    T_enc = result.T;
}

BatchMetrics
CacheAwareEncoder::batch_metrics() const {
    return batcher_->metrics();
}

}  // namespace nemo_speech::asr
