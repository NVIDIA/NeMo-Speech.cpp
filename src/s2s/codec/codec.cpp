// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

namespace gr = ggml_runtime;

namespace nemo_speech::s2s {

S2SCodec::S2SCodec(gr::BackendManager& bm, const std::string& gguf_path) : bm_(&bm) {
    loader_ = std::make_unique<gr::GGUFLoader>(gguf_path);
    cfg_ = CodecConfig::from_gguf(*loader_);

    decode_module_ = std::make_unique<CodecDecodeWavModule>(cfg_);
    tail_len_ = decode_module_->tail_len();
    decode_session_ = std::make_unique<gr::Session>(bm, decode_module_.get(), loader_.get());
    decode_session_->setup();

    encode_module_ = std::make_unique<CodecEncodeModule>(cfg_);
    encode_session_ = std::make_unique<gr::Session>(bm, encode_module_.get(), loader_.get());
    encode_session_->setup();
}

S2SCodec::~S2SCodec() = default;

S2SCodec::StreamState&
S2SCodec::get_or_create_stream(int64_t id) {
    auto it = streams_.find(id);
    if (it != streams_.end())
        return it->second;
    StreamState& st = streams_[id];
    const int km1 = cfg_.conv_next_kernel - 1;
    st.per_block_cache.resize(cfg_.n_blocks());
    // Decoder block channels: derive from stage_channels in REVERSE stage
    // order (decoder upsamples 1536 -> 768 -> 384); each stage holds
    // num_blocks_per_stage blocks.
    int bi = 0;
    for (int st_i = static_cast<int>(cfg_.stage_channels.size()) - 1; st_i >= 0; st_i--) {
        for (int j = 0; j < cfg_.num_blocks_per_stage; j++) {
            st.per_block_cache[bi++].assign(
                static_cast<size_t>(km1) * cfg_.stage_channels[st_i], 0.0f);
        }
    }
    st.tail_re.assign(static_cast<size_t>(cfg_.n_bins()) * tail_len_, 0.0f);
    st.tail_im.assign(static_cast<size_t>(cfg_.n_bins()) * tail_len_, 0.0f);
    return st;
}

void
S2SCodec::drop_stream(int64_t stream_id) {
    std::lock_guard<std::mutex> lk(streams_mutex_);
    streams_.erase(stream_id);
}

std::vector<float>
S2SCodec::compute_envelope_recip(int n_frames, int* out_trim) const {
    const int n_fft = cfg_.n_fft;
    const int hop = cfg_.hop_length;
    const int pad = (n_fft - hop) / 2;
    int out_t = n_frames;
    for (int r : cfg_.rates) out_t *= r;
    const int half_wav_padding = (tail_len_ / 2) * hop;
    const int eff_t = tail_len_ + out_t;
    const int out_size_full = (eff_t - 1) * hop + n_fft;
    const int out_size_trim = out_size_full - 2 * pad - 2 * half_wav_padding;
    *out_trim = out_size_trim;

    const double TWO_PI = 6.283185307179586476925286766559;
    std::vector<float> w_sq(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        const double w = 0.5 * (1.0 - std::cos(TWO_PI * i / n_fft));
        w_sq[i] = static_cast<float>(w * w);
    }
    std::vector<float> env_full(out_size_full, 0.0f);
    for (int f = 0; f < eff_t; ++f)
        for (int kk = 0; kk < n_fft; ++kk) env_full[f * hop + kk] += w_sq[kk];

    std::vector<float> recip(out_size_trim, 0.0f);
    const float eps = 1e-11f;
    const int off = pad + half_wav_padding;
    for (int i = 0; i < out_size_trim; ++i) {
        const float v = env_full[static_cast<size_t>(off) + i];
        recip[i] = (v > eps) ? (1.0f / v) : 0.0f;
    }
    return recip;
}

void
S2SCodec::decode_wav(
    int64_t stream_id, const int32_t* codes, int n_frames, std::vector<float>& out_wav) {
    // The current codec graph is scalar-stream. Serialize this short stage
    // while also protecting its graph/envelope caches and state map; all
    // other neural stages remain free to batch concurrently.
    std::lock_guard<std::mutex> state_lock(streams_mutex_);
    const int Q = cfg_.n_quantizers;
    const int n_blocks = cfg_.n_blocks();
    const int km1 = cfg_.conv_next_kernel - 1;
    const int n_bins = cfg_.n_bins();
    const int BT = n_frames;

    // Repack codes (n_frames, Q) row-major -> q-major (BT per q contiguous).
    std::vector<int32_t> codes_qbt(static_cast<size_t>(Q) * BT);
    for (int q = 0; q < Q; q++)
        for (int t = 0; t < BT; t++)
            codes_qbt[static_cast<size_t>(q) * BT + t] = codes[static_cast<size_t>(t) * Q + q];

    int out_trim = 0;
    auto env_it = envelope_cache_.find(n_frames);
    if (env_it == envelope_cache_.end()) {
        EnvelopeEntry entry;
        entry.reciprocal = compute_envelope_recip(n_frames, &entry.out_trim);
        env_it = envelope_cache_.emplace(n_frames, std::move(entry)).first;
    }
    out_trim = env_it->second.out_trim;
    const std::vector<float>& env = env_it->second.reciprocal;

    StreamState* st = &get_or_create_stream(stream_id);

    // ---- assemble inputs in the Module's fixed order ----
    std::vector<gr::Session::Input> inputs;
    inputs.push_back({"input.codes", GGML_TYPE_I32, codes_qbt.data(), {BT, Q}});
    char nm[48];
    std::vector<std::string> in_names(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        const int C = static_cast<int>(st->per_block_cache[b].size()) / km1;
        std::snprintf(nm, sizeof(nm), "input.cache_%d", b);
        in_names[b] = nm;
        inputs.push_back({in_names[b], GGML_TYPE_F32, st->per_block_cache[b].data(), {km1, C, 1}});
    }
    inputs.push_back({"input.tail_re", GGML_TYPE_F32, st->tail_re.data(), {n_bins, tail_len_, 1}});
    inputs.push_back({"input.tail_im", GGML_TYPE_F32, st->tail_im.data(), {n_bins, tail_len_, 1}});
    inputs.push_back(
        {"input.envelope", GGML_TYPE_F32, env.data(), {static_cast<int64_t>(env.size())}});

    // ---- outputs: wav by index, caches/tails by name ----
    std::vector<float> wav(out_trim);
    std::vector<std::vector<float>> cache_out(n_blocks);
    std::vector<gr::Session::Output> outputs;
    outputs.push_back({0, "", wav.data(), wav.size() * sizeof(float)});
    std::vector<std::string> out_names(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        cache_out[b].resize(st->per_block_cache[b].size());
        std::snprintf(nm, sizeof(nm), "cache_out_%d", b);
        out_names[b] = nm;
        outputs.push_back(
            {-1, out_names[b], cache_out[b].data(), cache_out[b].size() * sizeof(float)});
    }
    std::vector<float> tail_re_new(st->tail_re.size()), tail_im_new(st->tail_im.size());
    outputs.push_back(
        {-1, "spec_re_tail_out", tail_re_new.data(), tail_re_new.size() * sizeof(float)});
    outputs.push_back(
        {-1, "spec_im_tail_out", tail_im_new.data(), tail_im_new.size() * sizeof(float)});

    decode_session_->run(inputs, outputs);

    // ---- thread state forward ----
    for (int b = 0; b < n_blocks; b++) st->per_block_cache[b] = std::move(cache_out[b]);
    st->tail_re = std::move(tail_re_new);
    st->tail_im = std::move(tail_im_new);

    out_wav.insert(out_wav.end(), wav.begin(), wav.end());
}

std::vector<int32_t>
S2SCodec::compute_silence_codes() {
    // model.py: encode zeros of wav_to_token_ratio * 10 samples. The STFT of
    // zeros is zeros; NeMo's stft uses center padding so
    // spec_t = T_samples / hop + 1. The encoder requires spec_t divisible by
    // prod(rates) (= samples_per_frame / hop); T = 10 frames gives
    // spec_t = 10 * 441 + 1 -> trim the +1 center frame to 4410 like the
    // reference C path does (it receives the torch.stft output and the
    // encoder asserts divisibility; torch.stft(center=True) on
    // frame-multiple input yields spec_t = T/hop + 1 and NeMo's encoder
    // trims internally to a multiple).
    const int n_frames_target = 10;
    const int spec_t = n_frames_target * (cfg_.samples_per_frame() / cfg_.hop_length);
    const int C = cfg_.spec_channels;
    std::vector<float> zeros(static_cast<size_t>(spec_t) * C, 0.0f);

    const int Q = cfg_.n_quantizers;
    std::vector<gr::Session::Input> inputs = {
        {"input.spec", GGML_TYPE_F32, zeros.data(), {spec_t, C, 1}}};
    std::vector<std::vector<int32_t>> per_q(Q);
    std::vector<gr::Session::Output> outputs;
    char nm[32];
    std::vector<std::string> names(Q);
    for (int q = 0; q < Q; q++) {
        per_q[q].resize(n_frames_target);
        std::snprintf(nm, sizeof(nm), "code_%d", q);
        names[q] = nm;
        outputs.push_back({-1, names[q], per_q[q].data(), per_q[q].size() * sizeof(int32_t)});
    }
    encode_session_->run(inputs, outputs);

    // Most common 31-tuple across the 10 frames.
    std::map<std::vector<int32_t>, int> counts;
    for (int t = 0; t < n_frames_target; t++) {
        std::vector<int32_t> tup(Q);
        for (int q = 0; q < Q; q++) tup[q] = per_q[q][t];
        counts[tup]++;
    }
    const std::vector<int32_t>* best = nullptr;
    int best_n = -1;
    for (const auto& kv : counts) {
        if (kv.second > best_n) {
            best_n = kv.second;
            best = &kv.first;
        }
    }
    return *best;
}

}  // namespace nemo_speech::s2s
