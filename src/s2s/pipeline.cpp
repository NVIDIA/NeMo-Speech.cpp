// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace gr = ggml_runtime;
using Json = nlohmann::ordered_json;

namespace nemo_speech::s2s {

// Round f32 -> bf16 -> f32 (round-to-nearest-even). The reference computes
// all channel embeddings and their weighted sums in torch.bfloat16; the LLM's
// pad-vs-word pacing decision is sensitive to this rounding.
static inline float
bf16_round(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    std::memcpy(&x, &u, 4);
    return x;
}

static inline void
bf16_round_vec(float* v, size_t n) {
    for (size_t i = 0; i < n; i++) v[i] = bf16_round(v[i]);
}

namespace {

// Strip "<TOOL_ACK_MESSAGES>...</TOOL_ACK_MESSAGES>" from the prompt;
// returns the stripped prompt and the inner JSON (or "").
std::pair<std::string, std::string>
strip_tool_acks(const std::string& prompt) {
    const std::string open = "<TOOL_ACK_MESSAGES>", close = "</TOOL_ACK_MESSAGES>";
    const size_t a = prompt.find(open);
    if (a == std::string::npos)
        return {prompt, ""};
    const size_t b = prompt.find(close, a);
    if (b == std::string::npos)
        return {prompt, ""};
    std::string inner = prompt.substr(a + open.size(), b - a - open.size());
    std::string stripped = prompt.substr(0, a) + prompt.substr(b + close.size());
    return {stripped, inner};
}

std::string
trim_copy(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string
under_thousand_to_words(int value) {
    static const char* small[] = {"zero",    "one",     "two",       "three",    "four",
                                  "five",    "six",     "seven",     "eight",    "nine",
                                  "ten",     "eleven",  "twelve",    "thirteen", "fourteen",
                                  "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    static const char* tens[] = {"",      "",      "twenty",  "thirty", "forty",
                                 "fifty", "sixty", "seventy", "eighty", "ninety"};
    std::string out;
    if (value >= 100) {
        out = std::string(small[value / 100]) + " hundred";
        value %= 100;
        if (value)
            out += " ";
    }
    if (value >= 20) {
        out += tens[value / 10];
        if (value % 10)
            out += std::string("-") + small[value % 10];
    } else if (value > 0 || out.empty()) {
        out += small[value];
    }
    return out;
}

std::string
integer_to_words(int64_t value) {
    if (value == 0)
        return "zero";
    if (value == std::numeric_limits<int64_t>::min())
        return "minus nine quintillion two hundred twenty-three quadrillion three hundred "
               "seventy-two trillion thirty-six billion eight hundred fifty-four million "
               "seven hundred seventy-five thousand eight hundred eight";
    std::string out;
    if (value < 0) {
        out = "minus ";
        value = -value;
    }
    static const std::pair<int64_t, const char*> scales[] = {
        {1000000000000000000LL, "quintillion"},
        {1000000000000000LL, "quadrillion"},
        {1000000000000LL, "trillion"},
        {1000000000LL, "billion"},
        {1000000LL, "million"},
        {1000LL, "thousand"}};
    for (const auto& scale : scales) {
        if (value < scale.first)
            continue;
        if (!out.empty() && out.back() != ' ')
            out += " ";
        out += under_thousand_to_words(static_cast<int>(value / scale.first));
        out += " ";
        out += scale.second;
        value %= scale.first;
    }
    if (value) {
        if (!out.empty() && out.back() != ' ')
            out += " ";
        out += under_thousand_to_words(static_cast<int>(value));
    }
    return out;
}

std::string
decimal_to_words(double value) {
    std::ostringstream ss;
    ss << value;
    const std::string repr = ss.str();
    const size_t dot = repr.find('.');
    if (dot == std::string::npos || repr.find_first_of("eE") != std::string::npos)
        return integer_to_words(static_cast<int64_t>(value));
    std::string out = integer_to_words(static_cast<int64_t>(value));
    out += " point";
    for (size_t i = dot + 1; i < repr.size(); ++i) {
        if (repr[i] >= '0' && repr[i] <= '9')
            out += " " + integer_to_words(repr[i] - '0');
    }
    return out;
}

void
replace_all(std::string& value, const std::string& needle, const std::string& replacement) {
    size_t pos = 0;
    while ((pos = value.find(needle, pos)) != std::string::npos) {
        value.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

void
normalize_tool_json(Json& value) {
    if (value.is_object()) {
        for (auto& item : value.items()) normalize_tool_json(item.value());
    } else if (value.is_array()) {
        for (auto& item : value) normalize_tool_json(item);
    } else if (value.is_number_integer()) {
        value = integer_to_words(value.get<int64_t>());
    } else if (value.is_number_unsigned()) {
        const uint64_t number = value.get<uint64_t>();
        value = number <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
                    ? integer_to_words(static_cast<int64_t>(number))
                    : value.dump();
    } else if (value.is_number_float()) {
        value = decimal_to_words(value.get<double>());
    } else if (value.is_string()) {
        std::string text = value.get<std::string>();
        replace_all(text, "—", " ");
        replace_all(text, "–", " ");
        const size_t degree = text.find("°");
        if (degree != std::string::npos && text.find("°", degree + 2) == std::string::npos) {
            const std::string number_text = trim_copy(text.substr(0, degree));
            const std::string scale = trim_copy(text.substr(degree + std::string("°").size()));
            char* end = nullptr;
            const double number = std::strtod(number_text.c_str(), &end);
            if (end && *end == '\0' && !number_text.empty() && scale.size() == 1 &&
                (scale[0] == 'C' || scale[0] == 'c' || scale[0] == 'F' || scale[0] == 'f')) {
                text = decimal_to_words(number) + " degrees " +
                       ((scale[0] == 'C' || scale[0] == 'c') ? "Celsius" : "Fahrenheit");
            }
        }
        value = text;
    }
}

// Match the model's Python JSON serialization: ASCII escaped, comma-space,
// and colon-space. Spacing is significant because this string is tokenized.
std::string
python_json_dump(const Json& value) {
    if (value.is_object()) {
        std::string out = "{";
        bool first = true;
        for (const auto& item : value.items()) {
            if (!first)
                out += ", ";
            first = false;
            out += Json(item.key()).dump(-1, ' ', true) + ": " + python_json_dump(item.value());
        }
        return out + "}";
    }
    if (value.is_array()) {
        std::string out = "[";
        for (size_t i = 0; i < value.size(); ++i) {
            if (i)
                out += ", ";
            out += python_json_dump(value[i]);
        }
        return out + "]";
    }
    return value.dump(-1, ' ', true);
}

}  // namespace

// The model samples acoustic tokens naturally unless a reproducible seed is
// requested. A nonzero S2S_TTS_SEED produces a deterministic per-step stream.
static uint64_t
tts_seed_for_step(int decoder_global_step) {
    static const long long base = [] {
        const char* e = std::getenv("S2S_TTS_SEED");
        return e ? std::atoll(e) : 0LL;
    }();
    return base == 0 ? 0 : static_cast<uint64_t>(base + decoder_global_step);
}

class S2SPipeline::LLMStepBatcher {
   public:
    struct Request {
        S2SStream* stream = nullptr;
        std::vector<float> audio_row;
        int step = 0;
        int audio_idx = 0;
    };

    LLMStepBatcher(S2SPipeline* owner, const asr::BatchingConfig& cfg)
        : owner_(owner), queue_(cfg, [this](const int&, std::vector<Request>&& requests) {
              const int B = static_cast<int>(requests.size());
              const int H = owner_->llm_heads_->hidden_size();
              const int V = owner_->llm_->n_vocab();
              const int FV = owner_->llm_heads_->fn_out_dim();
              std::vector<int32_t> channel_ids(static_cast<size_t>(2) * B);
              for (int b = 0; b < B; ++b) {
                  const auto& req = requests[static_cast<size_t>(b)];
                  const auto& st = *req.stream;
                  channel_ids[static_cast<size_t>(2) * b] =
                      req.step == 0 ? owner_->pad_id_
                                    : static_cast<int32_t>(st.text_tokens[req.step - 1]);
                  channel_ids[static_cast<size_t>(2) * b + 1] =
                      req.step == 0 ? owner_->pad_id_
                                    : static_cast<int32_t>(st.function_tokens[req.step - 1]);
              }

              std::vector<float> channel_emb(static_cast<size_t>(2) * B * H);
              owner_->llm_heads_->embed_tokens(channel_ids.data(), 2 * B, channel_emb.data());
              std::vector<float> combined(static_cast<size_t>(B) * H);
              std::vector<int64_t> seq_ids(static_cast<size_t>(B));
              for (int b = 0; b < B; ++b) {
                  const auto& req = requests[static_cast<size_t>(b)];
                  const float* text_emb = req.step == 0
                                              ? owner_->bos_like_emb_.data()
                                              : channel_emb.data() + static_cast<size_t>(2 * b) * H;
                  const float* fn_emb =
                      req.step == 0 ? owner_->pad_emb_.data()
                                    : channel_emb.data() + static_cast<size_t>(2 * b + 1) * H;
                  if (req.step != 0) {
                      bf16_round_vec(
                          channel_emb.data() + static_cast<size_t>(2 * b) * H,
                          static_cast<size_t>(2) * H);
                  }
                  float* dst = combined.data() + static_cast<size_t>(b) * H;
                  for (int i = 0; i < H; ++i) {
                      float acc = bf16_round(
                          bf16_round(req.audio_row[static_cast<size_t>(i)]) *
                          owner_->cfg_.user_channel_weight);
                      acc = bf16_round(
                          acc + bf16_round(text_emb[i] * owner_->cfg_.text_channel_weight));
                      dst[i] = bf16_round(
                          acc + bf16_round(fn_emb[i] * owner_->cfg_.function_channel_weight));
                  }
                  seq_ids[static_cast<size_t>(b)] = req.stream->seq_id;
              }

              const auto t0 = std::chrono::steady_clock::now();
              std::vector<float> text_logits(static_cast<size_t>(B) * V);
              std::vector<float> hidden(static_cast<size_t>(B) * H);
              owner_->llm_->step_batch(
                  seq_ids.data(), combined.data(), B, text_logits.data(), hidden.data());
              const auto t1 = std::chrono::steady_clock::now();
              bf16_round_vec(text_logits.data(), text_logits.size());
              bf16_round_vec(hidden.data(), hidden.size());
              std::vector<float> fn_logits(static_cast<size_t>(B) * FV);
              owner_->llm_heads_->function_head(hidden.data(), B, fn_logits.data());
              bf16_round_vec(fn_logits.data(), fn_logits.size());
              const auto t2 = std::chrono::steady_clock::now();

              const bool dbg_tok = std::getenv("S2S_DEBUG_TOKENS") != nullptr;
              for (int b = 0; b < B; ++b) {
                  const auto& req = requests[static_cast<size_t>(b)];
                  S2SStream& st = *req.stream;
                  const float* fn_row = fn_logits.data() + static_cast<size_t>(b) * FV;
                  int fn_token = 0;
                  for (int i = 1; i < FV; ++i)
                      if (fn_row[i] > fn_row[fn_token])
                          fn_token = i;
                  const int sampled = owner_->sample_text_token(
                      text_logits.data() + static_cast<size_t>(b) * V, st);
                  st.text_tokens[req.step] =
                      st.fn_state == S2SStream::FnState::Idle ? sampled : owner_->pad_id_;
                  st.function_tokens[req.step] = fn_token;
                  if (dbg_tok)
                      std::fprintf(
                          stderr, "[s2s-tok] seq=%lld t=%d text=%lld fn=%d audio_idx=%d\n",
                          static_cast<long long>(st.seq_id), req.step,
                          static_cast<long long>(st.text_tokens[req.step]), fn_token,
                          req.audio_idx);

                  if (st.fn_state == S2SStream::FnState::SpeakingAck) {
                      if (!st.fn_ack_tokens.empty()) {
                          st.text_tokens[req.step] = st.fn_ack_tokens.front();
                          st.fn_ack_tokens.pop_front();
                      }
                      if (st.fn_ack_tokens.empty()) {
                          st.fn_state = st.fn_response_tokens.empty()
                                            ? S2SStream::FnState::WaitingForResponse
                                            : S2SStream::FnState::ProcessResponse;
                          st.fn_frames_in_state = 0;
                      }
                  } else if (st.fn_state == S2SStream::FnState::ProcessResponse) {
                      if (!st.fn_response_tokens.empty()) {
                          st.function_tokens[req.step] = st.fn_response_tokens.front();
                          st.fn_response_tokens.pop_front();
                      }
                      if (st.fn_response_tokens.empty())
                          owner_->reset_fn_call_state(st);
                  }
              }
              if (std::getenv("S2S_DEBUG_TIMING"))
                  std::fprintf(
                      stderr, "[s2s-batch] llm batch=%d decode=%.2f fnhead=%.2f\n", B,
                      std::chrono::duration<double, std::milli>(t1 - t0).count(),
                      std::chrono::duration<double, std::milli>(t2 - t1).count());
              return std::vector<uint8_t>(static_cast<size_t>(B), 0);
          }) {}

    void run(S2SStream& st, const float* audio_row, int step, int audio_idx) {
        Request request;
        request.stream = &st;
        request.audio_row.assign(
            audio_row, audio_row + static_cast<size_t>(owner_->llm_heads_->hidden_size()));
        request.step = step;
        request.audio_idx = audio_idx;
        (void)queue_.run(0, std::move(request));
    }
    asr::BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    S2SPipeline* owner_;
    asr::MicroBatcher<int, Request, uint8_t> queue_;
};

class S2SPipeline::TTSStepBatcher {
   public:
    struct Request {
        S2SStream* stream = nullptr;
        int step = 0;
    };

    TTSStepBatcher(S2SPipeline* owner, const asr::BatchingConfig& cfg)
        : owner_(owner), queue_(cfg, [this](const int&, std::vector<Request>&& requests) {
              const int B = static_cast<int>(requests.size());
              const int Q = owner_->codec_->config().n_quantizers;
              const int H = owner_->tts_->n_embd();
              std::vector<int32_t> codes(static_cast<size_t>(B) * Q);
              std::vector<int32_t> text(static_cast<size_t>(B));
              std::vector<float> text_mask(static_cast<size_t>(B), 1.0f);
              std::vector<float> bos_mask(static_cast<size_t>(B), 1e-20f);
              std::vector<int> cond_slots(static_cast<size_t>(B));
              std::vector<int> uncond_slots(static_cast<size_t>(B));
              std::vector<uint64_t> seeds(static_cast<size_t>(B));

              for (int b = 0; b < B; ++b) {
                  const auto& req = requests[static_cast<size_t>(b)];
                  S2SStream& st = *req.stream;
                  const int32_t subword = static_cast<int32_t>(st.text_tokens[req.step]);
                  text[static_cast<size_t>(b)] = subword;
                  std::vector<int32_t> step_codes = st.tts_last_codes;
                  if (subword == owner_->eos_id_ &&
                      (st.tts_agent_idle || st.tts_force_silence[req.step]))
                      step_codes = owner_->silence_codes_;
                  const bool speaking_ack = st.fn_state == S2SStream::FnState::SpeakingAck;
                  if (owner_->cfg_.tts_force_silence_on_pad && subword == owner_->pad_id_ &&
                      !speaking_ack) {
                      const bool tail_done = owner_->cfg_.tts_pad_tail_ratio > 0.0f &&
                                             !st.tts_agent_idle &&
                                             st.tts_in_turn_pads > owner_->cfg_.tts_pad_tail_ratio *
                                                                       st.tts_in_turn_content;
                      if (st.tts_agent_idle || tail_done)
                          step_codes = owner_->silence_codes_;
                  }
                  const int32_t* output_codes = subword == owner_->bos_id_
                                                    ? owner_->silence_codes_.data()
                                                    : step_codes.data();
                  std::memcpy(
                      st.audio_tokens.data() + static_cast<size_t>(req.step) * Q, output_codes,
                      static_cast<size_t>(Q) * sizeof(int32_t));
                  std::copy(
                      step_codes.begin(), step_codes.end(),
                      codes.begin() + static_cast<size_t>(b) * Q);

                  if (subword == owner_->bos_id_) {
                      st.tts_agent_idle = false;
                      st.tts_in_turn_content = 0;
                      st.tts_in_turn_pads = 0;
                  } else if (subword == owner_->eos_id_) {
                      st.tts_agent_idle = true;
                      st.tts_in_turn_content = 0;
                      st.tts_in_turn_pads = 0;
                  } else if (subword == owner_->pad_id_) {
                      if (!st.tts_agent_idle)
                          ++st.tts_in_turn_pads;
                  } else {
                      ++st.tts_in_turn_content;
                      st.tts_in_turn_pads = 0;
                  }
                  cond_slots[static_cast<size_t>(b)] = st.tts_cond_slot;
                  uncond_slots[static_cast<size_t>(b)] = st.tts_uncond_slot;
                  seeds[static_cast<size_t>(b)] = tts_seed_for_step(req.step);
              }

              const auto t0 = std::chrono::steady_clock::now();
              std::vector<float> cond(static_cast<size_t>(B) * H);
              std::vector<float> uncond(static_cast<size_t>(B) * H);
              owner_->tts_embedder_->embed_pair(
                  codes.data(), text.data(), text_mask.data(), bos_mask.data(), B, cond.data(),
                  uncond.data());
              const auto t1 = std::chrono::steady_clock::now();
              std::vector<float> h_cond(static_cast<size_t>(B) * H);
              std::vector<float> h_uncond(static_cast<size_t>(B) * H);
              // Side-network execution is partitioned separately from the
              // llama.cpp backbone to preserve deterministic sampling.
              constexpr int kTTSDeterministicBatch = 4;
              owner_->tts_->step_pairs(
                  cond_slots.data(), uncond_slots.data(), cond.data(), uncond.data(), B,
                  h_cond.data(), h_uncond.data());
              const auto t2 = std::chrono::steady_clock::now();

              const bool dbg_parity = std::getenv("S2S_DEBUG_BATCH_PARITY") != nullptr;
              if (dbg_parity && B > 1) {
                  auto max_abs_diff = [H](const float* lhs, const float* rhs) {
                      float result = 0.0f;
                      for (int i = 0; i < H; ++i)
                          result = std::max(result, std::abs(lhs[i] - rhs[i]));
                      return result;
                  };
                  const float* cond0 = cond.data();
                  const float* uncond0 = uncond.data();
                  const float* h_cond0 = h_cond.data();
                  const float* h_uncond0 = h_uncond.data();
                  for (int b = 0; b < B; ++b) {
                      const auto& req = requests[static_cast<size_t>(b)];
                      const bool codes_equal = std::equal(
                          codes.begin(), codes.begin() + Q,
                          codes.begin() + static_cast<size_t>(b) * Q);
                      std::fprintf(
                          stderr,
                          "[s2s-parity] seq=%lld step=%d text=%d codes_eq0=%d "
                          "embed_c=%.9g embed_u=%.9g hidden_c=%.9g hidden_u=%.9g\n",
                          static_cast<long long>(req.stream->seq_id), req.step,
                          text[static_cast<size_t>(b)], codes_equal ? 1 : 0,
                          max_abs_diff(cond0, cond.data() + static_cast<size_t>(b) * H),
                          max_abs_diff(uncond0, uncond.data() + static_cast<size_t>(b) * H),
                          max_abs_diff(h_cond0, h_cond.data() + static_cast<size_t>(b) * H),
                          max_abs_diff(h_uncond0, h_uncond.data() + static_cast<size_t>(b) * H));
                  }
              }
              std::vector<int32_t> sampled(static_cast<size_t>(B) * Q);
              for (int begin = 0; begin < B; begin += kTTSDeterministicBatch) {
                  const int count = std::min(kTTSDeterministicBatch, B - begin);
                  owner_->tts_sampler_->sample_batch(
                      h_cond.data() + static_cast<size_t>(begin) * H,
                      h_uncond.data() + static_cast<size_t>(begin) * H, owner_->cfg_.guidance_scale,
                      seeds.data() + begin, count, sampled.data() + static_cast<size_t>(begin) * Q);
              }
              const auto t3 = std::chrono::steady_clock::now();
              const bool dbg_codes = std::getenv("S2S_DEBUG_AUDIO_CODES") != nullptr;
              for (int b = 0; b < B; ++b) {
                  auto& dst = requests[static_cast<size_t>(b)].stream->tts_last_codes;
                  dst.assign(
                      sampled.begin() + static_cast<size_t>(b) * Q,
                      sampled.begin() + static_cast<size_t>(b + 1) * Q);
                  if (dbg_codes) {
                      std::fprintf(
                          stderr, "[s2s-code] seq=%lld step=%d codes=",
                          static_cast<long long>(requests[static_cast<size_t>(b)].stream->seq_id),
                          requests[static_cast<size_t>(b)].step);
                      for (int q = 0; q < Q; ++q)
                          std::fprintf(stderr, "%s%d", q ? "," : "", dst[static_cast<size_t>(q)]);
                      std::fputc('\n', stderr);
                  }
              }
              if (std::getenv("S2S_DEBUG_TIMING"))
                  std::fprintf(
                      stderr, "[s2s-batch] tts batch=%d embed=%.2f backbone=%.2f sampler=%.2f\n", B,
                      std::chrono::duration<double, std::milli>(t1 - t0).count(),
                      std::chrono::duration<double, std::milli>(t2 - t1).count(),
                      std::chrono::duration<double, std::milli>(t3 - t2).count());
              return std::vector<uint8_t>(static_cast<size_t>(B), 0);
          }) {}

    void run(S2SStream& st, int step) { (void)queue_.run(0, Request{&st, step}); }
    asr::BatchMetrics metrics() const { return queue_.metrics(); }

   private:
    S2SPipeline* owner_;
    asr::MicroBatcher<int, Request, uint8_t> queue_;
};

S2SStream::~S2SStream() {
    extract_cancelled = true;
    inject_cancelled = true;
    if (fast_path_thread.joinable())
        fast_path_thread.join();
}

S2SPipeline::S2SPipeline(gr::BackendManager& bm, const S2SPipelineConfig& cfg)
    : cfg_(cfg), bm_(&bm) {
    auto env_int = [](const char* name, int current) {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : current;
    };
    auto env_float = [](const char* name, float current) {
        const char* value = std::getenv(name);
        return value ? std::strtof(value, nullptr) : current;
    };
    auto env_bool = [](const char* name, bool current) {
        const char* value = std::getenv(name);
        if (!value)
            return current;
        const std::string v(value);
        return v != "0" && v != "false" && v != "False" && v != "no";
    };
    cfg_.top_p = env_float("LLM_TOP_P", cfg_.top_p);
    cfg_.temperature = env_float("LLM_TEMPERATURE", cfg_.temperature);
    cfg_.repetition_penalty = env_float("LLM_REPETITION_PENALTY", cfg_.repetition_penalty);
    cfg_.max_tool_tokens = env_int("MAX_TOOL_TOKENS", cfg_.max_tool_tokens);
    cfg_.rnnt_eos_silence_frames = env_int("RNNT_EOS_SILENCE_FRAMES", cfg_.rnnt_eos_silence_frames);
    cfg_.rnnt_fc_interrupt_frames =
        env_int("RNNT_FC_INTERRUPT_FRAMES", cfg_.rnnt_fc_interrupt_frames);
    cfg_.rnnt_display_gate_frames =
        env_int("RNNT_DISPLAY_GATE_FRAMES", cfg_.rnnt_display_gate_frames);
    cfg_.rnnt_display_fallback_clear_frames =
        env_int("RNNT_DISPLAY_FALLBACK_CLEAR_FRAMES", cfg_.rnnt_display_fallback_clear_frames);
    cfg_.rnnt_display_max_symbols =
        env_int("RNNT_DISPLAY_MAX_SYMBOLS", cfg_.rnnt_display_max_symbols);
    cfg_.rnnt_max_symbols = env_int("RNNT_MAX_SYMBOLS", cfg_.rnnt_max_symbols);
    cfg_.rnnt_punct_bias_enabled =
        env_bool("RNNT_PUNCT_BIAS_ENABLED", cfg_.rnnt_punct_bias_enabled);
    cfg_.rnnt_punct_bias_increment =
        env_float("RNNT_PUNCT_BIAS_INCREMENT", cfg_.rnnt_punct_bias_increment);
    cfg_.rnnt_punct_bias_min_silence_frames =
        env_int("RNNT_PUNCT_BIAS_MIN_SILENCE_FRAMES", cfg_.rnnt_punct_bias_min_silence_frames);
    cfg_.rnnt_bou_min_frames = env_int("RNNT_BOU_MIN_FRAMES", cfg_.rnnt_bou_min_frames);
    cfg_.rnnt_bou_min_frames_first_turn =
        env_int("RNNT_BOU_MIN_FRAMES_FIRST_TURN", cfg_.rnnt_bou_min_frames_first_turn);
    cfg_.rnnt_barge_in_frames = env_int("RNNT_BARGE_IN_FRAMES", cfg_.rnnt_barge_in_frames);
    cfg_.rnnt_density_alpha = env_float("RNNT_DENSITY_ALPHA", cfg_.rnnt_density_alpha);
    cfg_.rnnt_density_threshold = env_float("RNNT_DENSITY_THRESHOLD", cfg_.rnnt_density_threshold);
    cfg_.rnnt_density_low_min = env_int("RNNT_DENSITY_LOW_MIN", cfg_.rnnt_density_low_min);
    cfg_.rnnt_noise_reset_frames = env_int("RNNT_NOISE_RESET_FRAMES", cfg_.rnnt_noise_reset_frames);
    cfg_.rnnt_tts_ratio_cap = env_float("RNNT_TTS_RATIO_CAP", cfg_.rnnt_tts_ratio_cap);
    cfg_.rnnt_tts_min_tokens = env_int("RNNT_TTS_MIN_TOKENS", cfg_.rnnt_tts_min_tokens);
    if (const char* value = std::getenv("RNNT_MAX_AGENT_RESPONSE_SEC")) {
        const float seconds = std::strtof(value, nullptr);
        cfg_.rnnt_max_agent_response_frames = seconds > 0 ? static_cast<int>(seconds / 0.08f) : 0;
    }
    cfg_.tts_force_silence_on_pad =
        env_bool("S2S_INFERENCE_FORCE_SPEECH_SILENCE_ON_PAD", cfg_.tts_force_silence_on_pad);
    cfg_.tts_pad_tail_ratio = env_float("S2S_TTS_PAD_TAIL_RATIO", cfg_.tts_pad_tail_ratio);
    cfg_.max_streams = env_int("S2S_MAX_STREAMS", cfg_.max_streams);
    if (cfg_.max_streams < 1)
        throw std::invalid_argument("S2SPipeline: max_streams must be positive");

    batching_cfg_.enabled = cfg_.max_streams > 1;
    batching_cfg_.max_batch_size = cfg_.max_streams;
    batching_cfg_.max_queue_delay_us = env_int("S2S_BATCH_QUEUE_DELAY_US", 1000);
    batching_cfg_.ingress_cohort_delay_us = env_int("S2S_INGRESS_COHORT_DELAY_US", 2000);
    batching_cfg_.max_queue_depth = std::max(64, cfg_.max_streams * 8);
    // S2S deliberately owns two independent RNNT predictors per conversation:
    // turn evidence and user-visible transcript display.
    batching_cfg_.state_arena_slots = cfg_.max_streams * 2;

    perception_ = std::make_unique<S2SPerception>(bm, cfg.perception_gguf, batching_cfg_);
    llm_heads_ = std::make_unique<LLMHeads>(bm, cfg.llm_aux_gguf);
    llm_ = std::make_unique<LLMBackbone>(cfg.llm_gguf, cfg_.max_streams);
    tts_ = std::make_unique<EarTTSBackbone>(cfg.tts_backbone_gguf, cfg_.max_streams);
    // Embedder and sampler share the side-network GGUF (~500 MB) via one loader.
    auto side_loader = std::make_shared<ggml_runtime::GGUFLoader>(cfg.tts_side_gguf);
    tts_embedder_ = std::make_unique<EarTTSEmbedder>(bm, side_loader);
    tts_sampler_ = std::make_unique<EarTTSSampler>(bm, side_loader);
    tts_prompt_ = std::make_unique<EarTTSPrompt>(cfg.tts_prompt_gguf);
    codec_ = std::make_unique<S2SCodec>(bm, cfg.codec_gguf);

    cfg_.user_channel_weight = llm_heads_->user_channel_weight();
    cfg_.text_channel_weight = llm_heads_->text_channel_weight();
    cfg_.function_channel_weight = llm_heads_->function_channel_weight();
    if (tts_sampler_->has_default_guidance())
        cfg_.guidance_scale = tts_sampler_->default_guidance();

    // Every conversation uses the same immutable voice prompt. Decode it once
    // into reserved llama sequence rows; new streams clone the exact KV state.
    // The prompt-final hidden state and deterministic step-0 noise are also
    // identical, so cache the first acoustic frame codes once.
    {
        const int T = tts_prompt_->length();
        const int H = tts_->n_embd();
        std::vector<float> cond(static_cast<size_t>(T) * H), uncond(cond.size());
        tts_embedder_->embed_pair(
            tts_prompt_->acoustic_tokens().data(), tts_prompt_->text_tokens().data(),
            tts_prompt_->text_mask().data(), tts_prompt_->bos_mask().data(), T, cond.data(),
            uncond.data(),
            tts_prompt_->has_audio_prompt_latent() ? tts_prompt_->audio_prompt_latent().data()
                                                   : nullptr);
        std::vector<float> h_cond(H), h_uncond(H);
        tts_->initialize_prompt_template(
            cond.data(), uncond.data(), T, h_cond.data(), h_uncond.data());
        cfg_.max_steps = std::min(cfg_.max_steps, tts_->step_capacity());
        if (cfg_.max_steps <= 0)
            throw std::runtime_error("S2SPipeline: EarTTS prompt exhausts the position budget");
        if (tts_sampler_->config().num_quantizers != codec_->config().n_quantizers)
            throw std::runtime_error(
                "S2SPipeline: TTS sampler and codec disagree on num_quantizers");
        tts_initial_codes_.resize(tts_sampler_->config().num_quantizers);
        tts_sampler_->sample(
            h_cond.data(), h_uncond.data(), cfg_.guidance_scale, tts_seed_for_step(0), 1,
            tts_initial_codes_.data());
    }

    // The model runtime overrides the raw Hugging Face config after loading:
    //   bos="<s>" (1), eos="</s>" (2), pad="<SPECIAL_12>" (12).
    // The GGUF tokenizer metadata still reflects the un-overridden config
    // (eos=12, pad=0), so resolve the literal tokens directly.
    const auto bos = llm_->text_to_ids("<s>");
    const auto eos = llm_->text_to_ids("</s>");
    const auto pad = llm_->text_to_ids("<SPECIAL_12>");
    const auto sotc = llm_->text_to_ids("<SPECIAL_20>");
    const auto eotc = llm_->text_to_ids("<SPECIAL_21>");
    if (bos.size() != 1 || eos.size() != 1 || pad.size() != 1 || sotc.size() != 1 ||
        eotc.size() != 1)
        throw std::runtime_error("S2SPipeline: required special marker is not a single token");
    bos_id_ = bos.front();
    eos_id_ = eos.front();
    pad_id_ = pad.front();
    sotc_id_ = sotc.front();
    eotc_id_ = eotc.front();

    // RNN-T piece vocab (JSON string array; index == token id). The S2S
    // perception GGUF carries no vocab strings, so the transcript pieces
    // come from the bundle's rnnt_tokenizer/vocab.json.
    if (!cfg.rnnt_vocab_json.empty()) {
        std::ifstream input(cfg.rnnt_vocab_json, std::ios::binary);
        if (!input)
            throw std::runtime_error("S2SPipeline: cannot open " + cfg.rnnt_vocab_json);
        const Json pieces = Json::parse(input, nullptr, false);
        if (!pieces.is_array())
            throw std::runtime_error("S2SPipeline: rnnt vocab is not a JSON array");
        rnnt_vocab_.reserve(pieces.size());
        for (const auto& piece : pieces) {
            if (!piece.is_string())
                throw std::runtime_error("S2SPipeline: rnnt vocab entry is not a string");
            rnnt_vocab_.push_back(piece.get<std::string>());
        }
    }

    if (cfg_.rnnt_punct_bias_enabled) {
        const std::pair<const char*, float> punct[] = {
            {".", 1.5f}, {"?", 1.5f}, {",", 1.0f}, {"!", 1.0f}};
        for (const auto& item : punct) {
            for (const std::string& variant :
                 {std::string(item.first), std::string("▁") + item.first}) {
                const auto it = std::find(rnnt_vocab_.begin(), rnnt_vocab_.end(), variant);
                if (it == rnnt_vocab_.end())
                    continue;
                const int32_t id = static_cast<int32_t>(it - rnnt_vocab_.begin());
                if (std::find(rnnt_punct_ids_.begin(), rnnt_punct_ids_.end(), id) ==
                    rnnt_punct_ids_.end()) {
                    rnnt_punct_ids_.push_back(id);
                    rnnt_punct_increments_.push_back(item.second);
                }
            }
        }
    }

    // Silence codes (TTS eos substitution) + channel pad embeddings.
    silence_codes_ = codec_->compute_silence_codes();
    const int H = llm_heads_->hidden_size();
    pad_emb_.resize(H);
    llm_heads_->embed_tokens(&pad_id_, 1, pad_emb_.data());
    bf16_round_vec(pad_emb_.data(), H);
    // "_get_bos_embedding" in the reference embeds pad_id as well; keep a
    // distinct buffer for clarity but identical contents.
    bos_like_emb_ = pad_emb_;

    // llama.cpp's library API executes the batch supplied by its caller; it
    // does not collect independent S2S request threads (that scheduler lives
    // in llama-server). These thin coordinators rendezvous one pipeline wave
    // and submit native multi-sequence batches to the two backbones.
    llm_step_batcher_ = std::make_unique<LLMStepBatcher>(this, batching_cfg_);
    tts_step_batcher_ = std::make_unique<TTSStepBatcher>(this, batching_cfg_);
    ingress_batch_coordinator_ = std::make_unique<asr::IngressBatchCoordinator>(batching_cfg_);
}

S2SPipeline::~S2SPipeline() = default;

std::unique_ptr<S2SStream>
S2SPipeline::create_stream(int64_t seq_id) {
    int active = active_streams_.load(std::memory_order_relaxed);
    do {
        if (active >= cfg_.max_streams)
            throw std::runtime_error("S2SPipeline: maximum concurrent streams reached");
    } while (!active_streams_.compare_exchange_weak(
        active, active + 1, std::memory_order_acq_rel, std::memory_order_relaxed));

    std::unique_ptr<S2SStream> st;
    try {
        st = std::unique_ptr<S2SStream>(new S2SStream(seq_id));
        st->text_tokens.assign(cfg_.max_steps, pad_id_);
        st->function_tokens.assign(cfg_.max_steps, pad_id_);
        st->tts_force_silence.assign(cfg_.max_steps, 0);
        st->audio_tokens.assign(
            static_cast<size_t>(cfg_.max_steps) * codec_->config().n_quantizers, 0);
        const auto tts_pair = tts_->allocate_pair();
        st->tts_cond_slot = tts_pair.first;
        st->tts_uncond_slot = tts_pair.second;
        st->tts_last_codes = tts_initial_codes_;
        st->tts_started = true;
        const auto& rc = perception_->rnnt_config();
        for (auto* state : {&st->rnnt_eou, &st->rnnt_display}) {
            state->engine_state = perception_->rnnt_engine().make_rnnt_stream_state();
            state->prev_token = rc.blank_id;
            state->active_bank = 0;
            state->predictor_valid = false;
        }
        st->started = true;
        return st;
    }
    catch (...) {
        if (st && st->tts_started)
            tts_->free_pair(st->tts_cond_slot, st->tts_uncond_slot);
        active_streams_.fetch_sub(1, std::memory_order_release);
        throw;
    }
}

void
S2SPipeline::prefill_system_prompt(S2SStream& st, const std::string& prompt) {
    if (prompt.empty())
        return;
    auto [stripped, acks_json] = strip_tool_acks(prompt);

    // Current schema: [{"name": "tool", "ack_messages": ["...", ...]}].
    if (!acks_json.empty()) {
        const Json entries = Json::parse(acks_json, nullptr, false);
        if (entries.is_array()) {
            for (const auto& entry : entries) {
                if (!entry.is_object() || !entry.contains("name") || !entry["name"].is_string() ||
                    !entry.contains("ack_messages") || !entry["ack_messages"].is_array())
                    continue;
                std::vector<std::string> messages;
                for (const auto& message : entry["ack_messages"])
                    if (message.is_string() && !message.get_ref<const std::string&>().empty())
                        messages.push_back(message.get<std::string>());
                if (!messages.empty())
                    st.tool_ack_messages.emplace_back(
                        entry["name"].get<std::string>(), std::move(messages));
            }
        }
    }

    // Prompt collation uses two identical copies.
    std::vector<int32_t> single_ids;
    single_ids.push_back(bos_id_);
    auto body = llm_->text_to_ids(stripped);
    single_ids.insert(single_ids.end(), body.begin(), body.end());
    single_ids.push_back(eos_id_);
    std::vector<int32_t> ids = single_ids;
    ids.insert(ids.end(), single_ids.begin(), single_ids.end());
    const int L = static_cast<int>(ids.size());
    const int H = llm_heads_->hidden_size();

    // E = embed(ids); E[1:] += pad_emb (text channel); E[0] += pad-as-"BOS";
    // E[:] += pad_emb * function_channel_weight.
    std::vector<float> E(static_cast<size_t>(L) * H);
    llm_heads_->embed_tokens(ids.data(), L, E.data());
    bf16_round_vec(E.data(), E.size());
    for (int t = 0; t < L; t++) {
        float* row = E.data() + static_cast<size_t>(t) * H;
        const float* text_part = (t == 0) ? bos_like_emb_.data() : pad_emb_.data();
        for (int i = 0; i < H; i++) {
            row[i] = bf16_round(row[i] + text_part[i]);
            row[i] = bf16_round(row[i] + bf16_round(pad_emb_[i] * cfg_.function_channel_weight));
        }
    }

    llm_->prefill(st.seq_id, E.data(), L);
}

void
S2SPipeline::end_stream(S2SStream& st) {
    if (!st.started)
        return;
    st.started = false;
    active_streams_.fetch_sub(1, std::memory_order_relaxed);
    st.extract_cancelled = true;
    st.inject_cancelled = true;
    if (st.fast_path_thread.joinable())
        st.fast_path_thread.join();
    llm_->cleanup(st.seq_id);
    if (st.tts_started)
        tts_->free_pair(st.tts_cond_slot, st.tts_uncond_slot);
    codec_->drop_stream(st.seq_id);
}

S2SChunkResult
S2SPipeline::finish_stream(S2SStream& st) {
    S2SChunkResult out;
    if (!st.started)
        return out;

    st.extract_cancelled = true;
    st.inject_cancelled = true;
    if (st.fast_path_thread.joinable())
        st.fast_path_thread.join();

    try {
        append_rnnt_display_output(
            st, /*agent_bos_fired=*/false, /*force_publish=*/true, /*close_turn=*/true);
        out.asr_text = std::move(st.rnnt_display_pending_text);
    }
    catch (...) {
        end_stream(st);
        throw;
    }
    end_stream(st);
    return out;
}

// ---------------------------------------------------------------------------
// Text sampling (infer/utils.py sample_text_token).
// ---------------------------------------------------------------------------
int
S2SPipeline::sample_text_token(const float* logits, S2SStream& st) const {
    const int V = llm_heads_->vocab_size();
    int greedy = 0;
    float best = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > best) {
            best = logits[i];
            greedy = i;
        }

    const bool fast = cfg_.top_p >= 1.0f && cfg_.repetition_penalty == 1.0f &&
                      (cfg_.temperature == 0.0f || cfg_.temperature == 1.0f);
    if (fast)
        return greedy;
    if (greedy == pad_id_ || greedy == bos_id_ || greedy == eos_id_)
        return greedy;  // specials bypass sampling

    std::vector<float> l(logits, logits + V);
    if (cfg_.repetition_penalty != 1.0f && st.decoder_global_step > 0) {
        std::vector<bool> seen(V, false);
        for (int t = 0; t < st.decoder_global_step; t++) {
            const int id = static_cast<int>(st.text_tokens[t]);
            if (id == pad_id_ || id == bos_id_ || id == eos_id_)
                continue;
            if (id >= 0 && id < V)
                seen[id] = true;
        }
        for (int i = 0; i < V; i++) {
            if (!seen[i])
                continue;
            l[i] = l[i] > 0 ? l[i] / cfg_.repetition_penalty : l[i] * cfg_.repetition_penalty;
        }
    }
    // Temperature zero remains deterministic while applying repetition
    // penalty before the greedy argmax.
    if (cfg_.temperature == 0.0f) {
        int adjusted = 0;
        for (int i = 1; i < V; ++i)
            if (l[i] > l[adjusted])
                adjusted = i;
        return adjusted;
    }
    if (cfg_.temperature != 1.0f)
        for (auto& v : l) v /= cfg_.temperature;
    // top-p
    std::vector<int> order(V);
    for (int i = 0; i < V; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return l[a] > l[b]; });
    float mx = l[order[0]];
    double denom = 0;
    std::vector<double> p(V);
    for (int i = 0; i < V; i++) {
        p[i] = std::exp(static_cast<double>(l[order[i]]) - mx);
        denom += p[i];
    }
    double cum = 0;
    int keep = V;
    for (int i = 0; i < V; i++) {
        cum += p[i] / denom;
        if (cum > cfg_.top_p) {
            keep = i + 1;
            break;
        }  // shift-by-one: keep top-1+
    }
    double sub = 0;
    for (int i = 0; i < keep; i++) sub += p[i];
    double r = std::uniform_real_distribution<double>(0.0, sub)(st.rng);
    for (int i = 0; i < keep; i++) {
        r -= p[i];
        if (r <= 0)
            return order[i];
    }
    return order[0];
}

// ---------------------------------------------------------------------------
// Per-step engines
// ---------------------------------------------------------------------------
void
S2SPipeline::llm_step(S2SStream& st, const float* encoded_audio, int n_frames, int step_offset) {
    const int t = st.decoder_global_step;
    if (t >= cfg_.max_steps) {
        st.steps_exhausted = true;
        return;
    }
    const int H = llm_heads_->hidden_size();
    const int max_idx = cfg_.max_chunks_for_inference - 1;
    int audio_idx = st.audio_chunk_idx;
    if (audio_idx >= max_idx)
        audio_idx = max_idx - (cfg_.steps_per_call - 1) + step_offset;
    audio_idx = std::max(0, std::min(audio_idx, std::min(max_idx, n_frames - 1)));
    const float* audio_row = encoded_audio + static_cast<size_t>(audio_idx) * H;
    llm_step_batcher_->run(st, audio_row, t, audio_idx);
}

void
S2SPipeline::tts_step(S2SStream& st) {
    if (st.steps_exhausted)
        return;
    const int t = st.decoder_global_step;
    if (t >= cfg_.max_steps) {
        st.steps_exhausted = true;
        return;
    }
    tts_step_batcher_->run(st, t);
}

namespace {

bool
contains_token(const std::vector<int32_t>& tokens, int token) {
    return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

class RnntDecodeScope {
   public:
    explicit RnntDecodeScope(asr::RnntEngine& engine) : engine_(engine) {
        engine_.begin_decode_step();
    }
    ~RnntDecodeScope() { engine_.end_decode_step(); }

   private:
    asr::RnntEngine& engine_;
};

}  // namespace

void
S2SPipeline::reset_rnnt_display(S2SStream& st, bool reset_predictor) {
    auto& state = st.rnnt_display;
    state.blank_count = 0;
    state.nonblank_total = 0;
    state.speech_confirmed = false;
    state.y_sequence.clear();
    state.punct_word_acc.clear();
    state.punct_bias = 0.0f;
    if (reset_predictor) {
        state.engine_state.reset();
        state.engine_state = perception_->rnnt_engine().make_rnnt_stream_state();
        state.prev_token = perception_->rnnt_config().blank_id;
        state.active_bank = 0;
        state.predictor_valid = false;
    }
    st.rnnt_display_emitted_len = 0;
    st.rnnt_display_turn_open = false;
}

int
S2SPipeline::rnnt_joint_token(
    S2SStream::RnntDecoderState& state, const float* frame, const float* logit_bias) {
    if (!state.engine_state)
        throw std::runtime_error("S2S RNNT decoder state is not initialized");
    auto& engine = perception_->rnnt_engine();
    const int joint_dim = perception_->rnnt_config().joint_dim;
    if (!state.predictor_valid) {
        engine.predict_rnnt(*state.engine_state, state.prev_token, state.active_bank);
        state.predictor_valid = true;
    }
    int32_t token = perception_->rnnt_config().blank_id;
    engine.joint_argmax(*state.engine_state, frame, joint_dim, /*T=*/1, &token, logit_bias);
    return token;
}

void
S2SPipeline::rnnt_commit_token(S2SStream::RnntDecoderState& state, int token) {
    state.prev_token = token;
    state.active_bank ^= 1;
    state.predictor_valid = false;
}

void
S2SPipeline::append_rnnt_display_output(
    S2SStream& st, bool agent_bos_fired, bool force_publish, bool close_turn) {
    std::string full_text;
    for (const int32_t token : st.rnnt_display.y_sequence)
        if (token >= 0 && token < static_cast<int32_t>(rnnt_vocab_.size()))
            full_text += rnnt_vocab_[token];
    replace_all(full_text, "▁", " ");
    full_text = trim_copy(full_text);
    const bool has_word = std::any_of(
        full_text.begin(), full_text.end(), [](unsigned char c) { return std::isalpha(c) != 0; });

    if ((st.rnnt_display.speech_confirmed || force_publish || agent_bos_fired) &&
        !full_text.empty() && has_word) {
        if (full_text.size() < st.rnnt_display_emitted_len)
            st.rnnt_display_emitted_len = 0;
        const std::string delta = full_text.substr(st.rnnt_display_emitted_len);
        if (!delta.empty()) {
            if (!st.rnnt_display_turn_open) {
                st.rnnt_display_pending_text += "<s>";
                st.rnnt_display_turn_open = true;
            }
            st.rnnt_display_pending_text += delta;
            st.rnnt_display_emitted_len = full_text.size();
        }
    }

    if (close_turn && st.rnnt_display_turn_open)
        st.rnnt_display_pending_text += "</s>";

    if (agent_bos_fired) {
        if (st.rnnt_display_turn_open)
            st.rnnt_display_pending_text += "</s>";
        reset_rnnt_display(st);
        return;
    }

    if (st.rnnt_display.blank_count >= cfg_.rnnt_display_fallback_clear_frames &&
        st.rnnt_display.nonblank_total == 0)
        reset_rnnt_display(st, true);
}

void
S2SPipeline::rnnt_display_step(S2SStream& st, const float* asr_emb, int n_frames, int step_offset) {
    if (st.steps_exhausted || asr_emb == nullptr || rnnt_vocab_.empty())
        return;
    const auto& rc = perception_->rnnt_config();
    const int max_idx = cfg_.max_chunks_for_inference - 1;
    int audio_idx = st.audio_chunk_idx;
    if (audio_idx >= max_idx)
        audio_idx = max_idx - (cfg_.steps_per_call - 1) + step_offset;
    audio_idx = std::max(0, std::min(audio_idx, std::min(max_idx, n_frames - 1)));
    const float* frame = asr_emb + static_cast<size_t>(audio_idx) * rc.joint_dim;
    auto& state = st.rnnt_display;
    RnntDecodeScope decode_scope(perception_->rnnt_engine());

    const int first_token = rnnt_joint_token(state, frame);
    const bool is_blank = first_token == rc.blank_id;
    std::vector<int32_t> emitted;

    if (is_blank && state.punct_bias > 0.0f && !rnnt_punct_ids_.empty()) {
        std::vector<float> bias(static_cast<size_t>(rc.vocab_size), 0.0f);
        for (size_t i = 0; i < rnnt_punct_ids_.size(); ++i) {
            const int id = rnnt_punct_ids_[i];
            bias[static_cast<size_t>(id)] = state.punct_bias * rnnt_punct_increments_[i];
        }
        const int punct_token = rnnt_joint_token(state, frame, bias.data());
        if (contains_token(rnnt_punct_ids_, punct_token)) {
            emitted.push_back(punct_token);
            rnnt_commit_token(state, punct_token);
        }
    }

    int loop_token = first_token;
    for (int symbol = 0; loop_token != rc.blank_id && symbol < cfg_.rnnt_display_max_symbols;
         ++symbol) {
        emitted.push_back(loop_token);
        rnnt_commit_token(state, loop_token);
        loop_token = rnnt_joint_token(state, frame);
    }

    if (is_blank) {
        ++state.blank_count;
    } else {
        state.blank_count = 0;
        ++state.nonblank_total;
        if (state.nonblank_total >= cfg_.rnnt_display_gate_frames)
            state.speech_confirmed = true;
    }

    bool emitted_punct = false, emitted_nonpunct = false;
    for (const int32_t token : emitted) {
        if (contains_token(rnnt_punct_ids_, token))
            emitted_punct = true;
        else {
            emitted_nonpunct = true;
            state.punct_word_acc.push_back(token);
        }
    }
    if (emitted_punct) {
        state.punct_word_acc.clear();
        state.punct_bias = 0.0f;
    } else if (emitted_nonpunct) {
        state.punct_bias = 0.0f;
    } else if (
        !state.punct_word_acc.empty() && is_blank &&
        state.blank_count >= cfg_.rnnt_punct_bias_min_silence_frames) {
        state.punct_bias += 1.0f;
    }
    state.y_sequence.insert(state.y_sequence.end(), emitted.begin(), emitted.end());

    const bool agent_bos_fired = st.text_tokens[st.decoder_global_step] == bos_id_;
    append_rnnt_display_output(st, agent_bos_fired);
}

void
S2SPipeline::rnnt_eou_step(S2SStream& st, const float* asr_emb, int n_frames, int step_offset) {
    if (st.steps_exhausted || asr_emb == nullptr)
        return;
    const auto& rc = perception_->rnnt_config();
    const int max_idx = cfg_.max_chunks_for_inference - 1;
    int audio_idx = st.audio_chunk_idx;
    if (audio_idx >= max_idx)
        audio_idx = max_idx - (cfg_.steps_per_call - 1) + step_offset;
    audio_idx = std::max(0, std::min(audio_idx, std::min(max_idx, n_frames - 1)));
    const float* frame = asr_emb + static_cast<size_t>(audio_idx) * rc.joint_dim;
    auto& state = st.rnnt_eou;
    RnntDecodeScope decode_scope(perception_->rnnt_engine());

    const int first_token = rnnt_joint_token(state, frame);
    const bool is_blank = first_token == rc.blank_id;

    int loop_token = first_token;
    for (int symbol = 0; loop_token != rc.blank_id && symbol < cfg_.rnnt_max_symbols; ++symbol) {
        rnnt_commit_token(state, loop_token);
        loop_token = rnnt_joint_token(state, frame);
    }

    if (!st.rnnt_agent_speaking)
        st.rnnt_rolling_density = cfg_.rnnt_density_alpha * (is_blank ? 0.0f : 1.0f) +
                                  (1.0f - cfg_.rnnt_density_alpha) * st.rnnt_rolling_density;
    if (!is_blank) {
        st.rnnt_silent_frames = 0;
        ++st.rnnt_consecutive_speech_frames;
        ++st.rnnt_nonblank_total;
        if (st.rnnt_consecutive_speech_frames >= cfg_.rnnt_fc_interrupt_frames) {
            if (st.extracting_tool && !st.extract_cancelled)
                st.extract_cancelled = true;
            if (st.injecting_response && !st.inject_cancelled)
                st.inject_cancelled = true;
        }
    } else {
        st.rnnt_consecutive_speech_frames = 0;
        ++st.rnnt_silent_frames;
    }
}

void
S2SPipeline::apply_rnnt_turn_taking(S2SStream& st) {
    if (st.steps_exhausted)
        return;
    const int t = st.decoder_global_step;
    int32_t current_token = static_cast<int32_t>(st.text_tokens[t]);
    const auto reset_tts_for_turn = [&] {
        // Each utterance starts from the immutable voice prompt. Carrying
        // PAD-only or completed-turn acoustic history into a new BOS can
        // leave TTS on a silence trajectory even while text generation is
        // healthy. The language-model and conversation states are unchanged.
        tts_->reset_pair(st.tts_cond_slot, st.tts_uncond_slot);
        st.tts_last_codes = tts_initial_codes_;
        st.tts_agent_idle = true;
        st.tts_in_turn_content = 0;
        st.tts_in_turn_pads = 0;
    };
    const int lookback = std::max(0, t - cfg_.force_turn_taking_threshold);
    bool recent_bos = false;
    for (int i = lookback; i < t; ++i) recent_bos = recent_bos || st.text_tokens[i] == bos_id_;

    int effective_min = cfg_.rnnt_bou_min_frames;
    if (st.rnnt_first_turn)
        effective_min = cfg_.rnnt_bou_min_frames_first_turn;
    else if (
        st.rnnt_rolling_density > 0.0f && st.rnnt_rolling_density < cfg_.rnnt_density_threshold)
        effective_min = cfg_.rnnt_density_low_min;

    if (current_token == bos_id_ && st.rnnt_agent_speaking) {
        st.text_tokens[t] = eos_id_;
        st.tts_force_silence[t] = 1;
        current_token = eos_id_;
    }

    if (current_token == bos_id_)
        reset_tts_for_turn();

    if (current_token == bos_id_) {
        st.rnnt_agent_speaking = true;
        st.rnnt_first_turn = false;
        st.rnnt_user_speaking = false;
        st.rnnt_nonblank_total = 0;
        st.rnnt_turn_text_tokens = 0;
    }
    if (st.rnnt_agent_speaking && current_token == eos_id_) {
        st.rnnt_agent_speaking = false;
        st.rnnt_user_speaking = false;
        st.rnnt_nonblank_total = 0;
        st.rnnt_agent_talking_frames = 0;
    }

    if (st.rnnt_silent_frames >= cfg_.rnnt_noise_reset_frames && !st.rnnt_user_speaking &&
        !st.rnnt_agent_speaking)
        st.rnnt_nonblank_total = 0;

    if ((st.rnnt_consecutive_speech_frames >= effective_min ||
         st.rnnt_nonblank_total >= effective_min) &&
        !st.rnnt_agent_speaking)
        st.rnnt_user_speaking = true;

    if (st.rnnt_silent_frames >= cfg_.rnnt_eos_silence_frames && st.rnnt_user_speaking &&
        !st.rnnt_agent_speaking) {
        if (!recent_bos && current_token != bos_id_) {
            st.text_tokens[t] = bos_id_;
            reset_tts_for_turn();
            st.rnnt_user_speaking = false;
            st.rnnt_nonblank_total = 0;
            st.rnnt_agent_speaking = true;
            st.rnnt_first_turn = false;
            st.rnnt_turn_text_tokens = 0;
        }
        return;
    }

    if (st.rnnt_agent_speaking && current_token != eos_id_) {
        ++st.rnnt_agent_talking_frames;
        if (current_token != bos_id_ && current_token != eos_id_ && current_token != pad_id_)
            ++st.rnnt_turn_text_tokens;
    } else if (!st.rnnt_agent_speaking) {
        st.rnnt_agent_talking_frames = 0;
    }

    if (cfg_.rnnt_max_agent_response_frames > 0 && st.rnnt_agent_speaking &&
        current_token != eos_id_ &&
        st.rnnt_agent_talking_frames >= cfg_.rnnt_max_agent_response_frames) {
        st.text_tokens[t] = eos_id_;
        st.tts_force_silence[t] = 1;
        st.rnnt_agent_speaking = false;
        st.rnnt_agent_talking_frames = 0;
        return;
    }

    if (cfg_.rnnt_tts_ratio_cap > 0.0f && st.rnnt_agent_speaking && current_token != eos_id_ &&
        st.rnnt_turn_text_tokens >= cfg_.rnnt_tts_min_tokens &&
        st.rnnt_agent_talking_frames >= cfg_.rnnt_tts_ratio_cap * st.rnnt_turn_text_tokens) {
        st.text_tokens[t] = eos_id_;
        st.tts_force_silence[t] = 1;
        st.rnnt_agent_speaking = false;
        st.rnnt_agent_talking_frames = 0;
        st.rnnt_turn_text_tokens = 0;
        return;
    }

    if (st.rnnt_consecutive_speech_frames >= cfg_.rnnt_barge_in_frames && st.rnnt_agent_speaking &&
        current_token != eos_id_) {
        st.text_tokens[t] = eos_id_;
        st.tts_force_silence[t] = 1;
        st.rnnt_agent_speaking = false;
        st.rnnt_consecutive_speech_frames = 0;
        st.rnnt_nonblank_total = 0;
    }
}

// ---------------------------------------------------------------------------
// Function-call machinery
// ---------------------------------------------------------------------------
std::string
S2SPipeline::sanitize_function_text(const std::string& raw) const {
    const std::string open = "<TOOLCALL>", close = "</TOOLCALL>";
    const std::string text = trim_copy(raw);
    const size_t a = text.find(open);
    if (a == std::string::npos)
        return "";
    const size_t b = text.find(close, a);
    if (b == std::string::npos)
        return "";
    std::string inner = text.substr(a + open.size(), b - a - open.size());
    if (Json::parse(inner, nullptr, false).is_discarded())
        return "";
    return inner;
}

std::string
S2SPipeline::format_tool_response(const std::string& raw) const {
    static const std::string open = "<TOOL_RESPONSE>";
    static const std::string close = "</TOOL_RESPONSE>";
    const std::string text = trim_copy(raw);
    if (text.rfind(open, 0) == 0 && text.size() >= open.size() + close.size() &&
        text.compare(text.size() - close.size(), close.size(), close) == 0) {
        const std::string inner =
            text.substr(open.size(), text.size() - open.size() - close.size());
        Json payload = Json::parse(inner, nullptr, false);
        if (payload.is_discarded())
            return text;
        normalize_tool_json(payload);
        return open + python_json_dump(payload) + close;
    }

    Json payload = Json::parse(text, nullptr, false);
    if (payload.is_discarded())
        payload = text;
    Json wrapped = Json::array();
    wrapped.push_back(std::move(payload));
    normalize_tool_json(wrapped);
    return open + python_json_dump(wrapped) + close;
}

std::string
S2SPipeline::select_tool_ack(S2SStream& st, const std::string& sanitized) const {
    const Json calls = Json::parse(sanitized, nullptr, false);
    if (!calls.is_array() || calls.empty() || !calls[0].is_object() || !calls[0].contains("name") ||
        !calls[0]["name"].is_string())
        return "";
    const std::string name = calls[0]["name"].get<std::string>();
    for (const auto& item : st.tool_ack_messages) {
        if (item.first != name || item.second.empty())
            continue;
        return item
            .second[std::uniform_int_distribution<size_t>(0, item.second.size() - 1)(st.rng)];
    }
    return "";
}

void
S2SPipeline::reset_fn_call_state(S2SStream& st) {
    st.fn_state = S2SStream::FnState::Idle;
    st.fn_frames_in_state = 0;
    st.fn_request.clear();
    st.fn_response_tokens.clear();
    st.fn_ack_tokens.clear();
    st.pending_function_text.clear();
    st.tts_agent_idle = true;
    st.tts_in_turn_content = 0;
    st.tts_in_turn_pads = 0;
}

void
S2SPipeline::parse_function_tokens(
    S2SStream& st, const std::vector<int32_t>& toks, std::string& out_function_text) {
    // Timeouts advance once per response frame.
    if (st.fn_state == S2SStream::FnState::WaitingForRequest ||
        st.fn_state == S2SStream::FnState::WaitingForResponse) {
        st.fn_frames_in_state += static_cast<int>(toks.size());
        if (st.fn_frames_in_state >= cfg_.fn_call_timeout_frames) {
            reset_fn_call_state(st);
            return;
        }
    }

    std::string sanitized;
    for (int32_t tok : toks) {
        if (tok == pad_id_)
            continue;
        if (tok == sotc_id_) {
            if (st.fn_state != S2SStream::FnState::Idle)
                continue;
            reset_fn_call_state(st);
            st.fn_state = S2SStream::FnState::WaitingForRequest;
        } else if (tok == eotc_id_ && st.fn_state == S2SStream::FnState::WaitingForRequest) {
            sanitized = sanitize_function_text(st.fn_request);
            if (sanitized.empty()) {
                reset_fn_call_state(st);
            } else {
                st.fn_request = sanitized;
                st.fn_state = S2SStream::FnState::RequestReceived;
            }
            st.fn_frames_in_state = 0;
        } else if (st.fn_state == S2SStream::FnState::WaitingForRequest) {
            st.fn_request += llm_->ids_to_text(&tok, 1);
        }
    }

    if (!sanitized.empty()) {
        const std::string ack = select_tool_ack(st, sanitized);
        if (!ack.empty()) {
            auto ids = llm_->text_to_ids(ack);
            st.fn_ack_tokens.push_back(bos_id_);
            st.fn_ack_tokens.insert(st.fn_ack_tokens.end(), ids.begin(), ids.end());
            const int pad_count = std::max(17, static_cast<int>(std::ceil(0.5 * ack.size())));
            for (int i = 0; i < pad_count; ++i) st.fn_ack_tokens.push_back(pad_id_);
            st.fn_ack_tokens.push_back(eos_id_);
            st.fn_state = S2SStream::FnState::SpeakingAck;
        } else {
            st.fn_state = S2SStream::FnState::WaitingForResponse;
        }
        st.fn_frames_in_state = 0;
        out_function_text = sanitized;
    }
}

void
S2SPipeline::sync_audio_buffer_to_step(S2SStream& st) {
    const size_t want =
        static_cast<size_t>(st.decoder_global_step + cfg_.steps_per_call) * cfg_.samples_per_chunk;
    if (st.audio_buffer.size() < want)
        st.audio_buffer.resize(want, 0.0f);
}

void
S2SPipeline::fast_extract_tool_tokens(S2SStream& st) {
    const int H = llm_heads_->hidden_size();
    std::vector<float> dummy(static_cast<size_t>(cfg_.max_chunks_for_inference) * H, 0.0f);
    for (int iter = 0; iter < cfg_.max_tool_tokens; iter++) {
        // Barge-in only stops this fast-path attempt. Keep the accumulated
        // request and WaitingForRequest state so the next chunk can retry.
        // The normal request timeout remains the escape path.
        if (st.extract_cancelled)
            break;
        if (st.decoder_global_step >= cfg_.max_steps) {
            st.steps_exhausted = true;
            reset_fn_call_state(st);
            break;
        }
        llm_step(st, dummy.data(), cfg_.max_chunks_for_inference, 0);
        const int t = st.decoder_global_step;
        const int32_t fn_token = static_cast<int32_t>(st.function_tokens[t]);
        st.audio_chunk_idx = std::min(st.audio_chunk_idx + 1, cfg_.max_chunks_for_inference - 1);
        st.decoder_global_step++;

        if (fn_token == eotc_id_) {
            const std::string sanitized = sanitize_function_text(st.fn_request);
            if (!sanitized.empty()) {
                st.fn_request = sanitized;
                st.pending_function_text = sanitized;
                const std::string ack = select_tool_ack(st, sanitized);
                if (!ack.empty()) {
                    auto ids = llm_->text_to_ids(ack);
                    st.fn_ack_tokens.push_back(bos_id_);
                    st.fn_ack_tokens.insert(st.fn_ack_tokens.end(), ids.begin(), ids.end());
                    const int pad_count =
                        std::max(17, static_cast<int>(std::ceil(0.5 * ack.size())));
                    for (int i = 0; i < pad_count; ++i) st.fn_ack_tokens.push_back(pad_id_);
                    st.fn_ack_tokens.push_back(eos_id_);
                    st.fn_state = S2SStream::FnState::SpeakingAck;
                } else {
                    st.fn_state = S2SStream::FnState::WaitingForResponse;
                }
            } else {
                reset_fn_call_state(st);
            }
            break;
        }
        if (fn_token != pad_id_)
            st.fn_request += llm_->ids_to_text(&fn_token, 1);
        if (iter == cfg_.max_tool_tokens - 1) {
            reset_fn_call_state(st);
        }
    }
    sync_audio_buffer_to_step(st);
    st.extracting_tool = false;
}

void
S2SPipeline::fast_inject_response_tokens(S2SStream& st) {
    const int H = llm_heads_->hidden_size();
    std::vector<float> dummy(static_cast<size_t>(cfg_.max_chunks_for_inference) * H, 0.0f);
    while (st.fn_state == S2SStream::FnState::ProcessResponse) {
        if (st.inject_cancelled || st.decoder_global_step >= cfg_.max_steps) {
            if (st.decoder_global_step >= cfg_.max_steps)
                st.steps_exhausted = true;
            if (st.inject_cancelled)
                reset_fn_call_state(st);
            break;
        }
        llm_step(st, dummy.data(), cfg_.max_chunks_for_inference, 0);
        st.audio_chunk_idx = std::min(st.audio_chunk_idx + 1, cfg_.max_chunks_for_inference - 1);
        st.decoder_global_step++;
    }
    sync_audio_buffer_to_step(st);
    st.injecting_response = false;
}

void
S2SPipeline::maybe_start_fast_paths(S2SStream& st) {
    if (st.fn_state == S2SStream::FnState::WaitingForRequest && !st.extracting_tool) {
        st.extracting_tool = true;
        st.extract_cancelled = false;
        if (st.fast_path_thread.joinable())
            st.fast_path_thread.join();
        st.fast_path_thread = std::thread([this, &st] { fast_extract_tool_tokens(st); });
    } else if (
        st.fn_state == S2SStream::FnState::ProcessResponse && !st.fn_response_tokens.empty() &&
        !st.injecting_response) {
        st.injecting_response = true;
        st.inject_cancelled = false;
        if (st.fast_path_thread.joinable())
            st.fast_path_thread.join();
        st.fast_path_thread = std::thread([this, &st] { fast_inject_response_tokens(st); });
    }
}

// ---------------------------------------------------------------------------
// Output decode
// ---------------------------------------------------------------------------
void
S2SPipeline::decode_outputs(S2SStream& st, S2SChunkResult& out, int completed_steps) {
    const int end_idx = std::min(st.decoder_global_step, cfg_.max_steps);
    const int start_idx = std::max(0, end_idx - completed_steps);
    const int Q = codec_->config().n_quantizers;

    // Text: drop pads, render specials literally.
    for (int i = start_idx; i < end_idx; i++) {
        const int32_t tok = static_cast<int32_t>(st.text_tokens[i]);
        if (tok == pad_id_)
            continue;
        out.text += llm_->ids_to_text(&tok, 1);
    }

    // Function tokens of this chunk -> state machine + sanitized payload.
    std::vector<int32_t> fn_toks;
    for (int i = start_idx; i < end_idx; i++)
        fn_toks.push_back(static_cast<int32_t>(st.function_tokens[i]));
    parse_function_tokens(st, fn_toks, out.function_text);
    if (out.function_text.empty() && !st.pending_function_text.empty()) {
        out.function_text = st.pending_function_text;
        st.pending_function_text.clear();
    }

    if (!st.rnnt_display_pending_text.empty()) {
        out.asr_text = std::move(st.rnnt_display_pending_text);
        st.rnnt_display_pending_text.clear();
    }

    // Codec decode of exactly this chunk's frames.
    if (completed_steps > 0)
        codec_->decode_wav(
            st.seq_id, st.audio_tokens.data() + static_cast<size_t>(start_idx) * Q, completed_steps,
            out.audio);
    if (start_idx == 0 && !out.audio.empty()) {
        // First-chunk trim: drop the frame produced by the TTS prompt prefill.
        const int trim = codec_->samples_per_frame();
        out.audio.erase(
            out.audio.begin(), out.audio.begin() + std::min<size_t>(trim, out.audio.size()));
    }
}

// ---------------------------------------------------------------------------
// Per-chunk entry
// ---------------------------------------------------------------------------
void
S2SPipeline::warmup() {
    auto st = create_stream(/*seq_id=*/0);
    try {
        prefill_system_prompt(*st, "You are a helpful voice assistant.");
        const int chunk = cfg_.steps_per_call * cfg_.samples_per_chunk;
        std::vector<float> zeros(chunk, 0.0f);
        const int n_chunks = cfg_.max_chunks_for_inference / cfg_.steps_per_call + 5;
        for (int i = 0; i < n_chunks; i++) process_chunk(*st, zeros.data(), chunk, "");
    }
    catch (...) {
        end_stream(*st);
        throw;
    }
    end_stream(*st);
}

S2SChunkResult
S2SPipeline::process_chunk(
    S2SStream& st, const float* audio, int n_samples, const std::string& function_response) {
    static const bool dbg_total = std::getenv("S2S_DEBUG_TIMING") != nullptr;
    const auto tw0 = std::chrono::steady_clock::now();
    struct TotalLog {
        bool on;
        std::chrono::steady_clock::time_point t0;
        ~TotalLog() {
            if (on)
                std::fprintf(
                    stderr, "[s2s-timing] chunk_total=%.1f\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count());
        }
    } total_log{dbg_total, tw0};
    S2SChunkResult out;
    if (st.steps_exhausted)
        throw std::runtime_error("S2SStream: step budget exhausted");

    // Fast-path gate: drop audio, return silence.
    if (st.extracting_tool || st.injecting_response) {
        out.audio.assign(
            static_cast<size_t>(cfg_.steps_per_call) * codec_->samples_per_frame(), 0.0f);
        return out;
    }
    if (st.fast_path_thread.joinable() && st.fn_state != S2SStream::FnState::Idle &&
        !st.extracting_tool && !st.injecting_response) {
        st.fast_path_thread.join();
    }

    st.audio_buffer.insert(st.audio_buffer.end(), audio, audio + n_samples);

    // Tool-response intake accepts a raw JSON object/string or an already
    // framed response, then normalizes it before tokenizing.
    if (!function_response.empty()) {
        const std::string formatted = format_tool_response(function_response);
        auto body = llm_->text_to_ids(formatted);
        st.fn_response_tokens.clear();
        for (auto v : body) st.fn_response_tokens.push_back(v);
        if (st.fn_state == S2SStream::FnState::WaitingForResponse) {
            st.fn_state = S2SStream::FnState::ProcessResponse;
            st.fn_frames_in_state = 0;
        }
    }

    // Realtime gate.
    const size_t min_samples =
        static_cast<size_t>(st.decoder_global_step + cfg_.steps_per_call) * cfg_.samples_per_chunk;
    if (st.audio_buffer.size() < min_samples) {
        out.skipped = true;
        return out;
    }

    // Establish one ingress wave for this request thread. Every native
    // microbatcher below sees the same target and releases as soon as all
    // members of this actual wave arrive, instead of paying a timer per stage.
    // A max_streams > 1 build can still be serving only one live
    // conversation.  In that case bypass the ingress timer entirely; when a
    // concurrent cohort is open, wait for exactly that live cohort rather
    // than the configured reservation ceiling.
    const int expected_streams = std::max(1, active_streams_.load(std::memory_order_relaxed));
    const int cohort_size = ingress_batch_coordinator_->arrive(expected_streams);
    asr::ScopedBatchCohort batch_cohort(cohort_size);

    // Window: last min(min_samples, max) samples, padded UP to one of four
    // fixed buckets (1.0 / 2.0 / 3.5 / 5.6 s) so the perception Sessions see
    // at most 4 input shapes — without this every chunk is a fresh shape and
    // each step pays a full graph rebuild + warmup (multi-second stalls).
    // Mirrors PerceptionGGMLAdapter's S2S_AUDIO_BUCKETS behavior.
    const size_t max_samples =
        static_cast<size_t>(cfg_.max_chunks_for_inference) * cfg_.samples_per_chunk;
    const size_t window = std::min(min_samples, max_samples);
    static const size_t kBuckets[] = {16000, 32000, 56000, 89600};
    size_t bucket = kBuckets[3];
    for (size_t b : kBuckets)
        if (window <= b) {
            bucket = b;
            break;
        }
    std::vector<float> padded(bucket, 0.0f);
    std::memcpy(
        padded.data(), st.audio_buffer.data() + (st.audio_buffer.size() - window),
        window * sizeof(float));

    // Perception over the bucketed window. Frames produced for the zero
    // padding are sliced off so audio_chunk_idx indexing matches the real
    // audio (mirrors the adapter's output slice).
    std::vector<float> encoded, asr_emb;
    int n_frames = 0;
    {
        static const bool dbg_p = std::getenv("S2S_DEBUG_TIMING") != nullptr;
        const auto tp0 = std::chrono::steady_clock::now();
        n_frames = perception_->step(
            padded.data(), static_cast<int>(bucket), static_cast<int>(window), encoded, &asr_emb);
        if (dbg_p) {
            std::fprintf(
                stderr, "[s2s-timing] perception=%.1f (bucket=%zu)\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0)
                    .count(),
                bucket);
        }
    }
    bf16_round_vec(encoded.data(), encoded.size());
    // Keep only the frames corresponding to the real window: one encoder
    // frame per 1280 samples.
    const int real_frames =
        std::min<int>(n_frames, static_cast<int>(window / cfg_.samples_per_chunk));
    n_frames = real_frames;
    if (n_frames <= 0) {
        out.skipped = true;
        return out;
    }

    // Ordering: LLM -> RNN-T EOU evidence -> turn-token correction -> RNN-T
    // transcript display -> TTS. In particular, TTS must
    // see a BOS/EOS injected by RNN-T in this same 80 ms frame.
    static const bool dbg = std::getenv("S2S_DEBUG_TIMING") != nullptr;
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point a) {
        return std::chrono::duration<double, std::milli>(clk::now() - a).count();
    };
    double t_llm = 0, t_tts = 0, t_rnnt = 0;
    int completed_steps = 0;
    for (int step = 0; step < cfg_.steps_per_call; step++) {
        auto t0 = clk::now();
        llm_step(st, encoded.data(), n_frames, step);
        t_llm += ms_since(t0);
        if (st.steps_exhausted)
            break;
        t0 = clk::now();
        rnnt_eou_step(st, asr_emb.data(), n_frames, step);
        apply_rnnt_turn_taking(st);
        rnnt_display_step(st, asr_emb.data(), n_frames, step);
        t_rnnt += ms_since(t0);
        t0 = clk::now();
        tts_step(st);
        t_tts += ms_since(t0);
        st.audio_chunk_idx = std::min(st.audio_chunk_idx + 1, cfg_.max_chunks_for_inference - 1);
        st.decoder_global_step++;
        completed_steps++;
        if (st.decoder_global_step >= cfg_.max_steps) {
            st.steps_exhausted = true;
            break;
        }
    }

    auto t_dec0 = clk::now();
    decode_outputs(st, out, completed_steps);
    if (dbg) {
        std::fprintf(
            stderr, "[s2s-timing] step=%d llm=%.1f tts=%.1f rnnt=%.1f codec+dec=%.1f\n",
            st.decoder_global_step, t_llm, t_tts, t_rnnt, ms_since(t_dec0));
    }
    if (!st.steps_exhausted)
        maybe_start_fast_paths(st);
    return out;
}

}  // namespace nemo_speech::s2s
