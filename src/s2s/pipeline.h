// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// S2S VoiceChat pipeline orchestrator.
//
// One S2SPipeline owns shared engines and one S2SStream owns each
// conversation's state. process_chunk() advances one 160 ms input step.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "codec/codec.h"
#include "eartts/backbone.h"
#include "eartts/embedder.h"
#include "eartts/prompt.h"
#include "eartts/sampler.h"
#include "llm/backbone.h"
#include "llm/heads.h"
#include "perception/perception.h"
#include "runtime.h"

namespace nemo_speech::s2s {

struct S2SPipelineConfig {
    std::string perception_gguf;
    std::string llm_gguf;
    std::string llm_aux_gguf;
    std::string tts_backbone_gguf;
    std::string tts_side_gguf;
    std::string tts_prompt_gguf;
    std::string codec_gguf;
    std::string rnnt_vocab_json;  // rnnt_tokenizer/vocab.json (JSON string array)

    // Resource reservation ceiling, not an expected throughput value.
    int max_streams = 32;
    int steps_per_call = 2;             // MODEL_STEPS_PER_CALL
    int samples_per_chunk = 1280;       // 80 ms @ 16 kHz
    int max_chunks_for_inference = 70;  // 5.6 s perception window
    int max_steps = 5000;               // upper bound; clamped to EarTTS position capacity
    float user_channel_weight = 1.0f;
    float text_channel_weight = 1.0f;
    float function_channel_weight = 2.0f;
    float guidance_scale = 0.2f;
    float top_p = 1.0f;
    float temperature = 0.0f;  // 0 => greedy
    float repetition_penalty = 1.0f;
    int max_tool_tokens = 512;
    int fn_call_timeout_frames = 125;  // 10 s / 80 ms

    // RNN-T turn-taking and display defaults. Each frame is 80 ms.
    int force_turn_taking_threshold = 40;
    int rnnt_eos_silence_frames = 40;
    int rnnt_fc_interrupt_frames = 40;
    int rnnt_display_gate_frames = 3;
    int rnnt_display_fallback_clear_frames = 10;
    int rnnt_display_max_symbols = 10;
    int rnnt_max_symbols = 10;
    bool rnnt_punct_bias_enabled = true;
    float rnnt_punct_bias_increment = 1.5f;
    int rnnt_punct_bias_min_silence_frames = 5;
    int rnnt_bou_min_frames = 3;
    int rnnt_bou_min_frames_first_turn = 2;
    int rnnt_barge_in_frames = 40;
    float rnnt_density_alpha = 0.1f;
    float rnnt_density_threshold = 0.15f;
    int rnnt_density_low_min = 6;
    int rnnt_noise_reset_frames = 10;
    float rnnt_tts_ratio_cap = 16.0f;
    int rnnt_tts_min_tokens = 5;
    int rnnt_max_agent_response_frames = 0;  // 0 disables the absolute cap

    // TTS generation guards.
    bool tts_force_silence_on_pad = true;
    float tts_pad_tail_ratio = 3.0f;
};

// One processed chunk's outputs (mirrors the Triton output tensors).
struct S2SChunkResult {
    bool skipped = false;       // start placeholder or not-enough-audio
    std::string text;           // agent text delta (specials rendered literal)
    std::string asr_text;       // RNN-T transcript delta
    std::string function_text;  // sanitized tool-call JSON, emitted once
    std::vector<float> audio;   // 22050 Hz; empty when skipped
};

class S2SPipeline;

// Per-conversation state. Created by S2SPipeline::create_stream().
class S2SStream {
   public:
    ~S2SStream();

   private:
    friend class S2SPipeline;
    explicit S2SStream(int64_t id) : seq_id(id) {}

    int64_t seq_id;
    std::vector<float> audio_buffer;         // grows; only last window used
    std::vector<int64_t> text_tokens;        // [max_steps]
    std::vector<int64_t> function_tokens;    // [max_steps]
    std::vector<uint8_t> tts_force_silence;  // [max_steps]
    std::vector<int32_t> audio_tokens;       // [max_steps * Q]
    int decoder_global_step = 0;
    int audio_chunk_idx = 0;

    // TTS stream state.
    bool tts_started = false;
    int tts_cond_slot = -1, tts_uncond_slot = -1;
    std::vector<int32_t> tts_last_codes;  // (Q,) last sampled frame
    bool tts_agent_idle = true;
    int tts_in_turn_content = 0;
    int tts_in_turn_pads = 0;

    struct RnntDecoderState {
        std::unique_ptr<asr::RnntStreamState> engine_state;
        int prev_token = -1;
        int active_bank = 0;
        bool predictor_valid = false;
        int blank_count = 0;
        int nonblank_total = 0;
        bool speech_confirmed = false;
        std::vector<int32_t> y_sequence;
        std::vector<int32_t> punct_word_acc;
        float punct_bias = 0.0f;
    };

    // Separate predictors: one owns turn evidence and the other owns
    // user-visible transcript text.
    RnntDecoderState rnnt_eou;
    RnntDecoderState rnnt_display;
    size_t rnnt_display_emitted_len = 0;
    bool rnnt_display_turn_open = false;
    std::string rnnt_display_pending_text;

    // RNN-T turn-taking counters/state.
    bool rnnt_user_speaking = false;
    int rnnt_silent_frames = 0;
    int rnnt_consecutive_speech_frames = 0;
    int rnnt_nonblank_total = 0;
    float rnnt_rolling_density = 0.0f;
    bool rnnt_agent_speaking = false;
    bool rnnt_first_turn = true;
    int rnnt_agent_talking_frames = 0;
    int rnnt_turn_text_tokens = 0;

    // Function-call state machine.
    enum class FnState {
        Idle,
        WaitingForRequest,
        RequestReceived,
        SpeakingAck,
        WaitingForResponse,
        ProcessResponse
    };
    FnState fn_state = FnState::Idle;
    std::string fn_request;
    std::deque<int32_t> fn_response_tokens;
    std::deque<int32_t> fn_ack_tokens;
    int fn_frames_in_state = 0;
    std::string pending_function_text;
    std::atomic<bool> extracting_tool{false};
    std::atomic<bool> injecting_response{false};
    std::atomic<bool> extract_cancelled{false};
    std::atomic<bool> inject_cancelled{false};
    std::thread fast_path_thread;

    // Tool-ack messages parsed from the system prompt.
    std::vector<std::pair<std::string, std::vector<std::string>>> tool_ack_messages;

    bool steps_exhausted = false;
    bool started = false;
};

class S2SPipeline {
   public:
    S2SPipeline(ggml_runtime::BackendManager& bm, const S2SPipelineConfig& cfg);
    ~S2SPipeline();

    // Stream lifecycle.
    std::unique_ptr<S2SStream> create_stream(int64_t seq_id);
    // System-prompt prefill (call once right after create_stream when a
    // prompt is present). Strips <TOOL_ACK_MESSAGES> and registers acks.
    void prefill_system_prompt(S2SStream& st, const std::string& prompt);
    // Graceful completion publishes any gated RNN-T transcript tail before
    // releasing stream resources.
    S2SChunkResult finish_stream(S2SStream& st);
    // Immediate teardown for errors, cancellation, and abandoned streams.
    void end_stream(S2SStream& st);

    // Startup warmup: runs a throwaway zero-audio conversation through all
    // perception buckets so per-shape graph captures and cuDNN JIT happen
    // before the first real stream.
    void warmup();

    // Process one 160 ms input chunk (steps_per_call * samples_per_chunk
    // samples @ 16 kHz). function_response: tool result to inject ("" =
    // none). Mirrors model.py infer() for a single sequence.
    S2SChunkResult process_chunk(
        S2SStream& st, const float* audio, int n_samples, const std::string& function_response);

    const S2SPipelineConfig& config() const { return cfg_; }
    int output_sample_rate() const { return codec_->sample_rate(); }

   private:
    class LLMStepBatcher;
    class TTSStepBatcher;

    void llm_step(
        S2SStream& st, const float* encoded_audio /*[F, d_proj]*/, int n_frames,
        int step_offset = 0);
    void tts_step(S2SStream& st);
    void rnnt_eou_step(
        S2SStream& st, const float* asr_emb /*[F, joint_dim]*/, int n_frames, int step_offset);
    void rnnt_display_step(
        S2SStream& st, const float* asr_emb /*[F, joint_dim]*/, int n_frames, int step_offset);
    void apply_rnnt_turn_taking(S2SStream& st);
    void append_rnnt_display_output(
        S2SStream& st, bool agent_bos_fired, bool force_publish = false, bool close_turn = false);
    void reset_rnnt_display(S2SStream& st, bool reset_predictor = false);
    int rnnt_joint_token(
        S2SStream::RnntDecoderState& state, const float* frame, const float* logit_bias = nullptr);
    static void rnnt_commit_token(S2SStream::RnntDecoderState& state, int token);
    void decode_outputs(S2SStream& st, S2SChunkResult& out);
    void maybe_start_fast_paths(S2SStream& st);
    void fast_extract_tool_tokens(S2SStream& st);
    void fast_inject_response_tokens(S2SStream& st);
    void sync_audio_buffer_to_step(S2SStream& st);
    void parse_function_tokens(
        S2SStream& st, const std::vector<int32_t>& toks, std::string& out_function_text);
    int sample_text_token(const float* logits, S2SStream& st) const;
    std::string sanitize_function_text(const std::string& raw) const;
    std::string format_tool_response(const std::string& raw) const;
    std::string select_tool_ack(const S2SStream& st, const std::string& sanitized) const;
    void reset_fn_call_state(S2SStream& st);

    S2SPipelineConfig cfg_;
    asr::BatchingConfig batching_cfg_;
    ggml_runtime::BackendManager* bm_;

    std::unique_ptr<S2SPerception> perception_;
    std::unique_ptr<LLMBackbone> llm_;
    std::unique_ptr<LLMHeads> llm_heads_;
    std::unique_ptr<EarTTSBackbone> tts_;
    std::unique_ptr<EarTTSEmbedder> tts_embedder_;
    std::unique_ptr<EarTTSSampler> tts_sampler_;
    std::unique_ptr<EarTTSPrompt> tts_prompt_;
    std::unique_ptr<S2SCodec> codec_;
    std::unique_ptr<LLMStepBatcher> llm_step_batcher_;
    std::unique_ptr<TTSStepBatcher> tts_step_batcher_;
    std::unique_ptr<asr::IngressBatchCoordinator> ingress_batch_coordinator_;
    std::atomic<int> active_streams_{0};

    // Token ids resolved from the LLM vocab at startup.
    int32_t bos_id_ = 1, eos_id_ = 12, pad_id_ = 0;
    int32_t sotc_id_ = 20, eotc_id_ = 21;

    std::vector<std::string> rnnt_vocab_;  // RNN-T pieces (index == id)
    std::vector<int32_t> rnnt_punct_ids_;
    std::vector<float> rnnt_punct_increments_;
    std::vector<int32_t> silence_codes_;         // (Q,)
    std::vector<int32_t> tts_initial_codes_;     // prompt-final sampler output (Q,)
    std::vector<float> pad_emb_, bos_like_emb_;  // (4480,) each
};

}  // namespace nemo_speech::s2s
