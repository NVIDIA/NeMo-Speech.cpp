// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared process-level owner for the full-duplex speech-to-speech runtime.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "pipeline.h"

namespace nemo_speech::s2s {

struct VoiceChatConfig {
    S2SPipelineConfig pipeline;
    int gpu = 0;  // -1 selects the CPU backend.
    bool verbose = false;
};

// Resolve the conventional files produced by the unified S2S converter.
VoiceChatConfig voicechat_config_from_model_dir(
    const std::string& model_dir, int gpu = 0, int max_streams = 32);

// Protocol frontends share one VoiceChat. Conversation state remains isolated
// in S2SStream instances while model weights, schedulers, and backend resources
// are owned exactly once by this object.
class VoiceChat {
   public:
    explicit VoiceChat(VoiceChatConfig config);
    ~VoiceChat();
    VoiceChat(const VoiceChat&) = delete;
    VoiceChat& operator=(const VoiceChat&) = delete;

    std::unique_ptr<S2SStream> create_stream();
    void prefill_system_prompt(S2SStream& stream, const std::string& prompt);
    S2SChunkResult process_chunk(
        S2SStream& stream, const float* audio, int samples,
        const std::string& function_response = {});
    S2SChunkResult finish_stream(S2SStream& stream);
    void end_stream(S2SStream& stream);

    void warmup();
    const S2SPipelineConfig& config() const;
    int input_sample_rate() const { return 16000; }
    int output_sample_rate() const;
    int samples_per_step() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::s2s
