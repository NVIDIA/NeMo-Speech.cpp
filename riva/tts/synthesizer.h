// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Transport-neutral text-to-PCM orchestration.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "magpietts/runtime.h"
#include "tts/tokenizer/tokenizer.h"

namespace nemo_speech::tts {

struct SynthesizerConfig {
    MagpieRuntimeConfig runtime;
    std::string tokenizer_model_dir;
    std::string text_normalizer_model_dir;
    MagpieTokenizerConfig tokenizer;
    std::string default_language_code = "en-US";
    std::string default_voice_name;
};

struct SynthesisRequest {
    std::string text;
    std::string language_code;
    std::string voice_name;
    int output_sample_rate = 0;  // 0 = model rate
    MagpieSynthesisOptions options;
};

struct SynthesisMetadata {
    std::string original_text;
    std::string processed_text;
    std::string language_code;
    std::string tokenizer_name;
    int speaker = 0;
    int sample_rate = 0;
    size_t token_count = 0;
    size_t chunk_count = 0;
};

struct PreparedSynthesis {
    SynthesisMetadata metadata;
    std::vector<int32_t> tokens;
    std::vector<std::vector<int32_t>> token_chunks;
    MagpieSynthesisOptions options;
    double normalizer_ms = 0.0;
    double tokenizer_ms = 0.0;
};

struct SynthesisResult {
    SynthesisMetadata metadata;
    MagpieSynthesisStats stats;
    uint64_t output_samples = 0;
    bool cancelled = false;
};

class Synthesizer {
   public:
    using PcmCallback = std::function<bool(const SynthesisMetadata&, const std::string& pcm_s16le)>;

    explicit Synthesizer(SynthesizerConfig config);
    ~Synthesizer();

    Synthesizer(const Synthesizer&) = delete;
    Synthesizer& operator=(const Synthesizer&) = delete;

    PreparedSynthesis prepare(const SynthesisRequest& request) const;
    SynthesisResult synthesize(const PreparedSynthesis& request, const PcmCallback& callback = {});
    SynthesisResult synthesize(const SynthesisRequest& request, const PcmCallback& callback = {});
    SynthesisResult synthesize_tokens(
        const std::vector<int32_t>& tokens, const MagpieSynthesisOptions& options = {},
        int output_sample_rate = 0, const PcmCallback& callback = {});

    // Initialize lazy runtime state through the normal synthesis path.
    SynthesisResult warmup(const std::string& text, int steps = -1);

    int sample_rate() const;
    int speaker_count() const;
    const std::vector<std::string>& speaker_names() const;
    std::vector<std::string> supported_language_codes() const;
    const std::string& model_name() const;
    const std::string& default_language_code() const;
    int default_speaker() const;
    bool text_normalization_enabled() const;

   private:
    int resolve_speaker(const std::string& voice_name) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts
