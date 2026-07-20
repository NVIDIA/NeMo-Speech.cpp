// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Transport-neutral streaming speech translation and optional speech synthesis.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "recognizer.h"
#include "translator.h"
#include "tts/synthesizer.h"

namespace nemo_speech::speech {

struct SpeechTranslationOptions {
    asr::AsrRequestOptions recognition;
    std::string source_language;
    std::string target_language;
    bool synthesize_speech = false;
    tts::SynthesisRequest synthesis;
};

struct SpeechTranslationResult {
    std::string transcript;
    std::string text;
    std::string language_code;
};

struct SpeechTranslationCallbacks {
    // Called once for each translated ASR final, before any synthesized audio.
    std::function<bool(const SpeechTranslationResult&)> translation;
    // Called for every synthesized PCM chunk. PCM is mono signed-16 LE at
    // metadata.sample_rate.
    std::function<bool(
        const SpeechTranslationResult&, const tts::SynthesisMetadata&,
        const std::string& pcm_s16le)>
        audio;
    // Called after synthesis completes for one translation. This is a semantic
    // utterance boundary; transports decide how to encode it.
    std::function<bool(const SpeechTranslationResult&)> utterance_end;
};

class SpeechTranslationStream {
   public:
    ~SpeechTranslationStream();

    SpeechTranslationStream(const SpeechTranslationStream&) = delete;
    SpeechTranslationStream& operator=(const SpeechTranslationStream&) = delete;

    void push(const float* samples, size_t count, int sample_rate = 0);
    void force_endpoint();
    void finish();
    bool cancelled() const;

   private:
    friend class SpeechTranslator;
    struct Impl;
    explicit SpeechTranslationStream(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class SpeechTranslator {
   public:
    SpeechTranslator(
        std::shared_ptr<asr::Recognizer> recognizer, std::shared_ptr<nmt::Translator> translator,
        std::shared_ptr<tts::Synthesizer> synthesizer = nullptr);

    std::unique_ptr<SpeechTranslationStream> streaming_translate(
        SpeechTranslationOptions options, SpeechTranslationCallbacks callbacks = {}) const;
    SpeechTranslationResult translate_text(
        const std::string& transcript, const std::string& source_language,
        const std::string& target_language,
        const std::vector<std::string>& detected_languages = {}) const;

    int sample_rate() const;
    bool can_synthesize() const { return synthesizer_ != nullptr; }

   private:
    std::shared_ptr<asr::Recognizer> recognizer_;
    std::shared_ptr<nmt::Translator> translator_;
    std::shared_ptr<tts::Synthesizer> synthesizer_;
};

}  // namespace nemo_speech::speech
