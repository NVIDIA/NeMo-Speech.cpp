// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Process-level owner for loaded speech engines. Frontends translate their
// request/response formats and borrow these typed engines; model lifecycle,
// warmup, capability reporting, and cross-pipeline composition live here.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(NEMO_SPEECH_REGISTRY_DIAR)
#include "diarizer.h"
#endif
#if defined(NEMO_SPEECH_REGISTRY_ASR)
#include "recognizer.h"
#endif
#if defined(NEMO_SPEECH_REGISTRY_TTS)
#include "tts/synthesizer.h"
#endif
#if defined(NEMO_SPEECH_REGISTRY_NMT)
#include "translator.h"
#endif
#if defined(NEMO_SPEECH_REGISTRY_SPEECH)
#include "speech_translator.h"
#endif

namespace nemo_speech {

struct WarmupOptions {
    bool asr = true;
    bool tts = true;
    std::string tts_text = "Hello";
    int tts_steps = 1;
};

struct EngineRegistryConfig {
    bool asr = false;
    bool nmt = false;
};

class EngineRegistry {
   public:
    explicit EngineRegistry(EngineRegistryConfig config = {});
    ~EngineRegistry() = default;
    EngineRegistry(const EngineRegistry&) = delete;
    EngineRegistry& operator=(const EngineRegistry&) = delete;

#if defined(NEMO_SPEECH_REGISTRY_DIAR)
    std::shared_ptr<asr::Diarizer> load_diarization(
        int gpu, const std::string& model_path,
        asr::DiarGeometry geometry = asr::DiarGeometry::preset("streaming"));
    std::shared_ptr<asr::Diarizer> diarization() const;
#endif
#if defined(NEMO_SPEECH_REGISTRY_ASR)
    std::shared_ptr<asr::Recognizer> load_asr(asr::RecognizerConfig config);
    std::shared_ptr<asr::Recognizer> asr() const;
#endif
#if defined(NEMO_SPEECH_REGISTRY_TTS)
    std::shared_ptr<tts::Synthesizer> load_tts(tts::SynthesizerConfig config);
    std::shared_ptr<tts::Synthesizer> tts() const;
#endif
#if defined(NEMO_SPEECH_REGISTRY_NMT)
    std::shared_ptr<nmt::Translator> load_nmt(nmt::TranslatorConfig config);
    std::shared_ptr<nmt::Translator> nmt() const;
#endif
#if defined(NEMO_SPEECH_REGISTRY_SPEECH)
    std::shared_ptr<speech::SpeechTranslator> speech_translation() const;
#endif

    void warmup(const WarmupOptions& options = {});
    void set_device_label(std::string label);
    std::string device_label() const;
    std::vector<std::string> capabilities() const;
    bool ready() const;

   private:
    void rebuild_compositions_locked();

    mutable std::mutex mutex_;
    std::string device_label_ = "auto";
#if defined(NEMO_SPEECH_REGISTRY_DIAR)
    std::shared_ptr<asr::Diarizer> diar_;
    bool diar_from_asr_ = false;
#endif
#if defined(NEMO_SPEECH_REGISTRY_ASR)
    std::shared_ptr<asr::Recognizer> asr_;
#endif
#if defined(NEMO_SPEECH_REGISTRY_TTS)
    std::shared_ptr<tts::Synthesizer> tts_;
#endif
#if defined(NEMO_SPEECH_REGISTRY_NMT)
    std::shared_ptr<nmt::Translator> nmt_;
#endif
#if defined(NEMO_SPEECH_REGISTRY_SPEECH)
    std::shared_ptr<speech::SpeechTranslator> speech_;
#endif
};

}  // namespace nemo_speech
