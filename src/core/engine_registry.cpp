// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "engine_registry.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace nemo_speech {
namespace {

#if defined(NEMO_SPEECH_REGISTRY_TTS) || defined(NEMO_SPEECH_REGISTRY_NMT)
void
set_environment_default(const char* name, const char* value) {
    if (std::getenv(name))
        return;
#if defined(_WIN32)
    if (_putenv_s(name, value) != 0)
        throw std::runtime_error(std::string("could not set process default ") + name);
#else
    if (setenv(name, value, 0) != 0)
        throw std::runtime_error(std::string("could not set process default ") + name);
#endif
}
#endif

}  // namespace

EngineRegistry::EngineRegistry(EngineRegistryConfig config) {
#if defined(NEMO_SPEECH_REGISTRY_NMT)
    if (config.nmt)
        set_environment_default(config.asr ? "GGML_SKINNY_Q8_INPLACE" : "GGML_SKINNY_Q8", "0");
#else
    (void)config;
#endif
}

#if defined(NEMO_SPEECH_REGISTRY_DIAR)
std::shared_ptr<asr::Diarizer>
EngineRegistry::load_diarization(
    int gpu, const std::string& model_path, asr::DiarGeometry geometry,
    asr::BatchingConfig batching) {
    auto engine = asr::Diarizer::load(gpu, model_path, std::move(geometry), std::move(batching));
    std::lock_guard<std::mutex> lock(mutex_);
    diar_ = engine;
    diar_from_asr_ = false;
    return engine;
}

std::shared_ptr<asr::Diarizer>
EngineRegistry::diarization() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (diar_)
        return diar_;
    throw std::runtime_error("diarization is not loaded");
}
#endif

#if defined(NEMO_SPEECH_REGISTRY_ASR)
std::shared_ptr<asr::Recognizer>
EngineRegistry::load_asr(asr::RecognizerConfig config) {
    auto engine = std::make_shared<asr::Recognizer>(std::move(config));
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(NEMO_SPEECH_REGISTRY_DIAR)
    if (diar_from_asr_)
        diar_.reset();
#endif
    asr_ = engine;
#if defined(NEMO_SPEECH_REGISTRY_DIAR)
    if (engine->diar_model()) {
        auto model = std::shared_ptr<asr::DiarModel>(engine, engine->diar_model());
        diar_ = std::make_shared<asr::Diarizer>(
            std::move(model), engine->config().diar.resolved_geometry());
        diar_from_asr_ = true;
    }
#endif
    rebuild_compositions_locked();
    return engine;
}

std::shared_ptr<asr::Recognizer>
EngineRegistry::asr() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!asr_)
        throw std::runtime_error("ASR is not loaded");
    return asr_;
}
#endif

#if defined(NEMO_SPEECH_REGISTRY_TTS)
std::shared_ptr<tts::Synthesizer>
EngineRegistry::load_tts(tts::SynthesizerConfig config) {
    set_environment_default("GGML_CUDA_GRAPH_EVICT_AFTER_MS", "0");
    auto engine = std::make_shared<tts::Synthesizer>(std::move(config));
    std::lock_guard<std::mutex> lock(mutex_);
    tts_ = engine;
    rebuild_compositions_locked();
    return engine;
}

std::shared_ptr<tts::Synthesizer>
EngineRegistry::tts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tts_)
        throw std::runtime_error("TTS is not loaded");
    return tts_;
}
#endif

#if defined(NEMO_SPEECH_REGISTRY_NMT)
std::shared_ptr<nmt::Translator>
EngineRegistry::load_nmt(nmt::TranslatorConfig config) {
#if defined(NEMO_SPEECH_REGISTRY_ASR)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (asr_)
            set_environment_default("GGML_SKINNY_Q8_INPLACE", "0");
        else
            set_environment_default("GGML_SKINNY_Q8", "0");
    }
#else
    set_environment_default("GGML_SKINNY_Q8", "0");
#endif
    auto engine = std::make_shared<nmt::Translator>(std::move(config));
    std::lock_guard<std::mutex> lock(mutex_);
    nmt_ = engine;
    rebuild_compositions_locked();
    return engine;
}

std::shared_ptr<nmt::Translator>
EngineRegistry::nmt() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!nmt_)
        throw std::runtime_error("NMT is not loaded");
    return nmt_;
}
#endif

#if defined(NEMO_SPEECH_REGISTRY_SPEECH)
std::shared_ptr<speech::SpeechTranslator>
EngineRegistry::speech_translation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!speech_)
        throw std::runtime_error("speech translation requires loaded ASR and NMT engines");
    return speech_;
}
#endif

#if defined(NEMO_SPEECH_REGISTRY_S2S)
std::shared_ptr<s2s::VoiceChat>
EngineRegistry::load_voicechat(s2s::VoiceChatConfig config) {
    auto voicechat = std::make_shared<s2s::VoiceChat>(std::move(config));
    std::lock_guard<std::mutex> lock(mutex_);
    voicechat_ = voicechat;
    return voicechat;
}

std::shared_ptr<s2s::VoiceChat>
EngineRegistry::voicechat() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!voicechat_)
        throw std::runtime_error("VoiceChat is not loaded");
    return voicechat_;
}
#endif

void
EngineRegistry::rebuild_compositions_locked() {
#if defined(NEMO_SPEECH_REGISTRY_SPEECH)
    speech_.reset();
    if (asr_ && nmt_) {
#if defined(NEMO_SPEECH_REGISTRY_TTS)
        speech_ = std::make_shared<speech::SpeechTranslator>(asr_, nmt_, tts_);
#else
        speech_ = std::make_shared<speech::SpeechTranslator>(asr_, nmt_);
#endif
    }
#endif
}

void
EngineRegistry::warmup(const WarmupOptions& options) {
#if !defined(NEMO_SPEECH_REGISTRY_ASR) && !defined(NEMO_SPEECH_REGISTRY_TTS) && \
    !defined(NEMO_SPEECH_REGISTRY_S2S)
    (void)options;
#endif
#if defined(NEMO_SPEECH_REGISTRY_ASR)
    std::shared_ptr<asr::Recognizer> recognizer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        recognizer = asr_;
    }
    if (recognizer && options.asr)
        recognizer->warmup();
#endif
#if defined(NEMO_SPEECH_REGISTRY_TTS)
    std::shared_ptr<tts::Synthesizer> synthesizer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        synthesizer = tts_;
    }
    if (synthesizer && options.tts)
        synthesizer->warmup(options.tts_text, options.tts_steps);
#endif
#if defined(NEMO_SPEECH_REGISTRY_S2S)
    std::shared_ptr<s2s::VoiceChat> voicechat;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        voicechat = voicechat_;
    }
    if (voicechat && options.s2s)
        voicechat->warmup();
#endif
}

void
EngineRegistry::set_device_label(std::string label) {
    std::lock_guard<std::mutex> lock(mutex_);
    device_label_ = std::move(label);
}

std::string
EngineRegistry::device_label() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_label_;
}

std::vector<std::string>
EngineRegistry::capabilities() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
#if defined(NEMO_SPEECH_REGISTRY_ASR)
    if (asr_)
        result.emplace_back("asr");
#endif
#if defined(NEMO_SPEECH_REGISTRY_DIAR)
    if (diar_)
        result.emplace_back("diarization");
#endif
#if defined(NEMO_SPEECH_REGISTRY_TTS)
    if (tts_)
        result.emplace_back("tts");
#endif
#if defined(NEMO_SPEECH_REGISTRY_NMT)
    if (nmt_)
        result.emplace_back("translation");
#endif
#if defined(NEMO_SPEECH_REGISTRY_SPEECH)
    if (speech_) {
        result.emplace_back("speech-translation");
        if (speech_->can_synthesize())
            result.emplace_back("speech-to-speech");
    }
#endif
#if defined(NEMO_SPEECH_REGISTRY_S2S)
    if (voicechat_)
        result.emplace_back("realtime-speech-to-speech");
#endif
    return result;
}

bool
EngineRegistry::ready() const {
    return !capabilities().empty();
}

}  // namespace nemo_speech
