// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "speech_translator.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "langpairs.h"

namespace nemo_speech::speech {
namespace {

std::string
resolve_source_language(
    const asr::Recognizer& recognizer, std::string source_language,
    std::vector<std::string> detected_languages) {
    if (!source_language.empty())
        return source_language;
    detected_languages.erase(
        std::remove_if(
            detected_languages.begin(), detected_languages.end(),
            [](const std::string& language) { return language.empty() || language == "auto"; }),
        detected_languages.end());
    std::sort(detected_languages.begin(), detected_languages.end());
    detected_languages.erase(
        std::unique(detected_languages.begin(), detected_languages.end()),
        detected_languages.end());
    if (detected_languages.size() == 1)
        return detected_languages.front();

    auto supported = recognizer.supported_languages();
    supported.erase(
        std::remove_if(
            supported.begin(), supported.end(),
            [](const std::string& language) { return language.empty() || language == "auto"; }),
        supported.end());
    std::sort(supported.begin(), supported.end());
    supported.erase(std::unique(supported.begin(), supported.end()), supported.end());
    return supported.size() == 1 ? supported.front() : std::string();
}

SpeechTranslationResult
translate_text(
    const asr::Recognizer& recognizer, nmt::Translator& translator, const std::string& transcript,
    const std::string& requested_source, const std::string& requested_target,
    const std::vector<std::string>& detected_languages) {
    if (requested_target.empty())
        throw std::invalid_argument("target_language is required");
    const std::string source =
        resolve_source_language(recognizer, requested_source, detected_languages);
    if (source.empty())
        throw std::invalid_argument(
            "the source language could not be detected; provide source_language");
    const auto output = translator.translate({transcript}, source, requested_target);
    SpeechTranslationResult result;
    result.transcript = transcript;
    if (!output.empty()) {
        result.text = output.front().text;
        result.language_code = output.front().language;
    }
    return result;
}

}  // namespace

struct SpeechTranslationStream::Impl {
    Impl(
        std::shared_ptr<asr::Recognizer> recognizer, std::shared_ptr<nmt::Translator> translator,
        std::shared_ptr<tts::Synthesizer> synthesizer, SpeechTranslationOptions options,
        SpeechTranslationCallbacks callbacks,
        std::unique_ptr<asr::RecognitionStream> recognition_stream)
        : recognizer(std::move(recognizer)), translator(std::move(translator)),
          synthesizer(std::move(synthesizer)), options(std::move(options)),
          callbacks(std::move(callbacks)), recognition_stream(std::move(recognition_stream)) {}

    bool emit_final(const asr::Result& result) {
        if (cancelled)
            return false;
        if (result.alternatives.empty() || result.alternatives.front().transcript.empty())
            return true;

        SpeechTranslationResult translated = translate_text(
            *recognizer, *translator, result.alternatives.front().transcript,
            options.source_language, options.target_language,
            result.alternatives.front().language_codes);
        if (callbacks.translation && !callbacks.translation(translated)) {
            cancelled = true;
            return false;
        }
        if (!options.synthesize_speech)
            return true;

#if defined(NEMO_SPEECH_SPEECH_TTS)
        if (!translated.text.empty()) {
            tts::SynthesisRequest request = options.synthesis;
            request.text = translated.text;
            if (request.language_code.empty())
                request.language_code = translated.language_code;
            const auto synthesis_result = synthesizer->synthesize(
                request, [&](const tts::SynthesisMetadata& metadata, const std::string& pcm) {
                    if (!callbacks.audio)
                        return true;
                    if (!callbacks.audio(translated, metadata, pcm)) {
                        cancelled = true;
                        return false;
                    }
                    return true;
                });
            if (synthesis_result.cancelled) {
                cancelled = true;
                return false;
            }
        }
        if (callbacks.utterance_end && !callbacks.utterance_end(translated)) {
            cancelled = true;
            return false;
        }
        return true;
#else
        throw std::invalid_argument("speech synthesis requested without TTS support");
#endif
    }

    void drain_finals() {
        auto result = recognition_stream->next();
        while (result && result->is_final && !cancelled) {
            if (!emit_final(*result))
                return;
            result = recognition_stream->next();
        }
    }

    std::shared_ptr<asr::Recognizer> recognizer;
    std::shared_ptr<nmt::Translator> translator;
    std::shared_ptr<tts::Synthesizer> synthesizer;
    SpeechTranslationOptions options;
    SpeechTranslationCallbacks callbacks;
    std::unique_ptr<asr::RecognitionStream> recognition_stream;
    bool cancelled = false;
    bool finished = false;
};

SpeechTranslationStream::SpeechTranslationStream(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SpeechTranslationStream::~SpeechTranslationStream() = default;

void
SpeechTranslationStream::push(const float* samples, size_t count, int sample_rate) {
    if (impl_->finished)
        throw std::logic_error("cannot push audio after speech translation finish");
    if (impl_->cancelled)
        return;
    impl_->recognition_stream->push(samples, count, sample_rate);
    impl_->drain_finals();
}

void
SpeechTranslationStream::force_endpoint() {
    if (impl_->finished)
        throw std::logic_error("cannot force an endpoint after speech translation finish");
    if (impl_->cancelled)
        return;
    impl_->recognition_stream->force_endpoint();
    impl_->drain_finals();
}

void
SpeechTranslationStream::finish() {
    if (impl_->finished)
        return;
    impl_->finished = true;
    if (impl_->cancelled)
        return;
    impl_->emit_final(impl_->recognition_stream->finish());
}

bool
SpeechTranslationStream::cancelled() const {
    return impl_->cancelled;
}

SpeechTranslator::SpeechTranslator(
    std::shared_ptr<asr::Recognizer> recognizer, std::shared_ptr<nmt::Translator> translator,
    std::shared_ptr<tts::Synthesizer> synthesizer)
    : recognizer_(std::move(recognizer)), translator_(std::move(translator)),
      synthesizer_(std::move(synthesizer)) {
    if (!recognizer_)
        throw std::invalid_argument("SpeechTranslator requires an ASR recognizer");
    if (!translator_)
        throw std::invalid_argument("SpeechTranslator requires an NMT translator");
}

std::unique_ptr<SpeechTranslationStream>
SpeechTranslator::streaming_translate(
    SpeechTranslationOptions options, SpeechTranslationCallbacks callbacks) const {
    if (options.target_language.empty())
        throw std::invalid_argument("target_language is required");
    if (options.source_language.empty()) {
        auto languages = recognizer_->supported_languages();
        languages.erase(
            std::remove_if(
                languages.begin(), languages.end(),
                [](const std::string& language) { return language.empty() || language == "auto"; }),
            languages.end());
        std::sort(languages.begin(), languages.end());
        languages.erase(std::unique(languages.begin(), languages.end()), languages.end());
        if (languages.size() == 1)
            options.source_language = languages.front();
    }
    if (!options.source_language.empty()) {
        const std::string tag =
            nmt::langpairs::resolve_tag(options.source_language, options.target_language);
        if (tag.empty()) {
            throw std::invalid_argument(
                "unsupported language pair: " + options.source_language + " -> " +
                options.target_language);
        }
        const auto languages = nmt::langpairs::split_tag(tag);
        options.source_language = languages.first;
        options.target_language = languages.second;
    }
    if (options.synthesize_speech && !can_synthesize())
        throw std::invalid_argument("speech synthesis requested without a TTS synthesizer");
    const std::string language = options.recognition.language_code;
    auto recognition = recognizer_->streaming_recognize(options.recognition, language);
    auto impl = std::make_unique<SpeechTranslationStream::Impl>(
        recognizer_, translator_, synthesizer_, std::move(options), std::move(callbacks),
        std::move(recognition));
    return std::unique_ptr<SpeechTranslationStream>(new SpeechTranslationStream(std::move(impl)));
}

SpeechTranslationResult
SpeechTranslator::translate_text(
    const std::string& transcript, const std::string& source_language,
    const std::string& target_language, const std::vector<std::string>& detected_languages) const {
    return speech::translate_text(
        *recognizer_, *translator_, transcript, source_language, target_language,
        detected_languages);
}

int
SpeechTranslator::sample_rate() const {
    return recognizer_->sample_rate();
}

}  // namespace nemo_speech::speech
