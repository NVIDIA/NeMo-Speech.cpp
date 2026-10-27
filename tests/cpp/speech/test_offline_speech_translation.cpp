// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Speech translation over a complete-audio (speech-LLM) backend: the cascade buffers the audio and
// recognizes it whole at an endpoint or at finish. A fake backend stands in for SALM.
//   test_offline_speech_translation [NMT.gguf]   the model adds an end-to-end translation check
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "speech_translator.h"

using namespace nemo_speech;

namespace {

void
check(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error(message);
}

template <class F>
std::string
invalid_argument_message(F&& f) {
    try {
        f();
    }
    catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}

class FakeBackend final : public asr::OfflineRecognitionBackend {
   public:
    struct Call {
        size_t samples = 0;
        int rate = 0;
        std::string language;
        asr::AsrRequestOptions options;
    };
    asr::Result recognize(
        const float* samples, size_t n, asr::AsrRequestOptions opts, const std::string& language,
        int rate) override {
        check(samples != nullptr, "null samples");
        calls.push_back({n, rate, language, std::move(opts)});
        asr::Result result;
        asr::Alternative alternative;
        alternative.transcript = transcript;
        result.alternatives.push_back(std::move(alternative));
        return result;
    }
    void warmup() override {}
    int sample_rate() const override { return 16000; }
    bool supports_word_timestamps() const override { return false; }
    bool supports_speaker_diarization() const override { return false; }
    std::vector<std::string> supported_languages() const override { return {"auto"}; }
    asr::BatchMetrics batch_metrics() const override { return {}; }

    std::string transcript;  // empty: the cascade skips translation
    std::vector<Call> calls;
};

speech::SpeechTranslationOptions
options(const std::string& source) {
    speech::SpeechTranslationOptions result;
    result.source_language = source;
    result.target_language = "es-US";
    result.recognition.language_code = source;
    return result;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        asr::RecognizerConfig config;
        config.log_status = false;
        auto backend = std::make_unique<FakeBackend>();
        auto* fake = backend.get();
        auto recognizer = std::make_shared<asr::Recognizer>(config, std::move(backend));
        check(!recognizer->supports_streaming(), "fake backend must be complete-audio only");

        // Never dereferenced: an empty transcript ends the cascade before translation, and the
        // missing-source check runs before the translator. SpeechTranslator only requires non-null.
        alignas(nmt::Translator) unsigned char unused[sizeof(nmt::Translator)];
        speech::SpeechTranslator offline(
            recognizer, std::shared_ptr<nmt::Translator>(
                            std::shared_ptr<void>(), reinterpret_cast<nmt::Translator*>(unused)));

        // Chunks accumulate and are recognized once at finish, with the request options.
        const std::vector<float> chunk(800, 0.0f);
        auto stream = offline.streaming_translate(options("en-US"));
        for (int i = 0; i < 3; ++i) stream->push(chunk.data(), chunk.size(), 8000);
        check(fake->calls.empty(), "audio recognized before an endpoint");
        stream->finish();
        check(fake->calls.size() == 1, "finish did not recognize the buffered audio");
        check(
            fake->calls[0].samples == 3 * chunk.size() && fake->calls[0].rate == 8000 &&
                fake->calls[0].language == "en-US",
            "buffered audio, rate or language lost");
        check(
            fake->calls[0].options.enable_automatic_punctuation,
            "speech-LLM punctuation policy not applied");

        // An endpoint recognizes what is buffered so far; later audio forms the next utterance.
        fake->calls.clear();
        stream = offline.streaming_translate(options("en-US"));
        stream->force_endpoint();
        check(fake->calls.empty(), "endpoint without audio recognized nothing");
        stream->push(chunk.data(), chunk.size(), 0);  // 0 = model rate
        stream->force_endpoint();
        stream->push(chunk.data(), chunk.size(), 16000);
        stream->push(chunk.data(), chunk.size(), 16000);
        stream->finish();
        check(
            fake->calls.size() == 2 && fake->calls[0].samples == chunk.size() &&
                fake->calls[0].rate == 16000 && fake->calls[1].samples == 2 * chunk.size(),
            "endpoint did not split the buffered audio");

        // The input rate is fixed per stream, and the buffer is bounded (gRPC has no upload limit).
        stream = offline.streaming_translate(options("en-US"));
        stream->push(chunk.data(), chunk.size(), 16000);
        check(
            !invalid_argument_message([&] {
                 stream->push(chunk.data(), chunk.size(), 8000);
             }).empty(),
            "sample-rate change accepted");
        stream = offline.streaming_translate(options("en-US"));
        const std::vector<float> second(8000, 0.0f);
        const std::string overflow = invalid_argument_message([&] {
            for (int i = 0; i < 4 * 3600; ++i) stream->push(second.data(), second.size(), 8000);
        });
        check(!overflow.empty(), "unbounded speech translation buffer");

        // Without a source language a speech-LLM backend cannot name one: a client error.
        fake->transcript = "Hello world.";
        stream = offline.streaming_translate(options(""));
        stream->push(chunk.data(), chunk.size(), 16000);
        const std::string missing = invalid_argument_message([&] { stream->finish(); });
        check(
            missing.find("source language") != std::string::npos &&
                missing.find("source_language") == std::string::npos &&
                missing.find("--language") == std::string::npos,
            "missing source language message: " + missing);

        if (argc > 1) {
            nmt::TranslatorConfig nmt_config;
            nmt_config.model.path = argv[1];
            speech::SpeechTranslator cascade(
                recognizer, std::make_shared<nmt::Translator>(std::move(nmt_config)));
            std::vector<speech::SpeechTranslationResult> translated;
            speech::SpeechTranslationCallbacks callbacks;
            callbacks.translation = [&](const speech::SpeechTranslationResult& result) {
                translated.push_back(result);
                return true;
            };
            stream = cascade.streaming_translate(options("en-US"), std::move(callbacks));
            stream->push(chunk.data(), chunk.size(), 16000);
            stream->finish();
            check(
                translated.size() == 1 && translated[0].transcript == "Hello world." &&
                    !translated[0].text.empty(),
                "end-to-end translation failed");
            std::printf(
                "translated: %s -> %s\n", translated[0].transcript.c_str(),
                translated[0].text.c_str());
        }
        std::cout << "offline speech translation passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "test_offline_speech_translation: " << error.what() << '\n';
        return 1;
    }
}
