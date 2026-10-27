// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Riva Recognize over a complete-audio (speech-LLM) backend: status mapping, the punctuation
// policy, requested-language reporting and empty speech contexts. A fake backend stands in for
// SALM.
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "errors.h"
#include "grpc_asr.h"

using namespace nemo_speech;

namespace {

void
check(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error(message);
}

class FakeBackend final : public asr::OfflineRecognitionBackend {
   public:
    asr::Result recognize(
        const float*, size_t n, asr::AsrRequestOptions opts, const std::string& language,
        int) override {
        last = std::move(opts);
        if (fail)
            fail();
        asr::Result result;
        result.audio_processed = static_cast<float>(n) / 16000.0f;
        asr::Alternative alternative;
        alternative.transcript = "Hello, World.";
        // Like SALM, the backend reports the requested language; it does not detect one.
        if (!language.empty() && language != "auto")
            alternative.language_codes.push_back(language);
        result.alternatives.push_back(std::move(alternative));
        return result;
    }
    void warmup() override {}
    int sample_rate() const override { return 16000; }
    bool supports_word_timestamps() const override { return false; }
    bool supports_speaker_diarization() const override { return false; }
    std::vector<std::string> supported_languages() const override { return {"auto"}; }
    asr::BatchMetrics batch_metrics() const override { return {}; }

    std::function<void()> fail;
    asr::AsrRequestOptions last;
};

nr_asr::RecognizeRequest
make_request(const std::string& language, bool punctuation) {
    nr_asr::RecognizeRequest request;
    auto* config = request.mutable_config();
    config->set_encoding(nr_audio::LINEAR_PCM);
    config->set_sample_rate_hertz(16000);
    config->set_language_code(language);
    config->set_enable_automatic_punctuation(punctuation);
    request.set_audio(std::string(3200, '\0'));  // 100 ms of PCM16 silence
    return request;
}

}  // namespace

int
main() {
    try {
        asr::RecognizerConfig config;
        config.log_status = false;
        auto backend = std::make_unique<FakeBackend>();
        auto* fake = backend.get();
        GrpcAsrService service(std::make_shared<asr::Recognizer>(config, std::move(backend)));

        // Riva's proto default (punctuation off) must not strip the model's own formatting.
        auto request = make_request("en-US", false);
        nr_asr::RecognizeResponse response;
        auto status = service.Recognize(nullptr, &request, &response);
        check(status.ok(), "Recognize failed: " + status.error_message());
        check(fake->last.enable_automatic_punctuation, "speech-LLM punctuation policy not applied");
        const auto& alternative = response.results(0).alternatives(0);
        check(alternative.transcript() == "Hello, World.", "transcript changed by the transport");
        check(
            alternative.language_code_size() == 1 && alternative.language_code(0) == "en-US",
            "alternative language_code lost");
        check(alternative.words_size() == 0, "timeless WordInfo for a requested language");

        // Contexts without phrases are dropped, as on HTTP; non-empty ones still reach the backend.
        request = make_request("", true);
        request.mutable_config()->add_speech_contexts()->set_boost(4.0f);
        response.Clear();
        status = service.Recognize(nullptr, &request, &response);
        check(status.ok() && fake->last.speech_contexts.empty(), "empty speech context kept");
        request.mutable_config()->mutable_speech_contexts(0)->add_phrases("NeMo");
        response.Clear();
        status = service.Recognize(nullptr, &request, &response);
        check(status.ok() && fake->last.speech_contexts.size() == 1, "speech context dropped");

        struct Case {
            std::function<void()> fail;
            grpc::StatusCode expected;
        };
        const Case cases[] = {
            {[] { throw std::invalid_argument("bad request"); },
             grpc::StatusCode::INVALID_ARGUMENT},
            {[] { throw asr::ResourceExhausted("queue is full"); },
             grpc::StatusCode::RESOURCE_EXHAUSTED},
            {[] { throw asr::Unavailable("shutting down"); }, grpc::StatusCode::UNAVAILABLE},
            {[] { throw std::runtime_error("engine failed"); }, grpc::StatusCode::INTERNAL},
            {[] { throw 7; }, grpc::StatusCode::INTERNAL},
        };
        for (const auto& c : cases) {
            fake->fail = c.fail;
            request = make_request("", true);
            response.Clear();
            status = service.Recognize(nullptr, &request, &response);
            check(
                status.error_code() == c.expected, "unexpected status " +
                                                       std::to_string(status.error_code()) +
                                                       " for " + status.error_message());
        }
        std::cout << "gRPC Recognize over a complete-audio backend passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "test_grpc_asr_offline: " << error.what() << '\n';
        return 1;
    }
}
