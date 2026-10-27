// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Riva StreamingTranslateSpeechToText served by a speech-LLM recognizer without an NMT model (and
// in builds without NMT): the whole stream goes to one translate request, one final result comes
// back; text translation needs NMT. A fake backend stands in for SALM; the service runs in process.
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "grpc_nmt.h"

using namespace nemo_speech;

namespace {

void
check(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error(message);
}

class TranslatingBackend final : public asr::OfflineRecognitionBackend {
   public:
    asr::Result recognize(
        const float*, size_t n, asr::AsrRequestOptions opts, const std::string&,
        int rate) override {
        samples = n;
        rate_seen = rate;
        last = std::move(opts);
        asr::Result result;
        asr::Alternative alternative;
        alternative.transcript = "Hallo Welt.";
        result.alternatives.push_back(std::move(alternative));
        return result;
    }
    void warmup() override {}
    int sample_rate() const override { return 16000; }
    bool supports_word_timestamps() const override { return false; }
    bool supports_speaker_diarization() const override { return false; }
    bool supports_translation() const override { return true; }
    std::vector<std::string> translation_languages() const override { return {"en", "de"}; }
    std::vector<std::string> supported_languages() const override { return {"auto"}; }
    asr::BatchMetrics batch_metrics() const override { return {}; }

    size_t samples = 0;
    int rate_seen = 0;
    asr::AsrRequestOptions last;
};

}  // namespace

int
main() {
    try {
        asr::RecognizerConfig config;
        config.log_status = false;
        auto backend = std::make_unique<TranslatingBackend>();
        auto* fake = backend.get();
        GrpcNmtService service(
            nullptr, nullptr, std::make_shared<asr::Recognizer>(config, std::move(backend)));
        grpc::ServerBuilder builder;
        builder.RegisterService(&service);
        const auto server = builder.BuildAndStart();
        check(server != nullptr, "in-process server did not start");
        const auto stub = nr_nmt::RivaTranslation::NewStub(server->InProcessChannel({}));

        grpc::ClientContext context;
        auto stream = stub->StreamingTranslateSpeechToText(&context);
        nr_nmt::StreamingTranslateSpeechToTextRequest request;
        auto* translation = request.mutable_config()->mutable_translation_config();
        translation->set_source_language_code("en-US");
        translation->set_target_language_code("de");
        auto* recognition = request.mutable_config()->mutable_asr_config()->mutable_config();
        recognition->set_encoding(nvidia::riva::LINEAR_PCM);
        recognition->set_sample_rate_hertz(16000);
        check(stream->Write(request), "config write failed");
        for (int i = 0; i < 2; ++i) {  // 2 x 100 ms of PCM16 silence
            nr_nmt::StreamingTranslateSpeechToTextRequest audio;
            audio.set_audio_content(std::string(3200, '\0'));
            check(stream->Write(audio), "audio write failed");
        }
        stream->WritesDone();
        nr_nmt::StreamingTranslateSpeechToTextResponse response;
        int results = 0;
        std::string transcript;
        while (stream->Read(&response)) {
            ++results;
            check(response.results_size() == 1 && response.results(0).is_final(), "not final");
            transcript = response.results(0).alternatives(0).transcript();
        }
        const auto status = stream->Finish();
        check(status.ok(), "translation failed: " + status.error_message());
        check(results == 1 && transcript == "Hallo Welt.", "one final translation expected");
        check(fake->samples == 3200 && fake->rate_seen == 16000, "the stream was not one request");
        check(
            fake->last.speechlm.task == "translate" && fake->last.speechlm.target_language == "de",
            "translation options not passed");

        grpc::ClientContext text_context;
        nr_nmt::TranslateTextRequest text_request;
        text_request.add_texts("hello");
        nr_nmt::TranslateTextResponse text_response;
        const auto text_status = stub->TranslateText(&text_context, text_request, &text_response);
        check(
            text_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION,
            "text translation without NMT must fail as a precondition");
        server->Shutdown();
        std::cout << "speech-LLM gRPC translation without NMT passed\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
