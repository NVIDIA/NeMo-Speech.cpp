// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Riva NMT protocol adapter. Translation and speech composition live in core.
#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>

#include "riva/proto/riva_nmt.grpc.pb.h"
#include "speech_translator.h"
#include "translator.h"

namespace nr_nmt = nvidia::riva::nmt;

namespace nemo_speech {

class GrpcNmtService final : public nr_nmt::RivaTranslation::Service {
   public:
    GrpcNmtService(
        std::shared_ptr<nmt::Translator> translator,
        std::shared_ptr<speech::SpeechTranslator> speech_translator = nullptr);
    ~GrpcNmtService() override;

    grpc::Status TranslateText(
        grpc::ServerContext* ctx, const nr_nmt::TranslateTextRequest* req,
        nr_nmt::TranslateTextResponse* resp) override;

    grpc::Status ListSupportedLanguagePairs(
        grpc::ServerContext* ctx, const nr_nmt::AvailableLanguageRequest* req,
        nr_nmt::AvailableLanguageResponse* resp) override;

    grpc::Status StreamingTranslateSpeechToText(
        grpc::ServerContext* ctx,
        grpc::ServerReaderWriter<
            nr_nmt::StreamingTranslateSpeechToTextResponse,
            nr_nmt::StreamingTranslateSpeechToTextRequest>* stream) override;

    grpc::Status StreamingTranslateSpeechToSpeech(
        grpc::ServerContext* ctx,
        grpc::ServerReaderWriter<
            nr_nmt::StreamingTranslateSpeechToSpeechResponse,
            nr_nmt::StreamingTranslateSpeechToSpeechRequest>* stream) override;

   private:
    std::shared_ptr<nmt::Translator> translator_;
    std::shared_ptr<speech::SpeechTranslator> speech_translator_;
};

}  // namespace nemo_speech
