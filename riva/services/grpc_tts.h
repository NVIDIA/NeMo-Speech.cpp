// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Riva TTS protocol adapter. Product behavior lives in tts::Synthesizer.
#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>

#include "riva/proto/riva_tts.grpc.pb.h"
#include "tts/synthesizer.h"

namespace nr_tts = nvidia::riva::tts;

namespace nemo_speech {

class GrpcTtsService final : public nr_tts::RivaSpeechSynthesis::Service {
   public:
    explicit GrpcTtsService(std::shared_ptr<tts::Synthesizer> synthesizer, bool benchmark = false);
    ~GrpcTtsService() override;

    grpc::Status Synthesize(
        grpc::ServerContext* ctx, const nr_tts::SynthesizeSpeechRequest* req,
        nr_tts::SynthesizeSpeechResponse* resp) override;

    grpc::Status SynthesizeOnline(
        grpc::ServerContext* ctx,
        grpc::ServerReaderWriter<nr_tts::SynthesizeSpeechResponse, nr_tts::SynthesizeSpeechRequest>*
            stream) override;

    grpc::Status GetRivaSynthesisConfig(
        grpc::ServerContext* ctx, const nr_tts::RivaSynthesisConfigRequest* req,
        nr_tts::RivaSynthesisConfigResponse* resp) override;

   private:
    std::shared_ptr<tts::Synthesizer> synthesizer_;
    bool benchmark_ = false;
};

}  // namespace nemo_speech
