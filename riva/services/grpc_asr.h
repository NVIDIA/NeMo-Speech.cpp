// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Riva-compatible gRPC adapter over the transport-neutral ASR recognizer.
#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

#include "recognizer.h"  // asr::Recognizer + asr::RecognizerConfig
#include "riva/proto/riva_asr.grpc.pb.h"
#include "riva/proto/riva_audio.pb.h"

namespace nr_asr = nvidia::riva::asr;
namespace nr_audio = nvidia::riva;

namespace nemo_speech {

class GrpcAsrService final : public nr_asr::RivaSpeechRecognition::Service {
   public:
    explicit GrpcAsrService(std::shared_ptr<asr::Recognizer> recognizer);
    ~GrpcAsrService() override;

    grpc::Status Recognize(
        grpc::ServerContext* ctx, const nr_asr::RecognizeRequest* req,
        nr_asr::RecognizeResponse* resp) override;

    grpc::Status StreamingRecognize(
        grpc::ServerContext* ctx,
        grpc::ServerReaderWriter<
            nr_asr::StreamingRecognizeResponse, nr_asr::StreamingRecognizeRequest>* stream)
        override;

    grpc::Status GetRivaSpeechRecognitionConfig(
        grpc::ServerContext* ctx, const nr_asr::RivaSpeechRecognitionConfigRequest* req,
        nr_asr::RivaSpeechRecognitionConfigResponse* resp) override;

   private:
    // The library; owns the shared backend/model/postproc and is the runner
    // factory. This service is a thin transport adapter over it.
    std::shared_ptr<asr::Recognizer> recognizer_;
};

}  // namespace nemo_speech
