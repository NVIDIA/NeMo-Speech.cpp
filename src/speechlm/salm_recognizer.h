// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "recognizer.h"

namespace nemo_speech::speechlm {
struct SalmResult;
// Transport boundary: seconds -> milliseconds; speaker tags -> one-based IDs.
asr::Result to_asr_result(
    const SalmResult&, const asr::AsrRequestOptions&, const std::string& language,
    const asr::postproc::Postprocessor&);
// Used by EngineRegistry so all complete-audio transports share the same engine.
std::unique_ptr<asr::OfflineRecognitionBackend> make_salm_backend(
    const asr::RecognizerConfig& config);
}  // namespace nemo_speech::speechlm
