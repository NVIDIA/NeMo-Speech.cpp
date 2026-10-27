// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// One request-error mapping for the HTTP API, matching gRPC and the CLI: audio::AudioError -> its
// own status (422 for audio that cannot be decoded, 400 otherwise), std::invalid_argument -> 400,
// asr::ResourceExhausted -> 429, asr::Unavailable -> 503, asr::Cancelled -> 499 (the client is
// gone), anything else -> 500.
#pragma once

#include <exception>
#include <stdexcept>
#include <string>

#include "audio_ingest.h"  // audio::AudioError

#if defined(NEMO_SPEECH_REGISTRY_ASR)
#include "errors.h"  // asr::ResourceExhausted, asr::Unavailable, asr::Cancelled
#endif

namespace nemo_speech::http {

struct RequestError {
    int status = 500;
    std::string type = "server_error";    // OpenAI error.type
    std::string code = "internal_error";  // realtime error.code: the gRPC status name, lower case
    std::string message = "unknown server error";
};

inline RequestError
request_error(const std::exception_ptr& error) {
    try {
        if (error)
            std::rethrow_exception(error);
    }
    catch (const audio::AudioError& e) {  // before invalid_argument, which it derives from
        return {e.http_status(), "invalid_request_error", "invalid_argument", e.what()};
    }
    catch (const std::invalid_argument& e) {
        return {400, "invalid_request_error", "invalid_argument", e.what()};
    }
#if defined(NEMO_SPEECH_REGISTRY_ASR)
    catch (const asr::ResourceExhausted& e) {
        return {429, "rate_limit_error", "resource_exhausted", e.what()};
    }
    catch (const asr::Unavailable& e) {
        return {503, "server_error", "unavailable", e.what()};
    }
    catch (const asr::Cancelled& e) {
        return {499, "client_closed_request", "cancelled", e.what()};
    }
#endif
    catch (const std::exception& e) {
        return {500, "server_error", "internal_error", e.what()};
    }
    catch (...) {
    }
    return {};
}

}  // namespace nemo_speech::http
