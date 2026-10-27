// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// HTTP status and error type for each request-error class (the same classes gRPC and the CLI map).
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "http_errors.h"

namespace {

int failures = 0;

template <class E>
void
expect(E error, int status, const std::string& type, const std::string& code) {
    const auto mapped = nemo_speech::http::request_error(std::make_exception_ptr(error));
    if (mapped.status != status || mapped.type != type || mapped.code != code) {
        std::cerr << "FAIL: " << mapped.message << " -> " << mapped.status << ' ' << mapped.type
                  << ' ' << mapped.code << " (want " << status << ' ' << type << ' ' << code
                  << ")\n";
        ++failures;
    }
}

}  // namespace

int
main() {
    expect(std::invalid_argument("bad"), 400, "invalid_request_error", "invalid_argument");
    // audio errors keep their status: undecodable audio is 422, audio over the limits 400
    expect(
        nemo_speech::audio::AudioError(422, "unsupported audio format"), 422,
        "invalid_request_error", "invalid_argument");
    expect(
        nemo_speech::audio::AudioLimitError("audio is 2.0 s, the limit is 1 s"), 400,
        "invalid_request_error", "invalid_argument");
#if defined(NEMO_SPEECH_REGISTRY_ASR)
    expect(
        nemo_speech::asr::ResourceExhausted("full"), 429, "rate_limit_error", "resource_exhausted");
    expect(nemo_speech::asr::Unavailable("stopping"), 503, "server_error", "unavailable");
#endif
    expect(std::runtime_error("engine"), 500, "server_error", "internal_error");
    expect(42, 500, "server_error", "internal_error");
    if (failures == 0)
        std::cout << "HTTP request-error mapping passed\n";
    return failures == 0 ? 0 : 1;
}
