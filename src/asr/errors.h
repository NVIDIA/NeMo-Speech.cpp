// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Request failures that transports map to a specific status. std::invalid_argument is a client
// error (HTTP 400, gRPC INVALID_ARGUMENT); any other exception is an internal error (HTTP 500, gRPC
// INTERNAL).
#pragma once

#include <stdexcept>
#include <string>

namespace nemo_speech::asr {

// Capacity is exhausted right now (queue full, too many uploads): HTTP 429, gRPC
// RESOURCE_EXHAUSTED.
class ResourceExhausted : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

// The service cannot take the request (shutting down, admission wait expired): HTTP 503, gRPC
// UNAVAILABLE.
class Unavailable : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

// The client went away (AsrRequestOptions::cancelled) and the request was dropped: HTTP 499 (no one
// reads it), gRPC CANCELLED.
class Cancelled : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

}  // namespace nemo_speech::asr
