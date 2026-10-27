// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// One request-error mapping for the Riva gRPC services, matching the HTTP API and the CLI:
// std::invalid_argument -> INVALID_ARGUMENT, asr::ResourceExhausted -> RESOURCE_EXHAUSTED,
// asr::Unavailable -> UNAVAILABLE, asr::Cancelled -> CANCELLED, anything else -> INTERNAL.
#pragma once

#include <grpcpp/grpcpp.h>

#include <exception>
#include <stdexcept>

#include "errors.h"  // asr::ResourceExhausted, asr::Unavailable, asr::Cancelled

namespace nemo_speech {

inline grpc::Status
grpc_status_from_exception(const std::exception_ptr& error) {
    try {
        if (error)
            std::rethrow_exception(error);
    }
    catch (const std::invalid_argument& e) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    }
    catch (const asr::ResourceExhausted& e) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, e.what());
    }
    catch (const asr::Unavailable& e) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, e.what());
    }
    catch (const asr::Cancelled& e) {
        return grpc::Status(grpc::StatusCode::CANCELLED, e.what());
    }
    catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
    catch (...) {
    }
    return grpc::Status(grpc::StatusCode::INTERNAL, "unknown internal error");
}

}  // namespace nemo_speech
