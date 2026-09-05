// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace nemo_speech {
class EngineRegistry;
namespace http {

struct ServerConfig {
    std::string address = "127.0.0.1";
    int port = 8080;
    int threads = 4;
    int read_timeout_seconds = 30;
    int write_timeout_seconds = 30;
    size_t max_upload_bytes = 512ULL * 1024ULL * 1024ULL;
    bool access_log = false;
    bool json_logs = false;
    // Emit per-request TTS runtime timings to stderr.
    bool tts_benchmark = false;
    // Keep only the latest HTTP TTS request; newer requests cancel older synthesis.
    bool preempt_tts = false;
    std::string tls_certificate;
    std::string tls_private_key;
    std::string api_key;
    std::string cors_origin;
    bool enable_playground = true;
    int realtime_s2s_max_session_seconds = 300;
    size_t realtime_s2s_max_pending_function_responses = 64;
    bool realtime_s2s_output_text_events = false;
};

class Server {
   public:
    Server(EngineRegistry& engines, ServerConfig config);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Blocks until stop() is called or the listening socket fails.
    bool listen();
    bool bind();
    bool listen_after_bind();
    void stop();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace http
}  // namespace nemo_speech
