// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <stdexcept>
#include <string>

#include "engine_registry.h"
#include "http_server.h"

int
main() {
    nemo_speech::EngineRegistry engines;
    nemo_speech::http::ServerConfig config;
    config.threads = 1;
    config.preempt_tts = true;

    bool rejected = false;
    try {
        nemo_speech::http::Server server(engines, config);
    }
    catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("http.threads >= 2") != std::string::npos) {
            rejected = true;
        } else {
            std::cerr << "FAIL: unexpected validation error: " << error.what() << '\n';
            return 1;
        }
    }
    if (!rejected) {
        std::cerr << "FAIL: single-worker TTS preemption was accepted\n";
        return 1;
    }

    config.threads = 2;
    try {
        nemo_speech::http::Server server(engines, config);
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: two-worker TTS preemption was rejected: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
