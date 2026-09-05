// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <httplib.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::s2s {
class VoiceChat;
}

namespace nemo_speech::http {
struct ServerConfig;

void handle_s2s_realtime(
    const httplib::Request& request, httplib::ws::WebSocket& socket,
    std::shared_ptr<s2s::VoiceChat> voicechat, const ServerConfig& config,
    std::atomic<uint64_t>& event_ids);

std::string select_s2s_subprotocol(const std::vector<std::string>& protocols);

}  // namespace nemo_speech::http
