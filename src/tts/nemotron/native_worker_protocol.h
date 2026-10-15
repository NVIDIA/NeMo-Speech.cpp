// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef _WIN32
#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace nemo_speech::tts::nemotron::worker_protocol {

using Message = nlohmann::json;

Message unpack(std::string_view payload);
Message load_config(const std::string& path);
std::string pack(const Message& value);
uint64_t integer(const Message& value, const char* key, uint64_t maximum = UINT64_MAX);
std::string string(const Message& value, const char* key, size_t maximum = 1048576);

struct Slab {
    const char* data;
    size_t bytes;
    uint64_t samples;
    uint64_t sequence;
};

class SlabReader {
   public:
    SlabReader() = default;
    ~SlabReader();
    SlabReader(const SlabReader&) = delete;
    SlabReader& operator=(const SlabReader&) = delete;
    Slab read(const Message& value);

   private:
    const char* mapping_ = nullptr;
    size_t extent_ = 0;
    std::string arena_;
    std::map<uint64_t, uint64_t> generations_;
};

}
#endif
