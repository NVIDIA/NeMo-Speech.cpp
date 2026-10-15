// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "native_worker_protocol.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nemo_speech::tts::nemotron::worker_protocol {
namespace {

bool
valid_utf8(const std::string& text) {
    size_t offset = 0;
    while (offset < text.size()) {
        const auto first = static_cast<unsigned char>(text[offset++]);
        if (first < 0x80)
            continue;
        const int count = first >= 0xc2 && first <= 0xdf   ? 1
                          : first >= 0xe0 && first <= 0xef ? 2
                          : first >= 0xf0 && first <= 0xf4 ? 3
                                                           : 0;
        if (count == 0 || text.size() - offset < static_cast<size_t>(count))
            return false;
        const auto second = static_cast<unsigned char>(text[offset]);
        if ((first == 0xe0 && second < 0xa0) || (first == 0xed && second >= 0xa0) ||
            (first == 0xf0 && second < 0x90) || (first == 0xf4 && second >= 0x90))
            return false;
        for (int i = 0; i < count; ++i)
            if ((static_cast<unsigned char>(text[offset++]) & 0xc0) != 0x80)
                return false;
    }
    return true;
}

class BoundedSax final : public nlohmann::json_sax<Message> {
   public:
    Message result;
    std::string error;

    bool null() override { return append(nullptr); }
    bool boolean(bool value) override { return append(value); }
    bool number_integer(number_integer_t value) override { return append(value); }
    bool number_unsigned(number_unsigned_t value) override { return append(value); }
    bool number_float(number_float_t value, const string_t&) override {
        return std::isfinite(value) ? append(value) : fail("nonfinite native protocol number");
    }
    bool string(string_t& value) override {
        return valid_utf8(value) ? append(std::move(value)) : fail("invalid native protocol UTF-8");
    }
    bool binary(binary_t& value) override {
        return value.has_subtype() ? fail("native protocol extensions are unsupported")
                                   : append(Message::binary(std::move(value)));
    }
    bool start_object(std::size_t elements) override { return start(true, elements); }
    bool start_array(std::size_t elements) override { return start(false, elements); }
    bool key(string_t& value) override {
        if (stack_.empty() || stack_.size() > 64 || !stack_.back().value.is_object() ||
            !valid_utf8(value))
            return fail("invalid native protocol object key");
        auto& frame = stack_.back();
        if (frame.value.size() >= 16384 || frame.value.contains(value))
            return fail("duplicate or excessive native protocol object keys");
        frame.key = std::move(value);
        frame.pending_key = true;
        return true;
    }
    bool end_object() override { return finish(true); }
    bool end_array() override { return finish(false); }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override {
        return fail("malformed native protocol payload");
    }

   private:
    struct Frame {
        Message value;
        std::string key;
        bool pending_key = false;
    };
    std::vector<Frame> stack_;
    bool complete_ = false;

    bool fail(const char* reason) {
        if (error.empty())
            error = reason;
        return false;
    }
    bool append(Message value) {
        if (stack_.size() > 64)
            return fail("native protocol nesting limit exceeded");
        if (stack_.empty()) {
            if (complete_)
                return fail("multiple native protocol roots");
            result = std::move(value);
            complete_ = true;
            return true;
        }
        auto& frame = stack_.back();
        if (frame.value.is_object()) {
            if (!frame.pending_key)
                return fail("missing native protocol object key");
            frame.value.emplace(std::move(frame.key), std::move(value));
            frame.pending_key = false;
        } else {
            if (frame.value.size() >= 16384)
                return fail("excessive native protocol array entries");
            frame.value.push_back(std::move(value));
        }
        return true;
    }
    bool start(bool object, size_t elements) {
        if (stack_.size() > 64 || (elements != size_t(-1) && elements > 16384))
            return fail("native protocol collection limit exceeded");
        stack_.push_back({object ? Message::object() : Message::array(), {}, false});
        return true;
    }
    bool finish(bool object) {
        if (stack_.empty() || stack_.back().value.is_object() != object ||
            stack_.back().pending_key)
            return fail("invalid native protocol collection");
        auto value = std::move(stack_.back().value);
        stack_.pop_back();
        return append(std::move(value));
    }
};

Message
parse(std::string_view payload, Message::input_format_t format) {
    BoundedSax sax;
    if (!Message::sax_parse(payload.begin(), payload.end(), &sax, format, true) ||
        !sax.result.is_object())
        throw std::runtime_error(
            sax.error.empty() ? "native protocol root must be a map" : sax.error);
    return std::move(sax.result);
}

}

Message
unpack(std::string_view payload) {
    if (payload.empty() || payload.size() > 16 * 1024 * 1024)
        throw std::runtime_error("invalid native MessagePack payload size");
    return parse(payload, Message::input_format_t::msgpack);
}

Message
load_config(const std::string& path) {
    if (path.empty() || path.front() != '/' || path.find('\0') != std::string::npos)
        throw std::runtime_error("direct native worker configuration path must be absolute");
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read direct native worker configuration");
    std::string content(1048577, '\0');
    input.read(content.data(), static_cast<std::streamsize>(content.size()));
    const auto bytes = input.gcount();
    if (input.bad() || bytes < 2 || bytes > 1048576)
        throw std::runtime_error("invalid direct native worker configuration size");
    content.resize(static_cast<size_t>(bytes));
    return parse(content, Message::input_format_t::json);
}

std::string
pack(const Message& value) {
    const auto bytes = Message::to_msgpack(value);
    if (bytes.empty() || bytes.size() > 16 * 1024 * 1024)
        throw std::runtime_error("invalid native MessagePack payload size");
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

uint64_t
integer(const Message& value, const char* key, uint64_t maximum) {
    const auto& field = value.at(key);
    uint64_t result;
    if (field.is_number_unsigned())
        result = field.get<uint64_t>();
    else if (field.is_number_integer() && field.get<int64_t>() >= 0)
        result = static_cast<uint64_t>(field.get<int64_t>());
    else
        throw std::runtime_error(std::string("invalid native worker integer: ") + key);
    if (result > maximum)
        throw std::runtime_error(std::string("native worker integer exceeds limit: ") + key);
    return result;
}

std::string
string(const Message& value, const char* key, size_t maximum) {
    const auto& field = value.at(key);
    if (!field.is_string())
        throw std::runtime_error(std::string("invalid native worker string: ") + key);
    const auto& result = field.get_ref<const std::string&>();
    if (result.size() > maximum || !valid_utf8(result))
        throw std::runtime_error(std::string("native worker string exceeds contract: ") + key);
    return result;
}

SlabReader::~SlabReader() {
    if (mapping_)
        ::munmap(const_cast<char*>(mapping_), extent_);
}

Slab
SlabReader::read(const Message& value) {
    constexpr uint64_t slot_bytes = 8 * 1764 * 4;
    const auto arena = string(value, "arena", 40);
    const auto extent = integer(value, "arena_bytes", 4096 * slot_bytes);
    const auto slot = integer(value, "slot", 4095);
    const auto generation = integer(value, "generation");
    const auto offset = integer(value, "offset");
    const auto bytes = integer(value, "data_bytes", slot_bytes);
    const auto samples = integer(value, "samples", slot_bytes / 4);
    const auto sequence = integer(value, "sequence", UINT32_MAX);
    bool valid_name = arena.size() == 40 && arena.compare(0, 8, "r20-pcm-") == 0;
    for (size_t index = 8; valid_name && index < arena.size(); ++index)
        valid_name = (arena[index] >= '0' && arena[index] <= '9') ||
                     (arena[index] >= 'a' && arena[index] <= 'f');
    const auto previous = generations_.find(slot);
    if (!valid_name || integer(value, "slab_version") != 1 ||
        integer(value, "slot_bytes") != slot_bytes || extent == 0 || extent % slot_bytes ||
        slot >= extent / slot_bytes || offset != slot * slot_bytes || bytes == 0 ||
        bytes != samples * 4 || offset > extent || bytes > extent - offset || generation == 0 ||
        (previous != generations_.end() && generation <= previous->second) ||
        integer(value, "sample_rate_hz") != 22050 || integer(value, "channels") != 1 ||
        string(value, "encoding") != "pcm_f32_le" || !value.at("is_final").is_boolean() ||
        value.at("is_final").get<bool>())
        throw std::runtime_error("invalid native PCM slab metadata");
    if (!mapping_) {
        const std::string path = "/dev/shm/" + arena;
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0)
            throw std::runtime_error("cannot open native PCM arena");
        struct stat info {};
        if (::fstat(fd, &info) < 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
            static_cast<uint64_t>(info.st_size) != extent) {
            ::close(fd);
            throw std::runtime_error("native PCM arena extent differs from metadata");
        }
        void* mapping = ::mmap(nullptr, static_cast<size_t>(extent), PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (mapping == MAP_FAILED)
            throw std::runtime_error("cannot map native PCM arena");
        mapping_ = static_cast<const char*>(mapping);
        extent_ = static_cast<size_t>(extent);
        arena_ = arena;
    }
    if (arena != arena_ || extent != extent_)
        throw std::runtime_error("native PCM arena identity changed during a connection");
    generations_[slot] = generation;
    return {mapping_ + static_cast<size_t>(offset), static_cast<size_t>(bytes), samples, sequence};
}

}
#endif
