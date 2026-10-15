// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts/nemotron/native_worker_protocol.h"

namespace protocol = nemo_speech::tts::nemotron::worker_protocol;
using protocol::Message;

namespace {
constexpr uint64_t kSlotBytes = 8 * 1764 * 4;
size_t assertions = 0;

void
require(bool condition, const std::string& name) {
    ++assertions;
    if (!condition)
        throw std::runtime_error(name);
}

template <class Function>
void
rejects(Function function, const std::string& name) {
    ++assertions;
    try {
        function();
    }
    catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("accepted invalid " + name);
}

std::string
bytes(std::initializer_list<unsigned> values) {
    std::string result;
    for (const auto value : values) result.push_back(static_cast<char>(value));
    return result;
}

std::string
unique_name() {
    static uint64_t counter = 0;
    std::ostringstream value;
    value << "r20-pcm-" << std::hex << std::setfill('0') << std::setw(16)
          << static_cast<uint64_t>(::getpid()) << std::setw(16)
          << (static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) +
              ++counter);
    return value.str();
}

struct Arena {
    std::string name = unique_name();
    std::string path = "/dev/shm/" + name;
    int fd = -1;

    Arena() {
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0)
            throw std::runtime_error("cannot create test PCM arena");
        if (::ftruncate(fd, 2 * kSlotBytes) != 0) {
            ::close(fd);
            ::unlink(path.c_str());
            throw std::runtime_error("cannot allocate test PCM arena");
        }
    }
    ~Arena() {
        if (fd >= 0)
            ::close(fd);
        ::unlink(path.c_str());
    }
    void write(uint64_t slot, const std::vector<float>& samples) {
        const auto count = samples.size() * sizeof(float);
        require(
            ::pwrite(fd, samples.data(), count, slot * kSlotBytes) == static_cast<ssize_t>(count),
            "test arena write failed");
    }
    Message metadata(uint64_t slot = 0, uint64_t generation = 1) const {
        return {
            {"arena", name},
            {"arena_bytes", 2 * kSlotBytes},
            {"slot_bytes", kSlotBytes},
            {"slot", slot},
            {"generation", generation},
            {"offset", slot * kSlotBytes},
            {"data_bytes", 8},
            {"samples", 2},
            {"sequence", 0},
            {"slab_version", 1},
            {"sample_rate_hz", 22050},
            {"channels", 1},
            {"encoding", "pcm_f32_le"},
            {"is_final", false}};
    }
};

void
test_parser_and_types() {
    for (const uint64_t value :
         {uint64_t{0}, uint64_t{1}, (uint64_t{1} << 53) + 1, uint64_t{1} << 63, UINT64_MAX}) {
        const auto decoded = protocol::unpack(protocol::pack(Message{{"value", value}}));
        require(protocol::integer(decoded, "value") == value, "uint64 lost precision");
    }
    for (const Message value :
         {Message(true), Message(-1), Message(1.0), Message("1"), Message(nullptr)})
        rejects([&] { protocol::integer(Message{{"value", value}}, "value"); }, "integer type");
    rejects([] { protocol::integer(Message{{"value", 3}}, "value", 2); }, "integer maximum");
    rejects([] { protocol::string(Message{{"value", false}}, "value"); }, "string type");
    rejects([] { protocol::string(Message{{"value", "abc"}}, "value", 2); }, "string length");
    const std::string valid = "Gr\xc3\xbc\xc3\x9f";
    require(
        protocol::string(protocol::unpack(protocol::pack(Message{{"value", valid}})), "value") ==
            valid,
        "UTF-8 round trip differs");
    const std::string embedded("a\0b", 3);
    require(
        protocol::string(protocol::unpack(protocol::pack(Message{{"value", embedded}})), "value") ==
            embedded,
        "embedded NUL differs");
    for (const auto& invalid :
         {bytes({0xc0, 0x80}), bytes({0xed, 0xa0, 0x80}), bytes({0xf4, 0x90, 0x80, 0x80}),
          bytes({0xe2, 0x82})})
        rejects(
            [&] { protocol::unpack(protocol::pack(Message{{"value", invalid}})); },
            "UTF-8 encoding");
    for (const double value :
         {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()})
        rejects(
            [&] { protocol::unpack(protocol::pack(Message{{"value", value}})); },
            "nonfinite number");
    rejects([] { protocol::unpack(""); }, "empty payload");
    rejects([] { protocol::unpack(std::string(16 * 1024 * 1024 + 1, '\0')); }, "oversized payload");
    rejects([] { protocol::unpack(bytes({0x82, 0xa1, 'x', 1, 0xa1, 'x', 2})); }, "duplicate key");
    rejects([] { protocol::unpack(bytes({0x80, 0x80})); }, "trailing root");
    rejects([] { protocol::unpack(bytes({0x90})); }, "non-map root");
    rejects([] { protocol::unpack(bytes({0x81, 0xa1, 'x', 0xc7, 1, 1, 0})); }, "extension type");
    rejects(
        [] { protocol::unpack(bytes({0x81, 0xa1, 'x', 0xdd, 0, 0, 0x40, 1})); },
        "array size bound");
    Message deep = Message::object();
    for (size_t depth = 0; depth < 66; ++depth) deep = Message{{"nested", std::move(deep)}};
    rejects([&] { protocol::unpack(protocol::pack(deep)); }, "nesting bound");
    const auto packed =
        protocol::pack(Message{{"kind", "submit"}, {"token_ids", Message::array({1, 131073})}});
    for (size_t length = 0; length < packed.size(); ++length)
        rejects(
            [&] { protocol::unpack(std::string_view(packed).substr(0, length)); },
            "truncated message");
}

void
test_config() {
    Arena arena;
    const auto write = [&](const std::string& data) {
        std::ofstream output(arena.path, std::ios::binary | std::ios::trunc);
        output << data;
        require(static_cast<bool>(output), "test config write failed");
    };
    write("{\"value\":18446744073709551615}");
    require(
        protocol::integer(protocol::load_config(arena.path), "value") == UINT64_MAX,
        "JSON config uint64 lost precision");
    for (const std::string value : {"{} trailing", "[]", "{\"x\":1,\"x\":2}", ""}) {
        write(value);
        rejects([&] { protocol::load_config(arena.path); }, "config shape");
    }
    write(std::string(1048577, ' '));
    rejects([&] { protocol::load_config(arena.path); }, "config size");
    rejects([] { protocol::load_config("relative.json"); }, "relative config path");
    rejects(
        [&] { protocol::load_config(arena.path + std::string("\0extra", 6)); }, "NUL config path");
}

void
test_slab_ownership_and_bounds() {
    Arena arena;
    arena.write(0, {0.25f, -0.5f});
    arena.write(1, {0.5f, -0.25f});
    protocol::SlabReader reader;
    const auto initial = reader.read(arena.metadata());
    require(
        initial.bytes == 8 && initial.samples == 2 && initial.sequence == 0, "slab shape differs");
    const std::string owned(initial.data, initial.bytes);
    arena.write(0, {1.0f, 1.0f});
    const std::vector<float> expected{0.25f, -0.5f};
    require(
        std::memcmp(owned.data(), expected.data(), owned.size()) == 0,
        "owned PCM changed after arena reuse");
    rejects([&] { reader.read(arena.metadata()); }, "repeated generation");
    reader.read(arena.metadata(0, 2));
    rejects([&] { reader.read(arena.metadata(0, 1)); }, "stale generation");
    const auto other = reader.read(arena.metadata(1));
    require(
        other.data - initial.data == static_cast<ptrdiff_t>(kSlotBytes),
        "upper slot offset differs");
    reader.read(arena.metadata(1, (uint64_t{1} << 53) + 1));
    reader.read(arena.metadata(1, (uint64_t{1} << 53) + 2));
    reader.read(arena.metadata(1, UINT64_MAX));
    rejects([&] { reader.read(arena.metadata(1, UINT64_MAX)); }, "uint64 generation replay");
    const std::vector<std::pair<std::string, Message>> invalid = {
        {"arena", "../escape"},
        {"arena", "r20-pcm-" + std::string(32, 'G')},
        {"slab_version", 2},
        {"arena_bytes", 0},
        {"arena_bytes", kSlotBytes + 1},
        {"arena_bytes", 4097 * kSlotBytes},
        {"slot_bytes", kSlotBytes - 4},
        {"slot", true},
        {"slot", -1},
        {"slot", 2},
        {"slot", 4096},
        {"generation", 0},
        {"generation", 1.0},
        {"offset", 4},
        {"offset", UINT64_MAX},
        {"data_bytes", 0},
        {"data_bytes", 7},
        {"data_bytes", kSlotBytes + 4},
        {"samples", false},
        {"samples", -1},
        {"samples", 3},
        {"sequence", uint64_t{UINT32_MAX} + 1},
        {"sample_rate_hz", 16000},
        {"channels", 2},
        {"encoding", "pcm_s16_le"},
        {"is_final", true},
        {"is_final", 0}};
    for (const auto& entry : invalid) {
        protocol::SlabReader fresh;
        auto value = arena.metadata();
        value[entry.first] = entry.second;
        rejects([&] { fresh.read(value); }, "slab " + entry.first);
        fresh.read(arena.metadata());
    }
    auto changed = arena.metadata(0, 3);
    changed["arena"] = unique_name();
    rejects([&] { reader.read(changed); }, "arena identity change");
    changed = arena.metadata(0, 3);
    changed["arena_bytes"] = kSlotBytes;
    rejects([&] { reader.read(changed); }, "arena extent change");
    reader.read(arena.metadata(0, 3));
}

void
test_slab_file_contract() {
    Arena target;
    const auto name = unique_name();
    const auto path = "/dev/shm/" + name;
    require(::symlink(target.path.c_str(), path.c_str()) == 0, "test symlink creation failed");
    auto value = target.metadata();
    value["arena"] = name;
    protocol::SlabReader reader;
    rejects([&] { reader.read(value); }, "symlink arena");
    require(::unlink(path.c_str()) == 0, "test symlink removal failed");
    require(::mkdir(path.c_str(), 0700) == 0, "test directory creation failed");
    rejects([&] { reader.read(value); }, "directory arena");
    require(::rmdir(path.c_str()) == 0, "test directory removal failed");
    require(::ftruncate(target.fd, kSlotBytes) == 0, "test truncate failed");
    rejects([&] { reader.read(target.metadata()); }, "file extent mismatch");
    require(::ftruncate(target.fd, 2 * kSlotBytes) == 0, "test restore failed");
    reader.read(target.metadata());
}
}

int
main() {
    if (::access("/dev/shm", W_OK | X_OK) != 0) {
        std::cout << "SKIP: /dev/shm unavailable for native slab ownership tests\n";
        return 77;
    }
    try {
        test_parser_and_types();
        test_config();
        test_slab_ownership_and_bounds();
        test_slab_file_contract();
        std::cout << "Native worker protocol passed " << assertions << " checks\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
