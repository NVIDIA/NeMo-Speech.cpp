// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "audio_file.h"
#include "cli_util.h"
#include "engine_registry.h"
#include "json.h"

namespace {

void
append_u16(std::string& bytes, uint16_t value) {
    bytes.push_back(static_cast<char>(value));
    bytes.push_back(static_cast<char>(value >> 8));
}

void
append_u32(std::string& bytes, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) bytes.push_back(static_cast<char>(value >> shift));
}

std::string
wav_header(
    uint16_t format, uint16_t channels, uint32_t sample_rate, uint16_t bits, uint32_t data_size) {
    std::string bytes = "RIFF";
    append_u32(bytes, data_size + 36);
    bytes += "WAVEfmt ";
    append_u32(bytes, 16);
    append_u16(bytes, format);
    append_u16(bytes, channels);
    append_u32(bytes, sample_rate);
    const uint16_t align = channels * bits / 8;
    append_u32(bytes, sample_rate * align);
    append_u16(bytes, align);
    append_u16(bytes, bits);
    bytes += "data";
    append_u32(bytes, data_size);
    return bytes;
}

void
require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Function>
void
require_throws(Function&& function, const char* message) {
    try {
        function();
    }
    catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

template <typename Function>
void
require_throws_with(Function&& function, const char* expected, const char* message) {
    try {
        function();
    }
    catch (const std::exception& error) {
        require(std::string(error.what()).find(expected) != std::string::npos, message);
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int
main() {
    try {
        using nemo_speech::json::Value;
        const auto value = Value::parse(
            R"({"emoji":"\ud83d\ude80","items":[true,null,-12.5e2],"escaped":"a\nb"})");
        require(value.at("emoji").string() == "\xf0\x9f\x9a\x80", "Unicode surrogate decoding");
        require(value.at("items").array().size() == 3, "JSON array size");
        require(value.at("escaped").string() == "a\nb", "JSON escape decoding");
        const auto roundtrip = Value::parse(value.dump());
        require(roundtrip.at("emoji").string() == value.at("emoji").string(), "JSON round trip");
        require_throws([] { Value::parse(R"({"duplicate":1,"duplicate":2})"); }, "duplicate keys");
        require_throws([] { Value::parse("01"); }, "leading-zero number");
        require_throws(
            [] { Value::parse(std::string(129, '[') + "0" + std::string(129, ']')); },
            "excessive JSON parse depth");
        Value deeply_nested = 0;
        for (int i = 0; i < 129; ++i) {
            Value::Array parent;
            parent.push_back(std::move(deeply_nested));
            deeply_nested = Value(std::move(parent));
        }
        require_throws([&] { deeply_nested.dump(); }, "excessive JSON serialization depth");

        std::string pcm = wav_header(1, 2, 48000, 16, 8);
        append_u16(pcm, static_cast<uint16_t>(16384));
        append_u16(pcm, static_cast<uint16_t>(-16384));
        append_u16(pcm, static_cast<uint16_t>(32767));
        append_u16(pcm, static_cast<uint16_t>(32767));
        const auto stereo = nemo_speech::audio::load_wav_memory(pcm.data(), pcm.size(), "stereo");
        require(stereo.sample_rate == 48000 && stereo.source_channels == 2, "WAV metadata");
        require(stereo.samples.size() == 2, "WAV frame count");
        require(std::abs(stereo.samples[0]) < 1e-6f, "stereo downmix");
        require(stereo.samples[1] > 0.99f, "PCM16 decoding");

        const float samples[] = {0.25f, -0.5f};
        std::string floating = wav_header(3, 1, 44100, 32, sizeof(samples));
        floating.append(reinterpret_cast<const char*>(samples), sizeof(samples));
        const auto decoded =
            nemo_speech::audio::load_wav_memory(floating.data(), floating.size(), "float");
        require(decoded.sample_rate == 44100 && decoded.samples.size() == 2, "float WAV metadata");
        require(std::abs(decoded.samples[1] + 0.5f) < 1e-6f, "float WAV decoding");

        require_throws(
            [&] { nemo_speech::audio::load_wav_memory(pcm.data(), pcm.size() - 1, "truncated"); },
            "truncated WAV rejection");
        const std::string unsupported(44, '\0');
        require_throws_with(
            [&] {
                nemo_speech::audio::load_wav_memory(
                    unsupported.data(), unsupported.size(), "recording.mp3");
            },
            "ffmpeg -i INPUT -ac 1 -ar 16000 -c:a pcm_s16le output.wav",
            "unsupported container conversion guidance");
        require(nemo_speech::audio::is_wav_path("VOICE.WAVE"), "case-insensitive WAV path");
        require(!nemo_speech::audio::is_wav_path("voice.mp3"), "codec path rejection");

        nemo_speech::EngineRegistry registry;
        require(!registry.ready(), "empty engine registry readiness");
        require(registry.capabilities().empty(), "empty engine registry capabilities");
        registry.set_device_label("cpu");
        require(registry.device_label() == "cpu", "engine registry device label");

        namespace fs = std::filesystem;
        const fs::path root = fs::temp_directory_path() / "nemo-speech-input";
        require(
            relative_output_path(root, root / "nested/audio.wav") == fs::path("nested/audio.wav"),
            "nested output path");
        require(
            relative_output_path(root, root / "linked.wav") == fs::path("linked.wav"),
            "lexical output path");
        require(
            relative_output_path(root, root.parent_path() / "outside.wav") ==
                fs::path("outside.wav"),
            "outside path containment");
        std::cout << "shared utility tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "test_shared_utilities: " << error.what() << '\n';
        return 1;
    }
}
