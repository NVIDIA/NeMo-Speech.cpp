// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "model.h"
#include "nvtx_utils.h"

namespace nc = nemo_speech::tts::nanocodec;

struct nc_params {
    std::string model;
    std::string codes;
    std::string output;
    int chunk_frames = 3;
    int threads = 4;
    bool force_cpu = false;
    bool stream = false;
    bool show_help = false;
};

static std::string
read_file(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("failed to open " + path);
    }
    std::ostringstream ss;
    ss << fin.rdbuf();
    return ss.str();
}

static std::vector<int32_t>
parse_ints(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size());
    for (char ch : text) {
        if (ch == ',' || ch == '[' || ch == ']' || ch == ';') {
            cleaned.push_back(' ');
        } else {
            cleaned.push_back(ch);
        }
    }

    std::istringstream iss(cleaned);
    std::vector<int32_t> values;
    std::string token;
    while (iss >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string discard;
            std::getline(iss, discard);
            continue;
        }
        char* end = nullptr;
        errno = 0;
        const long value = std::strtol(token.c_str(), &end, 10);
        if (errno != 0 || end == token.c_str() || *end != '\0') {
            throw std::runtime_error("invalid integer token: " + token);
        }
        values.push_back((int32_t)value);
    }
    return values;
}

static nc::NanoCodecFrames
read_codes(const std::string& path, int codebooks) {
    if (codebooks != 8) {
        throw std::runtime_error("this example currently expects 8 codec codebooks");
    }

    const std::string text = read_file(path);
    std::vector<int32_t> values = parse_ints(text);
    if (values.empty() || values.size() % (size_t)codebooks != 0) {
        throw std::runtime_error("codec token file must contain a multiple of 8 integer tokens");
    }

    nc::NanoCodecFrames frames(values.size() / (size_t)codebooks);
    for (size_t i = 0; i < frames.size(); ++i) {
        for (int c = 0; c < codebooks; ++c) {
            frames[i][c] = values[i * (size_t)codebooks + c];
        }
    }
    return frames;
}

static void
write_u16(std::ofstream& out, uint16_t v) {
    out.put((char)(v & 0xff));
    out.put((char)((v >> 8) & 0xff));
}

static void
write_u32(std::ofstream& out, uint32_t v) {
    out.put((char)(v & 0xff));
    out.put((char)((v >> 8) & 0xff));
    out.put((char)((v >> 16) & 0xff));
    out.put((char)((v >> 24) & 0xff));
}

static bool
write_wav(const std::string& path, const std::vector<float>& audio, int sample_rate) {
    const ggml_nvtx::range nvtx_range("nanocodec_write_wav");
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "failed to open %s for writing\n", path.c_str());
        return false;
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = (uint32_t)sample_rate * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t data_bytes = (uint32_t)audio.size() * block_align;

    out.write("RIFF", 4);
    write_u32(out, 36 + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32(out, 16);
    write_u16(out, 1);
    write_u16(out, channels);
    write_u32(out, (uint32_t)sample_rate);
    write_u32(out, byte_rate);
    write_u16(out, block_align);
    write_u16(out, bits_per_sample);
    out.write("data", 4);
    write_u32(out, data_bytes);

    for (float x : audio) {
        x = std::max(-1.0f, std::min(1.0f, x));
        const int32_t v = (int32_t)std::lrintf(x * 32767.0f);
        write_u16(out, (uint16_t)(int16_t)v);
    }

    return (bool)out;
}

static void
print_usage(const char* argv0) {
    fprintf(
        stderr,
        "usage: %s -m nano-codec.gguf --codes codec_tokens.txt -o audio.wav [options]\n"
        "\n"
        "options:\n"
        "  -m, --model PATH   NanoCodec decoder GGUF\n"
        "  --codes PATH       MagpieTTS codec-token file, 8 integers per frame\n"
        "  -o, --output PATH  Output mono 16-bit PCM WAV\n"
        "  -t, --threads N    CPU threads (default: 4)\n"
        "  --stream           Decode with stateful streaming graph chunks\n"
        "  --chunk-frames N   Streaming chunk size in codec frames (default: 3)\n"
        "  --cpu              Force NanoCodec decoder onto CPU backend\n",
        argv0);
}

static bool
parse_args(int argc, char** argv, nc_params& params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* opt) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", opt);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model") {
            params.model = need_value(arg.c_str());
        } else if (arg == "--codes") {
            params.codes = need_value(arg.c_str());
        } else if (arg == "-o" || arg == "--output") {
            params.output = need_value(arg.c_str());
        } else if (arg == "-t" || arg == "--threads") {
            params.threads = std::max(1, std::atoi(need_value(arg.c_str())));
        } else if (arg == "--stream") {
            params.stream = true;
        } else if (arg == "--chunk-frames") {
            params.chunk_frames = std::max(1, std::atoi(need_value(arg.c_str())));
        } else if (arg == "--cpu") {
            params.force_cpu = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            params.show_help = true;
            return false;
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (params.model.empty() || params.codes.empty() || params.output.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

int
main(int argc, char** argv) {
    const ggml_nvtx::range nvtx_main("nanocodec_main");
    ggml_time_init();

    nc_params params;
    if (!parse_args(argc, argv, params)) {
        return params.show_help ? 0 : 1;
    }

    nc::NanoCodecModel model;
    if (!model.load(params.model, params.force_cpu)) {
        return 1;
    }

    nc::NanoCodecFrames frames;
    try {
        frames = read_codes(params.codes, model.numCodebooks());
    }
    catch (const std::exception& e) {
        fprintf(stderr, "failed to read codec tokens: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "decoding %zu codec frames\n", frames.size());
    nc::NanoCodecDecoder decoder(model);
    std::vector<float> audio;
    const int64_t t_start = ggml_time_us();
    const bool decoded =
        params.stream ? decoder.decodeStreamAll(frames, params.chunk_frames, params.threads, audio)
                      : decoder.decode(frames, params.threads, audio);
    if (!decoded) {
        return 1;
    }

    const double elapsed_ms = (ggml_time_us() - t_start) / 1000.0;
    const double audio_s =
        model.sampleRate() > 0 ? (double)audio.size() / (double)model.sampleRate() : 0.0;
    const double elapsed_s = elapsed_ms / 1000.0;
    const double rtf = audio_s > 0.0 ? elapsed_s / audio_s : 0.0;
    const double rtfx = elapsed_s > 0.0 ? audio_s / elapsed_s : 0.0;
    fprintf(
        stderr,
        "decode mode=%s frames=%zu samples=%zu audio=%.3f s elapsed=%.2f ms rtf=%.4f rtfx=%.2f\n",
        params.stream ? "stream" : "offline", frames.size(), audio.size(), audio_s, elapsed_ms, rtf,
        rtfx);

    if (!write_wav(params.output, audio, model.sampleRate())) {
        return 1;
    }

    fprintf(
        stderr, "wrote %s (%zu samples at %d Hz)\n", params.output.c_str(), audio.size(),
        model.sampleRate());
    return 0;
}
