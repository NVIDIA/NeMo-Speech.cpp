// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Text-to-speech example using only the stable nemo-speech C ABI.

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nemo_speech/tts.h"

namespace {

struct Params {
    std::string magpie_model;
    std::string codec_model;
    std::string tokenizer_model_dir;
    std::string text_normalizer_model_dir;
    std::string text;
    std::string text_file;
    std::string token_text;
    std::string tokens_file;
    std::string output;
    std::string language = "en-US";
    std::string voice;
    nemo_speech_tts_runtime_config runtime = nemo_speech_tts_runtime_config_default();
    nemo_speech_tts_synthesis_options options = nemo_speech_tts_synthesis_options_default();
    bool help = false;
};

void
usage(const char* argv0) {
    std::fprintf(
        stderr,
        "usage: %s --tts.magpie-model MODEL --tts.codec-model MODEL \\\n"
        "          --tts.tokenizer-model-dir DIR --tts.text TEXT --tts.wav-out FILE [options]\n"
        "\n"
        "Input (choose text or tokens):\n"
        "  --tts.text TEXT                 Text to synthesize\n"
        "  --tts.text-file FILE            Read text from FILE\n"
        "  --tts.tokens LIST               Comma/space-separated token IDs\n"
        "  --tts.tokens-file FILE          Read token IDs from FILE\n"
        "\n"
        "Models and output:\n"
        "  --tts.magpie-model FILE         MagpieTTS GGUF\n"
        "  --tts.codec-model FILE          NanoCodec decoder GGUF\n"
        "  --tts.tokenizer-model-dir DIR   Extracted Magpie tokenizer assets\n"
        "  --tts.tn-model-dir DIR          Optional text-normalization grammars\n"
        "  --tts.wav-out FILE              Output mono PCM WAV\n"
        "  --tts.output-sample-rate HZ     Resample output (default: model rate)\n"
        "\n"
        "Synthesis:\n"
        "  --tts.language-code CODE        Language code (default: en-US)\n"
        "  --tts.voice-name NAME           Voice name or speaker index\n"
        "  --tts.speaker N                 Speaker index\n"
        "  --tts.seed N                    Sampling seed\n"
        "  --tts.steps N                   Maximum decoder frames\n"
        "  --tts.top-k N                   Top-k sampling\n"
        "  --tts.temperature F             Sampling temperature\n"
        "  --tts.cfg-scale F               Classifier-free guidance scale\n"
        "  --threads N                     CPU threads\n"
        "  --tts.codec-threads N           Codec CPU threads\n"
        "  --tts.codec-cpu                 Run NanoCodec on CPU\n"
        "  --verbose                       Enable detailed runtime logging\n",
        argv0);
}

int
parse_int(const char* value, const char* option) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be an integer: %s\n", option, value);
        std::exit(2);
    }
    return static_cast<int>(parsed);
}

float
parse_float(const char* value, const char* option) {
    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)) {
        std::fprintf(stderr, "%s must be a finite number: %s\n", option, value);
        std::exit(2);
    }
    return parsed;
}

bool
parse_args(int argc, char** argv, Params& params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* option) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", option);
                return nullptr;
            }
            return argv[++i];
        };
        const char* next = nullptr;

        if (arg == "-h" || arg == "--help") {
            params.help = true;
            return true;
        } else if (arg == "--tts.magpie-model") {
            if (!(next = value(arg.c_str())))
                return false;
            params.magpie_model = next;
        } else if (arg == "--tts.codec-model") {
            if (!(next = value(arg.c_str())))
                return false;
            params.codec_model = next;
        } else if (arg == "--tts.tokenizer-model-dir") {
            if (!(next = value(arg.c_str())))
                return false;
            params.tokenizer_model_dir = next;
        } else if (arg == "--tts.tn-model-dir") {
            if (!(next = value(arg.c_str())))
                return false;
            params.text_normalizer_model_dir = next;
        } else if (arg == "--tts.text") {
            if (!(next = value(arg.c_str())))
                return false;
            params.text = next;
        } else if (arg == "--tts.text-file") {
            if (!(next = value(arg.c_str())))
                return false;
            params.text_file = next;
        } else if (arg == "--tts.tokens") {
            if (!(next = value(arg.c_str())))
                return false;
            params.token_text = next;
        } else if (arg == "--tts.tokens-file") {
            if (!(next = value(arg.c_str())))
                return false;
            params.tokens_file = next;
        } else if (arg == "--tts.wav-out" || arg == "--output" || arg == "-o") {
            if (!(next = value(arg.c_str())))
                return false;
            params.output = next;
        } else if (arg == "--tts.language-code") {
            if (!(next = value(arg.c_str())))
                return false;
            params.language = next;
        } else if (arg == "--tts.voice-name") {
            if (!(next = value(arg.c_str())))
                return false;
            params.voice = next;
        } else if (arg == "--tts.speaker") {
            if (!(next = value(arg.c_str())))
                return false;
            params.options.speaker = parse_int(next, arg.c_str());
        } else if (arg == "--tts.seed") {
            if (!(next = value(arg.c_str())))
                return false;
            params.options.seed = parse_int(next, arg.c_str());
        } else if (arg == "--tts.steps") {
            if (!(next = value(arg.c_str())))
                return false;
            params.options.steps = parse_int(next, arg.c_str());
        } else if (arg == "--tts.top-k") {
            if (!(next = value(arg.c_str())))
                return false;
            params.options.top_k = parse_int(next, arg.c_str());
        } else if (arg == "--tts.temperature") {
            if (!(next = value(arg.c_str())))
                return false;
            params.options.temperature = parse_float(next, arg.c_str());
            params.options.override_temperature = true;
        } else if (arg == "--tts.cfg-scale") {
            if (!(next = value(arg.c_str())))
                return false;
            params.options.cfg_scale = parse_float(next, arg.c_str());
            params.options.override_cfg_scale = true;
        } else if (arg == "--tts.output-sample-rate") {
            if (!(next = value(arg.c_str())))
                return false;
            params.options.output_sample_rate = parse_int(next, arg.c_str());
        } else if (arg == "--threads") {
            if (!(next = value(arg.c_str())))
                return false;
            params.runtime.threads = parse_int(next, arg.c_str());
        } else if (arg == "--tts.codec-threads") {
            if (!(next = value(arg.c_str())))
                return false;
            params.runtime.codec_threads = parse_int(next, arg.c_str());
        } else if (arg == "--tts.codec-cpu") {
            params.runtime.codec_cpu = true;
        } else if (arg == "--verbose") {
            params.runtime.verbose = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

std::string
read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::vector<int32_t>
parse_tokens(const std::string& input) {
    std::string cleaned;
    cleaned.reserve(input.size());
    for (char ch : input) {
        cleaned.push_back(ch == ',' || ch == '[' || ch == ']' || ch == ';' ? ' ' : ch);
    }

    std::istringstream stream(cleaned);
    std::vector<int32_t> tokens;
    std::string token;
    while (stream >> token) {
        char* end = nullptr;
        errno = 0;
        const long value = std::strtol(token.c_str(), &end, 10);
        if (errno != 0 || end == token.c_str() || *end != '\0' ||
            value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("invalid token ID: " + token);
        }
        tokens.push_back(static_cast<int32_t>(value));
    }
    return tokens;
}

bool
collect_pcm(const uint8_t* pcm, size_t size, void* user_data) {
    auto* audio = static_cast<std::vector<uint8_t>*>(user_data);
    try {
        audio->insert(audio->end(), pcm, pcm + size);
        return true;
    }
    catch (...) {
        return false;
    }
}

void
write_u16(std::ofstream& output, uint16_t value) {
    output.put(static_cast<char>(value & 0xffu));
    output.put(static_cast<char>((value >> 8) & 0xffu));
}

void
write_u32(std::ofstream& output, uint32_t value) {
    output.put(static_cast<char>(value & 0xffu));
    output.put(static_cast<char>((value >> 8) & 0xffu));
    output.put(static_cast<char>((value >> 16) & 0xffu));
    output.put(static_cast<char>((value >> 24) & 0xffu));
}

bool
write_wav(const std::string& path, int sample_rate, const std::vector<uint8_t>& pcm) {
    if (sample_rate <= 0 || pcm.empty() || pcm.size() % sizeof(int16_t) != 0 ||
        pcm.size() > std::numeric_limits<uint32_t>::max() - 36u) {
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }

    const uint32_t data_size = static_cast<uint32_t>(pcm.size());
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
    output.write("RIFF", 4);
    write_u32(output, 36u + data_size);
    output.write("WAVEfmt ", 8);
    write_u32(output, 16);
    write_u16(output, 1);
    write_u16(output, channels);
    write_u32(output, static_cast<uint32_t>(sample_rate));
    write_u32(output, byte_rate);
    write_u16(output, block_align);
    write_u16(output, bits_per_sample);
    output.write("data", 4);
    write_u32(output, data_size);
    output.write(
        reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(pcm.size()));
    return static_cast<bool>(output);
}

}  // namespace

int
main(int argc, char** argv) {
    Params params;
    if (!parse_args(argc, argv, params)) {
        usage(argv[0]);
        return 2;
    }
    if (params.help) {
        usage(argv[0]);
        return 0;
    }

    try {
        if (!params.text_file.empty()) {
            params.text = read_file(params.text_file);
        }
        if (!params.tokens_file.empty()) {
            params.token_text = read_file(params.tokens_file);
        }
    }
    catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    const bool has_text = !params.text.empty();
    const bool has_tokens = !params.token_text.empty();
    if (params.magpie_model.empty() || params.codec_model.empty() || params.output.empty() ||
        has_text == has_tokens || (has_text && params.tokenizer_model_dir.empty())) {
        usage(argv[0]);
        return 2;
    }

    nemo_speech_tts_model_config model{};
    model.size = sizeof(model);
    model.magpie_model = params.magpie_model.c_str();
    model.codec_model = params.codec_model.c_str();
    model.tokenizer_model_dir =
        params.tokenizer_model_dir.empty() ? nullptr : params.tokenizer_model_dir.c_str();
    model.text_normalizer_model_dir = params.text_normalizer_model_dir.empty()
                                          ? nullptr
                                          : params.text_normalizer_model_dir.c_str();

    nemo_speech_tts_synthesizer_config config{};
    config.size = sizeof(config);
    config.model = &model;
    config.runtime = &params.runtime;
    config.default_language_code = params.language.c_str();
    config.default_voice_name = params.voice.empty() ? nullptr : params.voice.c_str();

    nemo_speech_tts_synthesizer* synthesizer = nullptr;
    nemo_speech_tts_status status = nemo_speech_tts_create(&config, &synthesizer);
    if (status != NEMO_SPEECH_TTS_OK) {
        std::fprintf(stderr, "failed to create synthesizer: %s\n", nemo_speech_tts_last_error());
        return 1;
    }

    params.options.language_code = params.language.c_str();
    params.options.voice_name = params.voice.empty() ? nullptr : params.voice.c_str();
    std::vector<uint8_t> pcm;
    nemo_speech_tts_synthesis_stats stats = nemo_speech_tts_synthesis_stats_default();
    if (has_text) {
        status = nemo_speech_tts_synthesize_text(
            synthesizer, &params.options, params.text.c_str(), collect_pcm, &pcm, &stats);
    } else {
        try {
            const std::vector<int32_t> tokens = parse_tokens(params.token_text);
            status = nemo_speech_tts_synthesize_tokens(
                synthesizer, &params.options, tokens.data(), tokens.size(), collect_pcm, &pcm,
                &stats);
        }
        catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            nemo_speech_tts_destroy(synthesizer);
            return 1;
        }
    }

    if (status != NEMO_SPEECH_TTS_OK) {
        std::fprintf(stderr, "synthesis failed: %s\n", nemo_speech_tts_last_error());
        nemo_speech_tts_destroy(synthesizer);
        return 1;
    }
    const int sample_rate =
        stats.sample_rate > 0 ? stats.sample_rate : nemo_speech_tts_sample_rate(synthesizer);
    nemo_speech_tts_destroy(synthesizer);

    if (!write_wav(params.output, sample_rate, pcm)) {
        std::fprintf(stderr, "failed to write WAV: %s\n", params.output.c_str());
        return 1;
    }
    std::fprintf(
        stdout, "wrote %s (%llu samples at %d Hz, %.2fx realtime)\n", params.output.c_str(),
        static_cast<unsigned long long>(stats.samples_written), sample_rate, stats.e2e_rtfx);
    return 0;
}
