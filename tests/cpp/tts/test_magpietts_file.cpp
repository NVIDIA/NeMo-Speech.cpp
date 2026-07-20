// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// MagpieTTS file-output smoke test. Uses the streaming runtime, buffers emitted
// PCM chunks, then writes one WAV file after synthesis finishes.
//
// Usage:
//   test_magpietts_file --tts.magpie-model magpie.gguf --tts.codec-model codec.gguf
//       --tts.tokens-file tokens.txt --tts.wav-out out.wav [options]

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "tts/magpietts/magpietts.h"

namespace tts = nemo_speech::tts;

namespace {

void
usage(const char* argv0) {
    std::fprintf(
        stderr,
        "usage: %s --tts.magpie-model magpie.gguf --tts.codec-model codec.gguf "
        "--tts.tokens-file tokens.txt --tts.wav-out out.wav [options]\n"
        "\n"
        "required:\n"
        "  --tts.magpie-model PATH\n"
        "  --tts.codec-model PATH\n"
        "  --tts.tokens LIST            Comma/space-separated token IDs\n"
        "  --tts.tokens-file PATH       File containing token IDs\n"
        "  --tts.wav-out PATH           Output WAV written after synthesis completes\n"
        "\n"
        "options:\n"
        "  --tts.speaker N              Baked speaker index (default 0)\n"
        "  --tts.steps N                Max decoder frames\n"
        "  --threads N                  CPU threads (default 4)\n"
        "  --tts.codec-threads N        Codec CPU threads (default --threads)\n"
        "  --tts.chunk-frames N         Codec frames per internal callback chunk (default 3)\n"
        "  --tts.lt-backend auto|cpu|cuda\n"
        "                               Local-transformer backend\n"
        "  --tts.lt-fp32               Run the local transformer entirely in FP32\n"
        "  --tts.sampling-backend auto|cpu|cuda\n"
        "  --tts.codec-cpu              Force NanoCodec decoder onto CPU backend\n"
        "  --tts.seed N                 RNG seed\n"
        "  --tts.top-k N                Top-k sampling\n"
        "  --tts.temperature F          Sampling temperature\n"
        "  --tts.cfg-scale F            Classifier-free guidance scale\n"
        "  --tts.no-cfg                 Disable classifier-free guidance\n"
        "  --tts.no-local-transformer   Sample directly from decoder final projection\n"
        "  --tts.no-kv-cache            Recompute decoder prefix each frame\n"
        "  --tts.no-stateful-codec      Disable fast layer-state codec\n"
        "  --benchmark                  Print timing summary\n"
        "  --verbose                    Print detailed runtime logs\n",
        argv0);
}

int
parse_int(const char* value, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be an integer: %s\n", name, value);
        std::exit(1);
    }
    return (int)parsed;
}

float
parse_float(const char* value, const char* name) {
    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)) {
        std::fprintf(stderr, "%s must be a finite float: %s\n", name, value);
        std::exit(1);
    }
    return parsed;
}

void
write_le16(FILE* f, uint16_t value) {
    const unsigned char bytes[2] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
    };
    std::fwrite(bytes, 1, sizeof(bytes), f);
}

void
write_le32(FILE* f, uint32_t value) {
    const unsigned char bytes[4] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
        (unsigned char)((value >> 16) & 0xffu),
        (unsigned char)((value >> 24) & 0xffu),
    };
    std::fwrite(bytes, 1, sizeof(bytes), f);
}

bool
write_wav(const std::string& path, int sample_rate, const std::vector<uint8_t>& pcm) {
    if (sample_rate <= 0) {
        std::fprintf(stderr, "invalid sample rate %d\n", sample_rate);
        return false;
    }
    if (pcm.empty()) {
        std::fprintf(stderr, "no PCM samples were generated\n");
        return false;
    }
    if (pcm.size() % sizeof(int16_t) != 0) {
        std::fprintf(stderr, "PCM byte count is not 16-bit aligned: %zu\n", pcm.size());
        return false;
    }
    if (pcm.size() > (uint64_t)std::numeric_limits<uint32_t>::max() - 36u) {
        std::fprintf(stderr, "PCM output is too large for RIFF WAV: %zu bytes\n", pcm.size());
        return false;
    }

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "failed to open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    const uint32_t data_bytes = (uint32_t)pcm.size();
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = (uint32_t)sample_rate * block_align;

    std::fwrite("RIFF", 1, 4, f);
    write_le32(f, 36u + data_bytes);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f);
    write_le32(f, 16);
    write_le16(f, 1);
    write_le16(f, channels);
    write_le32(f, (uint32_t)sample_rate);
    write_le32(f, byte_rate);
    write_le16(f, block_align);
    write_le16(f, bits_per_sample);
    std::fwrite("data", 1, 4, f);
    write_le32(f, data_bytes);

    const bool wrote_pcm = std::fwrite(pcm.data(), 1, pcm.size(), f) == pcm.size();
    const bool closed = std::fclose(f) == 0;
    if (!wrote_pcm || !closed) {
        std::fprintf(stderr, "failed to write %s\n", path.c_str());
        return false;
    }
    return true;
}

}  // namespace

int
main(int argc, char** argv) {
    tts::magpie_stream_params params;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--tts.magpie-model") {
            params.magpie_model = need_value(arg.c_str());
        } else if (arg == "--tts.codec-model") {
            params.codec_model = need_value(arg.c_str());
        } else if (arg == "--tts.tokens") {
            params.tokens = tts::parse_token_list(need_value(arg.c_str()));
        } else if (arg == "--tts.tokens-file") {
            params.tokens_file = need_value(arg.c_str());
        } else if (arg == "--tts.wav-out" || arg == "--tts.output") {
            params.wav_out = need_value(arg.c_str());
        } else if (arg == "--tts.speaker") {
            params.speaker = parse_int(need_value(arg.c_str()), "--tts.speaker");
        } else if (arg == "--tts.steps") {
            params.steps = parse_int(need_value(arg.c_str()), "--tts.steps");
        } else if (arg == "--threads") {
            params.threads = parse_int(need_value(arg.c_str()), "--threads");
        } else if (arg == "--tts.codec-threads") {
            params.codec_threads = parse_int(need_value(arg.c_str()), "--tts.codec-threads");
        } else if (arg == "--tts.chunk-frames") {
            params.chunk_frames = parse_int(need_value(arg.c_str()), "--tts.chunk-frames");
        } else if (arg == "--tts.lt-backend") {
            if (!tts::parse_backend_preference(need_value(arg.c_str()), params.lt_backend)) {
                std::fprintf(stderr, "--tts.lt-backend must be auto, cpu, or cuda\n");
                return 1;
            }
        } else if (arg == "--tts.sampling-backend") {
            if (!tts::parse_backend_preference(need_value(arg.c_str()), params.sampling_backend)) {
                std::fprintf(stderr, "--tts.sampling-backend must be auto, cpu, or cuda\n");
                return 1;
            }
        } else if (arg == "--tts.lt-fp32") {
            params.lt_fp32 = true;
        } else if (arg == "--tts.codec-cpu") {
            params.codec_cpu = true;
        } else if (arg == "--tts.seed") {
            params.seed = parse_int(need_value(arg.c_str()), "--tts.seed");
        } else if (arg == "--tts.top-k") {
            params.top_k = parse_int(need_value(arg.c_str()), "--tts.top-k");
        } else if (arg == "--tts.temperature") {
            params.temperature = parse_float(need_value(arg.c_str()), "--tts.temperature");
        } else if (arg == "--tts.cfg-scale") {
            params.cfg_scale = parse_float(need_value(arg.c_str()), "--tts.cfg-scale");
        } else if (arg == "--tts.no-cfg") {
            params.use_cfg = false;
        } else if (arg == "--tts.no-local-transformer") {
            params.use_local_transformer = false;
        } else if (arg == "--tts.no-kv-cache") {
            params.use_kv_cache = false;
        } else if (arg == "--tts.no-stateful-codec") {
            params.use_stateful_codec = false;
        } else if (arg == "--benchmark") {
            params.benchmark = true;
        } else if (arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (!params.tokens_file.empty()) {
        params.tokens = tts::parse_token_list(tts::read_file(params.tokens_file));
    }
    params.warmup_tokens = params.tokens;

    if (params.magpie_model.empty() || params.codec_model.empty() || params.tokens.empty() ||
        params.wav_out.empty()) {
        usage(argv[0]);
        return 1;
    }

    tts::MagpieStreamingRuntime runtime;
    if (!runtime.load(
            params.magpie_model, params.codec_model, params.uma_mode, false, params.codec_cpu)) {
        return 1;
    }

    std::vector<uint8_t> pcm;
    tts::stream_run_metrics metrics;
    const bool ok = runtime.synthesize(
        params, params.tokens,
        [&](const std::vector<uint8_t>& bytes) {
            pcm.insert(pcm.end(), bytes.begin(), bytes.end());
            return true;
        },
        metrics);
    if (!ok) {
        return 1;
    }
    if (!write_wav(params.wav_out, runtime.sampleRate(), pcm)) {
        return 1;
    }

    std::fprintf(
        stderr, "wrote %s (%zu bytes PCM, %llu samples, %.3f s, %d Hz)\n", params.wav_out.c_str(),
        pcm.size(), (unsigned long long)(pcm.size() / sizeof(int16_t)),
        runtime.sampleRate() > 0 ? (double)(pcm.size() / sizeof(int16_t)) / runtime.sampleRate()
                                 : 0.0,
        runtime.sampleRate());
    if (params.benchmark) {
        std::fprintf(
            stderr,
            "frames=%d chunks=%d samples=%llu e2e_rtfx=%.2f ttfa_ms=%.2f "
            "decoder_ttft_ms=%.2f codec_ttfa_ms=%.2f\n",
            metrics.generated_frames, metrics.chunks, (unsigned long long)metrics.samples_written,
            metrics.e2e_rtfx, metrics.e2e.first_event_ms, metrics.decoder.first_event_ms,
            metrics.codec.first_event_ms);
    }
    return 0;
}
