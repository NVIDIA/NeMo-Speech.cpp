// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// MagpieTTS -> Nemotron RNNT ASR validation test.
//
// Takes plain text, tokenizes it for MagpieTTS, synthesizes audio, runs the
// generated audio through a Nemotron RNNT ASR model, then reports WER/CER
// between the lowercase input and ASR output text.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "model.h"
#include "recognizer.h"  // RecognizerConfig
#include "runner.h"
#include "runtime.h"
#include "tts/magpietts/magpietts.h"
#include "tts/tokenizer/tokenizer.h"
#include "utils/wer.h"

namespace asr = nemo_speech::asr;
namespace tts = nemo_speech::tts;
namespace test_utils = nemo_speech::tests;

namespace {

struct Args {
    std::string text;
    std::string magpie_model;
    std::string codec_model;
    std::string tokenizer_model_dir;
    std::string asr_model;
    std::string wav_out;
    std::string language_code = "en-US";
    int speaker = 0;
    int steps = -1;
    int seed = -1;
    int top_k = -1;
    int threads = 4;
    int codec_threads = 0;
    int chunk_frames = 3;
    int gpu = 0;
    int chunk_ms = 160;
    int right_ctx = 1;
    double max_wer = 0.0;
    double max_cer = 0.0;
    float temperature = 0.0f;
    float cfg_scale = 0.0f;
    bool override_temperature = false;
    bool override_cfg_scale = false;
    bool use_cfg = true;
    bool use_local_transformer = true;
    bool lt_fp32 = false;
    bool use_kv_cache = true;
    bool use_stateful_codec = true;
    bool codec_cpu = false;
    bool benchmark = false;
    bool verbose = false;
    tts::magpietts_backend_preference lt_backend = tts::MAGPIETTS_BACKEND_AUTO;
    tts::magpietts_backend_preference sampling_backend = tts::MAGPIETTS_BACKEND_AUTO;
};

void
usage(const char* argv0) {
    std::fprintf(
        stderr,
        "usage: %s --tts.text TEXT --tts.magpie-model magpie.gguf --tts.codec-model codec.gguf "
        "--tts.tokenizer-model-dir DIR --asr-model rnnt.gguf [options]\n"
        "\n"
        "required:\n"
        "  --tts.text TEXT              Text to synthesize and validate\n"
        "  --tts.magpie-model PATH      MagpieTTS GGUF token generator\n"
        "  --tts.codec-model PATH       NanoCodec decoder GGUF\n"
        "  --tts.tokenizer-model-dir PATH\n"
        "                               Extracted Magpie .nemo directory\n"
        "  --asr-model PATH             Nemotron RNNT ASR GGUF\n"
        "\n"
        "options:\n"
        "  --tts.wav-out PATH           Optional generated WAV output\n"
        "  --tts.language-code LANG     Magpie tokenizer language (default en-US)\n"
        "  --tts.speaker N              Baked speaker index (default 0)\n"
        "  --tts.steps N                Max decoder frames\n"
        "  --tts.seed N                 RNG seed\n"
        "  --tts.top-k N                Top-k sampling\n"
        "  --tts.temperature F          Sampling temperature\n"
        "  --tts.cfg-scale F            Classifier-free guidance scale\n"
        "  --tts.lt-backend auto|cpu|cuda\n"
        "                               Local-transformer backend\n"
        "  --tts.lt-fp32               Run the local transformer entirely in FP32\n"
        "  --tts.sampling-backend auto|cpu|cuda\n"
        "  --threads N                  CPU threads (default 4)\n"
        "  --tts.codec-threads N        Codec CPU threads (default --threads)\n"
        "  --tts.chunk-frames N         Codec frames per internal audio chunk (default 3)\n"
        "  --tts.codec-cpu              Force NanoCodec decoder onto CPU backend\n"
        "  --gpu N                      ASR GPU index (default 0; -1 for CPU)\n"
        "  --chunk-ms N                 ASR streaming chunk size (default 160)\n"
        "  --right-ctx N                ASR RNNT right context (default 1)\n"
        "  --max-wer F                  Maximum lowercase WER allowed (default 0)\n"
        "  --max-cer F                  Maximum lowercase CER allowed (default 0)\n"
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

double
parse_double(const char* value, const char* name) {
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)) {
        std::fprintf(stderr, "%s must be a finite number: %s\n", name, value);
        std::exit(1);
    }
    return parsed;
}

float
parse_float(const char* value, const char* name) {
    return (float)parse_double(value, name);
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
    if (sample_rate <= 0 || pcm.empty() || pcm.size() % sizeof(int16_t) != 0) {
        std::fprintf(
            stderr, "invalid WAV payload: sample_rate=%d bytes=%zu\n", sample_rate, pcm.size());
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

std::vector<float>
pcm_s16le_to_float(const std::vector<uint8_t>& pcm) {
    std::vector<float> out;
    out.reserve(pcm.size() / sizeof(int16_t));
    for (size_t i = 0; i + 1 < pcm.size(); i += 2) {
        const uint16_t lo = pcm[i];
        const uint16_t hi = pcm[i + 1];
        const int16_t sample = (int16_t)((hi << 8) | lo);
        out.push_back((float)sample / 32768.0f);
    }
    return out;
}

std::vector<float>
resample_linear(const std::vector<float>& input, int src_rate, int dst_rate) {
    if (input.empty() || src_rate <= 0 || dst_rate <= 0) {
        return {};
    }
    if (src_rate == dst_rate) {
        return input;
    }

    const double scale = (double)dst_rate / (double)src_rate;
    const size_t out_size = std::max<size_t>(1, (size_t)std::llround((double)input.size() * scale));
    std::vector<float> out(out_size);
    for (size_t i = 0; i < out.size(); ++i) {
        const double src_pos = (double)i * (double)src_rate / (double)dst_rate;
        const size_t j = (size_t)src_pos;
        const double frac = src_pos - (double)j;
        const float a = input[std::min(j, input.size() - 1)];
        const float b = input[std::min(j + 1, input.size() - 1)];
        out[i] = (float)((1.0 - frac) * (double)a + frac * (double)b);
    }
    return out;
}

Args
parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--tts.text") {
            args.text = need_value(arg.c_str());
        } else if (arg == "--tts.magpie-model") {
            args.magpie_model = need_value(arg.c_str());
        } else if (arg == "--tts.codec-model") {
            args.codec_model = need_value(arg.c_str());
        } else if (arg == "--tts.tokenizer-model-dir") {
            args.tokenizer_model_dir = need_value(arg.c_str());
        } else if (arg == "--asr-model") {
            args.asr_model = need_value(arg.c_str());
        } else if (arg == "--tts.wav-out" || arg == "--tts.output") {
            args.wav_out = need_value(arg.c_str());
        } else if (arg == "--tts.language-code") {
            args.language_code = need_value(arg.c_str());
        } else if (arg == "--tts.speaker") {
            args.speaker = parse_int(need_value(arg.c_str()), "--tts.speaker");
        } else if (arg == "--tts.steps") {
            args.steps = parse_int(need_value(arg.c_str()), "--tts.steps");
        } else if (arg == "--tts.seed") {
            args.seed = parse_int(need_value(arg.c_str()), "--tts.seed");
        } else if (arg == "--tts.top-k") {
            args.top_k = parse_int(need_value(arg.c_str()), "--tts.top-k");
        } else if (arg == "--threads") {
            args.threads = parse_int(need_value(arg.c_str()), "--threads");
        } else if (arg == "--tts.codec-threads") {
            args.codec_threads = parse_int(need_value(arg.c_str()), "--tts.codec-threads");
        } else if (arg == "--tts.chunk-frames") {
            args.chunk_frames = parse_int(need_value(arg.c_str()), "--tts.chunk-frames");
        } else if (arg == "--gpu") {
            args.gpu = parse_int(need_value(arg.c_str()), "--gpu");
        } else if (arg == "--chunk-ms") {
            args.chunk_ms = parse_int(need_value(arg.c_str()), "--chunk-ms");
        } else if (arg == "--right-ctx") {
            args.right_ctx = parse_int(need_value(arg.c_str()), "--right-ctx");
        } else if (arg == "--max-wer") {
            args.max_wer = parse_double(need_value(arg.c_str()), "--max-wer");
        } else if (arg == "--max-cer") {
            args.max_cer = parse_double(need_value(arg.c_str()), "--max-cer");
        } else if (arg == "--tts.temperature") {
            args.temperature = parse_float(need_value(arg.c_str()), "--tts.temperature");
            args.override_temperature = true;
        } else if (arg == "--tts.cfg-scale") {
            args.cfg_scale = parse_float(need_value(arg.c_str()), "--tts.cfg-scale");
            args.override_cfg_scale = true;
        } else if (arg == "--tts.lt-backend") {
            if (!tts::parse_backend_preference(need_value(arg.c_str()), args.lt_backend)) {
                std::fprintf(stderr, "--tts.lt-backend must be auto, cpu, or cuda\n");
                std::exit(1);
            }
        } else if (arg == "--tts.sampling-backend") {
            if (!tts::parse_backend_preference(need_value(arg.c_str()), args.sampling_backend)) {
                std::fprintf(stderr, "--tts.sampling-backend must be auto, cpu, or cuda\n");
                std::exit(1);
            }
        } else if (arg == "--tts.lt-fp32") {
            args.lt_fp32 = true;
        } else if (arg == "--tts.codec-cpu") {
            args.codec_cpu = true;
        } else if (arg == "--tts.no-cfg") {
            args.use_cfg = false;
        } else if (arg == "--tts.no-local-transformer") {
            args.use_local_transformer = false;
        } else if (arg == "--tts.no-kv-cache") {
            args.use_kv_cache = false;
        } else if (arg == "--tts.no-stateful-codec") {
            args.use_stateful_codec = false;
        } else if (arg == "--benchmark") {
            args.benchmark = true;
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            std::exit(1);
        }
    }
    return args;
}

bool
validate_required(const Args& args, const char* argv0) {
    if (!args.text.empty() && !args.magpie_model.empty() && !args.codec_model.empty() &&
        !args.tokenizer_model_dir.empty() && !args.asr_model.empty()) {
        return true;
    }
    usage(argv0);
    return false;
}

std::string
run_rnnt_asr(asr::RnntModel& model, const std::vector<float>& audio, int chunk_ms, int right_ctx) {
    asr::RecognizerConfig cfg;
    cfg.streaming.rnnt_right_context = right_ctx;
    asr::CacheStreamRunner runner(&model, cfg);

    const size_t chunk_samples =
        std::max<size_t>(1, (size_t)chunk_ms * (size_t)model.sample_rate() / 1000u);
    std::string last_partial;
    for (size_t off = 0; off < audio.size(); off += chunk_samples) {
        const size_t n = std::min(chunk_samples, audio.size() - off);
        runner.feed_audio(audio.data() + off, n);
        const auto update = runner.step();
        if (!update.new_token_ids.empty() && update.transcript_so_far != last_partial) {
            last_partial = update.transcript_so_far;
        }
    }
    const auto final_update = runner.finalize();
    return final_update.transcript_so_far;
}

}  // namespace

int
main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    if (!validate_required(args, argv[0])) {
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();

    tts::MagpieNativeTokenizer tokenizer(args.tokenizer_model_dir);
    const auto tokenized = tokenizer.tokenize(args.text, args.language_code);

    tts::magpie_stream_params params;
    params.magpie_model = args.magpie_model;
    params.codec_model = args.codec_model;
    params.tokens = tokenized.tokens;
    params.warmup_tokens = tokenized.tokens;
    params.speaker = args.speaker;
    params.threads = args.threads;
    params.codec_threads = args.codec_threads;
    params.seed = args.seed;
    params.steps = args.steps;
    params.top_k = args.top_k;
    params.chunk_frames = args.chunk_frames;
    params.temperature = args.override_temperature ? args.temperature : NAN;
    params.cfg_scale = args.override_cfg_scale ? args.cfg_scale : NAN;
    params.use_cfg = args.use_cfg;
    params.use_local_transformer = args.use_local_transformer;
    params.lt_fp32 = args.lt_fp32;
    params.use_kv_cache = args.use_kv_cache;
    params.use_stateful_codec = args.use_stateful_codec;
    params.codec_cpu = args.codec_cpu;
    params.benchmark = args.benchmark;
    params.verbose = args.verbose;
    params.lt_backend = args.lt_backend;
    params.sampling_backend = args.sampling_backend;

    tts::MagpieStreamingRuntime tts_runtime;
    if (!tts_runtime.load(
            params.magpie_model, params.codec_model, params.uma_mode, false, params.codec_cpu,
            params.verbose)) {
        return 1;
    }

    std::vector<uint8_t> pcm;
    tts::stream_run_metrics tts_metrics;
    const bool tts_ok = tts_runtime.synthesize(
        params, params.tokens,
        [&](const std::vector<uint8_t>& bytes) {
            pcm.insert(pcm.end(), bytes.begin(), bytes.end());
            return true;
        },
        tts_metrics);
    if (!tts_ok || pcm.empty()) {
        std::fprintf(stderr, "MagpieTTS synthesis failed or produced no audio\n");
        return 1;
    }

    if (!args.wav_out.empty() && !write_wav(args.wav_out, tts_runtime.sampleRate(), pcm)) {
        return 1;
    }

    ggml_runtime::Params bm_params;
    bm_params.use_gpu = args.gpu >= 0;
    bm_params.gpu_device_idx = std::max(args.gpu, 0);
    bm_params.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(bm_params);
    auto asr_model = asr::AsrModel::load(bm, args.asr_model);
    auto* asr_rnnt = dynamic_cast<asr::RnntModel*>(asr_model.get());
    if (!asr_rnnt) {
        std::fprintf(stderr, "expected an RNNT ASR model\n");
        return 1;
    }

    std::vector<float> audio = pcm_s16le_to_float(pcm);
    audio = resample_linear(audio, tts_runtime.sampleRate(), asr_model->sample_rate());
    if (audio.empty()) {
        std::fprintf(stderr, "generated audio became empty after conversion/resampling\n");
        return 1;
    }

    const std::string transcript = run_rnnt_asr(*asr_rnnt, audio, args.chunk_ms, args.right_ctx);
    const auto wer = test_utils::word_error_rate(args.text, transcript);
    const auto cer = test_utils::char_error_rate(args.text, transcript);

    std::fprintf(stdout, "input: %s\n", args.text.c_str());
    std::fprintf(stdout, "asr: %s\n", transcript.c_str());
    std::fprintf(
        stdout, "wer=%.6f edits=%zu ref_words=%zu hyp_words=%zu\n", wer.rate, wer.edits,
        wer.reference_units, wer.hypothesis_units);
    std::fprintf(
        stdout, "cer=%.6f edits=%zu ref_chars=%zu hyp_chars=%zu\n", cer.rate, cer.edits,
        cer.reference_units, cer.hypothesis_units);

    if (args.benchmark) {
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::fprintf(
            stderr,
            "text_tokens=%zu samples=%llu tts_frames=%d tts_chunks=%d "
            "tts_rtfx=%.2f elapsed_ms=%.2f\n",
            tokenized.tokens.size(), (unsigned long long)tts_metrics.samples_written,
            tts_metrics.generated_frames, tts_metrics.chunks, tts_metrics.e2e_rtfx, elapsed_ms);
    }
    if (!args.wav_out.empty()) {
        std::fprintf(
            stderr, "wrote %s (%zu bytes PCM, %d Hz)\n", args.wav_out.c_str(), pcm.size(),
            tts_runtime.sampleRate());
    }

    if (wer.rate > args.max_wer || cer.rate > args.max_cer) {
        std::fprintf(
            stderr, "text validation failed: wer=%.6f max_wer=%.6f cer=%.6f max_cer=%.6f\n",
            wer.rate, args.max_wer, cer.rate, args.max_cer);
        return 2;
    }
    return 0;
}
