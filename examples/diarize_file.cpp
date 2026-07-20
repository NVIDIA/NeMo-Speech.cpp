// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Standalone speaker diarization built on the stable C ABI (nemo_speech_diar_*).
//
// Reads a WAV file (16 kHz mono, 16-bit PCM or 32-bit float), diarizes it
// with the Sortformer model - no ASR involved - and prints "who spoke when"
// as speaker segments (or RTTM lines for scoring tools).
//
// Two modes:
//   default     streaming: audio is pushed in 160 ms slices exactly like a
//               live source, state (speaker cache/FIFO) threads across chunks;
//               works for arbitrarily long audio.
//   --offline   one stateless full-attention pass (best-effort quality trade,
//               capped at ~6.6 minutes by the model's positional table).
//
// Usage: ./diarize_file <sortformer.gguf> <audio.wav>
//            [--gpu N] [--offline] [--rttm NAME] [--preset streaming|offline]
//
// For word-level speaker tags on a transcript, see transcribe_file --diar
// (the ASR ABI's diarization integration).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nemo_speech/diar.h"
#include "wav_reader.h"

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s <sortformer.gguf> <audio.wav> [--gpu N] [--offline] [--rttm NAME]\n"
            "          [--preset streaming|offline]\n"
            "  --gpu N     GPU device index (default 0; -1 = CPU)\n"
            "  --offline   single stateless full-attention pass (<= ~6.6 min audio)\n"
            "  --rttm NAME print RTTM lines (for DER scoring) instead of readable segments\n"
            "  --preset P  streaming geometry preset (default: streaming)\n"
            "  Audio must be 16 kHz mono WAV (PCM16 or float32).\n",
            argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* audio_path = argv[2];
    int gpu = 0;
    bool offline = false;
    const char* rttm_name = nullptr;
    const char* preset = nullptr;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--offline")
            offline = true;
        else if (a == "--rttm" && i + 1 < argc)
            rttm_name = argv[++i];
        else if (a == "--preset" && i + 1 < argc)
            preset = argv[++i];
        else {
            std::fprintf(stderr, "[diarize_file] unknown argument: %s\n", a.c_str());
            return 1;
        }
    }

    std::vector<float> audio;
    std::string err;
    if (!examples::read_wav_mono_16k(audio_path, audio, err)) {
        std::fprintf(stderr, "[diarize_file] %s: %s\n", audio_path, err.c_str());
        return 2;
    }

    nemo_speech_diar_model_config cfg = {};
    cfg.size = sizeof(cfg);
    cfg.model_path = model_path;
    cfg.gpu = gpu;
    cfg.preset = preset;

    nemo_speech_diar_model* model = nullptr;
    if (nemo_speech_diar_create(&cfg, &model) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(
            stderr, "[diarize_file] nemo_speech_diar_create failed: %s\n",
            nemo_speech_asr_last_error());
        return 2;
    }

    // Produce one finished diarization job, streaming or offline.
    nemo_speech_diar_stream* job = nullptr;
    if (offline) {
        if (nemo_speech_diar_offline_f32(
                model, audio.data(), audio.size(), examples::kSampleRate, &job) !=
            NEMO_SPEECH_ASR_OK) {
            std::fprintf(
                stderr, "[diarize_file] offline failed: %s\n", nemo_speech_asr_last_error());
            nemo_speech_diar_destroy(model);
            return 2;
        }
    } else {
        if (nemo_speech_diar_stream_open(model, &job) != NEMO_SPEECH_ASR_OK) {
            std::fprintf(
                stderr, "[diarize_file] stream open failed: %s\n", nemo_speech_asr_last_error());
            nemo_speech_diar_destroy(model);
            return 2;
        }
        // Push in 160 ms slices, the way a live audio source would.
        const size_t push = examples::kSampleRate * 160 / 1000;
        for (size_t off = 0; off < audio.size(); off += push) {
            const size_t n = std::min(push, audio.size() - off);
            if (nemo_speech_diar_stream_push_f32(
                    job, audio.data() + off, n, examples::kSampleRate) != NEMO_SPEECH_ASR_OK) {
                std::fprintf(
                    stderr, "[diarize_file] push failed: %s\n", nemo_speech_asr_last_error());
                nemo_speech_diar_stream_close(job);
                nemo_speech_diar_destroy(model);
                return 2;
            }
        }
        if (nemo_speech_diar_stream_finish(job) != NEMO_SPEECH_ASR_OK) {
            std::fprintf(
                stderr, "[diarize_file] finish failed: %s\n", nemo_speech_asr_last_error());
            nemo_speech_diar_stream_close(job);
            nemo_speech_diar_destroy(model);
            return 2;
        }
    }

    // Two-call segment fetch: count, then fill. NULL config = the library's
    // NeMo-tuned postprocessing defaults.
    size_t count = 0;
    if (nemo_speech_diar_segments(job, nullptr, nullptr, 0, &count) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(
            stderr, "[diarize_file] segment count failed: %s\n", nemo_speech_asr_last_error());
        nemo_speech_diar_stream_close(job);
        nemo_speech_diar_destroy(model);
        return 2;
    }
    std::vector<nemo_speech_diar_segment> segs(count);
    if (count > 0 && nemo_speech_diar_segments(job, nullptr, segs.data(), segs.size(), &count) !=
                         NEMO_SPEECH_ASR_OK) {
        std::fprintf(stderr, "[diarize_file] segments failed: %s\n", nemo_speech_asr_last_error());
        nemo_speech_diar_stream_close(job);
        nemo_speech_diar_destroy(model);
        return 2;
    }

    if (rttm_name) {
        for (const auto& s : segs) {
            std::printf(
                "SPEAKER %s 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>\n", rttm_name, s.start_time,
                s.end_time - s.start_time, s.speaker);
        }
    } else {
        std::fprintf(
            stderr, "[diarize_file] %.1fs audio, %lld frames (%s), %zu segments\n",
            audio.size() / static_cast<double>(examples::kSampleRate),
            static_cast<long long>(nemo_speech_diar_frame_count(job)),
            offline ? "offline" : "streaming", count);
        for (const auto& s : segs) {
            std::printf("  [%8.3fs - %8.3fs] speaker %d\n", s.start_time, s.end_time, s.speaker);
        }
    }

    nemo_speech_diar_stream_close(job);
    nemo_speech_diar_destroy(model);
    return 0;
}
