// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Offline file transcription built on the stable C ABI (nemo_speech_asr_*).
//
// Reads a WAV file (16 kHz mono, 16-bit PCM or 32-bit float), runs a single
// offline recognition via nemo_speech_asr_recognize_f32, and prints the transcript.
// With --word-times it also prints per-word start/end offsets. With
// --diar <sortformer.gguf> it enables speaker diarization and prints the
// transcript as speaker turns ("speaker N: ..."), riva-style word tags.
//
// Usage: ./transcribe_file <model.gguf> <audio.wav> [--gpu N]
//                         [--language CODE] [--word-times]
//                         [--diar <sortformer.gguf>]
//
// A self-contained companion to transcribe_live: same C ABI, no microphone /
// PortAudio dependency. The WAV reader below is intentionally minimal (the
// canonical 16 kHz mono case the models expect) so the example stays focused
// on the recognition API rather than audio decoding.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nemo_speech/asr.h"
#include "wav_reader.h"

using examples::kSampleRate;

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s <model.gguf> <audio.wav> [--gpu N] [--language CODE] [--word-times]\n"
            "                       [--diar D.gguf]\n"
            "  --gpu N       GPU device index (default 0; -1 = CPU)\n"
            "  --language C  select a language prompt (for example en-US or es-US)\n"
            "  --word-times  print per-word start/end offsets (ms)\n"
            "  --diar D.gguf Sortformer diarizer GGUF: tag words with speakers and\n"
            "                print the transcript as speaker turns\n"
            "  Audio must be 16 kHz mono WAV (PCM16 or float32).\n",
            argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* audio_path = argv[2];
    int gpu = 0;
    bool word_times = false;
    std::string language;
    const char* diar_model = nullptr;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--language" && i + 1 < argc)
            language = argv[++i];
        else if (a == "--word-times")
            word_times = true;
        else if (a == "--diar" && i + 1 < argc)
            diar_model = argv[++i];
    }

    std::vector<float> audio;
    std::string err;
    if (!examples::read_wav_mono_16k(audio_path, audio, err)) {
        std::fprintf(stderr, "[transcribe_file] %s: %s\n", audio_path, err.c_str());
        return 2;
    }

    nemo_speech_asr_backend_config backend = {};
    backend.size = sizeof(backend);
    backend.gpu = gpu;

    nemo_speech_asr_model_config model = {};
    model.size = sizeof(model);
    model.path = model_path;

    nemo_speech_asr_diar_config diar = {};
    diar.size = sizeof(diar);
    diar.model_path = diar_model;  // NULL = diarization not available

    nemo_speech_asr_recognizer_config cfg = {};
    cfg.size = sizeof(cfg);
    cfg.backend = &backend;
    cfg.model = &model;
    if (diar_model)
        cfg.diar = &diar;

    nemo_speech_asr_recognizer* recognizer = nullptr;
    if (nemo_speech_asr_create(&cfg, &recognizer) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(
            stderr, "[transcribe_file] nemo_speech_asr_create failed: %s\n",
            nemo_speech_asr_last_error());
        return 2;
    }

    nemo_speech_asr_recognition_options opts = nemo_speech_asr_recognition_options_default();
    opts.enable_word_time_offsets = word_times;
    opts.enable_automatic_punctuation = true;
    opts.language_code = language.empty() ? nullptr : language.c_str();
    opts.enable_speaker_diarization = diar_model != nullptr;

    nemo_speech_asr_result* result = nullptr;
    nemo_speech_asr_status st = nemo_speech_asr_recognize_f32(
        recognizer, &opts, audio.data(), audio.size(), kSampleRate, &result);
    if (st != NEMO_SPEECH_ASR_OK || !result) {
        std::fprintf(
            stderr, "[transcribe_file] recognize failed: %s\n", nemo_speech_asr_last_error());
        nemo_speech_asr_destroy(recognizer);
        return 2;
    }

    if (diar_model) {
        // Group consecutive same-speaker words into turns (riva-style
        // presentation of WordInfo.speaker_tag).
        const size_t nwords = nemo_speech_asr_result_word_count(result, 0);
        int32_t cur = 0;
        for (size_t i = 0; i < nwords; i++) {
            const int32_t tag = nemo_speech_asr_result_word_speaker_tag(result, 0, i);
            if (tag != cur) {
                std::fprintf(stdout, "%sspeaker %d: ", i ? "\n" : "", tag);
                cur = tag;
            } else if (i) {
                std::fprintf(stdout, " ");
            }
            std::fprintf(stdout, "%s", nemo_speech_asr_result_word_text(result, 0, i));
        }
        std::fprintf(stdout, "\n");
    } else {
        const char* transcript = nemo_speech_asr_result_transcript(result, 0);
        std::fprintf(stdout, "%s\n", transcript ? transcript : "");
    }

    if (word_times) {
        const size_t nwords = nemo_speech_asr_result_word_count(result, 0);
        for (size_t i = 0; i < nwords; i++) {
            const int32_t tag = nemo_speech_asr_result_word_speaker_tag(result, 0, i);
            char spk[16] = "";
            if (tag > 0)
                std::snprintf(spk, sizeof(spk), " spk%d", tag);
            std::fprintf(
                stderr, "  [%5d–%5d ms]%s %s\n",
                nemo_speech_asr_result_word_start_time(result, 0, i),
                nemo_speech_asr_result_word_end_time(result, 0, i), spk,
                nemo_speech_asr_result_word_text(result, 0, i));
        }
    }

    nemo_speech_asr_result_destroy(result);
    nemo_speech_asr_destroy(recognizer);
    return 0;
}
