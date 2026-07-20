// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// ASR -> NMT -> TTS composition using only the installed stable C APIs.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nemo_speech/asr.h"
#include "nemo_speech/nmt.h"
#include "nemo_speech/tts.h"
#include "wav_reader.h"

namespace {

bool
collect_pcm(const uint8_t* data, size_t size, void* user_data) {
    auto* output = static_cast<std::vector<uint8_t>*>(user_data);
    try {
        output->insert(output->end(), data, data + size);
        return true;
    }
    catch (...) {
        return false;
    }
}

void
write_u16(std::ofstream& output, uint16_t value) {
    const uint8_t bytes[2] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8)};
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void
write_u32(std::ofstream& output, uint32_t value) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
        static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)};
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

bool
write_wav(const std::string& path, const std::vector<uint8_t>& pcm, int sample_rate) {
    if (pcm.size() > UINT32_MAX - 36 || sample_rate <= 0)
        return false;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        return false;
    output.write("RIFF", 4);
    write_u32(output, static_cast<uint32_t>(36 + pcm.size()));
    output.write("WAVEfmt ", 8);
    write_u32(output, 16);
    write_u16(output, 1);
    write_u16(output, 1);
    write_u32(output, static_cast<uint32_t>(sample_rate));
    write_u32(output, static_cast<uint32_t>(sample_rate * 2));
    write_u16(output, 2);
    write_u16(output, 16);
    output.write("data", 4);
    write_u32(output, static_cast<uint32_t>(pcm.size()));
    output.write(
        reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(pcm.size()));
    return output.good();
}

void
usage(const char* program) {
    std::fprintf(
        stderr,
        "Usage: %s ASR.gguf NMT.gguf TTS.gguf CODEC.gguf TOKENIZER_DIR AUDIO.wav SRC DST "
        "OUTPUT.wav [--gpu N] [--seed N] [--steps N]\n",
        program);
}

}  // namespace

int
main(int argc, char** argv) {
    int gpu = 0;
    int seed = -1;
    int steps = 0;
    bool steps_set = false;
    std::vector<const char*> args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
            steps_set = true;
        } else
            args.push_back(argv[i]);
    }
    if (args.size() != 9) {
        usage(argv[0]);
        return 2;
    }

    std::vector<float> audio;
    std::string error;
    if (!examples::read_wav_mono_16k(args[5], audio, error)) {
        std::fprintf(stderr, "%s: %s\n", args[5], error.c_str());
        return 2;
    }

    nemo_speech_asr_backend_config asr_backend{sizeof(asr_backend), gpu};
    nemo_speech_asr_model_config asr_model{sizeof(asr_model), args[0], nullptr};
    nemo_speech_asr_recognizer_config asr_config{};
    asr_config.size = sizeof(asr_config);
    asr_config.backend = &asr_backend;
    asr_config.model = &asr_model;
    nemo_speech_asr_recognizer* recognizer = nullptr;
    if (nemo_speech_asr_create(&asr_config, &recognizer) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(stderr, "ASR initialization failed: %s\n", nemo_speech_asr_last_error());
        return 1;
    }
    nemo_speech_asr_recognition_options asr_options = nemo_speech_asr_recognition_options_default();
    asr_options.language_code = args[6];
    asr_options.enable_automatic_punctuation = true;
    nemo_speech_asr_result* asr_result = nullptr;
    if (nemo_speech_asr_recognize_f32(
            recognizer, &asr_options, audio.data(), audio.size(), examples::kSampleRate,
            &asr_result) != NEMO_SPEECH_ASR_OK ||
        !asr_result) {
        std::fprintf(stderr, "ASR failed: %s\n", nemo_speech_asr_last_error());
        nemo_speech_asr_destroy(recognizer);
        return 1;
    }
    const char* transcript_text = nemo_speech_asr_result_alternative_count(asr_result) > 0
                                      ? nemo_speech_asr_result_transcript(asr_result, 0)
                                      : nullptr;
    if (!transcript_text) {
        std::fprintf(stderr, "ASR returned no transcript\n");
        nemo_speech_asr_result_destroy(asr_result);
        nemo_speech_asr_destroy(recognizer);
        return 1;
    }
    const std::string transcript = transcript_text;
    nemo_speech_asr_result_destroy(asr_result);
    nemo_speech_asr_destroy(recognizer);

    nemo_speech_nmt_backend_config nmt_backend{sizeof(nmt_backend), gpu};
    nemo_speech_nmt_model_config nmt_model{sizeof(nmt_model), args[1], 0};
    nemo_speech_nmt_translator_config nmt_config{};
    nmt_config.size = sizeof(nmt_config);
    nmt_config.backend = &nmt_backend;
    nmt_config.model = &nmt_model;
    nemo_speech_nmt_translator* translator = nullptr;
    if (nemo_speech_nmt_create(&nmt_config, &translator) != NEMO_SPEECH_NMT_OK) {
        std::fprintf(stderr, "NMT initialization failed: %s\n", nemo_speech_nmt_last_error());
        return 1;
    }
    const char* texts[] = {transcript.c_str()};
    nemo_speech_nmt_result* nmt_result = nullptr;
    if (nemo_speech_nmt_translate(translator, texts, 1, args[6], args[7], &nmt_result) !=
            NEMO_SPEECH_NMT_OK ||
        !nmt_result) {
        std::fprintf(stderr, "NMT failed: %s\n", nemo_speech_nmt_last_error());
        nemo_speech_nmt_destroy(translator);
        return 1;
    }
    const char* translation_text = nemo_speech_nmt_result_count(nmt_result) > 0
                                       ? nemo_speech_nmt_result_text(nmt_result, 0)
                                       : nullptr;
    if (!translation_text) {
        std::fprintf(stderr, "NMT returned no translation\n");
        nemo_speech_nmt_result_destroy(nmt_result);
        nemo_speech_nmt_destroy(translator);
        return 1;
    }
    const std::string translation = translation_text;
    nemo_speech_nmt_result_destroy(nmt_result);
    nemo_speech_nmt_destroy(translator);

    nemo_speech_tts_model_config tts_model{};
    tts_model.size = sizeof(tts_model);
    tts_model.magpie_model = args[2];
    tts_model.codec_model = args[3];
    tts_model.tokenizer_model_dir = args[4];
    nemo_speech_tts_runtime_config tts_runtime = nemo_speech_tts_runtime_config_default();
    if (gpu < 0) {
        tts_runtime.lt_backend = NEMO_SPEECH_TTS_BACKEND_CPU;
        tts_runtime.sampling_backend = NEMO_SPEECH_TTS_BACKEND_CPU;
        tts_runtime.codec_cpu = true;
    }
    nemo_speech_tts_synthesizer_config tts_config{};
    tts_config.size = sizeof(tts_config);
    tts_config.model = &tts_model;
    tts_config.runtime = &tts_runtime;
    tts_config.default_language_code = args[7];
    nemo_speech_tts_synthesizer* synthesizer = nullptr;
    if (nemo_speech_tts_create(&tts_config, &synthesizer) != NEMO_SPEECH_TTS_OK) {
        std::fprintf(stderr, "TTS initialization failed: %s\n", nemo_speech_tts_last_error());
        return 1;
    }
    nemo_speech_tts_synthesis_options tts_options = nemo_speech_tts_synthesis_options_default();
    tts_options.language_code = args[7];
    tts_options.seed = seed;
    if (steps_set)
        tts_options.steps = steps;
    nemo_speech_tts_synthesis_stats stats = nemo_speech_tts_synthesis_stats_default();
    std::vector<uint8_t> pcm;
    if (nemo_speech_tts_synthesize_text(
            synthesizer, &tts_options, translation.c_str(), collect_pcm, &pcm, &stats) !=
        NEMO_SPEECH_TTS_OK) {
        std::fprintf(stderr, "TTS failed: %s\n", nemo_speech_tts_last_error());
        nemo_speech_tts_destroy(synthesizer);
        return 1;
    }
    nemo_speech_tts_destroy(synthesizer);
    if (!write_wav(args[8], pcm, stats.sample_rate)) {
        std::fprintf(stderr, "Could not write %s\n", args[8]);
        return 1;
    }

    std::printf(
        "Transcript:  %s\nTranslation: %s\nAudio:       %s\n", transcript.c_str(),
        translation.c_str(), args[8]);
    return 0;
}
