// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Live-microphone streaming ASR built on the stable C ABI (nemo_speech_asr_*).
//
// Opens the default (or --device N) capture device at 16 kHz mono float32 via
// PortAudio, pushes samples into an nemo_speech_asr_stream, and prints interim
// transcripts as they update plus a final transcript on Ctrl-C. The C ABI
// routes by the model's head internally (RNNT cache-aware vs CTC buffered), so
// this one binary handles both without referencing any internal C++ class.
//
// Usage: ./transcribe_live <model.gguf> [--gpu N] [--right-ctx R] [--device N]
//
// Audio capture is PortAudio (system package: apt install portaudio19-dev /
// brew install portaudio); it wraps CoreAudio on macOS, WASAPI on Windows, and
// ALSA/JACK/PulseAudio on Linux.
#include <portaudio.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nemo_speech/asr.h"

namespace {

// All current ASR models (Parakeet CTC, Nemotron-Speech RNNT) run at 16 kHz
// mono. The C ABI validates this on push and errors otherwise, so a future
// non-16k model fails loudly rather than silently mis-decoding.
constexpr int kSampleRate = 16000;

// Samples captured on PortAudio's audio thread, drained by the main loop.
struct SampleQueue {
    std::mutex mu;
    std::vector<float> buf;
};
SampleQueue g_queue;
std::atomic<bool> g_running{true};

// PortAudio capture callback (audio thread): append mono float32 input to the
// shared queue. Keep it allocation-light and lock-brief; the main thread does
// the ASR work.
int
capture_cb(
    const void* input, void* /*output*/, unsigned long frame_count,
    const PaStreamCallbackTimeInfo* /*ti*/, PaStreamCallbackFlags /*flags*/, void* /*user*/) {
    if (input && frame_count) {
        const float* in = static_cast<const float*>(input);
        std::lock_guard<std::mutex> lock(g_queue.mu);
        g_queue.buf.insert(g_queue.buf.end(), in, in + frame_count);
    }
    return paContinue;
}

void
on_sigint(int /*sig*/) {
    g_running.store(false);
}

// Print the device list (capture-capable devices only) — shown when --device
// is out of range so the user can pick a valid index.
void
print_input_devices() {
    const int n = Pa_GetDeviceCount();
    std::fprintf(stderr, "[transcribe_live] available capture devices:\n");
    for (int i = 0; i < n; i++) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info && info->maxInputChannels > 0) {
            std::fprintf(stderr, "  %d: %s\n", i, info->name);
        }
    }
}

// Format a final's words as speaker turns ("speaker N: ..."), riva-style
// presentation of WordInfo.speaker_tag. Finals-only: interims carry no words.
std::string
speaker_turns(const nemo_speech_asr_result* res) {
    std::string out;
    const size_t nwords = nemo_speech_asr_result_word_count(res, 0);
    int32_t cur = 0;
    for (size_t i = 0; i < nwords; i++) {
        const int32_t tag = nemo_speech_asr_result_word_speaker_tag(res, 0, i);
        if (tag != cur) {
            out += (i ? "\n" : "") + std::string("speaker ") + std::to_string(tag) + ": ";
            cur = tag;
        } else if (i) {
            out += " ";
        }
        out += nemo_speech_asr_result_word_text(res, 0, i);
    }
    return out;
}

// Drain every result currently available from the stream, printing interims
// and finals. Returns the last transcript seen (final or interim).
void
drain_results(nemo_speech_asr_stream* stream, std::string& last_printed, bool diarize) {
    nemo_speech_asr_result* res = nullptr;
    while (nemo_speech_asr_stream_next(stream, &res) == NEMO_SPEECH_ASR_OK && res) {
        const char* text = nemo_speech_asr_result_transcript(res, 0);
        const bool is_final = nemo_speech_asr_result_is_final(res);
        const float t = nemo_speech_asr_result_audio_processed(res);
        if (text && (is_final || std::strcmp(text, last_printed.c_str()) != 0)) {
            if (is_final && diarize) {
                std::fprintf(stderr, "[final   @ %.2fs]\n%s\n", t, speaker_turns(res).c_str());
            } else {
                std::fprintf(
                    stderr, "[%s @ %.2fs] %s\n", is_final ? "final  " : "partial", t, text);
            }
            last_printed = text ? text : "";
        }
        nemo_speech_asr_result_destroy(res);
        res = nullptr;
    }
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(
            stderr,
            "Usage: %s <model.gguf> [--gpu N] [--right-ctx R] [--device N] [--diar D.gguf]\n"
            "  --gpu N        GPU device index (default 0; -1 = CPU)\n"
            "  --right-ctx R  cache-aware right context (RNNT; default 1, ~160ms latency)\n"
            "  --device N     PortAudio capture device index (default: system default)\n"
            "  --diar D.gguf  Sortformer diarizer GGUF: finals print as speaker turns\n",
            argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    int gpu = 0;
    int right_ctx = 1;
    int device_idx = -1;  // -1 = system default
    const char* diar_model = nullptr;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--right-ctx" && i + 1 < argc)
            right_ctx = std::atoi(argv[++i]);
        else if (a == "--device" && i + 1 < argc)
            device_idx = std::atoi(argv[++i]);
        else if (a == "--diar" && i + 1 < argc)
            diar_model = argv[++i];
    }

    // ---- Build the recognizer via the C ABI. Each config struct sets `size`
    // (append-only ABI) and is wired into the top-level recognizer config. ----
    nemo_speech_asr_backend_config backend = {};
    backend.size = sizeof(backend);
    backend.gpu = gpu;

    nemo_speech_asr_model_config model = {};
    model.size = sizeof(model);
    model.path = model_path;

    // Explicit defaults match the library's StreamingConfig (Riva low-latency
    // preset): CTC fields are ignored by RNNT models and vice-versa, so the
    // same struct is correct for either head. rnnt_right_context is the only
    // knob this example exposes.
    nemo_speech_asr_streaming_config streaming = {};
    streaming.size = sizeof(streaming);
    streaming.chunk_size = 0.16f;
    streaming.ctc_left_padding = 1.92f;
    streaming.ctc_right_padding = 1.92f;
    streaming.rnnt_right_context = right_ctx;

    nemo_speech_asr_diar_config diar = {};
    diar.size = sizeof(diar);
    diar.model_path = diar_model;  // NULL = diarization not available

    nemo_speech_asr_recognizer_config cfg = {};
    cfg.size = sizeof(cfg);
    cfg.backend = &backend;
    cfg.model = &model;
    cfg.streaming = &streaming;
    if (diar_model)
        cfg.diar = &diar;

    nemo_speech_asr_recognizer* recognizer = nullptr;
    if (nemo_speech_asr_create(&cfg, &recognizer) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(
            stderr, "[transcribe_live] nemo_speech_asr_create failed: %s\n",
            nemo_speech_asr_last_error());
        return 2;
    }
    std::fprintf(stderr, "[transcribe_live] model loaded (%s)\n", nemo_speech_asr_version());

    // Interim results so partial transcripts stream as you speak.
    nemo_speech_asr_recognition_options opts = nemo_speech_asr_recognition_options_default();
    opts.interim_results = true;
    opts.enable_speaker_diarization = diar_model != nullptr;

    nemo_speech_asr_stream* stream = nullptr;
    if (nemo_speech_asr_streaming_recognize(recognizer, &opts, &stream) != NEMO_SPEECH_ASR_OK) {
        std::fprintf(
            stderr, "[transcribe_live] nemo_speech_asr_streaming_recognize failed: %s\n",
            nemo_speech_asr_last_error());
        nemo_speech_asr_destroy(recognizer);
        return 2;
    }

    // ---- PortAudio capture at the model rate, mono float32. ----
    if (Pa_Initialize() != paNoError) {
        std::fprintf(stderr, "[transcribe_live] Pa_Initialize failed\n");
        nemo_speech_asr_stream_close(stream);
        nemo_speech_asr_destroy(recognizer);
        return 3;
    }

    PaDeviceIndex dev = (device_idx >= 0) ? device_idx : Pa_GetDefaultInputDevice();
    if (dev == paNoDevice || dev >= Pa_GetDeviceCount()) {
        std::fprintf(stderr, "[transcribe_live] no usable capture device\n");
        if (device_idx >= 0)
            print_input_devices();
        Pa_Terminate();
        nemo_speech_asr_stream_close(stream);
        nemo_speech_asr_destroy(recognizer);
        return 3;
    }
    const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(dev);
    if (!dev_info || dev_info->maxInputChannels < 1) {
        std::fprintf(stderr, "[transcribe_live] device %d has no input channels\n", (int)dev);
        print_input_devices();
        Pa_Terminate();
        nemo_speech_asr_stream_close(stream);
        nemo_speech_asr_destroy(recognizer);
        return 3;
    }

    PaStreamParameters in_params = {};
    in_params.device = dev;
    in_params.channelCount = 1;
    in_params.sampleFormat = paFloat32;
    in_params.suggestedLatency = dev_info->defaultLowInputLatency;
    in_params.hostApiSpecificStreamInfo = nullptr;

    PaStream* pa_stream = nullptr;
    PaError err = Pa_OpenStream(
        &pa_stream, &in_params, nullptr, kSampleRate, paFramesPerBufferUnspecified, paClipOff,
        capture_cb, nullptr);
    if (err != paNoError) {
        std::fprintf(stderr, "[transcribe_live] Pa_OpenStream failed: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        nemo_speech_asr_stream_close(stream);
        nemo_speech_asr_destroy(recognizer);
        return 3;
    }
    if (Pa_StartStream(pa_stream) != paNoError) {
        std::fprintf(stderr, "[transcribe_live] Pa_StartStream failed\n");
        Pa_CloseStream(pa_stream);
        Pa_Terminate();
        nemo_speech_asr_stream_close(stream);
        nemo_speech_asr_destroy(recognizer);
        return 3;
    }
    std::fprintf(
        stderr, "[transcribe_live] listening on \"%s\" — Ctrl-C to stop\n", dev_info->name);

    std::signal(SIGINT, on_sigint);

    // ---- Stream loop: drain mic -> push -> pull results. ----
    std::string last_printed;
    std::vector<float> scratch;
    while (g_running.load()) {
        scratch.clear();
        {
            std::lock_guard<std::mutex> lock(g_queue.mu);
            scratch.swap(g_queue.buf);
        }
        if (!scratch.empty()) {
            if (nemo_speech_asr_stream_push_f32(
                    stream, scratch.data(), scratch.size(), kSampleRate) != NEMO_SPEECH_ASR_OK) {
                std::fprintf(
                    stderr, "[transcribe_live] push failed: %s\n", nemo_speech_asr_last_error());
                break;
            }
        }
        drain_results(stream, last_printed, diar_model != nullptr);
        if (scratch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    // ---- Teardown: stop capture, flush the tail, print the final. ----
    Pa_StopStream(pa_stream);
    Pa_CloseStream(pa_stream);
    Pa_Terminate();

    nemo_speech_asr_stream_finish(stream);
    drain_results(stream, last_printed, diar_model != nullptr);

    std::fprintf(stderr, "\n[transcribe_live] done\n");
    std::fprintf(stdout, "%s\n", last_printed.c_str());

    nemo_speech_asr_stream_close(stream);
    nemo_speech_asr_destroy(recognizer);
    return 0;
}
