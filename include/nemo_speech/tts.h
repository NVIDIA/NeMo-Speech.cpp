// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// nemo-speech: stable C ABI for TTS.
//
// This is the installed binary contract: opaque handles, POD structs, explicit
// ownership, no exceptions, no STL, no C++ name mangling. The C++ MagpieTTS
// runtime sits behind these handles.
//
// Compatibility (v1):
//   - Stable exported symbols: nemo_speech_tts_*; C++ exports are internal and unstable.
//   - Config/stats structs are append-only and start with `size`; functions
//     tolerate smaller input structs by defaulting missing tail fields.
//   - Status enum values are append-only.
//   - Breaking changes require a new SONAME / v2 symbols.
#ifndef NEMO_SPEECH_TTS_H
#define NEMO_SPEECH_TTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(NEMO_SPEECH_TTS_BUILD)
#define NEMO_SPEECH_TTS_API __declspec(dllexport)
#else
#define NEMO_SPEECH_TTS_API __declspec(dllimport)
#endif
#else
#define NEMO_SPEECH_TTS_API __attribute__((visibility("default")))
#endif

typedef struct nemo_speech_tts_synthesizer nemo_speech_tts_synthesizer;

typedef enum nemo_speech_tts_status {
    NEMO_SPEECH_TTS_OK = 0,
    NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT = 1,
    NEMO_SPEECH_TTS_ERROR_OUT_OF_MEMORY = 2,
    NEMO_SPEECH_TTS_ERROR_RUNTIME = 3,
    NEMO_SPEECH_TTS_ERROR_CANCELLED = 4,
} nemo_speech_tts_status;

typedef enum nemo_speech_tts_backend_preference {
    NEMO_SPEECH_TTS_BACKEND_AUTO = 0,
    NEMO_SPEECH_TTS_BACKEND_CPU = 1,
    NEMO_SPEECH_TTS_BACKEND_CUDA = 2,
} nemo_speech_tts_backend_preference;

typedef enum nemo_speech_tts_uma_mode {
    NEMO_SPEECH_TTS_UMA_AUTO = 0,
    NEMO_SPEECH_TTS_UMA_OFF = 1,
    NEMO_SPEECH_TTS_UMA_ON = 2,
} nemo_speech_tts_uma_mode;

typedef enum nemo_speech_tts_longform_mode {
    NEMO_SPEECH_TTS_LONGFORM_AUTO = 0,
    NEMO_SPEECH_TTS_LONGFORM_OFF = 1,
    NEMO_SPEECH_TTS_LONGFORM_ON = 2,
} nemo_speech_tts_longform_mode;

// ---- Startup config (subsystem structs, referenced by optional pointer) ----

typedef struct nemo_speech_tts_model_config {
    size_t size;
    const char* magpie_model;
    const char* codec_model;
    // Optional for nemo_speech_tts_synthesize_tokens; required for nemo_speech_tts_synthesize_text.
    const char* tokenizer_model_dir;
    // Optional written-to-spoken TN grammar root. Empty keeps text unchanged.
    const char* text_normalizer_model_dir;
} nemo_speech_tts_model_config;

typedef struct nemo_speech_tts_runtime_config {
    size_t size;
    int32_t speaker;
    int32_t threads;
    int32_t codec_threads;
    int32_t seed;
    int32_t steps;
    int32_t top_k;
    int32_t chunk_frames;
    int32_t codec_queue_depth;
    int32_t codec_history_frames;
    int32_t codec_future_frames;
    int32_t window_ms;
    float temperature;
    bool override_temperature;
    float cfg_scale;
    bool override_cfg_scale;
    bool use_cfg;
    bool use_local_transformer;
    bool use_kv_cache;
    bool use_stateful_codec;
    bool codec_cpu;
    bool flush_partial_chunk;
    bool verbose;
    nemo_speech_tts_backend_preference lt_backend;
    nemo_speech_tts_backend_preference sampling_backend;
    nemo_speech_tts_uma_mode uma_mode;
    nemo_speech_tts_longform_mode longform_mode;
    bool lt_fp32;
} nemo_speech_tts_runtime_config;

typedef struct nemo_speech_tts_synthesizer_config {
    size_t size;
    const nemo_speech_tts_model_config* model;
    const nemo_speech_tts_runtime_config* runtime;
    const char* default_language_code;  // NULL/"" = en-US
    const char* default_voice_name;     // speaker name, model.name, or numeric index
} nemo_speech_tts_synthesizer_config;

// ---- Per-request options ----

typedef struct nemo_speech_tts_synthesis_options {
    size_t size;
    const char* request_id;     // echoed by transports; not used for state
    const char* language_code;  // text tokenization; NULL/"" = en-US
    int32_t speaker;            // < 0 = synthesizer default
    int32_t seed;               // < 0 = synthesizer default
    int32_t steps;              // <= 0 = synthesizer default
    int32_t top_k;              // <= 0 = synthesizer default
    float temperature;
    bool override_temperature;
    float cfg_scale;
    bool override_cfg_scale;
    const char* voice_name;      // ignored when speaker >= 0
    int32_t output_sample_rate;  // 0 = model rate; otherwise 8000..model rate
} nemo_speech_tts_synthesis_options;

// Raw PCM callback. `pcm` is little-endian signed 16-bit mono at the requested
// output sample rate (or nemo_speech_tts_sample_rate() when no override was supplied).
// Return false to cancel synthesis.
typedef bool (*nemo_speech_tts_pcm_callback)(const uint8_t* pcm, size_t n_bytes, void* user_data);

typedef struct nemo_speech_tts_synthesis_stats {
    size_t size;
    int32_t sample_rate;
    int32_t generated_frames;
    int32_t chunks;
    int32_t e2e_chunks;
    uint64_t samples_written;
    double tokenizer_ms;
    double encoder_ms;
    double audio_s;
    double elapsed_s;
    double rtf;
    double rtfx;
    double ttfa_ms;
    double icl_avg_ms;
    double icl_min_ms;
    double icl_max_ms;
    double decoder_audio_s;
    double decoder_elapsed_s;
    double decoder_rtfx;
    double decoder_ttft_ms;
    double decoder_itl_avg_ms;
    double decoder_itl_min_ms;
    double decoder_itl_max_ms;
    double decoder_itl_p95_ms;
    double decoder_itl_p99_ms;
    double codec_audio_s;
    double codec_elapsed_s;
    double codec_rtfx;
    double codec_ttfa_ms;
    double codec_icl_avg_ms;
    double codec_icl_min_ms;
    double codec_icl_max_ms;
    double codec_icl_p95_ms;
    double codec_icl_p99_ms;
    double e2e_ttfa_ms;
    double e2e_icl_avg_ms;
    double e2e_icl_min_ms;
    double e2e_icl_max_ms;
    double e2e_icl_p95_ms;
    double e2e_icl_p99_ms;
    double e2e_rtfx;
} nemo_speech_tts_synthesis_stats;

// Struct defaults with `size` set to the current sizeof.
NEMO_SPEECH_TTS_API nemo_speech_tts_runtime_config nemo_speech_tts_runtime_config_default(void);
NEMO_SPEECH_TTS_API nemo_speech_tts_synthesis_options
nemo_speech_tts_synthesis_options_default(void);
NEMO_SPEECH_TTS_API nemo_speech_tts_synthesis_stats nemo_speech_tts_synthesis_stats_default(void);

// ---- Synthesizer ----

NEMO_SPEECH_TTS_API nemo_speech_tts_status nemo_speech_tts_create(
    const nemo_speech_tts_synthesizer_config* cfg, nemo_speech_tts_synthesizer** out);

NEMO_SPEECH_TTS_API void nemo_speech_tts_destroy(nemo_speech_tts_synthesizer* synthesizer);

NEMO_SPEECH_TTS_API int32_t
nemo_speech_tts_sample_rate(const nemo_speech_tts_synthesizer* synthesizer);
NEMO_SPEECH_TTS_API int32_t
nemo_speech_tts_speaker_count(const nemo_speech_tts_synthesizer* synthesizer);
NEMO_SPEECH_TTS_API const char* nemo_speech_tts_speaker_name(
    const nemo_speech_tts_synthesizer* synthesizer, size_t i);

// Synthesize raw text through the native Magpie tokenizer. The synthesizer must
// have been created with model.tokenizer_model_dir.
NEMO_SPEECH_TTS_API nemo_speech_tts_status nemo_speech_tts_synthesize_text(
    nemo_speech_tts_synthesizer* synthesizer, const nemo_speech_tts_synthesis_options* options,
    const char* text, nemo_speech_tts_pcm_callback callback, void* user_data,
    nemo_speech_tts_synthesis_stats* stats_out);

// Synthesize pre-tokenized Magpie text tokens as a single chunk.
NEMO_SPEECH_TTS_API nemo_speech_tts_status nemo_speech_tts_synthesize_tokens(
    nemo_speech_tts_synthesizer* synthesizer, const nemo_speech_tts_synthesis_options* options,
    const int32_t* tokens, size_t token_count, nemo_speech_tts_pcm_callback callback,
    void* user_data, nemo_speech_tts_synthesis_stats* stats_out);

// ---- Misc ----

// Thread-local last error message for the most recent failed call on this
// thread. Valid until the next nemo_speech_tts_* call on the same thread.
NEMO_SPEECH_TTS_API const char* nemo_speech_tts_last_error(void);
NEMO_SPEECH_TTS_API const char* nemo_speech_tts_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NEMO_SPEECH_TTS_H
