// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// nemo-speech: stable C ABI for NMT (text translation).
//
// This is the installed binary contract: opaque handles, POD structs, explicit
// ownership, no exceptions, no STL, no C++ name mangling. The C++ library
// (nmt::Translator) sits behind these handles. Names and behavior mirror Riva
// RivaTranslation.TranslateText.
//
// Compatibility (v1):
//   - Exported symbols: only nemo_speech_nmt_*.
//   - Config structs are append-only and start with `size`; a function tolerates
//     a smaller `size` by defaulting the missing tail fields. Subsystem configs
//     are referenced by optional pointer (NULL = defaults).
//   - Status enum values and result accessors are append-only.
//   - Breaking changes require a new SONAME / v2 symbols.
#ifndef NEMO_SPEECH_NMT_H
#define NEMO_SPEECH_NMT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(NEMO_SPEECH_NMT_BUILD)
#define NEMO_SPEECH_NMT_API __declspec(dllexport)
#else
#define NEMO_SPEECH_NMT_API __declspec(dllimport)
#endif
#else
#define NEMO_SPEECH_NMT_API __attribute__((visibility("default")))
#endif

typedef struct nemo_speech_nmt_translator nemo_speech_nmt_translator;
typedef struct nemo_speech_nmt_result nemo_speech_nmt_result;

typedef enum nemo_speech_nmt_status {
    NEMO_SPEECH_NMT_OK = 0,
    NEMO_SPEECH_NMT_ERROR_INVALID_ARGUMENT = 1,
    NEMO_SPEECH_NMT_ERROR_OUT_OF_MEMORY = 2,
    NEMO_SPEECH_NMT_ERROR_RUNTIME = 3,
} nemo_speech_nmt_status;

// ---- Startup config (subsystem structs, referenced by optional pointer) ----

typedef struct nemo_speech_nmt_backend_config {
    size_t size;
    int32_t gpu;  // -1 = CPU
} nemo_speech_nmt_backend_config;

typedef struct nemo_speech_nmt_model_config {
    size_t size;
    const char* path;
    int32_t n_ctx;  // decode context length in tokens; 0 = default
} nemo_speech_nmt_model_config;

typedef struct nemo_speech_nmt_generation_config {
    size_t size;
    int32_t max_new_tokens;  // cap per input text; 0 = default
} nemo_speech_nmt_generation_config;

typedef struct nemo_speech_nmt_pool_config {
    size_t size;
    int32_t contexts;  // concurrent decode contexts; 0 = default
} nemo_speech_nmt_pool_config;

typedef struct nemo_speech_nmt_translator_config {
    size_t size;
    const nemo_speech_nmt_backend_config* backend;
    const nemo_speech_nmt_model_config* model;
    const nemo_speech_nmt_generation_config* generation;
    const nemo_speech_nmt_pool_config* pool;
} nemo_speech_nmt_translator_config;

// ---- Translator ----

NEMO_SPEECH_NMT_API nemo_speech_nmt_status nemo_speech_nmt_create(
    const nemo_speech_nmt_translator_config* cfg, nemo_speech_nmt_translator** out);

NEMO_SPEECH_NMT_API void nemo_speech_nmt_destroy(nemo_speech_nmt_translator* translator);

// Translate each text from source to target language (proto: TranslateText).
// Languages are two-character codes or the model's pair tag (e.g. "en"/"de" or
// "en-de"). On NEMO_SPEECH_NMT_OK, *out is a result handle with one translation per input
// text, in order; the caller must destroy it. Unsupported language pairs return
// NEMO_SPEECH_NMT_ERROR_INVALID_ARGUMENT.
NEMO_SPEECH_NMT_API nemo_speech_nmt_status nemo_speech_nmt_translate(
    nemo_speech_nmt_translator* translator, const char* const* texts, size_t n_texts,
    const char* source_language, const char* target_language, nemo_speech_nmt_result** out);

// ---- Result accessors (memory owned by the result; valid until destroy) ----

NEMO_SPEECH_NMT_API size_t nemo_speech_nmt_result_count(const nemo_speech_nmt_result* result);
NEMO_SPEECH_NMT_API const char* nemo_speech_nmt_result_text(
    const nemo_speech_nmt_result* result, size_t i);
NEMO_SPEECH_NMT_API const char* nemo_speech_nmt_result_language(
    const nemo_speech_nmt_result* result, size_t i);
NEMO_SPEECH_NMT_API void nemo_speech_nmt_result_destroy(nemo_speech_nmt_result* result);

// ---- Misc ----

// Thread-local last error message for the most recent failed call on this
// thread. Valid until the next nemo_speech_nmt_* call on the same thread.
NEMO_SPEECH_NMT_API const char* nemo_speech_nmt_last_error(void);
NEMO_SPEECH_NMT_API const char* nemo_speech_nmt_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NEMO_SPEECH_NMT_H
