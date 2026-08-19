// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// nemo-speech: stable C ABI for ASR.
//
// This is the installed binary contract: opaque handles, POD structs, explicit
// ownership, no exceptions, no STL, no C++ name mangling. The C++ library
// (Recognizer/RecognitionStream) sits behind these handles. Names and behavior
// mirror Riva ASR (Recognize / StreamingRecognize / word time offsets / finals
// / force_eou).
//
// Compatibility (v1):
//   - Exported symbols: only nemo_speech_asr_*.
//   - Config structs are append-only and start with `size`; a function tolerates
//     a smaller `size` by defaulting the missing tail fields. Subsystem configs
//     are referenced by optional pointer (NULL = defaults).
//   - Status enum values and result accessors are append-only.
//   - Breaking changes require a new SONAME / v2 symbols.
#ifndef NEMO_SPEECH_ASR_H
#define NEMO_SPEECH_ASR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(NEMO_SPEECH_ASR_BUILD)
#define NEMO_SPEECH_ASR_API __declspec(dllexport)
#else
#define NEMO_SPEECH_ASR_API __declspec(dllimport)
#endif
#else
#define NEMO_SPEECH_ASR_API __attribute__((visibility("default")))
#endif

typedef struct nemo_speech_asr_recognizer nemo_speech_asr_recognizer;
typedef struct nemo_speech_asr_stream nemo_speech_asr_stream;
typedef struct nemo_speech_asr_result nemo_speech_asr_result;

typedef enum nemo_speech_asr_status {
    NEMO_SPEECH_ASR_OK = 0,
    NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT = 1,
    NEMO_SPEECH_ASR_ERROR_OUT_OF_MEMORY = 2,
    NEMO_SPEECH_ASR_ERROR_RUNTIME = 3,
    NEMO_SPEECH_ASR_ERROR_CANCELLED = 4,
} nemo_speech_asr_status;

typedef enum nemo_speech_asr_decoder_kind {
    NEMO_SPEECH_ASR_DECODER_GREEDY = 0,
    NEMO_SPEECH_ASR_DECODER_FLASHLIGHT = 1,
} nemo_speech_asr_decoder_kind;

// ---- Startup config (subsystem structs, referenced by optional pointer) ----

typedef struct nemo_speech_asr_backend_config {
    size_t size;
    int32_t gpu;  // -1 = CPU
} nemo_speech_asr_backend_config;

typedef struct nemo_speech_asr_model_config {
    size_t size;
    const char* path;
    const char* name;  // optional; NULL/"" = derived from model
} nemo_speech_asr_model_config;

// Streaming execution policy (latency/throughput), not model identity.
typedef struct nemo_speech_asr_streaming_config {
    size_t size;
    float chunk_size;            // s (CTC buffered window)
    float ctc_left_padding;      // s
    float ctc_right_padding;     // s
    int32_t rnnt_right_context;  // encoder frames; -1 = model default
} nemo_speech_asr_streaming_config;

typedef struct nemo_speech_asr_decoder_config {
    size_t size;
    nemo_speech_asr_decoder_kind kind;
    const char* flashlight_lm;
    const char* flashlight_lexicon;
    const char* flashlight_tokenizer;
    int32_t beam_size;
    int32_t beam_size_token;
    double beam_threshold;
    double lm_weight;
    double word_insertion_score;
    // Max magnitude of a per-word speech-context boost. The boost is added to a
    // word's score on every emission, so an unbounded value makes the word
    // repeat spuriously; this caps it. 0 keeps the default (10.0).
    double max_boost;
} nemo_speech_asr_decoder_config;

typedef struct nemo_speech_asr_vad_config {
    size_t size;
    const char* model_path;  // empty/NULL = no VAD
    bool enable_masking;
    float onset;
    float offset;
} nemo_speech_asr_vad_config;

typedef struct nemo_speech_asr_endpointing_config {
    size_t size;
    bool enable;
    bool vad_based;
    int32_t stop_history_eou_ms;
} nemo_speech_asr_endpointing_config;

typedef struct nemo_speech_asr_postproc_config {
    size_t size;
    const char* profanity_list_path;
    // Direct grammar dir, or a root with per-language children (en/, es/, ...).
    const char* itn_model_dir;
    const char* pnc_model_path;
} nemo_speech_asr_postproc_config;

// Speaker diarization sidecar (Sortformer). Geometry uses streaming defaults
// unless overridden; left_context_frames uses a negative sentinel because zero is valid.
typedef struct nemo_speech_asr_diar_config {
    size_t size;
    const char* model_path;  // empty/NULL = diarization not available
    int32_t chunk_frames;
    int32_t right_context_frames;
    int32_t left_context_frames;  // 0 is a valid value; < 0 = default
    int32_t fifo_frames;
    int32_t spkcache_frames;
    int32_t update_period_frames;
} nemo_speech_asr_diar_config;

// Transparent dynamic batching across concurrent recognition calls. A single
// recognizer owns one model; callers submit independent utterances from
// multiple threads and compatible neural work is combined into GPU batches.
// This does not create additional recognizers or model copies.
typedef struct nemo_speech_asr_batching_config {
    size_t size;
    bool enable;
    int32_t max_batch_size;
    int32_t max_queue_delay_us;
    int32_t max_queue_depth;
    int32_t ingress_cohort_delay_us;
    int32_t state_arena_slots;
} nemo_speech_asr_batching_config;

typedef struct nemo_speech_asr_recognizer_config {
    size_t size;
    const nemo_speech_asr_backend_config* backend;
    const nemo_speech_asr_model_config* model;
    const nemo_speech_asr_streaming_config* streaming;
    const nemo_speech_asr_decoder_config* decoder;
    const nemo_speech_asr_vad_config* vad;
    const nemo_speech_asr_endpointing_config* endpointing;
    const nemo_speech_asr_postproc_config* postproc;
    const nemo_speech_asr_diar_config* diar;
    const nemo_speech_asr_batching_config* batching;
} nemo_speech_asr_recognizer_config;

// ---- Per-request options ----

typedef struct nemo_speech_asr_speech_context {
    size_t size;
    const char* const* phrases;
    size_t phrase_count;
    float boost;
} nemo_speech_asr_speech_context;

typedef struct nemo_speech_asr_recognition_options {
    size_t size;
    const char* request_id;     // echoed by transports; not used for state
    const char* language_code;  // prompt selection; NULL/"" = auto/default
    bool interim_results;
    bool enable_word_time_offsets;
    bool enable_automatic_punctuation;
    bool verbatim_transcripts;
    bool profanity_filter;
    int32_t stop_history_eou_ms;  // <= 0 = server default
    const nemo_speech_asr_speech_context* speech_contexts;
    size_t speech_context_count;
    // Number of ranked hypotheses (N-best) to return; <= 1 = 1-best (default).
    // >1 only takes effect with a beam decoder that implements N-best - greedy
    // heads always return 1. Use nemo_speech_asr_result_alternative_count() to read back
    // how many a result actually carries.
    int32_t max_alternatives;
    // Tag each word of the top alternative with a 1-based speaker id
    // (nemo_speech_asr_result_word_speaker_tag). Requires the recognizer to have been
    // created with a diar model (nemo_speech_asr_diar_config.model_path), else the
    // request fails with INVALID_ARGUMENT. max_speaker_count is accepted for
    // compatibility but is ignored; Sortformer v2 supports up to four speakers.
    bool enable_speaker_diarization;
    int32_t max_speaker_count;
} nemo_speech_asr_recognition_options;

// Zero-initialized structs with `size` set to the current sizeof, holding the
// library's defaults.
NEMO_SPEECH_ASR_API nemo_speech_asr_recognition_options
nemo_speech_asr_recognition_options_default(void);

// ---- Recognizer ----

NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_asr_create(
    const nemo_speech_asr_recognizer_config* cfg, nemo_speech_asr_recognizer** out);

NEMO_SPEECH_ASR_API void nemo_speech_asr_destroy(nemo_speech_asr_recognizer* recognizer);

// Offline (proto: Recognize): decode a whole utterance. `samples` is mono
// float32. Rates from 8-96 kHz are resampled to the model rate; 0 means the
// samples are already at the model rate.
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_asr_recognize_f32(
    nemo_speech_asr_recognizer* recognizer, const nemo_speech_asr_recognition_options* options,
    const float* samples, size_t n_samples, int32_t sample_rate, nemo_speech_asr_result** out);

// ---- Streaming (proto: StreamingRecognize) ----

// Start a streaming recognition; *out is a stream handle driven by the
// nemo_speech_asr_stream_* calls below and released with nemo_speech_asr_stream_close.
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_asr_streaming_recognize(
    nemo_speech_asr_recognizer* recognizer, const nemo_speech_asr_recognition_options* options,
    nemo_speech_asr_stream** out);

// Buffer mono float32 audio (no decode). Rates from 8-96 kHz are resampled to
// the model rate; 0 means model rate. The rate cannot change within a stream.
// next() drives decoding.
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_asr_stream_push_f32(
    nemo_speech_asr_stream* stream, const float* samples, size_t n_samples, int32_t sample_rate);

// Latch an immediate end-of-utterance, honored on the next nemo_speech_asr_stream_next.
NEMO_SPEECH_ASR_API nemo_speech_asr_status
nemo_speech_asr_stream_force_endpoint(nemo_speech_asr_stream* stream);

// No more audio: flush the tail. The end-of-stream final is then returned by
// nemo_speech_asr_stream_next.
NEMO_SPEECH_ASR_API nemo_speech_asr_status
nemo_speech_asr_stream_finish(nemo_speech_asr_stream* stream);

// Pull one result. On NEMO_SPEECH_ASR_OK, *out is a result handle the caller must
// destroy, or NULL when nothing is currently available (need more audio).
NEMO_SPEECH_ASR_API nemo_speech_asr_status
nemo_speech_asr_stream_next(nemo_speech_asr_stream* stream, nemo_speech_asr_result** out);

NEMO_SPEECH_ASR_API void nemo_speech_asr_stream_close(nemo_speech_asr_stream* stream);

// ---- Result accessors (memory owned by the result; valid until destroy) ----
//
// A result carries one or more ranked alternatives (N-best), best first. The
// per-alternative accessors take an `alt` index in [0, alternative_count);
// `alt = 0` is the top hypothesis. Today alternative_count is always 1 (greedy
// heads emit one hypothesis; beam N-best is wired but not yet implemented), so
// passing 0 is the common case. is_final / audio_processed / channel_tag are
// per-result.

NEMO_SPEECH_ASR_API bool nemo_speech_asr_result_is_final(const nemo_speech_asr_result* result);
NEMO_SPEECH_ASR_API float nemo_speech_asr_result_audio_processed(
    const nemo_speech_asr_result* result);
// 1-based input channel this result belongs to (proto channel_tag). Mono input
// is always 1; multi-channel recognition is not implemented, so this is 1 today.
NEMO_SPEECH_ASR_API int32_t
nemo_speech_asr_result_channel_tag(const nemo_speech_asr_result* result);

NEMO_SPEECH_ASR_API size_t
nemo_speech_asr_result_alternative_count(const nemo_speech_asr_result* result);
NEMO_SPEECH_ASR_API const char* nemo_speech_asr_result_transcript(
    const nemo_speech_asr_result* result, size_t alt);
NEMO_SPEECH_ASR_API float nemo_speech_asr_result_confidence(
    const nemo_speech_asr_result* result, size_t alt);

NEMO_SPEECH_ASR_API size_t
nemo_speech_asr_result_word_count(const nemo_speech_asr_result* result, size_t alt);
NEMO_SPEECH_ASR_API const char* nemo_speech_asr_result_word_text(
    const nemo_speech_asr_result* result, size_t alt, size_t i);
NEMO_SPEECH_ASR_API int32_t
nemo_speech_asr_result_word_start_time(const nemo_speech_asr_result* result, size_t alt, size_t i);
NEMO_SPEECH_ASR_API int32_t
nemo_speech_asr_result_word_end_time(const nemo_speech_asr_result* result, size_t alt, size_t i);
NEMO_SPEECH_ASR_API float nemo_speech_asr_result_word_confidence(
    const nemo_speech_asr_result* result, size_t alt, size_t i);
// 1-based speaker id when diarization was requested; 0 = untagged.
NEMO_SPEECH_ASR_API int32_t
nemo_speech_asr_result_word_speaker_tag(const nemo_speech_asr_result* result, size_t alt, size_t i);

NEMO_SPEECH_ASR_API size_t
nemo_speech_asr_result_language_count(const nemo_speech_asr_result* result, size_t alt);
NEMO_SPEECH_ASR_API const char* nemo_speech_asr_result_language_code(
    const nemo_speech_asr_result* result, size_t alt, size_t i);

NEMO_SPEECH_ASR_API void nemo_speech_asr_result_destroy(nemo_speech_asr_result* result);

// ---- Misc ----

// Thread-local last error message for the most recent failed call on this
// thread. Valid until the next nemo_speech_asr_* call on the same thread.
NEMO_SPEECH_ASR_API const char* nemo_speech_asr_last_error(void);
NEMO_SPEECH_ASR_API const char* nemo_speech_asr_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NEMO_SPEECH_ASR_H
