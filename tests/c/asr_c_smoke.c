// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/* C ABI smoke test - compiled as C (not C++) to prove include/nemo_speech/asr.h
 * is valid C and the nemo_speech_asr_c library links and runs end to end.
 *
 *   asr_c_smoke                 -> link/symbol/header check only (prints version)
 *   asr_c_smoke MODEL.gguf WAV  -> offline + streaming recognition via the C ABI
 *
 * WAV must be 16 kHz mono 16-bit PCM (the test reads it as such).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nemo_speech/asr.h"

/* Read a 16-bit mono PCM WAV (skip the 44-byte header) into float32 [-1,1]. */
static float*
read_wav_f32(const char* path, size_t* n_out) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 44) {
        fclose(f);
        return NULL;
    }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    size_t off = 44; /* assume canonical header; smoke test only */
    size_t n = ((size_t)sz - off) / 2;
    float* out = (float*)malloc(n * sizeof(float));
    const int16_t* s = (const int16_t*)(buf + off);
    for (size_t i = 0; i < n; i++) out[i] = (float)s[i] / 32768.0f;
    free(buf);
    *n_out = n;
    return out;
}

int
main(int argc, char** argv) {
    printf("nemo_speech_asr_version: %s\n", nemo_speech_asr_version());
    if (argc < 3) {
        printf("[link/header OK] pass MODEL WAV to run recognition\n");
        return 0;
    }
    const char* model = argv[1];
    const char* wav = argv[2];

    nemo_speech_asr_backend_config backend;
    memset(&backend, 0, sizeof(backend));
    backend.size = sizeof(backend);
    backend.gpu = (argc > 3) ? atoi(argv[3]) : 0;

    nemo_speech_asr_model_config mc;
    memset(&mc, 0, sizeof(mc));
    mc.size = sizeof(mc);
    mc.path = model;

    nemo_speech_asr_batching_config batching;
    memset(&batching, 0, sizeof(batching));
    batching.size = sizeof(batching);
    batching.enable = true;
    batching.max_batch_size = 4;
    batching.max_queue_depth = 16;
    batching.state_arena_slots = 4;

    nemo_speech_asr_recognizer_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.size = sizeof(cfg);
    cfg.backend = &backend;
    cfg.model = &mc;
    cfg.batching = &batching;

    nemo_speech_asr_recognizer* rec = NULL;
    if (nemo_speech_asr_create(&cfg, &rec) != NEMO_SPEECH_ASR_OK) {
        fprintf(stderr, "create failed: %s\n", nemo_speech_asr_last_error());
        return 1;
    }

    size_t n = 0;
    float* audio = read_wav_f32(wav, &n);
    if (!audio) {
        fprintf(stderr, "cannot read wav %s\n", wav);
        nemo_speech_asr_destroy(rec);
        return 1;
    }

    nemo_speech_asr_recognition_options opt = nemo_speech_asr_recognition_options_default();
    opt.enable_word_time_offsets = true;

    /* ---- offline ---- */
    nemo_speech_asr_result* r = NULL;
    if (nemo_speech_asr_recognize_f32(rec, &opt, audio, n, 16000, &r) != NEMO_SPEECH_ASR_OK) {
        fprintf(stderr, "recognize failed: %s\n", nemo_speech_asr_last_error());
        free(audio);
        nemo_speech_asr_destroy(rec);
        return 1;
    }
    printf("offline: %s\n", nemo_speech_asr_result_transcript(r, 0));
    printf("offline words: %zu\n", nemo_speech_asr_result_word_count(r, 0));
    nemo_speech_asr_result_destroy(r);

    /* ---- streaming (160 ms chunks) ---- */
    nemo_speech_asr_stream* st = NULL;
    if (nemo_speech_asr_streaming_recognize(rec, &opt, &st) != NEMO_SPEECH_ASR_OK) {
        fprintf(stderr, "stream_open failed: %s\n", nemo_speech_asr_last_error());
        free(audio);
        nemo_speech_asr_destroy(rec);
        return 1;
    }
    if (nemo_speech_asr_stream_push_f32(st, NULL, 1, 16000) !=
        NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "stream push accepted a null non-empty audio buffer\n");
        nemo_speech_asr_stream_close(st);
        free(audio);
        nemo_speech_asr_destroy(rec);
        return 1;
    }
    if (nemo_speech_asr_stream_push_f32(st, NULL, 0, 16000) != NEMO_SPEECH_ASR_OK) {
        fprintf(stderr, "stream push rejected an empty audio buffer\n");
        nemo_speech_asr_stream_close(st);
        free(audio);
        nemo_speech_asr_destroy(rec);
        return 1;
    }
    const size_t chunk = 16000 * 160 / 1000; /* 160 ms */
    char streamed[8192];
    streamed[0] = '\0';
    for (size_t pos = 0; pos < n; pos += chunk) {
        size_t cn = (pos + chunk <= n) ? chunk : (n - pos);
        nemo_speech_asr_stream_push_f32(st, audio + pos, cn, 16000);
        nemo_speech_asr_result* sr = NULL;
        while (nemo_speech_asr_stream_next(st, &sr) == NEMO_SPEECH_ASR_OK && sr) {
            if (nemo_speech_asr_result_is_final(sr)) {
                strncat(
                    streamed, nemo_speech_asr_result_transcript(sr, 0),
                    sizeof(streamed) - strlen(streamed) - 1);
            }
            nemo_speech_asr_result_destroy(sr);
            sr = NULL;
        }
    }
    nemo_speech_asr_stream_finish(st);
    nemo_speech_asr_result* sr = NULL;
    while (nemo_speech_asr_stream_next(st, &sr) == NEMO_SPEECH_ASR_OK && sr) {
        if (nemo_speech_asr_result_is_final(sr)) {
            strncat(
                streamed, nemo_speech_asr_result_transcript(sr, 0),
                sizeof(streamed) - strlen(streamed) - 1);
        }
        nemo_speech_asr_result_destroy(sr);
        sr = NULL;
    }
    nemo_speech_asr_stream_close(st);
    printf("streaming: %s\n", streamed);

    free(audio);
    nemo_speech_asr_destroy(rec);
    printf("[C ABI smoke OK]\n");
    return 0;
}
