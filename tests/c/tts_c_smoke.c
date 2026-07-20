// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/* C ABI smoke test - compiled as C (not C++) to prove include/nemo_speech/tts.h
 * is valid C and the unified nemo_speech_tts library links and runs end to end.
 *
 *   tts_c_smoke
 *       link/symbol/header check only (prints version)
 *   tts_c_smoke --tokens MAGPIE.gguf CODEC.gguf TOKENS
 *       synthesize comma-separated Magpie text token IDs
 *   tts_c_smoke --text MAGPIE.gguf CODEC.gguf TOKENIZER_DIR TEXT [LANG]
 *       tokenize raw text and synthesize it
 */
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nemo_speech/tts.h"

struct pcm_sink {
    size_t bytes;
    size_t chunks;
};

static bool
count_pcm(const uint8_t* pcm, size_t n_bytes, void* user_data) {
    struct pcm_sink* sink = (struct pcm_sink*)user_data;
    if (!pcm && n_bytes)
        return false;
    sink->bytes += n_bytes;
    sink->chunks += 1;
    return true;
}

static void
usage(const char* prog) {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "  %s\n", prog);
    fprintf(stderr, "  %s --tokens MAGPIE.gguf CODEC.gguf TOKENS\n", prog);
    fprintf(stderr, "  %s --text MAGPIE.gguf CODEC.gguf TOKENIZER_DIR TEXT [LANG]\n", prog);
}

static int32_t*
parse_tokens(const char* text, size_t* count_out) {
    size_t cap = 16;
    size_t count = 0;
    int32_t* values = (int32_t*)malloc(cap * sizeof(int32_t));
    const char* p = text;
    if (!values)
        return NULL;
    while (*p) {
        char* end = NULL;
        long v;
        while (*p && (isspace((unsigned char)*p) || *p == ',')) ++p;
        if (!*p)
            break;
        errno = 0;
        v = strtol(p, &end, 10);
        if (errno || end == p || v < INT32_MIN || v > INT32_MAX) {
            free(values);
            return NULL;
        }
        if (count == cap) {
            int32_t* grown;
            cap *= 2;
            grown = (int32_t*)realloc(values, cap * sizeof(int32_t));
            if (!grown) {
                free(values);
                return NULL;
            }
            values = grown;
        }
        values[count++] = (int32_t)v;
        p = end;
        while (*p && isspace((unsigned char)*p)) ++p;
        if (*p && *p != ',') {
            free(values);
            return NULL;
        }
    }
    *count_out = count;
    return values;
}

static nemo_speech_tts_synthesizer*
create_synth(const char* magpie, const char* codec, const char* tokenizer_dir) {
    nemo_speech_tts_model_config model;
    nemo_speech_tts_runtime_config runtime;
    nemo_speech_tts_synthesizer_config cfg;
    nemo_speech_tts_synthesizer* synth = NULL;

    memset(&model, 0, sizeof(model));
    model.size = sizeof(model);
    model.magpie_model = magpie;
    model.codec_model = codec;
    model.tokenizer_model_dir = tokenizer_dir;

    runtime = nemo_speech_tts_runtime_config_default();

    memset(&cfg, 0, sizeof(cfg));
    cfg.size = sizeof(cfg);
    cfg.model = &model;
    cfg.runtime = &runtime;

    if (nemo_speech_tts_create(&cfg, &synth) != NEMO_SPEECH_TTS_OK) {
        fprintf(stderr, "create failed: %s\n", nemo_speech_tts_last_error());
        return NULL;
    }
    return synth;
}

int
main(int argc, char** argv) {
    printf("nemo_speech_tts_version: %s\n", nemo_speech_tts_version());
    if (argc == 1) {
        printf("[link/header OK] pass --text or --tokens to run synthesis\n");
        return 0;
    }

    if (argc < 5) {
        usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--tokens") == 0) {
        const char* magpie = argv[2];
        const char* codec = argv[3];
        const char* token_text = argv[4];
        size_t token_count = 0;
        int32_t* tokens = parse_tokens(token_text, &token_count);
        nemo_speech_tts_synthesizer* synth;
        nemo_speech_tts_synthesis_options opt;
        nemo_speech_tts_synthesis_stats stats;
        struct pcm_sink sink;
        nemo_speech_tts_status st;

        if (!tokens || token_count == 0) {
            fprintf(stderr, "invalid token list\n");
            free(tokens);
            return 1;
        }

        synth = create_synth(magpie, codec, NULL);
        if (!synth) {
            free(tokens);
            return 1;
        }

        opt = nemo_speech_tts_synthesis_options_default();
        stats = nemo_speech_tts_synthesis_stats_default();
        sink.bytes = 0;
        sink.chunks = 0;
        st = nemo_speech_tts_synthesize_tokens(
            synth, &opt, tokens, token_count, count_pcm, &sink, &stats);
        free(tokens);
        if (st != NEMO_SPEECH_TTS_OK) {
            fprintf(stderr, "synthesize_tokens failed: %s\n", nemo_speech_tts_last_error());
            nemo_speech_tts_destroy(synth);
            return 1;
        }
        printf(
            "tokens: %zu pcm_bytes=%zu callback_chunks=%zu sample_rate=%d frames=%d "
            "samples=%llu e2e_rtfx=%.2f\n",
            token_count, sink.bytes, sink.chunks, stats.sample_rate, stats.generated_frames,
            (unsigned long long)stats.samples_written, stats.e2e_rtfx);
        nemo_speech_tts_destroy(synth);
        printf("[C ABI smoke OK]\n");
        return sink.bytes > 0 ? 0 : 1;
    }

    if (strcmp(argv[1], "--text") == 0) {
        const char* magpie = argv[2];
        const char* codec = argv[3];
        const char* tokenizer_dir = argv[4];
        const char* text = (argc > 5) ? argv[5] : "";
        const char* lang = (argc > 6) ? argv[6] : NULL;
        nemo_speech_tts_synthesizer* synth;
        nemo_speech_tts_synthesis_options opt;
        nemo_speech_tts_synthesis_stats stats;
        struct pcm_sink sink;
        nemo_speech_tts_status st;

        synth = create_synth(magpie, codec, tokenizer_dir);
        if (!synth)
            return 1;

        opt = nemo_speech_tts_synthesis_options_default();
        opt.language_code = lang;
        stats = nemo_speech_tts_synthesis_stats_default();
        sink.bytes = 0;
        sink.chunks = 0;
        st = nemo_speech_tts_synthesize_text(synth, &opt, text, count_pcm, &sink, &stats);
        if (st != NEMO_SPEECH_TTS_OK) {
            fprintf(stderr, "synthesize_text failed: %s\n", nemo_speech_tts_last_error());
            nemo_speech_tts_destroy(synth);
            return 1;
        }
        printf(
            "text: pcm_bytes=%zu callback_chunks=%zu sample_rate=%d frames=%d samples=%llu "
            "tokenizer_ms=%.2f e2e_rtfx=%.2f\n",
            sink.bytes, sink.chunks, stats.sample_rate, stats.generated_frames,
            (unsigned long long)stats.samples_written, stats.tokenizer_ms, stats.e2e_rtfx);
        nemo_speech_tts_destroy(synth);
        printf("[C ABI smoke OK]\n");
        return sink.bytes > 0 ? 0 : 1;
    }

    usage(argv[0]);
    return 1;
}
