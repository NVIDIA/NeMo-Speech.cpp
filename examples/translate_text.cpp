// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Text translation built on the stable C ABI (nemo_speech_nmt_*).
//
// Loads a Riva-Translate GGUF, translates one or more input texts from a source
// to a target language, and prints one translation per line. A self-contained
// companion to transcribe_file: same C ABI style, no model internals exposed.
//
// Usage: ./translate_text <translate.gguf> <src> <tgt> <text> [<text> ...] [--gpu N]
//   src/tgt are two-character codes or a pair tag (e.g. "en"/"de" or "en-de").
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nemo_speech/nmt.h"

int
main(int argc, char** argv) {
    int gpu = 0;
    std::vector<const char*> pos;  // model, src, tgt, text...
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else
            pos.push_back(argv[i]);
    }
    if (pos.size() < 4) {
        std::fprintf(
            stderr,
            "Usage: %s <translate.gguf> <src> <tgt> <text> [<text> ...] [--gpu N]\n"
            "  src/tgt  two-character codes or a pair tag (e.g. en de, or en-de)\n"
            "  --gpu N  GPU device index (default 0; -1 = CPU)\n",
            argv[0]);
        return 1;
    }
    const char* model_path = pos[0];
    const char* source_language = pos[1];
    const char* target_language = pos[2];
    const char* const* texts = pos.data() + 3;
    const size_t n_texts = pos.size() - 3;

    nemo_speech_nmt_backend_config backend = {};
    backend.size = sizeof(backend);
    backend.gpu = gpu;

    nemo_speech_nmt_model_config model = {};
    model.size = sizeof(model);
    model.path = model_path;

    nemo_speech_nmt_translator_config cfg = {};
    cfg.size = sizeof(cfg);
    cfg.backend = &backend;
    cfg.model = &model;

    nemo_speech_nmt_translator* translator = nullptr;
    if (nemo_speech_nmt_create(&cfg, &translator) != NEMO_SPEECH_NMT_OK) {
        std::fprintf(
            stderr, "[translate_text] nemo_speech_nmt_create failed: %s\n",
            nemo_speech_nmt_last_error());
        return 2;
    }

    nemo_speech_nmt_result* result = nullptr;
    nemo_speech_nmt_status st = nemo_speech_nmt_translate(
        translator, texts, n_texts, source_language, target_language, &result);
    if (st != NEMO_SPEECH_NMT_OK || !result) {
        std::fprintf(
            stderr, "[translate_text] translate failed: %s\n", nemo_speech_nmt_last_error());
        nemo_speech_nmt_destroy(translator);
        return 2;
    }

    const size_t count = nemo_speech_nmt_result_count(result);
    for (size_t i = 0; i < count; i++) {
        const char* text = nemo_speech_nmt_result_text(result, i);
        std::fprintf(stdout, "%s\n", text ? text : "");
    }

    nemo_speech_nmt_result_destroy(result);
    nemo_speech_nmt_destroy(translator);
    return 0;
}
