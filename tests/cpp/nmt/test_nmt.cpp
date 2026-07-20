// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// NMT smoke test: load the Riva-Translate GGUF and translate fixed inputs,
// checking each output contains an expected substring.
// Usage: ./test_nmt <translate.gguf> [--gpu N]   (skips if no model arg)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "translator.h"

using namespace nemo_speech::nmt;

namespace {

struct Case {
    const char* source;
    const char* target;
    const char* text;
    const char* expect;  // substring the translation must contain
};

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stdout, "[SKIP] usage: %s <translate.gguf> [--gpu N]\n", argv[0]);
        return 0;
    }
    int gpu = -1;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);

    TranslatorConfig cfg;
    cfg.backend.gpu = gpu;
    cfg.model.path = argv[1];
    cfg.generation.max_new_tokens = 128;
    Translator tr(std::move(cfg));

    const std::vector<Case> cases = {
        {"en", "de",
         "The GRACE mission is a collaboration between NASA and the German Aerospace "
         "Center.",
         "GRACE"},
        {"en", "fr", "Edge deployment keeps latency low and data on the device.", "latence"},
        {"de", "en", "Guten Morgen, wie geht es Ihnen heute?", "morning"},
    };

    int fail = 0;
    for (const auto& c : cases) {
        const auto out = tr.translate({c.text}, c.source, c.target);
        const std::string& got = out.front().text;
        const bool ok = got.find(c.expect) != std::string::npos;
        std::fprintf(
            stdout, "[%s-%s] %s\n  -> %s%s\n", c.source, c.target, c.text, got.c_str(),
            ok ? "" : "  (MISSING expected substring)");
        if (!ok)
            ++fail;
    }
    std::fprintf(stdout, fail ? "FAILED (%d)\n" : "OK\n", fail);
    return fail ? 1 : 0;
}
