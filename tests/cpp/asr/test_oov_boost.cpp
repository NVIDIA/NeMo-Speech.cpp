// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Deterministic OOV word-boosting test: synthesizes CTC log-probs spelling an
// OOV word's pieces and checks the FlashlightDecoder emits it only after the
// boost augments the trie. The decoder does its own SentencePiece encoding, so
// this does not link sentencepiece directly (it is provided by the library).
//
// Needs the real artifacts; skips (exit 0) if any is unset or built without
// flashlight:
//   LM_PATH LEXICON_PATH TOKENIZER_PATH VOCAB_PATH (token pieces, one per line).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight_decoder.h"
#include "greedy_ctc_decoder.h"  // CtcConfig
#include "types.h"               // AsrRequestOptions, Result

using namespace nemo_speech::asr;

static int g_fail = 0;
static void
check(bool ok, const char* what) {
    std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        g_fail++;
}

int
main() {
#ifndef NEMO_SPEECH_WITH_FLASHLIGHT
    std::fprintf(stdout, "[SKIP] built without flashlight (-DNEMO_SPEECH_WITH_FLASHLIGHT=ON)\n");
    return 0;
#else
    const char* lm = std::getenv("LM_PATH");
    const char* lex = std::getenv("LEXICON_PATH");
    const char* tok = std::getenv("TOKENIZER_PATH");
    const char* vocab_path = std::getenv("VOCAB_PATH");
    if (!lm || !lex || !tok || !vocab_path) {
        std::fprintf(stdout, "[SKIP] set LM_PATH, LEXICON_PATH, TOKENIZER_PATH, VOCAB_PATH\n");
        return 0;
    }

    // Token vocab (id order), matching the lexicon/LM and the GGUF.
    std::vector<std::string> vocab;
    {
        std::ifstream f(vocab_path);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            vocab.push_back(line);
        }
    }
    if (vocab.empty()) {
        std::fprintf(stderr, "empty vocab from %s\n", vocab_path);
        return 1;
    }
    std::unordered_map<std::string, int> piece2id;
    for (int i = 0; i < static_cast<int>(vocab.size()); ++i) piece2id[vocab[i]] = i;

    CtcConfig cc;
    cc.num_classes = static_cast<int>(vocab.size());
    cc.blank_id = static_cast<int>(vocab.size());  // blank appended after vocab
    const int n_classes = cc.blank_id + 1;

    FlashlightCtcCfg fcfg;
    fcfg.lm_path = lm;
    fcfg.lexicon_path = lex;
    fcfg.tokenizer_path = tok;

    FlashlightDecoder dec(cc, vocab, fcfg);

    // OOV word + its SentencePiece pieces (the decoder encodes the same way from
    // TOKENIZER_PATH; both use the same model, so they agree).
    const std::string oov = "prabhsimran";
    const std::vector<std::string> pieces = {"▁p", "ra", "b", "h", "s", "im", "ra", "n"};
    std::vector<int> piece_ids;
    for (const auto& p : pieces) {
        auto it = piece2id.find(p);
        if (it == piece2id.end()) {
            std::fprintf(stderr, "piece '%s' not in vocab\n", p.c_str());
            return 1;
        }
        piece_ids.push_back(it->second);
    }

    // Synthesize CTC log-probs: one frame per piece (strongly favoring it), then
    // blank frames to close the word. Frame-major [T][n_classes].
    auto make_logprobs = [&](std::vector<float>& lp) -> int {
        const float HI = 0.0f, LO = -30.0f;
        std::vector<int> seq = piece_ids;
        for (int k = 0; k < 4; ++k) seq.push_back(cc.blank_id);
        const int T = static_cast<int>(seq.size());
        lp.assign(static_cast<size_t>(T) * n_classes, LO);
        for (int t = 0; t < T; ++t) lp[static_cast<size_t>(t) * n_classes + seq[t]] = HI;
        return T;
    };
    auto contains = [](const std::string& hay, const std::string& needle) {
        return hay.find(needle) != std::string::npos;
    };

    // Case A: no boost. "prabhsimran" is OOV -> not in the trie -> cannot be emitted.
    {
        std::vector<float> lp;
        const int T = make_logprobs(lp);
        dec.reset();
        dec.step(lp.data(), n_classes, T, 0);
        dec.finalize();
        const std::string out = dec.final_transcript();
        std::fprintf(stdout, "baseline out: '%s'\n", out.c_str());
        check(!contains(out, oov), "OOV word NOT emitted without boost (not in lexicon)");
    }

    // Case B: boost the OOV word -> encoded + inserted into the trie -> emittable.
    {
        AsrRequestOptions opts;
        AsrRequestOptions::Boost b;
        b.phrases = {oov};
        b.boost = 20.0f;
        opts.speech_contexts = {b};
        dec.set_request_options(opts);

        std::vector<float> lp;
        const int T = make_logprobs(lp);
        dec.reset();
        dec.step(lp.data(), n_classes, T, 0);
        dec.finalize();
        const std::string out = dec.final_transcript();
        std::fprintf(stdout, "boosted out:  '%s'\n", out.c_str());
        check(contains(out, oov), "OOV word emitted after boosting");
    }

    // Case C: shared FlashlightResources (the server's per-model bundle). An
    // OOV boost must clone privately before mutating - the sharing decoder
    // emits the word, the bundle itself stays untouched for other streams.
    {
        auto res = std::make_shared<const FlashlightResources>(cc, vocab, fcfg);
        const int shared_words_before = res->word_count();

        FlashlightDecoder shared_dec(cc, vocab, fcfg, res);
        AsrRequestOptions opts;
        AsrRequestOptions::Boost b;
        b.phrases = {oov};
        b.boost = 20.0f;
        opts.speech_contexts = {b};
        shared_dec.set_request_options(opts);

        std::vector<float> lp;
        const int T = make_logprobs(lp);
        shared_dec.step(lp.data(), n_classes, T, 0);
        shared_dec.finalize();
        const std::string out = shared_dec.final_transcript();
        std::fprintf(stdout, "shared+boost: '%s'\n", out.c_str());
        check(contains(out, oov), "OOV boost works on a shared-resources decoder");
        check(
            res->word_count() == shared_words_before,
            "shared bundle not mutated by the OOV boost (private clone)");

        // A second decoder on the same bundle must not see the OOV word.
        FlashlightDecoder other(cc, vocab, fcfg, res);
        other.step(lp.data(), n_classes, T, 0);
        other.finalize();
        check(
            !contains(other.final_transcript(), oov),
            "sibling stream on the shared bundle unaffected");
    }

    std::fprintf(stdout, g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
#endif
}
