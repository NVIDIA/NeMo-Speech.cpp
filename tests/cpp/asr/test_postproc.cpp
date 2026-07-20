// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Unit tests for the model-free ASR postprocessing pieces: profanity masking,
// pipeline gating, and ITN timing-remap projection. In an ITN-enabled build,
// ITN_MODEL_DIR enables the grammar-backed normalization cases.
//
// Usage: ./test_postproc   (writes a temp word list; no args)
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "itn.h"
#include "itn_align.h"
#ifdef NEMO_SPEECH_WITH_NORM
#include "itn_test_cases.h"
#endif
#include "pipeline.h"
#include "profanity.h"
#include "types.h"  // AsrRequestOptions, Result

using namespace nemo_speech::asr;

static int g_fail = 0;

static void
check(bool ok, const char* what) {
    std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        g_fail++;
}

#ifdef NEMO_SPEECH_WITH_NORM
static void
check_eq(const std::string& actual, std::string_view expected, const std::string& what) {
    const bool ok = actual == expected;
    std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        std::fprintf(
            stdout, "       expected: [%.*s]\n       actual:   [%s]\n",
            static_cast<int>(expected.size()), expected.data(), actual.c_str());
        g_fail++;
    }
}

template <size_t N>
static void
run_itn_cases(const postproc::Itn& itn, const std::string& suite, const test::ItnCase (&cases)[N]) {
    for (const auto& tc : cases) {
        const std::string input(tc.input);
        const std::string actual = itn.normalize(input);
        check_eq(actual, tc.expected, suite + ": " + std::string(tc.category));
        if (actual != tc.expected)
            std::fprintf(stdout, "       links:\n%s", itn.alignment(input).c_str());
    }
}

#endif

int
main() {
    // Keep the last completed check visible when CI pipes stdout through tee.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    // Write a small profanity list to a temp file. temp_directory_path() honors
    // TMPDIR/TMP/TEMP and is portable (no hardcoded /tmp).
    const std::string list =
        (std::filesystem::temp_directory_path() / "test_prof_list.txt").string();
    {
        std::ofstream f(list);
        f << "darn\nheck\n"
          << "\xE0\xA4\xA8\xE0\xA4\xA6\xE0\xA5\x80\n";  // नदी
    }

    postproc::Profanity prof(list);
    check(prof.enabled(), "profanity list loaded");
    check(prof.mask("oh darn the heck") == "oh d*** the h***", "whole-word masking");
    // UTF-8 masking keeps the first codepoint and emits one '*' per remaining codepoint.
    check(
        prof.mask("\xE0\xA4\xAF\xE0\xA4\xB9 "              // यह
                  "\xE0\xA4\xA8\xE0\xA4\xA6\xE0\xA5\x80 "  // नदी
                  "\xE0\xA4\xB9\xE0\xA5\x88") ==           // है
            "\xE0\xA4\xAF\xE0\xA4\xB9 \xE0\xA4\xA8** \xE0\xA4\xB9\xE0\xA5\x88",
        "UTF-8 (Devanagari) word masked per codepoint, valid UTF-8");
    check(prof.mask("darn, it") == "d***, it", "trailing punctuation preserved");
    check(prof.mask("(darn)") == "(d***)", "leading + trailing punctuation preserved");
    check(prof.mask("darned") == "darned", "substring not masked (whole word only)");
    check(prof.mask("DARN") == "D***", "case-insensitive match, original case of 1st char kept");

    postproc::Profanity off("");
    check(!off.enabled() && off.mask("darn") == "darn", "empty list = no-op");

    // Pipeline gating: profanity only when the request asks for it.
    postproc::PostprocConfig cfg;
    cfg.profanity_list_path = list;
    cfg.cpu_workers = 2;
    cfg.max_queue_depth = 2;
    postproc::Postprocessor pp(cfg);
    AsrRequestOptions opts;  // all off
    check(pp.apply("oh darn", opts) == "oh darn", "pipeline: profanity off by default");
    opts.profanity_filter = true;
    check(pp.apply("oh darn", opts) == "oh d***", "pipeline: profanity on when requested");

    opts.profanity_filter = false;
    opts.enable_automatic_punctuation = true;
    struct PunctuationSpacingCase {
        const char* input;
        const char* expected;
    };
    static constexpr PunctuationSpacingCase kPunctuationSpacingCases[] = {
        {"The total was €500 .", "The total was €500."},
        {"Thunder boomed , echoing across the silent valley .",
         "Thunder boomed, echoing across the silent valley."},
        {"word : a single distinct element", "word: a single distinct element"},
        {"Oh no! Another stain ? ! Where are the napkins ?",
         "Oh no! Another stain?! Where are the napkins?"},
        {"The swing creaked ( a childhood forgotten )",
         "The swing creaked (a childhood forgotten)"},
        {"runner - up", "runner-up"},
        {"preserve trailing space .  ", "preserve trailing space.  "},
    };
    bool punctuation_spacing_ok = true;
    for (const auto& tc : kPunctuationSpacingCases)
        punctuation_spacing_ok &= pp.apply(tc.input, opts) == tc.expected;
    check(punctuation_spacing_ok, "pipeline: punctuation spacing");

    // More callers than workers + queue slots exercises bounded backpressure.
    opts.profanity_filter = true;
    std::vector<std::string> concurrent_out(12);
    std::vector<std::thread> callers;
    for (size_t i = 0; i < concurrent_out.size(); ++i) {
        callers.emplace_back([&, i] { concurrent_out[i] = pp.apply("oh darn", opts); });
    }
    for (auto& caller : callers) caller.join();
    bool concurrent_ok = true;
    for (const auto& out : concurrent_out) concurrent_ok &= (out == "oh d***");
    check(concurrent_ok, "pipeline: bounded concurrent postprocessing");

    // ITN gate: verbatim skips ITN; with no ITN build it's pass-through anyway.
    opts.profanity_filter = false;
    opts.verbatim_transcripts = true;
    check(pp.apply("twenty twenty three", opts) == "twenty twenty three", "verbatim -> no ITN");

    // ----- ITN timing-remap projection -----
    // Hardcoded ShowLinks-format alignment + spoken spans into update_word_timings().
    using postproc::update_word_timings;
    {
        // Collapse: "twenty twenty three" (3 spoken) -> "2023" (1 written).
        std::vector<WordTiming> w = {
            {"twenty", 10, 18, 1.f}, {"twenty", 20, 28, 1.f}, {"three", 30, 40, 1.f}};
        update_word_timings(w, "Token:\t0\ttwenty twenty three\t0,18\t0,0\nWord:\t0\t2023\t0\n");
        check(
            w.size() == 1 && w[0].word == "2023" && w[0].start_frame == 10 && w[0].end_frame == 40,
            "remap: 3 spoken collapse onto 1 written span");
    }
    {
        // Unchanged words stay 1:1 with their own spans.
        std::vector<WordTiming> w = {{"the", 0, 5, 1.f}, {"meeting", 6, 15, 1.f}};
        update_word_timings(
            w,
            "Token:\t0\tthe\t0,3\t0,0\nToken:\t1\tmeeting\t4,11\t1,1\n"
            "Word:\t0\tthe\t0\nWord:\t1\tmeeting\t1\n");
        check(
            w.size() == 2 && w[0].word == "the" && w[0].start_frame == 0 &&
                w[1].word == "meeting" && w[1].end_frame == 15,
            "remap: unchanged words keep their spans");
    }
    {
        // Mixed: 3 unchanged + a 3->1 collapse.
        std::vector<WordTiming> w = {{"call", 0, 4, 1.f},   {"me", 5, 7, 1.f},
                                     {"at", 8, 10, 1.f},    {"five", 11, 13, 1.f},
                                     {"five", 14, 16, 1.f}, {"five", 17, 20, 1.f}};
        update_word_timings(
            w,
            "Token:\t0\tcall\t0,4\t0,0\nToken:\t1\tme\t5,7\t1,1\nToken:\t2\tat\t8,10\t2,2\n"
            "Token:\t3\tfive five five\t11,25\t3,3\n"
            "Word:\t0\tcall\t0\nWord:\t1\tme\t1\nWord:\t2\tat\t2\nWord:\t3\t555\t3\n");
        check(
            w.size() == 4 && w[3].word == "555" && w[3].start_frame == 11 && w[3].end_frame == 20,
            "remap: mixed unchanged + collapse");
    }
    {
        // Preserve the unit's original timing while collapsing its number.
        std::vector<WordTiming> w = {{"two", 10, 18, 0.8f}, {"kilograms", 30, 40, 1.0f}};
        update_word_timings(
            w,
            "Token:\t0\ttwo kilograms\t0,12\t0,1\n"
            "Word:\t0\t2\t0\nWord:\t1\tkg\t0\n");
        check(
            w.size() == 2 && w[0].word == "2" && w[0].start_frame == 10 && w[0].end_frame == 18 &&
                w[1].word == "kg" && w[1].start_frame == 30 && w[1].end_frame == 40,
            "remap: exact/abbreviated unit keeps its own span");
    }
    {
        // Unparseable alignment is a no-op (timings kept).
        std::vector<WordTiming> w = {{"hello", 0, 5, 1.f}};
        update_word_timings(w, "garbage not links");
        check(w.size() == 1 && w[0].word == "hello", "remap: unparseable links keep timings");
    }
#ifdef NEMO_SPEECH_WITH_NORM
    // Real Sparrowhawk ITN, when ITN_MODEL_DIR points at a grammar dir.
    if (const char* itn_dir = std::getenv("ITN_MODEL_DIR")) {
        postproc::Itn itn(itn_dir);
        check(itn.enabled(), "ITN: grammar dir loaded");

        run_itn_cases(itn, "ITN English cases", test::kEnglishItnCases);

        // The normalizer is shared by concurrent streams. Exercise enough
        // calls to catch accidental mutation of loaded FARs or synchronization
        // regressions without turning this into a performance benchmark.
        std::atomic<bool> concurrent_ok{true};
        std::vector<std::thread> workers;
        for (int t = 0; t < 4; ++t) {
            workers.emplace_back([&] {
                for (int i = 0; i < 4; ++i) {
                    if (itn.normalize("twenty twenty three") != "2023" ||
                        itn.normalize("five hundred two") != "502")
                        concurrent_ok = false;
                }
            });
        }
        for (auto& worker : workers) worker.join();
        check(concurrent_ok, "ITN: concurrent normalization is deterministic");

        bool bad_dir_rejected = false;
        try {
            postproc::Itn bad("/path/that/does/not/exist/nemotron-itn-test");
        }
        catch (const std::exception&) {
            bad_dir_rejected = true;
        }
        check(bad_dir_rejected, "ITN: nonexistent grammar directory is rejected");

        // Riva's production loader needs only the classifier and verbalizer
        // FARs; ASCII proto sidecars are optional.
        const auto unique_suffix =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        const auto minimal_dir =
            std::filesystem::temp_directory_path() / ("nemo_speech_itn_two_far_" + unique_suffix);
        std::filesystem::remove_all(minimal_dir);
        std::filesystem::create_directories(minimal_dir);
        std::filesystem::copy_file(
            std::filesystem::path(itn_dir) / "tokenize_and_classify.far",
            minimal_dir / "tokenize_and_classify.far");
        bool incomplete_dir_rejected = false;
        try {
            postproc::Itn incomplete(minimal_dir.string());
        }
        catch (const std::exception&) {
            incomplete_dir_rejected = true;
        }
        check(incomplete_dir_rejected, "ITN: missing required FAR is rejected");
        std::filesystem::copy_file(
            std::filesystem::path(itn_dir) / "verbalize.far", minimal_dir / "verbalize.far");
        postproc::Itn minimal_itn(minimal_dir.string());
        check(minimal_itn.normalize("five hundred two") == "502", "ITN: two-FAR loader works");

        // A parent grammar directory selects a child by the request/detected
        // language code. Locale suffixes fall back to the base language, while
        // an unsupported language must not accidentally run the English FST.
        const auto multilingual_dir = std::filesystem::temp_directory_path() /
                                      ("nemo_speech_itn_multilingual_" + unique_suffix);
        const auto english_child = multilingual_dir / "en";
        std::filesystem::remove_all(multilingual_dir);
        std::filesystem::create_directories(english_child);
        std::filesystem::copy_file(
            minimal_dir / "tokenize_and_classify.far", english_child / "tokenize_and_classify.far");
        std::filesystem::copy_file(minimal_dir / "verbalize.far", english_child / "verbalize.far");
        postproc::PostprocConfig multilingual_cfg;
        multilingual_cfg.itn_model_dir = multilingual_dir.string();
        postproc::Postprocessor multilingual_itn(multilingual_cfg);
        AsrRequestOptions multilingual_opts;
        multilingual_opts.enable_automatic_punctuation = true;
        check(
            multilingual_itn.apply("five hundred two", multilingual_opts, nullptr, "en-US") ==
                "502",
            "ITN multilingual: locale selects its base-language grammar");
        check(
            multilingual_itn.apply("five hundred two", multilingual_opts, nullptr, "es-ES") ==
                "five hundred two",
            "ITN multilingual: missing language grammar is pass-through");
        std::filesystem::remove_all(multilingual_dir);
        std::filesystem::remove_all(minimal_dir);

        postproc::PostprocConfig icfg;
        icfg.itn_model_dir = itn_dir;
        postproc::Postprocessor ipp(icfg);
        AsrRequestOptions iopts;                    // verbatim off -> ITN runs
        iopts.enable_automatic_punctuation = true;  // do not apply plain-text rendering here
        check(
            ipp.apply("twenty twenty three", iopts) == "2023",
            "ITN pipeline: default -> normalized");
        check(
            ipp.apply("the total was five hundred euros.", iopts) == "the total was €500.",
            "ITN pipeline: normalized money keeps terminal punctuation attached");
        iopts.verbatim_transcripts = true;
        check(
            ipp.apply("twenty twenty three", iopts) == "twenty twenty three",
            "ITN pipeline: verbatim -> raw");

        // Word-timing remap: 3 spoken words -> 1 written word spanning their range.
        std::vector<nemo_speech::asr::WordTiming> words = {
            {"twenty", 10, 18, 1.0f}, {"twenty", 20, 28, 1.0f}, {"three", 30, 40, 1.0f}};
        AsrRequestOptions topts;
        topts.enable_word_time_offsets = true;
        const std::string rout = ipp.apply("twenty twenty three", topts, &words);
        check(rout == "2023", "ITN remap: text normalized");
        check(words.size() == 1 && words[0].word == "2023", "ITN remap: 3 words -> 1 ('2023')");
        check(
            !words.empty() && words[0].start_frame == 10 && words[0].end_frame == 40,
            "ITN remap: span covers the original words");

    } else {
        check(false, "ITN_MODEL_DIR is required by an ITN-enabled test build");
    }
#endif

    std::remove(list.c_str());  // clean up the unique temp word list
    std::fprintf(stdout, g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
