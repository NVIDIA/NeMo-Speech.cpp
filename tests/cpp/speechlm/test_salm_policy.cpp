// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// salm_policy against expected values (salm_policy_goldens.inc).
//
//   test_salm_policy                 # request-rule checks
//   test_salm_policy <decoder.gguf>  # + prompt token ids through the GGUF vocabulary
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llama.h"
#include "salm_policy.h"
#include "salm_policy_goldens.inc"

using namespace nemo_speech::speechlm::policy;

namespace {
int failures = 0;
int checks = 0;

std::string
show(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c >= 32 && c < 127) {
            out += static_cast<char>(c);
        } else {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", c);
            out += b;
        }
    }
    return out;
}

void
expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        if (failures <= 40)
            std::fprintf(stderr, "FAIL %s\n", what.c_str());
    }
}

void
expect_eq(const std::string& got, const char* want, const std::string& what) {
    const std::string w = want ? want : "<none>";
    expect(got == w, what + ": got '" + show(got) + "' want '" + show(w) + "'");
}

std::optional<std::string_view>
opt(const char* s) {
    return s ? std::optional<std::string_view>(s) : std::nullopt;
}

template <class F>
std::string
error_of(F&& f) {
    try {
        f();
    }
    catch (const PolicyError& e) {
        return e.what();
    }
    return "<none>";
}

std::vector<char32_t>
codepoints(const std::string& s) {
    std::vector<char32_t> out;
    const std::string decoded = utf8_lossy(s);
    for (size_t i = 0; i < decoded.size();) {
        const unsigned char b = decoded[i];
        const size_t n = b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
        char32_t cp = n == 1 ? b : n == 2 ? (b & 0x1F) : n == 3 ? (b & 0x0F) : (b & 0x07);
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (decoded[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

void
test_tables() {
    size_t spaces = 0;
    for (char32_t c = 0; c < 0x110000; ++c) spaces += py_isspace(c);
    expect(spaces == std::size(kPySpaces), "py_isspace size");
    for (char32_t c : kPySpaces) expect(py_isspace(c), "py_isspace member");
    const auto& names = language_names();
    expect(names.size() == std::size(kLanguageNames), "language map size");
    for (size_t i = 0; i < names.size() && i < std::size(kLanguageNames); ++i)
        expect(
            names[i].first == kLanguageNames[i].first &&
                names[i].second == kLanguageNames[i].second,
            "language map entry " + std::string(kLanguageNames[i].first));
    expect(kTranslationPrompt == kServedTranslationPrompt, "served translation prompt");
    expect(kMaxOutputTokens == kServedMaxTokens, "served max_tokens");
}

void
test_language() {
    for (const auto& c : kLangNameCases) {
        const auto got = language_name(opt(c.code));
        expect_eq(
            got ? *got : "<none>", c.name,
            "language_name(" + show(c.code ? c.code : "<null>") + ")");
    }
    for (const auto& c : kHintCases)
        expect_eq(
            language_hint(c.prompt, opt(c.language), c.enabled), c.expected,
            "language_hint(" + show(c.prompt).substr(0, 40) + ", " +
                (c.language ? c.language : "<null>") + ")");
}

void
test_prompts() {
    for (const auto& c : kValidateCases)
        expect_eq(
            error_of([&] { validate_prompt(c.prompt); }), c.error,
            "validate_prompt(" + show(c.prompt) + ")");
    for (const auto& c : kPlanCases) {
        PromptConfig config;
        config.transcription_prompt = kServedTranscriptionPrompt;
        if (c.override_prompt)
            config.transcription_override = c.override_prompt;
        config.language_hint = c.hint;
        PromptRequest request;
        request.task = c.translate ? Task::kTranslate : Task::kTranscribe;
        if (c.client_prompt)
            request.instruction = c.client_prompt;
        if (c.language)
            request.language = c.language;
        const std::string what =
            std::string(c.translate ? "translate" : "transcribe") +
            " client=" + show(c.client_prompt ? c.client_prompt : "<null>").substr(0, 40) +
            " lang=" + (c.language ? c.language : "<null>") +
            " override=" + show(c.override_prompt ? c.override_prompt : "<null>");
        try {
            const PromptPlan plan = plan_prompt(config, request);
            expect_eq(plan.user_turn, c.prompt, "plan " + what);
            expect(c.error == nullptr, "plan should fail " + what);
            expect_eq(plan.language, c.effective_language, "effective language " + what);
            expect(plan.from_client == c.from_client, "from_client " + what);
        }
        catch (const PolicyError& e) {
            expect_eq(e.what(), c.error, "plan error " + what);
        }
    }

    // Translation targets: any language of the table; English (or none) keeps the served prompt.
    {
        PromptConfig config;
        config.transcription_prompt = kServedTranscriptionPrompt;
        PromptRequest request;
        request.task = Task::kTranslate;
        expect_eq(plan_prompt(config, request).user_turn, kTranslationPrompt.data(), "no target");
        expect_eq(plan_prompt(config, request).language, "en", "no target language");
        request.target_language = "en-US";
        expect_eq(
            plan_prompt(config, request).user_turn, kTranslationPrompt.data(), "target en-US");
        request.target_language = "de";
        const auto german = plan_prompt(config, request);
        expect_eq(german.user_turn, "Translate the audio into German. <|audio|>", "target de");
        expect_eq(german.language, "de", "target de language");
        request.target_language = "ja-JP";
        expect_eq(
            plan_prompt(config, request).user_turn, "Translate the audio into Japanese. <|audio|>",
            "target ja-JP");
        request.target_language = "xx";
        try {
            plan_prompt(config, request);
            expect(false, "target xx should fail");
        }
        catch (const PolicyError& e) {
            expect_eq(e.what(), "target_language xx is not supported", "target xx error");
            expect_eq(e.param(), "target_language", "target xx param");
        }
        request.target_language = "de";
        request.instruction = "Translate the speech into French.";  // the client's instruction wins
        expect_eq(
            plan_prompt(config, request).user_turn, "Translate the speech into French. <|audio|>",
            "instruction over target");
    }
}

void
test_budget() {
    for (const auto& c : kBudgetCases) {
        const std::string what = "budget(" + std::to_string(c.seconds) + ", " +
                                 std::to_string(c.max_model_len) + ", " +
                                 std::to_string(c.max_tokens) + ")";
        try {
            const int64_t got = output_budget(c.seconds, {c.max_model_len, c.max_tokens});
            expect(got == c.budget && c.error == nullptr, what + " = " + std::to_string(got));
        }
        catch (const PolicyError& e) {
            expect_eq(e.what(), c.error, what);
        }
    }
    for (double d : kBadDurations)  // checked before the context
        expect_eq(error_of([&] { output_budget(d, {73728}); }), kBadDurationError, "bad duration");
    for (const auto& c : kDurationCases) {
        const auto got = duration_limit_error(c.seconds, c.limit);
        expect_eq(
            got ? *got : "<none>", c.error,
            "duration_limit_error(" + std::to_string(c.seconds) + ", " + std::to_string(c.limit) +
                ")");
    }
    // near-digital silence: peak < 1e-3 and RMS < 3e-5 (one PCM16 LSB is 3.05e-5)
    {
        const float lsb = 1.0f / 32768.0f;
        std::vector<float> zeros(16000, 0.0f), one_lsb(16000, lsb), sparse(16000, 0.0f),
            loud(16000, 0.0f);
        sparse[100] = 32 * lsb;  // a few 32-LSB samples: peak 9.8e-4, RMS well below 3e-5
        loud[100] = 33 * lsb;    // peak 1.007e-3: not silent
        expect(near_silent(nullptr, 0), "near_silent(empty)");
        expect(near_silent(zeros.data(), zeros.size()), "near_silent(zeros)");
        expect(
            !near_silent(one_lsb.data(), one_lsb.size()),
            "near_silent(constant 1 LSB): RMS 3.05e-5");
        expect(near_silent(sparse.data(), sparse.size()), "near_silent(one 32-LSB sample)");
        expect(!near_silent(loud.data(), loud.size()), "near_silent(one 33-LSB sample)");
    }
}

void
test_text_and_usage() {
    expect(
        utf8_lossy("a\xE0\x80"
                   "b\xED\xA0\x80\xF4\x90"
                   "c\xC3") ==
            "a\xEF\xBF\xBD\xEF\xBF\xBD"
            "b\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"
            "c\xEF\xBF\xBD",
        "maximal subpart replacement");
    expect(response_text("\xC2\xA0 hi there\n\xE3\x80\x80") == "hi there", "response text strip");
    expect(
        strip_speaker_tags("<spk:0> Hello there. <SPK:12> Hi.") == "Hello there. Hi." &&
            strip_speaker_tags("a <spk:> b <spk:x> c <spk:3") == "a <spk:> b <spk:x> c <spk:3" &&
            strip_speaker_tags("<spk:1>") == "",
        "speaker tags stripped");
    // Streamed text adds up to the whole: response_text() through ResponseTextStream in pieces of
    // whole characters, and generated bytes through utf8_lossy() of the bytes utf8_pending() leaves
    // settled, then utf8_finish() of the rest.
    for (const std::string text :
         {std::string(" \n Hello,\xE3\x80\x80world \t"),
          std::string("\xC2\xA0 hi there\n\xE3\x80\x80"), std::string(" \t\n"), std::string("a  b"),
          std::string("\xF0\x9F\x8E\x89 x\xE2\x80\xA8")}) {
        std::vector<size_t> bounds;
        for (size_t i = 0; i < text.size(); ++i)
            if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80)
                bounds.push_back(i);
        bounds.push_back(text.size());
        for (size_t width = 1; width <= 3; ++width) {
            ResponseTextStream stream;
            std::string shown;
            for (size_t j = 0; j + 1 < bounds.size(); j += width) {
                const size_t end = bounds[std::min(j + width, bounds.size() - 1)];
                shown += stream.push(std::string_view(text).substr(bounds[j], end - bounds[j]));
            }
            expect(shown == response_text(text), "streamed response text " + show(text));
        }
    }
    uint32_t state = 12345;
    const auto random = [&state] {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    };
    const unsigned char alphabet[] = {'a',  ' ',  0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0,
                                      0xC2, 0xDF, 0xE0, 0xED, 0xEF, 0xF0, 0xF4, 0xF5, 0xFF};
    for (int round = 0; round < 50000; ++round) {
        std::string bytes;
        for (size_t n = random() % 12; n > 0; --n)
            bytes += static_cast<char>(alphabet[random() % std::size(alphabet)]);
        std::string text, streamed;
        size_t settled = 0;
        for (size_t at = 0; at < bytes.size();) {
            const size_t take = std::min<size_t>(1 + random() % 3, bytes.size() - at);
            text.append(bytes, at, take);
            at += take;
            const size_t end = text.size() - utf8_pending(text);
            if (end > settled) {
                streamed += utf8_lossy(std::string_view(text).substr(settled, end - settled));
                settled = end;
            }
        }
        streamed += utf8_finish(std::string_view(text).substr(settled));
        expect(streamed == utf8_finish(bytes), "streamed utf8 " + show(bytes));
        // utf8_lossy replaces the incomplete tail that utf8_finish drops
        expect(
            utf8_finish(bytes) + (utf8_pending(bytes) ? "\xEF\xBF\xBD" : "") == utf8_lossy(bytes),
            "utf8 pending tail " + show(bytes));
    }
    expect(py_rstrip(" a \t") == " a" && py_strip("\xE2\x80\xA8x\xC2\x85") == "x", "strip");
    expect(codepoints("\xF0\x9F\x8E\x89").size() == 1, "codepoint helper");

    for (const auto& c : kUsageCases) {
        std::optional<Usage> total;
        for (const auto& step : c.steps) {
            Usage u{step.prompt, step.completion, std::nullopt};
            if (step.audio >= 0)
                u.audio_tokens = step.audio;
            add_usage(total, u);
        }
        expect(
            total && total->prompt_tokens == c.prompt && total->completion_tokens == c.completion &&
                total->audio_tokens.value_or(-1) == c.audio,
            "usage sum");
    }
    std::optional<Usage> none;
    add_usage(none, std::nullopt);
    expect(!none, "no decode, no usage");
    const Usage u = decode_usage(58, 13, 7, true);
    expect(
        u.prompt_tokens == 71 && u.completion_tokens == 8 && u.audio_tokens == 13,
        "decode usage counts EOS");
    expect(decode_usage(58, 13, 30, false).completion_tokens == 30, "length finish has no EOS");
}

void
test_vocabulary(const char* gguf) {
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model* model = llama_model_load_from_file(gguf, params);
    expect(model != nullptr, std::string("load vocabulary ") + gguf);
    if (!model)
        return;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto tokenize = [&](const std::string& text) {
        std::vector<llama_token> out(text.size() + 16);
        const int n = llama_tokenize(
            vocab, text.c_str(), static_cast<int32_t>(text.size()), out.data(),
            static_cast<int32_t>(out.size()), /*add_special=*/false, /*parse_special=*/true);
        out.resize(n > 0 ? n : 0);
        return std::vector<int>(out.begin(), out.end());
    };
    // The decoder's input around the audio rows, as the engine builds it: the chat prefix and the
    // instruction before the locator, the text after it and the generation prompt.
    for (const auto& c : kPromptIdsCases) {
        const std::string turn = c.user_turn;
        const size_t at = turn.find(kAudioLocator);
        const std::string prefix = std::string(kChatPrefix) + turn.substr(0, at);
        const std::string suffix =
            turn.substr(at + kAudioLocator.size()) + std::string(kChatSuffix);
        expect(prefix + std::string(kAudioLocator) + suffix == c.rendered, "rendered chat");
        expect(tokenize(prefix) == c.prefix, "prefix ids " + show(c.user_turn).substr(0, 50));
        expect(tokenize(suffix) == c.suffix, "suffix ids " + show(c.user_turn).substr(0, 50));
    }
    expect(llama_vocab_eos(vocab) == kEosId, "GGUF eos is <|im_end|>");
    std::fprintf(
        stderr, "vocabulary checks: eog(2)=%d eog(10)=%d eog(11)=%d\n",
        llama_vocab_is_eog(vocab, 2), llama_vocab_is_eog(vocab, 10), llama_vocab_is_eog(vocab, 11));
    llama_model_free(model);
    llama_backend_free();
}
}  // namespace

int
main(int argc, char** argv) {
    test_tables();
    test_language();
    test_prompts();
    test_budget();
    test_text_and_usage();
    if (argc > 1)
        test_vocabulary(argv[1]);
    std::fprintf(
        stderr, "salm_policy: %d checks, %d failures%s\n", checks, failures,
        argc > 1 ? " (with vocabulary)" : "");
    return failures ? 1 : 0;
}
