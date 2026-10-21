// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// latin_as_english: Korean (ByT5) text keeps its byte tokens, inline Latin runs become the
// English IPA tokens the "en" path would produce, and the chunk still ends with </s> + EOS.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tts/tokenizer/tokenizer.h"

namespace tts = nemo_speech::tts;

namespace {

bool
expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
    return ok;
}

std::vector<int32_t>
strip_eos(std::vector<int32_t> tokens) {
    // "en" chunks end with EOS (and may be padded before it for short text).
    while (!tokens.empty() && tokens.back() >= 3357) {
        tokens.pop_back();
    }
    return tokens;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <v2607 tokenizer dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    tts::MagpieTokenizerConfig on;
    on.latin_as_english = true;
    const tts::MagpieNativeTokenizer mixed(dir, on);
    const tts::MagpieNativeTokenizer plain(dir);

    constexpr int32_t kKoOffset = 2973;  // korean_chartokenizer in the v2607 profile
    constexpr int32_t kKoEos = kKoOffset + 1;
    constexpr int32_t kEos = 3358;
    bool ok = true;

    // 1) Korean-only text is unchanged by the flag.
    const std::string ko = "안녕하십니까, 민원상담센터입니다.";
    ok &= expect(
        mixed.tokenize(ko, "ko").tokens == plain.tokenize(ko, "ko").tokens,
        "korean-only text unchanged");

    // 2) Mixed text: head bytes + English IPA + tail bytes + </s> + EOS.
    const std::string head = "담당자가 ";
    const std::string latin = "Please hold the line";
    const std::string tail = " 이라고 안내할 수 있습니다.";
    const auto got = mixed.tokenize(head + latin + tail, "ko").tokens;
    std::vector<int32_t> want;
    for (unsigned char b : head) want.push_back(kKoOffset + b + 3);
    const auto en = strip_eos(plain.tokenize(latin, "en").tokens);
    ok &= expect(!en.empty(), "english tokens non-empty");
    for (int32_t t : en) ok &= expect(t >= 0 && t < 96, "english tokens in english_phoneme range");
    want.insert(want.end(), en.begin(), en.end());
    for (unsigned char b : tail) want.push_back(kKoOffset + b + 3);
    want.push_back(kKoEos);
    want.push_back(kEos);
    if (!expect(got == want, "mixed token stream")) {
        std::fprintf(stderr, "got (%zu):", got.size());
        for (int32_t t : got) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\nwant (%zu):", want.size());
        for (int32_t t : want) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");
    }

    // 3) Flag off: the same text is plain bytes (no english_phoneme-range tokens).
    const auto off = plain.tokenize(head + latin + tail, "ko").tokens;
    bool any_en = false;
    for (int32_t t : off) any_en |= (t < 96);
    ok &= expect(!any_en, "flag off keeps bytes");

    // 4) Hyphen/apostrophe runs stay one English run; French (Latin script) is untouched.
    const auto wifi = mixed.tokenize("서류는 Wi-Fi 연결을 확인해 주세요.", "ko").tokens;
    int en_count = 0;
    for (int32_t t : wifi) en_count += (t < 96);
    ok &= expect(en_count >= 3, "Wi-Fi run encoded as English");
    const std::string fr = "Bonjour, comment allez-vous?";
    ok &= expect(
        mixed.tokenize(fr, "fr").tokens == plain.tokenize(fr, "fr").tokens, "french unchanged");

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
