// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <string>

#include "langpairs.h"

namespace langpairs = nemo_speech::nmt::langpairs;

namespace {

int failures = 0;

void
check(const std::string& actual, const std::string& expected, const char* label) {
    const bool passed = actual == expected;
    std::fprintf(
        stdout, "[%s] %s%s%s\n", passed ? "PASS" : "FAIL", label, passed ? "" : ": expected ",
        passed ? "" : expected.c_str());
    if (!passed) {
        std::fprintf(stdout, "       actual: %s\n", actual.c_str());
        ++failures;
    }
}

}  // namespace

int
main() {
    check(langpairs::normalize_language_code("EN-US"), "en", "BCP-47 English fallback");
    check(langpairs::normalize_language_code("es-ES"), "es-es", "supported region is preserved");
    check(langpairs::resolve_tag("en-US", "de-DE"), "en-de", "regional English to German");
    check(langpairs::resolve_tag("es-ES", "en-US"), "es-es-en", "regional Spanish to English");
    check(langpairs::resolve_tag("en-de", "de"), "en-de", "ready pair tag is accepted");
    check(langpairs::resolve_tag("xx-YY", "en-US"), "", "unsupported language is rejected");
    check(
        langpairs::build_prompt("en-de", "Hello world."),
        "<s>System\nYou are an expert at translating text from English to German.</s>\n"
        "<s>User\nWhat is the German translation of the sentence: Hello world.</s>\n"
        "<s>Assistant\n",
        "v1.1 chat prompt");
    std::fprintf(stdout, failures ? "FAILED (%d)\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
