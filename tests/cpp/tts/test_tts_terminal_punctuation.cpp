// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <iostream>
#include <string>

#include "tts/tokenizer/tokenizer.h"

namespace {

void
check(const std::string& actual, const std::string& expected, const char* description) {
    if (actual == expected) {
        return;
    }
    std::cerr << "FAIL: " << description << "\nexpected: '" << expected << "'\nactual:   '"
              << actual << "'\n";
    std::exit(1);
}

}  // namespace

int
main() {
    using nemo_speech::tts::ensure_terminal_punctuation;

    check(ensure_terminal_punctuation("Hello", "en-US"), "Hello.", "English period");
    check(ensure_terminal_punctuation("Hola", "es-ES"), "Hola.", "Spanish period");
    check(ensure_terminal_punctuation("Bonjour", "fr-FR"), "Bonjour.", "French period");
    check(ensure_terminal_punctuation("Hallo", "de-DE"), "Hallo.", "German period");
    check(ensure_terminal_punctuation("Ciao", "it-IT"), "Ciao.", "Italian period");
    check(
        ensure_terminal_punctuation("Tôi muốn đặt bàn", "vi-VN"), "Tôi muốn đặt bàn.",
        "Vietnamese period");
    check(ensure_terminal_punctuation("नमस्ते दुनिया", "hi-IN"), "नमस्ते दुनिया।", "Hindi danda");
    check(ensure_terminal_punctuation("你好世界", "zh-CN"), "你好世界。", "Chinese full stop");
    check(
        ensure_terminal_punctuation("こんにちは世界", "ja-JP"), "こんにちは世界。",
        "Japanese full stop");

    check(ensure_terminal_punctuation("Already.", "en"), "Already.", "existing period");
    check(
        ensure_terminal_punctuation("Already .", "en"), "Already.",
        "remove whitespace before existing terminal");
    check(ensure_terminal_punctuation("Question?", "vi"), "Question?", "existing question mark");
    check(ensure_terminal_punctuation("Stop!", "hi"), "Stop!", "existing exclamation mark");
    check(ensure_terminal_punctuation("नमस्ते।", "hi"), "नमस्ते।", "existing Hindi danda");
    check(ensure_terminal_punctuation("你好。", "zh"), "你好。", "existing Chinese full stop");
    check(ensure_terminal_punctuation("你好？", "zh"), "你好？", "existing Chinese question mark");
    check(
        ensure_terminal_punctuation("كيف حالك؟", "ar-MSA"), "كيف حالك؟",
        "existing Arabic question mark");
    check(
        ensure_terminal_punctuation("你好！", "zh"), "你好！", "existing Chinese exclamation mark");
    check(
        ensure_terminal_punctuation("こんにちは。", "ja"), "こんにちは。",
        "existing Japanese full stop");
    check(
        ensure_terminal_punctuation("Trailing spaces  ", "en"), "Trailing spaces.  ",
        "terminator precedes trailing whitespace");
    check(ensure_terminal_punctuation("Continue,", "en"), "Continue.", "replace comma");
    check(ensure_terminal_punctuation("Continue ,", "en"), "Continue.", "replace spaced comma");
    check(ensure_terminal_punctuation("Continue:", "de"), "Continue.", "replace colon");
    check(
        ensure_terminal_punctuation("Continue;  ", "fr"), "Continue.  ",
        "replace semicolon before trailing whitespace");
    check(
        ensure_terminal_punctuation("जारी रखें,", "hi"), "जारी रखें।",
        "replace comma with Hindi danda");
    check(
        ensure_terminal_punctuation("继续:", "zh"), "继续。",
        "replace colon with Chinese full stop");
    check(
        ensure_terminal_punctuation("続ける;", "ja"), "続ける。",
        "replace semicolon with Japanese full stop");
    check(ensure_terminal_punctuation("   ", "en"), "   ", "whitespace-only input unchanged");
    check(
        ensure_terminal_punctuation("Unsupported", "xx"), "Unsupported",
        "unsupported language unchanged");
    check(
        ensure_terminal_punctuation("Unsupported,", "xx"), "Unsupported,",
        "unsupported language comma unchanged");

    return 0;
}
