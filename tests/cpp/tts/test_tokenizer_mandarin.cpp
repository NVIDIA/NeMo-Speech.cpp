// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tts/tokenizer/tokenizer.h"

#ifndef MANDARIN_TEST_DATA_DIR
#define MANDARIN_TEST_DATA_DIR ""
#endif

namespace {

std::vector<std::string>
read_lines(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to read " + path);
    }
    std::vector<std::string> result;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            result.push_back(line);
        }
    }
    return result;
}

std::vector<int32_t>
parse_tokens(const std::string& line) {
    std::vector<int32_t> result;
    std::istringstream input(line);
    std::string value;
    while (std::getline(input, value, ',')) {
        result.push_back(static_cast<int32_t>(std::stoi(value)));
    }
    return result;
}

bool
check_equal(
    const std::string& text, const std::vector<int32_t>& expected,
    const std::vector<int32_t>& actual) {
    if (expected == actual) {
        return true;
    }
    std::fprintf(stderr, "Mandarin token mismatch for: %s\nexpected:", text.c_str());
    for (const int32_t value : expected) {
        std::fprintf(stderr, " %d", value);
    }
    std::fprintf(stderr, "\nactual:");
    for (const int32_t value : actual) {
        std::fprintf(stderr, " %d", value);
    }
    std::fprintf(stderr, "\n");
    return false;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s TOKENIZER_MODEL_DIR [TEST_DATA_DIR]\n", argv[0]);
        return 2;
    }

    try {
        const std::string data_dir = argc == 3 ? argv[2] : MANDARIN_TEST_DATA_DIR;
        const auto utterances = read_lines(data_dir + "/mandarin.list");
        const auto golden_lines = read_lines(data_dir + "/mandarin.golden.tokens");
        if (utterances.size() != golden_lines.size() || utterances.empty()) {
            std::fprintf(stderr, "Mandarin fixtures are empty or golden counts differ\n");
            return 1;
        }

        nemo_speech::tts::MagpieNativeTokenizer tokenizer(argv[1]);
        const bool v2607 = tokenizer.profile_id() == "v2607";
        std::vector<std::vector<int32_t>> expected;
        bool ok = true;
        for (size_t index = 0; index < utterances.size(); ++index) {
            expected.push_back(parse_tokens(golden_lines[index]));
            if (v2607) {
                for (auto& token : expected.back()) {
                    token = token == 2361 ? 3358 : token + 384;
                }
            }
            const auto result = tokenizer.tokenize(utterances[index], "zh-CN");
            ok &= result.language == "zh";
            ok &= result.tokenizer_name == "mandarin_phoneme";
            ok &= result.chunks.size() == 1;
            ok &= check_equal(utterances[index], expected.back(), result.tokens);
        }

        nemo_speech::tts::MagpieTokenizerConfig chunking_config;
        chunking_config.sentence_limit.zh = 1;
        nemo_speech::tts::MagpieNativeTokenizer chunking_tokenizer(argv[1], chunking_config);
        int transformed_chunks = 0;
        const auto chunked =
            chunking_tokenizer.tokenize("你好。世界！", "zh", [&](const std::string& text) {
                ++transformed_chunks;
                return text;
            });
        if (chunked.chunks.size() != 2 || transformed_chunks != 2 ||
            chunked.chunks[0].text != "你好。" || chunked.chunks[1].text != "世界！") {
            std::fprintf(stderr, "Mandarin CJK sentence splitting failed\n");
            ok = false;
        }

        const size_t concurrent_index = utterances.size() > 1 && expected.size() > 1 ? 1 : 0;
        const std::string concurrent_utterance = utterances[concurrent_index];
        const std::vector<int32_t> concurrent_expected = expected[concurrent_index];
        std::atomic<bool> concurrent_ok{true};
        std::vector<std::thread> workers;
        for (int worker = 0; worker < 4; ++worker) {
            workers.emplace_back([&tokenizer, &concurrent_ok, concurrent_utterance,
                                  concurrent_expected]() {
                try {
                    for (int iteration = 0; iteration < 10; ++iteration) {
                        const auto result = tokenizer.tokenize(concurrent_utterance, "zh");
                        if (result.tokens != concurrent_expected) {
                            concurrent_ok = false;
                        }
                    }
                }
                catch (const std::exception& e) {
                    std::fprintf(stderr, "Mandarin concurrent tokenization failed: %s\n", e.what());
                    concurrent_ok = false;
                }
                catch (...) {
                    std::fprintf(
                        stderr, "Mandarin concurrent tokenization failed: unknown error\n");
                    concurrent_ok = false;
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        ok &= concurrent_ok.load();

        return ok ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Mandarin tokenizer test failed: %s\n", e.what());
        return 1;
    }
    catch (...) {
        std::fprintf(stderr, "Mandarin tokenizer test failed: unknown error\n");
        return 1;
    }
}
