// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>

namespace nemo_speech::asr::test {

struct ItnCase {
    std::string_view category;
    std::string_view input;
    std::string_view expected;
};

inline constexpr ItnCase kEnglishItnCases[] = {
    {"cardinal", "five hundred two", "502"},
    {"ordinal", "third", "3rd"},
    {"decimal", "point two o three", ".203"},
    {"money", "one dollar", "$1"},
    {"date", "eleventh of october two thousand four", "11th of october 2004"},
    {"time", "two thirty p m", "02:30 p.m."},
    {"measure", "two kilograms", "2 kg"},
    {"telephone", "four one three two three seven six nine seven two", "413-237-6972"},
    {"identity", "words already normalized", "words already normalized"},
    {"identity/empty", "", ""},
    {"whitespace", "words  ", "words  "},
};

}  // namespace nemo_speech::asr::test
