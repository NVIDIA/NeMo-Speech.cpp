// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "profanity.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace nemo_speech::asr::postproc {

namespace {
std::string
lower(const std::string& s) {
    std::string o = s;
    for (char& c : o)
        if ((unsigned char)c <= 0x7F)
            c = (char)std::tolower((unsigned char)c);
    return o;
}

// Byte length of the UTF-8 codepoint whose lead byte is `b` (1-4). An invalid or
// continuation lead is treated as 1 so iteration always advances.
size_t
utf8_cp_len(unsigned char b) {
    if (b < 0x80)
        return 1;
    if ((b >> 5) == 0x6)
        return 2;
    if ((b >> 4) == 0xE)
        return 3;
    if ((b >> 3) == 0x1E)
        return 4;
    return 1;
}
}  // namespace

Profanity::Profanity(const std::string& list_path) {
    if (list_path.empty())
        return;
    std::ifstream f(list_path);
    std::string line;
    while (std::getline(f, line)) {
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos)
            continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        words_.insert(lower(line.substr(a, b - a + 1)));
    }
}

std::string
Profanity::mask(const std::string& text) const {
    if (words_.empty())
        return text;
    std::ostringstream out;
    std::istringstream in(text);
    std::string tok;
    bool first = true;
    while (in >> tok) {
        // Split a token into leading punct + core word + trailing punct (keep
        // punct). Only ASCII punctuation is trimmed - guarding on < 0x80 keeps the
        // loop from ever cutting a multi-byte UTF-8 character mid-codepoint.
        size_t start = 0;
        size_t end = tok.size();
        while (start < end && (unsigned char)tok[start] < 0x80 &&
               std::ispunct((unsigned char)tok[start]))
            start++;
        while (end > start && (unsigned char)tok[end - 1] < 0x80 &&
               std::ispunct((unsigned char)tok[end - 1]))
            end--;
        const std::string core = tok.substr(start, end - start);
        const std::string punct_before = tok.substr(0, start);
        const std::string punct_after = tok.substr(end);
        std::string masked = tok;
        if (!core.empty() && words_.count(lower(core))) {
            // Mask per UTF-8 codepoint, not per byte: keep the first character,
            // replace each remaining character with '*'. Byte-wise masking split
            // multi-byte characters (e.g. Devanagari) and emitted invalid UTF-8,
            // which fails protobuf string serialization of the response.
            masked = punct_before;
            bool kept_first = false;
            for (size_t i = 0; i < core.size();) {
                const size_t n = std::min(utf8_cp_len((unsigned char)core[i]), core.size() - i);
                if (!kept_first) {
                    masked += core.substr(i, n);
                    kept_first = true;
                } else {
                    masked += '*';
                }
                i += n;
            }
            masked += punct_after;
        }
        if (!first)
            out << ' ';
        out << masked;
        first = false;
    }
    return out.str();
}

}  // namespace nemo_speech::asr::postproc
