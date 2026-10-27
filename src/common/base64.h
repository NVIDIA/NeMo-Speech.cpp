// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace nemo_speech {

// Strict RFC 4648 base64 (standard alphabet): no whitespace, padding required, non-zero trailing
// bits refused; nullopt for anything else.
inline std::optional<std::string>
decode_base64(std::string_view input) {
    static const auto table = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i)
            t[static_cast<unsigned char>(alphabet[i])] = static_cast<int8_t>(i);
        return t;
    }();
    if (input.size() % 4 != 0)
        return std::nullopt;
    std::string out(input.size() / 4 * 3, '\0');
    size_t n = 0;
    for (size_t i = 0; i < input.size(); i += 4) {
        const bool last = i + 4 == input.size();
        int v[4];
        int padding = 0;
        for (int k = 0; k < 4; ++k) {
            const auto c = static_cast<unsigned char>(input[i + k]);
            if (c == '=' && last && k >= 2) {
                ++padding;
                v[k] = 0;
                continue;
            }
            if (padding || table[c] < 0)
                return std::nullopt;
            v[k] = table[c];
        }
        const uint32_t bits = (v[0] << 18) | (v[1] << 12) | (v[2] << 6) | v[3];
        if ((padding == 2 && (bits & 0xFFFF)) || (padding == 1 && (bits & 0xFF)))
            return std::nullopt;  // non-zero trailing bits
        out[n++] = static_cast<char>(bits >> 16);
        if (padding < 2)
            out[n++] = static_cast<char>((bits >> 8) & 0xFF);
        if (padding < 1)
            out[n++] = static_cast<char>(bits & 0xFF);
    }
    out.resize(n);
    return out;
}

}  // namespace nemo_speech
