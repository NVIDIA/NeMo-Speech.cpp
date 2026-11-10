// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace nemo_speech::tts {

// Longform synthesis encodes each text chunk in a window of `history` earlier tokens followed by
// the chunk itself. The history rows of the encoder output are replaced with the context those
// tokens had when they were synthesized, kept in a rolling per-token cache.

// Overwrites the history rows of text_cond (the first text_len - current_chunk_len rows) with
// the newest rows of the cache.
inline bool
splice_longform_history_context(
    std::vector<float>& text_cond, int text_len, int current_chunk_len, int n_embd,
    const std::vector<float>& previous_context, int previous_context_len) {
    if (text_len < 0 || current_chunk_len < 0 || current_chunk_len > text_len || n_embd <= 0) {
        std::fprintf(stderr, "invalid longform text context dimensions\n");
        return false;
    }
    const int history_len = text_len - current_chunk_len;
    if (history_len <= 0) {
        return true;
    }
    if (previous_context_len < history_len ||
        previous_context.size() < (size_t)previous_context_len * (size_t)n_embd ||
        text_cond.size() < (size_t)text_len * (size_t)n_embd) {
        std::fprintf(
            stderr, "longform history context cache is too short: need %d token(s), have %d\n",
            history_len, previous_context_len);
        return false;
    }

    const float* src =
        previous_context.data() + (size_t)(previous_context_len - history_len) * (size_t)n_embd;
    std::copy(src, src + (size_t)history_len * (size_t)n_embd, text_cond.data());
    return true;
}

// Appends a chunk's own rows to the rolling cache: the per-token context of every token
// synthesized so far, in order, capped at max_tokens. A window's history is capped by its own
// chunk length, so caching only the previous window falls short when a short chunk is followed by
// a longer one. The history rows of text_cond were spliced from this cache, so the cache tail
// already holds them and only the current chunk's rows are new.
inline bool
append_longform_history_context(
    std::vector<float>& context, int& context_len, const std::vector<float>& text_cond,
    int text_len, int current_chunk_len, int n_embd, int max_tokens) {
    if (current_chunk_len < 0 || current_chunk_len > text_len || n_embd <= 0 || context_len < 0 ||
        context.size() != (size_t)context_len * (size_t)n_embd ||
        text_cond.size() < (size_t)text_len * (size_t)n_embd) {
        std::fprintf(stderr, "invalid longform text context dimensions\n");
        return false;
    }
    const float* current =
        text_cond.data() + (size_t)(text_len - current_chunk_len) * (size_t)n_embd;
    context.insert(context.end(), current, current + (size_t)current_chunk_len * (size_t)n_embd);
    context_len += current_chunk_len;
    if (max_tokens > 0 && context_len > max_tokens) {
        context.erase(
            context.begin(),
            context.begin() +
                (std::ptrdiff_t)((size_t)(context_len - max_tokens) * (size_t)n_embd));
        context_len = max_tokens;
    }
    return true;
}

}  // namespace nemo_speech::tts
