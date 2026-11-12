// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Longform text-context cache: a short chunk followed by a longer one must still find the
// history the longer chunk's window asks for.
#include <cstdio>
#include <vector>

#include "tts/magpietts/longform_context.h"

namespace tts = nemo_speech::tts;

namespace {

constexpr int kEmbd = 3;

bool
expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return false;
    }
    return true;
}

// Encoder output for a window whose current chunk starts at absolute token `first_current`:
// every row is tagged with a value that identifies (pass, absolute token).
std::vector<float>
encode_window(int history_len, int current_len, int first_current, float pass) {
    std::vector<float> rows;
    for (int i = -history_len; i < current_len; ++i) {
        for (int d = 0; d < kEmbd; ++d) {
            rows.push_back(pass * 1000.0f + (float)(first_current + i) + 0.1f * (float)d);
        }
    }
    return rows;
}

float
row_value(const std::vector<float>& rows, int row) {
    return rows[(size_t)row * kEmbd];
}

// Chunks of 30, 8 and 25 tokens with the default history of min(prior, current, 20): the 8-token
// chunk has an 8-token history (16-row window), and the 25-token chunk then needs 20 rows of
// history. Caching only the previous window failed with "need 20 token(s), have 16".
bool
short_chunk_then_long_chunk() {
    std::vector<float> cache;
    int cache_len = 0;
    const int n_ctx = 2048;
    bool ok = true;

    // Chunk 0: 30 tokens, no history.
    std::vector<float> c0 = encode_window(0, 30, 0, 1);
    ok &= expect(tts::splice_longform_history_context(c0, 30, 30, kEmbd, cache, cache_len), "c0");
    ok &= expect(
        tts::append_longform_history_context(cache, cache_len, c0, 30, 30, kEmbd, n_ctx),
        "append c0");

    // Chunk 1: 8 tokens with 8 tokens of history (tokens 22..29).
    std::vector<float> c1 = encode_window(8, 8, 30, 2);
    ok &= expect(tts::splice_longform_history_context(c1, 16, 8, kEmbd, cache, cache_len), "c1");
    ok &= expect(row_value(c1, 0) == 1000.0f + 22.0f, "c1 history comes from chunk 0's context");
    ok &= expect(
        tts::append_longform_history_context(cache, cache_len, c1, 16, 8, kEmbd, n_ctx),
        "append c1");
    ok &= expect(cache_len == 38, "cache holds all 38 synthesized tokens");

    // Chunk 2: 25 tokens with 20 tokens of history (tokens 18..37).
    std::vector<float> c2 = encode_window(20, 25, 38, 3);
    ok &= expect(
        tts::splice_longform_history_context(c2, 45, 25, kEmbd, cache, cache_len),
        "c2 finds 20 history tokens after a 16-row window");
    ok &= expect(row_value(c2, 0) == 1000.0f + 18.0f, "token 18 keeps its chunk 0 context");
    ok &= expect(row_value(c2, 12) == 2000.0f + 30.0f, "token 30 keeps its chunk 1 context");
    ok &= expect(row_value(c2, 20) == 3000.0f + 38.0f, "current rows are untouched");
    return ok;
}

// When the history fits in the previous window, the rolling cache splices exactly what caching
// that window alone did.
bool
matches_previous_window_cache() {
    std::vector<float> cache;
    int cache_len = 0;
    std::vector<float> c0 = encode_window(0, 12, 0, 1);
    tts::append_longform_history_context(cache, cache_len, c0, 12, 12, kEmbd, 2048);
    std::vector<float> c1 = encode_window(10, 10, 12, 2);
    tts::splice_longform_history_context(c1, 20, 10, kEmbd, cache, cache_len);
    const std::vector<float> previous_window = c1;
    tts::append_longform_history_context(cache, cache_len, c1, 20, 10, kEmbd, 2048);

    std::vector<float> from_cache = encode_window(10, 14, 22, 3);
    std::vector<float> from_window = from_cache;
    bool ok = expect(
        tts::splice_longform_history_context(from_cache, 24, 14, kEmbd, cache, cache_len),
        "splice from rolling cache");
    ok &= expect(
        tts::splice_longform_history_context(from_window, 24, 14, kEmbd, previous_window, 20),
        "splice from previous window");
    ok &= expect(from_cache == from_window, "rolling cache splices the same rows");
    return ok;
}

bool
cache_is_capped() {
    std::vector<float> cache;
    int cache_len = 0;
    bool ok = true;
    for (int chunk = 0; chunk < 5; ++chunk) {
        std::vector<float> rows = encode_window(0, 10, chunk * 10, 1);
        ok &= expect(
            tts::append_longform_history_context(cache, cache_len, rows, 10, 10, kEmbd, 32),
            "append");
    }
    ok &= expect(cache_len == 32 && cache.size() == (size_t)32 * kEmbd, "capped at 32 tokens");
    ok &= expect(row_value(cache, 0) == 1000.0f + 18.0f, "oldest tokens are dropped first");
    ok &= expect(row_value(cache, 31) == 1000.0f + 49.0f, "newest token is last");
    return ok;
}

// The chunk prefetch copies only the newest history rows of the cache for its thread. Splicing
// from that copy must give the same rows as splicing from the whole cache.
bool
tail_splices_the_same_rows() {
    std::vector<float> cache;
    int cache_len = 0;
    std::vector<float> c0 = encode_window(0, 30, 0, 1);
    tts::append_longform_history_context(cache, cache_len, c0, 30, 30, kEmbd, 2048);
    std::vector<float> c1 = encode_window(8, 8, 30, 2);
    tts::splice_longform_history_context(c1, 16, 8, kEmbd, cache, cache_len);
    tts::append_longform_history_context(cache, cache_len, c1, 16, 8, kEmbd, 2048);

    const std::vector<float> tail = tts::longform_history_context_tail(cache, cache_len, 20, kEmbd);
    bool ok = expect(tail.size() == (size_t)20 * kEmbd, "tail holds the 20 newest rows");
    std::vector<float> from_cache = encode_window(20, 25, 38, 3);
    std::vector<float> from_tail = from_cache;
    ok &= expect(
        tts::splice_longform_history_context(from_cache, 45, 25, kEmbd, cache, cache_len),
        "splice from the whole cache");
    ok &= expect(
        tts::splice_longform_history_context(from_tail, 45, 25, kEmbd, tail, 20),
        "splice from the tail");
    ok &= expect(from_cache == from_tail, "the tail splices the same rows");

    ok &= expect(
        tts::longform_history_context_tail(cache, cache_len, 100, kEmbd) == cache,
        "a tail longer than the cache is the whole cache");
    ok &= expect(
        tts::longform_history_context_tail(cache, cache_len, 0, kEmbd).empty(),
        "no history rows gives an empty tail");
    return ok;
}

bool
rejects_inconsistent_cache() {
    std::vector<float> cache(5 * kEmbd, 0.0f);
    int cache_len = 4;  // size does not match
    std::vector<float> rows = encode_window(0, 2, 0, 1);
    return expect(
        !tts::append_longform_history_context(cache, cache_len, rows, 2, 2, kEmbd, 2048),
        "mismatched cache length is rejected");
}

}  // namespace

int
main() {
    bool ok = true;
    ok &= short_chunk_then_long_chunk();
    ok &= matches_previous_window_cache();
    ok &= cache_is_capped();
    ok &= tail_splices_the_same_rows();
    ok &= rejects_inconsistent_cache();
    if (!ok) {
        return 1;
    }
    std::printf("test_magpietts_longform_context: OK\n");
    return 0;
}
