// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Decoding guards of the SALM engine: the Latin-script token mask and the loop guard that ends
// runaway output.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace nemo_speech::speechlm::guards {

// ---- script guard
// Language tags whose transcripts are written in Latin script ("en", "en-US", "pt_BR", ...): the
// script guard applies.
bool latin_script_language(std::string_view language);
// Whether a request's decode keeps to Latin script: its output language is written in Latin script
// and a served prompt (transcription, translation or speaker tags) asks for it. A client's
// instruction sets the output language itself (e.g. "translate into Japanese"), so it never does.
bool script_guard(std::string_view output_language, bool client_instruction);
// True when every letter and mark of the UTF-8 text is Latin (digits, punctuation, symbols and
// spaces pass); false for invalid or incomplete UTF-8 (a token holding part of a character) and for
// U+FFFD.
bool is_latin_script(std::string_view utf8);

// ---- loop guard
// The smallest period p (1..32) the last tokens repeat with over the window p needs, else 0.
// Periods 4-12 need max(15, 4p) tokens (a phrase said four times) unless a divisor period explains
// the window; the others need max(32, 4p).
int loop_period(const std::vector<int>& tokens, size_t n);

// One sequence's guard. Fires when the output repeats with a short period (checked every 4 tokens
// from the 15th) or when at least 90% of the last 256 16-token n-grams already appeared earlier in
// the output.
class LoopGuard {
   public:
    // Commit token `t`; true when the sequence must end after it. The check covers the tokens
    // before `t`, so the newest token is kept, and the guard stays fired.
    bool push(int t);
    void reset();

   private:
    bool fires(size_t n);
    bool replay(size_t n);

    std::vector<int> tokens_;
    size_t last_check_ = 0;
    bool fired_ = false;
    // replay
    std::vector<uint64_t> hashes_;
    std::unordered_set<uint64_t> seen_;
    std::deque<char> recent_;
    int repeats_ = 0;
    size_t scanned_ = 0;
};

}  // namespace nemo_speech::speechlm::guards
