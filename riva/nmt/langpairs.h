// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Language-pair table for Riva-Translate. Resolves a request to the model's
// pair tag (en-de, en-zh-cn, ...) and builds the prompt the model's chat
// template would produce, so a hand-built prompt matches what the embedded
// template emits.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace nemo_speech::nmt::langpairs {

// Normalize a client language code for this model: lowercase it, preserve a
// supported pair tag or region code, and otherwise fall back from BCP-47 to a
// supported base language (for example en-US -> en).
std::string normalize_language_code(std::string code);

// True if tag is a supported pair (e.g. "en-de").
bool is_supported(const std::string& tag);

// Resolve a request to a model tag. Accepts a ready tag passed in one field
// (with the other empty), or a "<source>-<target>" pair built from two codes.
// Returns "" if unsupported, or if both fields are set but disagree.
std::string resolve_tag(const std::string& source_language, const std::string& target_language);

// Split a tag into (source_code, target_code): "en-de" -> {"en","de"},
// "en-zh-cn" -> {"en","zh-cn"}. Every supported pair has English on one side.
std::pair<std::string, std::string> split_tag(const std::string& tag);

// Build the prompt for a supported tag and input text, matching the chat
// template with add_generation_prompt. Tokenize the result with
// add_special=false, parse_special=true. Returns "" if the tag is unsupported.
std::string build_prompt(const std::string& tag, const std::string& text);

// All supported pairs as (source_code, target_code), for the language-pair RPC.
const std::vector<std::pair<std::string, std::string>>& supported_pairs();

}  // namespace nemo_speech::nmt::langpairs
