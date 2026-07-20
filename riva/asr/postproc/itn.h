// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Inverse text normalization (spoken -> written, "twenty twenty three" ->
// "2023"). Wraps the Sparrowhawk WFST normalizer behind NEMO_SPEECH_WITH_NORM.
//
// Sparrowhawk uses OpenFST + per-language .far grammars; it is an optional
// dependency, so it is gated by a CMake option (default OFF). When the
// flag is OFF, this is a pass-through: normalize() returns its input. When ON,
// it loads the grammar dir and runs the WFST normalizer. Word-timing remap
// across the rewrite is handled separately in itn_align.h.
#pragma once

#include <memory>
#include <string>

namespace nemo_speech::asr::postproc {

class Itn {
   public:
    // `model_dir` = a Riva/Sparrowhawk grammar directory containing
    // tokenize_and_classify.far and verbalize.far. Empty = disabled.
    explicit Itn(const std::string& model_dir = "");
    ~Itn();

    bool enabled() const { return enabled_; }

    // Spoken-form -> written-form. Returns `text` unchanged when disabled or
    // when built without NEMO_SPEECH_WITH_NORM. When requested,
    // alignment_links describes this exact rewrite.
    std::string normalize(const std::string& text, std::string* alignment_links = nullptr) const;

    // Sparrowhawk ShowLinks alignment metadata for `input` (Token:/Word: lines
    // linking spoken text -> normalized words). Used by itn_align.h to project
    // word timings across the rewrite. Empty when disabled/unbuilt/on failure.
    std::string alignment(const std::string& input) const;

   private:
    // PIMPL is unconditional so the class layout is identical with or without
    // NEMO_SPEECH_WITH_NORM - consumers don't need the flag to match this ABI.
    // impl_ stays null in a non-ITN build (Impl is defined only there).
    bool enabled_ = false;
    struct Impl;  // wraps the Sparrowhawk normalizer; defined only in ITN builds
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::asr::postproc
