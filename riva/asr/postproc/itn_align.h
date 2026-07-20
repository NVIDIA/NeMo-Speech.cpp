// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Project word timings across an ITN rewrite using the Sparrowhawk alignment
// metadata (the Token:/Word: links from Normalizer::NormalizeAndShowLinks, via
// Itn::alignment()). Each Token carries the spoken text it matched and parents
// the normalized Words it produced; this reads that grouping and re-attaches the
// spoken words' spans to the written words.
//
// e.g. "twenty twenty three" (3 timed spoken words) -> "2023" (1 written word
// covering their combined span).
#pragma once

#include <string>
#include <vector>

#include "decoder.h"  // WordTiming

namespace nemo_speech::asr::postproc {

// In-place: rebuild `timings` to the written words from `alignment_links`,
// carrying the spoken spans across the rewrite. No-op if either is empty or the
// links don't parse.
void update_word_timings(std::vector<WordTiming>& timings, const std::string& alignment_links);

}  // namespace nemo_speech::asr::postproc
