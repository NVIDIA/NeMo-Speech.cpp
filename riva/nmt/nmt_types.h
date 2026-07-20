// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Plain NMT value types shared by the library and its transports, proto-free so
// the translator, the gRPC service, and any future C ABI speak one vocabulary.
//
// Named nmt_types.h (not types.h) so it never collides with riva/asr/types.h:
// the NMT speech cascades pull both the asr and nmt include dirs into one
// translation unit, where an unqualified "types.h" would otherwise resolve to
// whichever dir the build happens to search first.
#pragma once

#include <string>
#include <vector>

namespace nemo_speech::nmt {

// One translated text plus the target language it was produced in.
struct Translation {
    std::string text;
    std::string language;
};

}  // namespace nemo_speech::nmt
