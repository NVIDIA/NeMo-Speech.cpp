// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "itn.h"

#include <iostream>
#ifdef NEMO_SPEECH_WITH_NORM
#include "fst_normalizer.h"
#endif

namespace nemo_speech::asr::postproc {

#ifdef NEMO_SPEECH_WITH_NORM
// Loads the Sparrowhawk config + grammars from `model_dir` and normalizes per
// request.
struct Itn::Impl {
    text_normalization::FstNormalizer normalizer;
    explicit Impl(const std::string& model_dir) : normalizer(model_dir) {}
};
#else
// Non-ITN build: Impl must still be a complete type so unique_ptr<Impl> in ~Itn
// compiles. impl_ is never constructed here (enabled_ stays false).
struct Itn::Impl {};
#endif

Itn::Itn(const std::string& model_dir) {
    if (model_dir.empty())
        return;
#ifdef NEMO_SPEECH_WITH_NORM
    impl_ = std::make_unique<Impl>(model_dir);
    enabled_ = true;
#else
    std::cerr << "[itn] WARNING: --itn-model-dir set (" << model_dir
              << ") but this build has no ITN support "
                 "(configure with -DNEMO_SPEECH_WITH_NORM=ON). ITN disabled.\n";
#endif
}

Itn::~Itn() = default;

std::string
Itn::normalize(const std::string& text, std::string* alignment_links) const {
    if (alignment_links != nullptr)
        alignment_links->clear();
    if (!enabled_)
        return text;
#ifdef NEMO_SPEECH_WITH_NORM
    return impl_->normalizer.normalize(text, alignment_links);
#else
    static_cast<void>(alignment_links);
#endif
    return text;
}

std::string
Itn::alignment(const std::string& input) const {
    if (!enabled_)
        return "";
#ifdef NEMO_SPEECH_WITH_NORM
    return impl_->normalizer.alignment(input);
#else
    static_cast<void>(input);
#endif
    return "";
}

}  // namespace nemo_speech::asr::postproc
