// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "fst_normalizer.h"

#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>

// Must precede Sparrowhawk headers: bridges them to OpenFST 1.8.
// clang-format off
#include "sparrowhawk_compat.h"
#include "sparrowhawk/normalizer.h"
// clang-format on

namespace nemo_speech::text_normalization {
namespace {

struct PreparedInput {
    std::string text;
    std::string trailing_spaces;
};

PreparedInput
prepare_input(const std::string& input) {
    const size_t last = input.find_last_not_of(' ');
    if (last == std::string::npos)
        return {"", input};
    PreparedInput prepared{input.substr(0, last + 1), input.substr(last + 1)};
    // Sparrowhawk's byte grammar expects escaped quotes, matching Riva.
    for (size_t pos = 0; (pos = prepared.text.find('"', pos)) != std::string::npos; pos += 3)
        prepared.text.replace(pos, 1, "\\\" ");
    return prepared;
}

void
replace_nbsp(std::string* text) {
    static const std::string kNbsp = "\xC2\xA0";
    for (size_t pos = 0; (pos = text->find(kNbsp, pos)) != std::string::npos; ++pos)
        text->replace(pos, kNbsp.size(), " ");
}

}  // namespace

struct FstNormalizer::Impl {
    speech::sparrowhawk::Normalizer normalizer;
    mutable std::mutex mu;

    explicit Impl(const std::string& model_dir) {
        std::error_code ec;
        const std::filesystem::path dir = std::filesystem::canonical(model_dir, ec);
        if (ec || !std::filesystem::is_directory(dir))
            throw std::runtime_error("text normalization: cannot resolve model dir: " + model_dir);
        for (const char* required : {"tokenize_and_classify.far", "verbalize.far"}) {
            if (!std::filesystem::is_regular_file(dir / required)) {
                throw std::runtime_error(
                    "text normalization: missing grammar: " + (dir / required).string());
            }
        }
        const bool post_process = std::filesystem::is_regular_file(dir / "post_process.far");
        const bool pre_process = std::filesystem::is_regular_file(dir / "pre_process.far");
        if (!normalizer.Setup(dir.string(), post_process, pre_process)) {
            throw std::runtime_error(
                "text normalization: Sparrowhawk Setup failed for " + dir.string());
        }
    }
};

FstNormalizer::FstNormalizer(const std::string& model_dir)
    : impl_(std::make_unique<Impl>(model_dir)) {}
FstNormalizer::~FstNormalizer() = default;

std::string
FstNormalizer::normalize(const std::string& input, std::string* alignment_links) const {
    if (alignment_links)
        alignment_links->clear();
    const PreparedInput prepared = prepare_input(input);
    if (prepared.text.empty())
        return input;

    std::lock_guard<std::mutex> lock(impl_->mu);
    std::string output;
    if (!impl_->normalizer.Normalize(prepared.text, &output)) {
        std::cerr << "[text_normalization] WARNING: rewrite failed, returning text unchanged "
                  << "(redacted, " << input.size() << " bytes)\n";
        return input;
    }
    if (alignment_links &&
        !impl_->normalizer.NormalizeAndShowLinks(prepared.text, alignment_links)) {
        alignment_links->clear();
    }
    replace_nbsp(&output);
    output += prepared.trailing_spaces;
    return output;
}

std::string
FstNormalizer::alignment(const std::string& input) const {
    const PreparedInput prepared = prepare_input(input);
    if (prepared.text.empty())
        return "";
    std::lock_guard<std::mutex> lock(impl_->mu);
    std::string links;
    if (impl_->normalizer.NormalizeAndShowLinks(prepared.text, &links))
        return links;
    return "";
}

}  // namespace nemo_speech::text_normalization
