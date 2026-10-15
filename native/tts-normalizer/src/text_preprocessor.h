/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nemotron_speech::tts::preprocessing {

enum class PreprocessorErrorCode {
  kInvalidInput,
  kUnsupportedLanguage,
  kUnsupportedSsml,
  kMissingAsset,
  kInternal,
};

class TextPreprocessorException : public std::runtime_error {
 public:
  TextPreprocessorException(PreprocessorErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code)
  {
  }

  PreprocessorErrorCode code() const noexcept { return code_; }

 private:
  PreprocessorErrorCode code_;
};

struct LanguageConfig {
  std::string normalizer_path;
  std::string normalizer_config = "sparrowhawk_configuration.ascii_proto";
  std::string abbreviations_path;
  bool append_punctuation = true;
  // Load tokenize_and_classify.far and verbalize.far directly from
  // normalizer_path. This matches AggregatedTokenizer's non-Chinese path.
  bool direct_grammar = false;
};

struct TextPreprocessorConfig {
  std::unordered_map<std::string, LanguageConfig> languages;
  std::size_t max_input_bytes = 64 * 1024;
  std::size_t max_normalized_bytes = 256 * 1024;
  std::size_t max_segments = 256;
};

struct PreprocessOptions {
  /**
   * Normalize one caller-certified unit of a continuous text stream.
   *
   * Continuous mode deliberately does not run sentence splitting or append
   * punctuation. Boundary whitespace is retained canonically as at most one
   * ASCII space on either side of the returned segment. The caller must only
   * submit prefixes whose normalization cannot be changed by future text.
   */
  bool continuous_streaming = false;
};

struct TextSegment {
  std::string text;
  int32_t sentence_num = 0;
  bool is_last_sentence = false;
};

/**
 * CPU-only, model-agnostic TTS text preprocessing.
 *
 * The class deliberately returns UTF-8 text rather than model token IDs. It
 * does not depend on Triton, Torch, CUDA, G2P, character mappings, or Python.
 * Instances are immutable after construction but are not advertised as
 * thread-safe because the underlying WFST implementation has not established
 * a cross-platform concurrent-call contract.
 */
class TextPreprocessor {
 public:
  explicit TextPreprocessor(const TextPreprocessorConfig& config);
  ~TextPreprocessor();

  TextPreprocessor(const TextPreprocessor&) = delete;
  TextPreprocessor& operator=(const TextPreprocessor&) = delete;
  TextPreprocessor(TextPreprocessor&&) noexcept;
  TextPreprocessor& operator=(TextPreprocessor&&) noexcept;

  std::vector<TextSegment> Preprocess(
      const std::string& text, const std::string& language_code,
      const PreprocessOptions& options = {});

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron_speech::tts::preprocessing
