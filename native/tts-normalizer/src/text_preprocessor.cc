/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "text_preprocessor.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <pugixml.hpp>
#include <regex>
#include <sstream>
#include <utility>

#include "normalize.h"

namespace nemotron_speech::tts::preprocessing {
namespace {

class AbbreviationMapping {
 public:
  explicit AbbreviationMapping(const std::string& path)
  {
    std::ifstream input(path);
    if (!input.good()) {
      throw std::runtime_error("Unable to load requested abbreviations");
    }
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || !std::isalnum(static_cast<unsigned char>(line.front()))) {
        continue;
      }
      const auto separator = line.find(' ');
      if (separator == std::string::npos || separator + 1 >= line.size()) {
        continue;
      }
      const auto abbreviation = line.substr(0, separator);
      const auto expansion = line.substr(separator + 1);
      mappings_.emplace_back(
          std::regex(
              "\\b" + abbreviation + "(\\.|\\b)", std::regex_constants::ECMAScript |
                                                      std::regex_constants::optimize |
                                                      std::regex_constants::icase),
          expansion);
    }
    if (mappings_.empty()) {
      throw std::runtime_error("Abbreviation mapping is empty");
    }
  }

  std::string Transform(const std::string& input) const
  {
    std::string output = input;
    for (const auto& [pattern, expansion] : mappings_) {
      output = std::regex_replace(output, pattern, expansion);
    }
    return output;
  }

 private:
  std::vector<std::pair<std::regex, std::string>> mappings_;
};

bool
IsAsciiWhitespace(unsigned char c)
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::string
CollapseWhitespace(const std::string& input, bool preserve_boundary_whitespace = false)
{
  std::string output;
  output.reserve(input.size());
  bool pending_space = false;
  for (unsigned char c : input) {
    if (IsAsciiWhitespace(c)) {
      pending_space = preserve_boundary_whitespace || !output.empty();
      continue;
    }
    if (pending_space) {
      output.push_back(' ');
      pending_space = false;
    }
    output.push_back(static_cast<char>(c));
  }
  if (preserve_boundary_whitespace && pending_space && !output.empty() && output.back() != ' ') {
    output.push_back(' ');
  }
  return output;
}

std::string
Trim(const std::string& input)
{
  std::size_t begin = 0;
  while (begin < input.size() && IsAsciiWhitespace(static_cast<unsigned char>(input[begin]))) {
    ++begin;
  }
  std::size_t end = input.size();
  while (end > begin && IsAsciiWhitespace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(begin, end - begin);
}

bool
ContainsMarkup(const std::string& input)
{
  const std::string trimmed = Trim(input);
  if (trimmed.rfind("<?xml", 0) == 0) {
    if (trimmed.size() == 5) {
      return true;
    }
    const auto boundary = static_cast<unsigned char>(trimmed[5]);
    return IsAsciiWhitespace(boundary) || boundary == '?';
  }
  if (trimmed.rfind("<speak", 0) != 0 || trimmed.size() == 6) {
    return false;
  }
  const auto boundary = static_cast<unsigned char>(trimmed[6]);
  return IsAsciiWhitespace(boundary) || boundary == '>' || boundary == '/';
}

bool
IsValidUtf8(const std::string& input)
{
  std::size_t index = 0;
  while (index < input.size()) {
    const auto lead = static_cast<unsigned char>(input[index]);
    std::size_t continuation_count = 0;
    uint32_t codepoint = 0;
    if (lead <= 0x7f) {
      ++index;
      continue;
    } else if ((lead & 0xe0) == 0xc0) {
      continuation_count = 1;
      codepoint = lead & 0x1f;
    } else if ((lead & 0xf0) == 0xe0) {
      continuation_count = 2;
      codepoint = lead & 0x0f;
    } else if ((lead & 0xf8) == 0xf0) {
      continuation_count = 3;
      codepoint = lead & 0x07;
    } else {
      return false;
    }
    if (index + continuation_count >= input.size()) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
      const auto continuation = static_cast<unsigned char>(input[index + offset]);
      if ((continuation & 0xc0) != 0x80) {
        return false;
      }
      codepoint = (codepoint << 6) | (continuation & 0x3f);
    }
    if ((continuation_count == 1 && codepoint < 0x80) ||
        (continuation_count == 2 && codepoint < 0x800) ||
        (continuation_count == 3 && codepoint < 0x10000) ||
        (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

void
AppendSupportedSsml(const pugi::xml_node& node, std::string& output)
{
  for (const auto& child : node.children()) {
    if (child.type() == pugi::node_pcdata || child.type() == pugi::node_cdata) {
      output += child.value();
      continue;
    }
    if (child.type() == pugi::node_comment) {
      continue;
    }
    if (child.type() != pugi::node_element) {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kUnsupportedSsml, "Unsupported node in SSML input.");
    }

    const std::string name = child.name();
    if (name != "sub") {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kUnsupportedSsml,
          "Unsupported SSML tag <" + name + ">. Only <speak> and <sub alias> are supported.");
    }
    const auto alias = child.attribute("alias");
    if (!alias || std::string(alias.value()).empty()) {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kUnsupportedSsml,
          "SSML <sub> requires a non-empty alias attribute.");
    }
    for (const auto& nested : child.children()) {
      if (nested.type() == pugi::node_element) {
        throw TextPreprocessorException(
            PreprocessorErrorCode::kUnsupportedSsml,
            "Nested SSML tags are not supported inside <sub>.");
      }
    }
    output += alias.value();
  }
}

std::string
ExtractSpokenText(const std::string& input)
{
  if (!ContainsMarkup(input)) {
    return input;
  }

  pugi::xml_document document;
  const auto result =
      document.load_buffer(input.data(), input.size(), pugi::parse_default | pugi::parse_ws_pcdata);
  if (!result) {
    throw TextPreprocessorException(
        PreprocessorErrorCode::kUnsupportedSsml,
        std::string("Malformed SSML input: ") + result.description());
  }
  const auto root = document.document_element();
  if (!root || std::string(root.name()) != "speak") {
    throw TextPreprocessorException(
        PreprocessorErrorCode::kUnsupportedSsml, "SSML root element must be <speak>.");
  }
  for (const auto& attribute : root.attributes()) {
    const std::string name = attribute.name();
    if (name != "version" && name != "xml:lang" && name != "xmlns") {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kUnsupportedSsml,
          "Unsupported attribute on SSML <speak>: " + name);
    }
  }

  std::string output;
  AppendSupportedSsml(root, output);
  return output;
}

bool
HasSpokenContent(const std::string& input)
{
  for (unsigned char c : input) {
    if (c >= 0x80 || std::isalnum(c)) {
      return true;
    }
  }
  return false;
}

bool
PreviousCodePoint(const std::string& text, std::size_t& end, uint32_t& codepoint)
{
  if (end == 0 || end > text.size()) {
    return false;
  }

  std::size_t begin = end - 1;
  while (begin > 0 && (static_cast<unsigned char>(text[begin]) & 0xc0) == 0x80) {
    --begin;
  }
  const auto lead = static_cast<unsigned char>(text[begin]);
  if (lead <= 0x7f) {
    codepoint = lead;
  }
  else if ((lead & 0xe0) == 0xc0) {
    codepoint = lead & 0x1f;
  }
  else if ((lead & 0xf0) == 0xe0) {
    codepoint = lead & 0x0f;
  }
  else {
    codepoint = lead & 0x07;
  }
  for (std::size_t index = begin + 1; index < end; ++index) {
    codepoint = (codepoint << 6) | (static_cast<unsigned char>(text[index]) & 0x3f);
  }
  end = begin;
  return true;
}

bool
IsSentenceCloser(uint32_t codepoint)
{
  return codepoint == '\'' || codepoint == '"' || codepoint == ')' || codepoint == ']' ||
         codepoint == '}' || codepoint == 0x2019 || codepoint == 0x201d;
}

bool
IsTerminalPunctuation(uint32_t codepoint)
{
  // Keep the legacy semicolon/colon behavior and match the multilingual
  // sentence terminators recognized by the Rust streaming boundary scanner.
  return codepoint == '.' || codepoint == '!' || codepoint == '?' || codepoint == ';' ||
         codepoint == ':' || codepoint == 0x061f || codepoint == 0x06d4 ||
         codepoint == 0x0964 || codepoint == 0x0965 || codepoint == 0x3002 ||
         codepoint == 0xff01 || codepoint == 0xff1f;
}

bool
HasTerminalPunctuation(const std::string& text)
{
  std::size_t end = text.size();
  uint32_t codepoint = 0;
  while (PreviousCodePoint(text, end, codepoint)) {
    if (IsSentenceCloser(codepoint)) {
      continue;
    }
    return IsTerminalPunctuation(codepoint);
  }
  return false;
}

}  // namespace

class TextPreprocessor::Impl {
 public:
  explicit Impl(const TextPreprocessorConfig& config) : config_(config)
  {
    if (config_.languages.empty()) {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kUnsupportedLanguage,
          "At least one language must be configured for text preprocessing.");
    }
    if (config_.max_input_bytes == 0 || config_.max_normalized_bytes == 0 ||
        config_.max_segments == 0) {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kInvalidInput, "Preprocessor limits must be greater than zero.");
    }

    for (const auto& [language, language_config] : config_.languages) {
      if (language.empty()) {
        throw TextPreprocessorException(
            PreprocessorErrorCode::kUnsupportedLanguage,
            "Configured language code cannot be empty.");
      }
      auto resources = std::make_unique<LanguageResources>();
      resources->config = language_config;
      try {
        if (!language_config.normalizer_path.empty()) {
          const std::filesystem::path normalizer_path(language_config.normalizer_path);
          if (!std::filesystem::is_directory(normalizer_path)) {
            throw TextPreprocessorException(
                PreprocessorErrorCode::kMissingAsset,
                "Normalizer asset directory does not exist for " + language + ": " +
                    normalizer_path.string());
          }
          if (language_config.direct_grammar) {
            resources->normalizer =
                std::make_unique<nemotron_speech::utils::Normalizer>(normalizer_path.string(), false);
          }
          else {
            resources->normalizer = std::make_unique<nemotron_speech::utils::Normalizer>(
                language_config.normalizer_config, normalizer_path.string());
          }
        }
        if (!language_config.abbreviations_path.empty()) {
          if (!std::filesystem::is_regular_file(language_config.abbreviations_path)) {
            throw TextPreprocessorException(
                PreprocessorErrorCode::kMissingAsset, "Abbreviation asset does not exist for " +
                                                          language + ": " +
                                                          language_config.abbreviations_path);
          }
          resources->abbreviations =
              std::make_unique<AbbreviationMapping>(language_config.abbreviations_path);
        }
      }
      catch (const TextPreprocessorException&) {
        throw;
      }
      catch (const std::exception& error) {
        throw TextPreprocessorException(
            PreprocessorErrorCode::kMissingAsset,
            "Failed to initialize preprocessing assets for " + language + ": " + error.what());
      }
      resources_[language] = std::move(resources);
    }
  }

  std::vector<TextSegment> Preprocess(
      const std::string& input, const std::string& language_code,
      const PreprocessOptions& options)
  {
    const auto language_it = resources_.find(language_code);
    if (language_it == resources_.end()) {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kUnsupportedLanguage,
          "Unsupported preprocessing language: " + language_code);
    }
    if (input.empty() || input.size() > config_.max_input_bytes) {
      std::ostringstream message;
      message << "Input must contain 1-" << config_.max_input_bytes << " UTF-8 bytes.";
      throw TextPreprocessorException(PreprocessorErrorCode::kInvalidInput, message.str());
    }
    if (!IsValidUtf8(input)) {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kInvalidInput, "Input must be valid UTF-8.");
    }

    auto& resources = *language_it->second;
    std::string normalized = ExtractSpokenText(input);
    if (resources.normalizer) {
      std::string output;
      if (!resources.normalizer->Normalize(normalized, output)) {
        throw TextPreprocessorException(
            PreprocessorErrorCode::kInvalidInput,
            "WFST normalization failed for the supplied input.");
      }
      normalized = std::move(output);
    }
    if (resources.abbreviations) {
      normalized = resources.abbreviations->Transform(normalized);
    }
    normalized = CollapseWhitespace(normalized, options.continuous_streaming);
    if (!HasSpokenContent(normalized)) {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kInvalidInput, "Input must contain spoken text.");
    }
    if (normalized.size() > config_.max_normalized_bytes) {
      std::ostringstream message;
      message << "Normalized text exceeds " << config_.max_normalized_bytes << " UTF-8 bytes.";
      throw TextPreprocessorException(PreprocessorErrorCode::kInvalidInput, message.str());
    }

    if (options.continuous_streaming) {
      // A continuous request is already a caller-certified normalization unit.
      // Splitting or adding punctuation here would create model-visible tokens
      // that were not present in the append-only input stream.
      return {TextSegment{std::move(normalized), 0, true}};
    }

    std::vector<std::string> sentences;
    if (resources.normalizer) {
      sentences = resources.normalizer->SentenceSplitter(normalized);
    } else {
      sentences = {normalized};
    }

    std::vector<TextSegment> segments;
    segments.reserve(sentences.size());
    for (auto& sentence : sentences) {
      sentence = CollapseWhitespace(sentence);
      if (sentence.empty()) {
        continue;
      }
      if (resources.config.append_punctuation && !HasTerminalPunctuation(sentence)) {
        sentence.push_back('.');
      }
      segments.push_back(TextSegment{sentence, static_cast<int32_t>(segments.size()), false});
    }
    if (segments.empty()) {
      throw TextPreprocessorException(
          PreprocessorErrorCode::kInvalidInput, "Preprocessing produced no spoken text.");
    }
    if (segments.size() > config_.max_segments) {
      std::ostringstream message;
      message << "Preprocessing produced more than " << config_.max_segments << " segments.";
      throw TextPreprocessorException(PreprocessorErrorCode::kInvalidInput, message.str());
    }
    segments.back().is_last_sentence = true;
    return segments;
  }

 private:
  struct LanguageResources {
    LanguageConfig config;
    std::unique_ptr<nemotron_speech::utils::Normalizer> normalizer;
    std::unique_ptr<AbbreviationMapping> abbreviations;
  };

  TextPreprocessorConfig config_;
  std::unordered_map<std::string, std::unique_ptr<LanguageResources>> resources_;
};

TextPreprocessor::TextPreprocessor(const TextPreprocessorConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
}

TextPreprocessor::~TextPreprocessor() = default;
TextPreprocessor::TextPreprocessor(TextPreprocessor&&) noexcept = default;
TextPreprocessor& TextPreprocessor::operator=(TextPreprocessor&&) noexcept = default;

std::vector<TextSegment>
TextPreprocessor::Preprocess(
    const std::string& text, const std::string& language_code, const PreprocessOptions& options)
{
  try {
    return impl_->Preprocess(text, language_code, options);
  }
  catch (const TextPreprocessorException&) {
    throw;
  }
  catch (const std::exception& error) {
    throw TextPreprocessorException(PreprocessorErrorCode::kInternal, error.what());
  }
}

}  // namespace nemotron_speech::tts::preprocessing
