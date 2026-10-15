/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "normalize.h"

#include "absl/strings/str_replace.h"

namespace nemotron_speech::utils {

Normalizer::Normalizer(std::string config, std::string prefix, bool speech_hints)
    : config_(config), prefix_(prefix)
{
  normalizer_ = std::make_unique<speech::sparrowhawk::Normalizer>();
  if (speech_hints) {
    speechhints_ = std::make_unique<SpeechHints>();
    check_speech_hints_ = true;
  }
  if (!LoadFullGrammar()) {
    throw std::runtime_error("Error loading WFST grammar");
  }
}

Normalizer::Normalizer(const std::string& prefix, bool speech_hints) : prefix_(prefix)
{
  normalizer_ = std::make_unique<speech::sparrowhawk::Normalizer>();
  if (speech_hints) {
    speechhints_ = std::make_unique<SpeechHints>();
    check_speech_hints_ = true;
  }
  if (!LoadGrammar()) {
    throw std::runtime_error("Error loading WFST grammar");
  }
}

bool
Normalizer::LoadGrammar()
{
  struct stat filestat;
  if (-1 == stat(prefix_.c_str(), &filestat)) {
    LOG(ERROR) << "Grammar directory missing:" << prefix_ << std::endl;
    return false;
  }
  if (-1 == stat((prefix_ + "/speech_class.far").c_str(), &filestat)) {
    check_speech_hints_ = false;
  }
  if (-1 == stat((prefix_ + "/post_process.far").c_str(), &filestat)) {
    enable_postprocess_ = false;
  }
  if (-1 == stat((prefix_ + "/pre_process.far").c_str(), &filestat)) {
    enable_preprocess_ = false;
  }
  if (check_speech_hints_) {
    auto load_check = speechhints_->LoadGrammars((prefix_ + "/speech_class.far"));
    if (!load_check) {
      LOG(ERROR) << "Failed to load the grammar";
      return load_check;
    }
  }
  return normalizer_->Setup(prefix_, enable_postprocess_, enable_preprocess_);
}

bool
Normalizer::LoadFullGrammar()
{
  struct stat filestat;
  if (-1 == stat(prefix_.c_str(), &filestat) ||
      -1 == stat((prefix_ + "/" + config_).c_str(), &filestat)) {
    LOG(ERROR) << "Grammar files missing:" << prefix_ + "/" + config_ << std::endl;
    return false;
  }
  if (-1 == stat((prefix_ + "/speech_class.far").c_str(), &filestat)) {
    check_speech_hints_ = false;
  }
  if (check_speech_hints_) {
    auto load_check = speechhints_->LoadGrammars((prefix_ + "/speech_class.far"));
    if (!load_check) {
      LOG(ERROR) << "Failed to load the grammar";
      return load_check;
    }
  }

  return normalizer_->Setup(config_, prefix_);
}

// intentionally no sentence splitting as its pre punctuation
// remove any trailing whitepsace then reattach after normalizing.
bool
Normalizer::NormalizeAndShowAlignments(
    const std::string& input, std::string& output, bool enable_preprocessing) const
{
  int terminal_position = input.find_last_not_of(" ") + 1;
  std::string fixed_input = input.substr(0, terminal_position);
  std::string ws_substring = input.substr(terminal_position, input.length() - 1);
  fixed_input = absl::StrReplaceAll(fixed_input, {{"\"", "\\\" "}});
  auto rval = normalizer_->NormalizeAndShowLinks(fixed_input, &output, enable_preprocessing);
  if (!rval) {
    LOG(ERROR) << "normalizer failed for '" << fixed_input << "'. Returning output unchanged.";
    output = input;
    return rval;
  }

  if (fixed_input != input) {
    output += ws_substring;
  }

  // it is possible for the normalizer to inject unicode non-breaking spaces
  // we replace these with standard spaces if encountered
  output = absl::StrReplaceAll(output, {{"\xC2\xA0", " "}});

  // Strip lone trailing '<' characters left by SPARROWHAWK verbalizer grammars
  // (e.g. Korean verbalize.far leaves an unclosed token-class marker at end of sentence).
  // Only strip if the '<' has no matching '>'.
  while (!output.empty() && output.back() == '<') {
    output.pop_back();
  }

  return rval;
}

bool
Normalizer::IsHintValid(const std::string& speechhint) const
{
  if (check_speech_hints_) {
    return speechhints_->IsHintValid(speechhint);
  } else {
    LOG(ERROR) << "Speechhints not initiated!";
    return check_speech_hints_;
  }
}

bool
Normalizer::AreHintsValid(const std::vector<std::string>& speechhints) const
{
  if (check_speech_hints_) {
    return speechhints_->AreHintsValid(speechhints);
  } else {
    LOG(ERROR) << "Speechhints not initiated!";
    return check_speech_hints_;
  }
}

bool
Normalizer::Normalize(
    const std::string& input, std::string& output, const std::vector<std::string> classHints,
    const bool verbatim_transcripts, const bool enable_preprocessing) const
{
  if (input.empty()) {
    output.clear();
    return true;
  }
  static constexpr char kBoundaryWs[] = " \t\n\r\f\v";
  const auto first_non_space = input.find_first_not_of(kBoundaryWs);
  if (first_non_space == std::string::npos) {
    output = input;
    return true;
  }
  const auto terminal_position = input.find_last_not_of(kBoundaryWs) + 1;
  const std::string leading_ws = input.substr(0, first_non_space);
  const std::string trailing_ws = input.substr(terminal_position);
  std::string fixed_input = input.substr(first_non_space, terminal_position - first_non_space);
  fixed_input = absl::StrReplaceAll(fixed_input, {{"\"", "\\\" "}});
  bool result = false;
  if (check_speech_hints_) {
    Alignments alignments;
    if (!classHints.empty()) {
      result = speechhints_->ApplyHints(fixed_input, classHints, output, alignments);
      if (!result) {
        LOG(ERROR) << "normalizer failed to apply speechhints";
      }
      if (result && alignments.empty()) {
        LOG(ERROR) << "normalizer failed to apply speechhints, alignments empty";
        result = false;
      }
    }
    if (!result) {
      output = input;
    } else {
      NormalizedSegments segments;
      speechhints_->GetSegments(alignments, segments);
      if (segments.empty()) {
        output = leading_ws + fixed_input + trailing_ws;
        return false;
      }
      std::string final_output;
      for (auto& segment : segments) {
        if (!verbatim_transcripts && !segment.second &&
            !normalizer_->Normalize(segment.first, &segment.first, enable_preprocessing)) {
          LOG(ERROR) << "normalizer failed for segment, returning value unchanged";
        }
        final_output += (segment.first + " ");
      }
      output = final_output;
      boost::algorithm::trim(output);
    }
  }
  if (!check_speech_hints_ || !result) {
    auto rval = normalizer_->Normalize(fixed_input, &output, enable_preprocessing);
    if (!rval) {
      LOG(ERROR) << "normalizer failed, returning value unchanged";
      output = input;
      return rval;
    }
  }
  // A preprocessor may validly delete the entire input (for example, a final ASR segment that
  // consists only of a filler word). Do not turn that result back into boundary whitespace.
  if (output.empty()) {
    return true;
  }
  output = leading_ws + output + trailing_ws;
  output = absl::StrReplaceAll(output, {{"\xC2\xA0", " "}});
  return true;
}

}  // namespace nemotron_speech::utils
