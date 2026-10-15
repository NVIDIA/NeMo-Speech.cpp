/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 */

#include "text_preprocessor.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace nemotron_speech::tts::preprocessing {
namespace {

TextPreprocessor
MakePlainPreprocessor()
{
  TextPreprocessorConfig config;
  config.languages.emplace("en-US", LanguageConfig{});
  return TextPreprocessor(config);
}

PreprocessOptions
ContinuousOptions()
{
  PreprocessOptions options;
  options.continuous_streaming = true;
  return options;
}

TEST(TextPreprocessorTest, NormalizesWhitespaceAndAppendsPunctuation)
{
  auto preprocessor = MakePlainPreprocessor();
  const auto segments = preprocessor.Preprocess("  Hello \n\t portable world  ", "en-US");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0].text, "Hello portable world.");
  EXPECT_EQ(segments[0].sentence_num, 0);
  EXPECT_TRUE(segments[0].is_last_sentence);
}

TEST(TextPreprocessorTest, PreservesMultilingualTerminalPunctuation)
{
  auto preprocessor = MakePlainPreprocessor();
  const std::vector<std::string> terminators = {
      ".", "!", "?", ";", ":", "؟", "۔", "।", "॥", "。", "！", "？"};

  for (const auto& terminator : terminators) {
    const std::string input = "Spoken text" + terminator;
    const auto segments = preprocessor.Preprocess(input, "en-US");
    ASSERT_EQ(segments.size(), 1) << "terminator: " << terminator;
    EXPECT_EQ(segments[0].text, input) << "terminator: " << terminator;
  }
}

TEST(TextPreprocessorTest, DetectsTerminalPunctuationBeforeSentenceClosers)
{
  auto preprocessor = MakePlainPreprocessor();
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"Spoken text.\"", "Spoken text.\""},
      {"Spoken text?']", "Spoken text?']"},
      {"Spoken text。”", "Spoken text。”"},
      {"Spoken text؟’", "Spoken text؟’"},
      {"\"Spoken text\"", "\"Spoken text\"."},
  };

  for (const auto& [input, expected] : cases) {
    const auto segments = preprocessor.Preprocess(input, "en-US");
    ASSERT_EQ(segments.size(), 1) << "input: " << input;
    EXPECT_EQ(segments[0].text, expected) << "input: " << input;
  }
}

TEST(TextPreprocessorTest, PreservesUnicodeTerminatorAfterTrailingWhitespaceIsCollapsed)
{
  auto preprocessor = MakePlainPreprocessor();
  const auto segments = preprocessor.Preprocess("Spoken text। \n\t", "en-US");

  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0].text, "Spoken text।");
}

TEST(TextPreprocessorTest, ContinuousModeReturnsOneSegmentWithoutAppendingPunctuation)
{
  auto preprocessor = MakePlainPreprocessor();
  const auto segments = preprocessor.Preprocess(
      "First unit. Second unit without a terminator", "en-US", ContinuousOptions());

  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0].text, "First unit. Second unit without a terminator");
  EXPECT_EQ(segments[0].sentence_num, 0);
  EXPECT_TRUE(segments[0].is_last_sentence);
}

TEST(TextPreprocessorTest, ContinuousModeKeepsWfstAndAbbreviationsButSkipsSentenceSplitting)
{
  TextPreprocessorConfig config;
  LanguageConfig language;
  language.normalizer_path = "testdata/en-US";
  language.abbreviations_path = "testdata/en-US/abbr.txt";
  config.languages.emplace("en-US", std::move(language));
  TextPreprocessor preprocessor(config);

  const auto segments = preprocessor.Preprocess(
      "Dr Jones spent $2 today. Second unit", "en-US", ContinuousOptions());

  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0].text, "doctor Jones spent two dollars today. Second unit");
}

TEST(TextPreprocessorTest, DirectGrammarModeMatchesLegacyNonChineseNormalizerLoading)
{
  TextPreprocessorConfig config;
  LanguageConfig language;
  language.normalizer_path = "testdata/en-US";
  language.direct_grammar = true;
  config.languages.emplace("en-US", std::move(language));
  TextPreprocessor preprocessor(config);

  const auto segments =
      preprocessor.Preprocess("I spent $2 today", "en-US", ContinuousOptions());

  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0].text, "I spent two dollars today");
}

TEST(TextPreprocessorTest, ContinuousModeCanonicalizesBoundaryWhitespace)
{
  auto preprocessor = MakePlainPreprocessor();
  const auto segments = preprocessor.Preprocess(
      " \tHello\n\tcontinuous world \r\n", "en-US", ContinuousOptions());

  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0].text, " Hello continuous world ");
}

TEST(TextPreprocessorTest, LegacyModeRemainsSentenceOrientedAndPunctuated)
{
  auto preprocessor = MakePlainPreprocessor();
  const auto segments = preprocessor.Preprocess("  Legacy behavior  ", "en-US");

  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0].text, "Legacy behavior.");
}

TEST(TextPreprocessorTest, ExpandsSupportedSubSsml)
{
  auto preprocessor = MakePlainPreprocessor();
  const auto segments = preprocessor.Preprocess(
      "<speak>Visit the <sub alias=\"World Wide Web\">WWW</sub></speak>", "en-US");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0].text, "Visit the World Wide Web.");
}

TEST(TextPreprocessorTest, TreatsLeadingLessThanTextAsPlainText)
{
  auto preprocessor = MakePlainPreprocessor();
  const std::vector<std::string> inputs = {"< 3 is spoken", "<tag is spoken", "<speaker is spoken"};

  for (const auto& input : inputs) {
    const auto segments = preprocessor.Preprocess(input, "en-US");
    ASSERT_EQ(segments.size(), 1) << "input: " << input;
    EXPECT_EQ(segments[0].text, input + ".") << "input: " << input;
  }
}

TEST(TextPreprocessorTest, RejectsUnsupportedSsml)
{
  auto preprocessor = MakePlainPreprocessor();
  try {
    static_cast<void>(
        preprocessor.Preprocess("<speak><prosody rate=\"slow\">Hello</prosody></speak>", "en-US"));
    FAIL() << "Expected TextPreprocessorException";
  }
  catch (const TextPreprocessorException& error) {
    EXPECT_EQ(error.code(), PreprocessorErrorCode::kUnsupportedSsml);
  }
}

TEST(TextPreprocessorTest, RejectsUnsupportedLanguage)
{
  auto preprocessor = MakePlainPreprocessor();
  EXPECT_THROW(preprocessor.Preprocess("Hello", "fr-FR"), TextPreprocessorException);
}

TEST(TextPreprocessorTest, RejectsPunctuationOnlyInput)
{
  auto preprocessor = MakePlainPreprocessor();
  EXPECT_THROW(preprocessor.Preprocess("...?!", "en-US"), TextPreprocessorException);
}

TEST(TextPreprocessorTest, RejectsInvalidUtf8)
{
  auto preprocessor = MakePlainPreprocessor();
  const std::string invalid("\xc3\x28", 2);
  EXPECT_THROW(preprocessor.Preprocess(invalid, "en-US"), TextPreprocessorException);
}

TEST(TextPreprocessorTest, ContinuousModeRejectsInvalidUtf8)
{
  auto preprocessor = MakePlainPreprocessor();
  const std::string invalid("\xc3\x28", 2);
  EXPECT_THROW(
      preprocessor.Preprocess(invalid, "en-US", ContinuousOptions()), TextPreprocessorException);
}

TEST(TextPreprocessorTest, EnforcesConfiguredLimits)
{
  TextPreprocessorConfig config;
  config.max_input_bytes = 4;
  config.languages.emplace("en-US", LanguageConfig{});
  TextPreprocessor preprocessor(config);
  EXPECT_THROW(preprocessor.Preprocess("hello", "en-US"), TextPreprocessorException);
}

TEST(TextPreprocessorTest, ContinuousModeEnforcesInputAndNormalizedLimits)
{
  TextPreprocessorConfig input_limited_config;
  input_limited_config.max_input_bytes = 4;
  input_limited_config.languages.emplace("en-US", LanguageConfig{});
  TextPreprocessor input_limited(input_limited_config);
  EXPECT_THROW(
      input_limited.Preprocess("hello", "en-US", ContinuousOptions()),
      TextPreprocessorException);

  TextPreprocessorConfig output_limited_config;
  output_limited_config.max_normalized_bytes = 4;
  output_limited_config.languages.emplace("en-US", LanguageConfig{});
  TextPreprocessor output_limited(output_limited_config);
  EXPECT_THROW(
      output_limited.Preprocess("hello", "en-US", ContinuousOptions()),
      TextPreprocessorException);
}

}  // namespace
}  // namespace nemotron_speech::tts::preprocessing
