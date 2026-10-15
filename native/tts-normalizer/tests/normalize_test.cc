/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

#include "normalize.h"

#include <fst/extensions/far/far.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nemotron_speech::utils {
namespace {

class DirectNormalizerTest : public testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    normalizer_ = std::make_unique<Normalizer>("testdata/en-US");
  }

  static void TearDownTestSuite() { normalizer_.reset(); }

  inline static std::unique_ptr<Normalizer> normalizer_;
};

TEST_F(DirectNormalizerTest, NormalizesTextAndPreservesBoundaryWhitespace)
{
  std::string output;

  EXPECT_TRUE(normalizer_->Normalize(" \tI spent $2 today\r\n", output));
  EXPECT_EQ(output, " \tI spent two dollars today\r\n");
}

TEST_F(DirectNormalizerTest, EmptyAndWhitespaceOnlyInputsRemainStable)
{
  std::string output = "stale output";

  EXPECT_TRUE(normalizer_->Normalize("", output));
  EXPECT_TRUE(output.empty());
  EXPECT_TRUE(normalizer_->Normalize(" \t\r\n", output));
  EXPECT_EQ(output, " \t\r\n");
}

TEST_F(DirectNormalizerTest, DisablesSpeechHintsWhenTheArchiveIsAbsent)
{
  EXPECT_FALSE(normalizer_->IsHintValid("$OOV_ALPHA_SEQUENCE"));
  EXPECT_FALSE(normalizer_->AreHintsValid({"$OOV_ALPHA_SEQUENCE"}));
}

TEST_F(DirectNormalizerTest, PreservesPunctuationAndLiteralSil)
{
  for (const std::string input : {
           "Hello, world, this is a test.", "Stop. Look. Listen.", "The word sil remains sil."}) {
    std::string output;
    EXPECT_TRUE(normalizer_->Normalize(input, output));
    EXPECT_EQ(output, input);
  }
}

TEST(NormalizerTest, PreservesCommasInsideVerbalizedDates)
{
  const auto directory = std::filesystem::path(testing::TempDir()) / "date-comma-grammar";
  std::filesystem::create_directories(directory);
  std::filesystem::copy_file(
      "testdata/en-US/verbalize.far", directory / "verbalize.far",
      std::filesystem::copy_options::overwrite_existing);

  // The deployed date grammar embeds the comma inside the year field.
  const std::string input = "July 28, 1914";
  const std::string tagged =
      "tokens { date { month: \"july\" day: \"twenty eight\" "
      "year: \", nineteen fourteen\" preserve_order: true } }";
  fst::VectorFst<fst::StdArc> tagger;
  auto state = tagger.AddState();
  tagger.SetStart(state);
  for (size_t i = 0; i < tagged.size(); ++i) {
    const auto next = tagger.AddState();
    tagger.AddArc(state, fst::StdArc(
                            i < input.size() ? input[i] : 0, tagged[i],
                            fst::StdArc::Weight::One(), next));
    state = next;
  }
  tagger.SetFinal(state, fst::StdArc::Weight::One());
  {
    std::unique_ptr<fst::FarWriter<fst::StdArc>> writer(
        fst::FarWriter<fst::StdArc>::Create((directory / "tokenize_and_classify.far").string()));
    ASSERT_NE(writer, nullptr);
    writer->Add("TOKENIZE_AND_CLASSIFY", tagger);
    ASSERT_FALSE(writer->Error());
  }

  Normalizer normalizer(directory.string());
  std::string output;
  ASSERT_TRUE(normalizer.Normalize(input, output));
  EXPECT_EQ(output, "july twenty eighth , nineteen fourteen");
}

TEST(NormalizerTest, FullGrammarConfigurationNormalizesText)
{
  Normalizer normalizer(
      std::string("sparrowhawk_configuration.ascii_proto"), std::string("testdata/en-US"));
  std::string output;

  EXPECT_TRUE(normalizer.Normalize("I spent $2 today", output));
  EXPECT_EQ(output, "I spent two dollars today");
}

TEST(NormalizerTest, MissingGrammarDirectoryIsRejected)
{
  EXPECT_THROW(Normalizer("testdata/missing-normalizer"), std::runtime_error);
}

}  // namespace
}  // namespace nemotron_speech::utils
