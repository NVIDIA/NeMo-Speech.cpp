/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

#include "speech_hints.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace nemotron_speech::utils {
namespace {

TEST(SpeechHintsTest, RejectsOperationsBeforeGrammarLoading)
{
  SpeechHints hints;
  std::string output = "unchanged";
  Alignments alignments = {{"input", "output"}};

  EXPECT_TRUE(hints.GetGrammarList().empty());
  EXPECT_FALSE(hints.IsHintValid("$OOV_ALPHA_SEQUENCE"));
  EXPECT_FALSE(hints.AreHintsValid({"$OOV_ALPHA_SEQUENCE"}));
  EXPECT_FALSE(hints.ApplyHint("hello", "$OOV_ALPHA_SEQUENCE", output, alignments));
  EXPECT_EQ(output, "unchanged");
  EXPECT_EQ(alignments, Alignments({{"input", "output"}}));
  EXPECT_FALSE(hints.ApplyHints("hello", {"$OOV_ALPHA_SEQUENCE"}, output, alignments));
  EXPECT_EQ(output, "unchanged");
  EXPECT_EQ(alignments, Alignments({{"input", "output"}}));
}

TEST(SpeechHintsTest, SegmentConversionIsANoOpBeforeGrammarLoading)
{
  SpeechHints hints;
  const Alignments alignments = {{"plain", "plain"}, {"spoken", "normalized"}};
  NormalizedSegments normalized_segments = {{"existing", false}};
  std::vector<std::string> segments = {"existing"};

  hints.GetSegments(alignments, normalized_segments);
  hints.Alignments2Segments(alignments, segments);

  EXPECT_EQ(normalized_segments, NormalizedSegments({{"existing", false}}));
  EXPECT_EQ(segments, std::vector<std::string>({"existing"}));
}

}  // namespace
}  // namespace nemotron_speech::utils
