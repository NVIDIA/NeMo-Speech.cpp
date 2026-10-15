/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <sparrowhawk/normalizer.h>

#include "speech_hints.h"

namespace nemotron_speech::utils {

/*
 * This Normalizer class exists as an abstraction over however we implement text normalization.
 *
 */

class Normalizer {
 private:
  std::unique_ptr<speech::sparrowhawk::Normalizer> normalizer_;
  std::unique_ptr<SpeechHints> speechhints_;
  std::string config_;
  std::string prefix_;

 public:
  bool check_speech_hints_ = false;
  bool enable_preprocess_ = true;
  bool enable_postprocess_ = true;
  Normalizer() = delete;

  /**
   * @brief original path into sparrowhawk normalization - requires a sparrowhawk config (config)
   * and the six files in prefix_
   *
   * @detail sparrowhawk expected files in the config_ directory - these have not changed in the
   * first year plus of deployment. sentence_boundary_exceptions.txt
   * sparrowhawk_configuration.ascii_proto tokenize_and_classify.far  tokenizer.ascii_proto
   * verbalize.far  verbalizer.ascii_proto
   *
   * @param config
   * @param prefix
   * @param speechhints
   */
  Normalizer(std::string config, std::string prefix, bool speech_hint = false);

  Normalizer(const std::string& prefix, bool speech_hint = false);

  /**
   * @brief normalizes the input provided and applies speech hints.
   *
   * @detail Applies speechhints and then normalizing the remaining segments. The output is a
   * normalized string with speechhints applied:
   * Input: i work at the b b c studios
   * Speechints: {"$OOV_ALPHA_SEQUENCE"}
   * Output: i work at the bbc studios
   *
   * @param input input string to normalize
   * @param output normalized string
   * @param classHints speech hints
   * @param verbatim_transcripts run only speech hints if verbatim_transcripts is true
   * @param enable_preprocessing apply the configured pre_process.far before ITN
   */
  bool Normalize(
      const std::string& input, std::string& output, const std::vector<std::string> classHints = {},
      const bool verbatim_transcripts = false, const bool enable_preprocessing = false) const;

  /**
   * @brief normalizes the input provided and adds the alignments to the location in the original
   * string.
   *
   * @detail Normalize input based on grammars specified and adds alignments.  The output is an
   * overloaded string:
   * Token:	0	what	0,3	0,0
   * Token:	1	are  5,7	1,1
   * Token:	2	the	9,11	2,2
   * Token:	3	alignments	13,22	3,3
   * Token:	4	here	24,27	4,4
   * Token:	5	four three two one	29,46	5,5
   * Word:	0	what	0
   * Word:	1	are	1
   * Word:	2	the	2
   * Word:	3	alignments	3
   * Word:	4	here	4
   * Word:	5	4321	5
   *
   * If we need alignments we should return the Utterance object this is dervied from or something
   * containing the words/token pieces vs a string
   *
   * @param input
   * @param output
   */
  bool NormalizeAndShowAlignments(
      const std::string& input, std::string& output, bool enable_preprocessing = false) const;

  /**
   * @brief Uses as many defaults as possible to load grammars from a directory
   *
   * @detail the only two files required for this load path are tokenize_and_classify.far and
   * verbalize.far, the others are still configurable, but are setup to use defaults. see the
   * sparrowhawk repo for more details.
   */
  bool LoadGrammar();
  /**
   * @brief requires all files to be present in the directory stored in prefix_.
   *
   * @detail sparrowhawk by default requires six files to stand up normalization, this function
   * expect all files to be present and properly setup in the directory prefix_.
   *
   */

  bool LoadFullGrammar();

  /**
   * @brief Checks if the speechhints vector has valid hints.
   *
   * @detail This is an wrapper around the AreHintsValid speech_hints.cc function to check if the
   * input speechhints vector has all the phrases as valid speechhints.
   *
   * @param speechhints vector of speechhints strings
   */
  bool AreHintsValid(const std::vector<std::string>& speechhints) const;
  /**
   * @brief Checks if the speechhint is a valid hint.
   *
   * @detail This is an wrapper around the IsHintValid speech_hints.cc function to check if the
   * input speechhint phrase is valid.
   *
   * @param speechhint string
   */
  bool IsHintValid(const std::string& speechhint) const;
  /**
   * @brief splits the input string based on the configuration of the sentence splitting regex as
   * passed in when constructing the normalization class.
   */
  std::vector<std::string> SentenceSplitter(const std::string& input) const
  {
    return normalizer_->SentenceSplitter(input);
  }
};

}  // namespace nemotron_speech::utils
