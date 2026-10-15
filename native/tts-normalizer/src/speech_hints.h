/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <fst/arc.h>
#include <fst/bi-table.h>
#include <fst/compat.h>
#include <fst/expanded-fst.h>
#include <fst/extensions/far/far.h>
#include <fst/filter-state.h>
#include <fst/fst.h>
#include <fst/fstlib.h>
#include <fst/state-table.h>
#include <fst/string.h>
#include <thrax/abstract-grm-manager.h>
#include <thrax/compat/compat.h>
#include <thrax/compat/utils.h>
#include <thrax/grm-manager.h>
#include <thrax/stringfst.h>
#include <thrax/symbols.h>

#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>

#ifndef NEMOTRON_SPEECH_SPEECH_HINTS_H
#define NEMOTRON_SPEECH_SPEECH_HINTS_H

namespace nemotron_speech::utils {

using thrax::GrmManager;
typedef fst::VectorFst<fst::StdArc> MutableTransducer;
typedef fst::StringCompiler<fst::StdArc> Compiler;
typedef std::vector<std::pair<std::string, std::string>> Alignments;
typedef std::vector<std::pair<std::string, bool>> NormalizedSegments;
typedef boost::tokenizer<boost::char_separator<char>> tokenizer;


/**
 * @brief SpeechHints class used for supporting speech hints based normalization.
 */
class SpeechHints {
 public:
  SpeechHints();


  /**
   * @brief Loads class grammars for speech hints
   *
   * @param[in] far_filename Filename with path of the far file to be loaded for speech hints.
   *                         FAR should contain "$__PASSTHROUGH__" and "$__SPACE__" fsts at a
   * minimum
   *
   * @return True if grammar far files for speech hints is loaded successfully
   */
  bool LoadGrammars(const std::string& far_filename);

  /**
   * @brief Apply normalization on the string using the provided speech hints
   *
   * @param[in] input String to be normalized
   * @param[in] speech_hints List of speech hints to be applied
   * @param[out] output Normalized string
   * @param[out] alignments Alignments between the input word(s) and the normalized segments
   *
   * @return True if normalization is successful
   */
  bool ApplyHints(
      const std::string& input, const std::vector<std::string>& speech_hints, std::string& output,
      Alignments& alignments);

  /**
   * @brief Apply normalization on the string using the provided speech hint
   *
   * @param[in] input String to be normalized
   * @param[in] speech_hint Speech hint to be applied
   * @param[out] output Normalized string
   * @param[out] alignments Alignments between the input word(s) and the normalized output segments
   *
   * @return True if normalization is successful
   */
  bool ApplyHint(
      const std::string& input, const std::string& speech_hint, std::string& output,
      Alignments& alignments);

  /**
   * @brief Given alignments, split text into a list of text segments. Segments contain text and
   * bool (true if segment is normalized)
   *
   * @param[in] alignments
   * @param[out] segments
   */
  void GetSegments(Alignments alignments, NormalizedSegments& segments);

  /**
   * @brief Given alignments, split text into a list of text segments.
   *
   * @param[in] alignments
   * @param[out] segments
   */
  void Alignments2Segments(Alignments alignments, std::vector<std::string>& segments);

  /**
   * @brief Get list of valid grammars present in the far file
   *
   * @return  list of valid grammars
   */
  std::vector<std::string> GetGrammarList(void);

  /**
   * @brief Checks if speech hints contains valid grammars
   *
   * @param speech_hints List of speech hints
   * @return True if all grammars in speech hints are valid
   */
  bool AreHintsValid(const std::vector<std::string>& speech_hints);

  /**
   * @brief Checks if speech hint contains valid grammars
   *
   * @param speech_hints Speech hint
   * @return True if all grammars in speech hints are valid
   */
  bool IsHintValid(const std::string& speech_hint);

 private:
  const std::string space_grammar = "$__SPACE__";
  const std::string passthrough_grammar = "$__PASSTHROUGH__";
  boost::char_separator<char> word_sep;
  std::unique_ptr<GrmManager> sh_class_;
  bool grammar_loaded = false;
  std::vector<std::string> valid_grammars;

  MutableTransducer space_fst_, space_sigma_;
  MutableTransducer sent_fst_;
  bool StringFstToOutputLabels(
      const MutableTransducer& ifst, std::vector<fst::StdArc::Label>* labels);
  /**
   * @brief Converts a given word to an fst
   *
   * @param word word that has to be converted to an FST
   * @return FST for word
   */
  MutableTransducer Word2Fst(const std::string& word);

  /**
   * @brief Given a speech hint class name, returns the FST for that class
   *
   * @param[in] sh_class_name Name of the speech hint class
   * @param[out] word_fst FST for the speech hint class
   * @return True if speech hint class exists
   */
  bool GetClassFst(const std::string& sh_class_name, MutableTransducer& word_fst);

  /**
   * @brief Converts the provided speech hint into an FST
   *
   * @param[in] hint Speech hint string
   * @return MutableTransducer containing the speech hint FST
   */
  MutableTransducer Hint2Fst(const std::string& hint);

  /**
   * @brief Apply normalization on the string using the provided speech hint
   *
   * @param[in] input String to be normalized
   * @param[in] speech_hint Speech hint(s) (FST) to be applied
   * @param[out] output Normalized string
   * @param[out] alignments Alignments between the input word(s) and the normalized output segments
   *
   * @return True if normalization is successful
   */
  bool ApplyHintFst(
      const std::string& input, MutableTransducer speech_hint, std::string& output,
      Alignments& alignments);


  /**
   * @brief Given an FST (obtained after applying the speech hints FST on input FST, returns the
   * string and the alignments
   *
   * @param[in] ofst FST obtained after composing input FST with speech hints FST
   * @param[out] output Normalized string
   * @param[out] alignments Alignments between the input word(s) and the normalized output segments
   *
   * @return True if shortest path is obtained
   */
  bool StringifyFst(MutableTransducer& ofst, std::string& output, Alignments& alignments);
};

}  // namespace nemotron_speech::utils

#endif  // NEMOTRON_SPEECH_SPEECH_HINTS_H
