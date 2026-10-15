/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "speech_hints.h"
namespace nemotron_speech::utils {

SpeechHints::SpeechHints()
{
  word_sep = boost::char_separator<char>(" \t,.!?;./\"(){}[]<>");
}

bool
SpeechHints::StringFstToOutputLabels(
    const MutableTransducer& ifst, std::vector<fst::StdArc::Label>* labels)
{
  labels->clear();
  auto s = ifst.Start();

  if (s == fst::kNoStateId) {
    LOG(ERROR) << "StringFstToOutputLabels: Invalid start state";
    return false;
  }

  while (ifst.Final(s) == fst::StdArc::Weight::Zero()) {
    fst::ArcIterator<MutableTransducer> aiter(ifst, s);
    if (aiter.Done()) {
      LOG(ERROR) << "StringFstToOutputLabels: Does not reach final state";
      return false;
    }

    const auto& arc = aiter.Value();
    labels->push_back(arc.olabel);
    s = arc.nextstate;
    aiter.Next();

    if (!aiter.Done()) {
      LOG(ERROR) << "StringFstToOutputLabels: State " << s << " has multiple outgoing arcs";
      return false;
    }
  }

  if (ifst.NumArcs(s) != 0) {
    LOG(ERROR) << "StringFstToOutputLabels: Final state " << s << " has outgoing arc(s)";
    return false;
  }
  return true;
}


bool
SpeechHints::LoadGrammars(const std::string& far_filename)
{
  sh_class_.reset(new GrmManager);
  if (!sh_class_->LoadArchive(far_filename)) {
    LOG(ERROR) << "Unable to load grammars from " << far_filename << std::endl;
    return false;
  } else {
    for (auto& i : *sh_class_->GetFstMap()) {
      if ((i.first != space_grammar) && (i.first != passthrough_grammar))
        valid_grammars.push_back(i.first);
    }
    if (!GetClassFst(passthrough_grammar, sent_fst_)) {
      LOG(ERROR) << "Missing passthrough grammar" << far_filename << std::endl;
      return false;
    }
    if (!GetClassFst(space_grammar, space_fst_)) {
      LOG(ERROR) << "Missing whitespace grammar" << far_filename << std::endl;
      return false;
    } else {
      space_sigma_ = *space_fst_.Copy(true);
      fst::Closure(&space_sigma_, fst::CLOSURE_STAR);
    }
  }
  grammar_loaded = true;
  return true;
}

std::vector<std::string>
SpeechHints::GetGrammarList()
{
  if (grammar_loaded) {
    return valid_grammars;
  }
  return {};
}

MutableTransducer
SpeechHints::Word2Fst(const std::string& word)
{
  Compiler str2fst(fst::TokenType::BYTE);
  MutableTransducer word_fst;

  if ((word == passthrough_grammar) || (word == space_grammar)) {
    LOG(ERROR) << "Cannot specify one of '" << passthrough_grammar << "' or '" << space_grammar
               << " in speech hints";
  } else if ((!GetClassFst(word, word_fst)) && (!str2fst.operator()(word, &word_fst))) {
    // Get class fst if word is class else convert word to fst
    LOG(ERROR) << "Unable to create fst for word ' " << word << "'" << std::endl;
  }
  return word_fst;
}

bool
SpeechHints::GetClassFst(const std::string& sh_class_name, MutableTransducer& word_fst)
{
  if ((sh_class_->GetFstMap()->count(sh_class_name) == 0)) {
    if (sh_class_name[0] == '$')
      LOG(ERROR) << "Invalid speech class - " << sh_class_name;
    return false;
  }
  auto it = sh_class_->GetFstMap()->find(sh_class_name);

  /*Copy the speech hint fst for the specified class. Since this will be composed with other fsts,
  needs to be a copy. Refer https://www.openfst.org/twiki/bin/view/FST/FstConventions */
  word_fst = *it->second->Copy(true);

  return true;
}

MutableTransducer
SpeechHints::Hint2Fst(const std::string& hint)
{
  MutableTransducer hint_fst;
  if ((hint != passthrough_grammar) && (hint != space_grammar)) {
    if (!GetClassFst(hint, hint_fst)) {
      tokenizer word_tokenizer(hint, word_sep);
      tokenizer::iterator word = word_tokenizer.begin();
      hint_fst = Word2Fst(word.current_token());
      std::advance(word, 1);
      for (; word != word_tokenizer.end(); ++word) {
        MutableTransducer word_fst = Word2Fst(word.current_token());
        fst::Concat(&hint_fst, space_sigma_);
        fst::Concat(&hint_fst, word_fst);
      }
    }
  } else {
    LOG(ERROR) << "Cannot specify one of '" << passthrough_grammar << "' or '" << space_grammar
               << " in speech hints";
  }


  return hint_fst;
}

bool
SpeechHints::StringifyFst(MutableTransducer& ofst, std::string& output, Alignments& alignments)
{
  output.clear();
  alignments.clear();
  MutableTransducer temp;
  fst::ShortestPath(ofst, &temp);
  if (temp.Start() == ::fst::kNoStateId)
    return false;
  MutableTransducer ifst(*temp.Copy());
  fst::Invert(&ifst);
  std::vector<fst::StdArc::Label> ilabels;
  std::vector<fst::StdArc::Label> olabels;
  std::vector<fst::StdArc::Label>::iterator it, ot;
  StringFstToOutputLabels(ifst, &ilabels);
  StringFstToOutputLabels(temp, &olabels);
  std::string iword;
  std::string oword;

  for (it = ilabels.begin(), ot = olabels.begin(); it != ilabels.end() && ot != olabels.end();
       ++it, ++ot) {
    if ((*it == ' ') && (*ot == ' ')) {
      boost::algorithm::trim(iword);
      boost::algorithm::trim(oword);
      if (iword != "" || oword != "")
        alignments.push_back(std::make_pair(iword, oword));
      output += oword + ' ';
      iword = "";
      oword = "";


    } else {
      if (*it != 0)
        iword.push_back(*it);
      if (*ot != 0)
        oword.push_back(*ot);
    }
  }
  boost::algorithm::trim(output);
  return true;
}

bool
SpeechHints::AreHintsValid(const std::vector<std::string>& speech_hints)
{
  if (!grammar_loaded)
    return false;

  for (auto hint : speech_hints) {
    if (!IsHintValid(hint)) {
      return false;
    }
  }
  return true;
}

bool
SpeechHints::IsHintValid(const std::string& speech_hint)
{
  if (!grammar_loaded)
    return false;
  tokenizer word_tokenizer(speech_hint, word_sep);
  for (auto word : word_tokenizer) {
    if ((word == passthrough_grammar) || (word == space_grammar))
      return false;
    if (word[0] == '$') {
      if (std::find(valid_grammars.begin(), valid_grammars.end(), word) == valid_grammars.end()) {
        return false;
      }
    }
  }
  return true;
}

bool
SpeechHints::ApplyHint(
    const std::string& input, const std::string& speech_hint, std::string& output,
    Alignments& alignments)
{
  if (!grammar_loaded) {
    return false;
  }

  MutableTransducer sh_fst;
  fst::Union(&sh_fst, sent_fst_);
  fst::Union(&sh_fst, space_sigma_);
  std::string speech_hint_ = speech_hint;
  boost::algorithm::trim(speech_hint_);
  if (!speech_hint_.empty())
    fst::Union(&sh_fst, Hint2Fst(speech_hint_));

  fst::Closure(&sh_fst, fst::CLOSURE_STAR);

  return ApplyHintFst(input, sh_fst, output, alignments);
}

bool
SpeechHints::ApplyHints(
    const std::string& input, const std::vector<std::string>& speech_hints, std::string& output,
    Alignments& alignments)
{
  if (!grammar_loaded)
    return false;
  MutableTransducer hint_fst, sh_fst = sent_fst_;
  fst::Union(&sh_fst, space_fst_);

  if (!speech_hints.empty()) {
    std::vector<std::string>::const_iterator speech_hint = speech_hints.begin();
    std::string speech_hint_ = *speech_hint;
    boost::algorithm::trim(speech_hint_);
    if (!speech_hint_.empty())
      fst::Union(&sh_fst, Hint2Fst(speech_hint_));

    std::advance(speech_hint, 1);
    for (; speech_hint != speech_hints.end(); ++speech_hint) {
      speech_hint_ = *speech_hint;
      boost::algorithm::trim(speech_hint_);

      if (!speech_hint_.empty())
        fst::Union(&sh_fst, Hint2Fst(speech_hint_));
    }
  }

  fst::Closure(&sh_fst, fst::CLOSURE_STAR);

  return ApplyHintFst(input, sh_fst, output, alignments);
}

bool
SpeechHints::ApplyHintFst(
    const std::string& input, MutableTransducer speech_hint, std::string& output,
    Alignments& alignments)
{
  MutableTransducer input_fst, output_fst;
  std::string input_pad = " " + input + " ";
  Compiler str2fst(fst::TokenType::BYTE);
  if (!str2fst.operator()(input_pad, &input_fst)) {
    LOG(INFO) << "Unable to create fst for input ' " << input << "'";
  }
  fst::ComposeOptions opts(true, fst::ALT_SEQUENCE_FILTER);

  fst::Compose(input_fst, speech_hint, &output_fst, opts);
  if (!StringifyFst(output_fst, output, alignments)) {
    output = input;

    return false;
  }

  return true;
}

void
SpeechHints::GetSegments(Alignments alignments, NormalizedSegments& segments)
{
  if (!grammar_loaded)
    return;
  std::string segment;

  for (auto it : alignments) {
    if (it.first == it.second) {
      segment = segment + it.first + " ";
    } else {
      boost::algorithm::trim(segment);
      if (!segment.empty()) {
        segments.push_back(std::make_pair(segment, false));
      }
      segments.push_back(std::make_pair(it.second, true));
      segment = "";
    }
  }
  boost::algorithm::trim(segment);
  if (segments.empty()) {
    if (!segment.empty()) {
      segments.push_back(std::make_pair(segment, false));
    }
  } else if (segment != segments.back().first && !segment.empty()) {
    segments.push_back(std::make_pair(segment, false));
  }
}

void
SpeechHints::Alignments2Segments(Alignments alignments, std::vector<std::string>& segments)
{
  if (!grammar_loaded)
    return;
  std::string segment;
  for (auto it : alignments) {
    if (it.first == it.second) {
      segment = segment + it.first + " ";
    } else {
      boost::algorithm::trim(segment);
      if (!segment.empty()) {
        segments.push_back(segment);
      }
      segments.push_back(it.second);
      segment = "";
    }
  }
  boost::algorithm::trim(segment);
  if (segments.empty()) {
    if (!segment.empty()) {
      segments.push_back(segment);
    }
  } else if (segment != segments.back() && !segment.empty()) {
    segments.push_back(segment);
  }
}
}  // namespace nemotron_speech::utils
