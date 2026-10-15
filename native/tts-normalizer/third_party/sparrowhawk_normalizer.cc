// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2015 and onwards Google, Inc.
#include <sparrowhawk/normalizer.h>

#include <memory>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
using std::string;

#include <google/protobuf/text_format.h>
#include <sparrowhawk/items.pb.h>
#include <sparrowhawk/sentence_boundary.h>
#include <sparrowhawk/serialization_spec.pb.h>
#include <sparrowhawk/sparrowhawk_configuration.pb.h>
#include <sparrowhawk/io_utils.h>
#include <sparrowhawk/logger.h>
#include <sparrowhawk/protobuf_parser.h>
#include <sparrowhawk/protobuf_serializer.h>
#include <sparrowhawk/spec_serializer.h>
#include <sparrowhawk/string_utils.h>

namespace speech {
namespace sparrowhawk {
extern thread_local bool plain_string_normalization;
namespace {
class PlainStringScope {
 public:
  explicit PlainStringScope(bool enabled) : previous_(plain_string_normalization) {
    plain_string_normalization = enabled;
  }
  ~PlainStringScope() { plain_string_normalization = previous_; }
 private:
  bool previous_;
};

class StageTimer {
 public:
  StageTimer(const char* stage, const void* instance) : stage_(stage), instance_(instance) {
    static const bool enabled = std::getenv("NEMOTRON_TN_PROFILE") != nullptr;
    enabled_ = enabled;
    if (enabled_) {
      started_ = std::chrono::steady_clock::now();
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_started_);
    }
  }
  ~StageTimer() {
    if (!enabled_) return;
    const auto ended = std::chrono::steady_clock::now();
    timespec cpu_ended;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_ended);
    const double wall_us = std::chrono::duration<double, std::micro>(ended - started_).count();
    const double cpu_us = (cpu_ended.tv_sec - cpu_started_.tv_sec) * 1e6 +
                          (cpu_ended.tv_nsec - cpu_started_.tv_nsec) * 1e-3;
    std::fprintf(stderr, "TN_STAGE %s instance=%p wall_us=%.3f cpu_us=%.3f\n", stage_, instance_, wall_us, cpu_us);
  }
 private:
  const char* stage_;
  const void* instance_;
  bool enabled_;
  std::chrono::steady_clock::time_point started_;
  timespec cpu_started_;
};
}


const char kDefaultSentenceBoundaryRegexp[] = "[\\.:!\\?] ";

const std::string kDefaultVerbalizerProto =
  "grammar_file: \"verbalize.far\"\ngrammar_name: \"Verbalizer\"\nrules { main: \"ALL\" redup: \"REDUP\" }\n";
const std::string kDefaultTokenizerProto =
  "grammar_file: \"tokenize_and_classify.far\"\ngrammar_name: \"TokenizerClassifier\"\nrules { main: \"TOKENIZE_AND_CLASSIFY\" }\n";
const std::string kDefaultPostProcessorProto =
  "grammar_file: \"post_process.far\"\ngrammar_name: \"PostProcessor\"\nrules { main: \"POSTPROCESSOR\" }\n";
const std::string kDefaultPreProcessorProto =
  "grammar_file: \"pre_process.far\"\ngrammar_name: \"PreProcessor\"\nrules { main: \"PREPROCESSOR\" }\n";


std::vector<std::string> kDefaultSentenceBoundaryExceptions = {"Mr.", "Dr.", "Mrs.", "St.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec." };

Normalizer::Normalizer() { }

Normalizer::~Normalizer() { }


bool Normalizer::Setup(const string &pathname, bool post_process, bool pre_process)
{
  SparrowhawkConfiguration configuration;
  string proto_string = "tokenizer_grammar:  \"tokenizer.ascii_proto\"\n"
                        "verbalizer_grammar:  \"verbalizer.ascii_proto\"\n"
                        "sentence_boundary_regexp: \"[\\\\.:!\\\\?] \"\n"
    "sentence_boundary_exceptions_file: \"sentence_boundary_exceptions.txt\"\n";
  if (pre_process)
    proto_string += "preprocessor_grammar:  \"preprocessor.ascii_proto\"\n";
  if (post_process)
    proto_string += "postprocessor_grammar:  \"postprocessor.ascii_proto\"\n";
  LOG(INFO) << "Proto String: " << proto_string;
  try{
      if (!google::protobuf::TextFormat::ParseFromString(proto_string, &configuration))
        return false;
      }
  catch (...){
    LOG(ERROR) << "Failed to parse proto string" << proto_string;
    }

  tokenizer_classifier_rules_.reset(new RuleSystem);
  if (!tokenizer_classifier_rules_->LoadGrammarProtoFromString(
          kDefaultTokenizerProto,
          pathname))
    return false;

  verbalizer_rules_.reset(new RuleSystem);
  if (!verbalizer_rules_->LoadGrammarProtoFromString(kDefaultVerbalizerProto,
                                      pathname))
    return false;

  string sentence_boundary_regexp = kDefaultSentenceBoundaryRegexp;
  sentence_boundary_.reset(new SentenceBoundary(sentence_boundary_regexp));
  sentence_boundary_->AddSentenceBoundaryExceptions(kDefaultSentenceBoundaryExceptions);

  if (configuration.has_preprocessor_grammar()) {
    try{
      pre_processor_rules_.reset(new RuleSystem);
      if (pre_processor_rules_->LoadGrammarProtoFromString(
        kDefaultPreProcessorProto,
        pathname))
        this->do_preprocess = true;
      else
        LoggerWarn("Unable to load pre_processor_grammar from: ");
      }
    catch (...){
      LOG(ERROR) << "Failed to load preprocessor" << proto_string;
      }
    }
  if (configuration.has_postprocessor_grammar()) {
	try{
      post_processor_rules_.reset(new RuleSystem);
      if (post_processor_rules_->LoadGrammarProtoFromString(
        kDefaultPostProcessorProto,
        pathname))
        this->do_postprocess = true;
      else
        LoggerWarn("Unable to load post_processor_grammar");
      }
    catch (...){
      LOG(ERROR) << "Failed to load postprocessor" << proto_string;
      }
  }

  return true;
}

bool Normalizer::Setup(const string &configuration_proto,
                       const string &pathname_prefix) {
  SparrowhawkConfiguration configuration;
  string proto_string = IOStream::LoadFileToString(pathname_prefix +
                                                   "/" + configuration_proto);
  if (!google::protobuf::TextFormat::ParseFromString(proto_string, &configuration))
    return false;
  if (!(configuration.has_tokenizer_grammar()))
    LoggerError("Configuration does not define a tokenizer-classifier grammar");
  if (!(configuration.has_verbalizer_grammar()))
    LoggerError("Configuration does not define a verbalizer grammar");
  tokenizer_classifier_rules_.reset(new RuleSystem);
  if (!tokenizer_classifier_rules_->LoadGrammar(
          configuration.tokenizer_grammar(),
          pathname_prefix))
    return false;
  verbalizer_rules_.reset(new RuleSystem);
  if (!verbalizer_rules_->LoadGrammar(configuration.verbalizer_grammar(),
                                      pathname_prefix))
    return false;
  string sentence_boundary_regexp;
  if (configuration.has_sentence_boundary_regexp()) {
    sentence_boundary_regexp = configuration.sentence_boundary_regexp();
  } else {
    sentence_boundary_regexp = kDefaultSentenceBoundaryRegexp;
  }
  sentence_boundary_.reset(new SentenceBoundary(sentence_boundary_regexp));
  if (configuration.has_sentence_boundary_exceptions_file()) {
    if (!sentence_boundary_->LoadSentenceBoundaryExceptions(
            configuration.sentence_boundary_exceptions_file(), pathname_prefix)) {
      LoggerError("Cannot load sentence boundary exceptions file: %s",
                  configuration.sentence_boundary_exceptions_file().c_str());
    }
  }
  if (configuration.has_serialization_spec()) {
    string spec_string = IOStream::LoadFileToString(
        pathname_prefix + "/" + configuration.serialization_spec());
    SerializeSpec spec;
    if (spec_string.empty() ||
        !google::protobuf::TextFormat::ParseFromString(spec_string, &spec) ||
        (spec_serializer_ = Serializer::Create(spec)) == nullptr) {
      LoggerError("Failed to load a valid serialization spec from file: %s",
                  configuration.serialization_spec().c_str());
      return false;
    }
  }
  if (configuration.has_preprocessor_grammar()) {
    pre_processor_rules_.reset(new RuleSystem);
    if (pre_processor_rules_->LoadGrammar(
            configuration.preprocessor_grammar(),
            pathname_prefix))
      this->do_preprocess = true;
    else
      LoggerWarn("Unable to load pre_processor_grammar from: ");
  }
  if (configuration.has_postprocessor_grammar()) {
    post_processor_rules_.reset(new RuleSystem);
      if (post_processor_rules_->LoadGrammar(
              configuration.postprocessor_grammar(),
              pathname_prefix))
        this->do_postprocess = true;
    else
      LoggerWarn("Unable to load post_processor_grammar");
  }
  return true;
}


bool Normalizer::Normalize(
    const string &input, string *output, bool enable_preprocessing) const {
  PlainStringScope scope(true);
  std::unique_ptr<Utterance> utt;
  utt.reset(new Utterance);
  bool preprocessing_deleted_input = false;
  if (!Normalize(
          utt.get(), input, enable_preprocessing,
          &preprocessing_deleted_input)) {
    return false;
  }
  if (preprocessing_deleted_input) {
    output->clear();
    return true;
  }
  *output = LinearizeWords(utt.get());

  if (this->do_postprocess){
    const string pp_input = *output;
    StageTimer timer("postprocess", this);
    post_processor_rules_->ApplyRules(pp_input,output,false);
  }


  return true;
}

bool Normalizer::Normalize(
    Utterance *utt, const string &input, bool enable_preprocessing,
    bool *preprocessing_deleted_input) const {
  return TokenizeAndClassifyUtt(
             utt, input, enable_preprocessing,
             preprocessing_deleted_input) &&
         (*preprocessing_deleted_input || VerbalizeUtt(utt));
}

bool Normalizer::NormalizeAndShowLinks(
    const string &input, string *output, bool enable_preprocessing) const {
  PlainStringScope scope(false);
  std::unique_ptr<Utterance> utt;
  utt.reset(new Utterance);
  bool preprocessing_deleted_input = false;
  if (!Normalize(
          utt.get(), input, enable_preprocessing,
          &preprocessing_deleted_input)) {
    return false;
  }
  if (preprocessing_deleted_input) {
    output->clear();
    return true;
  }
  *output = ShowLinks(utt.get());
  return true;
}

bool Normalizer::TokenizeAndClassifyUtt(
    Utterance *utt, const string &input, bool enable_preprocessing,
    bool *preprocessing_deleted_input) const {
  StageTimer timer("classify", this);
  typedef fst::StringCompiler<fst::StdArc> Compiler;
  typedef fst::StringPrinter<fst::StdArc> Printer;
  Compiler compiler(fst::TokenType::BYTE);
  MutableTransducer input_fst;
  if (!compiler(input, &input_fst)) {
    LoggerError("Failed to compile input string \"%s\"", input.c_str());
    return false;
  }

  *preprocessing_deleted_input = false;
  MutableTransducer tokenizer_input(input_fst);
  if (this->do_preprocess && enable_preprocessing) {
    MutableTransducer preprocess_lattice;
    if (!pre_processor_rules_->ApplyRules(
            input_fst, &preprocess_lattice, false)) {
      return false;
    }





    fst::ShortestPath(preprocess_lattice, &tokenizer_input);

    MutableTransducer preprocessed_output(tokenizer_input);
    fst::Project(&preprocessed_output, fst::PROJECT_OUTPUT);
    fst::RmEpsilon(&preprocessed_output);
    Printer printer(fst::TokenType::BYTE);
    string preprocessed_text;
    if (!printer(preprocessed_output, &preprocessed_text)) {
      LoggerError("Failed to print preprocessed input");
      return false;
    }
    if (preprocessed_text.find_first_not_of(" \t\n\r\f\v") == string::npos) {
      *preprocessing_deleted_input = true;
      return true;
    }
  }

  MutableTransducer output;
  if (!tokenizer_classifier_rules_->ApplyRules(tokenizer_input,
                                               &output,
                                               true)) {
    LoggerError("Failed to tokenize \"%s\"", input.c_str());
    return false;
  }
  MutableTransducer shortest_path;
  fst::ShortestPath(output, &shortest_path);
  ProtobufParser parser(&shortest_path);
  if (!parser.ParseTokensFromFST(utt, true  )) {
    LoggerError("Failed to parse tokens from FST for \"%s\"", input.c_str());
    return false;
  }
  return true;
}








bool Normalizer::VerbalizeUtt(Utterance *utt) const {
  StageTimer timer("verbalize", this);
  for (int i = 0; i < utt->linguistic().tokens_size(); ++i) {
    Token *token = utt->mutable_linguistic()->mutable_tokens(i);
    string token_form = ToString(*token);
    token->set_first_daughter(-1);
    token->set_last_daughter(-1);



    if (token->type() == Token::PUNCT) {
      if (token->phrase_break() &&
          (utt->linguistic().words_size() == 0 ||
           utt->linguistic().words(
               utt->linguistic().words_size() - 1).id() != "sil")) {
        AddWord(utt, token, "sil");
      }
    } else if (token->type() == Token::SEMIOTIC_CLASS) {
      if (!token->skip()) {
        LoggerDebug("Verbalizing: [%s]\n", token_form.c_str());
        string words;
        if (VerbalizeSemioticClass(*token, &words)) {
          AddWords(utt, token, words);
        } else {
          LoggerWarn("First-pass verbalization FAILED for [%s]",
                     token_form.c_str());

          string original_token = token->name();
          token->Clear();
          token->set_name(original_token);
          token->set_verbatim(original_token);
          if (VerbalizeSemioticClass(*token, &words)) {
            LoggerWarn("Reversion to verbatim succeeded for [%s]",
                       original_token.c_str());
            AddWords(utt, token, words);
          } else {

            LoggerError("Verbalization FAILED for [%s]", token_form.c_str());
          }
        }
      }
    } else if (token->type() == Token::WORD) {
      if (token->has_wordid()) {
        AddWord(utt, token, token->wordid());
      } else {
        LoggerError("Token [%s] has type WORD but there is no word id",
                    token_form.c_str());
      }
    } else {
      LoggerError("No type found for [%s]", token_form.c_str());
    }
  }
  LoggerDebug("Verbalize output: Words\n%s\n\n", LinearizeWords(utt).c_str());
  return true;
}

bool Normalizer::VerbalizeSemioticClass(const Token &markup,
                                        string *words) const {
  Token local(markup);
  CleanFields(&local);
  MutableTransducer input_fst;
  if (spec_serializer_ == nullptr) {
    ProtobufSerializer serializer(&local, &input_fst);
    serializer.SerializeToFst();
  } else {
    input_fst = spec_serializer_->Serialize(local);
  }
  if (!verbalizer_rules_->ApplyRules(input_fst,
                                     words,
                                     false  )) {
    LoggerError("Failed to verbalize \"%s\"", ToString(local).c_str());
    return false;
  }
  return true;
}

std::vector<string> Normalizer::SentenceSplitter(const string &input) const {
  return sentence_boundary_->ExtractSentences(input);
}

}
}
