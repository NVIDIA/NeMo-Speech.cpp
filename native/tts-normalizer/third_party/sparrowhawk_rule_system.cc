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
#include <sparrowhawk/rule_system.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include "sparrowhawk_string_matcher.h"

#include <google/protobuf/text_format.h>
#include <sparrowhawk/io_utils.h>
#include <sparrowhawk/logger.h>
#include <sparrowhawk/items.pb.h>
#include <sparrowhawk/protobuf_parser.h>

namespace speech {
namespace sparrowhawk {
thread_local bool plain_string_normalization = false;
namespace {
std::mutex matcher_mutex;
std::map<const RuleSystem*, std::map<string, std::unique_ptr<StringMatcher>>> string_matchers;

StringMatcher* FindStringMatcher(const RuleSystem* owner, const string& name,
                                const Transducer& grammar) {
  std::lock_guard<std::mutex> lock(matcher_mutex);
  auto& matcher = string_matchers[owner][name];
  if (!matcher) {
    StringMatcher::Limits limits;
    limits.canonical_word_outputs = true;
    matcher.reset(new StringMatcher(grammar, limits));
  }
  return matcher.get();
}

bool HasStrictWordSyntax(const MutableTransducer& path) {
  string text;
  auto state = path.Start();
  for (size_t steps = 0; steps <= path.NumStates(); ++steps) {
    fst::ArcIterator<MutableTransducer> arcs(path, state);
    if (arcs.Done()) {
      if (path.Final(state) == fst::StdArc::Weight::Zero()) return false;
      break;
    }
    const auto arc = arcs.Value();
    arcs.Next();
    if (!arcs.Done() || arc.olabel < 0 || arc.olabel > 255) return false;
    if (arc.olabel) text.push_back(static_cast<char>(arc.olabel));
    state = arc.nextstate;
    if (steps == path.NumStates()) return false;
  }
  size_t position = 0;
  auto whitespace = [&]() {
    while (position < text.size() && text[position] == ' ') ++position;
  };
  auto literal = [&](const string& value) {
    whitespace();
    if (text.compare(position, value.size(), value) != 0) return false;
    position += value.size();
    return true;
  };
  size_t tokens = 0;
  while (true) {
    whitespace();
    if (position == text.size()) return tokens != 0;
    if (!literal("tokens") || !literal("{") || !literal("name") ||
        !literal(":") || !literal("\"")) return false;
    bool closed = false;
    while (position < text.size()) {
      const auto byte = static_cast<unsigned char>(text[position++]);
      if (byte == '"') { closed = true; break; }
      if (byte == '\\') {
        if (position == text.size()) return false;
        ++position;
      }
    }
    if (!closed || !literal("}")) return false;
    ++tokens;
  }
}

bool IsExplicitWordPath(const MutableTransducer& path) {
  if (path.Start() == fst::kNoStateId || !HasStrictWordSyntax(path)) return false;
  Utterance utterance;
  ProtobufParser parser(&path);
  if (!parser.ParseTokensFromFST(&utterance, true) ||
      utterance.linguistic().tokens_size() == 0) return false;
  for (const auto& token : utterance.linguistic().tokens()) {
    if (token.type() != Token::WORD || !token.has_wordid() || !token.has_name())
      return false;
    std::vector<const google::protobuf::FieldDescriptor*> fields;
    token.GetReflection()->ListFields(token, &fields);
    for (const auto* field : fields) {
      const auto& name = field->name();
      if (name != "name" && name != "wordid" && name != "start_index" &&
          name != "end_index") return false;
    }
  }
  return true;
}

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



using fst::LabelLookAheadRelabeler;
using fst::StdArc;

RuleSystem::~RuleSystem() {
  {
    std::lock_guard<std::mutex> lock(matcher_mutex);
    string_matchers.erase(this);
  }
  std::map<string, LookaheadFst*>::iterator iter;
  for (iter = lookaheads_.begin(); iter != lookaheads_.end(); iter++) {
    delete iter->second;
  }
}

bool RuleSystem::LoadGrammar(const string& filename, const string& prefix) {

  std::string sep = "";
  if (prefix[prefix.size() - 1] != '/') {
    sep = "/";
  }


  string proto_string = IOStream::LoadFileToString(prefix + sep + filename);
  return LoadGrammarProtoFromString(proto_string, prefix + sep);
}

bool RuleSystem::LoadGrammarProtoFromString(const string& proto_string, const string& prefix)
{
  {
    std::lock_guard<std::mutex> lock(matcher_mutex);
    string_matchers.erase(this);
  }
  std::string sep = "";
  if (prefix[prefix.size() - 1] != '/') {
    sep = "/";
  }

  if (!google::protobuf::TextFormat::ParseFromString(proto_string, &grammar_))
    return false;
  string grm_file = prefix + sep+ grammar_.grammar_file();
  grammar_name_ = grammar_.grammar_name();
  grm_.reset(new GrmManager);
  if (!grm_->LoadArchive(grm_file)) {
    LoggerError("Error loading archive \"%s\" from \"%s\"",
                grammar_name_.c_str(), grm_file.c_str());
    return false;
  }


  for (int i = 0; i < grammar_.rules_size(); ++i) {
    Rule rule = grammar_.rules(i);
    if (grm_->GetFst(rule.main()) == NULL) {
      LoggerError("Rule \"%s\" not found in \"%s\"",
                  rule.main().c_str(), grammar_name_.c_str());
      return false;
    }
    if (rule.has_parens() && grm_->GetFst(rule.parens()) == NULL) {
      LoggerError("Rule \"%s\" not found in \"%s\"",
                  rule.parens().c_str(), grammar_name_.c_str());
      return false;
    }
    if (rule.has_redup() && grm_->GetFst(rule.redup()) == NULL) {
      LoggerError("Rule \"%s\" not found in \"%s\"",
                  rule.redup().c_str(), grammar_name_.c_str());
      return false;
    }
  }
  return true;
}

bool RuleSystem::ApplyRules(const Transducer& input,
                            MutableTransducer* output,
                            bool use_lookahead) const {
  StageTimer all_timer(grammar_name_.c_str(), this);
  if (std::getenv("NEMOTRON_TN_STRING_MATCHER") != nullptr && plain_string_normalization && use_lookahead &&
      grammar_.rules_size() == 1 && !grammar_.rules(0).has_redup() &&
      !grammar_.rules(0).has_parens()) {
    const auto& name = grammar_.rules(0).main();
    auto* matcher = FindStringMatcher(this, name, *grm_->GetFst(name));
    MutableTransducer candidate;
    const bool matched = matcher->Match(input, &candidate);
    const bool accepted = matched && IsExplicitWordPath(candidate);
    if (std::getenv("NEMOTRON_TN_PROFILE") != nullptr) {
      const auto& d = matcher->diagnostics();
      std::fprintf(stderr, "TN_MATCH reason=%s states=%zu arcs=%zu nodes=%zu best=%.9g second=%.9g guard=%.9g\n",
                   accepted ? "matched_word_only" : matched ? "non_word_path" : d.reason,
                   d.visited_states, d.arc_relaxations, d.backpointers,
                   d.best_cost, d.second_cost, d.ambiguity_guard);
    }
    if (accepted) {
      *output = std::move(candidate);
      return true;
    }
  }
  MutableTransducer mutable_input(input);
  for (int i = 0; i < grammar_.rules_size(); ++i) {
    Rule rule = grammar_.rules(i);
    if (rule.has_redup()) {
      const string& redup_rule = rule.redup();
      MutableTransducer redup1;

      if (grm_->Rewrite(redup_rule, mutable_input, &redup1, "")) {
        MutableTransducer redup2(redup1);
        fst::Concat(redup1, &redup2);
        fst::Union(&mutable_input, redup2);
        fst::RmEpsilon(&mutable_input);
      }
    }
    const string& rule_name = rule.main();
    string parens_rule = rule.has_parens() ? rule.parens() : "";

    bool success = true;
    if (parens_rule.empty()
        && use_lookahead) {
      std::map<string, LookaheadFst*>::iterator iter =
          lookaheads_.find(rule_name);
      LookaheadFst *lookahead_rule_fst;
      if (iter == lookaheads_.end()) {
        const Transducer *rule_fst = grm_->GetFst(rule_name);
        lookahead_rule_fst = new LookaheadFst(*rule_fst);
        lookaheads_[rule_name] = lookahead_rule_fst;
      } else {
        lookahead_rule_fst = iter->second;
      }
      LabelLookAheadRelabeler<StdArc>::Relabel(&mutable_input,
                                               *lookahead_rule_fst,
                                               false);
      fst::ComposeFst<StdArc> tmp_output(mutable_input,
                                             *lookahead_rule_fst);
      *output = tmp_output;
      if (output->NumStates() == 0) {
        success = false;
      }

    } else if (!grm_->Rewrite(rule_name,
                              mutable_input,
                              output,
                              parens_rule
                              )
        || output->NumStates() == 0) {
      success = false;
    }
    if (!success) {
      LoggerError("Application of rule \"%s\" failed", rule_name.c_str());
      return false;
    }
    if (i + 1 < grammar_.rules_size()) mutable_input = *output;
  }



  { StageTimer epsilon_timer("rm_epsilon", this); fst::RmEpsilon(output); }
  return true;
}

typedef fst::StringCompiler<StdArc> Compiler;
typedef fst::StringPrinter<StdArc> Printer;

bool RuleSystem::ApplyRules(const string& input,
                            string* output,
                            bool use_lookahead) const {
  Compiler compiler(fst::TokenType::BYTE);
  MutableTransducer input_fst, output_fst;
  if (!compiler.operator()(input, &input_fst)) {
    LoggerError("Failed to compile input string \"%s\"", input.c_str());
    return false;
  }
  if (!ApplyRules(input_fst, &output_fst, use_lookahead)) return false;
  MutableTransducer shortest_path;
  fst::ShortestPath(output_fst, &shortest_path);
  fst::Project(&shortest_path, fst::PROJECT_OUTPUT);
  fst::RmEpsilon(&shortest_path);
  Printer printer(fst::TokenType::BYTE);
  if (!printer.operator()(shortest_path, output)) {
    LoggerError("Failed to print output string");
    return false;
  }
  return true;
}

bool RuleSystem::ApplyRules(const Transducer& input,
                            string* output,
                            bool use_lookahead) const {
  MutableTransducer output_fst;
  if (!ApplyRules(input, &output_fst, use_lookahead)) return false;
  MutableTransducer shortest_path;
  fst::ShortestPath(output_fst, &shortest_path);
  fst::Project(&shortest_path, fst::PROJECT_OUTPUT);
  fst::RmEpsilon(&shortest_path);
  Printer printer(fst::TokenType::BYTE);
  if (!printer.operator()(shortest_path, output)) {
    LoggerError("Failed to print to output string");
    return false;
  }
  return true;
}

const Transducer* RuleSystem::FindRule(const string& name) const {
  return grm_->GetFst(name);
}


}
}
