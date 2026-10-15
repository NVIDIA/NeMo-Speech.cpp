#include "sparrowhawk_string_matcher.h"

#include <cstdlib>
#include <iostream>
#include <string>

using Arc = fst::StdArc;
using Fst = fst::VectorFst<Arc>;
using Matcher = speech::sparrowhawk::StringMatcher;

static int checks = 0;
static void Check(bool condition, const char* label) {
  ++checks;
  if (!condition) {
    std::cerr << "FAIL " << label << '\n';
    std::exit(1);
  }
}

static Fst Input(const std::string& text) {
  Fst result;
  result.AddState();
  result.SetStart(0);
  for (unsigned char byte : text) {
    int next = result.AddState();
    result.AddArc(next - 1, Arc(byte, byte, 0, next));
  }
  result.SetFinal(result.NumStates() - 1, 0);
  return result;
}

static Fst Grammar(int states) {
  Fst result;
  for (int i = 0; i < states; ++i) result.AddState();
  result.SetStart(0);
  return result;
}

static std::string Output(const Fst& result, bool input = false) {
  std::string text;
  int state = result.Start();
  for (int i = 0; i < result.NumStates(); ++i) {
    fst::ArcIterator<Fst> it(result, state);
    if (it.Done()) return text;
    Arc arc = it.Value();
    if (input ? arc.ilabel : arc.olabel)
      text.push_back(char(input ? arc.ilabel : arc.olabel));
    state = arc.nextstate;
  }
  Check(false, "output not a simple path");
  return {};
}

static void Unchanged(const Fst& output) {
  Check(Output(output) == "sentinel", "fallback leaves output untouched");
}

static void AddOutputPath(Fst* grammar, const std::string& text, bool separate_input) {
  int state = 0;
  if (separate_input) {
    int next = grammar->AddState();
    grammar->AddArc(state, Arc('a', 0, 0, next));
    state = next;
  }
  bool first = true;
  for (unsigned char byte : text) {
    int next = grammar->AddState();
    grammar->AddArc(state, Arc(first && !separate_input ? 'a' : 0, byte, 0, next));
    first = false;
    state = next;
  }
  grammar->SetFinal(state, 0);
}

static void CanonicalCase(const std::string& first, const std::string& second,
                          bool accepted, const char* label) {
  Fst grammar = Grammar(1);
  AddOutputPath(&grammar, first, false);
  AddOutputPath(&grammar, second, true);
  Matcher::Limits limits;
  limits.canonical_word_outputs = true;
  Matcher matcher(grammar, limits);
  Fst output = Input("sentinel");
  Check(matcher.Match(Input("a"), &output) == accepted, label);
  if (accepted) {
    Check(Output(output) == first || Output(output) == second,
          "canonical identity never rewrites emitted output");
    Check(Output(output, true) == "a", "canonical mode retains actual input labels");
  } else {
    Unchanged(output);
  }
  Check(matcher.Match(Input("a"), &output) == accepted,
        "canonical prefix generations reset across calls");
}

int main() {
  {
    const std::string one = R"(tokens { name: "Alice" } tokens { name: "Bob" })";
    const std::string many = R"(tokens   {   name:   "Alice"   }   tokens   { name: "Bob" })";
    CanonicalCase(one, many, true, "outside repeated spaces coalesce");
    Fst grammar = Grammar(1);
    AddOutputPath(&grammar, one, false);
    AddOutputPath(&grammar, many, true);
    Matcher original(grammar);
    Fst output = Input("sentinel");
    Check(!original.Match(Input("a"), &output), "canonical mode is opt-in");
    Unchanged(output);
  }
  CanonicalCase(R"(tokens { name: "Alice  Bob" })", R"(tokens { name: "Alice Bob" })",
                false, "quoted repeated spaces remain distinct");
  CanonicalCase(R"(tokens { name: "Alice\"  Bob" })", R"(tokens { name: "Alice\" Bob" })",
                false, "escaped quote does not end quoted whitespace protection");
  CanonicalCase(R"(tokens { name: "A\\" } tokens { name: "B" })",
                R"(tokens { name: "A\\" }   tokens { name: "B" })",
                true, "even backslash parity permits outside collapse");
  CanonicalCase(R"(tokens { name: "A\\\"  B" })", R"(tokens { name: "A\\\" B" })",
                false, "odd backslash parity retains quoted spaces");
  CanonicalCase(R"(tokens { name: "Alice" })", R"(tokens { name: "Carol" })",
                false, "different word names are genuine output ambiguity");
  CanonicalCase(R"(tokens { name: "unterminated })", R"(tokens { name: "unterminated })",
                false, "unbalanced normal escape syntax cannot enter WORD shortcut");
  CanonicalCase(R"(tokens { date { month: "january" day: "two" } })",
                R"(tokens { date { month: "february" day: "three" } })",
                false, "different date semantics never coalesce");
  {
    Fst grammar = Grammar(3);
    grammar.AddArc(0, Arc('a', 'A', 0, 1));
    grammar.AddArc(0, Arc('a', 'A', 0, 1));
    grammar.AddArc(0, Arc('a', 'B', 10, 2));
    grammar.SetFinal(1, 0);
    grammar.SetFinal(2, 0);
    Matcher matcher(grammar);
    Fst output;
    Check(matcher.Match(Input("a"), &output), "same-prefix parallel paths coalesce");
    Check(Output(output) == "A", "same-prefix winning output");
    Check(matcher.diagnostics().second_cost == 10, "runner-up is distinct prefix");
  }
  {
    Fst grammar = Grammar(5);
    grammar.AddArc(0, Arc('a', 'A', 2, 1));
    grammar.AddArc(0, Arc(0, 0, 0, 2));
    grammar.AddArc(2, Arc('a', 'A', 0, 1));
    grammar.AddArc(1, Arc('b', 'B', 0, 3));
    grammar.AddArc(1, Arc('b', 'X', 5, 4));
    grammar.SetFinal(3, 0);
    grammar.SetFinal(4, 0);
    Matcher matcher(grammar);
    Fst output;
    Check(matcher.Match(Input("ab"), &output), "double-epsilon keeps canonical prefix");
    Check(Output(output) == "AB", "best weight retained when same prefix merges");
    Check(matcher.diagnostics().best_cost == 0, "better same-prefix raw weight");
    Check(matcher.diagnostics().second_cost == 5, "distinct runner-up propagates suffix");
  }
  {
    Fst grammar = Grammar(104);
    grammar.AddArc(0, Arc('a', 'A', 0, 103));
    grammar.SetFinal(103, 0);
    for (int i = 1; i <= 100; ++i) {
      grammar.AddArc(0, Arc(0, 'x', -1, i));
      grammar.AddArc(i, Arc('b', 'B', 0, 102));
    }
    grammar.SetFinal(102, 0);
    Matcher matcher(grammar);
    Fst output;
    Check(matcher.Match(Input("a"), &output), "next-byte pruning preserves winner");
    Check(matcher.diagnostics().visited_states == 2, "dead epsilon states not enqueued");
    Check(Output(output) == "A", "pruned output exact");
  }
  {
    Fst grammar = Grammar(5);
    grammar.AddArc(0, Arc('a', 'A', 0, 1));
    grammar.AddArc(0, Arc('a', 'B', -20, 2));
    grammar.AddArc(1, Arc(0, 'Z', 0, 3));
    grammar.AddArc(2, Arc('b', 'X', 0, 4));
    grammar.SetFinal(3, 0);
    grammar.SetFinal(4, 0);
    Matcher matcher(grammar);
    Fst output;
    Check(matcher.Match(Input("a"), &output), "end-of-input can-final pruning");
    Check(Output(output) == "AZ", "negative dead suffix not selected");
  }
  {
    Fst grammar = Grammar(2);
    grammar.AddArc(0, Arc('a', 'A', 0, 1));
    grammar.AddArc(0, Arc('a', 'B', 0, 1));
    grammar.AddArc(0, Arc('a', 'C', 0, 1));
    grammar.SetFinal(1, 0);
    Matcher matcher(grammar);
    Fst output = Input("sentinel");
    Check(!matcher.Match(Input("a"), &output), "third distinct tie stays guarded");
    Check(matcher.diagnostics().discarded_cost == 0, "discarded witness reaches final");
    Unchanged(output);
  }
  {
    Fst grammar = Grammar(5);
    grammar.AddArc(0, Arc(0, 'x', 1, 3));
    grammar.AddArc(0, Arc(0, 'y', 5, 1));
    grammar.AddArc(1, Arc(0, 'z', -10, 3));
    grammar.AddArc(3, Arc('a', 'A', 0, 4));
    grammar.SetFinal(4, 0);
    Matcher matcher(grammar);
    Fst output;
    Check(matcher.valid(), "negative epsilon DAG supported");
    Check(matcher.Match(Input("a"), &output), "negative epsilon path accepted");
    Check(Output(output) == "yzA", "epsilon topology updates before propagation");
    Check(Output(output, true) == "a", "input labels preserved");
    Check(matcher.diagnostics().best_cost == -5, "negative path cost");
    Check(matcher.diagnostics().second_cost == 1, "runner-up through shared suffix");
  }
  {
    Fst grammar = Grammar(4);
    grammar.AddArc(0, Arc('a', 'A', 3, 1));
    grammar.AddArc(0, Arc('a', 'X', 1, 2));
    grammar.AddArc(1, Arc('b', 'B', -10, 3));
    grammar.AddArc(2, Arc('b', 'Y', 0, 3));
    grammar.SetFinal(3, 0);
    Matcher matcher(grammar);
    Fst output;
    Check(matcher.Match(Input("ab"), &output), "negative consuming path");
    Check(Output(output) == "AB", "not greedy or Dijkstra early termination");
    output = Input("sentinel");
    Check(!matcher.Match(Input("a"), &output), "incomplete input has no path");
    Unchanged(output);
    Check(matcher.Match(Input("ab"), &output), "generation reuse after fallback");
    Check(Output(output) == "AB", "no stale previous-position state");
  }
  for (bool merge : {false, true}) {
    Fst grammar = Grammar(4);
    grammar.AddArc(0, Arc('a', 'A', 0, 1));
    grammar.AddArc(0, Arc('a', 'B', 0, merge ? 1 : 2));
    grammar.SetFinal(1, 0);
    grammar.SetFinal(2, 0);
    Matcher matcher(grammar);
    Fst output = Input("sentinel");
    Check(!matcher.Match(Input("a"), &output), "exact merge/final tie falls back");
    Check(std::string(matcher.diagnostics().reason) == "ambiguous_path", "tie reason");
    Unchanged(output);
  }
  {
    Fst grammar = Grammar(3);
    grammar.AddArc(0, Arc('a', 'A', 0, 1));
    grammar.AddArc(0, Arc('a', 'B', 0.00001f, 2));
    grammar.SetFinal(1, 0);
    grammar.SetFinal(2, 0);
    Matcher matcher(grammar);
    Fst output = Input("sentinel");
    Check(!matcher.Match(Input("a"), &output), "near tie falls back");
    Unchanged(output);
  }
  {
    Fst grammar = Grammar(3);
    grammar.AddArc(0, Arc(0, 'A', 0, 1));
    grammar.AddArc(1, Arc(0, 'B', -1, 0));
    grammar.AddArc(1, Arc('a', 'C', 0, 2));
    grammar.SetFinal(2, 0);
    Matcher matcher(grammar);
    Check(!matcher.valid(), "input epsilon cycle rejected at construction");
    Fst output = Input("sentinel");
    Check(!matcher.Match(Input("a"), &output), "cycle never enters relaxation");
    Unchanged(output);
  }
  {
    Fst grammar = Grammar(2);
    grammar.AddArc(0, Arc('a', 'A', 0, 1));
    grammar.SetFinal(1, 0);
    Matcher::Limits limits;
    limits.arc_relaxations = 0;
    Matcher matcher(grammar, limits);
    Fst output = Input("sentinel");
    Check(!matcher.Match(Input("a"), &output), "work budget falls back");
    Unchanged(output);
    limits.arc_relaxations = 10;
    limits.backpointers = 0;
    Matcher no_memory(grammar, limits);
    Check(!no_memory.Match(Input("a"), &output), "backpointer budget falls back");
    Unchanged(output);
    Matcher normal(grammar);
    Fst weighted = Input("a");
    weighted.SetFinal(1, 1);
    Check(!normal.Match(weighted, &output), "weighted/preprocessed input rejected");
    Fst branching = Input("a");
    branching.AddArc(0, Arc('a', 'a', 0, 1));
    Check(!normal.Match(branching, &output), "nonlinear input rejected");
    Fst nul = Input(std::string(1, '\0'));
    Check(!normal.Match(nul, &output), "NUL epsilon input rejected");
    Unchanged(output);
    limits.input_bytes = 8;
    Matcher short_limit(grammar, limits);
    Fst cyclic = Grammar(1);
    cyclic.AddArc(0, Arc('a', 'a', 0, 0));
    Check(!short_limit.Match(cyclic, &output), "cyclic input bounded");
  }
  std::cout << "PASS " << checks << " synthetic CPU checks\n";
}
