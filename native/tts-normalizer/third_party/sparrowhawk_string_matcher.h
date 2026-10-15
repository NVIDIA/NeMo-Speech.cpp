#ifndef NEMOTRON_SPARROWHAWK_STRING_MATCHER_H_
#define NEMOTRON_SPARROWHAWK_STRING_MATCHER_H_

#include <fst/fstlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace speech {
namespace sparrowhawk {

class StringMatcher {
 public:
  using Arc = fst::StdArc;
  using StateId = Arc::StateId;
  using Transducer = fst::Fst<Arc>;
  using MutableTransducer = fst::VectorFst<Arc>;

  struct Limits {
    bool canonical_word_outputs = false;
    std::size_t grammar_states = 1000000;
    std::size_t grammar_arcs = 4000000;
    std::size_t input_bytes = 65536;
    std::size_t visited_states = 1000000;
    std::size_t arc_relaxations = 4000000;
    std::size_t backpointers = 1000000;
    double absolute_margin = 0.001;
    double relative_margin = 0.000001;
  };

  struct Diagnostics {
    const char* reason = "not_called";
    std::size_t visited_states = 0;
    std::size_t arc_relaxations = 0;
    std::size_t backpointers = 0;
    float best_cost = std::numeric_limits<float>::infinity();
    float second_cost = std::numeric_limits<float>::infinity();
    float discarded_cost = std::numeric_limits<float>::infinity();
    double raw_path_roundoff_bound = 0;
    double ambiguity_guard = 0;
  };

  explicit StringMatcher(const Transducer& grammar)
      : StringMatcher(grammar, Limits{}) {}

  StringMatcher(const Transducer& grammar, Limits limits) : limits_(limits) {
    try {
      valid_ = Build(grammar);
    } catch (const std::bad_alloc&) {
      build_reason_ = "grammar_allocation";
    } catch (const std::length_error&) {
      build_reason_ = "grammar_allocation";
    }
  }

  bool valid() const { return valid_; }
  const Diagnostics& diagnostics() const { return diagnostics_; }

  bool Match(const Transducer& input, MutableTransducer* output) {
    diagnostics_ = Diagnostics{};
    try {
      if (!output) return Fail("null_output");
      if (!valid_) return Fail(build_reason_);
      return MatchImpl(input, output);
    } catch (const std::bad_alloc&) {
      return Fail("request_allocation");
    } catch (const std::length_error&) {
      return Fail("request_allocation");
    }
  }

 private:
  static constexpr std::uint32_t kNoNode =
      std::numeric_limits<std::uint32_t>::max();
  static constexpr float kInf = std::numeric_limits<float>::infinity();

  struct FlatArc {
    std::int32_t input;
    std::int32_t output;
    StateId next;
    float weight;
  };
  struct State {
    std::size_t epsilon_begin;
    std::size_t epsilon_end;
    std::size_t consume_begin;
    std::size_t consume_end;
    float final;
    std::uint32_t rank = 0;
    std::array<std::uint64_t, 4> next_bytes{};
    bool can_final = false;
  };
  struct Choice {
    float cost = kInf;
    std::uint32_t prefix = 0;
    std::uint32_t node = kNoNode;
    float final_weight = 0;
  };
  struct Cell {
    std::uint64_t generation = 0;
    Choice best;
    Choice second;
    float discarded = kInf;
    double absolute_sum = 0;
    std::uint32_t steps = 0;
  };
  struct PrefixKey {
    std::uint32_t parent;
    std::int32_t input;
    std::int32_t output;
    bool operator==(const PrefixKey& other) const {
      return parent == other.parent && input == other.input && output == other.output;
    }
  };
  struct PrefixMetadata {
    bool in_quotes = false;
    bool escaped = false;
    bool outside_space = false;
  };
  struct PrefixHash {
    std::size_t operator()(const PrefixKey& key) const {
      std::size_t value = key.parent;
      value = value * 1000003 + std::size_t(key.input);
      return value * 1000003 + std::size_t(key.output);
    }
  };
  using QueueItem = std::pair<std::uint32_t, StateId>;
  struct Layer {
    std::vector<Cell> cells;
    std::vector<QueueItem> queue;
    std::uint64_t generation = 0;

    void Begin() {
      queue.clear();
      if (++generation == 0) {
        for (auto& cell : cells) cell.generation = 0;
        generation = 1;
      }
    }

    Cell& Touch(StateId state, std::uint32_t rank) {
      Cell& cell = cells[state];
      if (cell.generation != generation) {
        cell = Cell{};
        cell.generation = generation;
        queue.emplace_back(rank, state);
        std::push_heap(queue.begin(), queue.end(), std::greater<QueueItem>());
      }
      return cell;
    }

    StateId Pop() {
      std::pop_heap(queue.begin(), queue.end(), std::greater<QueueItem>());
      StateId result = queue.back().second;
      queue.pop_back();
      return result;
    }
  };
  struct Backpointer {
    std::uint32_t parent;
    FlatArc arc;
  };

  bool Fail(const char* reason) {
    diagnostics_.reason = reason;
    diagnostics_.backpointers = path_.size();
    return false;
  }

  bool Build(const Transducer& grammar) {
    if (!std::isfinite(limits_.absolute_margin) ||
        !std::isfinite(limits_.relative_margin) ||
        limits_.absolute_margin < 0 || limits_.relative_margin < 0 ||
        limits_.grammar_states >= kNoNode || limits_.backpointers >= kNoNode) {
      build_reason_ = "invalid_limits";
      return false;
    }
    start_ = grammar.Start();
    for (fst::StateIterator<Transducer> it(grammar); !it.Done(); it.Next()) {
      const StateId id = it.Value();
      if (id < 0 || static_cast<std::size_t>(id) != states_.size() ||
          states_.size() >= limits_.grammar_states) return false;
      const float final = grammar.Final(id).Value();
      if (std::isnan(final) || final == -kInf) return false;
      State state{epsilons_.size(), 0, consuming_.size(), 0, final, 0};
      for (fst::ArcIterator<Transducer> arcs(grammar, id); !arcs.Done();
           arcs.Next()) {
        const Arc& arc = arcs.Value();
        const float weight = arc.weight.Value();
        if (arc.ilabel < 0 || arc.ilabel > 255 || arc.olabel < 0 ||
            arc.olabel > 255 || arc.nextstate < 0 || !std::isfinite(weight)) {
          return false;
        }
        if (epsilons_.size() + consuming_.size() >= limits_.grammar_arcs)
          return false;
        FlatArc flat{arc.ilabel, arc.olabel, arc.nextstate, weight};
        if (arc.ilabel == 0) epsilons_.push_back(flat);
        else consuming_.push_back(flat);
      }
      state.epsilon_end = epsilons_.size();
      state.consume_end = consuming_.size();
      std::stable_sort(consuming_.begin() + state.consume_begin,
                       consuming_.end(),
                       [](const FlatArc& a, const FlatArc& b) {
                         return a.input < b.input;
                       });
      states_.push_back(state);
    }
    if (start_ < 0 || static_cast<std::size_t>(start_) >= states_.size())
      return false;
    std::vector<std::size_t> indegree(states_.size(), 0);
    for (const auto& arc : consuming_)
      if (static_cast<std::size_t>(arc.next) >= states_.size()) return false;
    for (const auto& arc : epsilons_) {
      if (static_cast<std::size_t>(arc.next) >= states_.size()) return false;
      ++indegree[arc.next];
    }
    std::vector<StateId> ready;
    ready.reserve(states_.size());
    for (std::size_t i = 0; i < states_.size(); ++i)
      if (indegree[i] == 0) ready.push_back(static_cast<StateId>(i));
    for (std::size_t i = 0; i < ready.size(); ++i) {
      const StateId id = ready[i];
      states_[id].rank = static_cast<std::uint32_t>(i);
      for (std::size_t a = states_[id].epsilon_begin;
           a < states_[id].epsilon_end; ++a)
        if (--indegree[epsilons_[a].next] == 0)
          ready.push_back(epsilons_[a].next);
    }
    if (ready.size() != states_.size()) {
      build_reason_ = "input_epsilon_cycle";
      return false;
    }
    for (auto it = ready.rbegin(); it != ready.rend(); ++it) {
      State& state = states_[*it];
      state.can_final = state.final != kInf;
      for (std::size_t a = state.consume_begin; a < state.consume_end; ++a) {
        const unsigned label = consuming_[a].input;
        state.next_bytes[label / 64] |= std::uint64_t(1) << (label % 64);
      }
      for (std::size_t a = state.epsilon_begin; a < state.epsilon_end; ++a) {
        const State& next = states_[epsilons_[a].next];
        state.can_final = state.can_final || next.can_final;
        for (unsigned word = 0; word < 4; ++word)
          state.next_bytes[word] |= next.next_bytes[word];
      }
    }
    layers_[0].cells.resize(states_.size());
    layers_[1].cells.resize(states_.size());
    return true;
  }

  bool ReadInput(const Transducer& input) {
    bytes_.clear();
    StateId state = input.Start();
    if (state == fst::kNoStateId) return false;
    for (std::size_t count = 0; count <= limits_.input_bytes; ++count) {
      const float final = input.Final(state).Value();
      fst::ArcIterator<Transducer> it(input, state);
      if (it.Done()) return final == 0;
      if (final != kInf || count == limits_.input_bytes) return false;
      const Arc arc = it.Value();
      it.Next();
      if (!it.Done() || arc.ilabel < 1 || arc.ilabel > 255 ||
          arc.ilabel != arc.olabel || arc.weight.Value() != 0 ||
          arc.nextstate == fst::kNoStateId) return false;
      bytes_.push_back(arc.ilabel);
      state = arc.nextstate;
    }
    return false;
  }

  bool Reachable(StateId state, std::size_t position) const {
    const State& value = states_[state];
    if (position == bytes_.size()) return value.can_final;
    const unsigned label = bytes_[position];
    return (value.next_bytes[label / 64] & (std::uint64_t(1) << (label % 64))) != 0;
  }

  bool Prefix(std::uint32_t parent, const FlatArc& arc, std::uint32_t* output) {
    PrefixMetadata metadata;
    if (limits_.canonical_word_outputs) {
      metadata = prefix_metadata_[parent];
      if (arc.output == 0 ||
          (!metadata.in_quotes && metadata.outside_space && arc.output == ' ')) {
        *output = parent;
        return true;
      }
      if (metadata.in_quotes) {
        if (metadata.escaped) metadata.escaped = false;
        else if (arc.output == '\\') metadata.escaped = true;
        else if (arc.output == '"') metadata.in_quotes = false;
        metadata.outside_space = false;
      } else {
        metadata.in_quotes = arc.output == '"';
        metadata.escaped = false;
        metadata.outside_space = arc.output == ' ';
      }
    } else if (arc.input == 0 && arc.output == 0) {
      *output = parent;
      return true;
    }
    const PrefixKey key{parent, limits_.canonical_word_outputs ? 0 : arc.input, arc.output};
    const auto found = prefixes_.find(key);
    if (found != prefixes_.end()) {
      *output = found->second;
      return true;
    }
    if (prefixes_.size() >= limits_.backpointers) return Fail("prefix_budget");
    const auto id = static_cast<std::uint32_t>(prefixes_.size() + 1);
    prefixes_.emplace(key, id);
    prefix_metadata_.push_back(metadata);
    *output = id;
    return true;
  }

  bool Consider(Cell* target, Choice candidate, const FlatArc* arc) {
    Choice* slot = nullptr;
    if (target->best.cost != kInf && candidate.prefix == target->best.prefix) {
      if (candidate.cost >= target->best.cost) return true;
      slot = &target->best;
    } else if (target->second.cost != kInf && candidate.prefix == target->second.prefix) {
      if (candidate.cost >= target->second.cost) return true;
      slot = &target->second;
    } else if (candidate.cost < target->best.cost) {
      target->discarded = std::min(target->discarded, target->second.cost);
      target->second = target->best;
      slot = &target->best;
    } else if (candidate.cost < target->second.cost) {
      target->discarded = std::min(target->discarded, target->second.cost);
      slot = &target->second;
    } else {
      target->discarded = std::min(target->discarded, candidate.cost);
      return true;
    }
    if (arc) {
      if (path_.size() >= limits_.backpointers) return Fail("backpointer_budget");
      const auto node = static_cast<std::uint32_t>(path_.size());
      path_.push_back(Backpointer{candidate.node, *arc});
      candidate.node = node;
    }
    *slot = candidate;
    if (target->second.cost < target->best.cost)
      std::swap(target->best, target->second);
    return true;
  }

  bool Relax(const Cell& source, const FlatArc& arc, Layer* destination,
             std::size_t destination_position) {
    if (++diagnostics_.arc_relaxations > limits_.arc_relaxations)
      return Fail("arc_budget");
    if (!Reachable(arc.next, destination_position)) return true;
    Cell& target = destination->Touch(arc.next, states_[arc.next].rank);
    if (source.steps == std::numeric_limits<std::uint32_t>::max())
      return Fail("path_length");
    target.steps = std::max(target.steps, source.steps + 1);
    target.absolute_sum = std::max(target.absolute_sum,
                                  source.absolute_sum + std::abs(double(arc.weight)));
    for (const Choice* choice : {&source.best, &source.second}) {
      if (choice->cost == kInf) continue;
      Choice candidate = *choice;
      candidate.cost = choice->cost + arc.weight;
      if (!std::isfinite(candidate.cost)) return Fail("nonfinite_path");
      if (!Prefix(choice->prefix, arc, &candidate.prefix)) return false;
      if (!Consider(&target, candidate, &arc)) return false;
    }
    if (source.discarded != kInf) {
      const float candidate = source.discarded + arc.weight;
      if (!std::isfinite(candidate)) return Fail("nonfinite_discarded_path");
      target.discarded = std::min(target.discarded, candidate);
    }
    return true;
  }

  std::vector<FlatArc> DebugWitness(std::uint32_t node) const {
    std::vector<FlatArc> result;
    while (node != kNoNode) {
      if (node >= path_.size() || result.size() >= limits_.backpointers)
        throw std::length_error("invalid diagnostic witness");
      const auto& value = path_[node];
      if (value.arc.input != 0 || value.arc.output != 0)
        result.push_back(value.arc);
      node = value.parent;
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  static void DebugJsonString(const std::string& value) {
    std::fputc('"', stderr);
    for (unsigned char byte : value) {
      if (byte == '"' || byte == '\\') std::fprintf(stderr, "\\%c", byte);
      else if (byte < 32) std::fprintf(stderr, "\\u%04x", unsigned(byte));
      else std::fputc(byte, stderr);
    }
    std::fputc('"', stderr);
  }

  static void DebugPath(const Choice& choice, const std::vector<FlatArc>& arcs,
                        std::size_t first_difference) {
    if (choice.cost == kInf) {
      std::fprintf(stderr, "null");
      return;
    }
    std::string input, output;
    for (const auto& arc : arcs) {
      if (arc.input) input.push_back(char(arc.input));
      if (arc.output) output.push_back(char(arc.output));
    }
    std::fprintf(stderr, "{\"cost\":%.9g,\"prefix_id\":%u,\"final_weight\":%.9g,\"input_utf8\":",
                 choice.cost, choice.prefix, choice.final_weight);
    DebugJsonString(input);
    std::fprintf(stderr, ",\"output_utf8\":");
    DebugJsonString(output);
    std::fprintf(stderr, ",\"pairs\":[");
    for (std::size_t i = 0; i < arcs.size(); ++i) {
      if (i) std::fputc(',', stderr);
      std::fprintf(stderr, "[%d,%d]", arcs[i].input, arcs[i].output);
    }
    std::fprintf(stderr, "],\"difference_context\":[");
    const std::size_t begin = first_difference > 8 ? first_difference - 8 : 0;
    const std::size_t end = std::min(arcs.size(), first_difference + 9);
    std::size_t input_offset = 0, output_offset = 0;
    bool first = true;
    for (std::size_t i = 0; i < arcs.size(); ++i) {
      if (i >= begin && i < end) {
        if (!first) std::fputc(',', stderr);
        first = false;
        std::fprintf(stderr, "{\"pair\":%zu,\"input_offset\":%zu,\"output_offset\":%zu,\"ilabel\":%d,\"olabel\":%d}",
                     i, input_offset, output_offset, arcs[i].input, arcs[i].output);
      }
      input_offset += arcs[i].input != 0;
      output_offset += arcs[i].output != 0;
    }
    std::fprintf(stderr, "]}");
  }

  void DebugPaths(const Cell& final) const {
    if (!std::getenv("NEMOTRON_TN_PATH_DEBUG")) return;
    const auto best = DebugWitness(final.best.node);
    const auto second = DebugWitness(final.second.node);
    std::size_t difference = 0;
    while (difference < best.size() && difference < second.size() &&
           best[difference].input == second[difference].input &&
           best[difference].output == second[difference].output) ++difference;
    std::fprintf(stderr, "TN_PATH {\"scope\":\"diagnostic_only\",\"first_differing_pair\":%zu,\"guard\":%.17g,\"best\":",
                 difference, diagnostics_.ambiguity_guard);
    DebugPath(final.best, best, difference);
    std::fprintf(stderr, ",\"second\":");
    DebugPath(final.second, second, difference);
    std::fprintf(stderr, "}\n");
  }

  bool MatchImpl(const Transducer& input, MutableTransducer* output) {
    path_.clear();
    prefixes_.clear();
    prefix_metadata_.clear();
    prefix_metadata_.emplace_back();  // Canonical empty prefix is ID zero.
    if (!ReadInput(input)) return Fail("unsupported_input");
    if (!Reachable(start_, 0)) return Fail("no_path");
    Layer* current = &layers_[0];
    Layer* next = &layers_[1];
    current->Begin();
    next->Begin();
    current->Touch(start_, states_[start_].rank).best.cost = 0;
    Cell final;
    for (std::size_t position = 0; position <= bytes_.size(); ++position) {
      while (!current->queue.empty()) {
        const StateId id = current->Pop();
        if (++diagnostics_.visited_states > limits_.visited_states)
          return Fail("state_budget");
        const Cell cell = current->cells[id];
        const State& state = states_[id];
        for (std::size_t a = state.epsilon_begin; a < state.epsilon_end; ++a)
          if (!Relax(cell, epsilons_[a], current, position)) return false;
        if (position < bytes_.size()) {
          auto begin = consuming_.begin() + state.consume_begin;
          auto end = consuming_.begin() + state.consume_end;
          auto it = std::lower_bound(begin, end, bytes_[position],
                                    [](const FlatArc& arc, std::int32_t byte) {
                                      return arc.input < byte;
                                    });
          for (; it != end && it->input == bytes_[position]; ++it)
            if (!Relax(cell, *it, next, position + 1)) return false;
        } else if (state.final != kInf) {
          final.steps = std::max(final.steps, cell.steps + 1);
          final.absolute_sum = std::max(
              final.absolute_sum, cell.absolute_sum + std::abs(double(state.final)));
          for (const Choice* choice : {&cell.best, &cell.second}) {
            if (choice->cost == kInf) continue;
            Choice candidate = *choice;
            candidate.cost = choice->cost + state.final;
            candidate.final_weight = state.final;
            if (!std::isfinite(candidate.cost)) return Fail("nonfinite_final");
            if (!Consider(&final, candidate, nullptr)) return false;
          }
          if (cell.discarded != kInf) {
            const float candidate = cell.discarded + state.final;
            if (!std::isfinite(candidate)) return Fail("nonfinite_discarded_final");
            final.discarded = std::min(final.discarded, candidate);
          }
        }
      }
      if (position < bytes_.size()) {
        std::swap(current, next);
        next->Begin();
      }
    }
    diagnostics_.best_cost = final.best.cost;
    diagnostics_.second_cost = final.second.cost;
    diagnostics_.discarded_cost = final.discarded;
    if (final.best.cost == kInf) return Fail("no_path");
    const double nu = final.steps * (double(std::numeric_limits<float>::epsilon()) / 2);
    if (nu >= 0.125) return Fail("roundoff_budget");
    const double error = nu / (1 - nu) * final.absolute_sum;
    const double guard = 2 * error + limits_.absolute_margin +
                         limits_.relative_margin * std::abs(double(final.best.cost));
    diagnostics_.raw_path_roundoff_bound = error;
    diagnostics_.ambiguity_guard = guard;
    DebugPaths(final);
    const float alternative = std::min(final.second.cost, final.discarded);
    if (alternative != kInf && double(alternative) - final.best.cost <= guard)
      return Fail("ambiguous_path");
    if (limits_.canonical_word_outputs &&
        (prefix_metadata_[final.best.prefix].in_quotes ||
         prefix_metadata_[final.best.prefix].escaped))
      return Fail("unbalanced_output_quotes");
    reverse_.clear();
    for (std::uint32_t node = final.best.node; node != kNoNode;
         node = path_[node].parent) {
      if (node >= path_.size() || reverse_.size() >= limits_.backpointers)
        return Fail("invalid_backpointer");
      reverse_.push_back(node);
    }
    MutableTransducer result;
    StateId from = result.AddState();
    result.SetStart(from);
    for (auto it = reverse_.rbegin(); it != reverse_.rend(); ++it) {
      const FlatArc& arc = path_[*it].arc;
      const StateId to = result.AddState();
      result.AddArc(from, Arc(arc.input, arc.output, Arc::Weight(arc.weight), to));
      from = to;
    }
    result.SetFinal(from, Arc::Weight(final.best.final_weight));
    *output = std::move(result);
    diagnostics_.reason = "matched_experimental";
    diagnostics_.backpointers = path_.size();
    return true;
  }

  Limits limits_;
  bool valid_ = false;
  const char* build_reason_ = "unsupported_grammar";
  StateId start_ = fst::kNoStateId;
  std::vector<State> states_;
  std::vector<FlatArc> epsilons_;
  std::vector<FlatArc> consuming_;
  Layer layers_[2];
  std::vector<Backpointer> path_;
  std::unordered_map<PrefixKey, std::uint32_t, PrefixHash> prefixes_;
  std::vector<PrefixMetadata> prefix_metadata_;
  std::vector<std::uint32_t> reverse_;
  std::vector<std::int32_t> bytes_;
  Diagnostics diagnostics_;
};

}  // namespace sparrowhawk
}  // namespace speech

#endif  // NEMOTRON_SPARROWHAWK_STRING_MATCHER_H_
