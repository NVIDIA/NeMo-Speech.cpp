// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Aho-Corasick context-biasing trie for RNNT word boosting (shallow fusion).
//
// Ported from NeMo's context_biasing.ContextGraph + GPUBoostingTreeModel:
// boost phrases are tokenized to subword ids and compiled into a token-level trie
// with Aho-Corasick fail links. During greedy RNNT decode, each step adds an
// additive bias to the joint logits and advances a per-stream state on every
// emitted (non-blank) token.
//
// Scoring (RNN-T defaults context_score=1.0, depth_scaling=2.0):
//   token_score(i)     = context_score                       (i == 0)
//                      = context_score*depth_scaling + ln(i+1) (i > 0)
//   node_score         = sum of token_score along the path from root
//   backoff_w(node)    = 0                                   (node is a phrase end - boost is
//   banked)
//                      = node_score(fail) - node_score        (otherwise - undoes a partial match)
// The per-token bias for token v from state s is the WFST score: follow fail
// links accumulating backoff_w until v has a forward arc, then add that arc's
// token_score (unk_score = 0 for tokens off the trie). This nets to zero for
// abandoned partial matches and is "banked" once a phrase completes.
#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

namespace nemo_speech::asr {

class ContextBiasingTree {
   public:
    static constexpr int kRoot = 0;

    // Build from per-phrase subword-token-id sequences. context_scores holds the
    // per-phrase base per-token score (the request boost); it must have one entry
    // per phrase. depth_scaling is the NeMo RNN-T default 2.0.
    void build(
        const std::vector<std::vector<int>>& phrases, const std::vector<float>& context_scores,
        float depth_scaling);

    void clear() { nodes_.assign(1, Node{}); }
    bool empty() const { return nodes_.size() <= 1; }

    // Additive per-token bias from `state`, before the fusion alpha:
    //   base       applies to every token with no trie arc reachable from `state`
    //   specials   overrides specific tokens (token -> score)
    struct StepBoost {
        float base = 0.0f;
        std::vector<std::pair<int, float>> specials;
    };
    StepBoost step_boost(int state) const;

    // Aho-Corasick goto: state after emitting `token` from `state`.
    int advance(int state, int token) const;

   private:
    struct Node {
        float node_score = 0.0f;
        float token_score = 0.0f;
        float backoff_w = 0.0f;
        bool is_end = false;
        int fail = kRoot;
        std::unordered_map<int, int> next;  // token id -> child node index
    };
    std::vector<Node> nodes_{Node{}};  // nodes_[0] = root
    void fill_fail();
};

}  // namespace nemo_speech::asr
