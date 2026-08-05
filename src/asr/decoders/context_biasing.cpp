// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "context_biasing.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <unordered_set>

namespace nemo_speech::asr {

void
ContextBiasingTree::build(
    const std::vector<std::vector<int>>& phrases, const std::vector<float>& context_scores,
    float depth_scaling) {
    nodes_.assign(1, Node{});  // root
    if (context_scores.size() != phrases.size())
        throw std::invalid_argument("ContextBiasingTree: one score per phrase required");
    for (size_t p = 0; p < phrases.size(); ++p) {
        const std::vector<int>& toks = phrases[p];
        if (toks.empty())
            continue;
        const float cs = context_scores[p];
        int node = kRoot;
        for (size_t i = 0; i < toks.size(); ++i) {
            const int tok = toks[i];
            const bool is_end = (i + 1 == toks.size());
            const float token_score =
                (i == 0) ? cs : (cs * depth_scaling + std::log(static_cast<float>(i + 1)));
            auto it = nodes_[node].next.find(tok);
            if (it == nodes_[node].next.end()) {
                const int child = static_cast<int>(nodes_.size());
                nodes_.emplace_back();
                // Re-fetch: emplace_back may have reallocated nodes_.
                nodes_[child].token_score = token_score;
                nodes_[child].node_score = nodes_[node].node_score + token_score;
                nodes_[child].is_end = is_end;
                nodes_[node].next[tok] = child;
                node = child;
            } else {
                // Shared state: keep the larger token score (matches NeMo).
                const int child = it->second;
                const float ts = std::max(token_score, nodes_[child].token_score);
                nodes_[child].token_score = ts;
                nodes_[child].is_end = nodes_[child].is_end || is_end;
                node = child;
            }
        }
    }
    // Recompute cumulative scores top-down: a later phrase can raise a shared
    // prefix node's token_score after that node's subtree was inserted, which
    // would leave descendant node_score (and thus backoff_w) stale.
    std::deque<int> order{kRoot};
    while (!order.empty()) {
        const int cur = order.front();
        order.pop_front();
        for (const auto& kv : nodes_[cur].next) {
            nodes_[kv.second].node_score = nodes_[cur].node_score + nodes_[kv.second].token_score;
            order.push_back(kv.second);
        }
    }
    fill_fail();
}

void
ContextBiasingTree::fill_fail() {
    nodes_[kRoot].fail = kRoot;
    std::deque<int> queue;
    for (const auto& kv : nodes_[kRoot].next) {
        nodes_[kv.second].fail = kRoot;
        queue.push_back(kv.second);
    }
    while (!queue.empty()) {
        const int cur = queue.front();
        queue.pop_front();
        // Snapshot children first - we only read other nodes below, but keep it
        // explicit that nodes_ is not resized during the BFS.
        for (const auto& kv : nodes_[cur].next) {
            const int tok = kv.first;
            const int child = kv.second;
            int f = nodes_[cur].fail;
            while (f != kRoot && nodes_[f].next.find(tok) == nodes_[f].next.end())
                f = nodes_[f].fail;
            int fail = kRoot;
            auto fit = nodes_[f].next.find(tok);
            if (fit != nodes_[f].next.end() && fit->second != child)
                fail = fit->second;
            nodes_[child].fail = fail;
            queue.push_back(child);
        }
    }
    for (size_t i = 1; i < nodes_.size(); ++i) {
        nodes_[i].backoff_w =
            nodes_[i].is_end ? 0.0f : (nodes_[nodes_[i].fail].node_score - nodes_[i].node_score);
    }
}

ContextBiasingTree::StepBoost
ContextBiasingTree::step_boost(int state) const {
    StepBoost sb;
    std::unordered_set<int> seen;
    float acc = 0.0f;
    int cur = state;
    while (true) {
        for (const auto& kv : nodes_[cur].next) {
            if (seen.insert(kv.first).second)
                sb.specials.emplace_back(kv.first, acc + nodes_[kv.second].token_score);
        }
        if (cur == kRoot)
            break;
        acc += nodes_[cur].backoff_w;
        cur = nodes_[cur].fail;
    }
    sb.base = acc;  // off-trie tokens back off to root (+ unk_score 0)
    return sb;
}

int
ContextBiasingTree::advance(int state, int token) const {
    int cur = state;
    while (true) {
        auto it = nodes_[cur].next.find(token);
        if (it != nodes_[cur].next.end())
            return it->second;
        if (cur == kRoot)
            return kRoot;
        cur = nodes_[cur].fail;
    }
}

}  // namespace nemo_speech::asr
