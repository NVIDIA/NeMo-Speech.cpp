// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#include "batching.h"

using nemo_speech::asr::BatchingConfig;
using nemo_speech::asr::MicroBatcher;

int
main() {
    auto require = [](bool ok, const char* message) {
        if (!ok)
            throw std::runtime_error(message);
    };
    const BatchingConfig defaults;
    require(defaults.max_batch_size == 1024, "wrong default maximum batch size");
    require(defaults.max_queue_delay_us == 5000, "wrong default queue delay");
    require(defaults.max_queue_depth == 2048, "wrong default queue depth");
    require(defaults.ingress_cohort_delay_us == 20000, "wrong default ingress cohort delay");

    BatchingConfig cfg;
    cfg.enabled = true;
    cfg.max_batch_size = 4;
    cfg.max_queue_delay_us = 10000;
    cfg.max_queue_depth = 32;
    std::atomic<int> largest{0};
    MicroBatcher<int, int, int> batcher(cfg, [&largest](const int& key, std::vector<int>&& in) {
        largest.store(std::max(largest.load(), static_cast<int>(in.size())));
        std::vector<int> out;
        for (int v : in) out.push_back(key + v);
        return out;
    });

    std::vector<std::future<int>> calls;
    for (int i = 0; i < 4; ++i) {
        calls.push_back(
            std::async(std::launch::async, [&batcher, i] { return batcher.run(7, i); }));
    }
    for (int i = 0; i < 4; ++i) require(calls[i].get() == 7 + i, "wrong result");
    require(largest.load() == 4, "jobs did not coalesce");
    const auto m = batcher.metrics();
    require(m.batches == 1 && m.items == 4 && m.max_observed_batch == 4, "wrong metrics");

    // Incompatible keys must never share a callback invocation.
    auto a = std::async(std::launch::async, [&] { return batcher.run(1, 10); });
    auto b = std::async(std::launch::async, [&] { return batcher.run(2, 20); });
    require(a.get() == 11, "wrong result for first key");
    require(b.get() == 22, "wrong result for second key");

    // A live cohort target releases a complete ingress wave without waiting
    // for the much larger physical maximum or the queue deadline.
    BatchingConfig cohort_cfg = cfg;
    cohort_cfg.max_queue_delay_us = 500000;
    std::atomic<int> cohort_batch{0};
    MicroBatcher<int, int, int> cohort_batcher(
        cohort_cfg, [&cohort_batch](const int&, std::vector<int>&& in) {
            cohort_batch.store(static_cast<int>(in.size()));
            return std::move(in);
        });
    const auto started = std::chrono::steady_clock::now();
    auto c = std::async(std::launch::async, [&] { return cohort_batcher.run(3, 30, 2); });
    auto d = std::async(std::launch::async, [&] { return cohort_batcher.run(3, 40, 2); });
    require(c.get() == 30 && d.get() == 40, "wrong cohort result");
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    require(cohort_batch.load() == 2, "cohort did not release as one batch");
    require(elapsed_ms < 250, "cohort waited for the physical maximum");

    // Direct scalar and inline calls must reject work after shutdown without
    // entering the callback during owner teardown.
    BatchingConfig scalar_cfg;
    std::atomic<int> callbacks{0};
    MicroBatcher<int, int, int> scalar_batcher(
        scalar_cfg, [&callbacks](const int&, std::vector<int>&& in) {
            callbacks.fetch_add(1);
            return std::move(in);
        });
    scalar_batcher.shutdown();
    bool scalar_rejected = false;
    bool inline_rejected = false;
    try {
        (void)scalar_batcher.run(0, 1);
    }
    catch (const std::runtime_error&) {
        scalar_rejected = true;
    }
    try {
        (void)scalar_batcher.run_inline(0, 1);
    }
    catch (const std::runtime_error&) {
        inline_rejected = true;
    }
    require(scalar_rejected, "scalar call was accepted after shutdown");
    require(inline_rejected, "inline call was accepted after shutdown");
    require(callbacks.load() == 0, "callback ran after shutdown");

    std::cout << "batching tests passed\n";
    return 0;
}
