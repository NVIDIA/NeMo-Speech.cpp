// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// PnC smoke test: load a pnc.* GGUF and punctuate/capitalize fixed inputs.
// Usage: ./test_pnc <pnc.gguf> [--gpu N]   (skips if no model arg)
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "pnc_model.h"
#include "pnc_runner.h"
#include "runtime.h"

using namespace nemo_speech::asr;

int
main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stdout, "[SKIP] usage: %s <pnc.gguf> [--gpu N]\n", argv[0]);
        return 0;
    }
    int gpu = -1;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);

    ggml_runtime::Params p;
    p.use_gpu = (gpu >= 0);
    p.gpu_device_idx = std::max(gpu, 0);
    p.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(p);

    BatchingConfig batching;
    batching.enabled = true;
    batching.max_batch_size = 8;
    batching.max_queue_delay_us = 500000;
    pnc::PncModel model(bm, argv[1], batching);
    pnc::PncRunner runner(&model);
    const ScopedBatchCohort scalar_cohort(1);

    // Alternate sequence lengths to exercise recurring graph-cache entries.
    const char* inputs[] = {
        "and so my",
        "i live in new york and work at nvidia",
        "what time is it",
        "and so my",
        "and so my fellow americans ask not what your country can do for you ask what you "
        "can do for your country",
        "and so my",
    };

    int fail = 0;
    for (const char* in : inputs) {
        const std::string out = runner.postprocess(in);
        std::fprintf(stdout, "IN : %s\nOUT: %s\n", in, out.c_str());
        if (out.empty() || !std::isupper(static_cast<unsigned char>(out[0])))
            ++fail;  // every sentence should start capitalized
    }
    {
        const auto& c = model.config();
        const int32_t ids[] = {c.cls_id, c.unk_id, c.unk_id, c.sep_id};
        std::vector<int> ref_p, ref_c;
        model.infer(ids, 4, ref_p, ref_c);
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::future<bool>> calls;
        const auto metrics_before = model.batch_metrics();
        for (int i = 0; i < 4; ++i) {
            calls.push_back(std::async(std::launch::async, [&] {
                const ScopedBatchCohort cohort(4);
                ready.fetch_add(1);
                while (!go.load()) std::this_thread::yield();
                std::vector<int> p, c;
                model.infer(ids, 4, p, c);
                return p == ref_p && c == ref_c;
            }));
        }
        while (ready.load() != 4) std::this_thread::yield();
        go.store(true);
        for (auto& call : calls)
            if (!call.get())
                ++fail;
        const auto metrics_after = model.batch_metrics();
        if (metrics_after.deadline_batches != metrics_before.deadline_batches ||
            metrics_after.target_reached_batches <= metrics_before.target_reached_batches) {
            std::fprintf(stderr, "PnC cohort waited for the queue deadline\n");
            ++fail;
        }
        if (metrics_after.max_observed_batch < 4) {
            std::fprintf(stderr, "PnC requests did not coalesce\n");
            ++fail;
        } else {
            std::fprintf(stdout, "PnC B=4 parity OK\n");
        }
    }
    {
        // Exercise compact per-item I32 outputs on real WordPiece labels; the
        // synthetic unknown-token probe above does not cover their layout.
        constexpr const char* text = "how is the weather today";
        const std::string expected = runner.postprocess(text);
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::future<std::string>> calls;
        for (int i = 0; i < 4; ++i) {
            calls.push_back(std::async(std::launch::async, [&] {
                const ScopedBatchCohort cohort(4);
                pnc::PncRunner local(&model);
                ready.fetch_add(1);
                while (!go.load()) std::this_thread::yield();
                return local.postprocess(text);
            }));
        }
        while (ready.load() != 4) std::this_thread::yield();
        go.store(true);
        for (auto& call : calls) {
            const std::string got = call.get();
            if (got != expected) {
                std::fprintf(
                    stderr, "PnC full-runner batch mismatch: expected \"%s\", got \"%s\"\n",
                    expected.c_str(), got.c_str());
                ++fail;
            }
        }
    }
    {
        const auto& c = model.config();
        std::vector<std::vector<int32_t>> requests;
        for (int n : {4, 6, 10, 15}) {
            std::vector<int32_t> ids(static_cast<size_t>(n), c.unk_id);
            ids.front() = c.cls_id;
            ids.back() = c.sep_id;
            requests.push_back(std::move(ids));
        }
        std::vector<std::vector<int>> ref_p(requests.size()), ref_c(requests.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            model.infer(
                requests[i].data(), static_cast<int>(requests[i].size()), ref_p[i], ref_c[i]);
        }
        const auto before = model.batch_metrics();
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::future<bool>> calls;
        for (size_t i = 0; i < requests.size(); ++i) {
            calls.push_back(std::async(std::launch::async, [&, i] {
                const ScopedBatchCohort cohort(4);
                ready.fetch_add(1);
                while (!go.load()) std::this_thread::yield();
                std::vector<int> p, c_out;
                model.infer(requests[i].data(), static_cast<int>(requests[i].size()), p, c_out);
                return p == ref_p[i] && c_out == ref_c[i];
            }));
        }
        while (ready.load() != 4) std::this_thread::yield();
        go.store(true);
        for (auto& call : calls)
            if (!call.get())
                ++fail;
        const auto after = model.batch_metrics();
        if (after.batches - before.batches != 1 || after.items - before.items != 4) {
            std::fprintf(stderr, "mixed-length PnC requests did not share one batch\n");
            ++fail;
        } else {
            std::fprintf(stdout, "PnC mixed-length B=4 parity OK\n");
        }
    }
    std::fprintf(stdout, fail ? "FAILED (%d)\n" : "OK\n", fail);
    return fail ? 1 : 0;
}
