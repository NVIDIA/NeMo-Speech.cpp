// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Typed dynamic microbatching for ASR neural stages. Requests with the same
// compatibility key wait for a bounded interval, execute together, and resolve
// their original synchronous callers.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "parameter_parser.h"

namespace nemo_speech::asr {

struct BatchingConfig {
    // Disabled by default to avoid queueing single-stream inference.
    bool enabled = false;
    int max_batch_size = 1024;
    int max_queue_delay_us = 5000;
    int max_queue_depth = 2048;
    // Streaming ingestion needs a wider host-side collection window than an
    // already-aligned neural stage. Keeping this separate avoids fragmenting
    // concurrent streams into many exact-shape CUDA graphs.
    int ingress_cohort_delay_us = 20000;
    // Persistent recurrent/cache state is much larger than queued work, so
    // size its indexed row arena independently from queue backpressure.
    int state_arena_slots = 16;

    void Register(common::ParameterParser& p) {
        p.Register("enabled", &enabled, "Enable transparent dynamic batching");
        p.Register("max_batch_size", &max_batch_size, "Maximum neural microbatch size");
        p.Register(
            "max_queue_delay_us", &max_queue_delay_us,
            "Maximum time to wait for compatible work (microseconds)");
        p.Register("max_queue_depth", &max_queue_depth, "Maximum pending jobs per neural stage");
        p.Register(
            "ingress_cohort_delay_us", &ingress_cohort_delay_us,
            "Maximum time to align one streaming ingress wave (microseconds)");
        p.Register(
            "state_arena_slots", &state_arena_slots,
            "Maximum concurrent stateful streams held in device row arenas");
    }
};

struct BatchMetrics {
    uint64_t batches = 0;
    uint64_t items = 0;
    uint64_t singleton_batches = 0;
    uint64_t max_observed_batch = 0;
};

// A transport or pipeline coordinator can establish one expected cohort for
// the synchronous work performed by the current request thread. Neural stages
// use it as an early-release target: once the whole ingress wave has reached a
// compatible stage, that stage launches immediately instead of paying another
// independent queue timer. The scope is thread-local because each recognition
// stream is single-threaded by contract.
inline thread_local int g_batch_cohort_target = 0;

inline int
current_batch_cohort_target() {
    return g_batch_cohort_target;
}

class ScopedBatchCohort {
   public:
    explicit ScopedBatchCohort(int target) : previous_(g_batch_cohort_target) {
        g_batch_cohort_target = std::max(0, target);
    }
    ~ScopedBatchCohort() { g_batch_cohort_target = previous_; }
    ScopedBatchCohort(const ScopedBatchCohort&) = delete;
    ScopedBatchCohort& operator=(const ScopedBatchCohort&) = delete;

   private:
    int previous_;
};

// Forms one bounded streaming-ingress wave across RecognitionStreams.
class IngressBatchCoordinator {
   public:
    explicit IngressBatchCoordinator(const BatchingConfig& config)
        : enabled_(config.enabled), delay_us_(std::max(0, config.ingress_cohort_delay_us)) {
        if (const char* value = std::getenv("NEMO_SPEECH_INGRESS_COHORT"))
            enabled_ = enabled_ && std::atoi(value) != 0;
        if (const char* value = std::getenv("NEMO_SPEECH_INGRESS_COHORT_US"))
            delay_us_ = std::max(0, std::atoi(value));
    }

    int arrive() {
        if (!enabled_)
            return 0;
        std::shared_ptr<Group> group;
        {
            std::unique_lock<std::mutex> lock(mu_);
            if (!current_)
                current_ = std::make_shared<Group>();
            group = current_;
            if (group->waiters++ == 0)
                group->deadline = Clock::now() + std::chrono::microseconds(delay_us_);
            if (delay_us_ <= 0) {
                release_locked(group);
            } else {
                group->cv.wait_until(lock, group->deadline, [&] { return group->released; });
                if (!group->released)
                    release_locked(group);
            }
        }
        return group->released_size;
    }

   private:
    using Clock = std::chrono::steady_clock;
    struct Group {
        int waiters = 0;
        int released_size = 0;
        bool released = false;
        Clock::time_point deadline{};
        std::condition_variable cv;
    };

    void release_locked(const std::shared_ptr<Group>& group) {
        if (group->released)
            return;
        group->released = true;
        group->released_size = group->waiters;
        if (current_ == group)
            current_.reset();
        group->cv.notify_all();
    }

    bool enabled_ = false;
    int delay_us_ = 0;
    std::mutex mu_;
    std::shared_ptr<Group> current_;
};

template <class Key, class Request, class Result, class KeyEqual = std::equal_to<Key>>
class MicroBatcher {
   public:
    using RunBatch =
        std::function<std::vector<Result>(const Key&, std::vector<Request>&& requests)>;

    MicroBatcher(BatchingConfig cfg, RunBatch run_batch, KeyEqual equal = KeyEqual{})
        : cfg_(sanitize(cfg)), run_batch_(std::move(run_batch)), equal_(std::move(equal)) {
        if (!run_batch_)
            throw std::invalid_argument("MicroBatcher: run callback is empty");
        if (cfg_.enabled && cfg_.max_batch_size > 1)
            worker_ = std::thread([this] { worker_loop(); });
    }

    ~MicroBatcher() { shutdown(); }

    MicroBatcher(const MicroBatcher&) = delete;
    MicroBatcher& operator=(const MicroBatcher&) = delete;

    // `target_batch_size` is an optional live-concurrency hint. Stateful
    // decoders know how many step() calls are currently active, so they can
    // release a complete lock-step wave without waiting for the configured
    // physical maximum. Zero retains the generic max-batch/deadline policy.
    Result run(Key key, Request request, int target_batch_size = 0) {
        std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mu_);
        if (!worker_.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (stopping_)
                    throw std::runtime_error("MicroBatcher: submit after shutdown");
            }
            std::vector<Request> requests;
            requests.push_back(std::move(request));
            auto results = run_batch_(key, std::move(requests));
            if (results.size() != 1)
                throw std::runtime_error(
                    "MicroBatcher: scalar callback returned wrong result count");
            record_batch(1);
            return std::move(results.front());
        }
        // The queued path is synchronized by mu_ and shutdown joins worker_;
        // reserve the lifecycle lock for callbacks that execute on the caller.
        lifecycle_lock.unlock();

        auto job = std::make_shared<Job>(
            std::move(key), std::move(request),
            target_batch_size > 0 ? std::min(target_batch_size, cfg_.max_batch_size)
                                  : cfg_.max_batch_size);
        std::future<Result> future = job->promise.get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_)
                throw std::runtime_error("MicroBatcher: submit after shutdown");
            if (queue_.size() >= static_cast<size_t>(cfg_.max_queue_depth))
                throw std::runtime_error("MicroBatcher: queue is full");
            queue_.push_back(job);
        }
        cv_.notify_one();
        return future.get();
    }

    // Uncontended stateful inference should not cross a worker thread and a
    // promise for every tiny predictor/joint graph. The owner calls this only
    // when it can prove there is one active decode scope.
    Result run_inline(Key key, Request request) {
        std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mu_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_)
                throw std::runtime_error("MicroBatcher: submit after shutdown");
        }
        std::vector<Request> requests;
        requests.push_back(std::move(request));
        auto results = run_batch_(key, std::move(requests));
        if (results.size() != 1)
            throw std::runtime_error("MicroBatcher: inline callback returned wrong result count");
        record_batch(1);
        return std::move(results.front());
    }

    BatchMetrics metrics() const {
        return {
            batches_.load(std::memory_order_relaxed), items_.load(std::memory_order_relaxed),
            singleton_batches_.load(std::memory_order_relaxed),
            max_observed_batch_.load(std::memory_order_relaxed)};
    }

    void shutdown() {
        std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mu_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_)
                return;
            stopping_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable())
            worker_.join();
    }

   private:
    using Clock = std::chrono::steady_clock;
    struct Job {
        Job(Key k, Request r, int target)
            : key(std::move(k)), request(std::move(r)), queued_at(Clock::now()),
              target_batch_size(target) {}
        Key key;
        Request request;
        Clock::time_point queued_at;
        int target_batch_size;
        std::promise<Result> promise;
    };

    static BatchingConfig sanitize(BatchingConfig cfg) {
        cfg.max_batch_size = std::max(1, cfg.max_batch_size);
        cfg.max_queue_delay_us = std::max(0, cfg.max_queue_delay_us);
        cfg.max_queue_depth = std::max(cfg.max_batch_size, cfg.max_queue_depth);
        cfg.ingress_cohort_delay_us = std::max(0, cfg.ingress_cohort_delay_us);
        cfg.state_arena_slots = std::max(1, cfg.state_arena_slots);
        return cfg;
    }

    size_t compatible_count_locked(const Key& key) const {
        size_t n = 0;
        for (const auto& job : queue_)
            if (equal_(job->key, key) && ++n >= static_cast<size_t>(cfg_.max_batch_size))
                break;
        return n;
    }

    void worker_loop() {
        for (;;) {
            std::vector<std::shared_ptr<Job>> jobs;
            Key key{};
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (queue_.empty() && stopping_)
                    return;

                key = queue_.front()->key;
                const size_t target = static_cast<size_t>(queue_.front()->target_batch_size);
                const auto deadline =
                    queue_.front()->queued_at + std::chrono::microseconds(cfg_.max_queue_delay_us);
                cv_.wait_until(lock, deadline, [this, &key, target] {
                    return stopping_ || compatible_count_locked(key) >= target;
                });

                for (auto it = queue_.begin();
                     it != queue_.end() &&
                     jobs.size() < static_cast<size_t>(cfg_.max_batch_size);) {
                    if (equal_((*it)->key, key)) {
                        jobs.push_back(std::move(*it));
                        it = queue_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            try {
                std::vector<Request> requests;
                requests.reserve(jobs.size());
                for (auto& job : jobs) requests.push_back(std::move(job->request));
                auto results = run_batch_(key, std::move(requests));
                if (results.size() != jobs.size())
                    throw std::runtime_error("MicroBatcher: callback returned wrong result count");
                record_batch(jobs.size());
                for (size_t i = 0; i < jobs.size(); ++i)
                    jobs[i]->promise.set_value(std::move(results[i]));
            }
            catch (...) {
                const auto error = std::current_exception();
                for (auto& job : jobs) job->promise.set_exception(error);
            }
        }
    }

    void record_batch(size_t n) {
        batches_.fetch_add(1, std::memory_order_relaxed);
        items_.fetch_add(n, std::memory_order_relaxed);
        if (n == 1)
            singleton_batches_.fetch_add(1, std::memory_order_relaxed);
        uint64_t old = max_observed_batch_.load(std::memory_order_relaxed);
        while (old < n && !max_observed_batch_.compare_exchange_weak(
                              old, n, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }

    BatchingConfig cfg_;
    RunBatch run_batch_;
    KeyEqual equal_;
    mutable std::shared_mutex lifecycle_mu_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> queue_;
    bool stopping_ = false;
    std::thread worker_;
    std::atomic<uint64_t> batches_{0};
    std::atomic<uint64_t> items_{0};
    std::atomic<uint64_t> singleton_batches_{0};
    std::atomic<uint64_t> max_observed_batch_{0};
};

}  // namespace nemo_speech::asr
