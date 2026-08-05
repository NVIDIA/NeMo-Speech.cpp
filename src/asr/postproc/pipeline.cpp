// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "itn_align.h"
#include "pnc_model.h"
#include "pnc_runner.h"

namespace nemo_speech::asr::postproc {
namespace {

#ifdef NEMO_SPEECH_WITH_NORM
bool
has_runtime_fars(const std::filesystem::path& dir) {
    return std::filesystem::is_regular_file(dir / "tokenize_and_classify.far") &&
           std::filesystem::is_regular_file(dir / "verbalize.far");
}
#endif

std::string
normalize_language_code(const std::string& language_code) {
    std::string normalized;
    normalized.reserve(language_code.size());
    for (unsigned char c : language_code) {
        if (c == '_') {
            normalized.push_back('-');
        } else if (!std::isspace(c)) {
            normalized.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return normalized;
}

// Verbatim/no-punctuation rendering for enable_automatic_punctuation=false:
// lowercase and drop sentence punctuation, while keeping word characters,
// digits, apostrophes/hyphens, the profanity mask ('*'), other symbols, and
// non-ASCII (UTF-8) bytes. A no-op on already-plain CTC output; strips the
// casing + punctuation a self-punctuating model bakes in.
std::string
strip_formatting(const std::string& in) {
    static const std::string kDrop = ".,?!;:\"()[]{}";  // sentence punctuation
    std::string out;
    out.reserve(in.size());
    bool pending_space = false;
    for (unsigned char c : in) {
        if (c == ' ') {
            pending_space = !out.empty();  // collapse runs; suppress leading
            continue;
        }
        if (c < 0x80 && kDrop.find(static_cast<char>(c)) != std::string::npos)
            continue;  // drop punctuation
        const char ch =
            (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);
        if (pending_space) {
            out.push_back(' ');
            pending_space = false;
        }
        out.push_back(ch);
    }
    return out;
}

std::string
sanitize_punctuation_spacing(const std::string& in) {
    static const std::string kAttachLeft = ".,?!;:%)]}-";
    static const std::string kAttachRight = "([{-";
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        if (c < 0x80 && kAttachLeft.find(static_cast<char>(c)) != std::string::npos) {
            while (!out.empty() && out.back() == ' ') out.pop_back();
        }
        out.push_back(static_cast<char>(c));
        if (c < 0x80 && kAttachRight.find(static_cast<char>(c)) != std::string::npos) {
            while (i + 1 < in.size() && in[i + 1] == ' ') ++i;
        }
    }
    return out;
}

}  // namespace

struct Postprocessor::ItnRegistry {
    explicit ItnRegistry(const std::string& configured_dir) {
        if (configured_dir.empty())
            return;
#ifdef NEMO_SPEECH_WITH_NORM
        std::error_code ec;
        const std::filesystem::path root = std::filesystem::canonical(configured_dir, ec);
        if (ec || !std::filesystem::is_directory(root))
            throw std::runtime_error("ITN: cannot resolve grammar directory: " + configured_dir);
        if (has_runtime_fars(root)) {
            fallback = std::make_unique<Itn>(root.string());
            return;
        }
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (!entry.is_directory() || !has_runtime_fars(entry.path()))
                continue;
            const std::string language = normalize_language_code(entry.path().filename().string());
            const auto [it, inserted] = grammar_dirs.emplace(language, entry.path().string());
            if (!inserted) {
                throw std::runtime_error(
                    "ITN: duplicate normalized language directory '" + language +
                    "': " + it->second + " and " + entry.path().string());
            }
        }
        if (grammar_dirs.empty()) {
            throw std::runtime_error(
                "ITN: no two-FAR grammar directories found under " + root.string());
        }
        std::cerr << "[itn] discovered " << grammar_dirs.size()
                  << " language grammar directories under " << root.string() << "\n";
#else
        // Preserve the existing warning for a configured grammar in a build
        // without text-normalization support.
        fallback = std::make_unique<Itn>(configured_dir);
#endif
    }

    const Itn* select(const std::string& requested_language) const {
        if (fallback)
            return fallback.get();
        if (grammar_dirs.empty())
            return nullptr;

        const std::string language = normalize_language_code(requested_language);
        if (language == "auto")
            return nullptr;  // The recognizer replaces auto with its detected code.

        auto find_path = [&](const std::string& key) -> const std::string* {
            const auto it = grammar_dirs.find(key);
            return it == grammar_dirs.end() ? nullptr : &it->second;
        };
        const std::string* path = nullptr;
        if (!language.empty()) {
            path = find_path(language);
            if (!path) {
                const size_t separator = language.find('-');
                if (separator != std::string::npos)
                    path = find_path(language.substr(0, separator));
            }
        } else {
            path = find_path("en-us");
            if (!path)
                path = find_path("en");
            if (!path && grammar_dirs.size() == 1)
                path = &grammar_dirs.begin()->second;
        }
        if (!path)
            return nullptr;

        std::lock_guard<std::mutex> lock(mu);
        const auto found = loaded.find(*path);
        if (found != loaded.end())
            return found->second.get();
        auto normalizer = std::make_unique<Itn>(*path);
        const Itn* selected = normalizer.get();
        loaded.emplace(*path, std::move(normalizer));
        return selected;
    }

    std::unique_ptr<Itn> fallback;
    std::map<std::string, std::string> grammar_dirs;
    mutable std::mutex mu;
    mutable std::map<std::string, std::unique_ptr<Itn>> loaded;
};

class Postprocessor::Executor {
   public:
    Executor(int workers, int max_queue) : max_queue_(static_cast<size_t>(std::max(1, max_queue))) {
        workers = std::max(1, workers);
        threads_.reserve(static_cast<size_t>(workers));
        for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { worker(); });
    }

    ~Executor() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopping_ = true;
        }
        has_work_.notify_all();
        has_room_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    std::string run(std::function<std::string()> fn, double* queue_ms) {
        if (current_ == this)
            return fn();
        using Clock = std::chrono::steady_clock;
        const auto submitted = Clock::now();
        auto task = std::make_shared<std::packaged_task<std::string()>>(
            [fn = std::move(fn), submitted, queue_ms] {
                if (queue_ms) {
                    *queue_ms =
                        std::chrono::duration<double, std::milli>(Clock::now() - submitted).count();
                }
                return fn();
            });
        auto future = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mu_);
            has_room_.wait(lock, [&] { return stopping_ || queue_.size() < max_queue_; });
            if (stopping_)
                throw std::runtime_error("postprocessing executor is stopping");
            queue_.emplace_back([task] { (*task)(); });
        }
        has_work_.notify_one();
        return future.get();
    }

   private:
    void worker() {
        current_ = this;
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                has_work_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty())
                    break;
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            has_room_.notify_one();
            task();
        }
        current_ = nullptr;
    }

    static thread_local Executor* current_;
    size_t max_queue_;
    std::mutex mu_;
    std::condition_variable has_work_, has_room_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> threads_;
    bool stopping_ = false;
};

thread_local Postprocessor::Executor* Postprocessor::Executor::current_ = nullptr;

Postprocessor::Postprocessor(
    const PostprocConfig& cfg, ggml_runtime::BackendManager* bm, bool model_self_punctuates,
    const BatchingConfig& batching)
    : profanity_(cfg.profanity_list_path), itn_(std::make_unique<ItnRegistry>(cfg.itn_model_dir)) {
    if (!cfg.pnc_model_path.empty()) {
        if (model_self_punctuates) {
            std::cerr << "[pnc] WARNING: model emits its own punctuation/capitalization; "
                         "ignoring --pnc-model (the PnC BERT targets plain-text CTC models; "
                         "stacking it here would double/garble punctuation).\n";
        } else if (bm == nullptr) {
            std::cerr << "[pnc] WARNING: --pnc-model set but no backend; PnC disabled.\n";
        } else {
            pnc_model_ = std::make_unique<pnc::PncModel>(*bm, cfg.pnc_model_path, batching);
            pnc_runner_ = std::make_unique<pnc::PncRunner>(pnc_model_.get());
        }
    }
    executor_ = std::make_unique<Executor>(cfg.cpu_workers, cfg.max_queue_depth);
}

Postprocessor::~Postprocessor() = default;

BatchMetrics
Postprocessor::pnc_batch_metrics() const {
    return pnc_model_ ? pnc_model_->batch_metrics() : BatchMetrics{};
}

std::string
Postprocessor::apply(
    const std::string& transcript, const AsrRequestOptions& opts, std::vector<WordTiming>* words,
    const std::string& language_code) const {
    static const bool t_log = std::getenv("NEMO_SPEECH_TIMING") != nullptr;
    const auto begin = std::chrono::steady_clock::now();
    double queue_ms = 0.0;
    auto text = executor_->run(
        [this, &transcript, &opts, words, &language_code] {
            return apply_cpu(transcript, opts, words, language_code);
        },
        &queue_ms);

    const auto pnc_begin = std::chrono::steady_clock::now();
    if (pnc_runner_ && opts.enable_automatic_punctuation)
        text = pnc_runner_->postprocess(text);
    const auto pnc_end = std::chrono::steady_clock::now();

    if (opts.enable_automatic_punctuation)
        text = sanitize_punctuation_spacing(text);
    else
        text = strip_formatting(text);

    if (t_log) {
        const double total_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
        const double pnc_ms =
            std::chrono::duration<double, std::milli>(pnc_end - pnc_begin).count();
        std::fprintf(
            stderr, "[timing] postproc-dispatch queue=%.2f pnc=%.2f total=%.2f ms\n", queue_ms,
            pnc_ms, total_ms);
    }
    return text;
}

std::string
Postprocessor::apply_cpu(
    const std::string& transcript, const AsrRequestOptions& opts, std::vector<WordTiming>* words,
    const std::string& language_code) const {
    std::string text = transcript;

    static const bool t_log = std::getenv("NEMO_SPEECH_TIMING") != nullptr;
    using _clk = std::chrono::high_resolution_clock;
    const auto _t0 = _clk::now();

    // Ordering matches Riva: profanity, ITN, then PnC.
    if (opts.profanity_filter && profanity_.enabled()) {
        text = profanity_.mask(text);
        if (words) {
            for (auto& w : *words) w.word = profanity_.mask(w.word);
        }
    }
    const auto _t1 = _clk::now();
    const Itn* normalizer = itn_->select(language_code);
    if (normalizer && normalizer->enabled() && !opts.verbatim_transcripts) {
        std::string alignment;
        const std::string normalized =
            normalizer->normalize(text, words && !words->empty() ? &alignment : nullptr);
        if (words && !words->empty() && normalized != text)
            update_word_timings(*words, alignment);
        text = normalized;
    }
    const auto _t2 = _clk::now();
    if (t_log) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(
            stderr, "[timing] postproc-cpu chars=%zu profanity=%.2f itn=%.2f ms\n", text.size(),
            ms(_t0, _t1), ms(_t1, _t2));
    }
    return text;
}

}  // namespace nemo_speech::asr::postproc
