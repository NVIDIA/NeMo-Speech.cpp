// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "translator.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "langpairs.h"
#include "llama.h"

namespace nemo_speech::nmt {

namespace {

std::atomic<bool> verbose_llama_logs{false};
std::atomic<bool> continue_llama_log{false};

void
llama_log_callback(ggml_log_level level, const char* text, void*) {
    bool emit = verbose_llama_logs.load(std::memory_order_relaxed);
    if (level == GGML_LOG_LEVEL_CONT)
        emit = emit || continue_llama_log.load(std::memory_order_relaxed);
    else {
        emit = emit || level >= GGML_LOG_LEVEL_WARN;
        continue_llama_log.store(emit, std::memory_order_relaxed);
    }
    if (emit)
        std::fputs(text, stderr);
}

void
configure_llama_logging(bool verbose) {
    verbose_llama_logs.store(verbose, std::memory_order_relaxed);
    continue_llama_log.store(false, std::memory_order_relaxed);
    llama_log_set(llama_log_callback, nullptr);
}

void
ensure_backend() {
    static std::once_flag once;
    std::call_once(once, [] { llama_backend_init(); });
}

// NMT-only entry points (test_nmt, the nemo_speech_nmt_* C ABI, direct lib use). The
// skinny-q8 ggml kernel (see ggml-patches/0005) is an ASR-encoder optimization
// whose in-place repack is unsafe for the decoder and whose padded single-token
// decode is slower than the stock path; with no ASR in the process, disable it
// entirely. The flag is read once, on the first skinny repack, so this runs in
// the Translator ctor, before any decode. putenv (not setenv) keeps it portable
// with no _WIN32 branch; the string must outlive the call, hence static. Defer
// to the server (it sets one of these for the combined ASR+NMT case): act only
// when neither is already set.
void
force_skinny_q8_safe_for_nmt() {
    if (std::getenv("GGML_SKINNY_Q8") == nullptr &&
        std::getenv("GGML_SKINNY_Q8_INPLACE") == nullptr) {
        static char kv[] = "GGML_SKINNY_Q8=0";
        putenv(kv);
    }
}

std::string
path_stem(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const std::size_t dot = base.find_last_of('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

std::string
ltrim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
                return !std::isspace(c);
            }));
    return s;
}

// RAII holders for the llama handles. If construction throws after some are
// allocated, the Impl members are destroyed during unwinding and free them
// (contexts/samplers before the model, by reverse declaration order), so the
// Translator destructor has nothing to do.
struct model_deleter {
    void operator()(llama_model* m) const noexcept { llama_model_free(m); }
};
struct context_deleter {
    void operator()(llama_context* c) const noexcept { llama_free(c); }
};
struct sampler_deleter {
    void operator()(llama_sampler* s) const noexcept { llama_sampler_free(s); }
};
using model_ptr = std::unique_ptr<llama_model, model_deleter>;
using context_ptr = std::unique_ptr<llama_context, context_deleter>;
using sampler_ptr = std::unique_ptr<llama_sampler, sampler_deleter>;

}  // namespace

struct Translator::Impl {
    struct Slot {
        context_ptr ctx;
        sampler_ptr smpl;
    };

    TranslatorConfig cfg;
    std::string model_name;
    model_ptr model;
    const llama_vocab* vocab = nullptr;
    std::vector<Slot> slots;

    std::mutex mu;
    std::condition_variable cv;
    std::vector<int> free_slots;

    int acquire() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !free_slots.empty(); });
        const int idx = free_slots.back();
        free_slots.pop_back();
        return idx;
    }

    void release(int idx) {
        {
            std::lock_guard<std::mutex> lk(mu);
            free_slots.push_back(idx);
        }
        cv.notify_one();
    }

    std::string detok(llama_token tok) {
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
        if (n >= 0)
            return std::string(buf, n);
        std::string big(-n, '\0');
        n = llama_token_to_piece(vocab, tok, big.data(), static_cast<int>(big.size()), 0, false);
        return std::string(big.data(), std::max(n, 0));
    }

    std::string run_one(const Slot& slot, const std::string& tag, const std::string& text) {
        const std::string prompt = langpairs::build_prompt(tag, text);

        const int n = -llama_tokenize(
            vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, false, true);
        std::vector<llama_token> tokens(n);
        llama_tokenize(
            vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), n, false,
            true);

        // The whole prompt must fit the context (it stays resident in the KV
        // cache while we generate). Fail with a clear message instead of a
        // cryptic decode error when an over-long input exceeds n_ctx.
        const int n_ctx = static_cast<int>(llama_n_ctx(slot.ctx.get()));
        if (n >= n_ctx)
            throw std::runtime_error(
                "nmt: prompt too long (" + std::to_string(n) + " tokens) for context " +
                std::to_string(n_ctx) + "; raise nmt.model.n_ctx");

        llama_memory_clear(llama_get_memory(slot.ctx.get()), true);
        // Prefill in batches: n_batch caps a single llama_decode, so a prompt
        // longer than it must be fed in chunks. Positions continue across the
        // calls (the memory is not cleared between them), as in the loop below.
        const int n_batch = static_cast<int>(llama_n_batch(slot.ctx.get()));
        for (int off = 0; off < n; off += n_batch) {
            const int cur = std::min(n_batch, n - off);
            if (llama_decode(slot.ctx.get(), llama_batch_get_one(tokens.data() + off, cur)) != 0)
                throw std::runtime_error("nmt: prefill decode failed");
        }

        std::string out;
        for (int i = 0; i < cfg.generation.max_new_tokens; ++i) {
            const llama_token tok = llama_sampler_sample(slot.smpl.get(), slot.ctx.get(), -1);
            if (llama_vocab_is_eog(vocab, tok))
                break;
            out += detok(tok);
            llama_token next = tok;
            if (llama_decode(slot.ctx.get(), llama_batch_get_one(&next, 1)) != 0)
                throw std::runtime_error("nmt: decode failed");
        }
        return ltrim(std::move(out));
    }
};

Translator::Translator(TranslatorConfig cfg) : impl_(std::make_unique<Impl>()) {
    force_skinny_q8_safe_for_nmt();
    configure_llama_logging(cfg.verbose);
    ensure_backend();
    impl_->cfg = std::move(cfg);
    impl_->model_name = path_stem(impl_->cfg.model.path);

    llama_model_params mp = llama_model_default_params();
    const bool gpu = impl_->cfg.backend.gpu >= 0;
    mp.n_gpu_layers = gpu ? 999 : 0;
    mp.main_gpu = gpu ? impl_->cfg.backend.gpu : 0;

    impl_->model.reset(llama_model_load_from_file(impl_->cfg.model.path.c_str(), mp));
    if (!impl_->model)
        throw std::runtime_error("nmt: failed to load model: " + impl_->cfg.model.path);
    impl_->vocab = llama_model_get_vocab(impl_->model.get());

    const int n = std::max(impl_->cfg.pool.contexts, 1);
    impl_->slots.resize(n);
    impl_->free_slots.reserve(n);
    for (int i = 0; i < n; ++i) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = impl_->cfg.model.n_ctx;
        // Small prefill batch sized for the bs=1 common case; run_one chunks any
        // longer prompt to this size, so a big n_batch buffer is not needed.
        cp.n_batch = std::min(impl_->cfg.model.n_ctx, 512);
        impl_->slots[i].ctx.reset(llama_init_from_model(impl_->model.get(), cp));
        if (!impl_->slots[i].ctx)
            throw std::runtime_error("nmt: failed to create decode context");
        impl_->slots[i].smpl.reset(llama_sampler_init_greedy());
        impl_->free_slots.push_back(i);
    }
}

// The llama handles are owned by RAII members of Impl, so destruction is just
// impl_ going away (which frees contexts/samplers before the model).
Translator::~Translator() = default;

std::vector<Translation>
Translator::translate(
    const std::vector<std::string>& texts, const std::string& source_language,
    const std::string& target_language) {
    const std::string tag = langpairs::resolve_tag(source_language, target_language);
    if (tag.empty())
        throw std::invalid_argument(
            "nmt: unsupported language pair: " + source_language + " -> " + target_language);
    // Report the language actually decoded (the tag's target side), which can
    // differ from the raw target_language when a ready tag was passed in.
    const std::string out_language = langpairs::split_tag(tag).second;

    std::vector<Translation> out;
    out.reserve(texts.size());
    for (const auto& text : texts) {
        const int idx = impl_->acquire();
        std::string translated;
        try {
            translated = impl_->run_one(impl_->slots[idx], tag, text);
        }
        catch (...) {
            impl_->release(idx);
            throw;
        }
        impl_->release(idx);
        out.push_back(Translation{std::move(translated), out_language});
    }
    return out;
}

const std::string&
Translator::model_name() const {
    return impl_->model_name;
}

}  // namespace nemo_speech::nmt
