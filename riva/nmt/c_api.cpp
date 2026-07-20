// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// C ABI implementation: maps the nemo_speech_nmt_* C surface onto the internal C++
// nmt::Translator. Every exported function is noexcept in spirit: exceptions are
// caught and turned into a status code plus a thread-local last-error string. No
// C++ types or exceptions cross the ABI.
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "nemo_speech/nmt.h"
#include "nmt_types.h"   // nmt::Translation
#include "translator.h"  // nmt::Translator + nmt::TranslatorConfig

namespace nmt_core = nemo_speech::nmt;

struct nemo_speech_nmt_translator {
    std::unique_ptr<nmt_core::Translator> impl;
};
struct nemo_speech_nmt_result {
    std::vector<nmt_core::Translation> translations;
};

namespace {

thread_local std::string g_last_error;
void
set_last_error(const std::string& m) {
    g_last_error = m;
}

// A field is present iff the caller's struct `size` covers it (append-only ABI).
#define HAS_FIELD(ptr, type, member) \
    ((ptr) != nullptr && (ptr)->size >= offsetof(type, member) + sizeof(((type*)nullptr)->member))

std::string
str_or_empty(const char* s) {
    return s ? std::string(s) : std::string();
}

nmt_core::TranslatorConfig
to_config(const nemo_speech_nmt_translator_config* c) {
    nmt_core::TranslatorConfig cfg;  // library defaults
    if (!c)
        return cfg;
    if (HAS_FIELD(c, nemo_speech_nmt_translator_config, backend) && c->backend) {
        const auto* b = c->backend;
        if (HAS_FIELD(b, nemo_speech_nmt_backend_config, gpu))
            cfg.backend.gpu = b->gpu;
    }
    if (HAS_FIELD(c, nemo_speech_nmt_translator_config, model) && c->model) {
        const auto* m = c->model;
        if (HAS_FIELD(m, nemo_speech_nmt_model_config, path))
            cfg.model.path = str_or_empty(m->path);
        if (HAS_FIELD(m, nemo_speech_nmt_model_config, n_ctx) && m->n_ctx > 0)
            cfg.model.n_ctx = m->n_ctx;
    }
    if (HAS_FIELD(c, nemo_speech_nmt_translator_config, generation) && c->generation) {
        const auto* g = c->generation;
        if (HAS_FIELD(g, nemo_speech_nmt_generation_config, max_new_tokens) &&
            g->max_new_tokens > 0)
            cfg.generation.max_new_tokens = g->max_new_tokens;
    }
    if (HAS_FIELD(c, nemo_speech_nmt_translator_config, pool) && c->pool) {
        const auto* p = c->pool;
        if (HAS_FIELD(p, nemo_speech_nmt_pool_config, contexts) && p->contexts > 0)
            cfg.pool.contexts = p->contexts;
    }
    return cfg;
}

// Wrap a body, mapping exceptions to status codes + last-error.
template <class F>
nemo_speech_nmt_status
guard(F&& f) {
    try {
        return f();
    }
    catch (const std::bad_alloc&) {
        set_last_error("out of memory");
        return NEMO_SPEECH_NMT_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::invalid_argument& e) {
        set_last_error(e.what());
        return NEMO_SPEECH_NMT_ERROR_INVALID_ARGUMENT;
    }
    catch (const std::exception& e) {
        set_last_error(e.what());
        return NEMO_SPEECH_NMT_ERROR_RUNTIME;
    }
    catch (...) {
        set_last_error("unknown error");
        return NEMO_SPEECH_NMT_ERROR_RUNTIME;
    }
}

}  // namespace

extern "C" {

nemo_speech_nmt_status
nemo_speech_nmt_create(
    const nemo_speech_nmt_translator_config* cfg, nemo_speech_nmt_translator** out) {
    if (!out)
        return NEMO_SPEECH_NMT_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return guard([&] {
        nmt_core::TranslatorConfig tc = to_config(cfg);
        if (tc.model.path.empty()) {
            set_last_error("translator config: model.path is required");
            return NEMO_SPEECH_NMT_ERROR_INVALID_ARGUMENT;
        }
        auto h = std::make_unique<nemo_speech_nmt_translator>();
        h->impl = std::make_unique<nmt_core::Translator>(std::move(tc));
        *out = h.release();
        return NEMO_SPEECH_NMT_OK;
    });
}

void
nemo_speech_nmt_destroy(nemo_speech_nmt_translator* translator) {
    delete translator;
}

nemo_speech_nmt_status
nemo_speech_nmt_translate(
    nemo_speech_nmt_translator* translator, const char* const* texts, size_t n_texts,
    const char* source_language, const char* target_language, nemo_speech_nmt_result** out) {
    if (!translator || !out || (!texts && n_texts > 0))
        return NEMO_SPEECH_NMT_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return guard([&] {
        std::vector<std::string> in;
        in.reserve(n_texts);
        for (size_t i = 0; i < n_texts; ++i) in.push_back(str_or_empty(texts[i]));
        auto r = std::make_unique<nemo_speech_nmt_result>();
        r->translations = translator->impl->translate(
            in, str_or_empty(source_language), str_or_empty(target_language));
        *out = r.release();
        return NEMO_SPEECH_NMT_OK;
    });
}

size_t
nemo_speech_nmt_result_count(const nemo_speech_nmt_result* result) {
    return result ? result->translations.size() : 0;
}

const char*
nemo_speech_nmt_result_text(const nemo_speech_nmt_result* result, size_t i) {
    if (!result || i >= result->translations.size())
        return nullptr;
    return result->translations[i].text.c_str();
}

const char*
nemo_speech_nmt_result_language(const nemo_speech_nmt_result* result, size_t i) {
    if (!result || i >= result->translations.size())
        return nullptr;
    return result->translations[i].language.c_str();
}

void
nemo_speech_nmt_result_destroy(nemo_speech_nmt_result* result) {
    delete result;
}

const char*
nemo_speech_nmt_last_error(void) {
    return g_last_error.c_str();
}

#ifndef NEMO_SPEECH_VERSION_STR
#define NEMO_SPEECH_VERSION_STR "0.0.0"  // overridden by the build from ./VERSION
#endif
const char*
nemo_speech_nmt_version(void) {
    return "nemo-speech-nmt " NEMO_SPEECH_VERSION_STR;
}

}  // extern "C"
