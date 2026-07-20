// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Internal C++ NMT translator: the single library entry point transports drive.
// Wraps llama.cpp to run the Riva-Translate decoder. Owns the model and a pool
// of decode contexts; translate() is safe to call from many threads (each call
// takes one context from the pool). llama.h is hidden behind a pimpl so
// consumers link this without seeing llama or ggml.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nmt_types.h"
#include "parameter_parser.h"

namespace nemo_speech::nmt {

struct BackendConfig {
    int gpu = 0;  // -1 = CPU
    void Register(common::ParameterParser& p) {
        p.Register("gpu", &gpu, "GPU device index (-1 = CPU)", {"--gpu", "-g"});
    }
};

struct ModelConfig {
    std::string path;
    // Sized for the sentence/short-turn translation that is the common case
    // (one request per context, bs=1), keeping the per-context KV cache small.
    // The model supports up to 8192, but degrades on inputs far longer than a
    // sentence anyway; raise this for document-length inputs at a linear KV
    // memory cost (~0.13 MiB/token), times pool.contexts.
    int n_ctx = 1024;
    void Register(common::ParameterParser& p) {
        p.Register("path", &path, "Path to the translation GGUF");
        p.Register("n_ctx", &n_ctx, "Decode context length in tokens (model max 8192)");
    }
};

struct GenerationConfig {
    int max_new_tokens = 256;
    void Register(common::ParameterParser& p) {
        p.Register("max_new_tokens", &max_new_tokens, "Max tokens generated per input text");
    }
};

struct PoolConfig {
    // Number of decode contexts in the pool. translate() is thread-safe and
    // hands each concurrent call its own context, so this caps how many
    // requests decode in parallel; extra callers block until one frees. Default
    // 1 keeps memory minimal (one KV cache); raise it for concurrency at the
    // cost of one extra n_ctx-sized KV cache per context.
    int contexts = 1;
    void Register(common::ParameterParser& p) {
        p.Register("contexts", &contexts, "Concurrent decode contexts (each adds one KV cache)");
    }
};

struct TranslatorConfig {
    BackendConfig backend;
    ModelConfig model;
    GenerationConfig generation;
    PoolConfig pool;
    bool verbose = false;
    void Register(common::ParameterParser& p) {
        p.Register("backend", backend);
        p.Register("model", model);
        p.Register("generation", generation);
        p.Register("pool", pool);
        p.Register("verbose", &verbose, "Enable verbose llama.cpp runtime logging");
    }
};

class Translator {
   public:
    explicit Translator(TranslatorConfig cfg);
    ~Translator();

    // Translate each text from the source to the target language, returning one
    // Translation per input (best-first). Throws std::runtime_error on an
    // unsupported language pair or a decode failure.
    std::vector<Translation> translate(
        const std::vector<std::string>& texts, const std::string& source_language,
        const std::string& target_language);

    const std::string& model_name() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::nmt
