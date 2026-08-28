// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "decoder.h"
#include "encoder.h"
#include "graph.h"
#include "lt.h"
#include "nvtx_utils.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif

namespace nemo_speech::tts {

const char*
magpietts_backend_preference_name(magpietts_backend_preference backend) {
    switch (backend) {
        case MAGPIETTS_BACKEND_AUTO:
            return "auto";
        case MAGPIETTS_BACKEND_CPU:
            return "cpu";
        case MAGPIETTS_BACKEND_CUDA:
            return "cuda";
    }
    return "unknown";
}

bool
parse_backend_preference(const std::string& value, magpietts_backend_preference& backend) {
    if (value == "auto") {
        backend = MAGPIETTS_BACKEND_AUTO;
        return true;
    }
    if (value == "cpu") {
        backend = MAGPIETTS_BACKEND_CPU;
        return true;
    }
    if (value == "cuda") {
        backend = MAGPIETTS_BACKEND_CUDA;
        return true;
    }
    return false;
}

const char*
magpietts_uma_mode_name(magpietts_uma_mode mode) {
    switch (mode) {
        case MAGPIETTS_UMA_AUTO:
            return "auto";
        case MAGPIETTS_UMA_OFF:
            return "off";
        case MAGPIETTS_UMA_ON:
            return "on";
    }
    return "unknown";
}

bool
parse_uma_mode(const std::string& value, magpietts_uma_mode& mode) {
    if (value == "auto") {
        mode = MAGPIETTS_UMA_AUTO;
        return true;
    }
    if (value == "off") {
        mode = MAGPIETTS_UMA_OFF;
        return true;
    }
    if (value == "on") {
        mode = MAGPIETTS_UMA_ON;
        return true;
    }
    return false;
}

BackendTensor::~BackendTensor() {
    reset();
}

BackendTensor::BackendTensor(BackendTensor&& other) noexcept {
    *this = std::move(other);
}

BackendTensor&
BackendTensor::operator=(BackendTensor&& other) noexcept {
    if (this != &other) {
        reset();
        ctx = other.ctx;
        buffer = other.buffer;
        tensor = other.tensor;
        other.ctx = nullptr;
        other.buffer = nullptr;
        other.tensor = nullptr;
    }
    return *this;
}

void
BackendTensor::reset() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    tensor = nullptr;
}

bool
BackendTensor::alloc2d(
    const magpietts_model& model, enum ggml_type type, int64_t ne0, int64_t ne1, const char* name) {
    if (tensor && buffer && tensor->type == type && tensor->ne[0] == ne0 && tensor->ne[1] == ne1) {
        ggml_set_name(tensor, name);
        return true;
    }

    reset();
    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to allocate persistent tensor context for %s\n", name);
        return false;
    }
    tensor = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(tensor, name);
    buffer = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    if (!buffer) {
        fprintf(
            stderr, "failed to allocate persistent tensor %s on backend %s\n", name,
            ggml_backend_name(model.backend));
        reset();
        return false;
    }
    return true;
}

bool
magpietts_backend_is_cuda(ggml_backend_t backend) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    return ggml_backend_is_cuda(backend);
#else
    (void)backend;
    return false;
#endif
}

static constexpr size_t MAGPIETTS_PINNED_STAGING_MIN_BYTES = 4 * 1024;

MagpiePinnedHostScratch::~MagpiePinnedHostScratch() {
    reset();
}

MagpiePinnedHostScratch::MagpiePinnedHostScratch(MagpiePinnedHostScratch&& other) noexcept {
    *this = std::move(other);
}

MagpiePinnedHostScratch&
MagpiePinnedHostScratch::operator=(MagpiePinnedHostScratch&& other) noexcept {
    if (this != &other) {
        reset();
        buffer_ = other.buffer_;
        fallback_ = std::move(other.fallback_);
        capacity_ = other.capacity_;
        pinned_ = other.pinned_;
        data_ = buffer_ ? (uint8_t*)ggml_backend_buffer_get_base(buffer_)
                        : (fallback_.empty() ? nullptr : fallback_.data());

        other.buffer_ = nullptr;
        other.data_ = nullptr;
        other.capacity_ = 0;
        other.pinned_ = false;
    }
    return *this;
}

void
MagpiePinnedHostScratch::reset() {
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    fallback_.clear();
    data_ = nullptr;
    capacity_ = 0;
    pinned_ = false;
}

bool
MagpiePinnedHostScratch::reserve(const magpietts_model& model, size_t size) {
    if (size == 0) {
        return true;
    }

#if defined(MAGPIETTS_CUDA_SAMPLING)
    if (magpietts_backend_is_cuda(model.backend) && std::getenv("GGML_CUDA_NO_PINNED") == nullptr) {
        if (buffer_ && capacity_ >= size) {
            return true;
        }

        reset();
        ggml_backend_buffer_type_t buft = ggml_backend_cuda_host_buffer_type();
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
        if (buffer && ggml_backend_buffer_get_type(buffer) == buft) {
            void* base = ggml_backend_buffer_get_base(buffer);
            if (base) {
                buffer_ = buffer;
                data_ = (uint8_t*)base;
                capacity_ = ggml_backend_buffer_get_size(buffer_);
                pinned_ = true;
                return true;
            }
        }
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
#else
    (void)model;
#endif

    if (buffer_) {
        reset();
    }
    if (fallback_.size() < size) {
        fallback_.resize(size);
    }
    data_ = fallback_.empty() ? nullptr : fallback_.data();
    capacity_ = fallback_.size();
    pinned_ = false;
    return data_ != nullptr;
}

bool
magpietts_should_stage_pinned(const magpietts_model& model, size_t nbytes) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    return nbytes >= MAGPIETTS_PINNED_STAGING_MIN_BYTES &&
           magpietts_backend_is_cuda(model.backend) &&
           std::getenv("GGML_CUDA_NO_PINNED") == nullptr;
#else
    (void)model;
    (void)nbytes;
    return false;
#endif
}

void
magpietts_backend_tensor_set_staged(
    const magpietts_model& model, MagpiePinnedHostScratch& scratch, ggml_tensor* tensor,
    const void* data, size_t offset, size_t nbytes) {
    if (magpietts_should_stage_pinned(model, nbytes) && scratch.reserve(model, nbytes) &&
        scratch.pinned()) {
        std::memcpy(scratch.data(), data, nbytes);
        ggml_backend_tensor_set(tensor, scratch.data(), offset, nbytes);
        return;
    }
    ggml_backend_tensor_set(tensor, data, offset, nbytes);
}

void
magpietts_backend_tensor_get_staged(
    const magpietts_model& model, MagpiePinnedHostScratch& scratch, const ggml_tensor* tensor,
    void* data, size_t offset, size_t nbytes) {
    if (magpietts_should_stage_pinned(model, nbytes) && scratch.reserve(model, nbytes) &&
        scratch.pinned()) {
        ggml_backend_tensor_get(tensor, scratch.data(), offset, nbytes);
        std::memcpy(data, scratch.data(), nbytes);
        return;
    }
    ggml_backend_tensor_get(tensor, data, offset, nbytes);
}

static bool
magpietts_set_env(const char* name, const char* value) {
#if defined(_WIN32)
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

static bool
magpietts_unset_env(const char* name) {
#if defined(_WIN32)
    return _putenv_s(name, "") == 0;
#else
    return unsetenv(name) == 0;
#endif
}

static bool
magpietts_configure_unified_memory(magpietts_uma_mode mode) {
    const char* env_name = "GGML_CUDA_ENABLE_UNIFIED_MEMORY";
    if (mode == MAGPIETTS_UMA_OFF) {
        if (std::getenv(env_name) && !magpietts_unset_env(env_name)) {
            fprintf(stderr, "warning: failed to unset %s\n", env_name);
        }
        return false;
    }
    if (mode == MAGPIETTS_UMA_ON) {
        if (!magpietts_set_env(env_name, "1")) {
            fprintf(stderr, "warning: failed to set %s=1\n", env_name);
        }
        return true;
    }
    return std::getenv(env_name) != nullptr;
}

static int32_t
gguf_i32(const gguf_context* ctx, const char* key, int32_t def) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    const gguf_type t = gguf_get_kv_type(ctx, id);
    if (t == GGUF_TYPE_INT32) {
        return gguf_get_val_i32(ctx, id);
    }
    if (t == GGUF_TYPE_UINT32) {
        return (int32_t)gguf_get_val_u32(ctx, id);
    }
    if (t == GGUF_TYPE_INT64) {
        return (int32_t)gguf_get_val_i64(ctx, id);
    }
    return def;
}

static float
gguf_f32(const gguf_context* ctx, const char* key, float def) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    const gguf_type t = gguf_get_kv_type(ctx, id);
    if (t == GGUF_TYPE_FLOAT32) {
        return gguf_get_val_f32(ctx, id);
    }
    if (t == GGUF_TYPE_FLOAT64) {
        return (float)gguf_get_val_f64(ctx, id);
    }
    return def;
}

static bool
gguf_bool(const gguf_context* ctx, const char* key, bool def) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    const gguf_type t = gguf_get_kv_type(ctx, id);
    if (t == GGUF_TYPE_BOOL) {
        return gguf_get_val_bool(ctx, id);
    }
    if (t == GGUF_TYPE_INT32) {
        return gguf_get_val_i32(ctx, id) != 0;
    }
    if (t == GGUF_TYPE_UINT32) {
        return gguf_get_val_u32(ctx, id) != 0;
    }
    return def;
}

static std::string
gguf_string(const gguf_context* ctx, const char* key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return {};
    }
    const char* value = gguf_get_val_str(ctx, id);
    return value ? std::string(value) : std::string();
}

static std::vector<int32_t>
gguf_i32_array(const gguf_context* ctx, const char* key) {
    std::vector<int32_t> out;
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
        return out;
    }
    const gguf_type t = gguf_get_arr_type(ctx, id);
    const size_t n = gguf_get_arr_n(ctx, id);
    const void* data = gguf_get_arr_data(ctx, id);
    if (!data) {
        return out;
    }
    out.reserve(n);
    if (t == GGUF_TYPE_INT32) {
        const int32_t* p = static_cast<const int32_t*>(data);
        out.assign(p, p + n);
    } else if (t == GGUF_TYPE_INT64) {
        const int64_t* p = static_cast<const int64_t*>(data);
        for (size_t i = 0; i < n; ++i) {
            out.push_back((int32_t)p[i]);
        }
    } else if (t == GGUF_TYPE_UINT32) {
        const uint32_t* p = static_cast<const uint32_t*>(data);
        for (size_t i = 0; i < n; ++i) {
            out.push_back((int32_t)p[i]);
        }
    } else if (t == GGUF_TYPE_UINT64) {
        const uint64_t* p = static_cast<const uint64_t*>(data);
        for (size_t i = 0; i < n; ++i) {
            out.push_back((int32_t)p[i]);
        }
    }
    return out;
}

// Minimal config_json extractor used only for legacy Magpie GGUF metadata.
// Supported input is a flat object with double-quoted keys and scalar values,
// plus simple integer arrays consumed by json_i32_array, for example:
// {"apply_attention_prior":true,"attention_prior_lookahead_window":5,
//  "estimate_alignment_from_layers":[0,1,2]}
// It does not parse nested objects, escaped quotes in strings, or general JSON
// arrays. json_bool, json_i32, json_f32, and json_i32_array all rely on this:
// missing keys and invalid scalar values return their caller-provided defaults,
// while integer arrays stop at the first unparsable item. Switch to a full JSON
// parser if the embedded config format grows beyond this subset.
static const char*
json_key_value(const std::string& json, const char* key) {
    const std::string quoted = "\"" + std::string(key) + "\"";
    const size_t key_pos = json.find(quoted);
    if (key_pos == std::string::npos) {
        return nullptr;
    }
    const size_t colon = json.find(':', key_pos + quoted.size());
    if (colon == std::string::npos) {
        return nullptr;
    }
    const char* p = json.c_str() + colon + 1;
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') {
        ++p;
    }
    return p;
}

static bool
json_bool(const std::string& json, const char* key, bool def) {
    const char* p = json_key_value(json, key);
    if (!p) {
        return def;
    }
    if (std::strncmp(p, "true", 4) == 0) {
        return true;
    }
    if (std::strncmp(p, "false", 5) == 0) {
        return false;
    }
    return def;
}

static int32_t
json_i32(const std::string& json, const char* key, int32_t def) {
    const char* p = json_key_value(json, key);
    if (!p) {
        return def;
    }
    char* end = nullptr;
    const long parsed = std::strtol(p, &end, 10);
    return end != p ? (int32_t)parsed : def;
}

static float
json_f32(const std::string& json, const char* key, float def) {
    const char* p = json_key_value(json, key);
    if (!p) {
        return def;
    }
    char* end = nullptr;
    const float parsed = std::strtof(p, &end);
    return end != p ? parsed : def;
}

static std::vector<int32_t>
json_i32_array(const std::string& json, const char* key) {
    std::vector<int32_t> out;
    const char* p = json_key_value(json, key);
    if (!p || *p != '[') {
        return out;
    }
    ++p;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == ',') {
            ++p;
        }
        if (*p == ']') {
            break;
        }
        char* end = nullptr;
        const long parsed = std::strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        out.push_back((int32_t)parsed);
        p = end;
    }
    return out;
}

static std::string
format_i32_list(const std::vector<int32_t>& values) {
    if (values.empty()) {
        return "all";
    }
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << values[i];
    }
    return out.str();
}

static ggml_tensor*
require_tensor(const magpietts_model& model, const std::string& name) {
    ggml_tensor* t = ggml_get_tensor(model.ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "missing tensor: %s\n", name.c_str());
        std::exit(1);
    }
    return t;
}

static std::string
layer_prefix(const char* block, int il) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s.layers.%d.", block, il);
    return std::string(buf);
}

static void
load_transformer(
    magpietts_model& model, magpietts_transformer& tr, const char* prefix, int n_layer, int n_embd,
    int n_head, int kernel, bool causal, bool has_cross, int n_cross_head, int n_cross_dhead) {
    tr.n_embd = n_embd;
    tr.n_head = n_head;
    tr.kernel = kernel;
    tr.causal = causal;
    tr.has_cross = has_cross;
    tr.n_cross_head = n_cross_head;
    tr.n_cross_dhead = n_cross_dhead;
    tr.norm_out = ggml_get_tensor(model.ctx, (std::string(prefix) + ".norm_out.weight").c_str());
    if (!tr.norm_out && std::string(prefix) != "local_transformer") {
        fprintf(stderr, "missing tensor: %s\n", (std::string(prefix) + ".norm_out.weight").c_str());
        std::exit(1);
    }
    tr.pos_emb = require_tensor(model, std::string(prefix) + ".position_embeddings.weight");
    tr.layers.resize(n_layer);

    for (int il = 0; il < n_layer; ++il) {
        magpietts_layer& layer = tr.layers[il];
        const std::string p = layer_prefix(prefix, il);
        layer.has_cross = has_cross;
        layer.kernel = kernel;
        layer.norm_self = require_tensor(model, p + "norm_self.weight");
        layer.self_qkv = require_tensor(model, p + "self_attention.qkv_net.weight");
        layer.self_o = require_tensor(model, p + "self_attention.o_net.weight");
        if (has_cross) {
            layer.norm_xattn_query = require_tensor(model, p + "norm_xattn_query.weight");
            layer.cross_q = require_tensor(model, p + "cross_attention.q_net.weight");
            layer.cross_kv = require_tensor(model, p + "cross_attention.kv_net.weight");
            layer.cross_o = require_tensor(model, p + "cross_attention.o_net.weight");
            layer.norm_xattn_memory = require_tensor(model, p + "norm_xattn_memory.weight");
        }
        layer.norm_ff = require_tensor(model, p + "norm_pos_ff.weight");
        layer.ff_proj.resize(kernel);
        layer.ff_out.resize(kernel);
        for (int k = 0; k < kernel; ++k) {
            layer.ff_proj[k] =
                require_tensor(model, p + "pos_ff.proj.conv.weight.k" + std::to_string(k));
            layer.ff_out[k] =
                require_tensor(model, p + "pos_ff.o_net.conv.weight.k" + std::to_string(k));
        }
    }
}

static bool magpietts_model_load_impl(
    const std::string& fname, magpietts_model& model, magpietts_uma_mode uma_mode, bool force_cpu,
    bool verbose);

MagpieModel::~MagpieModel() {
    reset();
}

MagpieModel::MagpieModel(MagpieModel&& other) noexcept {
    *this = std::move(other);
}

MagpieModel&
MagpieModel::operator=(MagpieModel&& other) noexcept {
    if (this != &other) {
        reset();
        hparams = other.hparams;
        tokenizer_profile = std::move(other.tokenizer_profile);
        nemo_version = std::move(other.nemo_version);
        gguf = other.gguf;
        ctx = other.ctx;
        backend = other.backend;
        buffer = other.buffer;
        cuda_unified_memory = other.cuda_unified_memory;
        text_embedding = other.text_embedding;
        audio_embeddings = std::move(other.audio_embeddings);
        baked_context = other.baked_context;
        final_proj_w = other.final_proj_w;
        final_proj_b = other.final_proj_b;
        lt_in_w = other.lt_in_w;
        lt_in_b = other.lt_in_b;
        lt_out_w = std::move(other.lt_out_w);
        lt_out_b = std::move(other.lt_out_b);
        encoder = std::move(other.encoder);
        decoder = std::move(other.decoder);
        local = std::move(other.local);

        other.gguf = nullptr;
        other.ctx = nullptr;
        other.backend = nullptr;
        other.buffer = nullptr;
        other.cuda_unified_memory = false;
        other.text_embedding = nullptr;
        other.baked_context = nullptr;
        other.final_proj_w = nullptr;
        other.final_proj_b = nullptr;
        other.lt_in_w = nullptr;
        other.lt_in_b = nullptr;
    }
    return *this;
}

bool
MagpieModel::load(
    const std::string& fname, magpietts_uma_mode uma_mode, bool force_cpu, bool verbose) {
    if (!magpietts_model_load_impl(fname, *this, uma_mode, force_cpu, verbose)) {
        reset();
        return false;
    }
    return true;
}

void
MagpieModel::reset() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (backend) {
        ggml_backend_free(backend);
        backend = nullptr;
    }
    if (gguf) {
        gguf_free(gguf);
        gguf = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    hparams = {};
    tokenizer_profile.clear();
    nemo_version.clear();
    cuda_unified_memory = false;
    text_embedding = nullptr;
    audio_embeddings.clear();
    baked_context = nullptr;
    final_proj_w = nullptr;
    final_proj_b = nullptr;
    lt_in_w = nullptr;
    lt_in_b = nullptr;
    lt_out_w.clear();
    lt_out_b.clear();
    encoder = {};
    decoder = {};
    local = {};
}

static bool
magpietts_model_load_impl(
    const std::string& fname, magpietts_model& model, magpietts_uma_mode uma_mode, bool force_cpu,
    bool verbose) {
    const ggml_nvtx::range nvtx_range("magpietts_model_load");
    model.reset();

    gguf_init_params params = {
        /*.no_alloc =*/true,
        /*.ctx      =*/&model.ctx,
    };
    model.gguf = gguf_init_from_file(fname.c_str(), params);
    if (!model.gguf || !model.ctx) {
        fprintf(stderr, "failed to load GGUF: %s\n", fname.c_str());
        return false;
    }

    auto& h = model.hparams;
    h.text_vocab_size = gguf_i32(model.gguf, "magpietts.text_vocab_size", h.text_vocab_size);
    h.audio_codebooks = gguf_i32(model.gguf, "magpietts.audio_codebooks", h.audio_codebooks);
    h.audio_codebook_size =
        gguf_i32(model.gguf, "magpietts.audio_codebook_size", h.audio_codebook_size);
    h.audio_vocab_size = gguf_i32(model.gguf, "magpietts.audio_vocab_size", h.audio_vocab_size);
    h.audio_bos_id = gguf_i32(model.gguf, "magpietts.audio_bos_id", h.audio_bos_id);
    h.audio_eos_id = gguf_i32(model.gguf, "magpietts.audio_eos_id", h.audio_eos_id);
    h.mask_token_id = gguf_i32(model.gguf, "magpietts.mask_token_id", h.mask_token_id);
    h.frame_stacking_factor =
        gguf_i32(model.gguf, "magpietts.frame_stacking_factor", h.frame_stacking_factor);
    if (h.audio_codebooks < 1 || h.frame_stacking_factor < 1) {
        fprintf(
            stderr, "invalid stacked audio layout: codebooks=%d frame_stacking_factor=%d\n",
            h.audio_codebooks, h.frame_stacking_factor);
        return false;
    }
    const int64_t stacked_audio_codebooks_64 =
        static_cast<int64_t>(h.audio_codebooks) * h.frame_stacking_factor;
    if (stacked_audio_codebooks_64 > std::numeric_limits<int32_t>::max()) {
        fprintf(
            stderr,
            "stacked audio layout overflows int32: codebooks=%d frame_stacking_factor=%d "
            "slots=%lld\n",
            h.audio_codebooks, h.frame_stacking_factor,
            static_cast<long long>(stacked_audio_codebooks_64));
        return false;
    }
    const int32_t expected_stacked_codebooks = static_cast<int32_t>(stacked_audio_codebooks_64);
    model.tokenizer_profile = gguf_string(model.gguf, "magpietts.tokenizer_profile");
    model.nemo_version = gguf_string(model.gguf, "magpietts.nemo_version");
    const int32_t stored_stacked_codebooks =
        gguf_i32(model.gguf, "magpietts.stacked_audio_codebooks", expected_stacked_codebooks);
    h.n_embd = gguf_i32(model.gguf, "magpietts.embedding_dim", h.n_embd);
    h.n_ffn = gguf_i32(model.gguf, "magpietts.ffn_dim", h.n_ffn);
    h.n_ctx = gguf_i32(model.gguf, "magpietts.context_length", h.n_ctx);
    h.n_enc_layer = gguf_i32(model.gguf, "magpietts.encoder.layers", h.n_enc_layer);
    h.n_enc_head = gguf_i32(model.gguf, "magpietts.encoder.heads", h.n_enc_head);
    h.enc_kernel = gguf_i32(model.gguf, "magpietts.encoder.kernel_size", h.enc_kernel);
    h.n_dec_layer = gguf_i32(model.gguf, "magpietts.decoder.layers", h.n_dec_layer);
    h.n_dec_head = gguf_i32(model.gguf, "magpietts.decoder.heads", h.n_dec_head);
    h.n_cross_head = gguf_i32(model.gguf, "magpietts.decoder.cross_heads", h.n_cross_head);
    h.n_cross_dhead = gguf_i32(model.gguf, "magpietts.decoder.cross_head_dim", h.n_cross_dhead);
    h.dec_kernel = gguf_i32(model.gguf, "magpietts.decoder.kernel_size", h.dec_kernel);
    h.lt_layers = gguf_i32(model.gguf, "magpietts.local_transformer.layers", h.lt_layers);
    h.lt_heads = gguf_i32(model.gguf, "magpietts.local_transformer.heads", h.lt_heads);
    h.lt_hidden = gguf_i32(model.gguf, "magpietts.local_transformer.hidden_dim", h.lt_hidden);
    h.lt_ctx = gguf_i32(model.gguf, "magpietts.local_transformer.context_length", h.lt_ctx);
    h.baked_context_length =
        gguf_i32(model.gguf, "magpietts.baked_context_length", h.baked_context_length);
    h.baked_context_dim = gguf_i32(model.gguf, "magpietts.baked_context_dim", h.baked_context_dim);
    h.baked_speakers = gguf_i32(model.gguf, "magpietts.baked_speakers", h.baked_speakers);
    h.max_decoder_steps =
        gguf_i32(model.gguf, "magpietts.inference.max_decoder_steps", h.max_decoder_steps);
    h.top_k = gguf_i32(model.gguf, "magpietts.inference.topk", h.top_k);
    h.min_generated_frames =
        gguf_i32(model.gguf, "magpietts.inference.min_generated_frames", h.min_generated_frames);
    h.temperature = gguf_f32(model.gguf, "magpietts.inference.temperature", h.temperature);
    h.cfg_scale = gguf_f32(model.gguf, "magpietts.inference.cfg_scale", h.cfg_scale);
    const bool has_typed_attention_prior =
        gguf_find_key(model.gguf, "magpietts.inference.apply_attention_prior") >= 0;
    h.apply_attention_prior =
        gguf_bool(model.gguf, "magpietts.inference.apply_attention_prior", h.apply_attention_prior);
    h.attention_prior_epsilon = gguf_f32(
        model.gguf, "magpietts.inference.attention_prior_epsilon", h.attention_prior_epsilon);
    h.attention_prior_lookahead_window = gguf_i32(
        model.gguf, "magpietts.inference.attention_prior_lookahead_window",
        h.attention_prior_lookahead_window);
    h.start_prior_after_n_audio_steps = gguf_i32(
        model.gguf, "magpietts.inference.start_prior_after_n_audio_steps",
        h.start_prior_after_n_audio_steps);
    h.attention_prior_advance_threshold = gguf_i32(
        model.gguf, "magpietts.inference.attention_prior_advance_threshold",
        h.attention_prior_advance_threshold);
    h.attention_prior_decay_threshold = gguf_i32(
        model.gguf, "magpietts.inference.attention_prior_decay_threshold",
        h.attention_prior_decay_threshold);
    h.estimate_alignment_from_layers =
        gguf_i32_array(model.gguf, "magpietts.inference.estimate_alignment_from_layers");
    h.apply_prior_to_layers =
        gguf_i32_array(model.gguf, "magpietts.inference.apply_prior_to_layers");
    if (!has_typed_attention_prior) {
        const std::string config_json = gguf_string(model.gguf, "magpietts.config_json");
        if (!config_json.empty()) {
            h.apply_attention_prior =
                json_bool(config_json, "apply_attention_prior", h.apply_attention_prior);
            h.attention_prior_epsilon =
                json_f32(config_json, "attention_prior_epsilon", h.attention_prior_epsilon);
            h.attention_prior_lookahead_window = json_i32(
                config_json, "attention_prior_lookahead_window",
                h.attention_prior_lookahead_window);
            h.start_prior_after_n_audio_steps = json_i32(
                config_json, "start_prior_after_n_audio_steps", h.start_prior_after_n_audio_steps);
            h.attention_prior_advance_threshold = json_i32(
                config_json, "attention_prior_advance_threshold",
                h.attention_prior_advance_threshold);
            h.attention_prior_decay_threshold = json_i32(
                config_json, "attention_prior_decay_threshold", h.attention_prior_decay_threshold);
            h.estimate_alignment_from_layers =
                json_i32_array(config_json, "estimate_alignment_from_layers");
            h.apply_prior_to_layers = json_i32_array(config_json, "apply_prior_to_layers");
        }
    }

    if (h.attention_prior_advance_threshold < 0 || h.attention_prior_decay_threshold < 0) {
        fprintf(
            stderr, "attention prior thresholds must be non-negative: advance=%d decay=%d\n",
            h.attention_prior_advance_threshold, h.attention_prior_decay_threshold);
        return false;
    }

    if (model.tokenizer_profile.empty()) {
        model.tokenizer_profile = magpietts_infer_tokenizer_profile(h);
    }
    if (!magpietts_tokenizer_profile_matches(model.tokenizer_profile, h)) {
        fprintf(
            stderr,
            "unsupported or inconsistent Magpie tokenizer profile '%s': "
            "text_vocab_size=%d frame_stacking_factor=%d\n",
            model.tokenizer_profile.c_str(), h.text_vocab_size, h.frame_stacking_factor);
        return false;
    }
    if (stored_stacked_codebooks != expected_stacked_codebooks) {
        fprintf(
            stderr,
            "invalid stacked audio layout: codebooks=%d frame_stacking_factor=%d stored_slots=%d\n",
            h.audio_codebooks, h.frame_stacking_factor, stored_stacked_codebooks);
        return false;
    }

    model.cuda_unified_memory = force_cpu ? false : magpietts_configure_unified_memory(uma_mode);
    ggml_backend_load_all();
    if (!force_cpu) {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!model.backend) {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!model.backend) {
        fprintf(stderr, "failed to initialize ggml backend\n");
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(model.backend);
    if (verbose) {
        fprintf(
            stderr, "MagpieTTS backend: %s%s%s%s\n", ggml_backend_name(model.backend),
            dev ? " - " : "", dev ? ggml_backend_dev_description(dev) : "",
            force_cpu ? " (forced CPU)" : "");
        if (magpietts_backend_is_cuda(model.backend)) {
            fprintf(
                stderr, "MagpieTTS CUDA managed memory: %s (uma-mode=%s)\n",
                model.cuda_unified_memory ? "on" : "off", magpietts_uma_mode_name(uma_mode));
        }
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer) {
        fprintf(
            stderr, "failed to allocate MagpieTTS tensors on backend %s\n",
            ggml_backend_name(model.backend));
        return false;
    }

    FILE* f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s for tensor loading\n", fname.c_str());
        return false;
    }
    MagpiePinnedHostScratch read_buf;
    if (!read_buf.reserve(model, 16 * 1024 * 1024)) {
        fprintf(stderr, "failed to allocate MagpieTTS tensor read buffer\n");
        fclose(f);
        return false;
    }
    const size_t read_buf_size = read_buf.capacity();
    const int n_tensors = gguf_get_n_tensors(model.gguf);
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(model.gguf, i);
        ggml_tensor* tensor = ggml_get_tensor(model.ctx, name);
        if (!tensor) {
            continue;
        }

        const size_t tensor_offset =
            gguf_get_data_offset(model.gguf) + gguf_get_tensor_offset(model.gguf, i);
        if (fseek(f, (long)tensor_offset, SEEK_SET) != 0) {
            fprintf(stderr, "failed to seek tensor %s\n", name);
            fclose(f);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += read_buf_size) {
            const size_t ncopy = std::min(read_buf_size, nbytes - pos);
            if (fread(read_buf.data(), 1, ncopy, f) != ncopy) {
                fprintf(stderr, "failed to read tensor %s\n", name);
                fclose(f);
                return false;
            }
            ggml_backend_tensor_set(tensor, read_buf.data(), pos, ncopy);
        }
    }
    fclose(f);

    model.text_embedding = require_tensor(model, "text_embedding.weight");
    model.baked_context = require_tensor(model, "baked_context_embedding.weight");
    model.final_proj_w = require_tensor(model, "final_proj.weight");
    model.final_proj_b = require_tensor(model, "final_proj.bias");
    // v2602 projects decoder/audio embeddings into the local-transformer
    // dimension.  In v2607 those dimensions are both 768 and the checkpoint
    // intentionally omits the projection (identity input).
    model.lt_in_w = ggml_get_tensor(model.ctx, "local_transformer_in_projection.weight");
    model.lt_in_b = ggml_get_tensor(model.ctx, "local_transformer_in_projection.bias");
    if ((model.lt_in_w == nullptr) != (model.lt_in_b == nullptr)) {
        fprintf(stderr, "local-transformer input projection must include both weight and bias\n");
        return false;
    }
    if (model.lt_in_w == nullptr && h.n_embd != h.lt_hidden) {
        fprintf(
            stderr, "local-transformer input projection is missing for incompatible dimensions\n");
        return false;
    }

    model.audio_embeddings.resize(h.stacked_audio_codebooks());
    for (int i = 0; i < h.stacked_audio_codebooks(); ++i) {
        model.audio_embeddings[i] =
            require_tensor(model, "audio_embeddings." + std::to_string(i) + ".weight");
    }

    load_transformer(
        model, model.encoder, "encoder", h.n_enc_layer, h.n_embd, h.n_enc_head, h.enc_kernel, true,
        false, 0, 0);
    load_transformer(
        model, model.decoder, "decoder", h.n_dec_layer, h.n_embd, h.n_dec_head, h.dec_kernel, true,
        true, h.n_cross_head, h.n_cross_dhead);
    model.decoder.apply_attention_prior = h.apply_attention_prior;
    model.decoder.estimate_alignment_from_layers = h.estimate_alignment_from_layers;
    model.decoder.apply_prior_to_layers = h.apply_prior_to_layers;
    load_transformer(
        model, model.local, "local_transformer", h.lt_layers, h.lt_hidden, h.lt_heads, 1, true,
        false, 0, 0);
#if defined(GGML_USE_METAL)
    if (ggml_backend_is_metal(model.backend)) {
        model.encoder.portable_causal_mask = true;
        model.decoder.portable_causal_mask = true;
        model.local.portable_causal_mask = true;
    }
#endif

    model.lt_out_w.resize(h.stacked_audio_codebooks());
    model.lt_out_b.resize(h.stacked_audio_codebooks());
    for (int i = 0; i < h.stacked_audio_codebooks(); ++i) {
        model.lt_out_w[i] = require_tensor(
            model, "local_transformer_out_projections." + std::to_string(i) + ".weight");
        model.lt_out_b[i] = require_tensor(
            model, "local_transformer_out_projections." + std::to_string(i) + ".bias");
    }

    fprintf(
        stderr,
        "loaded MagpieTTS GGUF: text_vocab=%d audio_codebooks=%d stacked_slots=%d audio_vocab=%d "
        "speakers=%d "
        "attention_prior=%s epsilon=%.4g lookahead=%d start_step=%d advance_threshold=%d "
        "decay_threshold=%d estimate_layers=%s apply_layers=%s\n",
        h.text_vocab_size, h.audio_codebooks, h.stacked_audio_codebooks(), h.audio_vocab_size,
        h.baked_speakers, h.apply_attention_prior ? "on" : "off", h.attention_prior_epsilon,
        h.attention_prior_lookahead_window, h.start_prior_after_n_audio_steps,
        h.attention_prior_advance_threshold, h.attention_prior_decay_threshold,
        format_i32_list(h.estimate_alignment_from_layers).c_str(),
        format_i32_list(h.apply_prior_to_layers).c_str());
    return true;
}

ggml_tensor*
layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight) {
    return ggml_mul(ctx, ggml_norm(ctx, x, MAGPIETTS_LN_EPS), weight);
}

ggml_tensor*
linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

static ggml_tensor*
causal_shift(ggml_context* ctx, ggml_tensor* x, int shift) {
    if (shift == 0) {
        return x;
    }
    ggml_tensor* padded = ggml_pad(ctx, x, 0, shift, 0, 0);
    ggml_tensor* zeros = ggml_scale(ctx, padded, 0.0f);
    ggml_tensor* shifted = ggml_acc(
        ctx, zeros, x, zeros->nb[1], zeros->nb[2], zeros->nb[3], (size_t)shift * zeros->nb[1]);
    return ggml_view_2d(ctx, shifted, x->ne[0], x->ne[1], shifted->nb[1], 0);
}

ggml_tensor*
causal_conv1d(ggml_context* ctx, ggml_tensor* x, const std::vector<ggml_tensor*>& kernels) {
    ggml_tensor* y = nullptr;
    const int kernel = (int)kernels.size();
    for (int k = 0; k < kernel; ++k) {
        ggml_tensor* xs = causal_shift(ctx, x, kernel - 1 - k);
        ggml_tensor* part = ggml_mul_mat(ctx, kernels[k], xs);
        y = y ? ggml_add(ctx, y, part) : part;
    }
    return y;
}

static ggml_tensor*
causal_soft_max(
    ggml_context* ctx, const magpietts_transformer& tr, ggml_tensor* scores, int64_t n_past) {
    if (!tr.causal) {
        return ggml_soft_max(ctx, scores);
    }
    if (!tr.portable_causal_mask) {
        return ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, scores, n_past));
    }

    const int64_t n_keys = scores->ne[0];
    const int64_t n_queries = scores->ne[1];
    if (n_queries == 1) {
        return ggml_soft_max(ctx, scores);
    }

    // Metal does not implement GGML_OP_DIAG_MASK_INF. Build an equivalent
    // additive mask from portable operations without changing other backends'
    // graphs. A finite value avoids 0 * infinity in triangular kernels.
    ggml_tensor* storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_keys, n_keys);
    ggml_tensor* upper = ggml_tri(ctx, ggml_fill(ctx, storage, -1.0e9f), GGML_TRI_TYPE_UPPER);
    ggml_tensor* mask =
        ggml_view_2d(ctx, upper, n_keys, n_queries, upper->nb[1], (size_t)n_past * upper->nb[1]);
    return ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);
}

ggml_tensor*
self_attention(
    ggml_context* ctx, const magpietts_transformer& tr, const magpietts_layer& layer,
    ggml_tensor* x) {
    const int64_t n_embd = tr.n_embd;
    const int64_t n_head = tr.n_head;
    const int64_t d_head = n_embd / n_head;
    const int64_t n_tok = x->ne[1];

    ggml_tensor* qkv = linear(ctx, layer.self_qkv, x);
    ggml_tensor* qcur = ggml_view_2d(ctx, qkv, n_embd, n_tok, qkv->nb[1], 0);
    ggml_tensor* kcur =
        ggml_view_2d(ctx, qkv, n_embd, n_tok, qkv->nb[1], (size_t)ggml_element_size(qkv) * n_embd);
    ggml_tensor* vcur = ggml_view_2d(
        ctx, qkv, n_embd, n_tok, qkv->nb[1], (size_t)ggml_element_size(qkv) * n_embd * 2);

    ggml_tensor* q = ggml_permute(ctx, ggml_cont_3d(ctx, qcur, d_head, n_head, n_tok), 0, 2, 1, 3);
    ggml_tensor* k = ggml_permute(ctx, ggml_cont_3d(ctx, kcur, d_head, n_head, n_tok), 0, 2, 1, 3);
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt((float)d_head));
    ggml_tensor* kq_soft = causal_soft_max(ctx, tr, kq, 0);
    ggml_tensor* v_trans = ggml_cont_3d(
        ctx, ggml_permute(ctx, ggml_cont_3d(ctx, vcur, d_head, n_head, n_tok), 1, 2, 0, 3), n_tok,
        d_head, n_head);
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_trans, kq_soft);
    ggml_tensor* merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    ggml_tensor* out = ggml_cont_2d(ctx, merged, n_embd, n_tok);
    return linear(ctx, layer.self_o, out);
}

ggml_tensor*
self_attention_cached(
    ggml_context* ctx, ggml_cgraph* gf, const magpietts_transformer& tr,
    const magpietts_layer& layer, DecoderKvCache& kv, int layer_index, int n_past, ggml_tensor* x) {
    const int64_t n_embd = tr.n_embd;
    const int64_t n_head = tr.n_head;
    const int64_t d_head = n_embd / n_head;
    const int64_t n_tok = x->ne[1];
    const int64_t n_total = n_past + n_tok;

    ggml_tensor* qkv = linear(ctx, layer.self_qkv, x);
    ggml_tensor* qcur = ggml_view_2d(ctx, qkv, n_embd, n_tok, qkv->nb[1], 0);
    ggml_tensor* kcur =
        ggml_view_2d(ctx, qkv, n_embd, n_tok, qkv->nb[1], (size_t)ggml_element_size(qkv) * n_embd);
    ggml_tensor* vcur = ggml_view_2d(
        ctx, qkv, n_embd, n_tok, qkv->nb[1], (size_t)ggml_element_size(qkv) * n_embd * 2);

    const size_t layer_offset =
        (size_t)layer_index * kv.n_ctx * n_embd * ggml_element_size(kv.memory_k);
    const size_t write_offset =
        layer_offset + (size_t)n_past * n_embd * ggml_element_size(kv.memory_k);

    ggml_tensor* k_dst = ggml_view_1d(ctx, kv.memory_k, n_tok * n_embd, write_offset);
    ggml_tensor* v_dst = ggml_view_1d(ctx, kv.memory_v, n_tok * n_embd, write_offset);
    ggml_tensor* k_copy = ggml_cpy(ctx, kcur, k_dst);
    ggml_set_name(k_copy, "magpietts_decoder_kv_copy_k");
    ggml_build_forward_expand(gf, k_copy);
    ggml_tensor* v_copy = ggml_cpy(ctx, vcur, v_dst);
    ggml_set_name(v_copy, "magpietts_decoder_kv_copy_v");
    ggml_build_forward_expand(gf, v_copy);

    ggml_tensor* q = ggml_permute(ctx, ggml_cont_3d(ctx, qcur, d_head, n_head, n_tok), 0, 2, 1, 3);
    ggml_tensor* k = ggml_permute(
        ctx,
        ggml_reshape_3d(
            ctx, ggml_view_1d(ctx, kv.memory_k, n_total * n_embd, layer_offset), d_head, n_head,
            n_total),
        0, 2, 1, 3);

    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt((float)d_head));
    ggml_tensor* kq_soft = causal_soft_max(ctx, tr, kq, n_past);
    ggml_tensor* v_trans = ggml_cont_3d(
        ctx,
        ggml_permute(
            ctx,
            ggml_reshape_3d(
                ctx, ggml_view_1d(ctx, kv.memory_v, n_total * n_embd, layer_offset), d_head, n_head,
                n_total),
            1, 2, 0, 3),
        n_total, d_head, n_head);
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_trans, kq_soft);
    ggml_tensor* merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    ggml_tensor* out = ggml_cont_2d(ctx, merged, n_embd, n_tok);
    return linear(ctx, layer.self_o, out);
}

ggml_tensor*
cross_attention(
    ggml_context* ctx, const magpietts_transformer& tr, const magpietts_layer& layer,
    ggml_tensor* x, ggml_tensor* memory, ggml_tensor* attn_prior, ggml_tensor** last_attn) {
    const int64_t d_head = tr.n_cross_dhead;
    const int64_t n_head = tr.n_cross_head;
    const int64_t n_q = x->ne[1];
    const int64_t n_kv = memory->ne[1];

    ggml_tensor* q = linear(ctx, layer.cross_q, x);
    ggml_tensor* mem_norm = layer_norm(ctx, memory, layer.norm_xattn_memory);
    ggml_tensor* kv = linear(ctx, layer.cross_kv, mem_norm);
    ggml_tensor* kcur = ggml_view_2d(ctx, kv, d_head * n_head, n_kv, kv->nb[1], 0);
    ggml_tensor* vcur = ggml_view_2d(
        ctx, kv, d_head * n_head, n_kv, kv->nb[1], (size_t)ggml_element_size(kv) * d_head * n_head);

    ggml_tensor* qh = ggml_permute(ctx, ggml_cont_3d(ctx, q, d_head, n_head, n_q), 0, 2, 1, 3);
    ggml_tensor* kh = ggml_permute(ctx, ggml_cont_3d(ctx, kcur, d_head, n_head, n_kv), 0, 2, 1, 3);
    ggml_tensor* kq = ggml_mul_mat(ctx, kh, qh);
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt((float)d_head));
    ggml_tensor* kq_soft = ggml_soft_max(ctx, kq);
    if (attn_prior) {
        ggml_tensor* prior = ggml_repeat(ctx, attn_prior, kq_soft);
        kq_soft = ggml_mul(ctx, kq_soft, prior);
        ggml_tensor* normalizer = ggml_repeat(ctx, ggml_sum_rows(ctx, kq_soft), kq_soft);
        kq_soft = ggml_div(ctx, kq_soft, normalizer);
    }
    if (last_attn) {
        const size_t offset = (size_t)(n_q - 1) * kq_soft->nb[1];
        ggml_tensor* last = ggml_view_2d(ctx, kq_soft, n_kv, n_head, kq_soft->nb[2], offset);
        *last_attn = ggml_cont_2d(ctx, last, n_kv, n_head);
    }
    ggml_tensor* v_trans = ggml_cont_3d(
        ctx, ggml_permute(ctx, ggml_cont_3d(ctx, vcur, d_head, n_head, n_kv), 1, 2, 0, 3), n_kv,
        d_head, n_head);
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_trans, kq_soft);
    ggml_tensor* merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    ggml_tensor* out = ggml_cont_2d(ctx, merged, d_head * n_head, n_q);
    return linear(ctx, layer.cross_o, out);
}

ggml_tensor*
cross_attention_cached(
    ggml_context* ctx, const magpietts_transformer& tr, const magpietts_layer& layer,
    const DecoderCrossKvCache& cross_kv, int layer_index, ggml_tensor* x, ggml_tensor* attn_prior,
    ggml_tensor** last_attn, bool prior_is_log) {
    const int64_t d_head = tr.n_cross_dhead;
    const int64_t n_head = tr.n_cross_head;
    const int64_t cross_dim = d_head * n_head;
    const int64_t n_q = x->ne[1];
    const int64_t n_kv = cross_kv.text_len;

    ggml_tensor* q = linear(ctx, layer.cross_q, x);
    const size_t layer_offset =
        (size_t)layer_index * n_kv * cross_dim * ggml_element_size(cross_kv.memory_k);

    ggml_tensor* qh = ggml_permute(ctx, ggml_cont_3d(ctx, q, d_head, n_head, n_q), 0, 2, 1, 3);
    ggml_tensor* kh = ggml_permute(
        ctx,
        ggml_reshape_3d(
            ctx, ggml_view_1d(ctx, cross_kv.memory_k, n_kv * cross_dim, layer_offset), d_head,
            n_head, n_kv),
        0, 2, 1, 3);
    ggml_tensor* kq = ggml_mul_mat(ctx, kh, qh);
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt((float)d_head));
    ggml_tensor* kq_soft = nullptr;
    if (attn_prior && prior_is_log) {
        kq_soft = ggml_soft_max(ctx, ggml_add(ctx, kq, ggml_repeat(ctx, attn_prior, kq)));
    } else {
        kq_soft = ggml_soft_max(ctx, kq);
    }
    if (attn_prior && !prior_is_log) {
        ggml_tensor* prior = ggml_repeat(ctx, attn_prior, kq_soft);
        kq_soft = ggml_mul(ctx, kq_soft, prior);
        ggml_tensor* normalizer = ggml_repeat(ctx, ggml_sum_rows(ctx, kq_soft), kq_soft);
        kq_soft = ggml_div(ctx, kq_soft, normalizer);
    }
    if (last_attn) {
        const size_t offset = (size_t)(n_q - 1) * kq_soft->nb[1];
        ggml_tensor* last = ggml_view_2d(ctx, kq_soft, n_kv, n_head, kq_soft->nb[2], offset);
        *last_attn = ggml_cont_2d(ctx, last, n_kv, n_head);
    }
    ggml_tensor* v_trans = ggml_cont_3d(
        ctx,
        ggml_permute(
            ctx,
            ggml_reshape_3d(
                ctx, ggml_view_1d(ctx, cross_kv.memory_v, n_kv * cross_dim, layer_offset), d_head,
                n_head, n_kv),
            1, 2, 0, 3),
        n_kv, d_head, n_head);
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_trans, kq_soft);
    ggml_tensor* merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    ggml_tensor* out = ggml_cont_2d(ctx, merged, cross_dim, n_q);
    return linear(ctx, layer.cross_o, out);
}

static bool
layer_selected(const std::vector<int32_t>& layers, int layer_index) {
    return layers.empty() ||
           std::find(layers.begin(), layers.end(), (int32_t)layer_index) != layers.end();
}

ggml_tensor*
transformer_forward(
    ggml_context* ctx, const magpietts_transformer& tr, ggml_tensor* x, ggml_tensor* pos,
    ggml_tensor* cond, ggml_tensor* attn_prior, std::vector<ggml_tensor*>* alignment_outputs) {
    x = ggml_add(ctx, x, ggml_get_rows(ctx, tr.pos_emb, pos));

    for (int il = 0; il < (int)tr.layers.size(); ++il) {
        const magpietts_layer& layer = tr.layers[il];
        ggml_tensor* residual = x;
        ggml_tensor* cur = layer_norm(ctx, x, layer.norm_self);
        cur = self_attention(ctx, tr, layer, cur);
        x = ggml_add(ctx, residual, cur);

        if (tr.has_cross && cond != nullptr) {
            residual = x;
            cur = layer_norm(ctx, x, layer.norm_xattn_query);
            ggml_tensor* last_attn = nullptr;
            const bool apply_prior = tr.apply_attention_prior && attn_prior &&
                                     layer_selected(tr.apply_prior_to_layers, il);
            const bool collect_alignment =
                alignment_outputs && layer_selected(tr.estimate_alignment_from_layers, il);
            cur = cross_attention(
                ctx, tr, layer, cur, cond, apply_prior ? attn_prior : nullptr,
                collect_alignment ? &last_attn : nullptr);
            if (last_attn) {
                const std::string name = "magpietts_decoder_cross_attn_last_" + std::to_string(il);
                ggml_set_name(last_attn, name.c_str());
                ggml_set_output(last_attn);
                alignment_outputs->push_back(last_attn);
            }
            x = ggml_add(ctx, residual, cur);
        }

        residual = x;
        cur = layer_norm(ctx, x, layer.norm_ff);
        cur = causal_conv1d(ctx, cur, layer.ff_proj);
        cur = ggml_gelu(ctx, cur);
        cur = causal_conv1d(ctx, cur, layer.ff_out);
        x = ggml_add(ctx, residual, cur);
    }

    return tr.norm_out ? layer_norm(ctx, x, tr.norm_out) : x;
}

ggml_tensor*
transformer_forward_cached(
    ggml_context* ctx, ggml_cgraph* gf, const magpietts_transformer& tr, ggml_tensor* x,
    ggml_tensor* pos, ggml_tensor* cond, DecoderKvCache& kv, DecoderCrossKvCache* cross_kv,
    int n_past, ggml_tensor* attn_prior, std::vector<ggml_tensor*>* alignment_outputs) {
    x = ggml_add(ctx, x, ggml_get_rows(ctx, tr.pos_emb, pos));

    for (int il = 0; il < (int)tr.layers.size(); ++il) {
        const magpietts_layer& layer = tr.layers[il];
        ggml_tensor* residual = x;
        ggml_tensor* cur = layer_norm(ctx, x, layer.norm_self);
        cur = self_attention_cached(ctx, gf, tr, layer, kv, il, n_past, cur);
        x = ggml_add(ctx, residual, cur);

        const bool has_cached_cross = cross_kv && cross_kv->valid && cross_kv->memory_k &&
                                      cross_kv->memory_v && cross_kv->text_len > 0;
        if (tr.has_cross && (cond != nullptr || has_cached_cross)) {
            residual = x;
            cur = layer_norm(ctx, x, layer.norm_xattn_query);
            ggml_tensor* last_attn = nullptr;
            const bool apply_prior = tr.apply_attention_prior && attn_prior &&
                                     layer_selected(tr.apply_prior_to_layers, il);
            const bool collect_alignment =
                alignment_outputs && layer_selected(tr.estimate_alignment_from_layers, il);
            if (has_cached_cross) {
                cur = cross_attention_cached(
                    ctx, tr, layer, *cross_kv, il, cur, apply_prior ? attn_prior : nullptr,
                    collect_alignment ? &last_attn : nullptr);
            } else {
                cur = cross_attention(
                    ctx, tr, layer, cur, cond, apply_prior ? attn_prior : nullptr,
                    collect_alignment ? &last_attn : nullptr);
            }
            if (last_attn) {
                const std::string name =
                    "magpietts_decoder_cross_attn_last_cached_" + std::to_string(il);
                ggml_set_name(last_attn, name.c_str());
                ggml_set_output(last_attn);
                alignment_outputs->push_back(last_attn);
            }
            x = ggml_add(ctx, residual, cur);
        }

        residual = x;
        cur = layer_norm(ctx, x, layer.norm_ff);
        cur = causal_conv1d(ctx, cur, layer.ff_proj);
        cur = ggml_gelu(ctx, cur);
        cur = causal_conv1d(ctx, cur, layer.ff_out);
        x = ggml_add(ctx, residual, cur);
    }

    return tr.norm_out ? layer_norm(ctx, x, tr.norm_out) : x;
}

ggml_context*
new_graph_context() {
    const size_t buf_size = ggml_tensor_overhead() * MAGPIETTS_MAX_NODES +
                            ggml_graph_overhead_custom(MAGPIETTS_MAX_NODES, false);
    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    return ggml_init(params);
}

static bool
is_default_graph_node_name(const ggml_tensor* tensor) {
    if (!tensor) {
        return false;
    }
    const char* name = ggml_get_name(tensor);
    return !name || name[0] == '\0' || std::strncmp(name, "node_", 5) == 0;
}

void
tag_graph_first_node(ggml_cgraph* gf) {
    const int n_nodes = gf ? ggml_graph_n_nodes(gf) : 0;
    ggml_tensor* first = n_nodes > 0 ? ggml_graph_node(gf, 0) : nullptr;
    if (!first || !is_default_graph_node_name(first)) {
        return;
    }
    const char* label = ggml_get_name(ggml_graph_node(gf, n_nodes - 1));
    if (label && label[0]) {
        ggml_set_name(first, label);
    }
}

bool
compute_graph(
    const magpietts_model& model, ggml_context* ctx, ggml_cgraph* gf,
    const std::vector<std::pair<std::string, std::vector<int32_t>>>& i32_inputs,
    const std::vector<std::pair<std::string, std::vector<float>>>& f32_inputs, int threads,
    ggml_gallocr_t* keep_allocr) {
    const ggml_nvtx::range nvtx_range("magpietts_compute_graph");
    tag_graph_first_node(gf);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) {
        fprintf(stderr, "failed to create graph allocator\n");
        return false;
    }

    {
        const ggml_nvtx::range nvtx_alloc("magpietts_graph_alloc");
        ggml_gallocr_alloc_graph(allocr, gf);
    }

    {
        const ggml_nvtx::range nvtx_inputs("magpietts_graph_set_inputs");
        MagpiePinnedHostScratch input_staging;
        for (const auto& it : i32_inputs) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, it.first.c_str());
            if (!t) {
                fprintf(stderr, "missing graph input: %s\n", it.first.c_str());
                ggml_gallocr_free(allocr);
                return false;
            }
            magpietts_backend_tensor_set_staged(
                model, input_staging, t, it.second.data(), 0, it.second.size() * sizeof(int32_t));
        }
        for (const auto& it : f32_inputs) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, it.first.c_str());
            if (!t) {
                fprintf(stderr, "missing graph input: %s\n", it.first.c_str());
                ggml_gallocr_free(allocr);
                return false;
            }
            magpietts_backend_tensor_set_staged(
                model, input_staging, t, it.second.data(), 0, it.second.size() * sizeof(float));
        }
    }

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, threads);
    }

    ggml_status status = GGML_STATUS_FAILED;
    {
        const ggml_nvtx::range nvtx_compute("magpietts_graph_compute");
        status = ggml_backend_graph_compute(model.backend, gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_gallocr_free(allocr);
        return false;
    }

    if (keep_allocr) {
        *keep_allocr = allocr;
    } else {
        ggml_gallocr_free(allocr);
    }
    (void)ctx;
    return true;
}

std::vector<int32_t>
positions(int n) {
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; ++i) {
        pos[i] = i;
    }
    return pos;
}

std::vector<int32_t>
positions_range(int start, int n) {
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; ++i) {
        pos[i] = start + i;
    }
    return pos;
}

std::vector<int32_t>
parse_token_list(const std::string& text) {
    std::vector<int32_t> out;
    std::string cur;
    for (char ch : text) {
        if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t') {
            if (!cur.empty()) {
                out.push_back(std::stoi(cur));
                cur.clear();
            }
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) {
        out.push_back(std::stoi(cur));
    }
    return out;
}

std::string
read_file(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << fin.rdbuf();
    return ss.str();
}

static int
effective_attention_prior_advance_threshold(
    const magpietts_hparams& h, int attended_relative_index, int text_len) {
    (void)attended_relative_index;
    (void)text_len;
    return std::max(0, h.attention_prior_advance_threshold);
}

void
MagpieAttentionPriorState::reset() {
    text_len_ = 0;
    last_attended_ = 1;
    attended_counts_.clear();
    prior_.clear();
}

void
MagpieAttentionPriorState::ensureTextLen(int text_len) {
    if (text_len_ == text_len) {
        return;
    }
    text_len_ = text_len;
    last_attended_ = std::min(1, std::max(0, text_len - 1));
    attended_counts_.assign((size_t)std::max(text_len, 0), 0);
    prior_.clear();
}

bool
MagpieAttentionPriorState::enabled(const magpietts_hparams& h, int text_len) const {
    return h.apply_attention_prior && text_len > 0 && h.n_cross_head > 0 && h.n_cross_dhead > 0;
}

bool
MagpieAttentionPriorState::shouldCollect(const magpietts_hparams& h, int step, int text_len) const {
    return enabled(h, text_len) && step >= h.start_prior_after_n_audio_steps;
}

const std::vector<float>*
MagpieAttentionPriorState::priorForStep(const magpietts_hparams& h, int text_len) const {
    if (!enabled(h, text_len) || prior_.empty() || text_len_ != text_len) {
        return nullptr;
    }
    return &prior_;
}

void
MagpieAttentionPriorState::update(
    const magpietts_hparams& h, int step, int text_len,
    const std::vector<float>& alignment_scores) {
    if (!shouldCollect(h, step, text_len) || (int)alignment_scores.size() != text_len) {
        return;
    }
    ensureTextLen(text_len);
    if (text_len <= 0) {
        return;
    }

    const int previous_last = std::max(0, std::min(last_attended_, text_len - 1));
    const int max_next = std::min(previous_last + 1, text_len - 1);
    int last = previous_last;
    const int advance_threshold = effective_attention_prior_advance_threshold(h, last, text_len);
    if (last < (int)attended_counts_.size() &&
        attended_counts_[(size_t)last] >= advance_threshold) {
        last = std::min(last + 1, text_len - 1);
    }

    int attended = text_len - 1;
    const int window_end =
        std::min(last + std::max(0, h.attention_prior_lookahead_window), text_len - 3);
    if (window_end > last) {
        attended = last;
        float best = alignment_scores[(size_t)last];
        for (int i = last + 1; i < window_end; ++i) {
            if (alignment_scores[(size_t)i] > best) {
                best = alignment_scores[(size_t)i];
                attended = i;
            }
        }
    }

    last_attended_ = std::max(0, std::min({attended, max_next, text_len - 1}));
    if (last_attended_ < (int)attended_counts_.size()) {
        ++attended_counts_[(size_t)last_attended_];
    }

    prior_.assign((size_t)text_len, h.attention_prior_epsilon);
    if (text_len <= 5) {
        std::fill(prior_.begin(), prior_.end(), 1.0f);
    } else {
        prior_[(size_t)std::max(1, last_attended_ - 1)] = 1.0f;
        prior_[(size_t)last_attended_] = 1.0f;
        for (int i = 1; i <= h.attention_prior_lookahead_window; ++i) {
            prior_[(size_t)std::min(last_attended_ + i, text_len - 1)] = 1.0f;
        }
    }

    for (int i = 0; i < (int)attended_counts_.size(); ++i) {
        if (attended_counts_[(size_t)i] >= h.attention_prior_decay_threshold) {
            const int preserve_from = std::max(0, std::min(last_attended_, text_len - 1));
            const int fill_end = std::min(std::min(i + 1, text_len), preserve_from);
            std::fill(prior_.begin(), prior_.begin() + fill_end, h.attention_prior_epsilon);
        }
    }
}

void
MagpieLongformAttentionPriorState::reset() {
    initialized_ = false;
    first_chunk_ = true;
    left_offset_ = 0;
    text_len_ = 0;
    current_chunk_len_ = 0;
    last_attended_absolute_ = 1;
    attended_counts_.clear();
    prior_.clear();
}

bool
MagpieLongformAttentionPriorState::enabled(const magpietts_hparams& h, int text_len) const {
    return h.apply_attention_prior && text_len > 0 && h.n_cross_head > 0 && h.n_cross_dhead > 0;
}

bool
MagpieLongformAttentionPriorState::shouldCollect(
    const magpietts_hparams& h, int step, int text_len) const {
    return enabled(h, text_len) && initialized_ && step >= h.start_prior_after_n_audio_steps;
}

const std::vector<float>*
MagpieLongformAttentionPriorState::priorForStep(const magpietts_hparams& h, int text_len) const {
    if (!enabled(h, text_len) || prior_.empty() || text_len_ != text_len) {
        return nullptr;
    }
    return &prior_;
}

int
MagpieLongformAttentionPriorState::lastAttendedRelative() const {
    if (text_len_ <= 0) {
        return 0;
    }
    return std::max(0, std::min(last_attended_absolute_ - left_offset_, text_len_ - 1));
}

void
MagpieLongformAttentionPriorState::ensureAbsoluteCapacity(int absolute_len) {
    if (absolute_len <= 0) {
        return;
    }
    if ((int)attended_counts_.size() < absolute_len) {
        attended_counts_.resize((size_t)absolute_len, 0);
    }
}

void
MagpieLongformAttentionPriorState::beginChunk(
    const magpietts_hparams& h, int left_offset, int text_len, int current_chunk_len,
    bool first_chunk) {
    initialized_ = true;
    first_chunk_ = first_chunk;
    left_offset_ = std::max(0, left_offset);
    text_len_ = std::max(0, text_len);
    current_chunk_len_ = std::max(0, std::min(current_chunk_len, text_len_));
    ensureAbsoluteCapacity(left_offset_ + text_len_);

    if (first_chunk_ && text_len_ > 0) {
        last_attended_absolute_ = left_offset_ + std::min(1, text_len_ - 1);
        prior_.clear();
        return;
    }

    if (!enabled(h, text_len_)) {
        prior_.clear();
        return;
    }
    buildInitialChunkPrior(h);
}

void
MagpieLongformAttentionPriorState::buildInitialChunkPrior(const magpietts_hparams& h) {
    if (text_len_ <= 0) {
        prior_.clear();
        return;
    }

    const float eps = std::max(0.0f, h.attention_prior_epsilon);
    const float eps_sq = eps * eps;
    prior_.assign((size_t)text_len_, eps_sq);

    if (text_len_ <= 35) {
        std::fill(prior_.begin(), prior_.end(), 1.0f);
        return;
    }

    const int current_start = std::max(0, text_len_ - current_chunk_len_);
    if (current_start > 0) {
        prior_[(size_t)(current_start - 1)] = 0.2f;
    }
    static const float init_weights[] = {0.5f, 1.0f, 0.8f, 0.2f, 0.2f};
    for (int i = 0; i < (int)(sizeof(init_weights) / sizeof(init_weights[0])); ++i) {
        const int pos = current_start + i;
        if (pos >= 0 && pos < text_len_) {
            prior_[(size_t)pos] = init_weights[i];
        }
    }
}

void
MagpieLongformAttentionPriorState::buildPrior(const magpietts_hparams& h) {
    if (text_len_ <= 0) {
        prior_.clear();
        return;
    }

    const float eps = std::max(0.0f, h.attention_prior_epsilon);
    const float eps_sq = first_chunk_ ? eps : eps * eps;
    prior_.assign((size_t)text_len_, eps_sq);

    const int rel = lastAttendedRelative();
    if (first_chunk_) {
        if (text_len_ <= 5) {
            std::fill(prior_.begin(), prior_.end(), 1.0f);
        } else {
            prior_[(size_t)std::max(1, rel - 1)] = 1.0f;
            prior_[(size_t)rel] = 1.0f;
            for (int i = 1; i <= h.attention_prior_lookahead_window; ++i) {
                prior_[(size_t)std::min(rel + i, text_len_ - 1)] = 1.0f;
            }
        }
    } else if (text_len_ <= 35) {
        std::fill(prior_.begin(), prior_.end(), 1.0f);
    } else {
        if (rel > 0) {
            prior_[(size_t)(rel - 1)] = 0.2f;
        }
        prior_[(size_t)rel] = 1.0f;
        static const float lookahead_weights[] = {0.6f, 0.4f, 0.2f, 0.2f};
        for (int i = 0; i < (int)(sizeof(lookahead_weights) / sizeof(lookahead_weights[0])); ++i) {
            const int pos = rel + i + 1;
            if (pos >= text_len_) {
                break;
            }
            prior_[(size_t)pos] = lookahead_weights[i];
        }
    }

    for (int absolute = left_offset_; absolute < left_offset_ + text_len_; ++absolute) {
        if (absolute >= 0 && absolute < (int)attended_counts_.size() &&
            attended_counts_[(size_t)absolute] >= h.attention_prior_decay_threshold) {
            const int rel_pos = absolute - left_offset_;
            const int preserve_from = std::max(0, std::min(rel, text_len_ - 1));
            const int fill_end = std::min(std::min(rel_pos + 1, text_len_), preserve_from);
            std::fill(prior_.begin(), prior_.begin() + fill_end, eps_sq);
        }
    }
}

void
MagpieLongformAttentionPriorState::update(
    const magpietts_hparams& h, int step, int text_len,
    const std::vector<float>& alignment_scores) {
    if (!shouldCollect(h, step, text_len) || (int)alignment_scores.size() != text_len ||
        text_len != text_len_) {
        return;
    }
    if (text_len_ <= 0) {
        return;
    }

    const int previous_abs =
        std::max(left_offset_, std::min(last_attended_absolute_, left_offset_ + text_len_ - 1));
    const int previous_rel = std::max(0, std::min(previous_abs - left_offset_, text_len_ - 1));
    const int advance_threshold =
        effective_attention_prior_advance_threshold(h, previous_rel, text_len_);
    int search_start_abs = previous_abs;
    if (previous_abs >= 0 && previous_abs < (int)attended_counts_.size() &&
        attended_counts_[(size_t)previous_abs] >= advance_threshold) {
        search_start_abs = std::min(previous_abs + 1, left_offset_ + text_len_ - 1);
    }
    const int last_rel = std::max(0, std::min(search_start_abs - left_offset_, text_len_ - 1));

    int attended_rel = last_rel;
    const int search_end =
        std::min(last_rel + std::max(0, h.attention_prior_lookahead_window), text_len_ - 3);
    if (search_end > last_rel) {
        attended_rel = last_rel;
        float best = alignment_scores[(size_t)last_rel];
        for (int i = last_rel + 1; i < search_end; ++i) {
            if (alignment_scores[(size_t)i] > best) {
                best = alignment_scores[(size_t)i];
                attended_rel = i;
            }
        }
    }

    const int attended_abs = left_offset_ + attended_rel;
    last_attended_absolute_ =
        std::max(left_offset_, std::min(attended_abs, left_offset_ + text_len_ - 1));
    ensureAbsoluteCapacity(last_attended_absolute_ + 1);
    if (last_attended_absolute_ >= 0 && last_attended_absolute_ < (int)attended_counts_.size()) {
        ++attended_counts_[(size_t)last_attended_absolute_];
    }

    buildPrior(h);
}

void
magpietts_run_metrics::begin() {
    *this = {};
    inter_frame_min_ms = std::numeric_limits<double>::max();
    start_us = ggml_time_us();
}

double
magpietts_run_metrics::record_frame(int64_t now_us, bool& first_frame, int emitted_frames) {
    first_frame = first_frame_us == 0;
    double inter_ms = 0.0;
    if (first_frame) {
        first_frame_us = now_us;
        ttff_ms = start_us > 0 ? (double)(now_us - start_us) / 1000.0 : 0.0;
    } else {
        inter_ms = (double)(now_us - last_frame_us) / 1000.0;
        inter_frame_sum_ms += inter_ms;
        inter_frame_min_ms = std::min(inter_frame_min_ms, inter_ms);
        inter_frame_max_ms = std::max(inter_frame_max_ms, inter_ms);
    }
    last_frame_us = now_us;
    frames += emitted_frames;
    ++timing_events;
    return inter_ms;
}

void
magpietts_run_metrics::finish(size_t frames_generated, double codec_fps) {
    e2e_elapsed_s = start_us > 0 ? (double)(ggml_time_us() - start_us) / 1000000.0 : 0.0;
    audio_s = codec_fps > 0.0 ? (double)frames_generated / codec_fps : 0.0;
    e2e_rtf = audio_s > 0.0 ? e2e_elapsed_s / audio_s : 0.0;
    e2e_rtfx = e2e_elapsed_s > 0.0 ? audio_s / e2e_elapsed_s : 0.0;
}

double
magpietts_run_metrics::inter_frame_avg_ms() const {
    return timing_events > 1 ? inter_frame_sum_ms / (double)(timing_events - 1) : 0.0;
}

double
magpietts_run_metrics::inter_frame_min_value_ms() const {
    return timing_events > 1 ? inter_frame_min_ms : 0.0;
}

bool
magpietts_resolve_lt_backend(
    const magpietts_model& model, magpietts_backend_preference requested, bool& use_cuda_lt) {
    use_cuda_lt = false;
    if (requested == MAGPIETTS_BACKEND_CPU) {
        return true;
    }

    const bool cuda_backend = magpietts_backend_is_cuda(model.backend);
    if (requested == MAGPIETTS_BACKEND_CUDA) {
        if (!cuda_backend) {
            fprintf(
                stderr, "--lt-backend cuda requires a CUDA ggml backend; current backend is %s\n",
                ggml_backend_name(model.backend));
            return false;
        }
        use_cuda_lt = true;
        return true;
    }

    use_cuda_lt = cuda_backend;
    return true;
}

bool
magpietts_resolve_sampling_backend(
    const magpietts_model& model, magpietts_backend_preference requested, bool& use_cuda_sampling) {
    use_cuda_sampling = false;
    if (requested == MAGPIETTS_BACKEND_CPU) {
        return true;
    }

#if defined(MAGPIETTS_CUDA_SAMPLING)
    const bool cuda_backend = magpietts_backend_is_cuda(model.backend);
    if (requested == MAGPIETTS_BACKEND_AUTO) {
        use_cuda_sampling = cuda_backend;
        return true;
    }
    if (!cuda_backend) {
        fprintf(
            stderr, "--sampling-backend cuda requires a CUDA ggml backend; current backend is %s\n",
            ggml_backend_name(model.backend));
        return false;
    }
    use_cuda_sampling = true;
    return true;
#else
    (void)model;
    if (requested == MAGPIETTS_BACKEND_AUTO) {
        return true;
    }
    fprintf(
        stderr,
        "--sampling-backend cuda requires building MagpieTTS with GGML_CUDA=ON and CUDAToolkit\n");
    return false;
#endif
}

bool
MagpieCodeGenerator::generate(
    const magpietts_params& params, const std::vector<int32_t>& tokens,
    std::vector<std::vector<int32_t>>* generated_frames_out, magpietts_run_metrics& metrics,
    const char* run_label, const magpietts_model* local_transformer_cpu_model) {
    magpietts_model& model = model_;
    auto& h = model.hparams;
    const char* label = run_label ? run_label : "run";
    std::mt19937 rng((uint32_t)params.seed);
    bool use_cuda_lt = false;
    if (params.use_local_transformer &&
        !magpietts_resolve_lt_backend(model, params.lt_backend, use_cuda_lt)) {
        return false;
    }
    bool use_cuda_sampling = false;
    if (!magpietts_resolve_sampling_backend(model, params.sampling_backend, use_cuda_sampling)) {
        return false;
    }
    if (params.sampling_backend == MAGPIETTS_BACKEND_AUTO && params.use_local_transformer &&
        !use_cuda_lt) {
        use_cuda_sampling = false;
    }
    if (params.use_local_transformer && !use_cuda_lt && use_cuda_sampling) {
        fprintf(
            stderr,
            "--lt-backend cpu with --sampling-backend cuda is not supported when the local "
            "transformer is enabled\n");
        return false;
    }

    magpietts_model local_transformer_owner;
    const bool use_fp32_local = params.use_local_transformer && params.lt_fp32;
    const bool force_local_transformer_cpu =
        params.use_local_transformer && !use_cuda_lt && magpietts_backend_is_cuda(model.backend);
    const magpietts_model* local_transformer_override = local_transformer_cpu_model;
    if (use_fp32_local) {
        if (!magpietts_model_init_local_transformer_fp32(
                model, local_transformer_owner, use_cuda_lt)) {
            return false;
        }
        local_transformer_override = &local_transformer_owner;
    } else if (force_local_transformer_cpu && !local_transformer_override) {
        if (!magpietts_model_init_local_transformer_cpu(model, local_transformer_owner)) {
            return false;
        }
        local_transformer_override = &local_transformer_owner;
    }
    const magpietts_model& local_transformer_model =
        (use_fp32_local || force_local_transformer_cpu) ? *local_transformer_override : model;

    const char* local_transformer_effective = "off";
    if (params.use_local_transformer) {
        local_transformer_effective = use_cuda_lt ? "cuda" : "cpu";
    }

    fprintf(
        stderr,
        "%s encoding %zu text tokens with speaker=%d seed=%d kv-cache=%s "
        "lt-backend=%s effective=%s lt-precision=%s sampling-backend=%s effective=%s "
        "managed=%s codec-fps=%.2f\n",
        label, tokens.size(), params.speaker, params.seed, params.use_kv_cache ? "yes" : "no",
        magpietts_backend_preference_name(params.lt_backend), local_transformer_effective,
        params.lt_fp32 ? "fp32" : "native",
        magpietts_backend_preference_name(params.sampling_backend),
        use_cuda_sampling ? "cuda" : "cpu", model.cuda_unified_memory ? "on" : "off",
        params.codec_fps);

    metrics.begin();

    std::vector<float> text_cond;
    magpietts_backend_tensor text_cond_device;
    magpietts_backend_tensor cond_hidden_device;
    magpietts_backend_tensor uncond_hidden_device;

    std::vector<std::vector<int32_t>> audio_codes(h.audio_codebooks);
    for (int c = 0; c < h.audio_codebooks; ++c) {
        audio_codes[c].assign((size_t)h.frame_stacking_factor, h.audio_bos_id);
    }

    std::vector<std::vector<int32_t>> generated_frames;
    generated_frames.reserve(h.max_decoder_steps);
    DecoderKvCache cond_kv;
    DecoderKvCache uncond_kv;
    DecoderCrossKvCache cond_cross_kv;
    MagpieAttentionPriorState attention_prior;
    MagpieDecoder decoder(model);
    LocalCodebookSampler local_sampler(local_transformer_model, params.threads);
#if defined(MAGPIETTS_CUDA_SAMPLING)
    struct CudaSamplerDeleter {
        void operator()(magpietts_cuda_sampler* sampler) const {
            magpietts_cuda_sampler_free(sampler);
        }
    };
    std::unique_ptr<magpietts_cuda_sampler, CudaSamplerDeleter> cuda_sampler;
#endif

    MagpieEncoder encoder(model);
    if (use_cuda_sampling) {
        if (!encoder.evalDevice(tokens, params.threads, text_cond_device)) {
            return false;
        }
        if (params.use_local_transformer) {
            if (!cond_hidden_device.alloc2d(
                    model, GGML_TYPE_F32, h.n_embd, 1, "decoder_hidden_cond_device")) {
                return false;
            }
            if (params.use_cfg &&
                !uncond_hidden_device.alloc2d(
                    model, GGML_TYPE_F32, h.n_embd, 1, "decoder_hidden_uncond_device")) {
                return false;
            }
        }
#if defined(MAGPIETTS_CUDA_SAMPLING)
        cuda_sampler.reset(magpietts_cuda_sampler_create(h.stacked_audio_codebooks()));
        if (!cuda_sampler) {
            fprintf(stderr, "failed to create CUDA sampler\n");
            return false;
        }
#else
        fprintf(stderr, "CUDA sampling was not compiled into this MagpieTTS build\n");
        return false;
#endif
    } else if (!encoder.eval(tokens, params.threads, text_cond)) {
        return false;
    }

    const int64_t generation_start_us = ggml_time_us();
    {
        const ggml_nvtx::range nvtx_loop("magpietts_generate_loop");
        const int max_decoder_positions =
            (h.max_decoder_steps + h.frame_stacking_factor - 1) / h.frame_stacking_factor;
        for (int step = 0; step < max_decoder_positions; ++step) {
            const ggml_nvtx::range nvtx_step("magpietts_generate_step");
            const int64_t frame_start_us = ggml_time_us();
            const int frames_remaining = h.max_decoder_steps - step * h.frame_stacking_factor;
            if (frames_remaining <= 0) {
                break;
            }
            if (step % 10 == 0) {
                fprintf(stderr, "%s decoding frame %d/%d\n", label, step, max_decoder_positions);
            }

            const bool forbid_eos = step * h.frame_stacking_factor < h.min_generated_frames;
            std::vector<int32_t> next_codes;
            std::vector<int32_t> argmax_codes;
            std::vector<float> alignment_scores;
            magpietts_decoder_attention decoder_attention;
            decoder_attention.prior = attention_prior.priorForStep(h, (int)tokens.size());
            if (attention_prior.shouldCollect(h, step, (int)tokens.size())) {
                decoder_attention.alignment_scores = &alignment_scores;
            }
            const magpietts_decoder_attention* decoder_attention_arg =
                (decoder_attention.prior || decoder_attention.alignment_scores) ? &decoder_attention
                                                                                : nullptr;

            decoder_result cond;
            decoder_result uncond;
            cond.logits_required = !params.use_local_transformer;
            uncond.logits_required = !params.use_local_transformer;
            if (use_cuda_sampling) {
                if (params.use_local_transformer) {
                    const bool decode_ok =
                        params.use_cfg
                            ? (params.use_kv_cache
                                   ? decoder.evalCachedPair(
                                         text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                         params.threads, cond_kv, uncond_kv, cond, uncond,
                                         max_decoder_positions, nullptr, &text_cond_device,
                                         &cond_hidden_device, &uncond_hidden_device, &cond_cross_kv,
                                         decoder_attention_arg)
                                   : decoder.evalPair(
                                         text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                         params.threads, cond, uncond, nullptr, &text_cond_device,
                                         &cond_hidden_device, &uncond_hidden_device,
                                         decoder_attention_arg))
                            : (params.use_kv_cache
                                   ? decoder.evalCached(
                                         text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                         true, params.threads, cond_kv, cond, nullptr,
                                         &text_cond_device, &cond_hidden_device, &cond_cross_kv,
                                         decoder_attention_arg)
                                   : decoder.eval(
                                         text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                         true, params.threads, cond, nullptr, &text_cond_device,
                                         &cond_hidden_device, decoder_attention_arg));
                    if (!decode_ok) {
                        return false;
                    }
#if defined(MAGPIETTS_CUDA_SAMPLING)
                    if (!local_sampler.sampleCuda(
                            cond_hidden_device, uncond_hidden_device, params.use_cfg, h.cfg_scale,
                            h.temperature, h.top_k, forbid_eos, cuda_sampler.get(),
                            (uint64_t)(uint32_t)params.seed, step, next_codes, argmax_codes)) {
                        return false;
                    }
#else
                    fprintf(stderr, "CUDA sampling was not compiled into this MagpieTTS build\n");
                    return false;
#endif
                } else {
                    magpietts_cuda_sample_request cuda_sample;
#if defined(MAGPIETTS_CUDA_SAMPLING)
                    cuda_sample.sampler = cuda_sampler.get();
#endif
                    cuda_sample.use_cfg = params.use_cfg;
                    cuda_sample.cfg_scale = h.cfg_scale;
                    cuda_sample.temperature = h.temperature;
                    cuda_sample.top_k = h.top_k;
                    cuda_sample.forbid_audio_eos = forbid_eos;
                    cuda_sample.seed = (uint64_t)(uint32_t)params.seed;
                    cuda_sample.frame_index = step;

                    const bool decode_ok =
                        params.use_cfg
                            ? (params.use_kv_cache
                                   ? decoder.evalCachedPair(
                                         text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                         params.threads, cond_kv, uncond_kv, cond, uncond,
                                         max_decoder_positions, &cuda_sample, &text_cond_device,
                                         nullptr, nullptr, &cond_cross_kv, decoder_attention_arg)
                                   : decoder.evalPair(
                                         text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                         params.threads, cond, uncond, &cuda_sample,
                                         &text_cond_device, nullptr, nullptr,
                                         decoder_attention_arg))
                            : (params.use_kv_cache
                                   ? decoder.evalCached(
                                         text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                         true, params.threads, cond_kv, cond, &cuda_sample,
                                         &text_cond_device, nullptr, &cond_cross_kv,
                                         decoder_attention_arg)
                                   : decoder.eval(
                                         text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                         true, params.threads, cond, &cuda_sample,
                                         &text_cond_device, nullptr, decoder_attention_arg));
                    if (!decode_ok) {
                        return false;
                    }
                    next_codes = std::move(cuda_sample.codes);
                    argmax_codes = std::move(cuda_sample.argmax_codes);
                }
                if ((int)next_codes.size() != h.stacked_audio_codebooks() ||
                    (int)argmax_codes.size() != h.stacked_audio_codebooks()) {
                    fprintf(stderr, "CUDA sampler returned an unexpected number of codebooks\n");
                    return false;
                }
            } else {
                if (params.use_cfg) {
                    const bool pair_ok =
                        params.use_kv_cache
                            ? decoder.evalCachedPair(
                                  text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                  params.threads, cond_kv, uncond_kv, cond, uncond,
                                  max_decoder_positions, nullptr, nullptr, nullptr, nullptr,
                                  &cond_cross_kv, decoder_attention_arg)
                            : decoder.evalPair(
                                  text_cond, (int)tokens.size(), audio_codes, params.speaker,
                                  params.threads, cond, uncond, nullptr, nullptr, nullptr, nullptr,
                                  decoder_attention_arg);
                    if (!pair_ok) {
                        return false;
                    }
                } else {
                    const bool cond_ok =
                        params.use_kv_cache
                            ? decoder.evalCached(
                                  text_cond, (int)tokens.size(), audio_codes, params.speaker, true,
                                  params.threads, cond_kv, cond, nullptr, nullptr, nullptr,
                                  &cond_cross_kv, decoder_attention_arg)
                            : decoder.eval(
                                  text_cond, (int)tokens.size(), audio_codes, params.speaker, true,
                                  params.threads, cond, nullptr, nullptr, nullptr,
                                  decoder_attention_arg);
                    if (!cond_ok) {
                        return false;
                    }
                }

                if (params.use_local_transformer) {
                    if (!local_sampler.sample(
                            cond.hidden_last, uncond.hidden_last, params.use_cfg, h.cfg_scale,
                            h.temperature, h.top_k, forbid_eos, rng, next_codes, argmax_codes)) {
                        return false;
                    }
                } else {
                    next_codes = MagpieCodebookSampler::sampleParallel(
                        cond.logits_last, uncond.logits_last, h, params.use_cfg, h.cfg_scale,
                        h.temperature, h.top_k, forbid_eos, rng, &argmax_codes);
                }
            }

            if (decoder_attention.alignment_scores && !alignment_scores.empty()) {
                attention_prior.update(h, step, (int)tokens.size(), alignment_scores);
            }

            std::vector<std::vector<int32_t>> codec_frames;
            if (!magpietts_unstack_codes(next_codes, h, codec_frames)) {
                fprintf(stderr, "sampled an invalid stacked MagpieTTS frame\n");
                return false;
            }
            const int eos_lane =
                forbid_eos ? -1 : magpietts_first_eos_lane(next_codes, argmax_codes, h);
            for (int c = 0; c < h.audio_codebooks; ++c) {
                for (int lane = 0; lane < h.frame_stacking_factor; ++lane) {
                    audio_codes[c].push_back(next_codes[c + lane * h.audio_codebooks]);
                }
            }
            const int frames_to_emit =
                magpietts_frames_to_emit(frames_remaining, h.frame_stacking_factor, eos_lane);
            for (int lane = 0; lane < frames_to_emit; ++lane) {
                generated_frames.push_back(codec_frames[(size_t)lane]);
            }
            if (eos_lane >= 0) {
                ggml_nvtx::mark("magpietts_eos");
                fprintf(stderr, "%s EOS detected at frame %d\n", label, step);
            }

            if (frames_to_emit > 0) {
                const int64_t frame_done_us = ggml_time_us();
                bool first_frame = false;
                const double inter_ms =
                    metrics.record_frame(frame_done_us, first_frame, frames_to_emit);
                const double frame_latency_ms = (double)(frame_done_us - frame_start_us) / 1000.0;
                if (step < 4 || step % 10 == 0) {
                    if (first_frame) {
                        fprintf(
                            stderr, "%s frame %d latency=%.2f ms ttff=%.2f ms\n", label, step,
                            frame_latency_ms, metrics.ttff_ms);
                    } else {
                        fprintf(
                            stderr, "%s frame %d latency=%.2f ms inter=%.2f ms\n", label, step,
                            frame_latency_ms, inter_ms);
                    }
                }
            }
            if (eos_lane >= 0) {
                break;
            }
        }
    }

    metrics.finish(generated_frames.size(), params.codec_fps);
    const double generation_elapsed_s = (double)(ggml_time_us() - generation_start_us) / 1000000.0;
    const char* summary_prefix = std::strcmp(label, "warmup") == 0 ? "warmup " : "";
    fprintf(
        stderr,
        "%sgenerated %zu codec frames in %.2f s; "
        "e2e audio=%.3f s elapsed=%.2f s rtf=%.4f rtfx=%.2f "
        "ttff=%.2f ms inter_frame_avg=%.2f ms inter_frame_min=%.2f ms inter_frame_max=%.2f ms "
        "frames=%d\n",
        summary_prefix, generated_frames.size(), generation_elapsed_s, metrics.audio_s,
        metrics.e2e_elapsed_s, metrics.e2e_rtf, metrics.e2e_rtfx, metrics.ttff_ms,
        metrics.inter_frame_avg_ms(), metrics.inter_frame_min_value_ms(),
        metrics.inter_frame_max_ms, metrics.frames);

    if (generated_frames_out) {
        *generated_frames_out = std::move(generated_frames);
    }
    return true;
}

bool
generate_magpietts_codes(
    magpietts_model& model, const magpietts_params& params, const std::vector<int32_t>& tokens,
    std::vector<std::vector<int32_t>>* generated_frames_out, magpietts_run_metrics& metrics,
    const char* run_label, const magpietts_model* local_transformer_cpu_model) {
    MagpieCodeGenerator generator(model);
    return generator.generate(
        params, tokens, generated_frames_out, metrics, run_label, local_transformer_cpu_model);
}

}  // namespace nemo_speech::tts
