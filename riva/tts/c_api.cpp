// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// C ABI implementation: maps the generic nemo_speech_tts_* surface onto tts::Synthesizer.
// Every exported function is noexcept in spirit: exceptions become a status plus
// a thread-local last-error string. No C++ types or exceptions cross the ABI.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nemo_speech/tts.h"
#include "synthesizer.h"

namespace tts_core = nemo_speech::tts;

struct nemo_speech_tts_synthesizer {
    std::unique_ptr<tts_core::Synthesizer> synthesizer;
    std::vector<std::string> speaker_names;
    int sample_rate = 0;
};

namespace {

thread_local std::string g_last_error;

void
set_last_error(const std::string& m) {
    g_last_error = m;
}

// A field is present iff the caller's struct `size` covers it (append-only ABI:
// a caller compiled against an older header sends a smaller size; missing tail
// fields keep their library defaults).
#define HAS_FIELD(ptr, type, member) \
    ((ptr) != nullptr && (ptr)->size >= offsetof(type, member) + sizeof(((type*)nullptr)->member))

std::string
str_or_empty(const char* s) {
    return s ? std::string(s) : std::string();
}

tts_core::MagpieBackendPreference
to_backend(nemo_speech_tts_backend_preference backend) {
    switch (backend) {
        case NEMO_SPEECH_TTS_BACKEND_CPU:
            return tts_core::MagpieBackendPreference::Cpu;
        case NEMO_SPEECH_TTS_BACKEND_CUDA:
            return tts_core::MagpieBackendPreference::Cuda;
        case NEMO_SPEECH_TTS_BACKEND_AUTO:
        default:
            return tts_core::MagpieBackendPreference::Auto;
    }
}

tts_core::MagpieUmaMode
to_uma(nemo_speech_tts_uma_mode mode) {
    switch (mode) {
        case NEMO_SPEECH_TTS_UMA_OFF:
            return tts_core::MagpieUmaMode::Off;
        case NEMO_SPEECH_TTS_UMA_ON:
            return tts_core::MagpieUmaMode::On;
        case NEMO_SPEECH_TTS_UMA_AUTO:
        default:
            return tts_core::MagpieUmaMode::Auto;
    }
}

tts_core::MagpieLongformMode
to_longform(nemo_speech_tts_longform_mode mode) {
    switch (mode) {
        case NEMO_SPEECH_TTS_LONGFORM_OFF:
            return tts_core::MagpieLongformMode::Off;
        case NEMO_SPEECH_TTS_LONGFORM_ON:
            return tts_core::MagpieLongformMode::On;
        case NEMO_SPEECH_TTS_LONGFORM_AUTO:
        default:
            return tts_core::MagpieLongformMode::Auto;
    }
}

nemo_speech_tts_backend_preference
from_backend(tts_core::MagpieBackendPreference backend) {
    switch (backend) {
        case tts_core::MagpieBackendPreference::Cpu:
            return NEMO_SPEECH_TTS_BACKEND_CPU;
        case tts_core::MagpieBackendPreference::Cuda:
            return NEMO_SPEECH_TTS_BACKEND_CUDA;
        case tts_core::MagpieBackendPreference::Auto:
        default:
            return NEMO_SPEECH_TTS_BACKEND_AUTO;
    }
}

nemo_speech_tts_uma_mode
from_uma(tts_core::MagpieUmaMode mode) {
    switch (mode) {
        case tts_core::MagpieUmaMode::Off:
            return NEMO_SPEECH_TTS_UMA_OFF;
        case tts_core::MagpieUmaMode::On:
            return NEMO_SPEECH_TTS_UMA_ON;
        case tts_core::MagpieUmaMode::Auto:
        default:
            return NEMO_SPEECH_TTS_UMA_AUTO;
    }
}

nemo_speech_tts_longform_mode
from_longform(tts_core::MagpieLongformMode mode) {
    switch (mode) {
        case tts_core::MagpieLongformMode::Off:
            return NEMO_SPEECH_TTS_LONGFORM_OFF;
        case tts_core::MagpieLongformMode::On:
            return NEMO_SPEECH_TTS_LONGFORM_ON;
        case tts_core::MagpieLongformMode::Auto:
        default:
            return NEMO_SPEECH_TTS_LONGFORM_AUTO;
    }
}

tts_core::MagpieRuntimeConfig
runtime_config_defaults() {
    return tts_core::MagpieRuntimeConfig{};
}

tts_core::MagpieRuntimeConfig
to_config(const nemo_speech_tts_synthesizer_config* c) {
    tts_core::MagpieRuntimeConfig cfg = runtime_config_defaults();
    if (!c)
        return cfg;
    if (HAS_FIELD(c, nemo_speech_tts_synthesizer_config, model) && c->model) {
        const auto* m = c->model;
        if (HAS_FIELD(m, nemo_speech_tts_model_config, magpie_model))
            cfg.magpie_model = str_or_empty(m->magpie_model);
        if (HAS_FIELD(m, nemo_speech_tts_model_config, codec_model))
            cfg.codec_model = str_or_empty(m->codec_model);
    }
    if (HAS_FIELD(c, nemo_speech_tts_synthesizer_config, runtime) && c->runtime) {
        const auto* r = c->runtime;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, speaker))
            cfg.speaker = r->speaker;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, threads))
            cfg.threads = r->threads;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, codec_threads))
            cfg.codec_threads = r->codec_threads;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, seed))
            cfg.seed = r->seed;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, steps))
            cfg.steps = r->steps;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, top_k))
            cfg.top_k = r->top_k;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, chunk_frames))
            cfg.chunk_frames = r->chunk_frames;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, codec_queue_depth))
            cfg.codec_queue_depth = r->codec_queue_depth;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, codec_history_frames))
            cfg.codec_history_frames = r->codec_history_frames;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, codec_future_frames))
            cfg.codec_future_frames = r->codec_future_frames;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, window_ms))
            cfg.window_ms = r->window_ms;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, temperature))
            cfg.temperature = r->temperature;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, override_temperature))
            cfg.override_temperature = r->override_temperature;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, cfg_scale))
            cfg.cfg_scale = r->cfg_scale;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, override_cfg_scale))
            cfg.override_cfg_scale = r->override_cfg_scale;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, use_cfg))
            cfg.use_cfg = r->use_cfg;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, use_local_transformer))
            cfg.use_local_transformer = r->use_local_transformer;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, use_kv_cache))
            cfg.use_kv_cache = r->use_kv_cache;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, use_stateful_codec))
            cfg.use_stateful_codec = r->use_stateful_codec;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, codec_cpu))
            cfg.codec_cpu = r->codec_cpu;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, flush_partial_chunk))
            cfg.flush_partial_chunk = r->flush_partial_chunk;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, verbose))
            cfg.verbose = r->verbose;
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, lt_backend))
            cfg.lt_backend = to_backend(r->lt_backend);
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, sampling_backend))
            cfg.sampling_backend = to_backend(r->sampling_backend);
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, uma_mode))
            cfg.uma_mode = to_uma(r->uma_mode);
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, longform_mode))
            cfg.longform_mode = to_longform(r->longform_mode);
        if (HAS_FIELD(r, nemo_speech_tts_runtime_config, lt_fp32))
            cfg.lt_fp32 = r->lt_fp32;
    }
    return cfg;
}

std::string
tokenizer_dir_of(const nemo_speech_tts_synthesizer_config* c) {
    if (!c || !HAS_FIELD(c, nemo_speech_tts_synthesizer_config, model) || !c->model)
        return {};
    const auto* m = c->model;
    if (!HAS_FIELD(m, nemo_speech_tts_model_config, tokenizer_model_dir))
        return {};
    return str_or_empty(m->tokenizer_model_dir);
}

tts_core::SynthesizerConfig
synthesizer_config_of(const nemo_speech_tts_synthesizer_config* c) {
    tts_core::SynthesizerConfig cfg;
    cfg.runtime = to_config(c);
    cfg.tokenizer_model_dir = tokenizer_dir_of(c);
    if (c && HAS_FIELD(c, nemo_speech_tts_synthesizer_config, model) && c->model) {
        const auto* model = c->model;
        if (HAS_FIELD(model, nemo_speech_tts_model_config, text_normalizer_model_dir)) {
            cfg.text_normalizer_model_dir = str_or_empty(model->text_normalizer_model_dir);
        }
    }
    if (HAS_FIELD(c, nemo_speech_tts_synthesizer_config, default_language_code))
        cfg.default_language_code = str_or_empty(c->default_language_code);
    if (HAS_FIELD(c, nemo_speech_tts_synthesizer_config, default_voice_name))
        cfg.default_voice_name = str_or_empty(c->default_voice_name);
    return cfg;
}

tts_core::MagpieSynthesisOptions
to_options(const nemo_speech_tts_synthesis_options* o) {
    tts_core::MagpieSynthesisOptions opts;
    if (!o)
        return opts;
    if (HAS_FIELD(o, nemo_speech_tts_synthesis_options, speaker))
        opts.speaker = o->speaker;
    if (HAS_FIELD(o, nemo_speech_tts_synthesis_options, seed))
        opts.seed = o->seed;
    if (HAS_FIELD(o, nemo_speech_tts_synthesis_options, steps))
        opts.steps = o->steps;
    if (HAS_FIELD(o, nemo_speech_tts_synthesis_options, top_k))
        opts.top_k = o->top_k;
    if (HAS_FIELD(o, nemo_speech_tts_synthesis_options, temperature))
        opts.temperature = o->temperature;
    if (HAS_FIELD(o, nemo_speech_tts_synthesis_options, override_temperature))
        opts.override_temperature = o->override_temperature;
    if (HAS_FIELD(o, nemo_speech_tts_synthesis_options, cfg_scale))
        opts.cfg_scale = o->cfg_scale;
    if (HAS_FIELD(o, nemo_speech_tts_synthesis_options, override_cfg_scale))
        opts.override_cfg_scale = o->override_cfg_scale;
    return opts;
}

std::string
language_of(const nemo_speech_tts_synthesis_options* o) {
    return HAS_FIELD(o, nemo_speech_tts_synthesis_options, language_code)
               ? str_or_empty(o->language_code)
               : std::string();
}

std::string
voice_of(const nemo_speech_tts_synthesis_options* o) {
    return HAS_FIELD(o, nemo_speech_tts_synthesis_options, voice_name) ? str_or_empty(o->voice_name)
                                                                       : std::string();
}

int
output_rate_of(const nemo_speech_tts_synthesis_options* o) {
    return HAS_FIELD(o, nemo_speech_tts_synthesis_options, output_sample_rate)
               ? o->output_sample_rate
               : 0;
}

nemo_speech_tts_synthesis_stats
to_c_stats(const tts_core::MagpieSynthesisStats& s) {
    nemo_speech_tts_synthesis_stats out;
    std::memset(&out, 0, sizeof(out));
    out.size = sizeof(out);
    out.sample_rate = s.sample_rate;
    out.generated_frames = s.generated_frames;
    out.chunks = s.chunks;
    out.e2e_chunks = s.e2e_chunks;
    out.samples_written = s.samples_written;
    out.tokenizer_ms = s.tokenizer_ms;
    out.encoder_ms = s.encoder_ms;
    out.audio_s = s.audio_s;
    out.elapsed_s = s.elapsed_s;
    out.rtf = s.rtf;
    out.rtfx = s.rtfx;
    out.ttfa_ms = s.ttfa_ms;
    out.icl_avg_ms = s.icl_avg_ms;
    out.icl_min_ms = s.icl_min_ms;
    out.icl_max_ms = s.icl_max_ms;
    out.decoder_audio_s = s.decoder_audio_s;
    out.decoder_elapsed_s = s.decoder_elapsed_s;
    out.decoder_rtfx = s.decoder_rtfx;
    out.decoder_ttft_ms = s.decoder_ttft_ms;
    out.decoder_itl_avg_ms = s.decoder_itl_avg_ms;
    out.decoder_itl_min_ms = s.decoder_itl_min_ms;
    out.decoder_itl_max_ms = s.decoder_itl_max_ms;
    out.decoder_itl_p95_ms = s.decoder_itl_p95_ms;
    out.decoder_itl_p99_ms = s.decoder_itl_p99_ms;
    out.codec_audio_s = s.codec_audio_s;
    out.codec_elapsed_s = s.codec_elapsed_s;
    out.codec_rtfx = s.codec_rtfx;
    out.codec_ttfa_ms = s.codec_ttfa_ms;
    out.codec_icl_avg_ms = s.codec_icl_avg_ms;
    out.codec_icl_min_ms = s.codec_icl_min_ms;
    out.codec_icl_max_ms = s.codec_icl_max_ms;
    out.codec_icl_p95_ms = s.codec_icl_p95_ms;
    out.codec_icl_p99_ms = s.codec_icl_p99_ms;
    out.e2e_ttfa_ms = s.e2e_ttfa_ms;
    out.e2e_icl_avg_ms = s.e2e_icl_avg_ms;
    out.e2e_icl_min_ms = s.e2e_icl_min_ms;
    out.e2e_icl_max_ms = s.e2e_icl_max_ms;
    out.e2e_icl_p95_ms = s.e2e_icl_p95_ms;
    out.e2e_icl_p99_ms = s.e2e_icl_p99_ms;
    out.e2e_rtfx = s.e2e_rtfx;
    return out;
}

bool
validate_stats_out(const nemo_speech_tts_synthesis_stats* stats_out) {
    return !stats_out || stats_out->size >= sizeof(size_t);
}

void
copy_stats(nemo_speech_tts_synthesis_stats* stats_out, const tts_core::MagpieSynthesisStats& s) {
    if (!stats_out)
        return;
    nemo_speech_tts_synthesis_stats tmp = to_c_stats(s);
    const size_t n = std::min(stats_out->size, sizeof(tmp));
    std::memcpy(stats_out, &tmp, n);
}

tts_core::Synthesizer::PcmCallback
make_callback(nemo_speech_tts_pcm_callback callback, void* user_data) {
    if (!callback)
        return tts_core::Synthesizer::PcmCallback{};
    return [callback, user_data](const tts_core::SynthesisMetadata&, const std::string& pcm) {
        if (pcm.empty())
            return true;
        return callback(reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size(), user_data);
    };
}

// Wrap a body, mapping exceptions to status codes + last-error.
template <class F>
nemo_speech_tts_status
guard(F&& f) {
    try {
        return f();
    }
    catch (const std::bad_alloc&) {
        set_last_error("out of memory");
        return NEMO_SPEECH_TTS_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::invalid_argument& e) {
        set_last_error(e.what());
        return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
    }
    catch (const std::exception& e) {
        set_last_error(e.what());
        return NEMO_SPEECH_TTS_ERROR_RUNTIME;
    }
    catch (...) {
        set_last_error("unknown error");
        return NEMO_SPEECH_TTS_ERROR_RUNTIME;
    }
}

}  // namespace

extern "C" {

nemo_speech_tts_runtime_config
nemo_speech_tts_runtime_config_default(void) {
    const tts_core::MagpieRuntimeConfig d = runtime_config_defaults();
    nemo_speech_tts_runtime_config c;
    std::memset(&c, 0, sizeof(c));
    c.size = sizeof(c);
    c.speaker = d.speaker;
    c.threads = d.threads;
    c.codec_threads = d.codec_threads;
    c.seed = d.seed;
    c.steps = d.steps;
    c.top_k = d.top_k;
    c.chunk_frames = d.chunk_frames;
    c.codec_queue_depth = d.codec_queue_depth;
    c.codec_history_frames = d.codec_history_frames;
    c.codec_future_frames = d.codec_future_frames;
    c.window_ms = d.window_ms;
    c.temperature = d.temperature;
    c.override_temperature = d.override_temperature;
    c.cfg_scale = d.cfg_scale;
    c.override_cfg_scale = d.override_cfg_scale;
    c.use_cfg = d.use_cfg;
    c.use_local_transformer = d.use_local_transformer;
    c.use_kv_cache = d.use_kv_cache;
    c.use_stateful_codec = d.use_stateful_codec;
    c.codec_cpu = d.codec_cpu;
    c.flush_partial_chunk = d.flush_partial_chunk;
    c.verbose = d.verbose;
    c.lt_backend = from_backend(d.lt_backend);
    c.sampling_backend = from_backend(d.sampling_backend);
    c.uma_mode = from_uma(d.uma_mode);
    c.longform_mode = from_longform(d.longform_mode);
    c.lt_fp32 = d.lt_fp32;
    return c;
}

nemo_speech_tts_synthesis_options
nemo_speech_tts_synthesis_options_default(void) {
    nemo_speech_tts_synthesis_options o;
    std::memset(&o, 0, sizeof(o));
    o.size = sizeof(o);
    o.speaker = -1;
    o.seed = -1;
    o.steps = -1;
    o.top_k = -1;
    return o;
}

nemo_speech_tts_synthesis_stats
nemo_speech_tts_synthesis_stats_default(void) {
    nemo_speech_tts_synthesis_stats s;
    std::memset(&s, 0, sizeof(s));
    s.size = sizeof(s);
    return s;
}

nemo_speech_tts_status
nemo_speech_tts_create(
    const nemo_speech_tts_synthesizer_config* cfg, nemo_speech_tts_synthesizer** out) {
    if (!out)
        return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return guard([&] {
        tts_core::SynthesizerConfig synthesizer_config = synthesizer_config_of(cfg);
        if (synthesizer_config.runtime.magpie_model.empty()) {
            set_last_error("synthesizer config: model.magpie_model is required");
            return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
        }
        if (synthesizer_config.runtime.codec_model.empty()) {
            set_last_error("synthesizer config: model.codec_model is required");
            return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
        }
        auto h = std::make_unique<nemo_speech_tts_synthesizer>();
        h->synthesizer = std::make_unique<tts_core::Synthesizer>(std::move(synthesizer_config));
        h->speaker_names = h->synthesizer->speaker_names();
        h->sample_rate = h->synthesizer->sample_rate();

        *out = h.release();
        return NEMO_SPEECH_TTS_OK;
    });
}

void
nemo_speech_tts_destroy(nemo_speech_tts_synthesizer* synthesizer) {
    delete synthesizer;
}

int32_t
nemo_speech_tts_sample_rate(const nemo_speech_tts_synthesizer* synthesizer) {
    return synthesizer ? synthesizer->sample_rate : 0;
}

int32_t
nemo_speech_tts_speaker_count(const nemo_speech_tts_synthesizer* synthesizer) {
    return synthesizer ? static_cast<int32_t>(synthesizer->speaker_names.size()) : 0;
}

const char*
nemo_speech_tts_speaker_name(const nemo_speech_tts_synthesizer* synthesizer, size_t i) {
    if (!synthesizer || i >= synthesizer->speaker_names.size())
        return "";
    return synthesizer->speaker_names[i].c_str();
}

nemo_speech_tts_status
nemo_speech_tts_synthesize_text(
    nemo_speech_tts_synthesizer* synthesizer, const nemo_speech_tts_synthesis_options* options,
    const char* text, nemo_speech_tts_pcm_callback callback, void* user_data,
    nemo_speech_tts_synthesis_stats* stats_out) {
    if (!synthesizer)
        return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
    if (!validate_stats_out(stats_out)) {
        set_last_error("stats_out.size must be at least sizeof(size_t)");
        return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
    }
    return guard([&] {
        tts_core::SynthesisRequest request;
        request.text = str_or_empty(text);
        request.language_code = language_of(options);
        request.voice_name = voice_of(options);
        request.output_sample_rate = output_rate_of(options);
        request.options = to_options(options);
        const tts_core::SynthesisResult result =
            synthesizer->synthesizer->synthesize(request, make_callback(callback, user_data));
        if (result.cancelled) {
            set_last_error("synthesis cancelled");
            return NEMO_SPEECH_TTS_ERROR_CANCELLED;
        }
        copy_stats(stats_out, result.stats);
        return NEMO_SPEECH_TTS_OK;
    });
}

nemo_speech_tts_status
nemo_speech_tts_synthesize_tokens(
    nemo_speech_tts_synthesizer* synthesizer, const nemo_speech_tts_synthesis_options* options,
    const int32_t* tokens, size_t token_count, nemo_speech_tts_pcm_callback callback,
    void* user_data, nemo_speech_tts_synthesis_stats* stats_out) {
    if (!synthesizer)
        return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
    if (!validate_stats_out(stats_out)) {
        set_last_error("stats_out.size must be at least sizeof(size_t)");
        return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
    }
    return guard([&] {
        if (!tokens || token_count == 0) {
            set_last_error("text token list is empty");
            return NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT;
        }
        const std::vector<int32_t> input(tokens, tokens + token_count);
        const tts_core::SynthesisResult result = synthesizer->synthesizer->synthesize_tokens(
            input, to_options(options), output_rate_of(options),
            make_callback(callback, user_data));
        if (result.cancelled) {
            set_last_error("synthesis cancelled");
            return NEMO_SPEECH_TTS_ERROR_CANCELLED;
        }
        copy_stats(stats_out, result.stats);
        return NEMO_SPEECH_TTS_OK;
    });
}

const char*
nemo_speech_tts_last_error(void) {
    return g_last_error.c_str();
}

#ifndef NEMO_SPEECH_VERSION_STR
#define NEMO_SPEECH_VERSION_STR "0.0.0"  // overridden by the build from ./VERSION
#endif
const char*
nemo_speech_tts_version(void) {
    return "nemo-speech-tts " NEMO_SPEECH_VERSION_STR;
}

}  // extern "C"
