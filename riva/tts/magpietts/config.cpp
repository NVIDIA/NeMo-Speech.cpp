// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "config.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include "parameter_parser.h"

namespace nemo_speech::tts {

void
MagpieTokenizerSentenceLimits::Register(common::ParameterParser& p) {
    p.Register("en", &en, "English sentence-chunking word threshold");
    p.Register("es", &es, "Spanish sentence-chunking word threshold");
    p.Register("fr", &fr, "French sentence-chunking word threshold");
    p.Register("vi", &vi, "Vietnamese sentence-chunking word threshold");
    p.Register("it", &it, "Italian sentence-chunking word threshold");
    p.Register("de", &de, "German sentence-chunking word threshold");
    p.Register("zh", &zh, "Mandarin sentence-chunking character threshold");
    p.Register("hi", &hi, "Hindi sentence-chunking word threshold");
    p.Register("ja", &ja, "Japanese sentence-chunking character threshold");
}

void
MagpieTokenizerConfig::Register(common::ParameterParser& p) {
    p.Register("sentence-limit", sentence_limit);
}

namespace {

std::string
lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

[[noreturn]] void
bad_choice(const std::string& key, const std::string& value, const char* choices) {
    throw std::invalid_argument(
        "config " + key + ": invalid value '" + value + "' (expected " + choices + ")");
}

void
register_negated_bool(
    common::ParameterParser& p, const std::string& name, bool* field, const std::string& key,
    const std::string& doc) {
    p.Register(
        name,
        [field, key](const std::string& value) { *field = !common::detail::to_bool(key, value); },
        doc, {}, true);
}

MagpieBackendPreference
runtime_backend_from_string(const std::string& key, const std::string& value) {
    const std::string v = lower_ascii(value);
    if (v == "auto") {
        return MagpieBackendPreference::Auto;
    }
    if (v == "cpu") {
        return MagpieBackendPreference::Cpu;
    }
    if (v == "cuda") {
        return MagpieBackendPreference::Cuda;
    }
    bad_choice(key, value, "auto, cpu, or cuda");
}

MagpieUmaMode
runtime_uma_from_string(const std::string& key, const std::string& value) {
    const std::string v = lower_ascii(value);
    if (v == "auto") {
        return MagpieUmaMode::Auto;
    }
    if (v == "off") {
        return MagpieUmaMode::Off;
    }
    if (v == "on") {
        return MagpieUmaMode::On;
    }
    bad_choice(key, value, "auto, off, or on");
}

MagpieLongformMode
runtime_longform_from_string(const std::string& key, const std::string& value) {
    const std::string v = lower_ascii(value);
    if (v == "auto") {
        return MagpieLongformMode::Auto;
    }
    if (v == "off") {
        return MagpieLongformMode::Off;
    }
    if (v == "on") {
        return MagpieLongformMode::On;
    }
    bad_choice(key, value, "auto, off, or on");
}

magpietts_backend_preference
stream_backend_from_string(const std::string& key, const std::string& value) {
    magpietts_backend_preference backend = MAGPIETTS_BACKEND_AUTO;
    if (parse_backend_preference(lower_ascii(value), backend)) {
        return backend;
    }
    bad_choice(key, value, "auto, cpu, or cuda");
}

magpietts_uma_mode
stream_uma_from_string(const std::string& key, const std::string& value) {
    magpietts_uma_mode mode = MAGPIETTS_UMA_AUTO;
    if (parse_uma_mode(lower_ascii(value), mode)) {
        return mode;
    }
    bad_choice(key, value, "auto, off, or on");
}

magpie_longform_mode
stream_longform_from_string(const std::string& key, const std::string& value) {
    magpie_longform_mode mode = MAGPIE_LONGFORM_AUTO;
    if (parse_magpie_longform_mode(lower_ascii(value), mode)) {
        return mode;
    }
    bad_choice(key, value, "auto, off, or on");
}

void
register_runtime_config(common::ParameterParser& p, MagpieRuntimeConfig& c) {
    p.Register("magpie-model", &c.magpie_model, "MagpieTTS GGUF token generator path");
    p.Register("codec-model", &c.codec_model, "NanoCodec decoder GGUF path");
    p.Register("speaker", &c.speaker, "Default baked speaker index");
    p.Register("threads", &c.threads, "CPU threads for Magpie and codec", {"--threads"});
    p.Register("codec-threads", &c.codec_threads, "Codec CPU threads; 0 uses threads");
    p.Register("seed", &c.seed, "Default RNG seed; -1 uses current time");
    p.Register("steps", &c.steps, "Maximum decoder frames; -1 uses model default");
    p.Register("top-k", &c.top_k, "Top-k sampling; -1 uses model default");
    p.Register("chunk-frames", &c.chunk_frames, "Codec frames per streamed audio chunk");
    p.Register("codec-queue-depth", &c.codec_queue_depth, "Codec worker queue depth");
    p.Register("codec-history-frames", &c.codec_history_frames, "Rolling codec history frames");
    p.Register("codec-future-frames", &c.codec_future_frames, "Rolling codec future frames");
    p.Register("window-ms", &c.window_ms, "Overlap-add window duration in milliseconds");
    p.Register(
        "temperature",
        [&c](const std::string& value) {
            c.temperature = common::detail::to_float("tts.temperature", value);
            c.override_temperature = true;
        },
        "Sampling temperature; omitted uses model default");
    p.Register(
        "cfg-scale",
        [&c](const std::string& value) {
            c.cfg_scale = common::detail::to_float("tts.cfg-scale", value);
            c.override_cfg_scale = true;
        },
        "Classifier-free guidance scale; omitted uses model default");
    p.Register("use-cfg", &c.use_cfg, "Enable classifier-free guidance");
    register_negated_bool(
        p, "no-cfg", &c.use_cfg, "tts.no-cfg", "Disable classifier-free guidance");
    p.Register("use-local-transformer", &c.use_local_transformer, "Enable local transformer");
    register_negated_bool(
        p, "no-local-transformer", &c.use_local_transformer, "tts.no-local-transformer",
        "Disable local transformer");
    p.Register(
        "lt-fp32", &c.lt_fp32, "Run the local transformer entirely in FP32",
        {"--tts.local-transformer-fp32"});
    p.Register("use-kv-cache", &c.use_kv_cache, "Enable decoder KV cache");
    register_negated_bool(
        p, "no-kv-cache", &c.use_kv_cache, "tts.no-kv-cache",
        "Recompute the full decoder prefix each frame");
    p.Register("use-stateful-codec", &c.use_stateful_codec, "Enable fast layer-state codec");
    register_negated_bool(
        p, "no-stateful-codec", &c.use_stateful_codec, "tts.no-stateful-codec",
        "Disable fast layer-state codec");
    p.Register("codec-cpu", &c.codec_cpu, "Force NanoCodec decoder onto CPU backend");
    p.Register("flush-partial-chunk", &c.flush_partial_chunk, "Emit a final partial codec chunk");
    p.Register("verbose", &c.verbose, "Print detailed Magpie/NanoCodec logs", {"--verbose"});
    p.Register(
        "lt-backend",
        [&c](const std::string& value) {
            c.lt_backend = runtime_backend_from_string("tts.lt-backend", value);
        },
        "Local-transformer backend: auto, cpu, or cuda");
    p.Register(
        "sampling-backend",
        [&c](const std::string& value) {
            c.sampling_backend = runtime_backend_from_string("tts.sampling-backend", value);
        },
        "Sampling backend: auto, cpu, or cuda");
    p.Register(
        "uma-mode",
        [&c](const std::string& value) {
            c.uma_mode = runtime_uma_from_string("tts.uma-mode", value);
        },
        "CUDA managed-memory mode: auto, off, or on");
    p.Register(
        "longform",
        [&c](const std::string& value) {
            c.longform_mode = runtime_longform_from_string("tts.longform", value);
        },
        "Sentence-chunk longform synthesis mode: auto, off, or on");
}

void
register_stream_params(common::ParameterParser& p, magpie_stream_params& c) {
    p.Register("magpie-model", &c.magpie_model, "MagpieTTS GGUF token generator path");
    p.Register("codec-model", &c.codec_model, "NanoCodec decoder GGUF path");
    p.Register("wav-out", &c.wav_out, "Streaming mono 16-bit PCM WAV output path");
    p.Register("output", &c.wav_out, "Alias for wav-out");
    p.Register("raw-out", &c.raw_out, "Streaming raw s16le PCM output path");
    p.Register("audio-cmd", &c.audio_cmd, "Command that receives raw s16le PCM on stdin");
    p.Register("codes-out", &c.codes_out, "Path for generated codec frames");
    p.Register(
        "tokens", [&c](const std::string& value) { c.tokens = parse_token_list(value); },
        "Comma/space-separated text token IDs");
    p.Register("tokens-file", &c.tokens_file, "File containing text token IDs");
    p.Register(
        "warmup-tokens",
        [&c](const std::string& value) { c.warmup_tokens = parse_token_list(value); },
        "Comma/space-separated warmup text token IDs");
    p.Register("warmup-tokens-file", &c.warmup_tokens_file, "File containing warmup token IDs");
    p.Register("text", &c.text, "Raw text to synthesize");
    p.Register("text-file", &c.text_file, "File containing raw text to synthesize");
    p.Register("warmup-text", &c.warmup_text, "Raw text for warmup");
    p.Register("warmup-text-file", &c.warmup_text_file, "File containing warmup text");
    p.Register("tokenizer-model-dir", &c.tokenizer_model_dir, "Extracted Magpie .nemo directory");
    p.Register("tn-model-dir", &c.tn_model_dir, "Sparrowhawk TTS TN grammar dir");
    p.Register("tokenizer", c.tokenizer_config);
    p.Register("language-code", &c.language_code, "Text language code", {"--tts.language"});
    p.Register("speaker", &c.speaker, "Baked speaker index");
    p.Register("threads", &c.threads, "CPU threads for Magpie and codec", {"--threads"});
    p.Register("codec-threads", &c.codec_threads, "Codec CPU threads; 0 uses threads");
    p.Register("seed", &c.seed, "RNG seed; -1 uses current time");
    p.Register("steps", &c.steps, "Maximum decoder frames; -1 uses model default");
    p.Register("top-k", &c.top_k, "Top-k sampling; -1 uses model default");
    p.Register("chunk-frames", &c.chunk_frames, "Codec frames per streamed audio chunk");
    p.Register("codec-queue-depth", &c.codec_queue_depth, "Codec worker queue depth");
    p.Register("codec-history-frames", &c.codec_history_frames, "Rolling codec history frames");
    p.Register("codec-future-frames", &c.codec_future_frames, "Rolling codec future frames");
    p.Register("window-ms", &c.window_ms, "Overlap-add window duration in milliseconds");
    p.Register("fade-ms", &c.window_ms, "Legacy alias for window-ms");
    p.Register("temperature", &c.temperature, "Sampling temperature; omitted uses model default");
    p.Register("cfg-scale", &c.cfg_scale, "Classifier-free guidance scale");
    p.Register("use-cfg", &c.use_cfg, "Enable classifier-free guidance");
    register_negated_bool(
        p, "no-cfg", &c.use_cfg, "tts.no-cfg", "Disable classifier-free guidance");
    p.Register("use-local-transformer", &c.use_local_transformer, "Enable local transformer");
    register_negated_bool(
        p, "no-local-transformer", &c.use_local_transformer, "tts.no-local-transformer",
        "Disable local transformer");
    p.Register(
        "lt-fp32", &c.lt_fp32, "Run the local transformer entirely in FP32",
        {"--tts.local-transformer-fp32"});
    p.Register("use-kv-cache", &c.use_kv_cache, "Enable decoder KV cache");
    register_negated_bool(
        p, "no-kv-cache", &c.use_kv_cache, "tts.no-kv-cache",
        "Recompute the full decoder prefix each frame");
    p.Register("use-stateful-codec", &c.use_stateful_codec, "Enable fast layer-state codec");
    register_negated_bool(
        p, "no-stateful-codec", &c.use_stateful_codec, "tts.no-stateful-codec",
        "Disable fast layer-state codec");
    p.Register("codec-cpu", &c.codec_cpu, "Force NanoCodec decoder onto CPU backend");
    p.Register("flush-partial-chunk", &c.flush_partial_chunk, "Emit a final partial codec chunk");
    register_negated_bool(
        p, "no-flush-partial", &c.flush_partial_chunk, "tts.no-flush-partial",
        "Drop final partial codec chunk");
    p.Register(
        "benchmark", &c.benchmark, "Print generation timing for the main run", {"--benchmark"});
    p.Register(
        "verbose", &c.verbose, "Print detailed generation and codec chunk logs", {"--verbose"});
    p.Register(
        "lt-backend",
        [&c](const std::string& value) {
            c.lt_backend = stream_backend_from_string("tts.lt-backend", value);
        },
        "Local-transformer backend: auto, cpu, or cuda");
    p.Register(
        "sampling-backend",
        [&c](const std::string& value) {
            c.sampling_backend = stream_backend_from_string("tts.sampling-backend", value);
        },
        "Sampling backend: auto, cpu, or cuda");
    p.Register(
        "uma-mode",
        [&c](const std::string& value) {
            c.uma_mode = stream_uma_from_string("tts.uma-mode", value);
        },
        "CUDA managed-memory mode: auto, off, or on");
    p.Register(
        "longform",
        [&c](const std::string& value) {
            c.longform_mode = stream_longform_from_string("tts.longform", value);
        },
        "Sentence-chunk longform synthesis mode: auto, off, or on");
    p.Register(
        "no-longform",
        [&c](const std::string& value) {
            c.longform_mode = common::detail::to_bool("tts.no-longform", value)
                                  ? MAGPIE_LONGFORM_OFF
                                  : MAGPIE_LONGFORM_AUTO;
        },
        "Disable sentence-chunk longform synthesis", {}, true);
}

struct MagpieStreamParamsConfig {
    magpie_stream_params& params;

    void Register(common::ParameterParser& parser) { register_stream_params(parser, params); }
};

}  // namespace

void
MagpieTtsServerConfig::Register(common::ParameterParser& parser) {
    register_runtime_config(parser, runtime);
    parser.Register(
        "tokenizer-model-dir", &tokenizer_model_dir, "Extracted Magpie .nemo directory");
    parser.Register("tn-model-dir", &tn_model_dir, "Sparrowhawk TTS TN grammar dir");
    parser.Register("tokenizer", tokenizer_config);
    parser.Register("language-code", &default_language_code, "Default Riva language code");
    parser.Register("voice-name", &default_voice_name, "Default voice name or speaker index");
    parser.Register(
        "benchmark", &benchmark, "Print tokenizer/runtime timing per request", {"--benchmark"});
    parser.Register("warmup-enabled", &warmup, "Run tokenizer/runtime warmup at startup");
    register_negated_bool(
        parser, "no-warmup", &warmup, "tts.no-warmup", "Skip startup tokenizer/runtime warmup");
    parser.Register("warmup-text", &warmup_text, "Text used for startup warmup");
    parser.Register("warmup-steps", &warmup_steps, "Decoder frames used for startup warmup");
}

void
register_magpie_stream_params(common::ParameterParser& parser, magpie_stream_params& params) {
    MagpieStreamParamsConfig config{params};
    parser.Register("tts", config);
}

}  // namespace nemo_speech::tts
