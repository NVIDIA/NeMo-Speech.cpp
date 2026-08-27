// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "runtime.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "magpietts.h"
#include "token_utils.h"

namespace nemo_speech::tts {
namespace {

magpietts_backend_preference
to_internal(MagpieBackendPreference backend) {
    switch (backend) {
        case MagpieBackendPreference::Auto:
            return MAGPIETTS_BACKEND_AUTO;
        case MagpieBackendPreference::Cpu:
            return MAGPIETTS_BACKEND_CPU;
        case MagpieBackendPreference::Cuda:
            return MAGPIETTS_BACKEND_CUDA;
    }
    return MAGPIETTS_BACKEND_AUTO;
}

magpietts_uma_mode
to_internal(MagpieUmaMode mode) {
    switch (mode) {
        case MagpieUmaMode::Auto:
            return MAGPIETTS_UMA_AUTO;
        case MagpieUmaMode::Off:
            return MAGPIETTS_UMA_OFF;
        case MagpieUmaMode::On:
            return MAGPIETTS_UMA_ON;
    }
    return MAGPIETTS_UMA_AUTO;
}

magpie_longform_mode
to_internal(MagpieLongformMode mode) {
    switch (mode) {
        case MagpieLongformMode::Auto:
            return MAGPIE_LONGFORM_AUTO;
        case MagpieLongformMode::Off:
            return MAGPIE_LONGFORM_OFF;
        case MagpieLongformMode::On:
            return MAGPIE_LONGFORM_ON;
    }
    return MAGPIE_LONGFORM_AUTO;
}

std::vector<std::string>
default_speaker_names(int count) {
    static const char* names[] = {"John", "Sofia", "Aria", "Jason", "Leo"};
    std::vector<std::string> out;
    out.reserve(std::max(count, 0));
    for (int i = 0; i < count; ++i) {
        if (i < (int)(sizeof(names) / sizeof(names[0]))) {
            out.emplace_back(names[i]);
        } else {
            out.emplace_back("speaker-" + std::to_string(i));
        }
    }
    return out;
}

void
validate_config(const MagpieRuntimeConfig& config) {
    if (config.magpie_model.empty()) {
        throw std::invalid_argument("Magpie model path is required");
    }
    if (config.codec_model.empty()) {
        throw std::invalid_argument("NanoCodec model path is required");
    }
    if (config.threads <= 0) {
        throw std::invalid_argument("threads must be positive");
    }
    if (config.codec_threads < 0) {
        throw std::invalid_argument("codec_threads must be non-negative");
    }
    if (config.chunk_frames <= 0) {
        throw std::invalid_argument("chunk_frames must be positive");
    }
    if (config.codec_queue_depth <= 0) {
        throw std::invalid_argument("codec_queue_depth must be positive");
    }
    if (config.codec_history_frames < -1 || config.codec_future_frames < 0) {
        throw std::invalid_argument("codec history/future frame settings are invalid");
    }
    if (config.window_ms < 0) {
        throw std::invalid_argument("window_ms must be non-negative");
    }
}

}  // namespace

class MagpieTtsRuntime::Impl {
   public:
    explicit Impl(MagpieRuntimeConfig config) : config_(std::move(config)) {
        validate_config(config_);
        ggml_time_init();

        stream_ = std::make_unique<MagpieStreamingRuntime>();
        if (!stream_->load(
                config_.magpie_model, config_.codec_model, to_internal(config_.uma_mode),
                config_.magpie_cpu, config_.codec_cpu, config_.verbose)) {
            throw std::runtime_error("failed to load MagpieTTS/NanoCodec GGUFs");
        }
        tokenizer_profile_ = stream_->tokenizerProfile();
        text_vocab_size_ = stream_->textVocabSize();

        speaker_names_ = stream_->speakerNames();
        if (speaker_names_.empty()) {
            speaker_names_ = default_speaker_names(stream_->speakerCount());
        }
        model_name_ = "magpietts";
    }

    ~Impl() = default;

    int sample_rate() const { return stream_->sampleRate(); }
    int speaker_count() const { return stream_->speakerCount(); }
    const std::vector<std::string>& speaker_names() const { return speaker_names_; }
    const std::string& model_name() const { return model_name_; }
    const std::string& tokenizer_profile() const { return tokenizer_profile_; }
    int text_vocab_size() const { return text_vocab_size_; }

    MagpieSynthesisStats synthesize(
        const std::vector<int32_t>& tokens, const MagpieSynthesisOptions& options,
        const MagpieTtsRuntime::PcmCallback& pcm_callback) {
        if (tokens.empty()) {
            throw std::invalid_argument("text token list is empty");
        }
        return synthesize_impl({tokens}, tokens, options, pcm_callback);
    }

    MagpieSynthesisStats synthesize(
        const std::vector<std::vector<int32_t>>& token_chunks,
        const MagpieSynthesisOptions& options, const MagpieTtsRuntime::PcmCallback& pcm_callback) {
        if (token_chunks.empty()) {
            throw std::invalid_argument("text token chunk list is empty");
        }
        for (const auto& chunk : token_chunks) {
            if (chunk.empty()) {
                throw std::invalid_argument("text token chunk list contains an empty chunk");
            }
        }
        return synthesize_impl(
            token_chunks, flatten_token_chunks(token_chunks), options, pcm_callback);
    }

    MagpieSynthesisStats synthesize_impl(
        const std::vector<std::vector<int32_t>>& token_chunks, const std::vector<int32_t>& tokens,
        const MagpieSynthesisOptions& options, const MagpieTtsRuntime::PcmCallback& pcm_callback) {
        std::lock_guard<std::mutex> lock(mutex_);

        magpie_stream_params params;
        params.magpie_model = config_.magpie_model;
        params.codec_model = config_.codec_model;
        params.tokens = tokens;
        params.token_chunks = token_chunks;
        params.warmup_tokens = tokens;
        params.warmup_token_chunks = token_chunks;
        params.speaker = options.speaker >= 0 ? options.speaker : config_.speaker;
        params.threads = config_.threads;
        params.codec_threads = config_.codec_threads > 0 ? config_.codec_threads : config_.threads;
        params.seed = options.seed >= 0 ? options.seed : config_.seed;
        params.steps = options.steps > 0 ? options.steps : config_.steps;
        params.top_k = options.top_k > 0 ? options.top_k : config_.top_k;
        params.chunk_frames = config_.chunk_frames;
        params.codec_queue_depth = config_.codec_queue_depth;
        params.codec_history_frames = config_.codec_history_frames;
        params.codec_future_frames = config_.codec_future_frames;
        params.window_ms = config_.window_ms;
        params.temperature = options.override_temperature
                                 ? options.temperature
                                 : (config_.override_temperature ? config_.temperature : NAN);
        params.cfg_scale = options.override_cfg_scale
                               ? options.cfg_scale
                               : (config_.override_cfg_scale ? config_.cfg_scale : NAN);
        params.use_cfg = config_.use_cfg;
        params.use_local_transformer = config_.use_local_transformer;
        params.lt_fp32 = config_.lt_fp32;
        params.use_kv_cache = config_.use_kv_cache;
        params.use_stateful_codec = config_.use_stateful_codec;
        params.codec_cpu = config_.codec_cpu;
        params.flush_partial_chunk = config_.flush_partial_chunk;
        params.verbose = config_.verbose;
        params.lt_backend = to_internal(config_.lt_backend);
        params.sampling_backend = to_internal(config_.sampling_backend);
        params.uma_mode = to_internal(config_.uma_mode);
        params.longform_mode = to_internal(config_.longform_mode);

        stream_run_metrics metrics;
        if (!stream_->synthesize(
                params, token_chunks,
                [&](const std::vector<uint8_t>& bytes) {
                    if (!pcm_callback || bytes.empty()) {
                        return true;
                    }
                    return pcm_callback(
                        std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
                },
                metrics, "riva_tts", false)) {
            throw std::runtime_error("MagpieTTS synthesis failed");
        }

        MagpieSynthesisStats stats;
        stats.sample_rate = stream_->sampleRate();
        stats.generated_frames = metrics.generated_frames;
        stats.chunks = metrics.chunks;
        stats.e2e_chunks = metrics.e2e_chunks;
        stats.samples_written = metrics.samples_written;
        stats.tokenizer_ms = metrics.tokenizer_ms;
        stats.encoder_ms = metrics.encoder_ms;
        stats.audio_s = metrics.audio_s;
        stats.elapsed_s = metrics.e2e_elapsed_s;
        stats.rtf = metrics.e2e_rtf;
        stats.rtfx = metrics.e2e_rtfx;
        stats.ttfa_ms = metrics.e2e.first_event_ms;
        stats.icl_avg_ms = metrics.e2e.inter_event_avg_ms();
        stats.icl_min_ms = metrics.e2e.inter_event_min_value_ms();
        stats.icl_max_ms = metrics.e2e.inter_event_max_ms;
        stats.decoder_audio_s = metrics.decoder_audio_s;
        stats.decoder_elapsed_s = metrics.decoder.elapsed_s;
        stats.decoder_rtfx = metrics.decoder_rtfx;
        stats.decoder_ttft_ms = metrics.decoder.first_event_ms;
        stats.decoder_itl_avg_ms = metrics.decoder.inter_event_avg_ms();
        stats.decoder_itl_min_ms = metrics.decoder.inter_event_min_value_ms();
        stats.decoder_itl_max_ms = metrics.decoder.inter_event_max_ms;
        stats.decoder_itl_p95_ms = metrics.decoder.inter_event_p95_ms();
        stats.decoder_itl_p99_ms = metrics.decoder.inter_event_p99_ms();
        stats.codec_audio_s = metrics.codec_audio_s;
        stats.codec_elapsed_s = metrics.codec_elapsed_s;
        stats.codec_rtfx = metrics.codec_rtfx;
        stats.codec_ttfa_ms = metrics.codec.first_event_ms;
        stats.codec_icl_avg_ms = metrics.codec.inter_event_avg_ms();
        stats.codec_icl_min_ms = metrics.codec.inter_event_min_value_ms();
        stats.codec_icl_max_ms = metrics.codec.inter_event_max_ms;
        stats.codec_icl_p95_ms = metrics.codec.inter_event_p95_ms();
        stats.codec_icl_p99_ms = metrics.codec.inter_event_p99_ms();
        stats.e2e_ttfa_ms = metrics.e2e.first_event_ms;
        stats.e2e_icl_avg_ms = metrics.e2e.inter_event_avg_ms();
        stats.e2e_icl_min_ms = metrics.e2e.inter_event_min_value_ms();
        stats.e2e_icl_max_ms = metrics.e2e.inter_event_max_ms;
        stats.e2e_icl_p95_ms = metrics.e2e.inter_event_p95_ms();
        stats.e2e_icl_p99_ms = metrics.e2e.inter_event_p99_ms();
        stats.e2e_rtfx = metrics.e2e_rtfx;
        return stats;
    }

   private:
    MagpieRuntimeConfig config_;
    std::unique_ptr<MagpieStreamingRuntime> stream_;
    std::vector<std::string> speaker_names_;
    std::string model_name_;
    std::string tokenizer_profile_;
    int text_vocab_size_ = 0;
    std::mutex mutex_;
};

MagpieTtsRuntime::MagpieTtsRuntime(MagpieRuntimeConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

MagpieTtsRuntime::~MagpieTtsRuntime() = default;

int
MagpieTtsRuntime::sample_rate() const {
    return impl_->sample_rate();
}

int
MagpieTtsRuntime::speaker_count() const {
    return impl_->speaker_count();
}

const std::vector<std::string>&
MagpieTtsRuntime::speaker_names() const {
    return impl_->speaker_names();
}

const std::string&
MagpieTtsRuntime::model_name() const {
    return impl_->model_name();
}

const std::string&
MagpieTtsRuntime::tokenizer_profile() const {
    return impl_->tokenizer_profile();
}

int
MagpieTtsRuntime::text_vocab_size() const {
    return impl_->text_vocab_size();
}

MagpieSynthesisStats
MagpieTtsRuntime::synthesize(
    const std::vector<int32_t>& tokens, const MagpieSynthesisOptions& options,
    const PcmCallback& pcm_callback) {
    return impl_->synthesize(tokens, options, pcm_callback);
}

MagpieSynthesisStats
MagpieTtsRuntime::synthesize(
    const std::vector<std::vector<int32_t>>& token_chunks, const MagpieSynthesisOptions& options,
    const PcmCallback& pcm_callback) {
    return impl_->synthesize(token_chunks, options, pcm_callback);
}

}  // namespace nemo_speech::tts
