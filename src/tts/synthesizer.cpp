// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "synthesizer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "audio_resampler.h"
#include "nemotron/native_runtime.h"
#include "tts/preproc/text_normalizer.h"

namespace nemo_speech::tts {
namespace {

std::string
lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

template <typename T>
std::string
join(const std::vector<T>& values, const char* separator) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            out << separator;
        out << values[i];
    }
    return out.str();
}

void
add_preparation_timing(MagpieSynthesisStats& stats, const PreparedSynthesis& request) {
    stats.tokenizer_ms = request.tokenizer_ms;
    const double preparation_ms = request.normalizer_ms + request.tokenizer_ms;
    stats.elapsed_s += preparation_ms / 1000.0;
    stats.rtf = stats.audio_s > 0.0 ? stats.elapsed_s / stats.audio_s : 0.0;
    stats.rtfx = stats.elapsed_s > 0.0 ? stats.audio_s / stats.elapsed_s : 0.0;
    stats.e2e_ttfa_ms += preparation_ms;
    stats.ttfa_ms = stats.e2e_ttfa_ms;
    stats.e2e_rtfx = stats.rtfx;
}

MagpieSynthesisOptions
native_options(MagpieSynthesisOptions options, const MagpieRuntimeConfig& defaults) {
    if (options.seed < 0)
        options.seed = defaults.seed < 0 ? 0 : defaults.seed;
    if (options.steps <= 0)
        options.steps = defaults.steps;
    if (options.top_k <= 0)
        options.top_k = defaults.top_k;
    if (!options.override_temperature && defaults.override_temperature) {
        options.temperature = defaults.temperature;
        options.override_temperature = true;
    }
    if (options.steps > 0 && options.steps != 512)
        throw std::invalid_argument("Nemotron requires steps=512");
    if (options.top_k > 0 && options.top_k != 80)
        throw std::invalid_argument("Nemotron requires top_k=80");
    if (options.override_temperature &&
        (!std::isfinite(options.temperature) || std::abs(options.temperature - 0.7f) > 1e-6f))
        throw std::invalid_argument("Nemotron requires temperature=0.7");
    if (options.override_cfg_scale || defaults.override_cfg_scale)
        throw std::invalid_argument("Nemotron does not support cfg_scale");
    return options;
}

bool
valid_utf8(const std::string& text) {
    size_t offset = 0;
    while (offset < text.size()) {
        const auto first = static_cast<unsigned char>(text[offset++]);
        if (first < 0x80)
            continue;
        const int continuation = first >= 0xc2 && first <= 0xdf   ? 1
                                 : first >= 0xe0 && first <= 0xef ? 2
                                 : first >= 0xf0 && first <= 0xf4 ? 3
                                                                  : 0;
        if (continuation == 0 || text.size() - offset < static_cast<size_t>(continuation))
            return false;
        const auto second = static_cast<unsigned char>(text[offset]);
        if ((first == 0xe0 && second < 0xa0) || (first == 0xed && second >= 0xa0) ||
            (first == 0xf0 && second < 0x90) || (first == 0xf4 && second >= 0x90))
            return false;
        for (int i = 0; i < continuation; ++i)
            if ((static_cast<unsigned char>(text[offset++]) & 0xc0) != 0x80)
                return false;
    }
    return true;
}

}  // namespace

struct Synthesizer::Impl {
    explicit Impl(SynthesizerConfig config)
        : backend(std::move(config.backend)), defaults(config.runtime),
          default_language_code(
              config.default_language_code.empty() ? "en-US"
                                                   : std::move(config.default_language_code)),
          default_voice_name(std::move(config.default_voice_name)),
          default_speaker(config.runtime.speaker) {
        if (backend == "nemotron") {
            native_options({}, defaults);
            const auto language = lower_ascii(default_language_code);
            if (language != "en-us" && language != "en" && language != "english")
                throw std::invalid_argument("Nemotron supports only English");
            default_language_code = "en-US";
            native = std::make_unique<nemotron::NativeRuntime>(
                config.native_socket, config.native_timeout_seconds);
            return;
        }
        if (backend != "magpie")
            throw std::invalid_argument("unknown TTS backend '" + backend + "'");
        runtime = std::make_unique<MagpieTtsRuntime>(std::move(config.runtime));
        text_normalizer =
            std::make_unique<preproc::TextNormalizer>(std::move(config.text_normalizer_model_dir));
        if (!config.tokenizer_model_dir.empty()) {
            tokenizer = std::make_unique<MagpieNativeTokenizer>(
                std::move(config.tokenizer_model_dir), config.tokenizer);
            if (tokenizer->profile_id() != runtime->tokenizer_profile() ||
                tokenizer->text_vocab_size() != runtime->text_vocab_size()) {
                throw std::invalid_argument(
                    "Magpie model/tokenizer mismatch: GGUF requires tokenizer profile '" +
                    runtime->tokenizer_profile() + "' with text vocabulary " +
                    std::to_string(runtime->text_vocab_size()) + ", but tokenizer directory is '" +
                    tokenizer->profile_id() + "' with text vocabulary " +
                    std::to_string(tokenizer->text_vocab_size()));
            }
        }
    }

    std::string backend;
    MagpieRuntimeConfig defaults;
    std::string default_language_code;
    std::string default_voice_name;
    int default_speaker = 0;
    std::unique_ptr<MagpieTtsRuntime> runtime;
    std::unique_ptr<nemotron::NativeRuntime> native;
    std::unique_ptr<MagpieNativeTokenizer> tokenizer;
    std::unique_ptr<preproc::TextNormalizer> text_normalizer;
};

Synthesizer::Synthesizer(SynthesizerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
    if (!impl_->default_voice_name.empty())
        impl_->default_speaker = resolve_speaker(impl_->default_voice_name);
    if (impl_->default_speaker < 0 || impl_->default_speaker >= speaker_count()) {
        throw std::invalid_argument(
            "default speaker " + std::to_string(impl_->default_speaker) + " must be in [0, " +
            std::to_string(speaker_count() - 1) + "]");
    }
}

Synthesizer::~Synthesizer() = default;

int
Synthesizer::resolve_speaker(const std::string& voice_name) const {
    if (voice_name.empty())
        return impl_->default_speaker;

    size_t consumed = 0;
    try {
        const int value = std::stoi(voice_name, &consumed);
        if (consumed == voice_name.size()) {
            if (value >= 0 && value < speaker_count())
                return value;
            throw std::invalid_argument("voice speaker index is outside the model speaker range");
        }
    }
    catch (const std::invalid_argument&) {
        if (consumed == voice_name.size())
            throw;
    }
    catch (const std::out_of_range&) {
        throw std::invalid_argument("voice speaker index is outside the model speaker range");
    }

    std::string wanted = lower_ascii(voice_name);
    const std::string model_prefix = lower_ascii(model_name()) + ".";
    if (wanted.rfind(model_prefix, 0) == 0)
        wanted.erase(0, model_prefix.size());
    const auto& names = speaker_names();
    for (size_t i = 0; i < names.size(); ++i) {
        if (lower_ascii(names[i]) == wanted)
            return static_cast<int>(i);
    }
    throw std::invalid_argument("unknown voice_name '" + voice_name + "'");
}

PreparedSynthesis
Synthesizer::prepare(const SynthesisRequest& request) const {
    if (request.text.empty())
        throw std::invalid_argument("text is required");
    if (impl_->native) {
        if (request.text.find_first_not_of(" \t\r\n") == std::string::npos)
            throw std::invalid_argument("text is required");
        if (request.text.size() > 1000000 || !valid_utf8(request.text))
            throw std::invalid_argument(
                "Nemotron text must be valid UTF-8 and at most 1000000 bytes");
        if (request.output_sample_rate != 0 && request.output_sample_rate != 22050)
            throw std::invalid_argument("Nemotron requires sample_rate=22050");
        if (request.apply_normalize < -1 || request.apply_normalize > 1)
            throw std::invalid_argument("apply_normalize must be -1, 0, or 1");
        const auto language = lower_ascii(
            request.language_code.empty() ? impl_->default_language_code : request.language_code);
        if (language != "en" && language != "en-us" && language != "english")
            throw std::invalid_argument("Nemotron supports only English");
        PreparedSynthesis prepared;
        prepared.metadata.original_text = request.text;
        prepared.metadata.language_code = "en-US";
        prepared.metadata.sample_rate = 22050;
        prepared.metadata.tokenizer_name = "nemotron-native";
        prepared.metadata.chunk_count = 1;
        prepared.options = native_options(request.options, impl_->defaults);
        if (prepared.options.speaker < 0)
            prepared.options.speaker = resolve_speaker(
                request.voice_name.empty() ? impl_->default_voice_name : request.voice_name);
        if (prepared.options.speaker != 0)
            throw std::invalid_argument("Nemotron supports only voice english_sample");
        prepared.metadata.speaker = prepared.options.speaker;
        prepared.apply_normalize = request.apply_normalize;
        return prepared;
    }
    if (!impl_->tokenizer) {
        throw std::invalid_argument("text synthesis requires a tokenizer model directory");
    }
    const int output_rate =
        request.output_sample_rate == 0 ? sample_rate() : request.output_sample_rate;
    if (output_rate < 8000 || output_rate > sample_rate()) {
        throw std::invalid_argument(
            "output sample rate must be between 8000 and the model rate (" +
            std::to_string(sample_rate()) + ") Hz, or be 0 (auto)");
    }

    PreparedSynthesis prepared;
    prepared.metadata.original_text = request.text;
    prepared.metadata.language_code =
        request.language_code.empty() ? impl_->default_language_code : request.language_code;
    prepared.metadata.sample_rate = output_rate;
    prepared.options = request.options;
    if (prepared.options.speaker < 0) {
        prepared.options.speaker = resolve_speaker(
            request.voice_name.empty() ? impl_->default_voice_name : request.voice_name);
    }
    if (prepared.options.speaker < 0 || prepared.options.speaker >= speaker_count())
        throw std::invalid_argument("speaker is outside the model speaker range");
    prepared.metadata.speaker = prepared.options.speaker;

    const auto tokenizer_start = std::chrono::steady_clock::now();
    const auto tokenized = impl_->tokenizer->tokenize(
        request.text, prepared.metadata.language_code,
        [&](const std::string& chunk, bool is_final) {
            const auto normalizer_start = std::chrono::steady_clock::now();
            std::string normalized =
                impl_->text_normalizer->normalize(chunk, prepared.metadata.language_code);
            prepared.normalizer_ms += std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - normalizer_start)
                                          .count();
            if (is_final) {
                normalized =
                    ensure_terminal_punctuation(normalized, prepared.metadata.language_code);
            }
            return normalized;
        });
    const double total_preparation_ms = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - tokenizer_start)
                                            .count();
    prepared.tokenizer_ms = std::max(0.0, total_preparation_ms - prepared.normalizer_ms);
    prepared.metadata.language_code = tokenized.language;
    prepared.metadata.tokenizer_name = tokenized.tokenizer_name;
    prepared.tokens = tokenized.tokens;
    prepared.token_chunks.reserve(tokenized.chunks.size());
    std::vector<std::string> processed_chunks;
    processed_chunks.reserve(tokenized.chunks.size());
    for (const auto& chunk : tokenized.chunks) {
        prepared.token_chunks.push_back(chunk.tokens);
        processed_chunks.push_back(chunk.text);
    }
    prepared.metadata.processed_text = join(processed_chunks, " ");
    if (processed_chunks.empty() && !prepared.tokens.empty())
        prepared.metadata.processed_text = join(prepared.tokens, " ");
    if (prepared.token_chunks.empty() && !prepared.tokens.empty())
        prepared.token_chunks.push_back(prepared.tokens);
    if (prepared.token_chunks.empty())
        throw std::invalid_argument("tokenizer produced no text tokens");
    prepared.metadata.token_count = prepared.tokens.size();
    prepared.metadata.chunk_count = prepared.token_chunks.size();
    return prepared;
}

SynthesisResult
Synthesizer::synthesize(const PreparedSynthesis& request, const PcmCallback& callback) {
    if (impl_->native) {
        if (!request.tokens.empty() || !request.token_chunks.empty())
            throw std::invalid_argument("Nemotron does not support pretokenized synthesis");
        SynthesisRequest validation;
        validation.text = request.metadata.original_text;
        validation.language_code = request.metadata.language_code;
        validation.output_sample_rate = request.metadata.sample_rate;
        validation.options = request.options;
        validation.apply_normalize = request.apply_normalize;
        return impl_->native->synthesize(prepare(validation), callback);
    }
    SynthesisResult result;
    result.metadata = request.metadata;
    const int output_rate = request.metadata.sample_rate;
    const bool resample = output_rate != sample_rate();
    std::unique_ptr<audio::Pcm16Resampler> resampler;
    if (resample)
        resampler = std::make_unique<audio::Pcm16Resampler>(sample_rate(), output_rate);

    auto emit = [&](const std::string& pcm) {
        result.output_samples += pcm.size() / 2;
        if (pcm.empty() || !callback)
            return true;
        if (!callback(result.metadata, pcm)) {
            result.cancelled = true;
            return false;
        }
        return true;
    };
    auto process = [&](const std::string& pcm) {
        if (!resampler)
            return emit(pcm);
        std::vector<uint8_t> converted;
        resampler->process(reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size(), &converted);
        if (converted.empty())
            return true;
        return emit(std::string(reinterpret_cast<const char*>(converted.data()), converted.size()));
    };

    result.stats = impl_->runtime->synthesize(request.token_chunks, request.options, process);
    if (!result.cancelled && resampler) {
        std::vector<uint8_t> tail;
        resampler->finish(&tail);
        if (!tail.empty()) {
            emit(std::string(reinterpret_cast<const char*>(tail.data()), tail.size()));
        }
    }
    add_preparation_timing(result.stats, request);
    result.stats.sample_rate = output_rate;
    result.stats.samples_written = result.output_samples;
    result.stats.audio_s =
        output_rate > 0 ? static_cast<double>(result.output_samples) / output_rate : 0.0;
    result.stats.rtf =
        result.stats.audio_s > 0.0 ? result.stats.elapsed_s / result.stats.audio_s : 0.0;
    result.stats.rtfx =
        result.stats.elapsed_s > 0.0 ? result.stats.audio_s / result.stats.elapsed_s : 0.0;
    result.stats.e2e_rtfx = result.stats.rtfx;
    return result;
}

SynthesisResult
Synthesizer::synthesize(const SynthesisRequest& request, const PcmCallback& callback) {
    return synthesize(prepare(request), callback);
}

SynthesisResult
Synthesizer::synthesize_tokens(
    const std::vector<int32_t>& tokens, const MagpieSynthesisOptions& options,
    int output_sample_rate, const PcmCallback& callback) {
    if (impl_->native)
        throw std::invalid_argument("Nemotron does not support pretokenized synthesis");
    if (tokens.empty())
        throw std::invalid_argument("text token list is empty");
    const int output_rate = output_sample_rate == 0 ? sample_rate() : output_sample_rate;
    if (output_rate < 8000 || output_rate > sample_rate()) {
        throw std::invalid_argument(
            "output sample rate must be between 8000 and the model rate (" +
            std::to_string(sample_rate()) + ") Hz, or be 0 (auto)");
    }
    PreparedSynthesis prepared;
    prepared.tokens = tokens;
    prepared.token_chunks.push_back(tokens);
    prepared.options = options;
    if (prepared.options.speaker < 0)
        prepared.options.speaker = impl_->default_speaker;
    if (prepared.options.speaker < 0 || prepared.options.speaker >= speaker_count())
        throw std::invalid_argument("speaker is outside the model speaker range");
    prepared.metadata.processed_text = join(tokens, " ");
    prepared.metadata.tokenizer_name = "pretokenized";
    prepared.metadata.speaker = prepared.options.speaker;
    prepared.metadata.sample_rate = output_rate;
    prepared.metadata.token_count = tokens.size();
    prepared.metadata.chunk_count = 1;
    return synthesize(prepared, callback);
}

SynthesisResult
Synthesizer::warmup(const std::string& text, int steps) {
    if (text.empty())
        return {};
    if (impl_->native && impl_->native->batch_size() > 1)
        return {};
    SynthesisRequest request;
    request.text = text;
    request.language_code = impl_->default_language_code;
    request.options.speaker = impl_->default_speaker;
    request.options.steps = impl_->native ? -1 : steps;
    return synthesize(request);
}

int
Synthesizer::sample_rate() const {
    return impl_->native ? 22050 : impl_->runtime->sample_rate();
}

int
Synthesizer::speaker_count() const {
    return impl_->native ? static_cast<int>(impl_->native->speaker_names().size())
                         : impl_->runtime->speaker_count();
}

const std::vector<std::string>&
Synthesizer::speaker_names() const {
    return impl_->native ? impl_->native->speaker_names() : impl_->runtime->speaker_names();
}

std::vector<std::string>
Synthesizer::supported_language_codes() const {
    if (impl_->native)
        return {"en-US"};
    return impl_->tokenizer ? impl_->tokenizer->supported_language_codes()
                            : std::vector<std::string>{};
}

const std::string&
Synthesizer::model_name() const {
    return impl_->native ? impl_->native->model_name() : impl_->runtime->model_name();
}

const std::string&
Synthesizer::default_language_code() const {
    return impl_->default_language_code;
}

int
Synthesizer::default_speaker() const {
    return impl_->default_speaker;
}

bool
Synthesizer::text_normalization_enabled() const {
    return impl_->native ? impl_->native->text_normalization_enabled()
                         : impl_->text_normalizer->enabled();
}

const std::string&
Synthesizer::backend_name() const {
    return impl_->backend;
}

int
Synthesizer::backend_batch_size() const {
    return impl_->native ? impl_->native->batch_size() : 1;
}

const std::string&
Synthesizer::backend_mode() const {
    static const std::string serial = "serial";
    return impl_->native ? impl_->native->mode() : serial;
}

}  // namespace nemo_speech::tts
