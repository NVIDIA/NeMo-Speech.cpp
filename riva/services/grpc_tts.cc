// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "grpc_tts.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "riva/proto/riva_audio.pb.h"

namespace nemo_speech {
namespace {

namespace nr_audio = nvidia::riva;

grpc::Status
invalid_arg(const std::string& message) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, message);
}

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

std::string
json_escape(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0x0f];
                } else {
                    out += (char)c;
                }
                break;
        }
    }
    return out;
}

std::vector<std::string>
dotted_voice_names(const std::string& model_name, const std::vector<std::string>& speakers) {
    std::vector<std::string> voices;
    voices.reserve(speakers.size());
    for (const auto& speaker : speakers) {
        voices.push_back(model_name + "." + speaker);
    }
    return voices;
}

std::string
json_string_array(const std::vector<std::string>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out += ",";
        }
        out += "\"";
        out += json_escape(values[i]);
        out += "\"";
    }
    out += "]";
    return out;
}

std::string
voices_by_language_json(
    const std::vector<std::string>& language_codes, const std::vector<std::string>& voices) {
    std::string out = "{";
    const std::string rendered_voices = json_string_array(voices);
    for (size_t i = 0; i < language_codes.size(); ++i) {
        if (i) {
            out += ",";
        }
        out += "\"";
        out += json_escape(language_codes[i]);
        out += "\":{\"voices\":";
        out += rendered_voices;
        out += "}";
    }
    out += "}";
    return out;
}

int
parse_int(const std::string& key, const std::string& value) {
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    }
    catch (const std::exception&) {
        throw std::invalid_argument("custom_configuration '" + key + "' must be an integer");
    }
    if (consumed != value.size())
        throw std::invalid_argument("custom_configuration '" + key + "' must be an integer");
    return parsed;
}

float
parse_float(const std::string& key, const std::string& value) {
    size_t consumed = 0;
    float parsed = 0.0f;
    try {
        parsed = std::stof(value, &consumed);
    }
    catch (const std::exception&) {
        throw std::invalid_argument("custom_configuration '" + key + "' must be a float");
    }
    if (consumed != value.size())
        throw std::invalid_argument("custom_configuration '" + key + "' must be a float");
    return parsed;
}

tts::SynthesisRequest
map_request(const nr_tts::SynthesizeSpeechRequest& req) {
    const auto encoding = req.encoding();
    if (encoding != nr_audio::LINEAR_PCM && encoding != nr_audio::ENCODING_UNSPECIFIED)
        throw std::invalid_argument("Only LINEAR_PCM encoding is supported.");
    if (req.has_zero_shot_data())
        throw std::invalid_argument("zero_shot_data is not supported by MagpieTTS.");
    if (!req.custom_dictionary().empty())
        throw std::invalid_argument("custom_dictionary is not supported by MagpieTTS.");

    tts::SynthesisRequest out;
    out.text = req.text();
    out.language_code = req.language_code();
    out.voice_name = req.voice_name();
    out.output_sample_rate = req.sample_rate_hz();
    for (const auto& entry : req.custom_configuration()) {
        const std::string key = lower_ascii(entry.first);
        const std::string& value = entry.second;
        if (key == "speaker") {
            out.options.speaker = parse_int(entry.first, value);
        } else if (key == "seed") {
            out.options.seed = parse_int(entry.first, value);
        } else if (key == "steps" || key == "max_decoder_steps") {
            out.options.steps = parse_int(entry.first, value);
        } else if (key == "top_k" || key == "top-k") {
            out.options.top_k = parse_int(entry.first, value);
        } else if (key == "temperature") {
            out.options.temperature = parse_float(entry.first, value);
            out.options.override_temperature = true;
        } else if (key == "cfg_scale" || key == "cfg-scale") {
            out.options.cfg_scale = parse_float(entry.first, value);
            out.options.override_cfg_scale = true;
        } else {
            throw std::invalid_argument("Unsupported custom_configuration key: " + entry.first);
        }
    }
    return out;
}

void
fill_response_metadata(
    nr_tts::SynthesizeSpeechResponse& response, const nvidia::riva::RequestId& id,
    const tts::SynthesisMetadata& metadata) {
    *response.mutable_id() = id;
    response.mutable_meta()->set_text(metadata.original_text);
    response.mutable_meta()->set_processed_text(metadata.processed_text);
}

grpc::Status
map_exception() {
    try {
        throw;
    }
    catch (const std::invalid_argument& e) {
        return invalid_arg(e.what());
    }
    catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

void
log_benchmark(
    bool enabled, const char* mode, const nr_tts::SynthesizeSpeechRequest& request,
    const tts::SynthesisResult& result) {
    if (!enabled)
        return;
    const std::string id = request.id().value().empty() ? "-" : request.id().value();
    const auto& metadata = result.metadata;
    const auto& stats = result.stats;
    std::cerr << "[riva_tts][benchmark]"
              << " mode=" << mode << " id=" << id << " text_chars=" << metadata.original_text.size()
              << " processed_text_chars=" << metadata.processed_text.size()
              << " text_tokens=" << metadata.token_count << " text_chunks=" << metadata.chunk_count
              << " speaker=" << metadata.speaker << " frames=" << stats.generated_frames
              << " samples=" << stats.samples_written << " audio_s=" << stats.audio_s
              << " output_sample_rate=" << metadata.sample_rate << " elapsed_s=" << stats.elapsed_s
              << " tokenizer_ms=" << stats.tokenizer_ms << " encoder_ms=" << stats.encoder_ms
              << " decoder_ttft_ms=" << stats.decoder_ttft_ms
              << " decoder_rtfx=" << stats.decoder_rtfx << " codec_ttfa_ms=" << stats.codec_ttfa_ms
              << " codec_rtfx=" << stats.codec_rtfx << " e2e_ttfa_ms=" << stats.e2e_ttfa_ms
              << " e2e_rtfx=" << stats.e2e_rtfx << " codec_chunks=" << stats.chunks
              << " e2e_chunks=" << stats.e2e_chunks << "\n";
}

}  // namespace

GrpcTtsService::GrpcTtsService(std::shared_ptr<tts::Synthesizer> synthesizer, bool benchmark)
    : synthesizer_(std::move(synthesizer)), benchmark_(benchmark) {
    if (!synthesizer_)
        throw std::invalid_argument("GrpcTtsService requires a TTS synthesizer");
}

GrpcTtsService::~GrpcTtsService() = default;

grpc::Status
GrpcTtsService::Synthesize(
    grpc::ServerContext* ctx, const nr_tts::SynthesizeSpeechRequest* req,
    nr_tts::SynthesizeSpeechResponse* resp) {
    try {
        std::string audio;
        auto result =
            synthesizer_->synthesize(map_request(*req), [&](const auto&, const std::string& pcm) {
                if (ctx->IsCancelled())
                    return false;
                audio.append(pcm);
                return true;
            });
        if (ctx->IsCancelled())
            return grpc::Status(grpc::StatusCode::CANCELLED, "client cancelled");
        resp->set_audio(std::move(audio));
        fill_response_metadata(*resp, req->id(), result.metadata);
        log_benchmark(benchmark_, "unary", *req, result);
        return grpc::Status::OK;
    }
    catch (...) {
        return map_exception();
    }
}

grpc::Status
GrpcTtsService::SynthesizeOnline(
    grpc::ServerContext* ctx,
    grpc::ServerReaderWriter<nr_tts::SynthesizeSpeechResponse, nr_tts::SynthesizeSpeechRequest>*
        stream) {
    nr_tts::SynthesizeSpeechRequest req;
    while (stream->Read(&req)) {
        try {
            bool write_failed = false;
            auto result = synthesizer_->synthesize(
                map_request(req),
                [&](const tts::SynthesisMetadata& metadata, const std::string& pcm) {
                    if (ctx->IsCancelled())
                        return false;
                    nr_tts::SynthesizeSpeechResponse resp;
                    resp.set_audio(pcm);
                    fill_response_metadata(resp, req.id(), metadata);
                    if (!stream->Write(resp)) {
                        write_failed = true;
                        return false;
                    }
                    return true;
                });
            if (write_failed || ctx->IsCancelled())
                return grpc::Status(grpc::StatusCode::CANCELLED, "client stopped reading");
            log_benchmark(benchmark_, "streaming", req, result);
        }
        catch (...) {
            return map_exception();
        }
    }
    return grpc::Status::OK;
}

grpc::Status
GrpcTtsService::GetRivaSynthesisConfig(
    grpc::ServerContext* /*ctx*/, const nr_tts::RivaSynthesisConfigRequest* req,
    nr_tts::RivaSynthesisConfigResponse* resp) {
    try {
        if (!req->model_name().empty() && req->model_name() != synthesizer_->model_name()) {
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND, "unknown TTS model '" + req->model_name() + "'");
        }

        const std::vector<std::string> language_codes = tts::supported_language_codes();
        const std::vector<std::string> speaker_names = synthesizer_->speaker_names();
        const std::vector<std::string> dotted_voices =
            dotted_voice_names(synthesizer_->model_name(), speaker_names);
        const std::string subvoices = join(speaker_names, ",");
        const std::string voices_by_language =
            voices_by_language_json(language_codes, dotted_voices);
        for (const auto& language_code : language_codes) {
            auto* mc = resp->add_model_config();
            mc->set_model_name(synthesizer_->model_name());
            auto& params = *mc->mutable_parameters();
            params["language_code"] = language_code;
            params["sample_rate_hz"] = std::to_string(synthesizer_->sample_rate());
            params["encoding"] = "LINEAR_PCM";
            params["tokenizer"] = "native";
            params["voice_name"] = synthesizer_->model_name();
            params["subvoices"] = subvoices;
            params["voices"] = subvoices;
            params["voices_by_language"] = voices_by_language;
        }
        return grpc::Status::OK;
    }
    catch (...) {
        return map_exception();
    }
}

}  // namespace nemo_speech
