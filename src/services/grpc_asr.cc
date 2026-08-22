// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Riva-compatible gRPC ASR adapter.
#include "grpc_asr.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "audio_decoder.h"
#include "audio_resampler.h"
#include "model.h"
#include "types.h"  // AsrRequestOptions, Result

namespace nemo_speech {

namespace {

void
dump_batch_metrics(const asr::Recognizer& recognizer, const char* label) {
    if (std::getenv("NEMO_SPEECH_BATCH_METRICS") == nullptr)
        return;
    const auto print = [label](const char* stage, const asr::BatchMetrics& m) {
        const double mean = m.batches > 0 ? static_cast<double>(m.items) / m.batches : 0.0;
        const double requested =
            m.batches > 0 ? static_cast<double>(m.requested_items) / m.batches : 0.0;
        const double wait_us =
            m.batches > 0 ? static_cast<double>(m.queue_wait_ns) / m.batches / 1000.0 : 0.0;
        const double ready = m.batches > 0 ? static_cast<double>(m.ready_items) / m.batches : 0.0;
        const double compatible =
            m.batches > 0 ? static_cast<double>(m.compatible_items) / m.batches : 0.0;
        const double execution_us =
            m.batches > 0 ? static_cast<double>(m.execution_ns) / m.batches / 1000.0 : 0.0;
        std::cerr << "[grpc batch metrics] " << label << " stage=" << stage
                  << " batches=" << m.batches << " items=" << m.items << " mean=" << mean
                  << " singleton=" << m.singleton_batches << " max=" << m.max_observed_batch
                  << " target_mean=" << requested << " target_hit=" << m.target_reached_batches
                  << " full=" << m.capacity_batches << " ready_set=" << m.ready_set_batches
                  << " deadline=" << m.deadline_batches << " ready_mean=" << ready
                  << " compatible_mean=" << compatible << " wait_us=" << wait_us
                  << " execution_us=" << execution_us << "\n";
    };
    const auto* model = recognizer.model();
    print("frontend", model->fe().batch_metrics());
    if (model->head_kind() == asr::HeadKind::Ctc) {
        print("encoder", static_cast<const asr::CtcModel*>(model)->batch_metrics());
    } else {
        const auto* rnnt = static_cast<const asr::RnntModel*>(model);
        print("encoder", rnnt->encoder_batch_metrics());
        print("predictor", rnnt->predictor_batch_metrics());
        print("joint", rnnt->joint_batch_metrics());
    }
}

// Validate a RecognitionConfig against engine capabilities. Returns an empty
// Status on success, or an INVALID_ARGUMENT with a helpful message.
grpc::Status
validate_config(const nr_asr::RecognitionConfig& cfg) {
    // Riva clients commonly leave encoding unset (ENCODING_UNSPECIFIED=0) and
    // ship s16le PCM bytes. Treat unspecified as LINEAR_PCM.
    const auto enc = cfg.encoding();
    if (enc != nr_audio::LINEAR_PCM && enc != nr_audio::ENCODING_UNSPECIFIED) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT, "Only LINEAR_PCM encoding is supported.");
    }
    if (cfg.sample_rate_hertz() != 0 &&
        !audio::supported_input_sample_rate(cfg.sample_rate_hertz())) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            "sample_rate_hertz must be between 8000 and 96000, or 0 (model rate/WAV header).");
    }
    if (cfg.audio_channel_count() > 1) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Only mono audio supported.");
    }
    return grpc::Status::OK;
}

// Surface detected language(s) on the response: on
// SpeechRecognitionAlternative.language_code plus per-word
// WordInfo.language_code (matching Riva). No-op when none were detected.
void
add_detected_languages(
    nr_asr::SpeechRecognitionAlternative* alt, const std::string& transcript,
    const std::vector<std::string>& langs) {
    if (langs.empty())
        return;
    for (const auto& lang : langs) {
        alt->add_language_code(lang);
    }
    std::istringstream iss(transcript);
    std::string w;
    while (iss >> w) {
        auto* wi = alt->add_words();
        wi->set_word(w);
        wi->set_language_code(langs.front());
    }
}

// Mirror per-request RecognitionConfig knobs into the runner-facing options.
asr::AsrRequestOptions
extract_options(const nr_asr::RecognitionConfig& cfg) {
    asr::AsrRequestOptions o;
    // max_alternatives: proto default 0 means "unset" -> 1-best. N-best beyond 1
    // only takes effect once a beam decoder implements it (see Decoder).
    o.max_alternatives = std::max(1, cfg.max_alternatives());
    o.enable_word_time_offsets = cfg.enable_word_time_offsets();
    o.verbatim_transcripts = cfg.verbatim_transcripts();
    o.enable_automatic_punctuation = cfg.enable_automatic_punctuation();
    o.profanity_filter = cfg.profanity_filter();
    if (cfg.has_diarization_config()) {
        o.enable_speaker_diarization = cfg.diarization_config().enable_speaker_diarization();
        if (cfg.diarization_config().max_speaker_count() > 0)
            o.max_speaker_count = cfg.diarization_config().max_speaker_count();
    }
    for (const auto& sc : cfg.speech_contexts()) {
        asr::AsrRequestOptions::Boost b;
        b.boost = sc.boost();
        for (const auto& p : sc.phrases()) b.phrases.push_back(p);
        o.speech_contexts.push_back(std::move(b));
    }
    // Per-request EOU threshold. The riva proto carries it in
    // RecognitionConfig.endpointing_config.stop_history_eou (ms); accept a
    // custom_configuration["stop_history_eou"] string as an alias.
    if (cfg.has_endpointing_config() && cfg.endpointing_config().has_stop_history_eou()) {
        o.stop_history_eou_ms = static_cast<float>(cfg.endpointing_config().stop_history_eou());
    } else {
        const auto& cc = cfg.custom_configuration();
        auto cc_it = cc.find("stop_history_eou");
        if (cc_it != cc.end()) {
            try {
                o.stop_history_eou_ms = std::stof(cc_it->second);
            }
            catch (const std::exception&) {
                std::cerr << "[grpc_asr] ignoring non-numeric custom_configuration"
                             "[stop_history_eou]\n";
            }
        }
    }
    return o;
}

// riva clients join utterance segments in their cumulative display with the
// leading space the server puts on every result after the first final. Single
// source for both interim and final emission.
std::string
join_continuation(bool any_final, const std::string& text) {
    return (any_final && !text.empty()) ? " " + text : text;
}

// Map a library Alternative (already post-processed; word times in ms) onto the
// proto. `continuation` adds the riva leading space that joins utterance
// segments. When word offsets were requested the library populated alt.words
// (with per-word language tags) and alt.language_codes; otherwise we synthesize
// language-only WordInfo from the transcript (riva's no-offset language tagging).
void
fill_proto_alternative(
    nr_asr::SpeechRecognitionAlternative* a, const asr::Alternative& alt, bool want_offsets,
    bool continuation) {
    a->set_transcript(join_continuation(continuation, alt.transcript));
    a->set_confidence(alt.confidence);
    if (want_offsets) {
        for (const auto& w : alt.words) {
            auto* wi = a->add_words();
            wi->set_word(w.word);
            wi->set_start_time(w.start_time);
            wi->set_end_time(w.end_time);
            wi->set_confidence(w.confidence);
            if (w.speaker_tag > 0)
                wi->set_speaker_tag(w.speaker_tag);
            if (!w.language_code.empty())
                wi->set_language_code(w.language_code);
        }
        for (const auto& lc : alt.language_codes) a->add_language_code(lc);
    } else {
        add_detected_languages(a, alt.transcript, alt.language_codes);
    }
}

template <class Fn>
grpc::Status
guard_streaming_call(const char* operation, Fn&& fn) {
    try {
        return fn();
    }
    catch (const std::invalid_argument& e) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    }
    catch (const std::exception& e) {
        std::cerr << "[grpc_asr] StreamingRecognize " << operation << " EXCEPTION: " << e.what()
                  << "\n";
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
    catch (...) {
        std::cerr << "[grpc_asr] StreamingRecognize " << operation
                  << " EXCEPTION: unknown (non-std)\n";
        return grpc::Status(grpc::StatusCode::INTERNAL, "unknown internal error");
    }
}

// Own one RPC's decoder, recognition stream, response state, and temporary
// buffers. Model work completes before the outer service loop performs network
// writes, and cancellation releases state without flushing an unusable tail.
class StreamingAsrSession {
   public:
    explicit StreamingAsrSession(asr::Recognizer* recognizer) : recognizer_(recognizer) {}

    grpc::Status start(const nr_asr::StreamingRecognizeRequest& first) {
        return guard_streaming_call("start", [&]() -> grpc::Status {
            if (started_) {
                return grpc::Status(
                    grpc::StatusCode::INVALID_ARGUMENT, "streaming_config was already received");
            }
            if (!first.has_streaming_config()) {
                return grpc::Status(
                    grpc::StatusCode::INVALID_ARGUMENT,
                    "first message must contain streaming_config");
            }
            const auto& config = first.streaming_config();
            const auto& recognition = config.config();
            auto status = validate_config(recognition);
            if (!status.ok())
                return status;

            interim_results_ = config.interim_results();
            options_ = extract_options(recognition);
            request_id_ = first.has_id() ? first.id().value() : std::string();
            decoder_ = std::make_unique<audio::Pcm16StreamDecoder>(recognition.sample_rate_hertz());
            stream_ = recognizer_->streaming_recognize(
                options_, recognition.language_code(), /*coordinate_ingress=*/true);
            started_ = true;
            return grpc::Status::OK;
        });
    }

    grpc::Status process(
        const nr_asr::StreamingRecognizeRequest& request,
        std::vector<nr_asr::StreamingRecognizeResponse>& responses) {
        return guard_streaming_call("process", [&]() -> grpc::Status {
            responses.clear();
            if (!started_ || !stream_) {
                return grpc::Status(
                    grpc::StatusCode::FAILED_PRECONDITION, "streaming_config was not received");
            }

            const auto& runtime = request.runtime_config();
            const auto force_it = runtime.find("force_eou");
            const bool forced = force_it != runtime.end() && force_it->second == "true";
            if (forced)
                stream_->force_endpoint();
            if (!request.has_audio_content() || request.audio_content().empty()) {
                if (forced)
                    (void)drain_finals(responses);
                return grpc::Status::OK;
            }

            audio_chunk_.clear();
            decoder_->process(request.audio_content(), &audio_chunk_);
            if (audio_chunk_.empty())
                return grpc::Status::OK;
            const int sample_rate =
                decoder_->sample_rate() > 0 ? decoder_->sample_rate() : recognizer_->sample_rate();
            stream_->push(audio_chunk_.data(), audio_chunk_.size(), sample_rate);
            append_partial(drain_finals(responses), responses);
            return grpc::Status::OK;
        });
    }

    grpc::Status finish(std::vector<nr_asr::StreamingRecognizeResponse>& responses) {
        return guard_streaming_call("finish", [&]() -> grpc::Status {
            responses.clear();
            if (!started_ || !stream_) {
                return grpc::Status(
                    grpc::StatusCode::FAILED_PRECONDITION, "streaming_config was not received");
            }

            audio_chunk_.clear();
            decoder_->finish(&audio_chunk_);
            if (!audio_chunk_.empty()) {
                const int sample_rate = decoder_->sample_rate() > 0 ? decoder_->sample_rate()
                                                                    : recognizer_->sample_rate();
                stream_->push(audio_chunk_.data(), audio_chunk_.size(), sample_rate);
                (void)drain_finals(responses);
            }
            const asr::Result result = stream_->finish();
            stream_.reset();
            if (!result.alternatives.front().transcript.empty() || !any_final_)
                append_final(result, responses);
            return grpc::Status::OK;
        });
    }

    void cancel() { stream_.reset(); }

   private:
    void add_request_id(nr_asr::StreamingRecognizeResponse& response) const {
        if (!request_id_.empty())
            response.mutable_id()->set_value(request_id_);
    }

    void append_partial(
        const std::optional<asr::Result>& result,
        std::vector<nr_asr::StreamingRecognizeResponse>& responses) {
        nr_asr::StreamingRecognizeResponse response;
        if (result && interim_results_) {
            const std::string& transcript = result->alternatives.front().transcript;
            if (!transcript.empty() && transcript != last_emitted_transcript_) {
                auto* proto_result = response.add_results();
                proto_result->set_is_final(false);
                proto_result->set_stability(0.0f);
                proto_result->set_audio_processed(result->audio_processed);
                proto_result->set_channel_tag(result->channel_tag);
                auto* alternative = proto_result->add_alternatives();
                alternative->set_transcript(join_continuation(any_final_, transcript));
                alternative->set_confidence(0.0f);
                last_emitted_transcript_ = transcript;
            }
        }
        add_request_id(response);
        responses.push_back(std::move(response));
    }

    void append_final(
        const asr::Result& result, std::vector<nr_asr::StreamingRecognizeResponse>& responses) {
        nr_asr::StreamingRecognizeResponse response;
        auto* proto_result = response.add_results();
        proto_result->set_is_final(true);
        proto_result->set_stability(result.stability);
        proto_result->set_audio_processed(result.audio_processed);
        proto_result->set_channel_tag(result.channel_tag);
        for (const auto& alternative : result.alternatives) {
            fill_proto_alternative(
                proto_result->add_alternatives(), alternative, options_.needs_word_timings(),
                any_final_);
        }
        add_request_id(response);
        responses.push_back(std::move(response));
        last_emitted_transcript_.clear();
        any_final_ = true;
    }

    std::optional<asr::Result> drain_finals(
        std::vector<nr_asr::StreamingRecognizeResponse>& responses) {
        auto result = stream_->next();
        while (result && result->is_final) {
            if (!result->alternatives.front().transcript.empty())
                append_final(*result, responses);
            result = stream_->next();
        }
        return result;
    }

    asr::Recognizer* recognizer_ = nullptr;
    std::unique_ptr<asr::RecognitionStream> stream_;
    std::unique_ptr<audio::Pcm16StreamDecoder> decoder_;
    asr::AsrRequestOptions options_;
    std::string request_id_;
    std::string last_emitted_transcript_;
    std::vector<float> audio_chunk_;
    bool started_ = false;
    bool interim_results_ = false;
    bool any_final_ = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// GrpcAsrService - thin transport adapter over asr::Recognizer
// ---------------------------------------------------------------------------
GrpcAsrService::GrpcAsrService(std::shared_ptr<asr::Recognizer> recognizer)
    : recognizer_(std::move(recognizer)) {
    if (!recognizer_)
        throw std::invalid_argument("GrpcAsrService requires an ASR recognizer");
}

GrpcAsrService::~GrpcAsrService() {
    dump_batch_metrics(*recognizer_, "shutdown");
}

grpc::Status
GrpcAsrService::Recognize(
    grpc::ServerContext* /*ctx*/, const nr_asr::RecognizeRequest* req,
    nr_asr::RecognizeResponse* resp) {
    // The whole body is wrapped: any failure - validation, the PCM-decode
    // allocation, or inference - is request-fatal, never server-fatal. The
    // catch(...) backstop also turns a non-std-exception throw into a clean
    // gRPC status instead of std::terminate.
    try {
        auto status = validate_config(req->config());
        if (!status.ok())
            return status;
        const asr::AsrRequestOptions opts = extract_options(req->config());

        audio::Pcm16StreamDecoder decoder(req->config().sample_rate_hertz());
        std::vector<float> audio;
        decoder.process(req->audio(), &audio);
        decoder.finish(&audio);
        if (audio.empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "empty audio");
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        const int input_sample_rate =
            decoder.sample_rate() > 0 ? decoder.sample_rate() : recognizer_->sample_rate();
        const asr::Result r = recognizer_->recognize(
            audio.data(), audio.size(), opts, req->config().language_code(), input_sample_rate);
        const auto t1 = std::chrono::high_resolution_clock::now();

        const float audio_sec =
            static_cast<float>(audio.size()) / static_cast<float>(input_sample_rate);
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "[grpc_asr] Recognize: " << audio_sec << "s audio in " << ms << " ms (RTF "
                  << (ms / 1000.0 / audio_sec) << ")\n";

        if (req->has_id())
            *resp->mutable_id() = req->id();
        auto* result = resp->add_results();
        result->set_audio_processed(r.audio_processed);
        result->set_channel_tag(r.channel_tag);
        // All alternatives (N-best); the library already capped at max_alternatives.
        for (const auto& alt : r.alternatives) {
            fill_proto_alternative(
                result->add_alternatives(), alt, opts.needs_word_timings(),
                /*continuation=*/false);
        }
    }
    catch (const std::invalid_argument& e) {
        // Client error (e.g. diarization requested without a loaded diarizer).
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    }
    catch (const std::exception& e) {
        std::cerr << "[grpc_asr] Recognize EXCEPTION: " << e.what() << "\n";
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
    catch (...) {
        std::cerr << "[grpc_asr] Recognize EXCEPTION: unknown (non-std)\n";
        return grpc::Status(grpc::StatusCode::INTERNAL, "unknown internal error");
    }
    return grpc::Status::OK;
}

grpc::Status
GrpcAsrService::StreamingRecognize(
    grpc::ServerContext* ctx,
    grpc::ServerReaderWriter<nr_asr::StreamingRecognizeResponse, nr_asr::StreamingRecognizeRequest>*
        stream) {
    nr_asr::StreamingRecognizeRequest request;
    if (!stream->Read(&request)) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "no messages received");
    }

    StreamingAsrSession session(recognizer_.get());
    auto status = session.start(request);
    if (!status.ok())
        return status;

    std::vector<nr_asr::StreamingRecognizeResponse> responses;
    while (stream->Read(&request)) {
        if (ctx->IsCancelled()) {
            session.cancel();
            return grpc::Status::CANCELLED;
        }
        status = session.process(request, responses);
        if (!status.ok())
            return status;
        for (auto& response : responses) {
            if (!stream->Write(response)) {
                session.cancel();
                return grpc::Status::CANCELLED;
            }
        }
    }

    if (ctx->IsCancelled()) {
        session.cancel();
        return grpc::Status::CANCELLED;
    }
    status = session.finish(responses);
    if (!status.ok())
        return status;
    for (auto& response : responses) {
        if (!stream->Write(response)) {
            session.cancel();
            return grpc::Status::CANCELLED;
        }
    }
    return grpc::Status::OK;
}

grpc::Status
GrpcAsrService::GetRivaSpeechRecognitionConfig(
    grpc::ServerContext* /*ctx*/, const nr_asr::RivaSpeechRecognitionConfigRequest* req,
    nr_asr::RivaSpeechRecognitionConfigResponse* resp) {
    // riva semantics: an empty model_name lists everything; a name filters.
    if (!req->model_name().empty() && req->model_name() != recognizer_->model_name()) {
        return grpc::Status::OK;  // no matching model -> empty response
    }
    auto* mc = resp->add_model_config();
    mc->set_model_name(recognizer_->model_name());
    auto& params = *mc->mutable_parameters();
    // Keys follow riva's model_registry convention so riva tooling can match.
    std::string langs;
    for (const auto& language : recognizer_->supported_languages()) {
        if (!langs.empty())
            langs += ",";
        langs += language;
    }
    params["language_code"] = langs;
    params["sample_rate_hertz"] = std::to_string(recognizer_->sample_rate());
    params["streaming"] = "True";
    params["type"] = "online";
    return grpc::Status::OK;
}

}  // namespace nemo_speech
