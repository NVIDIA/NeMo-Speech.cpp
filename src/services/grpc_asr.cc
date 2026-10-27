// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Riva-compatible gRPC ASR adapter.
#include "grpc_asr.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "audio_decoder.h"
#include "audio_ingest.h"
#include "audio_resampler.h"
#include "grpc_status.h"
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
                  << " execution_us=" << execution_us << " cancelled=" << m.cancelled_items << "\n";
    };
    for (const auto& stage : recognizer.batch_metrics()) print(stage.first.c_str(), stage.second);
}

// Validate a RecognitionConfig against engine capabilities. Returns an empty
// Status on success, or an INVALID_ARGUMENT with a helpful message.
// Recognize decodes like the HTTP routes (audio_ingest.h): LINEAR_PCM and the G.711 encodings as
// raw samples or WAV, FLAC (and WAV/MP3) by content. OGGOPUS is not decoded by this build.
// Streaming takes raw or WAV LINEAR_PCM, MULAW and ALAW. Multi-channel audio is downmixed by the
// integer channel mean (not Riva's "first channel only").
grpc::Status
validate_config(const nr_asr::RecognitionConfig& cfg, bool streaming) {
    // Riva clients commonly leave encoding unset (ENCODING_UNSPECIFIED=0) and
    // ship s16le PCM bytes. Treat unspecified as LINEAR_PCM.
    switch (cfg.encoding()) {
        case nr_audio::ENCODING_UNSPECIFIED:
        case nr_audio::LINEAR_PCM:
        case nr_audio::MULAW:
        case nr_audio::ALAW:
            break;
        case nr_audio::FLAC:
            if (streaming) {
                return grpc::Status(
                    grpc::StatusCode::INVALID_ARGUMENT,
                    "FLAC is accepted by Recognize; StreamingRecognize takes LINEAR_PCM, MULAW or "
                    "ALAW.");
            }
            break;
        case nr_audio::OGGOPUS:
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "OGGOPUS is not supported by this build: use LINEAR_PCM, FLAC, MULAW or ALAW.");
        default:
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "Unsupported encoding: use LINEAR_PCM, FLAC, MULAW or ALAW.");
    }
    if (cfg.sample_rate_hertz() < 0 ||
        (streaming && cfg.sample_rate_hertz() != 0 &&
         !audio::supported_input_sample_rate(cfg.sample_rate_hertz()))) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            streaming
                ? "sample_rate_hertz must be between 8000 and 96000, or 0 (model rate/WAV header)."
                : "sample_rate_hertz must not be negative.");
    }
    if (cfg.audio_channel_count() < 0 || cfg.audio_channel_count() > 255) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT, "audio_channel_count must be between 0 and 255.");
    }
    return grpc::Status::OK;
}

audio::StreamEncoding
stream_encoding(const nr_asr::RecognitionConfig& cfg) {
    switch (cfg.encoding()) {
        case nr_audio::MULAW:
            return audio::StreamEncoding::Mulaw;
        case nr_audio::ALAW:
            return audio::StreamEncoding::Alaw;
        default:
            return audio::StreamEncoding::Pcm16;
    }
}

bool
is_riff_wave(const std::string& bytes) {
    return bytes.size() >= 12 && bytes.compare(0, 4, "RIFF") == 0 &&
           bytes.compare(8, 4, "WAVE") == 0;
}

// A RecognizeRequest's audio as canonical 16 kHz mono samples, decoded exactly like the HTTP
// routes: containers by content; headerless LINEAR_PCM / MULAW / ALAW with the rate and channel
// count of their RecognitionConfig (rate 0: the model rate for PCM, 8 kHz for G.711; channels 0:
// mono), decoded as the same samples in an uploaded WAV (integer downmix, then resampling to 16
// kHz).
audio::DecodedAudio
decode_recognize_audio(
    const nr_asr::RecognitionConfig& cfg, const std::string& bytes, int model_rate) {
    if (bytes.empty())
        throw std::invalid_argument("empty audio");
    const auto enc = cfg.encoding();
    const bool raw_pcm = enc == nr_audio::ENCODING_UNSPECIFIED || enc == nr_audio::LINEAR_PCM ||
                         enc == nr_audio::MULAW || enc == nr_audio::ALAW;
    if (!raw_pcm || is_riff_wave(bytes))
        return audio::decode_any(bytes.data(), bytes.size(), "audio");
    const auto encoding = enc == nr_audio::MULAW  ? audio::RawEncoding::Mulaw
                          : enc == nr_audio::ALAW ? audio::RawEncoding::Alaw
                                                  : audio::RawEncoding::Pcm16;
    const int rate = cfg.sample_rate_hertz() > 0             ? cfg.sample_rate_hertz()
                     : encoding == audio::RawEncoding::Pcm16 ? model_rate
                                                             : 8000;
    return audio::decode_raw(
        bytes.data(), bytes.size(), encoding, rate, std::max(1, cfg.audio_channel_count()));
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
        // A context without phrases boosts nothing; HTTP drops it the same way.
        if (sc.phrases().empty())
            continue;
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
// language-only WordInfo from the transcript (riva's no-offset language tagging)
// when `detected_languages`. Complete-audio backends echo the requested language
// rather than detecting one, so they keep only the alternative-level code.
void
fill_proto_alternative(
    nr_asr::SpeechRecognitionAlternative* a, const asr::Alternative& alt, bool want_offsets,
    bool continuation, bool detected_languages = true) {
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
    } else if (detected_languages) {
        add_detected_languages(a, alt.transcript, alt.language_codes);
    } else {
        for (const auto& lc : alt.language_codes) a->add_language_code(lc);
    }
}

// Client errors are returned quietly; server-side failures are also logged.
grpc::Status
request_failure(const std::string& operation) {
    auto status = grpc_status_from_exception(std::current_exception());
    if (status.error_code() == grpc::StatusCode::INTERNAL)
        std::cerr << "[grpc_asr] " << operation << " EXCEPTION: " << status.error_message() << "\n";
    return status;
}

template <class Fn>
grpc::Status
guard_streaming_call(const char* operation, Fn&& fn) {
    try {
        return fn();
    }
    catch (...) {
        return request_failure(std::string("StreamingRecognize ") + operation);
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
            auto status = validate_config(recognition, /*streaming=*/true);
            if (!status.ok())
                return status;

            interim_results_ = config.interim_results();
            options_ = extract_options(recognition);
            request_id_ = first.has_id() ? first.id().value() : std::string();
            // Without a configured rate the WAV header decides; headerless G.711 is then 8 kHz
            // telephony audio and headerless PCM takes the model rate.
            const auto encoding = stream_encoding(recognition);
            default_rate_ =
                encoding == audio::StreamEncoding::Pcm16 ? recognizer_->sample_rate() : 8000;
            decoder_ = std::make_unique<audio::Pcm16StreamDecoder>(
                recognition.sample_rate_hertz(), encoding,
                std::max(1, recognition.audio_channel_count()));
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
                decoder_->sample_rate() > 0 ? decoder_->sample_rate() : default_rate_;
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
                const int sample_rate =
                    decoder_->sample_rate() > 0 ? decoder_->sample_rate() : default_rate_;
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
        // One audio request gets one response, including any endpoint finals
        // and the following partial. Sending a separate empty acknowledgement
        // after a final breaks the official client's send/receive pairing.
        if (responses.empty())
            responses.emplace_back();
        auto& response = responses.back();
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
    }

    void append_final(
        const asr::Result& result, std::vector<nr_asr::StreamingRecognizeResponse>& responses) {
        if (responses.empty())
            responses.emplace_back();
        auto& response = responses.back();
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
    int default_rate_ = 0;
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
    grpc::ServerContext* ctx, const nr_asr::RecognizeRequest* req,
    nr_asr::RecognizeResponse* resp) {
    // The whole body is wrapped: any failure - validation, the PCM-decode
    // allocation, or inference - is request-fatal, never server-fatal. The
    // catch(...) backstop also turns a non-std-exception throw into a clean
    // gRPC status instead of std::terminate.
    try {
        auto status = validate_config(req->config(), /*streaming=*/false);
        if (!status.ok())
            return status;
        asr::AsrRequestOptions opts = extract_options(req->config());
        // A cancelled call (client gone, deadline) drops its request (speech-LLM backends).
        opts.cancelled = [ctx] { return ctx->IsCancelled(); };

        // Same decoders, resampler, downmix and limits as the HTTP routes (audio_ingest.h).
        const audio::DecodedAudio decoded =
            decode_recognize_audio(req->config(), req->audio(), recognizer_->sample_rate());
        const std::vector<float>& audio = decoded.samples;

        const auto t0 = std::chrono::high_resolution_clock::now();
        const int input_sample_rate = 16000;
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
        // Only an acoustic model detects languages; offline backends report the request's.
        const bool detected_languages = recognizer_->model() != nullptr;
        for (const auto& alt : r.alternatives) {
            fill_proto_alternative(
                result->add_alternatives(), alt, opts.needs_word_timings(),
                /*continuation=*/false, detected_languages);
        }
    }
    catch (...) {
        // invalid_argument (e.g. diarization without a diarizer) -> INVALID_ARGUMENT, a full
        // queue -> RESOURCE_EXHAUSTED, shutdown -> UNAVAILABLE, anything else -> INTERNAL.
        return request_failure("Recognize");
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
    params["streaming"] = recognizer_->supports_streaming() ? "True" : "False";
    params["type"] = recognizer_->supports_streaming() ? "online" : "offline";
    params["word_timestamps"] = recognizer_->supports_word_timestamps() ? "True" : "False";
    params["speaker_diarization"] = recognizer_->supports_speaker_diarization() ? "True" : "False";
    return grpc::Status::OK;
}

}  // namespace nemo_speech
