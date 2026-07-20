// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Riva-compatible gRPC ASR adapter.
#include "grpc_asr.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "audio_decoder.h"
#include "audio_resampler.h"
#include "types.h"  // AsrRequestOptions, Result

namespace nemo_speech {

namespace {

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

}  // namespace

// ---------------------------------------------------------------------------
// GrpcAsrService - thin transport adapter over asr::Recognizer
// ---------------------------------------------------------------------------
GrpcAsrService::GrpcAsrService(std::shared_ptr<asr::Recognizer> recognizer)
    : recognizer_(std::move(recognizer)) {
    if (!recognizer_)
        throw std::invalid_argument("GrpcAsrService requires an ASR recognizer");
}

GrpcAsrService::~GrpcAsrService() = default;

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
    // ----- 1. Read first message; must be streaming_config. -----
    nr_asr::StreamingRecognizeRequest first;
    if (!stream->Read(&first)) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "no messages received");
    }
    if (!first.has_streaming_config()) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT, "first message must contain streaming_config");
    }
    const auto& scfg = first.streaming_config();
    const auto& rcfg = scfg.config();
    auto v = validate_config(rcfg);
    if (!v.ok())
        return v;
    const bool interim_results = scfg.interim_results();
    const asr::AsrRequestOptions opts = extract_options(rcfg);
    // Echoed on every response when the client supplied one (riva contract).
    const std::string request_id = first.has_id() ? first.id().value() : std::string();

    // We emit exactly one StreamingRecognizeResponse per input audio chunk so
    // `riva_streaming_asr_client --simulate_realtime` can match request/
    // response 1:1 for per-chunk latency stats.
    //
    // During warm-up (before chunk_sec + right_pad_sec of audio is buffered)
    // and on chunks that produce no new tokens, we send an EMPTY response
    // (no `results` entries). The Riva clients gate printing on
    // `response.results_size() > 0`, so empty responses are silent on the
    // wire but still count for recv-side latency bookkeeping.

    std::string last_emitted_transcript;
    // True once a final has been emitted on this stream. Subsequent streaming
    // results (interims + finals) get a leading space so the riva client's
    // cumulative display joins utterance segments with a separator (matches
    // riva; our detokenizer trims each segment's own leading space).
    bool any_final = false;
    // Flipped when the client vanishes (failed Write); stops decode work.
    bool write_failed = false;
    auto send = [&](nr_asr::StreamingRecognizeResponse& resp) {
        // Once a write fails the client is gone: stop attempting writes so a
        // trailing final left in drain_finals isn't re-emitted as an interim,
        // and no further work writes to a dead stream.
        if (write_failed)
            return;
        if (!request_id.empty())
            resp.mutable_id()->set_value(request_id);
        if (!stream->Write(resp))
            write_failed = true;
    };

    try {
        // ----- 2. Per-stream recognition (CTC: buffered, RNNT: cache-aware). -----
        auto strm = recognizer_->streaming_recognize(opts, rcfg.language_code());

        // Emit an interim (non-final) result for the trailing partial. Dedups on
        // unchanged text and respects interim_results; always sends a response
        // (empty if nothing to report) to keep the 1:1 chunk/response contract.
        auto write_partial = [&](const std::optional<asr::Result>& r) {
            nr_asr::StreamingRecognizeResponse resp;
            if (r && interim_results) {
                const std::string& t = r->alternatives.front().transcript;
                if (!t.empty() && t != last_emitted_transcript) {
                    auto* result = resp.add_results();
                    result->set_is_final(false);
                    result->set_stability(0.0f);
                    result->set_audio_processed(r->audio_processed);
                    result->set_channel_tag(r->channel_tag);
                    auto* alt = result->add_alternatives();
                    alt->set_transcript(join_continuation(any_final, t));
                    alt->set_confidence(0.0f);  // WordInfo is final-only (riva)
                    last_emitted_transcript = t;
                }
            }
            send(resp);
        };

        // Send a FINAL result (mid-stream EOU or end-of-stream). The library
        // already post-processed it; this just maps to proto. One stream can
        // carry multiple finals (riva's multi-utterance behaviour).
        auto emit_final = [&](const asr::Result& r) {
            nr_asr::StreamingRecognizeResponse resp;
            auto* result = resp.add_results();
            result->set_is_final(true);
            result->set_stability(r.stability);
            result->set_audio_processed(r.audio_processed);
            result->set_channel_tag(r.channel_tag);
            // All alternatives (N-best); the library already capped at max_alternatives.
            for (const auto& alt : r.alternatives) {
                fill_proto_alternative(
                    result->add_alternatives(), alt, opts.needs_word_timings(),
                    /*continuation=*/any_final);
            }
            send(resp);
            last_emitted_transcript.clear();  // next utterance's partials start fresh
            any_final = true;
        };

        // Drive decoding for the audio buffered so far: emit each final, then
        // return the trailing interim for the caller to emit. Empty-transcript
        // finals (a VAD blip that decoded to nothing) are swallowed, matching the
        // end-of-stream guard below. Stops the moment a write fails: each step
        // holds the global GGML compute mutex, so decoding for a vanished client
        // delays active streams.
        auto drain_finals = [&]() -> std::optional<asr::Result> {
            auto r = strm->next();
            while (r && r->is_final && !write_failed) {
                if (!r->alternatives.front().transcript.empty())
                    emit_final(*r);
                r = strm->next();
            }
            return r;
        };

        // ----- 3. Loop: read audio_content, advance runner, write partials. -----
        nr_asr::StreamingRecognizeRequest in;
        std::vector<float> audio_chunk;
        audio::Pcm16StreamDecoder decoder(rcfg.sample_rate_hertz());
        while (!write_failed && stream->Read(&in)) {
            // riva force_eou: per-message runtime_config latches an immediate
            // EOU, honored on the next runner poll.
            const auto& rtc = in.runtime_config();
            const auto rc_it = rtc.find("force_eou");
            const bool forced = (rc_it != rtc.end() && rc_it->second == "true");
            if (forced)
                strm->force_endpoint();

            // Messages with no (or empty) audio still honor a latched force_eou.
            if (!in.has_audio_content() || in.audio_content().empty()) {
                if (forced)
                    drain_finals();  // fire the forced EOU without new audio
                continue;
            }
            audio_chunk.clear();
            decoder.process(in.audio_content(), &audio_chunk);
            if (audio_chunk.empty())
                continue;
            const int input_sample_rate =
                decoder.sample_rate() > 0 ? decoder.sample_rate() : recognizer_->sample_rate();
            strm->push(audio_chunk.data(), audio_chunk.size(), input_sample_rate);

            write_partial(drain_finals());
        }

        // Skip the tail decode for a vanished client: finalize() flushes the
        // whole buffered window through the encoder (+ PnC) under the compute
        // mutex.
        if (write_failed || ctx->IsCancelled()) {
            return grpc::Status::CANCELLED;
        }

        // Complete raw-header detection and reject a trailing half sample.
        audio_chunk.clear();
        decoder.finish(&audio_chunk);
        if (!audio_chunk.empty()) {
            const int input_sample_rate =
                decoder.sample_rate() > 0 ? decoder.sample_rate() : recognizer_->sample_rate();
            strm->push(audio_chunk.data(), audio_chunk.size(), input_sample_rate);
            drain_finals();
        }

        // ----- 4. End of audio: finalize, then emit the end-of-stream final.
        // Skip it only when endpointing already finalized every utterance and
        // the tail is empty, to avoid a trailing empty final.
        const asr::Result final_r = strm->finish();
        if (!final_r.alternatives.front().transcript.empty() || !any_final) {
            emit_final(final_r);
        }
        // A failed final write means the client is gone: report cancellation
        // rather than OK so the RPC status reflects reality.
        if (write_failed) {
            return grpc::Status::CANCELLED;
        }
    }
    catch (const std::invalid_argument& e) {
        // Client error (e.g. diarization requested without a loaded diarizer).
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    }
    catch (const std::exception& e) {
        std::cerr << "[grpc_asr] EXCEPTION: " << e.what() << "\n";
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
    catch (...) {
        std::cerr << "[grpc_asr] EXCEPTION: unknown (non-std)\n";
        return grpc::Status(grpc::StatusCode::INTERNAL, "unknown internal error");
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
