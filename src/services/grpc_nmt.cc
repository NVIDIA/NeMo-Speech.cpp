// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "grpc_nmt.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "audio_decoder.h"
#include "audio_ingest.h"
#include "audio_resampler.h"
#include "grpc_status.h"
#if defined(NEMO_SPEECH_BUILD_NMT)
#include "langpairs.h"
#endif
#include "riva/proto/riva_audio.pb.h"

namespace nemo_speech {
namespace {

#if defined(NEMO_SPEECH_BUILD_NMT)
speech::SpeechTranslationOptions
map_options(
    const nr_nmt::TranslationConfig& translation,
    const nvidia::riva::asr::RecognitionConfig& recognition) {
    speech::SpeechTranslationOptions options;
    options.source_language = translation.source_language_code();
    options.target_language = translation.target_language_code();
    options.recognition.language_code = recognition.language_code();
    options.recognition.max_alternatives = std::max(1, recognition.max_alternatives());
    options.recognition.enable_word_time_offsets = recognition.enable_word_time_offsets();
    options.recognition.verbatim_transcripts = recognition.verbatim_transcripts();
    options.recognition.enable_automatic_punctuation = recognition.enable_automatic_punctuation();
    options.recognition.profanity_filter = recognition.profanity_filter();
    if (recognition.has_diarization_config()) {
        options.recognition.enable_speaker_diarization =
            recognition.diarization_config().enable_speaker_diarization();
        if (recognition.diarization_config().max_speaker_count() > 0) {
            options.recognition.max_speaker_count =
                recognition.diarization_config().max_speaker_count();
        }
    }
    for (const auto& context : recognition.speech_contexts()) {
        if (context.phrases().empty())  // boosts nothing; dropped as on Recognize and HTTP
            continue;
        asr::AsrRequestOptions::Boost boost;
        boost.boost = context.boost();
        boost.phrases.assign(context.phrases().begin(), context.phrases().end());
        options.recognition.speech_contexts.push_back(std::move(boost));
    }
    return options;
}
#endif

void
validate_audio_config(const nvidia::riva::asr::RecognitionConfig& config) {
    const auto encoding = config.encoding();
    if (encoding != nvidia::riva::LINEAR_PCM && encoding != nvidia::riva::ENCODING_UNSPECIFIED) {
        throw std::invalid_argument("Only LINEAR_PCM encoding is supported.");
    }
    if (config.sample_rate_hertz() != 0 &&
        !audio::supported_input_sample_rate(config.sample_rate_hertz())) {
        throw std::invalid_argument("sample_rate_hertz must be between 8000 and 96000, or 0");
    }
    if (config.audio_channel_count() > 1)
        throw std::invalid_argument("Only mono audio is supported.");
}

grpc::Status
map_exception() {
    return grpc_status_from_exception(std::current_exception());
}

#if defined(NEMO_SPEECH_BUILD_NMT)
template <typename Request, typename ReadAudio>
grpc::Status
drive_audio(
    grpc::ServerContext* context, int configured_sample_rate, int model_sample_rate,
    ReadAudio read_audio, speech::SpeechTranslationStream& translation) {
    audio::Pcm16StreamDecoder decoder(configured_sample_rate);
    std::vector<float> samples;
    Request message;
    while (read_audio(message)) {
        samples.clear();
        decoder.process(message.audio_content(), &samples);
        if (!samples.empty()) {
            const int rate = decoder.sample_rate() > 0 ? decoder.sample_rate() : model_sample_rate;
            translation.push(samples.data(), samples.size(), rate);
        }
        if (translation.cancelled() || context->IsCancelled())
            return grpc::Status(grpc::StatusCode::CANCELLED, "client cancelled");
    }
    samples.clear();
    decoder.finish(&samples);
    if (!samples.empty()) {
        const int rate = decoder.sample_rate() > 0 ? decoder.sample_rate() : model_sample_rate;
        translation.push(samples.data(), samples.size(), rate);
    }
    translation.finish();
    return translation.cancelled()
               ? grpc::Status(grpc::StatusCode::CANCELLED, "client stopped reading")
               : grpc::Status::OK;
}
#endif

}  // namespace

GrpcNmtService::GrpcNmtService(
    std::shared_ptr<nmt::Translator> translator,
    std::shared_ptr<speech::SpeechTranslator> speech_translator,
    std::shared_ptr<asr::Recognizer> speech_llm)
    : translator_(std::move(translator)), speech_translator_(std::move(speech_translator)),
      speech_llm_(std::move(speech_llm)) {
    if (speech_llm_ && !speech_llm_->supports_translation())
        throw std::invalid_argument("GrpcNmtService: the recognizer does not translate");
    if (!translator_ && !speech_llm_)
        throw std::invalid_argument(
            "GrpcNmtService requires an NMT translator or a speech-LLM recognizer that translates");
}

GrpcNmtService::~GrpcNmtService() = default;

grpc::Status
GrpcNmtService::TranslateText(
    grpc::ServerContext* /*ctx*/, const nr_nmt::TranslateTextRequest* req,
    nr_nmt::TranslateTextResponse* resp) {
    if (!translator_)
        return grpc::Status(
            grpc::StatusCode::FAILED_PRECONDITION, "text translation requires an NMT model");
#if defined(NEMO_SPEECH_BUILD_NMT)
    if (req->texts_size() == 0)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "no texts to translate");
    try {
        const std::vector<std::string> texts(req->texts().begin(), req->texts().end());
        const auto output =
            translator_->translate(texts, req->source_language(), req->target_language());
        if (req->has_id())
            *resp->mutable_id() = req->id();
        for (const auto& value : output) {
            auto* translation = resp->add_translations();
            translation->set_text(value.text);
            translation->set_language(value.language);
        }
        return grpc::Status::OK;
    }
    catch (...) {
        return map_exception();
    }
#else
    (void)req;
    (void)resp;
    return grpc::Status(
        grpc::StatusCode::FAILED_PRECONDITION, "text translation requires an NMT build");
#endif
}

grpc::Status
GrpcNmtService::ListSupportedLanguagePairs(
    grpc::ServerContext* /*ctx*/, const nr_nmt::AvailableLanguageRequest* /*req*/,
    nr_nmt::AvailableLanguageResponse* resp) {
    if (!translator_) {  // the speech-LLM translates between the languages it knows
        auto& pair = (*resp->mutable_languages())[speech_llm_->model_name()];
        for (const auto& language : speech_llm_->supported_languages()) pair.add_src_lang(language);
        for (const auto& language : speech_llm_->translation_languages())
            pair.add_tgt_lang(language);
        return grpc::Status::OK;
    }
#if defined(NEMO_SPEECH_BUILD_NMT)
    auto& pair = (*resp->mutable_languages())[translator_->model_name()];
    for (const auto& languages : nmt::langpairs::supported_pairs()) {
        pair.add_src_lang(languages.first);
        pair.add_tgt_lang(languages.second);
    }
#endif
    return grpc::Status::OK;
}

grpc::Status
GrpcNmtService::StreamingTranslateSpeechToText(
    grpc::ServerContext* ctx, grpc::ServerReaderWriter<
                                  nr_nmt::StreamingTranslateSpeechToTextResponse,
                                  nr_nmt::StreamingTranslateSpeechToTextRequest>* stream) {
    try {
        nr_nmt::StreamingTranslateSpeechToTextRequest first;
        if (!stream->Read(&first))
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "no messages received");
        if (!first.has_config()) {
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "first message must contain a StreamingTranslateSpeechToTextConfig");
        }
        const auto config = first.config();
        validate_audio_config(config.asr_config().config());
        const std::string request_id = first.has_id() ? first.id().value() : std::string();
        if (!speech_translator_ && speech_llm_)
            return translate_with_speech_llm(ctx, stream, config, request_id);
#if defined(NEMO_SPEECH_BUILD_NMT)
        if (!speech_translator_) {
            return grpc::Status(
                grpc::StatusCode::FAILED_PRECONDITION,
                "speech-to-text translation requires an ASR recognizer");
        }
        auto options = map_options(config.translation_config(), config.asr_config().config());
        speech::SpeechTranslationCallbacks callbacks;
        callbacks.translation = [&](const speech::SpeechTranslationResult& translation) {
            nr_nmt::StreamingTranslateSpeechToTextResponse response;
            auto* result = response.add_results();
            result->set_is_final(true);
            result->add_alternatives()->set_transcript(translation.text);
            if (!request_id.empty())
                response.mutable_id()->set_value(request_id);
            return stream->Write(response);
        };
        auto translation =
            speech_translator_->streaming_translate(std::move(options), std::move(callbacks));
        auto read_audio = [&](auto& message) {
            while (stream->Read(&message)) {
                if (message.has_audio_content() && !message.audio_content().empty())
                    return true;
            }
            return false;
        };
        return drive_audio<nr_nmt::StreamingTranslateSpeechToTextRequest>(
            ctx, config.asr_config().config().sample_rate_hertz(),
            speech_translator_->sample_rate(), read_audio, *translation);
#else
        return grpc::Status(
            grpc::StatusCode::FAILED_PRECONDITION,
            "speech-to-text translation requires a speech-LLM recognizer that translates");
#endif
    }
    catch (...) {
        return map_exception();
    }
}

grpc::Status
GrpcNmtService::translate_with_speech_llm(
    grpc::ServerContext* ctx,
    grpc::ServerReaderWriter<
        nr_nmt::StreamingTranslateSpeechToTextResponse,
        nr_nmt::StreamingTranslateSpeechToTextRequest>* stream,
    const nr_nmt::StreamingTranslateSpeechToTextConfig& config, const std::string& request_id) {
    const std::string& target = config.translation_config().target_language_code();
    // The speech-LLM translates complete audio: read the whole stream first, up to the upload
    // limit.
    const auto& recognition = config.asr_config().config();
    audio::Pcm16StreamDecoder decoder(recognition.sample_rate_hertz());
    const auto stream_rate = [&] {
        return decoder.sample_rate() > 0 ? decoder.sample_rate() : speech_llm_->sample_rate();
    };
    std::vector<float> samples, chunk;
    const auto append = [&] {
        const double limit = std::floor(audio::kMaxUploadSeconds * stream_rate());
        if (static_cast<double>(samples.size() + chunk.size()) > limit)
            return false;
        samples.insert(samples.end(), chunk.begin(), chunk.end());
        return true;
    };
    const grpc::Status too_long(
        grpc::StatusCode::INVALID_ARGUMENT,
        "audio exceeds the limit of " + std::to_string(static_cast<int>(audio::kMaxUploadSeconds)) +
            " s");
    nr_nmt::StreamingTranslateSpeechToTextRequest message;
    while (stream->Read(&message)) {
        if (ctx->IsCancelled())
            return grpc::Status(grpc::StatusCode::CANCELLED, "client cancelled");
        if (message.audio_content().empty())
            continue;
        chunk.clear();
        decoder.process(message.audio_content(), &chunk);
        if (!append())
            return too_long;
    }
    chunk.clear();
    decoder.finish(&chunk);
    if (!append())
        return too_long;
    if (samples.empty())
        return grpc::Status::OK;  // no audio, no translation
    const int rate = stream_rate();
    asr::AsrRequestOptions options;
    options.language_code = config.translation_config().source_language_code().empty()
                                ? recognition.language_code()
                                : config.translation_config().source_language_code();
    options.speechlm.task = "translate";
    options.speechlm.target_language = target;  // empty = English; unknown = INVALID_ARGUMENT
    options.cancelled = [ctx] { return ctx->IsCancelled(); };
    const auto result = speech_llm_->recognize(
        samples.data(), samples.size(), options, options.language_code, rate);
    nr_nmt::StreamingTranslateSpeechToTextResponse response;
    auto* out = response.add_results();
    out->set_is_final(true);
    out->add_alternatives()->set_transcript(
        result.alternatives.empty() ? std::string() : result.alternatives.front().transcript);
    if (!request_id.empty())
        response.mutable_id()->set_value(request_id);
    if (!stream->Write(response))
        return grpc::Status(grpc::StatusCode::CANCELLED, "client stopped reading");
    return grpc::Status::OK;
}

grpc::Status
GrpcNmtService::StreamingTranslateSpeechToSpeech(
    grpc::ServerContext* ctx, grpc::ServerReaderWriter<
                                  nr_nmt::StreamingTranslateSpeechToSpeechResponse,
                                  nr_nmt::StreamingTranslateSpeechToSpeechRequest>* stream) {
    try {
        nr_nmt::StreamingTranslateSpeechToSpeechRequest first;
        if (!stream->Read(&first))
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "no messages received");
        if (!first.has_config()) {
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "first message must contain a StreamingTranslateSpeechToSpeechConfig");
        }
#if defined(NEMO_SPEECH_BUILD_NMT)
        if (!speech_translator_ || !speech_translator_->can_synthesize()) {
            return grpc::Status(
                grpc::StatusCode::FAILED_PRECONDITION,
                "speech-to-speech translation requires ASR and TTS");
        }
        const auto config = first.config();
        validate_audio_config(config.asr_config().config());
        if (config.tts_config().encoding() != nvidia::riva::LINEAR_PCM &&
            config.tts_config().encoding() != nvidia::riva::ENCODING_UNSPECIFIED) {
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "speech-to-speech output supports only LINEAR_PCM encoding");
        }
        const std::string request_id = first.has_id() ? first.id().value() : std::string();
        auto options = map_options(config.translation_config(), config.asr_config().config());
        options.synthesize_speech = true;
        options.synthesis.language_code = config.tts_config().language_code().empty()
                                              ? options.target_language
                                              : config.tts_config().language_code();
        options.synthesis.voice_name = config.tts_config().voice_name();
        options.synthesis.output_sample_rate = config.tts_config().sample_rate_hz();

        speech::SpeechTranslationCallbacks callbacks;
        callbacks.audio = [&](const auto&, const auto&, const std::string& pcm) {
            nr_nmt::StreamingTranslateSpeechToSpeechResponse response;
            response.mutable_speech()->set_audio(pcm);
            if (!request_id.empty())
                response.mutable_id()->set_value(request_id);
            return stream->Write(response);
        };
        callbacks.utterance_end = [&](const auto&) {
            nr_nmt::StreamingTranslateSpeechToSpeechResponse response;
            response.mutable_speech()->clear_audio();
            if (!request_id.empty())
                response.mutable_id()->set_value(request_id);
            return stream->Write(response);
        };
        auto translation =
            speech_translator_->streaming_translate(std::move(options), std::move(callbacks));
        auto read_audio = [&](auto& message) {
            while (stream->Read(&message)) {
                if (message.has_audio_content() && !message.audio_content().empty())
                    return true;
            }
            return false;
        };
        return drive_audio<nr_nmt::StreamingTranslateSpeechToSpeechRequest>(
            ctx, config.asr_config().config().sample_rate_hertz(),
            speech_translator_->sample_rate(), read_audio, *translation);
#else
        (void)ctx;
        return grpc::Status(
            grpc::StatusCode::FAILED_PRECONDITION,
            "speech-to-speech translation requires ASR, NMT and TTS");
#endif
    }
    catch (...) {
        return map_exception();
    }
}

}  // namespace nemo_speech
