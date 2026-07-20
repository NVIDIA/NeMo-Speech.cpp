// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "grpc_nmt.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "audio_decoder.h"
#include "audio_resampler.h"
#include "langpairs.h"
#include "riva/proto/riva_audio.pb.h"

namespace nemo_speech {
namespace {

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
        asr::AsrRequestOptions::Boost boost;
        boost.boost = context.boost();
        boost.phrases.assign(context.phrases().begin(), context.phrases().end());
        options.recognition.speech_contexts.push_back(std::move(boost));
    }
    return options;
}

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
    try {
        throw;
    }
    catch (const std::invalid_argument& e) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
    }
    catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
    catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "unknown internal error");
    }
}

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

}  // namespace

GrpcNmtService::GrpcNmtService(
    std::shared_ptr<nmt::Translator> translator,
    std::shared_ptr<speech::SpeechTranslator> speech_translator)
    : translator_(std::move(translator)), speech_translator_(std::move(speech_translator)) {
    if (!translator_)
        throw std::invalid_argument("GrpcNmtService requires an NMT translator");
}

GrpcNmtService::~GrpcNmtService() = default;

grpc::Status
GrpcNmtService::TranslateText(
    grpc::ServerContext* /*ctx*/, const nr_nmt::TranslateTextRequest* req,
    nr_nmt::TranslateTextResponse* resp) {
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
}

grpc::Status
GrpcNmtService::ListSupportedLanguagePairs(
    grpc::ServerContext* /*ctx*/, const nr_nmt::AvailableLanguageRequest* /*req*/,
    nr_nmt::AvailableLanguageResponse* resp) {
    auto& pair = (*resp->mutable_languages())[translator_->model_name()];
    for (const auto& languages : nmt::langpairs::supported_pairs()) {
        pair.add_src_lang(languages.first);
        pair.add_tgt_lang(languages.second);
    }
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
        if (!speech_translator_) {
            return grpc::Status(
                grpc::StatusCode::FAILED_PRECONDITION,
                "speech-to-text translation requires an ASR recognizer");
        }
        const auto config = first.config();
        validate_audio_config(config.asr_config().config());
        const std::string request_id = first.has_id() ? first.id().value() : std::string();
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
    }
    catch (...) {
        return map_exception();
    }
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
    }
    catch (...) {
        return map_exception();
    }
}

}  // namespace nemo_speech
