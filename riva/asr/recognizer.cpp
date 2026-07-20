// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "recognizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "model.h"
#include "silero_vad.h"
#ifdef NEMO_SPEECH_WITH_FLASHLIGHT
#include "flashlight_decoder.h"
#endif

namespace nemo_speech::asr {

namespace {

std::unique_ptr<ggml_runtime::BackendManager>
make_backend(int gpu_idx) {
    ggml_runtime::Params p;
    p.use_gpu = (gpu_idx >= 0);
    p.gpu_device_idx = std::max(gpu_idx, 0);
    p.pe_bin_path = const_cast<char*>("");
    return std::make_unique<ggml_runtime::BackendManager>(p);
}

// The model self-punctuates if its vocab carries sentence-terminator tokens
// (a plain-text CTC vocab is lowercase letters + space, no '.'/'?'). Used to
// skip the PnC BERT, which would double the model's own punctuation.
bool
vocab_self_punctuates(const std::vector<std::string>& vocab) {
    static constexpr std::string_view kSentencePiecePrefix = "\xE2\x96\x81";
    static constexpr std::array<std::string_view, 5> kTerminators = {
        ".", "?", "!", "\xE0\xA5\xA4", "\xE0\xA5\xA5"};
    for (const auto& p : vocab) {
        std::string_view token = p;
        if (token.substr(0, kSentencePiecePrefix.size()) == kSentencePiecePrefix)
            token.remove_prefix(kSentencePiecePrefix.size());
        if (std::find(kTerminators.begin(), kTerminators.end(), token) != kTerminators.end())
            return true;
    }
    return false;
}

}  // namespace

Recognizer::Recognizer(RecognizerConfig cfg)
    : bm_(make_backend(cfg.backend.gpu)), cfg_(std::move(cfg)), ingress_batches_(cfg_.batching) {
    if (cfg_.streaming.chunk_size <= 0.0f || cfg_.streaming.ctc_left_padding < 0.0f ||
        cfg_.streaming.ctc_right_padding < 0.0f)
        throw std::invalid_argument(
            "invalid streaming geometry: chunk_size must be > 0 and paddings must be >= 0");
    model_ = AsrModel::load(*bm_, cfg_.model.path, cfg_.batching);
    model_name_ = cfg_.model.name.empty() ? model_->model_name() : cfg_.model.name;
    // Self-punctuating models must bypass the separate PnC model.
    postproc_ = std::make_unique<postproc::Postprocessor>(
        cfg_.postproc, bm_.get(), vocab_self_punctuates(model_->vocab()), cfg_.batching);
    const bool need_vad =
        !cfg_.vad.model_path.empty() &&
        (cfg_.vad.masker.mask_enable || (cfg_.endpointing.enable && cfg_.endpointing.vad_based));
    if (need_vad) {
        vad_model_ = std::make_shared<SileroVadModel>(*bm_, cfg_.vad.model_path, cfg_.batching);
        std::cerr << "[recognizer] VAD weights/session loaded once and shared across streams\n";
    }
#ifdef NEMO_SPEECH_WITH_FLASHLIGHT
    // Share the loaded language model and lexicon trie across streams.
    if (model_->head_kind() == HeadKind::Ctc &&
        cfg_.decoder.kind == DecoderConfig::Kind::Flashlight) {
        const auto* ctc =
            static_cast<const CtcModel*>(model_.get());  // head_kind() guarantees CtcModel
        FlashlightCtcCfg fcfg;
        fcfg.lm_path = cfg_.decoder.lm_path;
        fcfg.lexicon_path = cfg_.decoder.lexicon_path;
        fcfg.tokenizer_path = cfg_.decoder.tokenizer_path;
        flashlight_resources_ =
            std::make_shared<const FlashlightResources>(ctc->ctc_config(), ctc->vocab(), fcfg);
        std::cerr << "[recognizer] flashlight resources loaded (lm + lexicon trie, shared)\n";
    }
#endif
    std::cerr << "[recognizer] streaming cfg: chunk=" << cfg_.streaming.chunk_size
              << "s left=" << cfg_.streaming.ctc_left_padding
              << "s right=" << cfg_.streaming.ctc_right_padding << "s\n";

    if (!cfg_.diar.model_path.empty()) {
        diar_model_ = std::make_unique<DiarModel>(*bm_, cfg_.diar.model_path);
        const DiarGeometry geo = cfg_.diar.resolved_geometry();
        std::cerr << "[recognizer] diarizer loaded: " << cfg_.diar.model_path
                  << " (spkcache=" << geo.spkcache_len << " fifo=" << geo.fifo_len
                  << " chunk=" << geo.chunk_len << " rc=" << geo.chunk_right_context << ")\n";
    }
}

Recognizer::~Recognizer() = default;

std::unique_ptr<AsrRunner>
Recognizer::make_runner() const {
    // head_kind() is the authoritative runner discriminator.
    if (model_->head_kind() != HeadKind::Ctc) {
        auto* transducer = static_cast<RnntModel*>(model_.get());
        if (!transducer->supports_cache_streaming())
            throw std::runtime_error(
                "this transducer encoder is offline-only and cannot serve StreamingRecognize");
        return std::make_unique<CacheStreamRunner>(transducer, cfg_, vad_model_);
    }
    return std::make_unique<BufferedStreamRunner>(
        static_cast<CtcModel*>(model_.get()), cfg_, flashlight_resources_, vad_model_);
}

void
Recognizer::warmup() {
    // Push ~4 s of silence through a real runner so the lazy per-shape graph
    // build (and, for RNNT, the cache-aware Session setup) happens now. 0.1 s
    // chunks: enough for the buffered runner to emit one full window and the
    // cache-aware runner to advance several chunks; finalize() flushes the tail.
    std::unique_ptr<AsrRunner> runner;
    if (model_->head_kind() != HeadKind::Ctc &&
        !static_cast<RnntModel*>(model_.get())->supports_cache_streaming())
        runner = std::make_unique<OfflineRunner>(model_.get(), cfg_, flashlight_resources_);
    else
        runner = make_runner();
    if (model_->has_prompt()) {
        runner->set_prompt_index(model_->prompt_index_for_lang("auto"));
    }
    const int sr = model_->sample_rate();
    const float warmup_sec = std::max(
        4.0f, cfg_.streaming.ctc_left_padding + cfg_.streaming.chunk_size +
                  cfg_.streaming.ctc_right_padding);
    const int iters = std::max(1, static_cast<int>(warmup_sec * 10.0f + 0.999f));
    const size_t chunk_samples = std::max<size_t>(1, static_cast<size_t>(sr) / 10);
    std::vector<float> chunk(chunk_samples, 0.0f);
    for (int i = 0; i < iters; ++i) {
        runner->feed_audio(chunk.data(), chunk.size());
        runner->step();
    }
    runner->finalize();

    // Warm the diarizer Session's graph shapes too: the streaming warmup
    // ladder (growing fifo), the first spkcache compression (~19 s in), and
    // the steady-state fifo cycle all key distinct cached graphs.
    if (diar_model_) {
        DiarStream ds(*diar_model_, cfg_.diar.resolved_geometry());
        std::vector<float> dchunk(static_cast<size_t>(sr), 0.0f);  // 1 s of silence
        for (int i = 0; i < 24; ++i) ds.feed_audio(dchunk.data(), dchunk.size());
        ds.finish();
    }
}

RecognitionStream::RecognitionStream(
    Recognizer* recognizer, std::unique_ptr<AsrRunner> runner, AsrRequestOptions opts,
    bool coordinate_ingress)
    : recognizer_(recognizer), runner_(std::move(runner)), opts_(std::move(opts)),
      coordinate_ingress_(coordinate_ingress) {
    runner_->set_request_options(opts_);
    if (opts_.enable_speaker_diarization) {
        if (recognizer_->diar_model() == nullptr) {
            throw std::invalid_argument(
                "speaker diarization requested but no diarizer model is loaded "
                "(start the server with --diar-model)");
        }
        diar_ = std::make_unique<DiarStream>(
            *recognizer_->diar_model(), recognizer_->config().diar.resolved_geometry());
    }
}

void
RecognitionStream::push(const float* samples, size_t n, int sample_rate) {
    if (n == 0)
        return;
    if (samples == nullptr)
        throw std::invalid_argument("audio samples must not be null");
    if (sample_rate < 0)
        throw std::invalid_argument("input sample rate must be 0 or positive");
    const int rate = sample_rate > 0 ? sample_rate : recognizer_->sample_rate();
    if (!audio::supported_input_sample_rate(rate))
        throw std::invalid_argument("input sample rate must be between 8000 and 96000 Hz");
    if (input_sample_rate_ == 0) {
        input_sample_rate_ = rate;
        if (rate != recognizer_->sample_rate())
            resampler_ = std::make_unique<audio::AudioResampler>(rate, recognizer_->sample_rate());
    } else if (rate != input_sample_rate_) {
        throw std::invalid_argument("input sample rate cannot change within a stream");
    }
    if (coordinate_ingress_)
        pending_cohort_target_ = recognizer_->arrive_streaming_ingress();
    // The diarizer consumes the same model-rate audio the runner does, so it
    // is fed after resampling, mirroring both paths below.
    if (!resampler_) {
        runner_->feed_audio(samples, n);
        if (diar_)
            diar_->feed_audio(samples, n);
        return;
    }
    resampled_audio_.clear();
    resampler_->process(samples, n, &resampled_audio_);
    if (!resampled_audio_.empty()) {
        runner_->feed_audio(resampled_audio_.data(), resampled_audio_.size());
        if (diar_)
            diar_->feed_audio(resampled_audio_.data(), resampled_audio_.size());
    }
}

void
RecognitionStream::force_endpoint() {
    runner_->force_eou();
}

Result
RecognitionStream::build_result_(const StreamingUpdate& u, bool is_final) const {
    Result r;
    r.is_final = is_final;
    r.stability = is_final ? 1.0f : 0.0f;
    r.channel_tag = 1;
    r.audio_processed = u.audio_processed_sec;

    if (!is_final) {
        // Interim results contain only the raw top hypothesis.
        Alternative alt;
        alt.transcript = u.transcript_so_far;
        alt.confidence = 0.0f;
        r.alternatives.push_back(std::move(alt));
        return r;
    }

    // Detected languages are per-stream (shared by every alternative).
    const auto langs = runner_->detected_languages();
    const bool want_off = opts_.needs_word_timings();
    const double ms = recognizer_->ms_per_enc_frame();
    const std::string lang0 = langs.empty() ? std::string() : langs.front();
    std::string postproc_language = opts_.language_code;
    std::string normalized_requested = postproc_language;
    std::transform(
        normalized_requested.begin(), normalized_requested.end(), normalized_requested.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if ((normalized_requested.empty() || normalized_requested == "auto") && !lang0.empty())
        postproc_language = lang0;

    // The prompt-conditioned multilingual model emits a "<lang>" token (e.g.
    // <en-US>). The runner strips it from the transcript text and records it in
    // detected_languages(), but the decoder's per-word list still carries it as
    // a word - drop those entries so the language id can't leak into word time
    // offsets. Empty for non-multilingual models (no-op).
    std::vector<std::string> tag_words;
    tag_words.reserve(langs.size());
    for (const auto& l : langs) tag_words.push_back("<" + l + ">");

    // Build one Alternative from a decoder hypothesis. Postproc (PnC/ITN/
    // profanity) may remap word spans, so it runs before frame->ms conversion;
    // word offsets are produced only when requested. Speaker tags: riva
    // semantics - mean diarizer frame probability over the word's span,
    // argmax, 1-based; only the top alternative is tagged.
    auto make_alt = [&](const std::string& transcript, float confidence,
                        const std::vector<WordTiming>& words_in, bool tag_speakers) {
        Alternative alt;
        std::vector<WordTiming> words;
        words.reserve(words_in.size());
        for (const auto& w : words_in)
            if (std::find(tag_words.begin(), tag_words.end(), w.word) == tag_words.end())
                words.push_back(w);
        alt.transcript = recognizer_->postproc().apply(
            transcript, opts_, want_off ? &words : nullptr, postproc_language);
        alt.confidence = confidence;
        alt.language_codes.assign(langs.begin(), langs.end());
        if (want_off) {
            for (const auto& w : words) {
                Word ww;
                ww.word = w.word;
                ww.start_time = static_cast<int32_t>(w.start_frame * ms + 0.5);
                ww.end_time = static_cast<int32_t>(w.end_frame * ms + 0.5);
                ww.confidence = w.confidence;
                ww.language_code = lang0;
                if (tag_speakers && diar_) {
                    const int spk =
                        diar_->speaker_for_time(ww.start_time / 1000.0, ww.end_time / 1000.0);
                    ww.speaker_tag = spk >= 0 ? spk + 1 : 0;
                }
                alt.words.push_back(std::move(ww));
            }
        }
        return alt;
    };

    // The scalar update fields carry the top hypothesis; append the N-best tail.
    const int cap = std::max(1, opts_.max_alternatives);
    r.alternatives.push_back(
        make_alt(u.transcript_so_far, u.confidence, u.words, /*tag_speakers=*/true));
    for (const auto& h : u.extra_alternatives) {
        if (static_cast<int>(r.alternatives.size()) >= cap)
            break;
        r.alternatives.push_back(make_alt(h.transcript, h.confidence, h.words, false));
    }
    return r;
}

void
RecognitionStream::flush_diar_deficit_(const StreamingUpdate& u) {
    if (!diar_ || u.words.empty())
        return;
    const double end_sec = u.words.back().end_frame * recognizer_->ms_per_enc_frame() / 1000.0;
    const auto target = static_cast<int64_t>(std::ceil(end_sec / diar_->seconds_per_frame()));
    if (target > diar_->n_frames()) {
        const int64_t before = diar_->n_frames();
        diar_->flush_available(target);
        if (std::getenv("NEMO_SPEECH_TIMING")) {
            std::cerr << "[diar] on-demand flush at final: frontier " << before << " -> "
                      << diar_->n_frames() << " frames (target " << target << ")\n";
        }
    }
}

std::optional<Result>
RecognitionStream::next() {
    const ScopedBatchCohort cohort_scope(pending_cohort_target_);
    auto u = runner_->step();
    if (u.is_final) {
        flush_diar_deficit_(u);
        last_processed_sec_ = std::max(last_processed_sec_, u.audio_processed_sec);
        return build_result_(u, /*is_final=*/true);
    }
    // Deliver an interim only when this step advanced the audio cursor (decoded
    // at least one new window). Once the buffered audio is consumed the cursor
    // stops, next() returns nullopt, and a `while (next())` drain terminates -
    // honoring the contract that nullopt means "need more audio". Interim dedup
    // (suppressing unchanged transcripts on the wire) stays in the transport.
    const bool advanced = u.audio_processed_sec > last_processed_sec_;
    last_processed_sec_ = std::max(last_processed_sec_, u.audio_processed_sec);
    if (advanced && !u.transcript_so_far.empty()) {
        pending_cohort_target_ = 0;
        return build_result_(u, /*is_final=*/false);
    }
    pending_cohort_target_ = 0;
    return std::nullopt;
}

Result
RecognitionStream::finish() {
    const ScopedBatchCohort cohort_scope(pending_cohort_target_);
    pending_cohort_target_ = 0;
    if (resampler_ && !resampler_flushed_) {
        resampled_audio_.clear();
        resampler_->finish(&resampled_audio_);
        if (!resampled_audio_.empty()) {
            runner_->feed_audio(resampled_audio_.data(), resampled_audio_.size());
            if (diar_)
                diar_->feed_audio(resampled_audio_.data(), resampled_audio_.size());
        }
        resampler_flushed_ = true;
    }
    if (diar_)
        diar_->finish();  // flush the diarizer tail before tagging final words
    auto u = runner_->finalize();
    return build_result_(u, /*is_final=*/true);
}

std::unique_ptr<RecognitionStream>
Recognizer::streaming_recognize(AsrRequestOptions opts, const std::string& language_code) {
    opts.language_code = language_code;
    auto runner = make_runner();
    if (model_->has_prompt())
        runner->set_prompt_index(model_->prompt_index_for_lang(language_code));
    return std::make_unique<RecognitionStream>(this, std::move(runner), std::move(opts));
}

Result
Recognizer::recognize(
    const float* samples, size_t n, AsrRequestOptions opts, const std::string& language_code,
    int sample_rate) {
    const ggml_backend_t gpu = bm_->gpu_backend_handle();
    const bool vulkan =
        gpu != nullptr && std::string(ggml_backend_name(gpu)).rfind("Vulkan", 0) == 0;
    std::unique_ptr<AsrRunner> runner;
    if (vulkan && model_->head_kind() != HeadKind::Ctc &&
        static_cast<RnntModel*>(model_.get())->supports_cache_streaming()) {
        runner = make_runner();
    } else {
        runner = std::make_unique<OfflineRunner>(model_.get(), cfg_, flashlight_resources_);
    }
    if (model_->has_prompt())
        runner->set_prompt_index(model_->prompt_index_for_lang(language_code));
    opts.language_code = language_code;
    auto stream = std::make_unique<RecognitionStream>(
        this, std::move(runner), std::move(opts), /*coordinate_ingress=*/false);
    stream->push(samples, n, sample_rate);
    return stream->finish();
}

int
Recognizer::sample_rate() const {
    return model_->sample_rate();
}

std::vector<std::string>
Recognizer::supported_languages() const {
    auto languages = model_->prompt_languages();
    if (languages.empty())
        languages.push_back("en-US");
    return languages;
}

double
Recognizer::ms_per_enc_frame() const {
    return model_->ms_per_enc_frame();
}

}  // namespace nemo_speech::asr
