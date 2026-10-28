// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "salm_recognizer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "audio_resampler.h"
#include "errors.h"
#include "salm_engine.h"
#include "salm_guards.h"
#include "salm_policy.h"

namespace nemo_speech::speechlm {
namespace {
// The speaker-tagged transcription prompt the model was trained with.
constexpr const char* kSpeakerPrompt =
    "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. "
    "Use no punctuation and no capitalization. Start every speaker turn with a speaker tag such as "
    "<spk:0>, <spk:1>, <spk:2>, etc.; assign one stable tag per speaker and keep speaker "
    "identities "
    "consistent. Output only the speaker-tagged transcript. <|audio|>";

SalmConfig
engine_config(const asr::RecognizerConfig& c) {
    if (c.model.path.empty())
        throw std::invalid_argument("SALM requires an LLM GGUF in asr.model.path");
    if (!c.diar.model_path.empty())
        throw std::invalid_argument(
            "SALM uses its PEE speaker expert and CTC adapter; remove the separate diarization "
            "model");
    if (!c.vad.model_path.empty())
        throw std::invalid_argument("SALM does not yet support the ASR VAD sidecar");
    const auto& s = c.salm;
    // The batched path rolls rejected drafts back through a 16-row replay cache per sequence, which
    // holds the last accepted token and up to 15 drafts.
    constexpr int kMaxMtpDraft = 15;
    if (s.max_sequences < 1 || s.n_ctx_per_sequence < 1 || s.max_context < 1 || s.n_batch < 1 ||
        s.encoder_max_batch < 1 || s.encoder_ahead_seconds < 0 || s.max_new_tokens < 1 ||
        s.mtp_draft < 0 || s.mtp_draft > kMaxMtpDraft)
        throw std::invalid_argument("invalid SALM scheduling configuration");
    SalmConfig out;
    out.encoder_path = s.encoder_path;
    out.ctc_path = s.ctc_path;
    out.llm_path = c.model.path;
    out.gpu = c.backend.gpu;
    // host-side decoder work: asr.backend.threads, as for the other models (0 = min(8, hardware
    // threads))
    const unsigned hw = std::thread::hardware_concurrency();
    out.threads = c.backend.threads > 0 ? c.backend.threads
                  : hw > 0              ? std::min(8, static_cast<int>(hw))
                                        : 8;
    out.max_sequences = s.max_sequences;
    out.n_ctx_per_sequence = s.n_ctx_per_sequence;
    out.max_context = s.max_context;
    out.n_batch = s.n_batch;
    out.encoder_max_batch = s.encoder_max_batch;
    out.encoder_ahead_seconds = s.encoder_ahead_seconds;
    out.max_new_tokens = s.max_new_tokens;
    out.mtp_draft = s.mtp_draft;
    out.dflash_path = s.dflash_path;
    return out;
}

// One worker thread keeps a SalmEngine::serve loop running, so a request that
// arrives while others are decoding is admitted into the next free slot instead
// of waiting for the whole running group to finish. Requests are admitted in
// arrival order whatever their prompt (diarization, language hint, instructions).
class SalmRecognitionBackend final : public asr::OfflineRecognitionBackend {
   public:
    explicit SalmRecognitionBackend(const asr::RecognizerConfig& c)
        : engine_(engine_config(c)), postproc_(c.postproc, nullptr, true, c.batching),
          max_queue_depth_(static_cast<size_t>(std::max(1, c.batching.max_queue_depth))),
          worker_([this] { run(); }) {}

    ~SalmRecognitionBackend() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }

    asr::Result recognize(
        const float* samples, size_t n, asr::AsrRequestOptions opts, const std::string& language,
        int rate) override {
        if (opts.needs_word_timings() && engine_.config().ctc_path.empty())
            throw std::invalid_argument(
                "SALM word timestamps and diarization need the CTC timestamp head, which this "
                "encoder file does not carry");
        if (!opts.speech_contexts.empty())
            throw std::invalid_argument(
                "SALM does not yet support speech_contexts / phrase boosting");
        if (rate == 0)
            rate = sample_rate();
        if (rate < 8000 || rate > 96000 || (n && !samples))
            throw std::invalid_argument(
                "invalid SALM audio: expected mono samples at 8000..96000 Hz");
        const auto& sl = opts.speechlm;
        if (sl.task != "transcribe" && sl.task != "translate")
            throw std::invalid_argument("task must be transcribe or translate");
        if (!(sl.temperature >= 0 && sl.temperature <= 1))
            throw std::invalid_argument("temperature must be between 0 and 1");
        // A special token in the instruction would forge or close a chat-template turn.
        if (const auto token = engine_.special_token_in(sl.instructions); !token.empty())
            throw std::invalid_argument(
                "the final text must not contain chat template control tokens (" + token +
                " is a special token of the served tokenizer)");
        // Request rules (salm_policy): the served prompt with its language hint, the translation
        // prompt or the client's instructions, and an output budget from the duration.
        policy::PromptRequest prompt_request;
        prompt_request.task =
            sl.task == "translate" ? policy::Task::kTranslate : policy::Task::kTranscribe;
        if (!sl.instructions.empty())
            prompt_request.instruction = sl.instructions;
        else if (opts.enable_speaker_diarization)
            prompt_request.instruction = std::string(kSpeakerPrompt);
        if (!language.empty() && language != "auto")
            prompt_request.language = language;
        if (!sl.target_language.empty())
            prompt_request.target_language = sl.target_language;
        const auto plan = policy::plan_prompt(prompt_config_, prompt_request);
        const double seconds = static_cast<double>(n) / rate;
        if (const auto error = policy::duration_limit_error(seconds))
            throw std::invalid_argument(*error);
        const int64_t budget = policy::output_budget(seconds, {engine_.config().max_context});
        if (n == 0)
            return {};
        // Near-digital silence: an empty transcript, no encoder or decoder work and no usage.
        if (policy::near_silent(samples, n)) {
            auto details = std::make_shared<asr::SpeechLmDetails>();
            details->language = plan.language;
            details->task = sl.task;
            details->duration_seconds = seconds;
            details->timestamps_requested = opts.enable_word_time_offsets;
            if (opts.needs_word_timings()) {
                details->alignment_status = "unavailable";
                details->failure_reason = "empty_transcript";
            }
            asr::Result result;
            result.audio_processed = static_cast<float>(seconds);
            asr::Alternative alt;
            if (!language.empty() && language != "auto")
                alt.language_codes.push_back(language);
            result.alternatives.push_back(std::move(alt));
            result.speechlm = std::move(details);
            return result;
        }
        opts.language_code = language;
        // Other rates are converted here, on the request's thread, rather than serially on the
        // engine's encoder thread; the request keeps the original duration.
        std::vector<float> resampled;
        if (rate != sample_rate()) {
            resampled = audio::resample_audio(samples, n, rate, sample_rate());
            samples = resampled.data();
            n = resampled.size();
            rate = sample_rate();
        }
        // The synchronous submit keeps borrowed samples alive until inference completes.
        SalmRequest request{"", samples, n, rate};
        request.seconds = seconds;
        request.prompt = plan.user_turn;
        request.max_new_tokens = static_cast<int>(budget);
        request.timestamps = opts.needs_word_timings();
        request.temperature = sl.temperature;
        request.latin_script = guards::script_guard(plan.language, !sl.instructions.empty());
        if (sl.has_seed)
            request.seed = sl.seed;
        // The response text as it is generated, whitespace stripped. A first decode that ends
        // empty (the fallback decode below runs) has shown nothing, so the stream carries on.
        if (sl.on_text)
            request.on_text = [stream = std::make_shared<policy::ResponseTextStream>(),
                               sink = sl.on_text](std::string_view piece) {
                if (const std::string shown = stream->push(piece); !shown.empty())
                    sink(shown);
            };
        // An empty transcript without timestamps gets one more decode of the same audio (its
        // encoder output) with the fallback prompt (same budget, sampling and script guard); usage
        // adds up. The fallback prompt asks for a transcription, so translations and custom
        // instructions keep their empty result.
        if (!request.timestamps && sl.task == "transcribe" && !plan.from_client)
            request.fallback_prompt = std::string(policy::kFallbackPrompt);
        auto raw = submit(std::move(request), opts.cancelled);
        if (raw.cancelled)
            throw asr::Cancelled("SALM: request cancelled");
        if (!raw.error.empty()) {
            if (raw.invalid_request)
                throw std::invalid_argument("SALM: " + raw.error);
            throw std::runtime_error("SALM: " + raw.error);
        }
        std::optional<policy::Usage> usage;
        if (const auto& first = raw.first_attempt)
            usage = policy::decode_usage(
                first->prompt_tokens - first->audio_tokens, first->audio_tokens,
                first->generated_tokens, first->end_of_turn);
        policy::add_usage(
            usage, policy::decode_usage(
                       raw.prompt_tokens - raw.audio_tokens, raw.audio_tokens, raw.generated_tokens,
                       raw.end_of_turn));
        // A failed alignment keeps the transcript (status "unavailable", no words).
        if (!raw.timestamp_error.empty())
            std::fprintf(
                stderr, "[salm] word timestamps unavailable: %s\n", raw.timestamp_error.c_str());
        auto details = raw.details ? std::make_shared<asr::SpeechLmDetails>(*raw.details)
                                   : std::make_shared<asr::SpeechLmDetails>();
        details->text = policy::response_text(raw.text);
        details->language = plan.language;
        details->task = sl.task;
        details->transcript_origin = "llm";
        details->duration_seconds = seconds;
        details->timestamps_requested = opts.enable_word_time_offsets;
        details->prompt_tokens = static_cast<int>(usage->prompt_tokens);
        details->audio_tokens = static_cast<int>(usage->audio_tokens.value_or(0));
        details->completion_tokens = static_cast<int>(usage->completion_tokens);
        auto result = to_asr_result(raw, opts, language, postproc_);
        result.speechlm = std::move(details);
        return result;
    }

    // The batched path serves clips of 3 to 30 s (low noise, 4 generated tokens each) together, as
    // traffic does, so the encoder and frontend graph caches and the compute buffers of the large
    // prefill and decode shapes are set up before the first real request. Otherwise one second of
    // silence.
    void warmup() override {
        const int rate = sample_rate();
        const size_t n = engine_.batched() ? std::min<size_t>(96, max_queue_depth_) : 1;
        std::vector<std::vector<float>> audio(n);
        std::mt19937 rng(12345);
        std::normal_distribution<float> noise(0.0f, 0.02f);
        std::deque<Job> jobs;  // stable addresses for the queue
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_)
                throw asr::Unavailable("SALM: backend is shutting down");
            for (size_t i = 0; i < n; ++i) {
                if (n == 1) {
                    audio[i].assign(static_cast<size_t>(rate), 0.0f);
                } else {
                    audio[i].resize(static_cast<size_t>(rate) * (3 + (i * 7) % 28));
                    for (float& v : audio[i]) v = noise(rng);
                }
                SalmRequest r{"warmup", audio[i].data(), audio[i].size(), rate};
                r.max_new_tokens = 4;
                r.decode_without_voice = true;  // the noise clips must reach the decoder
                jobs.push_back(Job{std::move(r), Clock::now(), {}, {}});
                queue_.push_back(&jobs.back());
            }
        }
        cv_.notify_all();
        // The worker borrows `jobs` and `audio`: wait for every job before reporting a failure.
        std::string error;
        std::exception_ptr exception;
        for (Job& job : jobs) {
            try {
                const SalmResult result = job.result.get_future().get();
                if (error.empty() && !result.error.empty())
                    error = result.error;
            }
            catch (...) {
                if (!exception)
                    exception = std::current_exception();
            }
        }
        if (exception)
            std::rethrow_exception(exception);
        if (!error.empty())
            throw std::runtime_error("SALM warmup: " + error);
    }
    bool supports_word_timestamps() const override { return !engine_.config().ctc_path.empty(); }
    bool supports_speaker_diarization() const override { return supports_word_timestamps(); }
    bool supports_offline_diarization() const override { return supports_word_timestamps(); }
    // The encoder's speaker branch alone: the request ends at the encoder output (no decode).
    std::vector<asr::SpeechLmSegment> diarize(const float* samples, size_t n, int rate) override {
        if (!supports_offline_diarization())
            throw std::invalid_argument(
                "SALM diarization needs the CTC timestamp head, which this encoder file does not "
                "carry");
        if (rate == 0)
            rate = sample_rate();
        if (rate < 8000 || rate > 96000 || (n && !samples))
            throw std::invalid_argument(
                "invalid SALM audio: expected mono samples at 8000..96000 Hz");
        const double seconds = static_cast<double>(n) / rate;
        if (const auto error = policy::duration_limit_error(seconds))
            throw std::invalid_argument(*error);
        // Near-digital silence has no speaker turns (and no encoder work), as in recognize().
        if (n == 0 || policy::near_silent(samples, n))
            return {};
        std::vector<float> resampled;
        if (rate != sample_rate()) {
            resampled = audio::resample_audio(samples, n, rate, sample_rate());
            samples = resampled.data();
            n = resampled.size();
            rate = sample_rate();
        }
        SalmRequest request{"", samples, n, rate};
        request.seconds = seconds;
        request.timestamps = true;
        request.diarization_only = true;
        auto raw = submit(std::move(request));
        if (!raw.error.empty()) {
            if (raw.invalid_request)
                throw std::invalid_argument("SALM: " + raw.error);
            throw std::runtime_error("SALM: " + raw.error);
        }
        return raw.details ? raw.details->diarization : std::vector<asr::SpeechLmSegment>{};
    }
    bool supports_translation() const override { return true; }
    bool supports_instructions() const override { return true; }
    std::vector<std::string> translation_languages() const override {
        std::vector<std::string> codes;
        for (const auto& entry : policy::language_names()) codes.emplace_back(entry.first);
        return codes;
    }
    int sample_rate() const override { return engine_.sample_rate(); }
    std::vector<std::string> supported_languages() const override { return {"auto"}; }
    asr::BatchMetrics batch_metrics() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return metrics_;
    }

   private:
    using Clock = std::chrono::steady_clock;
    // How often a waiting request checks its client (AsrRequestOptions::cancelled).
    static constexpr std::chrono::milliseconds kCancelPoll{200};
    struct Job {
        SalmRequest request;
        Clock::time_point queued;
        std::promise<SalmResult> result;
        std::optional<size_t> ticket;  // under mutex_: the engine's ticket once pulled
    };

    // Blocks the caller until its own request completes; the borrowed samples
    // stay alive for that long. With `cancelled`, the wait checks the client every kCancelPoll
    // and, once it is gone, drops the request: from the queue at once, or through the engine
    // (whose result comes back cancelled once it no longer reads the samples).
    SalmResult submit(SalmRequest request, const std::function<bool()>& cancelled = {}) {
        Job job{std::move(request), Clock::now(), {}, {}};
        auto future = job.result.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_)
                throw asr::Unavailable("SALM: backend is shutting down");
            if (queue_.size() >= max_queue_depth_)
                throw asr::ResourceExhausted("SALM: queue is full");
            queue_.push_back(&job);
            ++metrics_.requested_items;
        }
        cv_.notify_all();
        if (cancelled) {
            while (future.wait_for(kCancelPoll) != std::future_status::ready) {
                if (cancelled()) {
                    cancel(job);
                    break;
                }
            }
        }
        SalmResult result = future.get();
        if (result.cancelled) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++metrics_.cancelled_items;
        }
        return result;
    }

    // The client of `job` (still waiting in submit) went away.
    void cancel(Job& job) {
        size_t queued_total = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = std::find(queue_.begin(), queue_.end(), &job);
            if (it == queue_.end()) {
                // Pulled: the engine drops it at its next step boundary. A finished ticket is
                // ignored; tickets are reset under this lock when a serve() call ends.
                if (job.ticket)
                    engine_.cancel(*job.ticket);
                return;
            }
            queue_.erase(it);
            queued_total = ++cancelled_queued_;
        }
        SalmResult result;
        result.cancelled = true;
        result.error = "request cancelled";
        job.result.set_value(std::move(result));
        std::fprintf(
            stderr, "[salm] cancelled 1 request before the engine took it; %zu in total\n",
            queued_total);
    }

    void run() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (stop_) {
                    for (Job* job : queue_)
                        job->result.set_exception(std::make_exception_ptr(
                            std::runtime_error("SALM: backend is shutting down")));
                    queue_.clear();
                    return;
                }
            }
            serve();
        }
    }

    // Runs until the backend stops (or the engine fails: in-flight and queued requests get the
    // error).
    void serve() {
        std::unordered_map<size_t, Job*> in_flight;
        size_t next_ticket = 0;
        auto pull = [&](std::vector<SalmRequest>& out, size_t max, bool block) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (block)
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            const auto now = Clock::now();
            while (out.size() < max && !queue_.empty()) {
                Job* job = queue_.front();
                queue_.pop_front();
                metrics_.queue_wait_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - job->queued)
                        .count());
                job->ticket = next_ticket;
                in_flight.emplace(next_ticket++, job);
                out.push_back(job->request);
            }
            return !stop_;
        };
        auto done = [&](size_t ticket, SalmResult&& result) {
            const auto it = in_flight.find(ticket);
            Job* job = it->second;
            in_flight.erase(it);
            job->result.set_value(std::move(result));
        };
        SalmStats stats;
        const auto started = Clock::now();
        try {
            engine_.serve(pull, done, &stats);
        }
        catch (...) {
            // a failed engine fails what is queued too (no retry loop); the tickets of this
            // serve() call are reset first, so no cancellation reaches the next one
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& entry : in_flight) {
                entry.second->ticket.reset();
                entry.second->result.set_exception(std::current_exception());
            }
            in_flight.clear();
            for (Job* job : queue_) job->result.set_exception(std::current_exception());
            queue_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            metrics_.batches += stats.refills;
            metrics_.items += next_ticket;
            metrics_.max_observed_batch = std::max<uint64_t>(
                metrics_.max_observed_batch, static_cast<uint64_t>(stats.peak_sequences));
            metrics_.execution_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
                    .count());
        }
    }

    SalmEngine engine_;
    // The served prompts and language hint; the transcription prompt is SalmConfig's.
    policy::PromptConfig prompt_config_{std::string(SalmConfig{}.prompt)};
    asr::postproc::Postprocessor postproc_;
    const size_t max_queue_depth_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Job*> queue_;
    size_t cancelled_queued_ = 0;  // requests cancelled before the engine took them
    bool stop_ = false;
    asr::BatchMetrics metrics_;
    // Last member: started after everything it uses, joined in the destructor.
    std::thread worker_;
};
}  // namespace

asr::Result
to_asr_result(
    const SalmResult& raw, const asr::AsrRequestOptions& opts, const std::string& language,
    const asr::postproc::Postprocessor& postproc) {
    asr::Result result;
    result.audio_processed = static_cast<float>(raw.audio_seconds);
    asr::Alternative alt;
    if (!language.empty() && language != "auto")
        alt.language_codes.push_back(language);
    // Words of the alignment
    const std::vector<asr::SpeechLmWord> none;
    const auto& aligned = raw.details ? raw.details->words : none;
    // The decoded transcript; the speakers the server's prompt asked for go on the words, not in
    // the text.
    const auto decoded_text = [&] {
        return postproc.apply(
            opts.enable_speaker_diarization ? policy::strip_speaker_tags(raw.text) : raw.text, opts,
            nullptr, language);
    };
    if (!opts.needs_word_timings() || aligned.empty()) {
        alt.transcript = decoded_text();
    } else {
        // The text keeps the decoder's order (word_index): the aligned words are sorted by time,
        // which reorders overlapping speakers. Each contiguous speaker turn is normalized on its
        // own so ITN cannot merge words across speakers; `words` ends up in time order again.
        // Timings use an integer millisecond grid.
        std::vector<const asr::SpeechLmWord*> ordered;
        ordered.reserve(aligned.size());
        for (const auto& w : aligned) ordered.push_back(&w);
        std::stable_sort(ordered.begin(), ordered.end(), [](const auto* a, const auto* b) {
            return a->word_index < b->word_index;
        });
        size_t begin = 0;
        while (begin < ordered.size()) {
            size_t end = begin + 1;
            const int speaker = ordered[begin]->speaker;
            while (end < ordered.size() &&
                   (!opts.enable_speaker_diarization || ordered[end]->speaker == speaker))
                ++end;
            std::string text;
            std::vector<asr::WordTiming> words;
            for (size_t i = begin; i < end; ++i) {
                const auto& w = *ordered[i];
                if (!text.empty() && !w.joins_previous)
                    text += ' ';
                text += w.display.empty() ? w.word : w.display;
                words.push_back(
                    {w.word, std::llround(w.start * 1000), std::llround(w.end * 1000), 0.f});
            }
            text = postproc.apply(text, opts, &words, language);
            // Running text of character units, while the words are still the aligned ones.
            const bool same_words = words.size() == end - begin;
            if (!alt.transcript.empty() && !text.empty())
                alt.transcript += ' ';
            alt.transcript += text;
            for (size_t k = 0; k < words.size(); ++k) {
                const auto& w = words[k];
                asr::Word out;
                out.word = w.word;
                if (same_words && w.word == ordered[begin + k]->word) {  // not masked
                    out.display = ordered[begin + k]->display;
                    out.joins_previous = ordered[begin + k]->joins_previous;
                }
                out.start_time = static_cast<int32_t>(w.start_frame);
                out.end_time = static_cast<int32_t>(w.end_frame);
                out.language_code = language == "auto" ? "" : language;
                out.speaker_tag = opts.enable_speaker_diarization && speaker >= 0 ? speaker + 1 : 0;
                alt.words.push_back(std::move(out));
            }
            begin = end;
        }
        std::stable_sort(
            alt.words.begin(), alt.words.end(), [](const asr::Word& a, const asr::Word& b) {
                return a.start_time != b.start_time ? a.start_time < b.start_time
                                                    : a.end_time < b.end_time;
            });
        // Some speakers' words could not be aligned: the transcript keeps them (the decoded text);
        // `words` has the aligned ones.
        if (raw.details->unaligned_words > 0)
            alt.transcript = decoded_text();
    }
    result.alternatives.push_back(std::move(alt));
    return result;
}

std::unique_ptr<asr::OfflineRecognitionBackend>
make_salm_backend(const asr::RecognizerConfig& config) {
    return std::make_unique<SalmRecognitionBackend>(config);
}
}  // namespace nemo_speech::speechlm
