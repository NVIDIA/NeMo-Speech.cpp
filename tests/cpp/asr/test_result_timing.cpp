// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// No arguments: projection policy. Optional MODEL.gguf AUDIO.wav [GPU] [DIAR.gguf] [VAD.gguf]
// exercises actual SDK results, short EOF, input packets, resampling and EOU drain.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "itn_align.h"
#include "recognizer.h"
#include "word_time.h"

using namespace nemo_speech::asr;

static void
require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void
check_projection() {
    WordTiming late_word{"late", 14, 15, 1.0f};
    bound_word_frames(late_word, 8);
    require(
        late_word.start_frame == 7 && late_word.end_frame == 8 && !late_word.has_audio_position,
        "EOS text gets the last real interval, not speaker evidence");
    bound_word_frames(late_word, 8);
    require(!late_word.has_audio_position, "projection must preserve provenance when repeated");
    WordTiming partial_word{"partial", 6, 15, 1.0f};
    bound_word_frames(partial_word, 8);
    require(
        partial_word.start_frame == 6 && partial_word.end_frame == 8 &&
            partial_word.has_audio_position,
        "partly real word keeps its original onset");
    WordTiming empty_word{"empty", 0, 1, 1.0f};
    bound_word_frames(empty_word, 0);
    require(
        empty_word.start_frame == 0 && empty_word.end_frame == 0 && !empty_word.has_audio_position,
        "empty input has no real frame to borrow");
    for (uint64_t samples = 256; samples <= 32000; samples += 17) {
        const int64_t mels = (samples - 256) / 160 + 1;
        const int64_t frames = (mels - 1) / 8 + 1;
        WordTiming word{"tail", frames + 5, frames + 6, 1.0f};
        bound_word_frames(word, frames);
        const auto time = bound_word_time(word.start_frame, word.end_frame, 80, samples, 16000);
        require(
            time.start_ms < time.end_ms && time.end_ms <= samples * 1000.0 / 16000,
            "frame bounding plus sample bounding handles fractional tail frames");
    }
    const auto empty = bound_word_time(0, 1, 80.0, 0, 16000);
    require(
        real_audio_extent_ms(28224, 44100) == 640 && real_audio_extent_ms(1001, 44100) == 22,
        "all exporters must use the same sample-derived millisecond bound");
    require(empty.start_ms == 0 && empty.end_ms == 0 && !empty.has_audio_position, "empty bounds");
    const auto normal = bound_word_time(2, 4, 80.0, 16000, 16000);
    const auto long_audio = bound_word_time(89999, 90000, 80.0, 7200ULL * 44100, 44100);
    require(
        long_audio.start_ms == 7199920 && long_audio.end_ms == 7200000,
        "long recordings retain absolute time without per-chunk rounding drift");
    require(
        normal.start_ms == 160 && normal.end_ms == 320 && normal.has_audio_position,
        "valid nominal word times must not receive the decode-frontier offset");
    const auto partial = bound_word_time(7, 15, 80.0, 10240, 16000);
    require(
        partial.start_ms == 560 && partial.end_ms == 640 && partial.has_audio_position,
        "partly observed word keeps its real interval");
    const auto late = bound_word_time(14, 15, 80.0, 10240, 16000);
    require(
        late.start_ms == 640 && late.end_ms == 640 && !late.has_audio_position,
        "millisecond safeguard rejects unbounded coordinates from other producers");
    const auto point = bound_word_time(2, 2, 80.0, 16000, 16000);
    require(point.has_audio_position, "valid zero-width ITN spans still have an audio position");
    const auto fraction = bound_word_time(0, 2, 80.0, 1001, 44100);
    require(fraction.end_ms == 22, "millisecond rounding must not exceed actual source duration");
    const auto large = bound_word_time(
        std::numeric_limits<int64_t>::max() - 1, std::numeric_limits<int64_t>::max(), 80.0, 16000,
        16000);
    require(
        large.start_ms == 1000 && large.end_ms == 1000 && !large.has_audio_position,
        "large model positions cannot overflow exported milliseconds");
}

static void
check_result(const Result& result, double received, bool eof) {
    require(
        result.audio_processed >= 0 && result.audio_processed <= received + 1e-5,
        "processed audio exceeds received real samples");
    if (eof)
        require(
            std::fabs(result.audio_processed - received) < 1e-5, "EOF must report full duration");
    require(
        result.audio_extent_ms >= 0 && result.audio_extent_ms <= received * 1000.0 + 1e-6,
        "sample-derived display extent exceeds input");
    for (const auto& alt : result.alternatives) {
        int32_t previous = 0;
        for (const auto& word : alt.words) {
            require(
                word.start_time >= previous && word.end_time >= word.start_time &&
                    word.end_time <= received * 1000.0 + 1e-6,
                "word range outside real audio");
            previous = word.start_time;
            if (!word.has_audio_position)
                require(word.speaker_tag == 0, "EOS-backfilled words must remain unattributed");
        }
    }
}

static void
check_stream(Recognizer& rec, const std::vector<float>& audio, int rate, size_t packet) {
    AsrRequestOptions opts;
    opts.enable_word_time_offsets = rec.diar_model() == nullptr;
    opts.enable_speaker_diarization = rec.diar_model() != nullptr;
    auto stream = rec.streaming_recognize(opts, "auto");
    stream->set_interim_words(true);
    float last = 0;
    for (size_t off = 0; off < audio.size(); off += packet) {
        const size_t n = std::min(packet, audio.size() - off);
        stream->push(audio.data() + off, n, rate);
        int drains = 0;
        while (auto result = stream->next()) {
            require(++drains < 100, "stream drain must terminate");
            check_result(*result, double(off + n) / rate, false);
            require(result->audio_processed >= last, "decoded progress must be monotonic");
            last = result->audio_processed;
        }
    }
    auto final = stream->finish();
    check_result(final, double(audio.size()) / rate, true);
    if (rec.config().streaming.rnnt_right_context == 13 && audio.size() == size_t(rate * 0.64)) {
        require(
            !final.alternatives.empty() && !final.alternatives.front().words.empty(),
            "0.64s JFK/R13 regression must not pass with an empty transcript");
        for (const auto& word : final.alternatives.front().words)
            require(
                word.end_time > word.start_time && word.end_time <= 640,
                "JFK/R13 EOF words must have positive, in-recording intervals");
    }
    stream->refresh_speaker_tags(final);
    check_result(final, double(audio.size()) / rate, true);
    check_result(stream->finish(), double(audio.size()) / rate, true);
    bool rejected = false;
    const float sample = 0.0f;
    try {
        stream->push(&sample, 1, rate);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "closed input extent cannot grow after EOF");
}

static void
check_model(int argc, char** argv) {
    std::vector<float> audio;
    int sr = 0;
    require(read_wav_mono_16k(argv[2], audio, sr), "cannot read fixture WAV");
    RecognizerConfig base;
    base.model.path = argv[1];
    base.backend.gpu = argc >= 4 ? std::stoi(argv[3]) : -1;
    base.log_status = false;
    base.batching.enabled = false;
    if (argc >= 5)
        base.diar.model_path = argv[4];
    if (argc >= 6) {
        base.vad.model_path = argv[5];
        base.endpointing.enable = true;
        base.endpointing.vad_based = true;
    }
    for (int right : {0, 3, 13}) {
        auto cfg = base;
        cfg.streaming.rnnt_right_context = right;
        Recognizer rec(cfg);
        if (right == 13) {
            auto* model = static_cast<RnntModel*>(rec.model());
            CacheStreamRunner runner(model, cfg);
            runner.set_prompt_index(model->prompt_index_for_lang("auto"));
            AsrRequestOptions opts;
            opts.enable_word_time_offsets = true;
            runner.set_request_options(opts);
            const size_t count = size_t(sr * 0.64);
            require(audio.size() >= count, "JFK fixture must contain at least 0.64s");
            runner.feed_audio(audio.data(), count);
            runner.step();
            const auto result = runner.finalize();
            require(!result.words.empty(), "raw runner JFK/R13 must preserve tail text");
            const int64_t mels = (count - model->fe().n_fft() / 2) / model->fe().hop_length() + 1;
            const int64_t frames = (mels - 1) / model->encoder_config().subsampling_factor + 1;
            for (const auto& word : result.words)
                require(
                    word.start_frame >= 0 && word.end_frame > word.start_frame &&
                        word.end_frame <= frames,
                    "raw runner must bound EOF words before any postprocessing");
            require(
                std::any_of(
                    result.words.begin(), result.words.end(),
                    [](const auto& w) { return !w.has_audio_position; }),
                "JFK/R13 must exercise EOS-backfill provenance");
            runner.reset();
            runner.feed_audio(audio.data(), count);
            runner.step();
            const auto repeated = runner.finalize();
            require(
                result.transcript_so_far == repeated.transcript_so_far &&
                    result.words.size() == repeated.words.size(),
                "reset preserves EOF text");
            for (size_t i = 0; i < result.words.size(); ++i)
                require(
                    result.words[i].start_frame == repeated.words[i].start_frame &&
                        result.words[i].end_frame == repeated.words[i].end_frame &&
                        result.words[i].has_audio_position == repeated.words[i].has_audio_position,
                    "reset cannot leak timing bounds or provenance");
        }
        for (double seconds : {0.0, 0.016, 0.05, 0.16, 0.625, 0.64, 0.66, 2.0, 11.0}) {
            const size_t n = std::min(audio.size(), size_t(std::llround(seconds * sr)));
            const std::vector<float> clip(audio.begin(), audio.begin() + n);
            for (size_t packet : {size_t(1279), size_t(2560), std::max(size_t(1), n)})
                check_stream(rec, clip, sr, packet);
        }
        if (right == 3) {
            const auto resampled =
                nemo_speech::audio::resample_audio(audio.data(), audio.size(), sr, 44100);
            check_stream(rec, resampled, 44100, 7056);
        }
        std::printf("[PASS] R=%d SDK bounds/EOF/packets\n", right);
    }
    auto cfg = base;
    cfg.streaming.rnnt_right_context = 3;
    cfg.endpointing.enable = true;
    cfg.log_status = false;
    Recognizer rec(cfg);
    auto twice = audio;
    twice.insert(twice.end(), size_t(sr) * 2, 0.0f);
    twice.insert(twice.end(), audio.begin(), audio.end());
    AsrRequestOptions opts;
    opts.enable_word_time_offsets = true;
    opts.enable_speaker_diarization = rec.diar_model() != nullptr;
    auto stream = rec.streaming_recognize(opts, "auto");
    stream->push(twice.data(), twice.size(), sr);
    const double duration = double(twice.size()) / sr;
    int finals = 0, interims_after_final = 0, drains = 0;
    float last = 0;
    while (auto result = stream->next()) {
        require(++drains < 100, "large-input drain must terminate");
        check_result(*result, duration, false);
        require(result->audio_processed >= last, "EOU progress must not reset");
        if (result->is_final) {
            ++finals;
            require(
                result->audio_processed < duration,
                "mid-stream final must not report all queued audio");
        } else if (finals > 0) {
            ++interims_after_final;
            require(
                result->audio_processed > last,
                "remaining audio must produce fresh progress without push");
        }
        last = result->audio_processed;
    }
    require(
        finals > 0 && interims_after_final > 0,
        "exercise finals then a trailing interim from one push");
    check_result(stream->finish(), duration, true);
    std::puts("[PASS] EOU drain and result delivery");
}

static void
check_itn() {
    std::vector<WordTiming> words{{"two", 14, 15, 0.8f}, {"kilograms", 21, 22, 0.9f}};
    for (auto& word : words) bound_word_frames(word, 8);
    postproc::update_word_timings(
        words, "Token:\t0\ttwo kilograms\t0,12\t0,1\nWord:\t0\t2\t0\nWord:\t1\tkg\t0\n");
    require(
        words.size() == 2 && words[0].word == "2" && words[1].word == "kg",
        "ITN output order must survive equal EOF-clamped timestamps");
    for (const auto& word : words)
        require(
            !word.has_audio_position && word.start_frame == 7 && word.end_frame == 8,
            "ITN exact matching and numeric collapse preserve EOS provenance");
    words = {{"abbr", 7, 8, 1.0f, false}};
    postproc::update_word_timings(
        words, "Token:\t0\tabbr\t0,4\t0,1\nWord:\t0\tfirst\t0\nWord:\t1\tsecond\t0\n");
    require(
        words.size() == 2 && words[0].start_frame == words[0].end_frame,
        "ITN expansion can legitimately produce a zero-width frame span");
    for (const auto& word : words)
        require(
            !word.has_audio_position && word.start_frame >= 7 && word.end_frame <= 8,
            "ITN expansion retains bounds and conservative provenance");
    words = {{"one", 5, 6, 1.0f}, {"two", 6, 7, 1.0f}, {"three", 7, 8, 1.0f, false}};
    postproc::update_word_timings(
        words, "Token:\t0\tone two three\t0,13\t0,1\nWord:\t0\talpha\t0\nWord:\t1\tbeta\t0\n");
    require(
        words.size() == 2 && words[0].has_audio_position && !words[1].has_audio_position,
        "ITN regrouping only invalidates groups that include an EOS-only word");
}

int
main(int argc, char** argv) try {
    check_projection();
    check_itn();
    if (argc >= 3)
        check_model(argc, argv);
    else
        std::puts("[SKIP] SDK integration: supply model and WAV");
    std::puts("[PASS] result timing");
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
}
