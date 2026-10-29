// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// No arguments: model-independent causal first/steady-state shape checks.
// Optional: MODEL.gguf [--gpu N] exercises actual streaming, reset and EOF,
// plus word coordinates and end-of-stream word bounds on the repository's JFK sample.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "recognizer.h"

using namespace nemo_speech::asr;

static void
require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

static void
check_shapes() {
    for (int right : {0, 1, 3, 6, 13}) {
        const auto cfg = make_cache_aware_config(EncoderConfig{}, right);
        const int first = 1 + 8 * right;
        const int steady = 9 + 8 * (1 + right);
        require(cfg.cache_first_chunk_mel_frames() == first, "wrong first mel length");
        require(cfg.subsample_time_length(first) == 1 + right, "first chunk needs no drop");
        require(
            cfg.subsample_time_length(steady) - cfg.cache_drop_extra == 1 + right,
            "steady chunk must drop only the overlap");
        require(cfg.cache_decoded_mel_frames(0) == 0, "empty stream has a nonzero decode clock");
        require(cfg.cache_decoded_mel_frames(1) == 1, "first encoder frame must end at mel zero");
        for (int chunks : {1, 2, 100}) {
            const int64_t emitted = static_cast<int64_t>(chunks) * (1 + right);
            const int64_t consumed = first + static_cast<int64_t>(chunks - 1) * 8 * (1 + right);
            require(cfg.cache_decoded_mel_frames(emitted) == consumed, "VAD decode clock is ahead");
        }
    }

    // R13: 14 emitted frames cover 105 real mel frames, not 112. At a 10ms
    // hop, the old clock would fire an 800ms endpoint after only 730ms silence.
    const auto cfg = make_cache_aware_config(EncoderConfig{}, 13);
    VadEndpointer endpoint(VadEndpointerCfg{true, true, 800.0f});
    require(!endpoint.poll(320.0, 320.0), "speech must not immediately endpoint");
    require(
        !endpoint.poll(cfg.cache_decoded_mel_frames(14) * 10.0, 320.0),
        "VAD endpoint fired on the old 70ms-ahead clock");
    require(
        endpoint.poll(cfg.cache_decoded_mel_frames(15) * 10.0, 320.0),
        "VAD endpoint did not fire after decoded silence reached the threshold");
}

// Speaker attribution places encoder frame f at the centre of its receptive
// field. Derive that field by walking the k=3/stride-2 stages: stage output j
// reads input [2j - pad_left, 2j - pad_left + 2].
static void
check_frame_centers() {
    for (int sub : {2, 4, 8}) {
        for (auto context : {ConvContext::Causal, ConvContext::Symmetric}) {
            EncoderConfig cfg;
            cfg.subsampling_factor = sub;
            cfg.conv_context = context;
            const int pad_left = context == ConvContext::Causal ? 2 : 1;
            for (int f : {0, 1, 13, 14, 1000}) {
                int lo = f, hi = f;
                for (int stage = sub; stage > 1; stage /= 2) {
                    lo = 2 * lo - pad_left;
                    hi = 2 * hi - pad_left + 2;
                }
                require(
                    lo + hi == 2 * (f * sub + cfg.frame_center_mel_offset()),
                    "frame centre offset disagrees with the receptive field");
                if (context == ConvContext::Causal)
                    require(hi == f * sub, "causal frame must end at mel f*sub");
            }
        }
    }
    // Nemotron 8x causal subsampling: frame f spans mel [8f-14, 8f], so word
    // times (f * 80 ms) sit 70 ms after the centre the diarizer compares with.
    const auto cfg = make_cache_aware_config(EncoderConfig{}, 13);
    require(cfg.frame_center_mel_offset() == -7, "Nemotron frame centre must be 7 mel early");
}

static void
same_encoder(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "encoder size differs from explicit reference chunk");
    float largest = 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite encoder projection");
        largest = std::max(largest, std::fabs(a[i] - b[i]));
        sum += static_cast<double>(a[i] - b[i]) * (a[i] - b[i]);
    }
    if (largest > 1e-5f)
        std::fprintf(
            stderr, "encoder max abs error=%g RMS=%g\n", largest, std::sqrt(sum / a.size()));
    require(largest <= 1e-5f, "encoder differs from explicit first/history chunk");
}

static std::vector<float>
jfk_audio() {
    std::vector<float> audio;
    int rate = 0;
    require(
        read_wav_mono_16k(NEMO_SPEECH_TEST_FILES_DIR "/asr/wav/test/jfk.wav", audio, rate) &&
            rate == 16000,
        "cannot read test_files/asr/wav/test/jfk.wav");
    return audio;
}

// Runner words must sit on an independently derived coordinate: decode
// explicitly assembled chunks with a fresh decoder at absolute frame
// chunk * (R + 1). Packets match chunk boundaries, as in check_model.
static void
check_first_word_frame(const RecognizerConfig& cfg, RnntModel* model, int right) {
    const auto audio = jfk_audio();
    const auto enc = make_cache_aware_config(model->encoder_config(), right);
    const auto& fe = model->fe_config();
    const int hop = model->fe().hop_length();
    const int first_mel = 1 + enc.subsampling_factor * right;
    const int new_mel = enc.subsampling_factor * (1 + right);
    const int overlap = 9;
    const size_t first_samples = (first_mel - 1) * hop + fe.n_fft / 2;
    const size_t stride_samples = new_mel * hop;
    const int prompt = model->prompt_index_for_lang("auto");

    AsrRequestOptions opts;
    opts.enable_word_time_offsets = true;
    CacheStreamRunner runner(model, cfg);
    runner.set_prompt_index(prompt);
    runner.set_request_options(opts);
    auto reference = model->make_transducer_decoder();
    reference->set_compute_timestamps(true);
    auto state = model->make_cache_state();
    std::vector<float> all_mel;
    int consumed = 0, cached = 0;
    bool compared = false;
    for (int chunk = 0; !compared; ++chunk) {
        const size_t available = first_samples + chunk * stride_samples;
        require(available <= audio.size(), "no word in the JFK sample");
        const std::vector<float> prefix(audio.begin(), audio.begin() + available);
        std::vector<float> fresh_mel;
        produce_new_mel_frames(
            model->fe(), prefix, 0, static_cast<int64_t>(all_mel.size() / fe.n_mels), fresh_mel);
        all_mel.insert(all_mel.end(), fresh_mel.begin(), fresh_mel.end());
        const int lead = chunk == 0 ? 0 : overlap;
        const int frames = lead + (chunk == 0 ? first_mel : new_mel);
        std::vector<float> mel(static_cast<size_t>(frames) * fe.n_mels, 0.0f);
        for (int f = 0; f < frames; ++f)
            if (consumed - lead + f >= 0)
                std::copy_n(
                    all_mel.data() + static_cast<size_t>(consumed - lead + f) * fe.n_mels,
                    fe.n_mels, mel.data() + static_cast<size_t>(f) * fe.n_mels);
        std::vector<float> mask(enc.cache_left_ctx + enc.cache_chunk_frames, 0.0f);
        std::fill_n(mask.begin(), std::max(0, enc.cache_left_ctx - cached), -1e9f);
        std::vector<float> projected;
        int count = 0;
        model->encode_cache_aware(
            state, mel.data(), frames, mask.data(), static_cast<int>(mask.size()), projected, count,
            prompt);
        require(count == right + 1, "wrong reference encoder frame count");
        reference->step(
            projected.data(), model->rnnt_config().joint_dim, count,
            static_cast<int64_t>(chunk) * (right + 1));
        consumed += chunk == 0 ? first_mel : new_mel;
        cached = std::min(enc.cache_left_ctx, cached + count);

        const size_t begin = chunk == 0 ? 0 : first_samples + (chunk - 1) * stride_samples;
        runner.feed_audio(audio.data() + begin, available - begin);
        const auto update = runner.step();
        require(runner.chunks_processed() == chunk + 1, "runner did not consume one chunk");
        const double decoded_sec = static_cast<double>(enc.cache_decoded_mel_frames(
                                       static_cast<int64_t>(chunk + 1) * (right + 1))) *
                                   hop / fe.sample_rate;
        require(
            std::fabs(update.audio_processed_sec - decoded_sec) < 1e-6,
            "audio_processed is not the decoded mel frontier");
        const auto& expected = reference->word_timings();
        require(update.words.size() == expected.size(), "runner and reference word counts differ");
        if (!expected.empty()) {
            require(
                update.words.front().word == expected.front().word &&
                    update.words.front().start_frame == expected.front().start_frame &&
                    update.words.front().end_frame == expected.front().end_frame,
                "first word is not at the independently derived emission frame");
            std::fprintf(
                stdout, "[PASS] R=%d first word '%s' at encoder frame %lld (chunk %d)\n", right,
                expected.front().word.c_str(), static_cast<long long>(expected.front().start_frame),
                chunk);
            compared = true;
        }
    }
}

// End-of-stream flush words must stay within the audio on every path that
// reaches the SDK (PR #46 report: 0.64 s of JFK at R13 timed "And" at 1.12 s).
static void
check_eof_word_times(const RecognizerConfig& cfg, int right) {
    const auto full = jfk_audio();
    const std::vector<float> audio(full.begin(), full.begin() + 10240);  // 0.64 s
    std::vector<float> audio_8k;                                         // the same audio at 8 kHz
    const std::vector<float>* inputs[] = {&audio, &audio_8k};
    for (size_t i = 0; i < audio.size(); i += 2) audio_8k.push_back(audio[i]);
    Recognizer recognizer(cfg);
    AsrRequestOptions opts;
    opts.enable_word_time_offsets = true;
    auto check_words = [&](const Result& result, int32_t limit_ms, const char* what) {
        require(!result.alternatives.empty(), what);
        int32_t previous = 0;
        for (const auto& word : result.alternatives.front().words) {
            if (!(word.start_time >= previous && word.start_time < word.end_time &&
                  word.end_time <= limit_ms)) {
                std::fprintf(
                    stderr, "%s: '%s' %d-%d ms, audio %d ms\n", what, word.word.c_str(),
                    word.start_time, word.end_time, limit_ms);
                throw std::runtime_error("word time outside the audio");
            }
            previous = word.start_time;
        }
        return result.alternatives.front().words.size();
    };
    size_t streamed_words = 0;
    for (const auto* input : inputs) {
        const int rate = input == &audio ? 16000 : 8000;
        auto stream = recognizer.streaming_recognize(opts, "auto");
        stream->set_interim_words(true);
        for (size_t off = 0; off < input->size(); off += rate / 10) {
            stream->push(
                input->data() + off, std::min<size_t>(rate / 10, input->size() - off), rate);
            const int32_t pushed_ms = static_cast<int32_t>(
                std::min(input->size(), off + rate / 10) * 1000 / static_cast<size_t>(rate));
            while (auto result = stream->next()) {
                check_words(*result, pushed_ms, "interim/EOU word");
                require(result->audio_processed * 1000 <= pushed_ms, "progress past pushed audio");
                if (!result->is_final)
                    break;
            }
        }
        const auto final = stream->finish();
        const size_t words = check_words(final, 640, "end-of-stream word");
        require(std::fabs(final.audio_processed - 0.64f) < 1e-6f, "final progress is not 0.64 s");
        if (rate == 16000)
            streamed_words = words;
        check_words(
            recognizer.recognize(input->data(), input->size(), opts, "auto", rate), 640,
            "offline word");
    }
    // The flush must keep the tail word, only its time moves onto real audio.
    if (right == 13)
        require(streamed_words > 0, "end-of-stream flush lost the tail word");
    std::fprintf(
        stdout, "[PASS] R=%d end-of-stream words within 0.64 s (%zu streamed)\n", right,
        streamed_words);
}

static void
check_model(const std::string& path, int gpu, int right) {
    RecognizerConfig cfg;
    cfg.model.path = path;
    cfg.backend.gpu = gpu;
    cfg.streaming.rnnt_right_context = right;
    cfg.log_status = false;
    Recognizer recognizer(cfg);
    auto* model = dynamic_cast<RnntModel*>(recognizer.model());
    require(model != nullptr, "expected cache-aware RNNT model");
    const auto enc = make_cache_aware_config(model->encoder_config(), right);
    const auto& fe = model->fe_config();
    require(
        std::fabs(
            recognizer.enc_frame_center_offset_ms() -
            (1 - enc.subsampling_factor) * recognizer.ms_per_enc_frame() / enc.subsampling_factor) <
            1e-9,
        "word attribution must use the causal frame centre");
    const int first_mel = 1 + enc.subsampling_factor * right;
    const int new_mel = enc.subsampling_factor * (1 + right);
    const int overlap = 9;  // Existing runner policy, not inferred from subsampling.
    const int hop = model->fe().hop_length();
    const size_t first_samples = (first_mel - 1) * hop + fe.n_fft / 2;
    const size_t stride_samples = new_mel * hop;
    std::vector<float> audio(first_samples + 2 * stride_samples);
    for (size_t i = 0; i < audio.size(); ++i)
        audio[i] = 0.08f * std::sin(0.071f * i) + 0.03f * std::cos(0.019f * i);

    const int prompt = model->prompt_index_for_lang("auto");
    CacheStreamRunner runner(model, cfg);
    runner.set_prompt_index(prompt);
    runner.feed_audio(audio.data(), first_samples - 1);
    runner.step();
    require(runner.chunks_processed() == 0, "first chunk ran before its last stable mel frame");
    runner.feed_audio(audio.data() + first_samples - 1, 1);
    runner.step();
    require(
        runner.chunks_processed() == 1, "first chunk incorrectly waits for steady-state length");

    // Independent chunk assembly, including negative history for R=0. Use
    // another cache slot so runner buffer movement cannot affect the oracle.
    std::vector<float> all_mel;
    std::vector<float> initial_audio(audio.begin(), audio.begin() + first_samples - 1);
    produce_new_mel_frames(model->fe(), initial_audio, 0, 0, all_mel);
    auto state = model->make_cache_state();
    int consumed = 0;
    int cached = 0;
    std::vector<float> first_output;
    for (int step = 0; step < 3; ++step) {
        std::fprintf(stderr, "[check] R=%d chunk=%d\n", right, step);
        // Match the available audio frontier, not future audio: the first
        // reflected STFT frame of a minimal R=0 packet must use the same
        // boundary in the runner and in this chunk-assembly reference.
        const size_t available = first_samples + step * stride_samples;
        std::vector<float> prefix_audio(audio.begin(), audio.begin() + available);
        std::vector<float> new_features;
        produce_new_mel_frames(
            model->fe(), prefix_audio, 0, static_cast<int64_t>(all_mel.size() / fe.n_mels),
            new_features);
        all_mel.insert(all_mel.end(), new_features.begin(), new_features.end());
        const int prefix = step == 0 ? 0 : overlap;
        const int fresh = step == 0 ? first_mel : new_mel;
        const int frames = prefix + fresh;
        std::vector<float> chunk(static_cast<size_t>(frames) * fe.n_mels, 0.0f);
        for (int f = 0; f < frames; ++f) {
            const int source = consumed - prefix + f;
            if (source >= 0) {
                require(
                    static_cast<size_t>(source + 1) * fe.n_mels <= all_mel.size(),
                    "reference mel underrun");
                std::copy_n(
                    all_mel.data() + static_cast<size_t>(source) * fe.n_mels, fe.n_mels,
                    chunk.data() + static_cast<size_t>(f) * fe.n_mels);
            }
        }
        std::vector<float> mask(enc.cache_left_ctx + enc.cache_chunk_frames, 0.0f);
        std::fill_n(mask.begin(), std::max(0, enc.cache_left_ctx - cached), -1e9f);
        std::vector<float> expected;
        int expected_t = 0;
        model->encode_cache_aware(
            state, chunk.data(), frames, mask.data(), static_cast<int>(mask.size()), expected,
            expected_t, prompt);
        require(expected_t == right + 1, "wrong reference encoder frame count");
        if (step > 0) {
            runner.feed_audio(
                audio.data() + first_samples + (step - 1) * stride_samples, stride_samples);
            runner.step();
        }
        std::vector<float> actual;
        int actual_t = 0, dim = 0;
        runner.take_encoder_frames(actual, actual_t, dim);
        require(actual_t == expected_t, "wrong runner encoder frame count");
        require(runner.chunks_processed() == step + 1, "wrong chunk consumption");
        const auto emitted = static_cast<int64_t>(runner.rnnt_decode_stats().encoder_frames);
        require(
            emitted == static_cast<int64_t>(step + 1) * (right + 1),
            "first/steady transition changed the absolute decoder frame offset");
        require(
            enc.cache_decoded_mel_frames(emitted) == consumed + fresh,
            "decoded frontier differs from independently assembled real Mel coverage");
        same_encoder(actual, expected);
        if (step == 0)
            first_output = actual;
        consumed += fresh;
        cached = std::min(enc.cache_left_ctx, cached + expected_t);
    }
    runner.reset();
    require(
        runner.rnnt_decode_stats().encoder_frames == 0,
        "new stream retained the previous decoder frame clock");
    // Preserve FE packet shapes as well: Metal FFT rounding can differ across
    // packet shapes independently of the encoder's chunk geometry.
    runner.feed_audio(audio.data(), first_samples - 1);
    runner.step();
    runner.feed_audio(audio.data() + first_samples - 1, 1);
    runner.step();
    require(runner.chunks_processed() == 1, "reset did not restore first-chunk geometry");
    std::vector<float> reset_output;
    int reset_t = 0, reset_dim = 0;
    runner.take_encoder_frames(reset_output, reset_t, reset_dim);
    same_encoder(first_output, reset_output);

    auto decode = [&](size_t size, size_t packet, bool drain) {
        runner.reset();
        std::vector<int> ids;
        for (size_t off = 0; off < size;) {
            const size_t count = std::min(packet, size - off);
            runner.feed_audio(audio.data() + off, count);
            off += count;
            if (drain) {
                const auto update = runner.step();
                ids.insert(ids.end(), update.new_token_ids.begin(), update.new_token_ids.end());
            }
        }
        const auto final = runner.finalize();
        ids.insert(ids.end(), final.new_token_ids.begin(), final.new_token_ids.end());
        require(final.is_final, "EOF not final");
        require(runner.finalize().new_token_ids.empty(), "EOF duplicated tokens");
        return ids;
    };
    for (size_t size :
         {size_t(0), size_t(1), size_t(159), first_samples - 1, first_samples, first_samples + 1,
          audio.size()}) {
        const auto expected = decode(size, audio.size(), false);
        require(decode(size, 1279, true) == expected, "odd packet EOF differs");
        require(decode(size, 1600, true) == expected, "100ms packet EOF differs");
    }
    // EOU publishes text without ending the acoustic stream. Preserve the
    // first chunk's cache and require steady-state geometry after the event.
    // Use identical FE packet shapes for the EOU and no-EOU controls: otherwise
    // Metal FFT rounding differences would test packetization, not the boundary.
    auto after_boundary = [&](bool force) {
        runner.reset();
        runner.feed_audio(audio.data(), first_samples - 1);
        runner.step();
        runner.feed_audio(audio.data() + first_samples - 1, 1);
        if (force)
            runner.force_eou();
        require(runner.step().is_final == force, "unexpected forced EOU result");
        require(runner.cache_filled_frames() == right + 1, "EOU changed encoder cache");
        const int before = runner.chunks_processed();
        require(before == 1, "EOU synthesized extra chunks");
        require(
            runner.rnnt_decode_stats().encoder_frames == static_cast<uint64_t>(right + 1),
            "EOU reset the absolute decoder clock after the first chunk");
        runner.feed_audio(audio.data() + first_samples, stride_samples - 1);
        runner.step();
        require(runner.chunks_processed() == before, "post-EOU reused first-chunk geometry");
        runner.feed_audio(audio.data() + first_samples + stride_samples - 1, 1);
        runner.step();
        require(runner.chunks_processed() == before + 1, "post-EOU steady chunk was not processed");
        require(
            runner.rnnt_decode_stats().encoder_frames == static_cast<uint64_t>(2 * (right + 1)),
            "post-EOU decoder clock did not retain the first chunk offset");
        std::vector<float> output;
        int count = 0, dim = 0;
        runner.take_encoder_frames(output, count, dim);
        return output;
    };
    const auto no_eou = after_boundary(false);
    same_encoder(no_eou, after_boundary(true));
    std::fprintf(stdout, "[PASS] R=%d first/history encoder, reset, EOF and EOU\n", right);
    check_first_word_frame(cfg, model, right);
    check_eof_word_times(cfg, right);
}

int
main(int argc, char** argv) try {
    check_shapes();
    check_frame_centers();
    if (argc > 1) {
        int gpu = -1;
        if (argc == 4 && std::string(argv[2]) == "--gpu")
            gpu = std::stoi(argv[3]);
        else if (argc != 2)
            throw std::runtime_error("Usage: test_cache_stream_geometry [MODEL.gguf [--gpu N]]");
        for (int right : {0, 3, 6, 13}) check_model(argv[1], gpu, right);
    } else {
        std::fprintf(stdout, "[SKIP] model checks: no cache-aware RNNT model supplied\n");
    }
    std::fprintf(stdout, "[PASS] cache-aware chunk geometry\n");
    return 0;
}
catch (const std::exception& error) {
    std::fprintf(stderr, "[FAIL] %s\n", error.what());
    return 1;
}
