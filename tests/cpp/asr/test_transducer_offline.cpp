// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Full-utterance RNNT/TDT smoke + dynamic-batch parity test.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "fe.h"
#include "model.h"
#include "runtime.h"

using namespace nemo_speech::asr;

struct DecodeResult {
    std::vector<int> ids;
    std::string text;
    std::vector<float> enc;
    int T = 0;
};

static size_t
edit_distance(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<size_t> row(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) row[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        size_t diagonal = row[0];
        row[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t above = row[j];
            row[j] = std::min(
                {row[j] + 1, row[j - 1] + 1, diagonal + static_cast<size_t>(a[i - 1] != b[j - 1])});
            diagonal = above;
        }
    }
    return row.back();
}

static DecodeResult
run_offline(RnntModel& model, const std::vector<float>& audio, int prompt_index) {
    std::vector<float> enc;
    int T = 0;
    model.infer_offline(audio.data(), audio.size(), enc, T, prompt_index);
    auto decoder = model.make_transducer_decoder();
    const auto ids = decoder->step(enc.data(), model.rnnt_config().joint_dim, T, 0);
    decoder->finalize();
    return {ids, detokenize_sentencepiece(ids, model.vocab()), std::move(enc), T};
}

int
main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr, "Usage: %s <rnnt-or-tdt.gguf> <audio.wav> [--gpu N] [--language CODE]\n",
            argv[0]);
        return 1;
    }
    int gpu = 0;
    std::string language;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (arg == "--language" && i + 1 < argc)
            language = argv[++i];
    }

    ggml_runtime::Params params;
    params.use_gpu = gpu >= 0;
    params.gpu_device_idx = std::max(0, gpu);
    params.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager backend(params);
    BatchingConfig batching;
    batching.enabled = true;
    batching.max_batch_size = 4;
    batching.max_queue_delay_us = 20000;
    batching.state_arena_slots = 8;
    auto base = AsrModel::load(backend, argv[1], batching);
    if (base->head_kind() == HeadKind::Ctc) {
        std::fprintf(stderr, "[FAIL] expected a transducer model\n");
        return 2;
    }
    auto& model = static_cast<RnntModel&>(*base);
    std::vector<float> audio;
    int sample_rate = 0;
    if (!read_wav_mono_16k(argv[2], audio, sample_rate) || sample_rate != model.sample_rate()) {
        std::fprintf(stderr, "[FAIL] could not load compatible WAV\n");
        return 2;
    }
    const int prompt = model.prompt_index_for_lang(language);
    const auto reference = run_offline(model, audio, prompt);
    std::fprintf(
        stderr, "[offline %s reference] %s\n", model.head_kind() == HeadKind::Tdt ? "TDT" : "RNNT",
        reference.text.c_str());

    // Isolate the acoustic batch graph from frontend and decoder effects by
    // feeding the exact same precomputed mel tensor to B=1 and B=4.
    std::vector<float> mel;
    int mel_frames = 0;
    model.fe().compute(audio.data(), audio.size(), mel, mel_frames);
    std::vector<float> mel_reference;
    int mel_reference_T = 0;
    model.infer_offline_from_mel(mel.data(), mel_frames, mel_reference, mel_reference_T, prompt);
    std::atomic<int> enc_ready{0};
    std::atomic<bool> enc_go{false};
    std::vector<std::future<std::pair<std::vector<float>, int>>> encoder_calls;
    for (int b = 0; b < 4; ++b) {
        encoder_calls.push_back(std::async(std::launch::async, [&] {
            enc_ready.fetch_add(1);
            while (!enc_go.load()) std::this_thread::yield();
            std::pair<std::vector<float>, int> result;
            model.infer_offline_from_mel(
                mel.data(), mel_frames, result.first, result.second, prompt);
            return result;
        }));
    }
    while (enc_ready.load() != 4) std::this_thread::yield();
    enc_go.store(true);
    float direct_encoder_error = 0.0f;
    for (auto& call : encoder_calls) {
        const auto result = call.get();
        if (result.second != mel_reference_T || result.first.size() != mel_reference.size()) {
            std::fprintf(stderr, "[FAIL] direct offline encoder shape mismatch\n");
            return 3;
        }
        for (size_t i = 0; i < result.first.size(); ++i)
            direct_encoder_error =
                std::max(direct_encoder_error, std::fabs(result.first[i] - mel_reference[i]));
    }
    std::fprintf(stderr, "[offline encoder direct] B=4 max abs error=%.8g\n", direct_encoder_error);
    if (!std::isfinite(direct_encoder_error) || direct_encoder_error > 0.5f) {
        std::fprintf(stderr, "[FAIL] offline encoder batch drift exceeds Q8 tolerance\n");
        return 3;
    }

    constexpr int B = 4;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::future<DecodeResult>> calls;
    for (int b = 0; b < B; ++b) {
        calls.push_back(std::async(std::launch::async, [&] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            return run_offline(model, audio, prompt);
        }));
    }
    while (ready.load() != B) std::this_thread::yield();
    go.store(true);
    float max_encoder_error = 0.0f;
    size_t max_token_edits = 0;
    std::vector<int> first_batched_ids;
    int item = 0;
    for (auto& call : calls) {
        const auto result = call.get();
        if (result.T != reference.T || result.enc.size() != reference.enc.size()) {
            std::fprintf(stderr, "[FAIL] full-utterance B=4 encoder shape mismatch\n");
            return 3;
        }
        for (size_t i = 0; i < result.enc.size(); ++i)
            max_encoder_error =
                std::max(max_encoder_error, std::fabs(result.enc[i] - reference.enc[i]));
        if (item == 0)
            first_batched_ids = result.ids;
        else if (result.ids != first_batched_ids) {
            std::fprintf(stderr, "[FAIL] B=4 peers produced non-deterministic token sequences\n");
            return 3;
        }
        const size_t edits = edit_distance(reference.ids, result.ids);
        max_token_edits = std::max(max_token_edits, edits);
        const size_t edit_limit = std::max<size_t>(2, (reference.ids.size() + 19) / 20);
        if (edits > edit_limit) {
            std::fprintf(
                stderr,
                "[FAIL] full-utterance B=4 decode drift item=%d edits=%zu limit=%zu\n"
                "  encoder max abs error: %.8g\n"
                "  reference: %s\n"
                "  result:    %s\n",
                item, edits, edit_limit, max_encoder_error, reference.text.c_str(),
                result.text.c_str());
            return 3;
        }
        ++item;
    }
    const auto em = model.offline_encoder_batch_metrics();
    const auto fm = model.offline_frontend_batch_metrics();
    std::fprintf(
        stderr,
        "[offline batch] frontend max B=%llu encoder max B=%llu encoder max abs error=%.8g\n",
        static_cast<unsigned long long>(fm.max_observed_batch),
        static_cast<unsigned long long>(em.max_observed_batch), max_encoder_error);
    if (em.max_observed_batch < B ||
        (model.offline_frontend_uses_gpu() && fm.max_observed_batch < B)) {
        std::fprintf(stderr, "[FAIL] offline requests did not batch end to end\n");
        return 4;
    }
    std::fprintf(
        stderr, "[PASS] full-utterance transducer B=4 parity (max token edits=%zu)\n",
        max_token_edits);
    return 0;
}
