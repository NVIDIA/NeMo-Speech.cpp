// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Endpointing tests.
//
//   1. Policy unit test (always runs, no model): drive VadEndpointer with a
//      synthetic speech/silence timeline and assert it fires once at the
//      threshold, re-arms on new speech, honours force(), and stays silent when
//      disabled.
//   2. Integration (opt-in, needs a model + WAV): build a [utterance | silence |
//      utterance] clip, stream it through the runner with endpointing enabled,
//      and assert >= 2 finals with the right split. Token-silence mode by
//      default; pass --vad-model to also exercise the VAD-driven path.
//
// Usage: ./test_endpointer [<model.gguf> <audio.wav> [--gpu N] [--vad-model F]
//                           [--chunk-ms N] [--gap-ms N] [--eou-ms N]]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "fe.h"
#include "model.h"
#include "recognizer.h"  // RecognizerConfig
#include "runner.h"
#include "runtime.h"
#include "vad_endpointer.h"

using namespace nemo_speech::asr;

static int g_fail = 0;
static void
check(bool ok, const char* what) {
    std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        g_fail++;
}

static bool
starts_with_attach_punctuation(const std::string& text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return false;
    if (std::string(".,!?;:%)]}").find(text[first]) != std::string::npos)
        return true;
    const std::string tail = text.substr(first);
    return tail.rfind("\xE0\xA5\xA4", 0) == 0 || tail.rfind("\xE0\xA5\xA5", 0) == 0 ||
           tail.rfind("\xE3\x80\x82", 0) == 0 || tail.rfind("\xEF\xBC\x81", 0) == 0 ||
           tail.rfind("\xEF\xBC\x9F", 0) == 0;
}

// ---------------------------------------------------------------------------
// 1. Policy unit test (model-free).
// ---------------------------------------------------------------------------
static void
test_policy() {
    // Threshold 800 ms (riva default). The runner feeds (now_ms, last_speech_ms).
    {
        VadEndpointer ep(
            VadEndpointerCfg{/*enable=*/true, /*vad_based=*/true,
                             /*stop_history_eou_ms=*/800.0f});
        // Speech up to 1000 ms, then silence. Poll every 100 ms.
        bool fired_before = false, fired_at = false, fired_after = false;
        for (double now = 1000.0; now < 1800.0; now += 100.0)  // 0..800 ms silence
            fired_before |= ep.poll(now, 1000.0);
        fired_at = ep.poll(1800.0, 1000.0);                    // exactly 800 ms of trailing silence
        for (double now = 1900.0; now < 3000.0; now += 100.0)  // keep polling past threshold
            fired_after |= ep.poll(now, 1000.0);
        check(!fired_before, "policy: does not fire before stop_history_eou_ms");
        check(fired_at, "policy: fires exactly at stop_history_eou_ms");
        check(!fired_after, "policy: latches - fires once per silence gap");
    }
    // Re-arm: after firing, new speech must re-enable a second EOU.
    {
        VadEndpointer ep(VadEndpointerCfg{true, true, 800.0f});
        check(ep.poll(1800.0, 1000.0), "policy: first EOU fires");  // 800 ms silence
        // New speech to 2000 ms, then another 800 ms gap.
        check(!ep.poll(2050.0, 2000.0), "policy: re-armed, not yet fired");
        bool second = false;
        for (double now = 2100.0; now <= 2900.0; now += 100.0) second |= ep.poll(now, 2000.0);
        check(second, "policy: second EOU fires after speech resumes");
    }
    // Disabled: never fires automatically.
    {
        VadEndpointer ep(VadEndpointerCfg{/*enable=*/false, true, 800.0f});
        bool any = false;
        for (double now = 1000.0; now <= 5000.0; now += 100.0) any |= ep.poll(now, 1000.0);
        check(!any, "policy: disabled never fires automatically");
    }
    // force(): fires on the next poll regardless of enable / silence.
    {
        VadEndpointer ep(VadEndpointerCfg{/*enable=*/false, true, 800.0f});
        ep.force();
        check(ep.poll(1000.0, 1000.0), "policy: force() fires immediately");
        check(!ep.poll(1000.0, 1000.0), "policy: force() is one-shot");
    }
    // Threshold override: set, then clear (<= 0 restores the server cfg).
    {
        VadEndpointer ep(VadEndpointerCfg{true, true, 800.0f});
        ep.set_stop_history_eou_ms(300.0f);
        check(!ep.poll(1200.0, 1000.0), "policy: override - 200ms < 300ms no fire");
        check(ep.poll(1300.0, 1000.0), "policy: override - 300ms fires");
        ep.set_stop_history_eou_ms(-1.0f);  // next request leaves it default
        check(!ep.poll(2500.0, 2000.0), "policy: cleared override - 500ms no fire");
        check(ep.poll(2800.0, 2000.0), "policy: cleared override - 800ms fires");
    }
}

// ---------------------------------------------------------------------------
// 2. Integration: stream a 2-utterance clip, count finals.
// ---------------------------------------------------------------------------
static int
test_integration(int argc, char** argv) {
    const std::string model_path = argv[1];
    const std::string audio_path = argv[2];
    int gpu = 0, chunk_ms = 160, gap_ms = 1500, eou_ms = 800;
    std::string vad_model;
    bool no_masking = false;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            gpu = std::atoi(argv[++i]);
        else if (a == "--vad-model" && i + 1 < argc)
            vad_model = argv[++i];
        else if (a == "--no-masking")
            no_masking = true;  // VAD loaded for endpointing only (decoupling test)
        else if (a == "--chunk-ms" && i + 1 < argc)
            chunk_ms = std::atoi(argv[++i]);
        else if (a == "--gap-ms" && i + 1 < argc)
            gap_ms = std::atoi(argv[++i]);
        else if (a == "--eou-ms" && i + 1 < argc)
            eou_ms = std::atoi(argv[++i]);
    }

    ggml_runtime::Params p;
    p.use_gpu = (gpu >= 0);
    p.gpu_device_idx = std::max(gpu, 0);
    p.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(p);
    auto model = AsrModel::load(bm, model_path);

    std::vector<float> one;
    int sr = 0;
    if (!read_wav_mono_16k(audio_path, one, sr)) {
        std::fprintf(stderr, "failed to read WAV: %s\n", audio_path.c_str());
        return 2;
    }
    // Build [utterance | gap silence | utterance].
    std::vector<float> audio = one;
    audio.insert(audio.end(), static_cast<size_t>(gap_ms) * sr / 1000, 0.0f);
    audio.insert(audio.end(), one.begin(), one.end());
    std::fprintf(
        stdout, "[integ] 2-utterance clip: %.2fs (gap %dms), eou=%dms, vad=%s\n",
        static_cast<double>(audio.size()) / sr, gap_ms, eou_ms,
        vad_model.empty() ? "token-silence" : "vad-driven");

    VadEndpointerCfg ep;
    ep.enable = true;
    ep.stop_history_eou_ms = static_cast<float>(eou_ms);
    ep.vad_based = !vad_model.empty();

    RecognizerConfig cfg;
    cfg.vad.model_path = vad_model;
    cfg.vad.masker.mask_enable = !no_masking;
    cfg.endpointing = ep;
    cfg.streaming.rnnt_right_context = 1;  // R=1 for RNNT; the CTC runner ignores it

    std::unique_ptr<AsrRunner> runner;
    if (model->head_kind() == HeadKind::Rnnt) {
        runner = std::make_unique<CacheStreamRunner>(static_cast<RnntModel*>(model.get()), cfg);
    } else {
        runner = std::make_unique<BufferedStreamRunner>(static_cast<CtcModel*>(model.get()), cfg);
    }
    AsrRequestOptions request_options;
    request_options.enable_word_time_offsets = true;
    runner->set_request_options(request_options);

    const size_t chunk_samples = static_cast<size_t>(chunk_ms) * sr / 1000;

    // force_eou (riva runtime_config["force_eou"]) with endpointing disabled:
    // force fires regardless of `enable`, and no natural EOU can race the
    // probe. Feed the first 4 s, latch a forced EOU, and expect an immediate
    // final carrying the partial transcript.
    {
        VadEndpointerCfg ep_off = ep;
        ep_off.enable = false;
        RecognizerConfig pcfg;
        pcfg.endpointing = ep_off;
        pcfg.streaming.rnnt_right_context = 1;  // R=1 for RNNT; the CTC runner ignores it
        std::unique_ptr<AsrRunner> probe;
        if (model->head_kind() == HeadKind::Rnnt) {
            probe = std::make_unique<CacheStreamRunner>(static_cast<RnntModel*>(model.get()), pcfg);
        } else {
            probe =
                std::make_unique<BufferedStreamRunner>(static_cast<CtcModel*>(model.get()), pcfg);
        }
        const size_t head = std::min(audio.size(), static_cast<size_t>(4) * sr);
        for (size_t off = 0; off < head; off += chunk_samples) {
            const size_t n = std::min(chunk_samples, head - off);
            probe->feed_audio(audio.data() + off, n);
            probe->step();
        }
        probe->force_eou();
        auto fu = probe->step();
        std::fprintf(
            stdout, "[integ] forced final @ %.2fs: '%s'\n", fu.audio_processed_sec,
            fu.transcript_so_far.c_str());
        check(fu.is_final, "integration: force_eou yields an immediate final (endpointing off)");
        check(!fu.transcript_so_far.empty(), "integration: forced final carries the transcript");
    }

    std::vector<std::string> finals;
    std::vector<int64_t> first_word_frames;
    const auto record_final = [&](const StreamingUpdate& update) {
        finals.push_back(update.transcript_so_far);
        if (!update.words.empty())
            first_word_frames.push_back(update.words.front().start_frame);
    };
    for (size_t off = 0; off < audio.size(); off += chunk_samples) {
        const size_t n = std::min(chunk_samples, audio.size() - off);
        runner->feed_audio(audio.data() + off, n);
        // Drain finals like the service does: an EOU breaks the runner's chunk
        // loop, so re-step until the update is non-final.
        auto u = runner->step();
        while (u.is_final) {
            std::fprintf(
                stdout, "[integ] EOU final @ %.2fs: '%s'\n", u.audio_processed_sec,
                u.transcript_so_far.c_str());
            record_final(u);
            u = runner->step();
        }
    }
    auto fin = runner->finalize();
    if (!fin.transcript_so_far.empty() || finals.empty()) {
        std::fprintf(stdout, "[integ] end final: '%s'\n", fin.transcript_so_far.c_str());
        record_final(fin);
    }

    // We expect at least one mid-stream EOU (the gap) plus the end-of-stream
    // final = >= 2 finals, each non-empty (the same utterance twice).
    int non_empty = 0;
    int leading_punctuation = 0;
    for (const auto& f : finals)
        if (!f.empty()) {
            non_empty++;
            leading_punctuation += starts_with_attach_punctuation(f) ? 1 : 0;
        }
    std::fprintf(stdout, "[integ] %zu finals (%d non-empty)\n", finals.size(), non_empty);
    check(finals.size() >= 2, "integration: >= 2 finals (mid-stream EOU + end)");
    check(non_empty >= 2, "integration: both utterances transcribed");
    check(
        leading_punctuation == 0,
        "integration: terminal punctuation does not leak into the next final");
    check(
        first_word_frames.size() >= 2 &&
            std::is_sorted(first_word_frames.begin(), first_word_frames.end()),
        "integration: word timestamps remain absolute across endpoint resets");
    return 0;
}

int
main(int argc, char** argv) {
    test_policy();
    if (argc >= 3) {
        // A non-zero return is an early failure (bad model/WAV) that ran no
        // check() - count it so the binary can't print ALL PASS on a broken
        // fixture.
        if (test_integration(argc, argv) != 0)
            g_fail++;
    } else {
        std::fprintf(stdout, "[SKIP] integration (pass <model.gguf> <audio.wav> to enable)\n");
    }
    std::fprintf(stdout, g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
