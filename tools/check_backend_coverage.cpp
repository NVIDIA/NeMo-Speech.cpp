// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Backend coverage diagnostic: load a GGUF, run one inference step of every
// Session in the ASR pipeline, then dump which ggml backend each compiled
// graph node was assigned to. Flags scheduler fallback (an op the GPU backend
// can't run, silently moved to CPU) which is catastrophic for streaming
// latency: each fallback op forces one GPU<->CPU roundtrip per audio chunk.
//
// Usage:
//   check_backend_coverage <model.gguf> [--gpu N]
//
//   --gpu N    GPU device index (default 0). Use -1 to force pure CPU.
//
// Exit codes:
//   0 - success, no CPU-fallback ops detected (or pure-CPU run).
//   1 - argument error / setup failure.
//   2 - at least one Session ran on a GPU-target config but ended up with
//       one or more ops scheduled on the CPU backend.
//
// The "fallback" check only fires when a non-CPU backend is present in the
// schedule. Pure-CPU runs (--gpu -1) never trip it; their output ends with
// "CPU-only run (no GPU backend to fall back from) ✓".

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "diar/diar_pipeline.h"
#include "model.h"
#include "recognizer.h"  // RecognizerConfig
#include "runner.h"
#include "runtime.h"

using namespace nemo_speech::asr;

namespace {

// True if `out` contains a CPU-fallback ✗ verdict line for a session that had
// any non-CPU backend in its schedule. We can't easily share state with
// Session::dump_schedule (it's intentionally just a text dumper), so we just
// re-scan the captured output. Cheap and unambiguous.
bool
has_cpu_fallback(const std::string& dump) {
    return dump.find("CPU-fallback ops ✗") != std::string::npos;
}

void
dump_one(std::ostream& dest, ggml_runtime::Session* s, const std::string& label) {
    if (!s) {
        dest << "== " << label << " ==\n  <session not present for this model>\n\n";
        return;
    }
    s->dump_schedule(dest, label);
}

}  // namespace

int
main(int argc, char** argv) {
    // Schedule capture is opt-in (it heap-allocates strings per graph node);
    // this tool exists to read it, so turn it on before any Session runs.
    if (std::getenv("NEMO_SPEECH_SCHEDULE_CAPTURE") == nullptr) {
        static char kv[] = "NEMO_SPEECH_SCHEDULE_CAPTURE=1";
        putenv(kv);
    }
    if (argc < 2) {
        std::fprintf(
            stderr,
            "Usage: %s <model.gguf> [--gpu N]\n"
            "  --gpu N   GPU device index (default 0). -1 = pure CPU.\n",
            argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    int gpu = 0;
    std::string diar_path;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc) {
            gpu = std::atoi(argv[++i]);
        } else if (a == "--diar" && i + 1 < argc) {
            diar_path = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 1;
        }
    }

    std::fprintf(stderr, "[coverage] model=%s gpu=%d\n", model_path.c_str(), gpu);

    ggml_runtime::Params bm_params;
    bm_params.use_gpu = (gpu >= 0);
    bm_params.gpu_device_idx = std::max(gpu, 0);
    bm_params.pe_bin_path = const_cast<char*>("");
    ggml_runtime::BackendManager bm(bm_params);
    auto model = AsrModel::load(bm, model_path);
    const bool is_rnnt = (model->head_kind() == HeadKind::Rnnt);
    auto* rnnt = is_rnnt ? static_cast<RnntModel*>(model.get()) : nullptr;
    auto* ctc = is_rnnt ? nullptr : static_cast<CtcModel*>(model.get());
    const int sr = model->sample_rate();

    // ~2 seconds of zero audio. We don't care about the transcript here,
    // just that ggml_backend_sched has assigned every node before we
    // inspect, which requires every Session to actually run() at least once.
    // For the cache-aware CacheStreamRunner this means feeding enough audio
    // to produce at least one chunk (~250–300 ms of audio at the lowest
    // right-context setting we use); 2 s gives plenty of headroom across
    // models / R values.
    std::vector<float> audio(static_cast<size_t>(sr) * 2, 0.0f);

    // ---- Force every Session to run once. ----
    // Running each Session populates ggml_backend_sched's per-node placement,
    // which diagnostic_sessions() then exposes for inspection. CTC: infer_ctc
    // runs the fused encoder+head Session. RNNT: the staged predictor/joint
    // calls run the decoder Session, and the CacheStreamRunner below runs the
    // cache-aware encoder Session (built lazily on the first chunk).
    if (is_rnnt) {
        // Exercise both hot decoder-stage graph shapes. Runs on every backend,
        // CPU included.
        const auto& rcfg = rnnt->rnnt_config();
        std::vector<float> enc_proj(rcfg.joint_dim, 0.0f);
        int32_t best = -1;
        auto state = rnnt->make_rnnt_stream_state();
        rnnt->predict_rnnt(*state, rcfg.blank_id, /*active_bank=*/0);
        rnnt->joint_argmax(*state, enc_proj.data(), rcfg.joint_dim, 1, &best);
        std::fprintf(stderr, "[coverage] RNNT decoder stages ran (argmax=%d)\n", best);
    } else {
        std::vector<float> lp;
        int T_out = 0, n_classes = 0;
        ctc->infer_ctc(audio.data(), audio.size(), lp, T_out, n_classes);
        std::fprintf(
            stderr, "[coverage] encoder+ctc Session ran: T_out=%d n_classes=%d\n", T_out,
            n_classes);
        std::vector<float> mel, best_probs;
        std::vector<int32_t> best_ids;
        int n_mel = 0, T_greedy = 0;
        ctc->fe().compute(audio.data(), audio.size(), mel, n_mel);
        ctc->infer_ctc_greedy_from_mel(mel.data(), n_mel, best_ids, best_probs, T_greedy);
        std::fprintf(stderr, "[coverage] compact greedy CTC ran: T_out=%d\n", T_greedy);
    }

    // CacheStreamRunner has its OWN encoder Session (cache-aware shape, fixed
    // chunk). Only constructed for RNNT models. The model-handle ctor shares
    // the loader and wires the RNNT head, so runner.step() exercises the
    // cache-aware encoder Session plus staged RNNT decode on each chunk - on any
    // backend, CPU included.
    std::unique_ptr<CacheStreamRunner> runner;
    if (is_rnnt) {
        RecognizerConfig cfg;
        cfg.streaming.rnnt_right_context = 1;  // Riva low-latency preset.
        runner = std::make_unique<CacheStreamRunner>(rnnt, cfg);
        if (rnnt->has_prompt()) {
            const auto languages = rnnt->prompt_languages();
            if (!languages.empty())
                runner->set_prompt_index(rnnt->prompt_index_for_lang(languages.front()));
        }
        runner->feed_audio(audio.data(), audio.size());
        (void)runner->step();
        std::fprintf(
            stderr, "[coverage] CacheStreamRunner.session_ ran: chunks=%d\n",
            runner->chunks_processed());
    }

    std::fprintf(stderr, "\n");

    // ---- Dump per-Session schedules and collect verdicts. ----
    // diagnostic_sessions() enumerates every Session the model owns (labeled);
    // each was run above, so their schedules are populated.
    std::ostringstream all;
    for (const auto& ds : model->diagnostic_sessions()) {
        std::ostringstream ss;
        dump_one(ss, ds.session, ds.label);
        all << ss.str();
        std::cout << ss.str() << std::flush;
    }

    // Sortformer diarizer Session (optional sidecar; steady-state shape needs
    // several chunks of audio, so feed 3 s and flush).
    if (!diar_path.empty()) {
        DiarModel diar(bm, diar_path);
        DiarGeometry geo;
        DiarStream stream(diar, geo);
        std::vector<float> daudio(static_cast<size_t>(sr) * 3, 0.0f);
        stream.feed_audio(daudio.data(), daudio.size());
        stream.finish();
        std::fprintf(
            stderr, "[coverage] diarizer Session ran: %ld frames\n", (long)stream.n_frames());
        std::ostringstream ss;
        dump_one(ss, diar.model().session(), "sortformer_diarizer");
        all << ss.str();
        std::cout << ss.str() << std::flush;
    }

    // Aggregate verdict.
    const bool fallback = has_cpu_fallback(all.str());
    if (fallback) {
        std::fprintf(
            stderr, "[coverage] FAIL - one or more GPU-target Sessions have CPU-fallback ops.\n");
        return 2;
    }
    std::fprintf(stderr, "[coverage] OK - no CPU-fallback ops detected.\n");
    return 0;
}
