// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Dynamic-batching load and latency diagnostic for every ASR neural workload.
//
// Modes:
//   stream  complete buffered CTC or cache-aware RNNT recognizer pipeline,
//           paced PCM arrivals;
//           latency is completion minus the chunk's scheduled arrival (so
//           overload/backlog is visible)
//   offline complete CTC/RNNT/TDT Recognize pipeline over an in-memory utterance
//
// Examples:
//   bench_asr_batching stream model.gguf audio.wav --concurrency 1,2,4,8,16
//   bench_asr_batching offline model.gguf audio.wav --reps 3
//   bench_asr_batching stream model.gguf audio.wav --vad-model silero.gguf
//   bench_asr_batching offline model.gguf audio.wav --pnc-model pnc.gguf

// The executable does no file IO inside timed regions.  Use the same binary
// with and without the shim on LD_LIBRARY_PATH for a clean cuBLAS A/B.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(GGML_USE_CUDA)
#include <cuda_profiler_api.h>
#endif

#include "fe.h"
#include "model.h"
#include "nvtx_utils.h"
#include "recognizer.h"
#include "runner.h"
#include "runtime.h"

using namespace nemo_speech::asr;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string mode;
    std::string model;
    std::string audio;
    std::string language;
    std::string json_path;
    std::string vad_model;
    std::string pnc_model;
    std::string diar_model;
    std::string lm_path;
    std::string lexicon_path;
    std::string tokenizer_path;
    std::vector<int> concurrency;
    int gpu = 0;
    int reps = 0;
    int chunk_ms = 160;
    int right_context = 1;
    bool batching = true;
    bool vad_masking = true;
    bool vad_endpointing = false;
    bool acoustic_endpointing = false;
    int max_batch = 8;
    int queue_us = 1000;
    int state_slots = 0;
};

struct StageRow {
    std::string name;
    double mean_batch = 0.0;
    double singleton_pct = 0.0;
    uint64_t batches = 0;
    uint64_t items = 0;
    uint64_t max_batch = 0;
};

struct Row {
    int concurrency = 0;
    size_t samples = 0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double max_ms = 0.0;
    double final_mean_ms = 0.0;
    double final_p95_ms = 0.0;
    double wall_ms = 0.0;
    double compute_wall_ms = 0.0;
    double compute_rtfx = 0.0;
    double items_per_s = 0.0;
    double rtfx = 0.0;
    bool parity = true;
    std::vector<StageRow> stages;
};

struct Gate {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    Clock::time_point start;
};

void
wait_for_gate(Gate& gate, int n) {
    while (gate.ready.load(std::memory_order_acquire) != n) std::this_thread::yield();
    // Leave enough lead time that every worker observes `go` before the first
    // scheduled PCM arrival.  Runner/state construction is already complete.
    gate.start = Clock::now() + std::chrono::milliseconds(20);
    gate.go.store(true, std::memory_order_release);
}

void
arrive_at_gate(Gate& gate) {
    gate.ready.fetch_add(1, std::memory_order_acq_rel);
    while (!gate.go.load(std::memory_order_acquire)) std::this_thread::yield();
}

double
ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

std::vector<int>
parse_concurrency(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const int n = std::atoi(item.c_str());
        if (n > 0 && std::find(out.begin(), out.end(), n) == out.end())
            out.push_back(n);
    }
    if (out.empty())
        throw std::invalid_argument("--concurrency must contain a positive integer");
    return out;
}

void
usage(const char* exe) {
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s stream  <asr.gguf>  <audio.wav> [options]\n"
        "  %s offline <asr.gguf>  <audio.wav> [options]\n"
        "\n"
        "Options: --gpu N --concurrency 1,2,4,8 --reps N --batching 0|1\n"
        "         --max-batch N --queue-us N --state-slots N --chunk-ms N\n"
        "         --right-context N --language CODE --vad-model PATH\n"
        "         --vad-masking 0|1 --vad-endpointing 0|1\n"
        "         --acoustic-endpointing 0|1\n"
        "         --pnc-model PATH --diar-model PATH\n"
        "         --lm-path PATH --lexicon PATH [--tokenizer PATH] --json PATH\n",
        exe, exe);
}

Options
parse_options(int argc, char** argv) {
    if (argc < 3)
        throw std::invalid_argument("missing benchmark mode/model");
    Options o;
    o.mode = argv[1];
    o.model = argv[2];
    int i = 3;
    if (i >= argc)
        throw std::invalid_argument("this mode requires an audio path");
    o.audio = argv[i++];
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--gpu")
            o.gpu = std::atoi(value());
        else if (a == "--concurrency")
            o.concurrency = parse_concurrency(value());
        else if (a == "--reps")
            o.reps = std::atoi(value());
        else if (a == "--batching")
            o.batching = std::atoi(value()) != 0;
        else if (a == "--max-batch")
            o.max_batch = std::atoi(value());
        else if (a == "--queue-us")
            o.queue_us = std::atoi(value());
        else if (a == "--state-slots")
            o.state_slots = std::atoi(value());
        else if (a == "--chunk-ms")
            o.chunk_ms = std::atoi(value());
        else if (a == "--right-context")
            o.right_context = std::atoi(value());
        else if (a == "--language")
            o.language = value();
        else if (a == "--vad-model")
            o.vad_model = value();
        else if (a == "--vad-masking")
            o.vad_masking = std::atoi(value()) != 0;
        else if (a == "--vad-endpointing")
            o.vad_endpointing = std::atoi(value()) != 0;
        else if (a == "--acoustic-endpointing")
            o.acoustic_endpointing = std::atoi(value()) != 0;
        else if (a == "--pnc-model")
            o.pnc_model = value();
        else if (a == "--diar-model")
            o.diar_model = value();
        else if (a == "--lm-path")
            o.lm_path = value();
        else if (a == "--lexicon")
            o.lexicon_path = value();
        else if (a == "--tokenizer")
            o.tokenizer_path = value();
        else if (a == "--json")
            o.json_path = value();
        else
            throw std::invalid_argument("unknown option: " + a);
    }
    if (o.mode != "stream" && o.mode != "offline")
        throw std::invalid_argument("mode must be stream or offline");
    if (o.concurrency.empty()) {
        o.concurrency = o.mode == "stream" ? std::vector<int>{1, 2, 4, 8, 12, 16, 24, 32}
                                           : std::vector<int>{1, 2, 4, 8, 16};
    }
    if (o.reps <= 0)
        o.reps = o.mode == "stream" ? 1 : 3;
    o.max_batch = std::max(1, o.max_batch);
    o.queue_us = std::max(0, o.queue_us);
    o.chunk_ms = std::max(1, o.chunk_ms);
    const int largest = *std::max_element(o.concurrency.begin(), o.concurrency.end());
    if (o.state_slots <= 0)
        o.state_slots = largest;
    if (o.state_slots < largest)
        throw std::invalid_argument("--state-slots is smaller than the concurrency sweep");
    return o;
}

BatchingConfig
batching_config(const Options& o) {
    BatchingConfig c;
    c.enabled = o.batching;
    c.max_batch_size = o.max_batch;
    c.max_queue_delay_us = o.queue_us;
    c.max_queue_depth = std::max(256, 2 * o.state_slots);
    c.state_arena_slots = o.state_slots;
    return c;
}

BatchMetrics
delta(const BatchMetrics& after, const BatchMetrics& before) {
    BatchMetrics d;
    d.batches = after.batches - before.batches;
    d.items = after.items - before.items;
    d.singleton_batches = after.singleton_batches - before.singleton_batches;
    d.max_observed_batch = after.max_observed_batch;
    return d;
}

StageRow
stage_row(std::string name, const BatchMetrics& d) {
    StageRow r;
    r.name = std::move(name);
    r.batches = d.batches;
    r.items = d.items;
    r.max_batch = d.max_observed_batch;
    if (d.batches) {
        r.mean_batch = static_cast<double>(d.items) / static_cast<double>(d.batches);
        r.singleton_pct =
            100.0 * static_cast<double>(d.singleton_batches) / static_cast<double>(d.batches);
    }
    return r;
}

double
percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty())
        return 0.0;
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double f = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - f) + sorted[hi] * f;
}

void
fill_latency(Row& row, std::vector<double> values) {
    row.samples = values.size();
    if (values.empty())
        return;
    double sum = 0.0;
    for (double v : values) sum += v;
    row.mean_ms = sum / static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    row.p50_ms = percentile(values, 0.50);
    row.p95_ms = percentile(values, 0.95);
    row.max_ms = values.back();
}

std::string
json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(static_cast<char>(c));
                break;
        }
    }
    return out;
}

void
emit_results(const Options& o, const std::vector<Row>& rows) {
    std::fprintf(
        stderr, "\n%-6s %9s %9s %9s %9s %10s %10s %11s %9s %8s %7s\n", "conc", "mean_ms", "p50_ms",
        "p95_ms", "max_ms", "items/s", "RTFx", "compute_ms", "final_ms", "parity", "samples");
    for (const Row& r : rows) {
        std::fprintf(
            stderr, "%6d %9.2f %9.2f %9.2f %9.2f %10.2f %10.2f %11.2f %9.2f %8s %7zu\n",
            r.concurrency, r.mean_ms, r.p50_ms, r.p95_ms, r.max_ms, r.items_per_s, r.rtfx,
            r.compute_wall_ms, r.final_mean_ms, r.parity ? "yes" : "NO", r.samples);
        for (const StageRow& s : r.stages) {
            std::fprintf(
                stderr,
                "       %-9s mean_batch=%5.2f max_batch=%llu singleton=%5.1f%% "
                "batches=%llu items=%llu\n",
                s.name.c_str(), s.mean_batch, (unsigned long long)s.max_batch, s.singleton_pct,
                (unsigned long long)s.batches, (unsigned long long)s.items);
        }
    }

    std::ostringstream js;
    js << std::fixed << std::setprecision(6);
    js << "{\"mode\":\"" << json_escape(o.mode) << "\",\"model\":\"" << json_escape(o.model)
       << "\",\"audio\":\"" << json_escape(o.audio) << "\",\"vad_model\":\""
       << json_escape(o.vad_model) << "\",\"pnc_model\":\"" << json_escape(o.pnc_model)
       << "\",\"diar_model\":\"" << json_escape(o.diar_model) << "\",\"lm_path\":\""
       << json_escape(o.lm_path) << "\",\"batching\":" << (o.batching ? "true" : "false")
       << ",\"max_batch\":" << o.max_batch << ",\"queue_us\":" << o.queue_us
       << ",\"chunk_ms\":" << o.chunk_ms << ",\"reps\":" << o.reps << ",\"rows\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i)
            js << ',';
        const Row& r = rows[i];
        js << "{\"concurrency\":" << r.concurrency << ",\"samples\":" << r.samples
           << ",\"mean_ms\":" << r.mean_ms << ",\"p50_ms\":" << r.p50_ms
           << ",\"p95_ms\":" << r.p95_ms << ",\"max_ms\":" << r.max_ms
           << ",\"final_mean_ms\":" << r.final_mean_ms << ",\"final_p95_ms\":" << r.final_p95_ms
           << ",\"wall_ms\":" << r.wall_ms << ",\"compute_wall_ms\":" << r.compute_wall_ms
           << ",\"compute_rtfx\":" << r.compute_rtfx << ",\"items_per_s\":" << r.items_per_s
           << ",\"rtfx\":" << r.rtfx << ",\"parity\":" << (r.parity ? "true" : "false")
           << ",\"stages\":[";
        for (size_t j = 0; j < r.stages.size(); ++j) {
            if (j)
                js << ',';
            const StageRow& s = r.stages[j];
            js << "{\"name\":\"" << json_escape(s.name) << "\",\"mean_batch\":" << s.mean_batch
               << ",\"singleton_pct\":" << s.singleton_pct << ",\"batches\":" << s.batches
               << ",\"items\":" << s.items << ",\"max_batch\":" << s.max_batch << '}';
        }
        js << "]}";
    }
    js << "]}";
    if (!o.json_path.empty()) {
        std::ofstream f(o.json_path, std::ios::binary | std::ios::trunc);
        if (!f)
            throw std::runtime_error("cannot write JSON path: " + o.json_path);
        f << js.str() << '\n';
    }
    std::printf("%s\n", js.str().c_str());
}

struct StreamRun {
    std::vector<double> latencies;
    std::vector<double> final_latencies;
    std::vector<std::string> transcripts;
    double wall_ms = 0.0;
};

StreamRun
run_stream_once(
    Recognizer& recognizer, const AsrRequestOptions& opts, const std::string& language,
    const std::vector<float>& audio, int sr, int chunk_ms, int concurrency, bool paced) {
    const size_t chunk_samples = std::max<size_t>(1, static_cast<size_t>(chunk_ms) * sr / 1000);
    const size_t chunks = (audio.size() + chunk_samples - 1) / chunk_samples;
    Gate gate;
    std::vector<std::vector<double>> times(static_cast<size_t>(concurrency));
    std::vector<double> final_times(static_cast<size_t>(concurrency));
    std::vector<std::string> transcripts(static_cast<size_t>(concurrency));
    std::vector<std::exception_ptr> errors(static_cast<size_t>(concurrency));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(concurrency));
    for (int b = 0; b < concurrency; ++b) {
        threads.emplace_back([&, b] {
            try {
                const bool cohort_enabled = std::getenv("NEMO_SPEECH_BENCH_COHORT") != nullptr;
                std::optional<ScopedBatchCohort> cohort;
                if (cohort_enabled)
                    cohort.emplace(concurrency);
                auto stream = recognizer.streaming_recognize(opts, language);
                times[b].reserve(chunks);
                arrive_at_gate(gate);
                for (size_t k = 0, off = 0; off < audio.size(); ++k, off += chunk_samples) {
                    const auto scheduled = gate.start + std::chrono::milliseconds(chunk_ms * k);
                    if (paced)
                        std::this_thread::sleep_until(scheduled);
                    else if (k == 0)
                        std::this_thread::sleep_until(gate.start);
                    const size_t n = std::min(chunk_samples, audio.size() - off);
                    stream->push(audio.data() + off, n);
                    (void)stream->next();
                    if (paced)
                        times[b].push_back(ms(Clock::now() - scheduled));
                }
                const auto final_begin = Clock::now();
                const auto result = stream->finish();
                if (!result.alternatives.empty())
                    transcripts[b] = result.alternatives.front().transcript;
                // Store finalization separately: it includes the acoustic tail
                // and, when configured, the production postprocessing/PnC path.
                // It is deliberately not part of the steady 160 ms chunk SLA.
                final_times[b] = ms(Clock::now() - final_begin);
            }
            catch (...) {
                errors[b] = std::current_exception();
            }
        });
    }
    wait_for_gate(gate, concurrency);
    for (auto& t : threads) t.join();
    const auto end = Clock::now();
    for (const auto& e : errors)
        if (e)
            std::rethrow_exception(e);
    StreamRun out;
    out.wall_ms = ms(end - gate.start);
    out.transcripts = std::move(transcripts);
    out.final_latencies = std::move(final_times);
    for (auto& v : times) out.latencies.insert(out.latencies.end(), v.begin(), v.end());
    return out;
}

std::vector<Row>
bench_stream(const Options& o, const std::vector<float>& audio, int sr) {
    RecognizerConfig cfg;
    cfg.backend.gpu = o.gpu;
    cfg.batching = batching_config(o);
    cfg.model.path = o.model;
    cfg.streaming.rnnt_right_context = o.right_context;
    cfg.vad.model_path = o.vad_model;
    cfg.vad.masker.mask_enable = !o.vad_model.empty() && o.vad_masking;
    cfg.endpointing.enable = o.acoustic_endpointing || (!o.vad_model.empty() && o.vad_endpointing);
    cfg.endpointing.vad_based = !o.vad_model.empty() && o.vad_endpointing;
    cfg.postproc.pnc_model_path = o.pnc_model;
    cfg.diar.model_path = o.diar_model;
    cfg.decoder.lm_path = o.lm_path;
    cfg.decoder.lexicon_path = o.lexicon_path;
    cfg.decoder.tokenizer_path = o.tokenizer_path;
    if (!o.lm_path.empty())
        cfg.decoder.kind = DecoderConfig::Kind::Flashlight;
    Recognizer recognizer(std::move(cfg));
    auto* model = recognizer.model();
    if (sr != recognizer.sample_rate())
        throw std::runtime_error("audio/model sample-rate mismatch");
    AsrRequestOptions opts;
    // Keep request semantics constant across configurations.  On a
    // self-punctuating RNNT this preserves model formatting; its production
    // postprocessor intentionally ignores an incompatible external PnC model.
    opts.enable_automatic_punctuation = true;
    opts.enable_speaker_diarization = !o.diar_model.empty();

    // B=1 establishes both the transcript reference and all scalar graph
    // caches.  Each sweep point then gets one unpaced shape warmup.
    const auto ref_run =
        run_stream_once(recognizer, opts, o.language, audio, sr, o.chunk_ms, 1, false);
    const std::string reference = ref_run.transcripts.front();
    std::fprintf(stderr, "[reference] %s\n", reference.c_str());

    std::vector<Row> rows;
    for (int n : o.concurrency) {
        // First build every graph shape outside both measurements. Then average
        // compute-only, unpaced passes separately from the paced 160 ms SLA
        // passes. This makes compute_wall_ms directly comparable with engines
        // that expose whole-clip streaming compute rather than paced latency.
        // Pass 1 grows the shared scheduler pool to this concurrency's largest
        // steady-state shape. Pool growth invalidates older cached placements.
        // Pass 2 then rebuilds startup/tail shapes against the final pool
        // generation so the measured utterance reuses their CUDA graphs.
        const std::string warmup_range = "asr.bench.warmup.C" + std::to_string(n);
        {
            const ggml_nvtx::range nvtx(warmup_range.c_str());
            (void)run_stream_once(recognizer, opts, o.language, audio, sr, o.chunk_ms, n, false);
            (void)run_stream_once(recognizer, opts, o.language, audio, sr, o.chunk_ms, n, false);
        }
        const auto fb = model->fe().batch_metrics();
        BatchMetrics eb;
        BatchMetrics pb;
        BatchMetrics jb;
        const auto vb = recognizer.vad_batch_metrics();
        const auto db =
            recognizer.diar_model() ? recognizer.diar_model()->batch_metrics() : BatchMetrics{};
        const auto pnb = recognizer.postproc().pnc_batch_metrics();
        if (model->head_kind() == HeadKind::Ctc) {
            eb = static_cast<CtcModel*>(model)->batch_metrics();
        } else {
            auto* transducer = static_cast<RnntModel*>(model);
            eb = transducer->encoder_batch_metrics();
            pb = transducer->predictor_batch_metrics();
            jb = transducer->joint_batch_metrics();
        }
        Row row;
        row.concurrency = n;
        const std::string compute_range = "asr.bench.compute.C" + std::to_string(n);
        for (int rep = 0; rep < o.reps; ++rep) {
            const ggml_nvtx::range nvtx(compute_range.c_str());
#if defined(GGML_USE_CUDA)
            const bool profile_cuda =
                rep == 0 && std::getenv("NEMO_SPEECH_BENCH_CUDA_PROFILE") != nullptr;
            if (profile_cuda)
                cudaProfilerStart();
#endif
            row.compute_wall_ms +=
                run_stream_once(recognizer, opts, o.language, audio, sr, o.chunk_ms, n, false)
                    .wall_ms;
#if defined(GGML_USE_CUDA)
            if (profile_cuda)
                cudaProfilerStop();
#endif
        }
        row.compute_wall_ms /= o.reps;
        const double audio_s = static_cast<double>(audio.size()) / sr;
        row.compute_rtfx =
            row.compute_wall_ms > 0.0 ? audio_s * n / (row.compute_wall_ms / 1000.0) : 0.0;
        double total_wall = 0.0;
        std::vector<double> all;
        std::vector<double> finals;
        const std::string paced_range = "asr.bench.paced.C" + std::to_string(n);
        for (int rep = 0; rep < o.reps; ++rep) {
            const ggml_nvtx::range nvtx(paced_range.c_str());
            auto run =
                run_stream_once(recognizer, opts, o.language, audio, sr, o.chunk_ms, n, true);
            total_wall += run.wall_ms;
            for (size_t i = 0; i < run.transcripts.size(); ++i) {
                if (run.transcripts[i] != reference) {
                    row.parity = false;
                    std::fprintf(
                        stderr,
                        "[stream mismatch] concurrency=%d item=%zu expected=\"%s\" "
                        "got=\"%s\"\n",
                        n, i, reference.c_str(), run.transcripts[i].c_str());
                }
            }
            all.insert(all.end(), run.latencies.begin(), run.latencies.end());
            finals.insert(finals.end(), run.final_latencies.begin(), run.final_latencies.end());
        }
        row.wall_ms = total_wall / o.reps;
        fill_latency(row, std::move(all));
        if (!finals.empty()) {
            double sum = 0.0;
            for (double v : finals) sum += v;
            row.final_mean_ms = sum / finals.size();
            std::sort(finals.begin(), finals.end());
            row.final_p95_ms = percentile(finals, 0.95);
        }
        const double wall_s = row.wall_ms / 1000.0;
        row.items_per_s = wall_s > 0 ? static_cast<double>(row.samples) / wall_s : 0.0;
        row.rtfx = wall_s > 0 ? audio_s * n / wall_s : 0.0;
        row.stages.push_back(stage_row("frontend", delta(model->fe().batch_metrics(), fb)));
        if (model->head_kind() == HeadKind::Ctc) {
            auto* ctc = static_cast<CtcModel*>(model);
            row.stages.push_back(stage_row("ctc", delta(ctc->batch_metrics(), eb)));
        } else {
            auto* transducer = static_cast<RnntModel*>(model);
            row.stages.push_back(
                stage_row("encoder", delta(transducer->encoder_batch_metrics(), eb)));
            row.stages.push_back(
                stage_row("predictor", delta(transducer->predictor_batch_metrics(), pb)));
            row.stages.push_back(stage_row("joint", delta(transducer->joint_batch_metrics(), jb)));
        }
        if (!o.vad_model.empty())
            row.stages.push_back(stage_row("vad", delta(recognizer.vad_batch_metrics(), vb)));
        if (!o.diar_model.empty()) {
            row.stages.push_back(
                stage_row("diar", delta(recognizer.diar_model()->batch_metrics(), db)));
        }
        if (!o.pnc_model.empty()) {
            row.stages.push_back(
                stage_row("pnc", delta(recognizer.postproc().pnc_batch_metrics(), pnb)));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

struct OfflineRun {
    std::vector<double> latencies;
    std::vector<double> final_latencies;
    std::vector<std::string> transcripts;
    double wall_ms = 0.0;
};

OfflineRun
run_offline_once(
    Recognizer& recognizer, const AsrRequestOptions& opts, const std::string& language,
    const std::vector<float>& audio, int concurrency) {
    Gate gate;
    std::vector<double> times(static_cast<size_t>(concurrency));
    std::vector<double> final_times(static_cast<size_t>(concurrency));
    std::vector<std::string> transcripts(static_cast<size_t>(concurrency));
    std::vector<std::exception_ptr> errors(static_cast<size_t>(concurrency));
    std::vector<std::thread> threads;
    for (int b = 0; b < concurrency; ++b) {
        threads.emplace_back([&, b] {
            try {
                arrive_at_gate(gate);
                std::this_thread::sleep_until(gate.start);
                const auto begin = Clock::now();
                const auto result =
                    recognizer.recognize(audio.data(), audio.size(), opts, language);
                const auto end = Clock::now();
                times[b] = ms(end - begin);
                final_times[b] = times[b];
                if (!result.alternatives.empty())
                    transcripts[b] = result.alternatives.front().transcript;
            }
            catch (...) {
                errors[b] = std::current_exception();
            }
        });
    }
    wait_for_gate(gate, concurrency);
    for (auto& t : threads) t.join();
    const auto end = Clock::now();
    for (const auto& e : errors)
        if (e)
            std::rethrow_exception(e);
    return {std::move(times), std::move(final_times), std::move(transcripts), ms(end - gate.start)};
}

std::vector<Row>
bench_offline(const Options& o, const std::vector<float>& audio, int sr) {
    RecognizerConfig cfg;
    cfg.backend.gpu = o.gpu;
    cfg.batching = batching_config(o);
    cfg.model.path = o.model;
    cfg.vad.model_path = o.vad_model;
    cfg.vad.masker.mask_enable = !o.vad_model.empty() && o.vad_masking;
    cfg.endpointing.enable = o.acoustic_endpointing || (!o.vad_model.empty() && o.vad_endpointing);
    cfg.endpointing.vad_based = !o.vad_model.empty() && o.vad_endpointing;
    cfg.postproc.pnc_model_path = o.pnc_model;
    cfg.diar.model_path = o.diar_model;
    cfg.decoder.lm_path = o.lm_path;
    cfg.decoder.lexicon_path = o.lexicon_path;
    cfg.decoder.tokenizer_path = o.tokenizer_path;
    if (!o.lm_path.empty())
        cfg.decoder.kind = DecoderConfig::Kind::Flashlight;
    Recognizer recognizer(std::move(cfg));
    auto* model = recognizer.model();
    if (sr != recognizer.sample_rate())
        throw std::runtime_error("audio/model sample-rate mismatch");
    AsrRequestOptions opts;
    opts.enable_automatic_punctuation = true;
    opts.enable_speaker_diarization = !o.diar_model.empty();
    auto ref_run = run_offline_once(recognizer, opts, o.language, audio, 1);
    const auto reference = ref_run.transcripts.front();
    std::fprintf(stderr, "[reference] %s\n", reference.c_str());
    std::vector<Row> rows;
    for (int n : o.concurrency) {
        (void)run_offline_once(recognizer, opts, o.language, audio, n);
        BatchMetrics fe_before;
        BatchMetrics enc_before;
        BatchMetrics pred_before;
        BatchMetrics joint_before;
        const auto vad_before = recognizer.vad_batch_metrics();
        const auto diar_before =
            recognizer.diar_model() ? recognizer.diar_model()->batch_metrics() : BatchMetrics{};
        const auto pnc_before = recognizer.postproc().pnc_batch_metrics();
        if (model->head_kind() == HeadKind::Ctc) {
            auto* ctc = static_cast<CtcModel*>(model);
            fe_before = ctc->offline_frontend_batch_metrics();
            enc_before = ctc->batch_metrics();
        } else {
            auto* transducer = static_cast<RnntModel*>(model);
            fe_before = transducer->offline_frontend_batch_metrics();
            enc_before = transducer->offline_encoder_batch_metrics();
            pred_before = transducer->predictor_batch_metrics();
            joint_before = transducer->joint_batch_metrics();
        }
        Row row;
        row.concurrency = n;
        std::vector<double> all;
        std::vector<double> finals;
        for (int rep = 0; rep < o.reps; ++rep) {
            auto run = run_offline_once(recognizer, opts, o.language, audio, n);
            row.wall_ms += run.wall_ms;
            all.insert(all.end(), run.latencies.begin(), run.latencies.end());
            finals.insert(finals.end(), run.final_latencies.begin(), run.final_latencies.end());
            for (size_t i = 0; i < run.transcripts.size(); ++i) {
                if (run.transcripts[i] != reference) {
                    row.parity = false;
                    std::fprintf(
                        stderr,
                        "[pipeline mismatch] concurrency=%d item=%zu expected=\"%s\" "
                        "got=\"%s\"\n",
                        n, i, reference.c_str(), run.transcripts[i].c_str());
                }
            }
        }
        row.wall_ms /= o.reps;
        fill_latency(row, std::move(all));
        if (!finals.empty()) {
            double sum = 0.0;
            for (double v : finals) sum += v;
            row.final_mean_ms = sum / finals.size();
            std::sort(finals.begin(), finals.end());
            row.final_p95_ms = percentile(finals, 0.95);
        }
        const double wall_s = row.wall_ms / 1000.0;
        const double audio_s = static_cast<double>(audio.size()) / sr;
        row.items_per_s = wall_s > 0 ? n / wall_s : 0.0;
        row.rtfx = wall_s > 0 ? audio_s * n / wall_s : 0.0;
        if (model->head_kind() == HeadKind::Ctc) {
            auto* ctc = static_cast<CtcModel*>(model);
            row.stages.push_back(
                stage_row("frontend", delta(ctc->offline_frontend_batch_metrics(), fe_before)));
            row.stages.push_back(stage_row("ctc", delta(ctc->batch_metrics(), enc_before)));
        } else {
            auto* transducer = static_cast<RnntModel*>(model);
            row.stages.push_back(stage_row(
                "frontend", delta(transducer->offline_frontend_batch_metrics(), fe_before)));
            row.stages.push_back(stage_row(
                "encoder", delta(transducer->offline_encoder_batch_metrics(), enc_before)));
            row.stages.push_back(
                stage_row("predictor", delta(transducer->predictor_batch_metrics(), pred_before)));
            row.stages.push_back(
                stage_row("joint", delta(transducer->joint_batch_metrics(), joint_before)));
        }
        if (!o.vad_model.empty()) {
            row.stages.push_back(
                stage_row("vad", delta(recognizer.vad_batch_metrics(), vad_before)));
        }
        if (!o.diar_model.empty()) {
            row.stages.push_back(
                stage_row("diar", delta(recognizer.diar_model()->batch_metrics(), diar_before)));
        }
        if (!o.pnc_model.empty()) {
            row.stages.push_back(
                stage_row("pnc", delta(recognizer.postproc().pnc_batch_metrics(), pnc_before)));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        const Options o = parse_options(argc, argv);
        std::vector<float> audio;
        int sr = 0;
        if (!read_wav_mono_16k(o.audio, audio, sr))
            throw std::runtime_error("failed to read WAV: " + o.audio);

        std::fprintf(
            stderr, "[bench] mode=%s batching=%s max_batch=%d queue_us=%d state_slots=%d reps=%d\n",
            o.mode.c_str(), o.batching ? "on" : "off", o.max_batch, o.queue_us, o.state_slots,
            o.reps);
        std::vector<Row> rows;
        if (o.mode == "stream")
            rows = bench_stream(o, audio, sr);
        else
            rows = bench_offline(o, audio, sr);
        emit_results(o, rows);
        return std::all_of(rows.begin(), rows.end(), [](const Row& r) { return r.parity; }) ? 0 : 3;
    }
    catch (const std::exception& e) {
        usage(argv[0]);
        std::fprintf(stderr, "[FAIL] %s\n", e.what());
        return 2;
    }
}
