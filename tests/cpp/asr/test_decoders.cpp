// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Decoder unit tests - model-free, synthetic inputs. Covers the contracts the
// runners and endpointing depend on:
//
//   GreedyCtcDecoder   collapse rule across chunk boundaries, blank handling,
//                      last_emit_frame exactness, utterance confidence, word
//                      timings, reset_utterance vs reset semantics.
//   RnntGreedyDecoder  scripted MockRnntEngine: emission sequence, blank =
//                      frame advance, max_symbols_per_step cap, LSTM state /
//                      prev_token preserved across reset_utterance (riva EOU).
//
// Usage: ./test_decoders   (no model files, no GPU)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "greedy_ctc_decoder.h"
#include "rnnt_greedy_decoder.h"

using namespace nemo_speech::asr;

static int g_fail = 0;
static void
check(bool ok, const char* what) {
    std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        g_fail++;
}

static bool
near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

// ---------------------------------------------------------------------------
// GreedyCtcDecoder. 4 classes: 0="▁hel", 1="lo", 2="▁world", 3=<blank>.
// Frames are log-prob rows; emit_frame() builds a row whose argmax is `id`
// with softmax probability `p` (the rest share 1-p uniformly).
// ---------------------------------------------------------------------------
namespace {

constexpr int kClasses = 4;
constexpr int kBlank = 3;

std::vector<float>
frame_for(int id, float p = 0.8f) {
    std::vector<float> row(kClasses);
    const float rest = (1.0f - p) / (kClasses - 1);
    for (int c = 0; c < kClasses; c++) row[c] = std::log(c == id ? p : rest);
    return row;
}

std::vector<float>
frames_for(const std::vector<int>& ids, float p = 0.8f) {
    std::vector<float> out;
    for (int id : ids) {
        auto row = frame_for(id, p);
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

GreedyCtcDecoder
make_greedy() {
    CtcConfig cfg;
    cfg.d_model = 8;  // unused by the decoder
    cfg.num_classes = kClasses - 1;
    cfg.blank_id = kBlank;
    return GreedyCtcDecoder(cfg, {"\xE2\x96\x81hel", "lo", "\xE2\x96\x81world"});
}

void
test_sentencepiece_punctuation_detokenization() {
    const std::vector<std::string> vocab = {
        "\xE2\x96\x81"
        "How",
        "\xE2\x96\x81old",
        "\xE2\x96\x81is",
        "\xE2\x96\x81"
        "Brooklyn",
        "\xE2\x96\x81"
        "Bridge",
        "\xE2\x96\x81?",
        "\xE2\x96\x81!",
        "\xE2\x96\x81\xE0\xA5\xA4"};
    check(
        detokenize_sentencepiece({0, 1, 2, 3, 4, 5}, vocab) == "How old is Brooklyn Bridge?",
        "sentencepiece: boundary-prefixed terminator attaches to preceding word");
    check(
        detokenize_sentencepiece({0, 6}, vocab) == "How!" &&
            detokenize_sentencepiece({0, 7}, vocab) == "How\xE0\xA5\xA4",
        "sentencepiece: ASCII and Devanagari terminators attach consistently");

    std::string incremental;
    append_sentencepiece_tokens(incremental, {0, 1}, vocab);
    append_sentencepiece_tokens(incremental, {2, 3, 4}, vocab);
    append_sentencepiece_tokens(incremental, {5}, vocab);
    check(
        incremental == detokenize_sentencepiece({0, 1, 2, 3, 4, 5}, vocab),
        "sentencepiece: incremental detokenization matches full detokenization");
}

void
test_greedy_collapse_and_blank() {
    auto dec = make_greedy();
    // [0, 0, blank, 0] -> repeat collapses, blank separates -> emits {0, 0}.
    auto fr = frames_for({0, 0, kBlank, 0});
    auto ids = dec.step(fr.data(), kClasses, 4, /*frame_offset=*/0);
    check(ids == std::vector<int>({0, 0}), "greedy: collapse rule + blank separation");
}

void
test_greedy_cross_chunk_collapse() {
    auto dec = make_greedy();
    auto a = frames_for({kBlank, 0});  // chunk 1 ends emitting id 0
    auto b = frames_for({0, kBlank});  // chunk 2 starts with the same id
    auto ids1 = dec.step(a.data(), kClasses, 2, 0);
    auto ids2 = dec.step(b.data(), kClasses, 2, 2);
    check(ids1 == std::vector<int>({0}), "greedy: chunk 1 emits once");
    check(ids2.empty(), "greedy: repeated id across chunk boundary collapses");
}

void
test_greedy_seam_dedup() {
    // Buffered-window seam: the same word straddling an emit boundary can
    // spike once per window with a blank between (each window encodes the
    // seam with slightly different context). A window-leading repeat of the
    // last committed token within kSeamMergeMaxGap(=2) frames is suppressed.
    {
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank});  // window 1: spike at frame 0
        auto b = frames_for({0, kBlank});  // window 2: re-spike at frame 2
        auto ids1 = dec.step(a.data(), kClasses, 2, 0);
        auto ids2 = dec.step(b.data(), kClasses, 2, 2);
        check(ids1 == std::vector<int>({0}), "greedy: seam window 1 emits once");
        check(ids2.empty(), "greedy: seam re-spike within gap suppressed");
        // Window 3 re-spikes again: the suppressed frame extended the
        // committed token's reach, so the chain keeps deduping.
        auto ids3 = dec.step(b.data(), kClasses, 2, 4);
        check(ids3.empty(), "greedy: chained seam re-spike suppressed");
    }
    {
        // A genuine blank-separated repeat beyond the seam gap still emits.
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank, kBlank, kBlank});
        auto b = frames_for({0, kBlank});
        dec.step(a.data(), kClasses, 4, 0);
        auto ids = dec.step(b.data(), kClasses, 2, 4);  // gap 4 > 2
        check(ids == std::vector<int>({0}), "greedy: distant repeat re-emits");
    }
    {
        // Mid-window blank-separated repeats are not seam artifacts and
        // still emit even within the gap.
        auto dec = make_greedy();
        auto fr = frames_for({0, kBlank, 0});
        auto ids = dec.step(fr.data(), kClasses, 3, 0);
        check(ids == std::vector<int>({0, 0}), "greedy: mid-window repeat unaffected");
    }
    {
        // reset_utterance clears the dedup state: the next utterance's first
        // token emits even as an immediate repeat.
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank});
        dec.step(a.data(), kClasses, 2, 0);
        dec.reset_utterance();
        auto ids = dec.step(a.data(), kClasses, 2, 2);
        check(ids == std::vector<int>({0}), "greedy: reset_utterance clears seam state");
    }
}

void
test_greedy_last_emit_frame() {
    auto dec = make_greedy();
    check(dec.last_emit_frame() == -1, "greedy: last_emit_frame -1 before audio");
    // Non-blank at local frames 1 and 2; offset 100 -> last speech = 102.
    // The repeat at frame 2 emits no token but is still speech evidence.
    auto fr = frames_for({kBlank, 0, 0, kBlank});
    dec.step(fr.data(), kClasses, 4, /*frame_offset=*/100);
    check(dec.last_emit_frame() == 102, "greedy: last_emit_frame = last non-blank frame");
}

void
test_greedy_confidence() {
    auto dec = make_greedy();
    // Two emissions at known softmax probs 0.8 and 0.6 -> mean 0.7.
    auto f1 = frame_for(0, 0.8f);
    auto f2 = frame_for(2, 0.6f);
    std::vector<float> fr;
    fr.insert(fr.end(), f1.begin(), f1.end());
    fr.insert(fr.end(), f2.begin(), f2.end());
    dec.step(fr.data(), kClasses, 2, 0);
    check(near(dec.confidence(), 0.7f, 1e-3f), "greedy: confidence = mean token softmax");
}

void
test_greedy_word_timings() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    // ▁hel(f0) lo(f1) blank ▁world(f3) -> words "hel"+"lo" [0,2) and "world" [3,4).
    auto fr = frames_for({0, 1, kBlank, 2});
    dec.step(fr.data(), kClasses, 4, 0);
    dec.finalize();  // flush the in-progress word
    const auto& w = dec.word_timings();
    bool ok = w.size() == 2 && w[0].word == "hello" && w[0].start_frame == 0 &&
              w[0].end_frame == 2 && w[1].word == "world" && w[1].start_frame == 3 &&
              w[1].end_frame == 4;
    check(ok, "greedy: word timings (boundary split, start/end frames)");
}

void
test_greedy_punctuation_word_timing() {
    CtcConfig cfg;
    cfg.d_model = 8;
    cfg.num_classes = kClasses - 1;
    cfg.blank_id = kBlank;
    GreedyCtcDecoder dec(cfg, {"\xE2\x96\x81hello", "\xE2\x96\x81?", "\xE2\x96\x81world"});
    dec.set_compute_timestamps(true);
    auto fr = frames_for({0, 1, kBlank, 2});
    dec.step(fr.data(), kClasses, 4, 0);
    dec.finalize();
    const auto& w = dec.word_timings();
    const bool ok = w.size() == 2 && w[0].word == "hello?" && w[0].start_frame == 0 &&
                    w[0].end_frame == 2 && w[1].word == "world";
    check(ok, "greedy: boundary-prefixed terminator stays with timed word");
}

void
test_greedy_reset_utterance_vs_reset() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    auto fr = frames_for({0});
    dec.step(fr.data(), kClasses, 1, 0);
    dec.finalize();
    dec.reset_utterance();
    check(dec.word_timings().empty(), "greedy: reset_utterance clears word timings");
    check(near(dec.confidence(), 1.0f), "greedy: reset_utterance clears confidence");
    check(dec.last_emit_frame() == 0, "greedy: reset_utterance keeps last_emit_frame");
    // reset_utterance clears the collapse state (utterance boundary): the same
    // id right after the EOU is a new utterance's first token and must emit.
    auto ids = dec.step(fr.data(), kClasses, 1, 1);
    check(ids == std::vector<int>({0}), "greedy: reset_utterance clears prev id (re-emits)");
    // Full reset also clears collapse state.
    dec.reset();
    check(dec.last_emit_frame() == -1, "greedy: reset clears last_emit_frame");
    ids = dec.step(fr.data(), kClasses, 1, 0);
    check(ids == std::vector<int>({0}), "greedy: reset clears prev id (re-emits)");
}

void
test_greedy_open_word_needs_finalize() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    // "▁hel" opens a word with no closing boundary piece - it stays open.
    auto fr = frames_for({0});
    dec.step(fr.data(), kClasses, 1, 0);
    check(dec.word_timings().empty(), "greedy: open trailing word absent before finalize");
    dec.finalize();
    const auto& w = dec.word_timings();
    check(
        w.size() == 1 && w[0].word == "hel",
        "greedy: finalize flushes the open trailing word into word_timings");
}

// ---------------------------------------------------------------------------
// RnntGreedyDecoder with a scripted staged engine. Each joint round supplies
// argmax ids for all remaining frames under one predictor state. Predictor h/c
// candidates are stamped with a per-call serial so blank caching and state
// threading are observable.
// ---------------------------------------------------------------------------
class MockRnntEngine : public RnntEngine {
   public:
    MockRnntEngine() {
        cfg_.d_model = 4;
        cfg_.vocab_size = 5;  // ids 0..3 + blank
        cfg_.blank_id = 4;
        cfg_.pred_hidden = 3;
        cfg_.pred_num_layers = 1;
        cfg_.joint_dim = 4;
        cfg_.max_symbols_per_step = 3;
        vocab_ = {
            "\xE2\x96\x81"
            "a",
            "\xE2\x96\x81"
            "b",
            "\xE2\x96\x81"
            "c",
            "\xE2\x96\x81?"};
    }

    const RnntConfig& rnnt_config() const override { return cfg_; }
    const std::vector<std::string>& vocab() const override { return vocab_; }

    struct State : RnntStreamState {
        float bank[2] = {0.0f, 0.0f};
    };

    std::unique_ptr<RnntStreamState> make_rnnt_stream_state() override {
        return std::make_unique<State>();
    }

    void predict_rnnt(RnntStreamState& opaque, int prev_token, int active_bank) override {
        auto& state = static_cast<State&>(opaque);
        last_prev_token = prev_token;
        last_h_in = state.bank[active_bank];
        predictor_calls++;
        serial += 1.0f;
        state.bank[active_bank ^ 1] = serial;
    }

    void joint_argmax(
        RnntStreamState& /*state*/, const float* /*enc_proj*/, int joint_dim, int T,
        int32_t* token_ids, const float* logit_bias = nullptr) override {
        check(joint_dim == cfg_.joint_dim, "rnnt mock: projected width");
        joint_calls++;
        if (logit_bias != nullptr) {
            last_logit_bias.assign(logit_bias, logit_bias + cfg_.vocab_size);
        } else {
            last_logit_bias.clear();
        }
        std::vector<int> round;
        if (!rounds.empty()) {
            round = std::move(rounds.front());
            rounds.pop_front();
        }
        for (int i = 0; i < T; ++i) {
            token_ids[i] = (i < static_cast<int>(round.size())) ? round[i] : cfg_.blank_id;
        }
    }

    void joint_tdt_argmax(RnntStreamState&, const float*, int, int, int32_t*, int32_t*) override {
        throw std::runtime_error("RNNT mock does not implement TDT");
    }

    std::deque<std::vector<int>> rounds;
    int last_prev_token = -100;
    float last_h_in = -1.0f;
    float serial = 0.0f;
    int predictor_calls = 0;
    int joint_calls = 0;
    std::vector<float> last_logit_bias;

   private:
    RnntConfig cfg_;
    std::vector<std::string> vocab_;
};

void
test_rnnt_emission_and_blank() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    // T=3 frames. f0 emits 0 then blank; f1 blank; f2 emits 1, 2, blank.
    // Each vector is one speculative joint pass over the remaining frames.
    eng.rounds = {{0, blank, blank}, {blank, blank, 1}, {2}, {blank}};
    std::vector<float> enc(3 * 4, 0.0f);
    auto ids = dec.step(enc.data(), 4, 3, /*frame_offset=*/10);
    check(ids == std::vector<int>({0, 1, 2}), "rnnt: scripted emission sequence");
    check(dec.last_emit_frame() == 12, "rnnt: last_emit_frame = last emitting frame");
    check(eng.predictor_calls == 4, "rnnt: predictor runs only initially and after emits");
    check(eng.joint_calls == 4, "rnnt: joint consumes blank runs in vectorized rounds");
    const auto& stats = dec.stats();
    check(
        stats.encoder_frames == 3 && stats.emitted_tokens == 3 && stats.predictor_calls == 4 &&
            stats.joint_calls == 4 && stats.joint_frames == 8,
        "rnnt: staged execution counters");
}

void
test_rnnt_max_symbols_cap() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    // Script never returns blank: the inner loop must stop at
    // max_symbols_per_step (3) emissions for the single frame.
    eng.rounds = {{0}, {1}, {2}, {3}, {0}, {1}};
    std::vector<float> enc(4, 0.0f);
    auto ids = dec.step(enc.data(), 4, 1, 0);
    check(static_cast<int>(ids.size()) == 3, "rnnt: max_symbols_per_step caps the inner loop");
}

void
test_rnnt_eou_punctuation_bias() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);

    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.last_logit_bias.empty(), "rnnt: no punctuation bias mid-stream");

    dec.set_finalizing(true);
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 1);
    check(
        eng.last_logit_bias.size() == static_cast<size_t>(eng.rnnt_config().vocab_size),
        "rnnt: EOU bias spans the joint vocabulary including blank");
    check(
        eng.last_logit_bias.size() > 4 && near(eng.last_logit_bias[3], 7.5f) &&
            near(eng.last_logit_bias[blank], 0.0f),
        "rnnt: EOU floors terminal punctuation without biasing blank");
}

void
test_rnnt_state_threading_and_reset_utterance() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);

    // Frame emits token 2 (state serial 1 latched), then blank (serial 2
    // not latched: blank leaves the LSTM state untouched).
    eng.rounds = {{2}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.predictor_calls == 2, "rnnt: predictor candidate cached after blank");

    // Next call: the decoder must present the post-emit state (serial 1)
    // and prev_token=2.
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 1);
    check(eng.last_prev_token == 2, "rnnt: prev_token carried across steps");
    check(near(eng.last_h_in, 1.0f), "rnnt: LSTM state latched on emit only (not on blank)");
    check(eng.predictor_calls == 2, "rnnt: blank cache survives step boundary");

    // riva EOU semantics: reset_utterance keeps predictor state + prev_token.
    dec.reset_utterance();
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 2);
    check(eng.last_prev_token == 2, "rnnt: prev_token survives reset_utterance");
    check(near(eng.last_h_in, 1.0f), "rnnt: LSTM state survives reset_utterance");
    check(dec.last_emit_frame() == 0, "rnnt: last_emit_frame survives reset_utterance");
    check(eng.predictor_calls == 2, "rnnt: blank cache survives reset_utterance");

    // Full reset: prev_token back to blank, state zeroed.
    dec.reset();
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.last_prev_token == blank, "rnnt: reset restores prev_token = blank");
    check(near(eng.last_h_in, 0.0f), "rnnt: reset zeroes the LSTM state");
    check(dec.last_emit_frame() == -1, "rnnt: reset clears last_emit_frame");
}

void
test_rnnt_word_timings() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // f0 emits "▁a", f1 emits "▁b" -> two words at global frames 5 and 6.
    eng.rounds = {{0, blank}, {blank, 1}, {blank}};
    std::vector<float> enc(2 * 4, 0.0f);
    dec.step(enc.data(), 4, 2, /*frame_offset=*/5);
    dec.finalize();
    const auto& w = dec.word_timings();
    bool ok = w.size() == 2 && w[0].word == "a" && w[0].start_frame == 5 && w[1].word == "b" &&
              w[1].start_frame == 6;
    check(ok, "rnnt: word timings on scripted emissions");
}

void
test_rnnt_punctuation_word_timing() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // f0 emits "▁a", f1 emits "▁?": punctuation extends the first word.
    eng.rounds = {{0, blank}, {blank, 3}, {blank}};
    std::vector<float> enc(2 * 4, 0.0f);
    dec.step(enc.data(), 4, 2, /*frame_offset=*/5);
    dec.finalize();
    const auto& w = dec.word_timings();
    const bool ok =
        w.size() == 1 && w[0].word == "a?" && w[0].start_frame == 5 && w[0].end_frame == 7;
    check(ok, "rnnt: boundary-prefixed terminator stays with timed word");
}

void
test_rnnt_open_word_needs_finalize() {
    // The contract CacheStreamRunner::finalize() relies on: an open trailing
    // word reaches word_timings() only after finalize(). Without the runner
    // calling head_->finalize() at EOS, the last word's WordInfo is dropped.
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    eng.rounds = {{0}, {eng.rnnt_config().blank_id}};  // emit "▁a", then blank
    std::vector<float> enc(eng.rnnt_config().d_model, 0.0f);
    dec.step(enc.data(), eng.rnnt_config().d_model, 1, /*frame_offset=*/0);
    check(dec.word_timings().empty(), "rnnt: open trailing word absent before finalize");
    dec.finalize();
    const auto& w = dec.word_timings();
    check(
        w.size() == 1 && w[0].word == "a",
        "rnnt: finalize flushes the open trailing word into word_timings");
}

class MockTdtEngine : public RnntEngine {
   public:
    MockTdtEngine() {
        cfg_.d_model = cfg_.joint_dim = 2;
        cfg_.vocab_size = 4;
        cfg_.blank_id = 3;
        cfg_.max_symbols_per_step = 3;
        cfg_.durations = {0, 1, 2};
        vocab_ = {
            "\xE2\x96\x81"
            "a",
            "\xE2\x96\x81"
            "b",
            "c"};
    }
    const RnntConfig& rnnt_config() const override { return cfg_; }
    const std::vector<std::string>& vocab() const override { return vocab_; }
    struct State : RnntStreamState {};
    std::unique_ptr<RnntStreamState> make_rnnt_stream_state() override {
        return std::make_unique<State>();
    }
    void predict_rnnt(RnntStreamState&, int prev_token, int) override {
        last_prev_token = prev_token;
        ++predictor_calls;
    }
    void joint_argmax(RnntStreamState&, const float*, int, int, int32_t*, const float*) override {
        throw std::runtime_error("TDT mock does not implement RNNT argmax");
    }
    void joint_tdt_argmax(
        RnntStreamState&, const float*, int joint_dim, int T, int32_t* tokens,
        int32_t* durations) override {
        check(joint_dim == cfg_.joint_dim && T == 1, "tdt: scalar joint shape");
        ++joint_calls;
        const auto next = rounds.empty() ? std::pair<int, int>{cfg_.blank_id, 1} : rounds.front();
        if (!rounds.empty())
            rounds.pop_front();
        tokens[0] = next.first;
        durations[0] = next.second;
    }

    std::deque<std::pair<int, int>> rounds;
    int predictor_calls = 0;
    int joint_calls = 0;
    int last_prev_token = -1;

   private:
    RnntConfig cfg_;
    std::vector<std::string> vocab_;
};

void
test_tdt_duration_and_state_commit() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // t=0 emits a with duration 0, then blank advances to t=1. A duration-2
    // blank skips to t=3, where b emits with duration 2 and finishes T=5.
    eng.rounds = {{0, 0}, {blank, 1}, {blank, 2}, {1, 2}};
    std::vector<float> enc(5 * 2, 0.0f);
    const auto ids = dec.step(enc.data(), 2, 5, 10);
    dec.finalize();
    check(ids == std::vector<int>({0, 1}), "tdt: token and duration sequence");
    check(eng.predictor_calls == 2, "tdt: predictor commits only on non-blank");
    check(dec.last_emit_frame() == 13, "tdt: duration advances encoder frame");
    const auto& words = dec.word_timings();
    check(
        words.size() == 2 && words[0].start_frame == 10 && words[0].end_frame == 11 &&
            words[1].start_frame == 13 && words[1].end_frame == 15,
        "tdt: token timestamps include predicted span");
}

void
test_tdt_zero_duration_guard() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    eng.rounds = {{blank, 0}};
    std::vector<float> enc(2, 0.0f);
    const auto ids = dec.step(enc.data(), 2, 1, 0);
    check(ids.empty() && eng.joint_calls == 1, "tdt: blank duration zero advances immediately");
}

void
test_tdt_duration_crosses_step_boundary() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> frame(2, 0.0f);
    eng.rounds = {{blank, 2}};
    check(dec.step(frame.data(), 2, 1, 0).empty(), "tdt: boundary setup is blank");

    const int calls_before_skip = eng.joint_calls;
    check(dec.step(frame.data(), 2, 1, 1).empty(), "tdt: carried duration skips a whole block");
    check(eng.joint_calls == calls_before_skip, "tdt: skipped block performs no joint call");

    eng.rounds = {{0, 1}};
    const auto ids = dec.step(frame.data(), 2, 1, 2);
    check(ids == std::vector<int>({0}), "tdt: emits after carried duration");
    check(dec.last_emit_frame() == 2, "tdt: carries duration across step boundaries");
}

}  // namespace

int
main() {
    test_sentencepiece_punctuation_detokenization();
    test_greedy_collapse_and_blank();
    test_greedy_cross_chunk_collapse();
    test_greedy_seam_dedup();
    test_greedy_last_emit_frame();
    test_greedy_confidence();
    test_greedy_word_timings();
    test_greedy_punctuation_word_timing();
    test_greedy_reset_utterance_vs_reset();
    test_greedy_open_word_needs_finalize();
    test_rnnt_emission_and_blank();
    test_rnnt_max_symbols_cap();
    test_rnnt_eou_punctuation_bias();
    test_rnnt_state_threading_and_reset_utterance();
    test_rnnt_word_timings();
    test_rnnt_punctuation_word_timing();
    test_rnnt_open_word_needs_finalize();
    test_tdt_duration_and_state_commit();
    test_tdt_zero_duration_guard();
    test_tdt_duration_crosses_step_boundary();
    std::fprintf(stdout, g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
