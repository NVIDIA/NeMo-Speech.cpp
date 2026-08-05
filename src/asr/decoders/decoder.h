// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Converts encoder chunks into tokens and owns per-stream decoder state.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "parameter_parser.h"
#include "runtime.h"
#include "types.h"

namespace nemo_speech::asr {

struct DecoderConfig {
    enum class Kind { Greedy, Flashlight };
    Kind kind = Kind::Greedy;
    // Flashlight (LM beam) artifacts; used only when kind == Flashlight.
    std::string lm_path;
    std::string lexicon_path;
    std::string tokenizer_path;  // SentencePiece, for OOV boost phrases
    int beam_size = 32;
    int beam_size_token = 16;
    double beam_threshold = 20.0;
    double lm_weight = 0.8;
    double word_insertion_score = 1.0;
    // Upper bound on the magnitude of a per-word speech-context boost. The boost
    // is added to a word's LM score on every emission, so an unbounded value
    // makes re-emitting the boosted word net-positive in the beam and the word
    // repeats spuriously. Clamping caps that at the largest value that still
    // selects without repeating; raise it only knowingly.
    double max_boost = 10.0;
    // RNNT context-biasing (word boosting) shallow-fusion weight: the boosting
    // tree contributes alpha * arc_score to the joint logits at each greedy
    // step. 0 disables RNNT boosting. Independent of the flashlight (CTC) path.
    double boosting_tree_alpha = 1.0;
    // Per-token depth scaling for the RNNT boosting tree (NeMo RNN-T default
    // 2.0): tokens after the first in a phrase get a larger score.
    double boosting_depth_scaling = 2.0;
    // Upper clamp on the per-request boost folded into the RNNT boosting
    // shallow-fusion alpha. riva clamps to 10 (for beam search); the edge RNNT
    // path is greedy, which has no competing hypothesis, so a high boost makes
    // the boosted words insert spuriously / repeat. 5.0 keeps enough headroom
    // for harder words while staying well below riva's beam-oriented 10.
    double boosting_max_boost = 5.0;
    void Register(common::ParameterParser& p) {
        p.Register(
            "kind",
            [this](const std::string& v) {
                if (v == "greedy")
                    kind = Kind::Greedy;
                else if (v == "flashlight")
                    kind = Kind::Flashlight;
                else
                    throw std::runtime_error(
                        "config decoder.kind: invalid value '" + v + "' (greedy|flashlight)");
            },
            "Decoder: greedy | flashlight");
        p.Register(
            "lm_path",
            [this](const std::string& v) {
                lm_path = v;
                if (!v.empty())
                    kind = Kind::Flashlight;
            },
            "Flashlight KenLM path (implies flashlight)", {"--lm-path"});
        p.Register("lexicon_path", &lexicon_path, "Flashlight lexicon path", {"--lexicon"});
        p.Register(
            "tokenizer_path", &tokenizer_path, "SentencePiece tokenizer for OOV boosting",
            {"--tokenizer"});
        p.Register("beam_size", &beam_size, "Flashlight beam size", {"--beam-size"});
        p.Register("beam_size_token", &beam_size_token, "Flashlight token beam size");
        p.Register(
            "beam_threshold", &beam_threshold, "Flashlight beam threshold", {"--beam-threshold"});
        p.Register("lm_weight", &lm_weight, "Flashlight LM weight", {"--lm-weight"});
        p.Register(
            "word_insertion_score", &word_insertion_score, "Flashlight word insertion score",
            {"--word-score"});
        p.Register(
            "max_boost", &max_boost, "Max magnitude of a speech-context word boost",
            {"--max-boost"});
        p.Register(
            "boosting_tree_alpha", &boosting_tree_alpha,
            "RNNT word-boosting shallow-fusion weight (0 disables)", {"--boosting-tree-alpha"});
        p.Register(
            "boosting_depth_scaling", &boosting_depth_scaling,
            "RNNT word-boosting per-token depth scaling", {"--boosting-depth-scaling"});
        p.Register(
            "boosting_max_boost", &boosting_max_boost,
            "Upper clamp on the RNNT word-boosting alpha (greedy-safe default 5.0)",
            {"--boosting-max-boost"});
    }
};

// Per-word span in global encoder frames, start inclusive and end exclusive.
struct WordTiming {
    std::string word;
    int64_t start_frame = 0;
    int64_t end_frame = 0;
    float confidence = 1.0f;
};

// One N-best hypothesis. Times use the same encoder-frame clock as WordTiming.
struct Hypothesis {
    std::string transcript;
    float confidence = 1.0f;
    std::vector<WordTiming> words;  // empty unless word offsets were requested
};

// SentencePiece word-boundary marker U+2581 ('▁') = 0xE2 0x96 0x81. A piece
// starting with it begins a new word (what detokenize_sentencepiece splits on).
inline bool
sp_is_word_boundary(const std::string& p) {
    return p.size() >= 3 && (uint8_t)p[0] == 0xE2 && (uint8_t)p[1] == 0x96 && (uint8_t)p[2] == 0x81;
}
// Piece text with any leading ▁ stripped (the boundary becomes a word split).
inline std::string
sp_piece_text(const std::string& p) {
    return sp_is_word_boundary(p) ? p.substr(3) : p;
}

// Some SentencePiece vocabularies contain both bare and boundary-prefixed
// sentence terminators (for example "?" and "▁?"). The latter still
// attaches to the preceding word when rendered; its boundary marker must not
// introduce a space or a separate timed word.
inline bool
sp_is_sentence_terminator(const std::string& p) {
    const std::string text = sp_piece_text(p);
    return text == "." || text == "?" || text == "!" || text == "\xE0\xA5\xA4" ||
           text == "\xE0\xA5\xA5";
}

inline bool
sp_starts_new_word(const std::string& p) {
    return sp_is_word_boundary(p) && !sp_is_sentence_terminator(p);
}

class Decoder {
   public:
    virtual ~Decoder() = default;

    // Reset per-stream state at start of a new utterance.
    virtual void reset() = 0;

    // Prepare for a new utterance without discarding stream-level acoustic or
    // predictor context. Unlike reset(), this keeps the continuing stream
    // alive; CTC still clears collapse state at the boundary.
    virtual void reset_utterance() {}

    // Consume one chunk of encoder output (d_model x T frames).
    //
    //   enc_out:     host-side buffer of shape (d_model, T) in column-major
    //                (the same layout ggml_backend_tensor_get returns for an
    //                (d_model, T, 1, 1) ggml tensor - that is, frame t starts
    //                at enc_out + t*d_model).
    //   d_model, T:  shape.
    //   frame_offset: global frame index of enc_out[0] within the stream.
    //                For timestamps. Caller is expected to maintain it.
    //
    // Returns the new token IDs emitted by this call (relative to the running
    // stream state). The head also internally accumulates whatever it needs
    // for cross-call deduplication (e.g. CTC's `prev` id).
    virtual std::vector<int> step(
        const float* enc_out, int d_model, int T, int64_t frame_offset) = 0;

    // CUDA RNNT may keep the encoder projection resident and bind it directly
    // into the decoder graph. Other decoders retain the host-buffer path.
    virtual std::vector<int> step_device(const ggml_runtime::DeviceTensor&, int, int, int64_t) {
        throw std::runtime_error("decoder does not accept device-resident encoder output");
    }

    virtual int blank_id() const = 0;
    virtual const std::vector<std::string>& vocab() const = 0;

    // Optional: heads that own decoding internally (e.g. a beam decoder with
    // an LM) can return the transcript directly. When non-empty, the runner
    // uses this verbatim instead of detokenizing the accumulated token ids.
    // Default empty so greedy GreedyCtcDecoder is unchanged.
    virtual std::string partial_transcript() const { return ""; }
    virtual std::string final_transcript() const { return ""; }

    // Flush any non-causal tail state. Beam decoders run getBestHypothesis
    // mid-stream against an unstable beam; finalize() commits the final
    // hypothesis. Called once at end-of-stream.
    virtual void finalize() {}

    // Global encoder frame of the most recent speech evidence this decoder
    // observed (last non-blank argmax/emission), or -1 before any. Drives
    // token-silence endpointing: trailing silence = frames since this.
    // Survives reset_utterance() (the silence timeline continues across
    // utterances so the endpointer can re-arm); cleared by reset().
    virtual int64_t last_emit_frame() const { return -1; }

    // Hint: last_emit_frame() will be polled (the runner runs token-silence
    // endpointing on this head). Default no-op - heads that derive the frame as
    // part of their normal decode ignore it; FlashlightDecoder uses it to skip
    // an extra per-frame argmax it would otherwise run only to feed this signal.
    virtual void set_track_speech_frame(bool /*on*/) {}

    // Hint: the next step() processes an end-of-utterance (is_last) chunk. RNNT
    // greedy uses it to apply the end-of-utterance punctuation-logit floor (see
    // RnntGreedyDecoder); other heads ignore it.
    virtual void set_finalizing(bool /*on*/) {}

    // Utterance confidence. Greedy CTC returns the mean emitted-token posterior;
    // heads without a usable posterior return 1.0.
    virtual float confidence() const { return 1.0f; }

    // Word-level time spans accumulated so far (global encoder-frame indices),
    // for `enable_word_time_offsets`. Default empty; decoders that track
    // timestamps override. Enable tracking via set_compute_timestamps(true)
    // before the first step() (some decoders skip the bookkeeping otherwise).
    virtual void set_compute_timestamps(bool /*on*/) {}
    virtual const std::vector<WordTiming>& word_timings() const {
        static const std::vector<WordTiming> kEmpty;
        return kEmpty;
    }

    // Additional hypotheses beyond the top result, best-first and capped at
    // max_alternatives - 1. Greedy decoders return none.
    virtual std::vector<Hypothesis> additional_hypotheses(int /*max_alternatives*/) const {
        return {};
    }

    // Per-request options. Base wires timestamps; decoders that support
    // boosting (Flashlight) override to also consume speech_contexts. Called
    // once after construction, before the first step().
    virtual void set_request_options(const AsrRequestOptions& opts) {
        set_compute_timestamps(opts.needs_word_timings());
        // Boosting is LM-based - only the Flashlight head (which overrides this)
        // applies it. Warn once so speech_contexts aren't dropped silently on
        // the greedy CTC / RNNT heads.
        if (!opts.speech_contexts.empty()) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                std::fprintf(
                    stderr,
                    "[asr] speech_contexts (word boosting) ignored: this head has no LM; "
                    "boosting requires the Flashlight decoder (--lm-path/--lexicon).\n");
            }
        }
    }
};

// Detokenize a SentencePiece BPE token-id stream into text.
// Replaces U+2581 ('▁') with a single space and trims leading space.
std::string detokenize_sentencepiece(
    const std::vector<int>& ids, const std::vector<std::string>& vocab);
void append_sentencepiece_tokens(
    std::string& text, const std::vector<int>& ids, const std::vector<std::string>& vocab);

}  // namespace nemo_speech::asr
