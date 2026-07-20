// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Streaming CTC decoder backed by Flashlight LexiconDecoder and KenLM.
#pragma once

#ifdef NEMO_SPEECH_WITH_FLASHLIGHT

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "decoder.h"
#include "greedy_ctc_decoder.h"

namespace fl {
namespace lib {
namespace text {
class LexiconDecoder;
class Trie;
class Dictionary;
class KenLM;
class LM;
}  // namespace text
}  // namespace lib
}  // namespace fl

namespace sentencepiece {
class SentencePieceProcessor;
}  // namespace sentencepiece

namespace nemo_speech::asr {

struct FlashlightCtcCfg {
    std::string lm_path;
    std::string lexicon_path;
    // SentencePiece tokenizer.model (the model's own tokenizer). Optional; only
    // used to encode OOV / multi-word boost phrases into token pieces. Empty =
    // OOV phrases are skipped (in-vocab single-word boosting still works).
    std::string tokenizer_path;

    int beam_size = 32;
    int beam_size_token = 16;
    double beam_threshold = 20.0;
    double lm_weight = 0.8;
    double word_insertion_score = 1.0;
    // See DecoderConfig::max_boost.
    double max_boost = 10.0;
    double unk_score = -std::numeric_limits<double>::infinity();
    double sil_score = 0.0;
    bool log_add = true;

    // The beam spans the whole utterance, matching riva (its LexiconDecoder
    // is fed continuously and reset only by external endpointing); EOU and
    // finalize restart it via reset_utterance(). max_segment_frames is a
    // memory backstop for pause-free speech on endpointing-off streams; keep
    // it above any WER-set utterance length. Units are encoder frames
    // (subsampling=8 -> ~80 ms/frame at 100 Hz mel).
    int max_segment_frames = 1500;  // ~2 min safety cap

    // Token names - parakeet-ctc SentencePiece convention.
    std::string sil_token = "▁";  // SentencePiece ▁
    std::string unk_token = "<unk>";
    std::string blank_token = "<blank>";  // appended after the SP vocab
};

// Immutable dictionaries, KenLM, and LM-scored trie shared across streams.
// Streams that add OOV boost words clone the mutable resources first.
struct FlashlightResources {
    // Word -> token-piece spellings (flashlight's LexiconMap, a pure STL type).
    // Kept resident so a private clone can rebuild the trie without reloading
    // the lexicon or KenLM from disk.
    using LexiconMap = std::unordered_map<std::string, std::vector<std::vector<std::string>>>;

    FlashlightResources(
        const CtcConfig& ctc_cfg, const std::vector<std::string>& vocab,
        const FlashlightCtcCfg& fl_cfg);

    // Private copy for OOV-boost mutation: shares the immutable KenLM, token
    // dictionary, and lexicon (no disk reload, no ~700 MB KenLM duplication);
    // deep-copies the word dictionary and rebuilds the trie so insertions don't
    // touch the shared bundle. The trie rebuild reproduces the shared trie
    // exactly (same words, same KenLM-smeared scores), ready for OOV inserts.
    FlashlightResources clone_for_mutation() const;

    int word_count() const;

    std::shared_ptr<fl::lib::text::Dictionary> token_dict;
    std::shared_ptr<fl::lib::text::Dictionary> word_dict;
    std::shared_ptr<fl::lib::text::KenLM> kenlm;
    std::shared_ptr<fl::lib::text::Trie> trie;
    std::shared_ptr<const LexiconMap> lexicon;
    int sil_idx = 0;
    int blank_idx = 0;
    int unk_idx = 0;
    // word_dict size after the lexicon build; indices >= this are OOV boost
    // words added at request time (KenLM cannot score them).
    int base_word_count = 0;
    // KenLM <unk> unigram - the LM base for OOV boost words, so OOV and
    // in-vocab boosts are calibrated the same way (unigram base + boost).
    float oov_lm_score = 0.0f;
};

class FlashlightDecoder : public Decoder {
   public:
    // Blank is appended at ctc_cfg.blank_id. `resources` shares prebuilt state;
    // null builds a private instance.
    FlashlightDecoder(
        CtcConfig ctc_cfg, std::vector<std::string> vocab, FlashlightCtcCfg fl_cfg,
        std::shared_ptr<const FlashlightResources> resources = nullptr);
    ~FlashlightDecoder() override;

    void reset() override;
    // EOU: the beam + committed words + transcript are owned internally here,
    // so a clean per-utterance restart is a full reset (the LM beam re-decodes
    // each utterance independently anyway). last_emit_frame survives per the
    // Decoder contract (the endpointing silence timeline spans utterances).
    void reset_utterance() override {
        const int64_t lef = last_emit_frame_;
        reset();
        last_emit_frame_ = lef;
    }
    std::vector<int> step(
        const float* log_probs, int n_classes, int T, int64_t frame_offset) override;
    int blank_id() const override { return ctc_cfg_.blank_id; }
    const std::vector<std::string>& vocab() const override { return vocab_; }

    std::string partial_transcript() const override { return partial_; }
    std::string final_transcript() const override { return final_; }
    void finalize() override;

    // Word boosting (speech_contexts): in-vocab words get a per-word LM bump
    // via a BoostedLm over KenLM (no mutation, works on shared resources); OOV
    // words are SentencePiece-encoded and inserted into the word dict + trie
    // so the LexiconDecoder can emit them - that mutates, so a decoder on a
    // shared bundle clones private resources first. Call once before decoding;
    // OOV entries are not reverted - use a fresh decoder to change the boost set.
    void set_request_options(const AsrRequestOptions& opts) override;

    // Word timestamps. Upstream flashlight's DecodeResult carries per-frame
    // `words`/`tokens` vectors (index = decode frame, -1 = none), so the frame a
    // word completes at is just its index - no separate CTC alignment pass.
    void set_compute_timestamps(bool on) override { compute_timestamps_ = on; }
    const std::vector<WordTiming>& word_timings() const override { return word_timings_; }

    // Speech evidence for token-silence EOU: the last non-blank argmax frame.
    // The beam emits word ids at commit time, not acoustically, so the
    // runner's token stream carries no silence signal for this head.
    int64_t last_emit_frame() const override { return last_emit_frame_; }

    // The per-frame argmax that maintains last_emit_frame_ runs only when the
    // runner will poll it (token-silence endpointing). Default on.
    void set_track_speech_frame(bool on) override { track_speech_frame_ = on; }

   private:
    std::string detokenize_word_ids(const std::vector<int>& word_ids) const;
    // Append WordTimings for one finalized segment's hypothesis. `words`/`tokens`
    // are the DecodeResult per-frame vectors; `seg_start` is the global encoder
    // frame of frame 0 of this segment.
    void extract_timings(
        const std::vector<int>& words, const std::vector<int>& tokens, int64_t seg_start);
    // (Re)build decoder_ with the given LM (kenlm_ or a boosting wrapper).
    void build_decoder(std::shared_ptr<fl::lib::text::LM> lm);
    // Detokenize (committed ++ current). `current` is a flashlight DecodeResult
    // `words` field - a per-frame vector where most entries are -1 (no word
    // emitted that frame) and the rest are word ids. detokenize_word_ids skips
    // the -1 entries automatically.
    std::string combine_word_lists(
        const std::vector<int>& committed, const std::vector<int>& current) const;
    // Finalize the current segment's best hypothesis into committed_words_
    // and restart the beam (decodeEnd + decodeBegin), so the decoder only
    // ever runs over a bounded segment.
    void commit_segment();

    // Switch to a private FlashlightResources copy before any mutation (OOV
    // boost insertion). No-op when the resources are already private.
    void ensure_private_resources();
    // Copy the resource handles + derived values into the working members.
    void adopt_resources(const FlashlightResources& res);

    CtcConfig ctc_cfg_;
    std::vector<std::string> vocab_;
    FlashlightCtcCfg cfg_;

    // Working handles into a FlashlightResources bundle. Mutation (OOV boost) is
    // only legal once the bundle is private, i.e. when shared_resources_ is null
    // (built our own) or after ensure_private_resources() clones + clears it.
    std::shared_ptr<fl::lib::text::Dictionary> token_dict_;
    std::shared_ptr<fl::lib::text::Dictionary> word_dict_;
    std::shared_ptr<fl::lib::text::KenLM> kenlm_;
    std::shared_ptr<fl::lib::text::Trie> trie_;
    // Non-null => the adopted bundle is shared (not ours to mutate); kept so
    // ensure_private_resources() can clone-for-mutation off it. Null => private
    // (built our own, or already cloned).
    std::shared_ptr<const FlashlightResources> shared_resources_;
    std::unique_ptr<fl::lib::text::LexiconDecoder> decoder_;
    // Non-null when speech_contexts are active: a BoostedLm wrapping kenlm_,
    // kept alive for the decoder's lifetime.
    std::shared_ptr<fl::lib::text::LM> boosted_lm_;

    // SentencePiece encoder for OOV boost words, lazily loaded from
    // cfg_.tokenizer_path on first need (null if no tokenizer is configured).
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spm_;
    int base_word_count_ = 0;
    float oov_lm_score_ = 0.0f;

    int sil_idx_ = 0;
    int blank_idx_ = 0;
    int unk_idx_ = 0;

    // Per-segment streaming state. The decoder runs over one bounded segment;
    // commit_segment() finalizes it at the max_segment_frames backstop and
    // restarts the beam, so the hypothesis buffer never grows unbounded.
    int64_t seg_frames_ = 0;          // frames decoded in the current segment
    int64_t last_emit_frame_ = -1;    // last non-blank argmax frame (global)
    bool track_speech_frame_ = true;  // maintain last_emit_frame_ (see setter)

    // Word ids finalized from completed segments. partial_ / final_ are
    // committed_words_ ++ the current segment's best hypothesis.
    std::vector<int> committed_words_;
    std::string partial_;
    std::string final_;

    // Timestamps. seg_global_start_frame_ is the global encoder frame of the
    // current segment's frame 0 (set on the segment's first step). word_timings_
    // accumulates committed segments' spans; populated only when requested.
    bool compute_timestamps_ = false;
    int64_t seg_global_start_frame_ = 0;
    std::vector<WordTiming> word_timings_;
};

}  // namespace nemo_speech::asr

#endif  // NEMO_SPEECH_WITH_FLASHLIGHT
