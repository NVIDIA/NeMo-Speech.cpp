// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "flashlight_decoder.h"

#ifdef NEMO_SPEECH_WITH_FLASHLIGHT

#include <sentencepiece_processor.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"
#include "runtime.h"

namespace nemo_speech::asr {

using fl::lib::text::createWordDict;
using fl::lib::text::CriterionType;
using fl::lib::text::Dictionary;
using fl::lib::text::KenLM;
using fl::lib::text::LexiconDecoder;
using fl::lib::text::LexiconDecoderOptions;
using fl::lib::text::LexiconMap;
using fl::lib::text::LM;
using fl::lib::text::loadWords;
using fl::lib::text::SmearingMode;
using fl::lib::text::Trie;

namespace {
// Word-level boosting LM: delegates to a base LM (KenLM) and adds a per-word
// boost to the score when a boosted word index is queried. The LexiconDecoder
// (isLmToken=false) scores the LM with WORD indices at word completion, so a
// flat wordIdx->boost map suffices for per-word boosting.
//
// OOV boost words are appended to the word dict past the base lexicon, so their
// indices (>= base_word_count) are NOT in KenLM's vocab map - scoring them
// through KenLM would index out of bounds. For those, contribute only the boost
// and leave the LM context unchanged.
class BoostedLm : public LM {
   public:
    BoostedLm(
        std::shared_ptr<LM> base, std::unordered_map<int, float> boost, int base_word_count,
        float oov_base_score)
        : base_(std::move(base)), boost_(std::move(boost)), base_word_count_(base_word_count),
          oov_base_score_(oov_base_score) {}

    fl::lib::text::LMStatePtr start(bool startWithNothing) override {
        return base_->start(startWithNothing);
    }
    std::pair<fl::lib::text::LMStatePtr, float> score(
        const fl::lib::text::LMStatePtr& state, const int usrTokenIdx) override {
        // OOV boost words (idx past the base lexicon) aren't in KenLM's vocab
        // map. Score them as the <unk> unigram + boost - symmetric with the
        // in-vocab path (own unigram + boost), so the boost magnitude means the
        // same thing for both. LM context is left unchanged.
        if (usrTokenIdx >= base_word_count_) {
            auto it = boost_.find(usrTokenIdx);
            const float boost = it != boost_.end() ? it->second : 0.0f;
            return {state, oov_base_score_ + boost};
        }
        auto r = base_->score(state, usrTokenIdx);
        auto it = boost_.find(usrTokenIdx);
        if (it != boost_.end())
            r.second += it->second;
        return r;
    }
    std::pair<fl::lib::text::LMStatePtr, float> finish(
        const fl::lib::text::LMStatePtr& state) override {
        return base_->finish(state);
    }

   private:
    std::shared_ptr<LM> base_;
    std::unordered_map<int, float> boost_;
    int base_word_count_;
    float oov_base_score_;
};

void
to_lower_ascii(std::string& s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

std::vector<std::string>
split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string w;
    while (iss >> w) out.push_back(w);
    return out;
}

// Shared by initial construction and clone-for-mutation so both produce an
// identical KenLM-scored, MAX-smeared trie.
std::shared_ptr<Trie>
build_lexicon_trie(
    const LexiconMap& lexicon, const Dictionary& token_dict, const Dictionary& word_dict,
    KenLM& kenlm, int sil_idx) {
    const int max_children = static_cast<int>(token_dict.indexSize());
    auto trie = std::make_shared<Trie>(max_children, sil_idx);
    auto start_state = kenlm.start(false);
    size_t n_seqs = 0, n_dropped = 0;
    for (const auto& kv : lexicon) {
        const std::string& word = kv.first;
        const auto& token_seqs = kv.second;
        int word_idx = word_dict.getIndex(word);
        auto scored = kenlm.score(start_state, word_idx);
        const float lm_score = static_cast<float>(scored.second);
        for (const auto& pieces : token_seqs) {
            ++n_seqs;
            std::vector<int> token_ids;
            token_ids.reserve(pieces.size());
            bool skip = false;
            for (const auto& p : pieces) {
                if (!token_dict.contains(p)) {
                    skip = true;
                    break;
                }
                token_ids.push_back(token_dict.getIndex(p));
            }
            if (skip || token_ids.empty()) {
                ++n_dropped;
                continue;
            }
            trie->insert(token_ids, word_idx, lm_score);
        }
    }
    // A non-trivial drop rate means lexicon token pieces are absent from the
    // model's CTC vocab - i.e. the lexicon/tokenizer were built for a DIFFERENT
    // model. The trie is then missing words and decoding silently degrades, so
    // warn loudly at load (a matched lexicon drops ~0%).
    if (n_seqs > 0) {
        const double drop_pct =
            100.0 * static_cast<double>(n_dropped) / static_cast<double>(n_seqs);
        if (drop_pct >= 2.0) {
            std::cerr << "[flashlight] WARNING: dropped " << n_dropped << "/" << n_seqs << " ("
                      << drop_pct
                      << "%) lexicon entries - their token pieces are not in the model's CTC "
                         "vocab. The lexicon/tokenizer likely do not match this model; "
                         "transcription will be degraded.\n";
        }
    }
    trie->smear(SmearingMode::MAX);
    return trie;
}
}  // namespace

FlashlightResources::FlashlightResources(
    const CtcConfig& ctc_cfg, const std::vector<std::string>& vocab,
    const FlashlightCtcCfg& fl_cfg) {
    // Indices must match the encoder's CTC output axis (C_out = num_classes+1).
    // The GGUF vocab has num_classes entries; blank is appended at
    // ctc_cfg.blank_id (= vocab.size() for parakeet-ctc-1.1b).
    token_dict = std::make_shared<Dictionary>();
    for (size_t i = 0; i < vocab.size(); ++i) {
        token_dict->addEntry(vocab[i], static_cast<int>(i));
    }
    if (!token_dict->contains(fl_cfg.blank_token)) {
        token_dict->addEntry(fl_cfg.blank_token, ctc_cfg.blank_id);
    }
    blank_idx = ctc_cfg.blank_id;
    sil_idx = token_dict->contains(fl_cfg.sil_token) ? token_dict->getIndex(fl_cfg.sil_token)
                                                     : blank_idx;  // fallback: silScore bonus only

    LexiconMap lex = loadWords(fl_cfg.lexicon_path, -1);
    if (lex.empty()) {
        throw std::runtime_error(
            "FlashlightResources: loadWords returned empty lexicon: " + fl_cfg.lexicon_path);
    }
    word_dict = std::make_shared<Dictionary>(createWordDict(lex));
    if (!word_dict->contains(fl_cfg.unk_token)) {
        word_dict->addEntry(fl_cfg.unk_token);
    }
    unk_idx = word_dict->getIndex(fl_cfg.unk_token);
    // Word indices beyond this are OOV boost words added at request time.
    base_word_count = word_dict->indexSize();

    kenlm = std::make_shared<KenLM>(fl_cfg.lm_path, *word_dict);
    // <unk> unigram - the LM base for OOV boost words (see BoostedLm / OOV
    // trie insertion), so OOV and in-vocab boosts are calibrated the same way.
    oov_lm_score = static_cast<float>(kenlm->score(kenlm->start(false), unk_idx).second);

    trie = build_lexicon_trie(lex, *token_dict, *word_dict, *kenlm, sil_idx);

    // Keep the lexicon resident so clone_for_mutation() can rebuild the trie
    // without reloading it (or KenLM) from disk.
    lexicon = std::make_shared<const LexiconMap>(std::move(lex));
}

FlashlightResources
FlashlightResources::clone_for_mutation() const {
    FlashlightResources r = *this;  // shares token_dict, kenlm, lexicon, scalars
    r.word_dict = std::make_shared<Dictionary>(*word_dict);  // private deep copy
    r.trie = build_lexicon_trie(*lexicon, *token_dict, *r.word_dict, *kenlm, sil_idx);
    return r;
}

int
FlashlightResources::word_count() const {
    return static_cast<int>(word_dict->indexSize());
}

FlashlightDecoder::FlashlightDecoder(
    CtcConfig ctc_cfg, std::vector<std::string> vocab, FlashlightCtcCfg fl_cfg,
    std::shared_ptr<const FlashlightResources> resources)
    : ctc_cfg_(ctc_cfg), vocab_(std::move(vocab)), cfg_(std::move(fl_cfg)) {
    if (resources) {
        shared_resources_ = resources;  // non-null => bundle is shared, not ours to mutate
        adopt_resources(*resources);
    } else {
        adopt_resources(FlashlightResources(ctc_cfg_, vocab_, cfg_));
    }
    build_decoder(std::static_pointer_cast<LM>(kenlm_));
}

void
FlashlightDecoder::adopt_resources(const FlashlightResources& res) {
    token_dict_ = res.token_dict;
    word_dict_ = res.word_dict;
    kenlm_ = res.kenlm;
    trie_ = res.trie;
    sil_idx_ = res.sil_idx;
    blank_idx_ = res.blank_idx;
    unk_idx_ = res.unk_idx;
    base_word_count_ = res.base_word_count;
    oov_lm_score_ = res.oov_lm_score;
}

void
FlashlightDecoder::ensure_private_resources() {
    if (!shared_resources_)
        return;  // already private (built our own, or already cloned)
    // Clone off the shared bundle: reuses the immutable KenLM, token dict, and
    // lexicon (no ~700 MB disk reload), giving this stream a private word dict
    // + trie for OOV boost words. The trie is rebuilt (KenLM-score + insert +
    // smear over the whole lexicon) since flashlight's Trie has no deep-copy -
    // a one-time per-stream cost paid only on an OOV boost.
    GGMLF_LOG_INFO(
        "[flashlight] OOV speech_context: building private lexicon trie "
        "(one-time for this stream; shared KenLM reused, no reload)\n");
    adopt_resources(shared_resources_->clone_for_mutation());
    shared_resources_.reset();  // now private: shared_resources_ == nullptr means owned
}

void
FlashlightDecoder::build_decoder(std::shared_ptr<fl::lib::text::LM> lm) {
    LexiconDecoderOptions opts;
    opts.beamSize = cfg_.beam_size;
    opts.beamSizeToken = cfg_.beam_size_token;
    opts.beamThreshold = cfg_.beam_threshold;
    opts.lmWeight = cfg_.lm_weight;
    opts.wordScore = cfg_.word_insertion_score;
    opts.unkScore = cfg_.unk_score;
    opts.silScore = cfg_.sil_score;
    opts.logAdd = cfg_.log_add;
    opts.criterionType = CriterionType::CTC;

    decoder_ = std::make_unique<LexiconDecoder>(
        opts, trie_, lm, sil_idx_, blank_idx_, unk_idx_,
        /*transitions=*/std::vector<float>{},
        /*isLmToken=*/false);

    decoder_->decodeBegin();
}

void
FlashlightDecoder::set_request_options(const AsrRequestOptions& opts) {
    set_compute_timestamps(opts.enable_word_time_offsets);
    if (opts.speech_contexts.empty())
        return;

    bool spm_load_failed = false;
    auto ensure_spm = [this, &spm_load_failed]() -> bool {
        if (active_spm_ != nullptr)
            return true;
        if (spm_load_failed)
            return false;
        // Explicit tokenizer_path overrides; otherwise use the tokenizer
        // embedded in the GGUF (guaranteed to match the model's vocab).
        if (!cfg_.tokenizer_path.empty()) {
            auto sp = std::make_unique<sentencepiece::SentencePieceProcessor>();
            const auto st = sp->Load(cfg_.tokenizer_path);
            if (!st.ok()) {
                std::cerr << "[boost] failed to load tokenizer '" << cfg_.tokenizer_path
                          << "': " << st.ToString() << "\n";
                spm_load_failed = true;
                return false;
            }
            spm_ = std::move(sp);
            active_spm_ = spm_.get();
            return true;
        }
        if (cfg_.embedded_spm != nullptr) {
            active_spm_ = cfg_.embedded_spm;
            return true;
        }
        return false;
    };
    // SentencePiece-encode `word` into token-dict ids. False if any piece is not
    // in the model's token vocab (then the word can't be represented/emitted).
    auto encode_to_token_ids = [this](const std::string& word, std::vector<int>& out) -> bool {
        std::vector<std::string> pieces;
        const auto st = active_spm_->Encode(word, &pieces);
        if (!st.ok() || pieces.empty())
            return false;
        out.clear();
        out.reserve(pieces.size());
        for (const auto& p : pieces) {
            if (!token_dict_->contains(p))
                return false;
            out.push_back(token_dict_->getIndex(p));
        }
        return true;
    };
    bool warned_no_spm = false;

    // Bound the per-request work: speech_contexts come from the client, and each
    // word can mutate the word dict + trie and trigger a re-smear. Cap the total
    // boosted-word count so a pathological request can't blow up decoder setup.
    constexpr size_t kMaxBoostWords = 256;
    size_t boost_words_seen = 0;
    bool warned_cap = false;

    std::unordered_map<int, float> boost;  // wordIdx -> boost (in-vocab + OOV)
    bool trie_changed = false;
    bool warned_clamp = false;

    for (const auto& sc : opts.speech_contexts) {
        // BoostedLm adds the boost to a word's score on every emission, so an
        // unbounded value makes re-emitting the boosted word net-positive in the
        // beam and it repeats. Clamp the magnitude (sign preserved so negative
        // de-boosting still works).
        const float clamped = std::max(
            static_cast<float>(-cfg_.max_boost),
            std::min(static_cast<float>(cfg_.max_boost), sc.boost));
        if (clamped != sc.boost && !warned_clamp) {
            std::cerr << "[boost] clamping boost " << sc.boost << " to " << clamped
                      << " (|boost| > max_boost=" << cfg_.max_boost
                      << "; larger values cause spurious word repetition)\n";
            warned_clamp = true;
        }
        for (const auto& phrase : sc.phrases) {
            // The decoder scores completed words; the n-gram LM still favors
            // the phrase's in-order sequence.
            for (std::string word : split_ws(phrase)) {
                to_lower_ascii(word);
                if (word.empty())
                    continue;
                if (++boost_words_seen > kMaxBoostWords) {
                    if (!warned_cap) {
                        std::cerr << "[boost] capping at " << kMaxBoostWords
                                  << " boosted words; extra phrases ignored\n";
                        warned_cap = true;
                    }
                    break;
                }

                if (word_dict_->contains(word)) {
                    boost[word_dict_->getIndex(word)] = clamped;  // in-vocab
                    continue;
                }
                // OOV word (not in the lexicon). Encode + add to this stream's
                // word dict + trie so the LexiconDecoder can emit it.
                if (!ensure_spm()) {
                    if (!warned_no_spm) {
                        std::cerr << "[boost] OOV word(s) skipped: no tokenizer "
                                     "(pass --tokenizer). e.g. '"
                                  << word << "'\n";
                        warned_no_spm = true;
                    }
                    continue;
                }
                std::vector<int> token_ids;
                if (!encode_to_token_ids(word, token_ids)) {
                    std::cerr << "[boost] OOV word '" << word
                              << "' skipped: SentencePiece pieces not in model vocab\n";
                    continue;
                }
                constexpr size_t kMaxOovTokens = 40;
                if (token_ids.size() > kMaxOovTokens) {
                    std::cerr << "[boost] OOV word '" << word << "' skipped: too many tokens ("
                              << token_ids.size() << " > " << kMaxOovTokens << ")\n";
                    continue;
                }
                ensure_private_resources();
                word_dict_->addEntry(word);
                const int widx = word_dict_->getIndex(word);
                // Leaf score = the <unk> unigram (same basis as in-vocab leaves,
                // which carry their own KenLM unigram); the boost is applied once
                // via BoostedLm, so OOV and in-vocab boosts are calibrated alike.
                trie_->insert(token_ids, widx, oov_lm_score_);
                boost[widx] = clamped;
                trie_changed = true;
            }
        }
    }

    if (boost.empty())
        return;
    if (trie_changed)
        trie_->smear(SmearingMode::MAX);  // re-propagate scores for new leaves

    boosted_lm_ = std::make_shared<BoostedLm>(
        std::static_pointer_cast<LM>(kenlm_), std::move(boost), base_word_count_, oov_lm_score_);
    build_decoder(boosted_lm_);  // rebuild beam with the boosting LM
}

FlashlightDecoder::~FlashlightDecoder() = default;

void
FlashlightDecoder::reset() {
    decoder_->decodeBegin();
    seg_frames_ = 0;
    last_emit_frame_ = -1;
    committed_words_.clear();
    partial_.clear();
    final_.clear();
    seg_global_start_frame_ = 0;
    word_timings_.clear();
}

std::vector<int>
FlashlightDecoder::step(const float* log_probs, int n_classes, int T, int64_t frame_offset) {
    if (T <= 0)
        return {};

    // First chunk of a new segment: record where this segment starts globally,
    // so DecodeResult per-frame indices map back to global encoder frames.
    if (seg_frames_ == 0)
        seg_global_start_frame_ = frame_offset;

    decoder_->decodeStep(log_probs, T, n_classes);
    seg_frames_ += T;

    // Per-frame argmax drives the token-silence endpointing signal
    // (last_emit_frame), the only consumer. Skipped when the runner won't poll
    // it (no token-silence endpointing on this head): the beam emits words at
    // commit time, not per-frame, so nothing else needs this scan. The beam is
    // not segmented at silence: riva feeds its LexiconDecoder continuously and
    // resets only on external endpointing, and LM context spanning pauses
    // matters for WER parity.
    if (track_speech_frame_) {
        for (int t = 0; t < T; ++t) {
            const float* fr = log_probs + static_cast<size_t>(t) * n_classes;
            int best = 0;
            for (int c = 1; c < n_classes; ++c) {
                if (fr[c] > fr[best])
                    best = c;
            }
            if (best != blank_idx_)
                last_emit_frame_ = frame_offset + t;
        }
    }

    // Memory backstop only (see FlashlightCtcCfg::max_segment_frames): commit
    // the beam on pause-free speech beyond the cap so the hypothesis buffer
    // stays bounded when endpointing is off.
    if (seg_frames_ >= cfg_.max_segment_frames) {
        commit_segment();
    }

    auto hyp = decoder_->getBestHypothesis(/*lookBack=*/0);
    partial_ = combine_word_lists(committed_words_, hyp.words);

    // Text is exposed directly through partial_transcript().
    return {};
}

void
FlashlightDecoder::commit_segment() {
    decoder_->decodeEnd();
    auto hyps = decoder_->getAllFinalHypothesis();
    if (!hyps.empty()) {
        const auto& h = hyps.front();
        for (int wid : h.words) {
            if (wid >= 0) {
                committed_words_.push_back(wid);
            }
        }
        if (compute_timestamps_)
            extract_timings(h.words, h.tokens, seg_global_start_frame_);
    }
    decoder_->decodeBegin();
    seg_frames_ = 0;
}

void
FlashlightDecoder::finalize() {
    decoder_->decodeEnd();
    auto hyps = decoder_->getAllFinalHypothesis();
    const std::vector<int> cur = hyps.empty() ? std::vector<int>{} : hyps.front().words;
    final_ = combine_word_lists(committed_words_, cur);
    partial_ = final_;
    if (compute_timestamps_ && !hyps.empty())
        extract_timings(hyps.front().words, hyps.front().tokens, seg_global_start_frame_);
}

void
FlashlightDecoder::extract_timings(
    const std::vector<int>& words, const std::vector<int>& tokens, int64_t seg_start) {
    // words[i] >= 0 marks a word completing at frame i. Its start is the first
    // non-blank token after the previous word's completion; end is i (exclusive
    // end = i+1). Frames are segment-local; add seg_start for global indices.
    int prev_end = -1;
    for (int i = 0; i < static_cast<int>(words.size()); ++i) {
        if (words[i] < 0)
            continue;
        const std::string w = word_dict_->getEntry(words[i]);
        if (w == cfg_.unk_token) {  // dropped from the transcript too
            prev_end = i;
            continue;
        }
        int start = i;
        for (int j = prev_end + 1; j <= i && j < static_cast<int>(tokens.size()); ++j) {
            if (tokens[j] >= 0 && tokens[j] != blank_idx_) {
                start = j;
                break;
            }
        }
        WordTiming wt;
        wt.word = w;
        wt.start_frame = seg_start + start;
        wt.end_frame = seg_start + i + 1;  // end exclusive
        wt.confidence = 1.0f;
        word_timings_.push_back(wt);
        prev_end = i;
    }
}

std::string
FlashlightDecoder::combine_word_lists(
    const std::vector<int>& committed, const std::vector<int>& current) const {
    std::vector<int> joined;
    joined.reserve(committed.size() + current.size());
    joined.insert(joined.end(), committed.begin(), committed.end());
    joined.insert(joined.end(), current.begin(), current.end());
    return detokenize_word_ids(joined);
}

std::string
FlashlightDecoder::detokenize_word_ids(const std::vector<int>& word_ids) const {
    std::string out;
    out.reserve(word_ids.size() * 6);
    bool first = true;
    for (int wid : word_ids) {
        if (wid < 0)
            continue;
        const auto& w = word_dict_->getEntry(wid);
        if (w == cfg_.unk_token)
            continue;
        if (!first)
            out.push_back(' ');
        out += w;
        first = false;
    }
    return out;
}

}  // namespace nemo_speech::asr

#endif  // NEMO_SPEECH_WITH_FLASHLIGHT
