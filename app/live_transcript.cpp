// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_transcript.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace nemo_speech::cli {
namespace {

double
speaker_ready_time(const asr::Word& word) {
    return std::max(word.end_time / 1000.0, word.start_time / 1000.0 + 0.160);
}

bool
ascii_word_byte(unsigned char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
}

void
append_text(std::string& destination, const std::string& text) {
    if (text.empty())
        return;
    if (!destination.empty() && !subtitle::attaches_to_previous(text) &&
        !std::isspace(static_cast<unsigned char>(destination.back())))
        destination += ' ';
    destination += text;
}

std::string
take_leading_punctuation(std::string& text) {
    size_t end = 0;
    while (end < text.size() && std::isspace(static_cast<unsigned char>(text[end]))) ++end;
    std::string punctuation;
    while (end < text.size() && subtitle::attaches_to_previous(text.substr(end))) {
        // Advance a whole UTF-8 code point.
        const size_t begin = end;
        ++end;
        while (end < text.size() && (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80) ++end;
        punctuation.append(text, begin, end - begin);
        while (end < text.size() && std::isspace(static_cast<unsigned char>(text[end]))) ++end;
    }
    if (!punctuation.empty())
        text.erase(0, end);
    return punctuation;
}

// Slice the transcript at aligned word boundaries; decoder words may be
// subword fragments, so never rebuild text by space-joining them.
std::vector<subtitle::SpeakerTurn>
speaker_turns(const Alternative& alternative, bool diarize) {
    const auto& text = alternative.transcript;
    const auto& words = alternative.words;
    if (text.empty())
        return {};
    const int start = words.empty() ? 0 : words.front().start_time;
    const int end = words.empty() ? 0 : words.back().end_time;
    if (!diarize || words.empty())
        return {{0, start, end, text}};

    std::vector<size_t> starts;
    size_t cursor = 0;
    for (const auto& word : words) {
        while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor])))
            ++cursor;
        if (word.word.empty() || text.compare(cursor, word.word.size(), word.word) != 0) {
            // Words don't match the text (e.g. PnC/ITN rewrite): keep one turn,
            // tagged only if all words agree.
            int speaker = words.front().speaker_tag;
            for (const auto& value : words)
                if (value.speaker_tag != speaker)
                    speaker = 0;
            return {{speaker, start, end, text}};
        }
        starts.push_back(cursor);
        cursor += word.word.size();
    }

    std::vector<subtitle::SpeakerTurn> turns;
    size_t begin = 0;
    int speaker = words.front().speaker_tag;
    auto emit = [&](size_t next) {
        const size_t first = begin == 0 ? 0 : starts[begin];
        size_t last = next == words.size() ? text.size() : starts[next];
        while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1]))) --last;
        turns.push_back(
            {speaker, words[begin].start_time, words[next - 1].end_time,
             text.substr(first, last - first)});
    };
    for (size_t i = 1; i < words.size(); ++i) {
        const int next_speaker = words[i].speaker_tag;
        // Never split between ASCII subword fragments; other scripts can have
        // turn boundaries without whitespace.
        if (subtitle::attaches_to_previous(words[i].word) ||
            (starts[i] > 0 && ascii_word_byte(text[starts[i] - 1]) &&
             ascii_word_byte(text[starts[i]])))
            continue;
        if (next_speaker != speaker) {
            emit(i);
            begin = i;
            speaker = next_speaker;
        }
    }
    emit(words.size());
    return turns;
}

void
append_turn(std::vector<subtitle::SpeakerTurn>& turns, subtitle::SpeakerTurn turn) {
    if (!turns.empty()) {
        append_text(turns.back().text, take_leading_punctuation(turn.text));
        if (turn.text.empty())
            return;
        auto& previous = turns.back();
        if (previous.speaker == turn.speaker) {
            previous.start_ms = std::min(previous.start_ms, turn.start_ms);
            previous.end_ms = std::max(previous.end_ms, turn.end_ms);
            append_text(previous.text, turn.text);
            return;
        }
    }
    turns.push_back(std::move(turn));
}

}  // namespace

std::vector<subtitle::SpeakerTurn>
LiveTranscriptPresenter::settle_pending() {
    std::vector<subtitle::SpeakerTurn> settled;
    if (pending_turn_valid_)
        settled.push_back(std::move(pending_turn_));
    pending_turn_valid_ = false;
    return settled;
}

LivePartialPresentation
LiveTranscriptPresenter::present_partial(const Alternative& alternative) const {
    LivePartialPresentation presentation;
    if (diarize_ && pending_turn_valid_)
        presentation.turns.push_back(pending_turn_);
    for (auto& turn : speaker_turns(alternative, diarize_))
        append_turn(presentation.turns, std::move(turn));
    for (const auto& turn : presentation.turns) append_text(presentation.text, turn.text);
    return presentation;
}

std::vector<subtitle::SpeakerTurn>
LiveTranscriptPresenter::present_final(const Result& result) {
    if (result.alternatives.empty() || result.alternatives.front().transcript.empty())
        return {};
    auto incoming = speaker_turns(result.alternatives.front(), diarize_);
    if (!diarize_)
        return incoming;
    auto turns = settle_pending();
    for (auto& turn : incoming) append_turn(turns, std::move(turn));
    if (!turns.empty()) {
        pending_turn_ = std::move(turns.back());
        pending_turn_valid_ = true;
        turns.pop_back();
    }
    return turns;
}

std::vector<subtitle::SpeakerTurn>
LiveTranscriptPresenter::finish() {
    return settle_pending();
}

bool
LiveTranscriptPresenter::attach_late_punctuation(const std::string& punctuation) {
    if (!pending_turn_valid_ || punctuation.empty())
        return false;
    pending_turn_.text += punctuation;
    return true;
}

bool
LiveDiarizationBuffer::attach_late_punctuation(const std::string& punctuation) {
    if (pending_.empty() || punctuation.empty())
        return false;
    auto& alternative = pending_.back().alternatives.front();
    alternative.transcript += punctuation;
    if (!alternative.words.empty())
        alternative.words.back().word += punctuation;
    return true;
}

void
LiveDiarizationBuffer::push(Result result) {
    if (!result.alternatives.empty() && !result.alternatives.front().transcript.empty())
        pending_.push_back(std::move(result));
}

std::vector<Result>
LiveDiarizationBuffer::update(
    double stable_before_seconds, const std::function<void(Result&)>& retag) {
    for (auto& result : pending_) retag(result);
    std::vector<Result> ready;
    while (!pending_.empty()) {
        const auto& result = pending_.front();
        const auto& words = result.alternatives.front().words;
        // Wait past each word's 160 ms onset window (2 V2 / 16 V3 frames), even
        // when zero-duration, so extrapolated labels are not frozen.
        double end = result.audio_processed;
        if (!words.empty()) {
            end = 0.0;
            for (const auto& word : words) end = std::max(end, speaker_ready_time(word));
        }
        if (end >= stable_before_seconds)
            break;
        ready.push_back(std::move(pending_.front()));
        pending_.pop_front();
    }
    return ready;
}

LivePartialPresentation
LiveDiarizationBuffer::preview(
    const LiveTranscriptPresenter& committed, const Alternative* partial,
    double stable_before_seconds) const {
    auto presenter = committed;
    LivePartialPresentation presentation;
    auto confirmed_labels = [&](Alternative& alternative) {
        for (auto& word : alternative.words)
            if (speaker_ready_time(word) >= stable_before_seconds)
                word.speaker_tag = 0;
    };
    // Mask unstable labels on display copies only; the terminal shows untagged
    // text under the active speaker.
    for (auto result : pending_) {
        confirmed_labels(result.alternatives.front());
        for (auto& turn : presenter.present_final(result))
            append_turn(presentation.turns, std::move(turn));
    }
    std::vector<subtitle::SpeakerTurn> tail;
    if (partial) {
        auto visible = *partial;
        confirmed_labels(visible);
        tail = presenter.present_partial(visible).turns;
    } else {
        tail = presenter.finish();
    }
    for (const auto& turn : tail) append_turn(presentation.turns, turn);
    for (const auto& turn : presentation.turns) append_text(presentation.text, turn.text);
    return presentation;
}

}  // namespace nemo_speech::cli
