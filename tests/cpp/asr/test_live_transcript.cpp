// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "live_transcript.h"

using nemo_speech::asr::Alternative;
using nemo_speech::asr::Result;
using nemo_speech::asr::Word;
using nemo_speech::cli::LiveTranscriptPresenter;

namespace {

int failures = 0;

void
check(bool condition, const char* message) {
    std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", message);
    if (!condition)
        ++failures;
}

Word
word(std::string text, int start_ms, int end_ms, int speaker) {
    Word result;
    result.word = std::move(text);
    result.start_time = start_ms;
    result.end_time = end_ms;
    result.speaker_tag = speaker;
    return result;
}

Alternative
alternative(std::string transcript, std::vector<Word> words) {
    Alternative result;
    result.transcript = std::move(transcript);
    result.words = std::move(words);
    return result;
}

Result
final_result(Alternative value, float audio_processed = 0.0f) {
    Result result;
    result.is_final = true;
    result.audio_processed = audio_processed;
    result.alternatives.push_back(std::move(value));
    return result;
}

void
test_canonical_single_speaker_text() {
    LiveTranscriptPresenter presenter(/*diarize=*/true);
    auto result = final_result(alternative(
        "alpha compound omega.", {word("alpha", 0, 160, 1), word("com", 160, 240, 1),
                                  word("pound", 240, 400, 1), word("omega.", 400, 560, 1)}));

    check(
        presenter.present_final(result).empty(),
        "a diarized turn remains open until its speaker changes");
    const auto turns = presenter.finish();
    check(turns.size() == 1, "single-speaker final produces one visible turn");
    check(
        turns.size() == 1 && turns.front().text == "alpha compound omega.",
        "single-speaker final preserves canonical decoder text");
}

void
test_silence_final_settles_without_more_speech() {
    LiveTranscriptPresenter presenter(/*diarize=*/true);
    const auto partial =
        presenter.present_partial(alternative("alpha beta gamma", {word("alpha", 0, 160, 1)}));
    check(!partial.text.empty(), "partial is visible before endpoint");

    auto result = final_result(
        alternative(
            "alpha beta gamma.",
            {word("alpha", 0, 160, 1), word("beta", 160, 320, 1), word("gamma.", 320, 480, 1)}),
        1.5f);
    check(
        presenter.present_final(result).empty(),
        "silence commits text without closing its speaker");
    check(
        presenter.current_turn() != nullptr &&
            presenter.current_turn()->text == "alpha beta gamma.",
        "the committed current turn remains available for live display");
    const auto turns = presenter.finish();
    check(
        turns.size() == 1 && turns.front().text == "alpha beta gamma.",
        "silence final settles even when no more speech arrives");
}

void
test_punctuation_stays_with_previous_speaker() {
    LiveTranscriptPresenter presenter(/*diarize=*/true);
    auto result = final_result(alternative(
        "alpha beta? gamma delta.",
        {word("alpha", 0, 160, 1), word("beta", 160, 320, 1), word("?", 320, 400, 2),
         word("gamma", 400, 560, 2), word("delta.", 560, 720, 2)}));

    auto turns = presenter.present_final(result);
    auto trailing = presenter.finish();
    turns.insert(turns.end(), trailing.begin(), trailing.end());
    check(turns.size() == 2, "multi-speaker final produces two turns");
    check(
        turns.size() == 2 && turns[0].speaker == 1 && turns[0].text == "alpha beta?",
        "closing punctuation stays with the preceding speaker");
    check(
        turns.size() == 2 && turns[1].speaker == 2 && turns[1].text == "gamma delta.",
        "next speaker begins with lexical text");
}

void
test_delayed_punctuation_moves_to_previous_final() {
    for (const std::string punctuation : {".", "?", "!"}) {
        LiveTranscriptPresenter presenter(/*diarize=*/true);
        auto first = final_result(
            alternative("alpha beta", {word("alpha", 0, 160, 1), word("beta", 160, 320, 1)}), 0.5f);
        check(
            presenter.present_final(first).empty(),
            "an incomplete final remains pending for a possible closing token");

        const auto partial = presenter.present_partial(alternative(
            punctuation + " gamma delta",
            {word(punctuation, 640, 720, 2), word("gamma", 720, 880, 2)}));
        check(
            presenter.current_turn()->text == "alpha beta",
            "an interim speaker label does not close the committed turn");
        check(
            partial.turns.size() == 2 && partial.turns[0].text == "alpha beta" + punctuation &&
                partial.turns[1].text == "gamma delta",
            "preview punctuation stays with the prior speaker");

        const auto revised = presenter.present_partial(
            alternative(punctuation + " gamma delta epsilon", {word(punctuation, 640, 720, 2)}));
        check(
            revised.text == "alpha beta" + punctuation + " gamma delta epsilon",
            "leading punctuation stays suppressed as the partial revises");

        auto second = final_result(
            alternative(
                punctuation + " gamma delta epsilon.",
                {word(punctuation, 640, 720, 2), word("gamma", 720, 880, 2),
                 word("delta", 880, 1040, 2), word("epsilon.", 1040, 1200, 2)}),
            1.4f);
        auto settled = presenter.present_final(second);
        auto tail = presenter.finish();
        settled.insert(settled.end(), tail.begin(), tail.end());
        check(
            settled.size() == 2 && settled[0].speaker == 1 &&
                settled[0].text == "alpha beta" + punctuation,
            "delayed punctuation is attached to the previous speaker");
        check(
            settled.size() == 2 && settled[1].speaker == 2 &&
                settled[1].text == "gamma delta epsilon.",
            "next final begins with lexical text");
    }
}

void
test_same_speaker_endpoints_form_one_turn() {
    LiveTranscriptPresenter presenter(/*diarize=*/true);
    check(
        presenter
            .present_final(final_result(
                alternative("alpha beta.", {word("alpha", 0, 160, 2), word("beta.", 160, 320, 2)})))
            .empty(),
        "first endpoint keeps the active speaker open");
    check(
        presenter
            .present_final(final_result(alternative(
                "gamma delta.", {word("gamma", 640, 800, 2), word("delta.", 800, 960, 2)})))
            .empty(),
        "silence does not split the same speaker");
    check(
        presenter.current_turn() != nullptr &&
            presenter.current_turn()->text == "alpha beta. gamma delta.",
        "the live current turn includes same-speaker endpoint continuations");

    const auto settled = presenter.present_final(final_result(alternative(
        "epsilon zeta.", {word("epsilon", 1120, 1280, 3), word("zeta.", 1280, 1440, 3)})));
    check(
        settled.size() == 1 && settled.front().speaker == 2 &&
            settled.front().text == "alpha beta. gamma delta.",
        "a real speaker change closes one consolidated turn");
    const auto tail = presenter.finish();
    check(
        tail.size() == 1 && tail.front().speaker == 3 && tail.front().text == "epsilon zeta.",
        "end-of-stream closes the active speaker");
}

void
test_non_diarized_finals_release_at_each_endpoint() {
    LiveTranscriptPresenter presenter(/*diarize=*/false);
    const auto turns = presenter.present_final(final_result(alternative("alpha beta.", {})));
    check(
        turns.size() == 1 && turns.front().text == "alpha beta.",
        "non-diarized finals are released immediately");
}

void
test_interim_revisions_never_consume_final_text() {
    LiveTranscriptPresenter presenter(true);
    const auto interim = alternative(
        "alpha beta gamma delta", {word("alpha", 0, 200, 1), word("beta", 200, 400, 1),
                                   word("gamma", 400, 600, 2), word("delta", 600, 800, 2)});
    const auto preview = presenter.present_partial(interim);
    check(
        preview.turns.size() == 2 && preview.turns[0].speaker == 1 && preview.turns[1].speaker == 2,
        "speaker changes appear immediately in a replaceable preview");
    check(presenter.current_turn() == nullptr, "interims do not mutate committed state");
    const auto corrected = alternative(
        "alpha beta? gamma delta.", {word("alpha", 0, 200, 2), word("beta?", 200, 400, 2),
                                     word("gamma", 400, 600, 2), word("delta.", 600, 800, 2)});
    const auto revised = presenter.present_partial(corrected);
    check(
        revised.turns.size() == 1 && revised.turns[0].speaker == 2 &&
            revised.text == corrected.transcript,
        "label and punctuation revisions replace the full preview");
    check(
        presenter.present_final(final_result(corrected)).empty(), "one final speaker remains open");
    const auto turns = presenter.finish();
    check(
        turns.size() == 1 && turns[0].speaker == 2 && turns[0].text == corrected.transcript,
        "final uses all canonical words, corrected tags and punctuation");
}

void
test_short_final_turns_are_preserved() {
    for (int duration : {100, 1000}) {
        LiveTranscriptPresenter presenter(true);
        auto turns = presenter.present_final(final_result(alternative(
            "Ready now? No. All right.",
            {word("Ready", 0, 400, 1), word("now?", 400, 800, 1),
             word("No.", 1000, 1000 + duration, 2), word("All", 2200, 2600, 1),
             word("right.", 2600, 3000, 1)})));
        auto tail = presenter.finish();
        turns.insert(turns.end(), tail.begin(), tail.end());
        check(
            turns.size() == 3 && turns[1].speaker == 2 && turns[1].text == "No.",
            "a one-word response retains its speaker regardless of duration");
    }
}

void
test_pending_turn_and_partial_are_reconciled_once() {
    LiveTranscriptPresenter presenter(true);
    presenter.present_final(final_result(
        alternative("alpha beta", {word("alpha", 0, 200, 1), word("beta", 200, 400, 1)})));
    const auto preview = presenter.present_partial(alternative(
        "? gamma delta",
        {word("?", 480, 560, 2), word("gamma", 560, 800, 2), word("delta", 800, 1040, 2)}));
    check(
        preview.turns.size() == 2 && preview.turns[0].text == "alpha beta?" &&
            preview.turns[1].text == "gamma delta",
        "preview places late punctuation on prior turn");
    check(
        presenter.current_turn()->text == "alpha beta",
        "preview punctuation does not mutate pending final");
    const auto final = presenter.present_final(final_result(alternative(
        "? gamma delta.",
        {word("?", 480, 560, 2), word("gamma", 560, 800, 2), word("delta.", 800, 1040, 2)})));
    const auto tail = presenter.finish();
    check(
        final.size() == 1 && final[0].text == "alpha beta?" && tail.size() == 1 &&
            tail[0].text == "gamma delta.",
        "final reconciliation emits each word and punctuation once");
}

void
test_canonical_multispeaker_subwords_and_unicode() {
    LiveTranscriptPresenter presenter(true);
    auto turns = presenter.present_final(final_result(alternative(
        "alpha computer. Écho oui！",
        {word("alpha", 0, 200, 1), word("com", 200, 400, 1), word("puter.", 400, 600, 2),
         word("Écho", 600, 800, 2), word("oui", 800, 1000, 2), word("！", 1000, 1200, 1)})));
    const auto tail = presenter.finish();
    turns.insert(turns.end(), tail.begin(), tail.end());
    check(
        turns.size() == 2 && turns[0].text == "alpha computer." && turns[1].text == "Écho oui！",
        "speaker boundaries never split a lexical word or detach Unicode punctuation");
}

void
test_changed_word_alignment_keeps_canonical_text() {
    LiveTranscriptPresenter presenter(true);
    presenter.present_partial(alternative(
        "twenty one items",
        {word("twenty", 0, 200, 1), word("one", 200, 400, 1), word("items", 400, 600, 2)}));
    presenter.present_final(final_result(alternative(
        "21 items.",
        {word("twenty", 0, 200, 1), word("one", 200, 400, 1), word("items", 400, 600, 2)})));
    const auto tail = presenter.finish();
    check(
        tail.size() == 1 && tail[0].text == "21 items." && tail[0].speaker == 0,
        "unaligned final text is kept intact without inventing speaker boundaries");
}

void
test_turns_without_spaces() {
    for (const auto& pieces : std::vector<std::pair<std::string, std::string>>{
             {"你好。", "再见。"},
             {"こんにちは", "さようなら"},
             {"你好", "再见"},
             {"Yes.", "No."}}) {
        LiveTranscriptPresenter presenter(true);
        auto value = alternative(
            pieces.first + pieces.second,
            {word(pieces.first, 0, 400, 1), word(pieces.second, 500, 900, 2)});
        check(
            presenter.present_partial(value).turns.size() == 2,
            "no-space partial honors aligned speaker boundaries");
        auto turns = presenter.present_final(final_result(value));
        const auto tail = presenter.finish();
        turns.insert(turns.end(), tail.begin(), tail.end());
        check(
            turns.size() == 2 && turns[0].speaker == 1 && turns[1].speaker == 2 &&
                turns[0].text == pieces.first && turns[1].text == pieces.second,
            "no-space finals retain both speakers and exact canonical text");
    }
}

void
test_unknown_does_not_inherit_neighbor() {
    LiveTranscriptPresenter presenter(true);
    presenter.present_final(final_result(alternative("Hello.", {word("Hello.", 0, 200, 1)})));
    auto unknown = alternative(
        "21 birds. Yes.", {word("twenty", 300, 500, 2), word("one", 500, 700, 2),
                           word("birds.", 700, 900, 2), word("Yes.", 1000, 1200, 3)});
    const auto preview = presenter.present_partial(unknown);
    check(
        preview.turns.size() == 2 && preview.turns[1].speaker == 0,
        "unaligned partial cannot inherit a known preceding speaker");
    auto turns = presenter.present_final(final_result(unknown));
    auto next =
        presenter.present_final(final_result(alternative("Next.", {word("Next.", 1300, 1500, 2)})));
    turns.insert(turns.end(), next.begin(), next.end());
    const auto tail = presenter.finish();
    turns.insert(turns.end(), tail.begin(), tail.end());
    check(
        turns.size() == 3 && turns[0].speaker == 1 && turns[1].speaker == 0 &&
            turns[1].text == "21 birds. Yes." && turns[2].speaker == 2,
        "unknown final stays separate from both known neighbors");
}

void
test_final_speaker_revisions() {
    nemo_speech::cli::LiveDiarizationBuffer buffer;
    LiveTranscriptPresenter presenter(true);
    buffer.push(final_result(alternative("Alpha.", {word("Alpha.", 0, 400, 1)}), 1));
    buffer.push(final_result(alternative("Beta.", {word("Beta.", 500, 900, 1)}), 1.5));
    int revised_speaker = 1;
    auto retag = [&](Result& result) {
        for (auto& w : result.alternatives.front().words)
            if (w.start_time >= 500)
                w.speaker_tag = revised_speaker;
    };
    check(
        buffer.update(0, retag).empty(), "ASR finals remain mutable before diarization is stable");
    auto preview = buffer.preview(presenter);
    check(
        preview.turns.size() == 1 && preview.text == "Alpha. Beta.",
        "ASR-final words remain visible during silence before labels stabilize");
    revised_speaker = 2;
    check(buffer.update(0, retag).empty(), "new speaker evidence does not freeze revisable words");
    preview = buffer.preview(presenter);
    check(
        preview.turns.size() == 2 && preview.turns[1].speaker == 2,
        "a late speaker correction updates an already ASR-final preview");
    for (const auto& result : buffer.update(0.45, retag)) presenter.present_final(result);
    // A later speaker birth must remain revisable even after two labels exist.
    revised_speaker = 3;
    buffer.update(0.45, retag);
    preview = buffer.preview(presenter);
    check(
        preview.turns.size() == 2 && preview.turns[0].speaker == 1 &&
            preview.turns[1].speaker == 3 && preview.text == "Alpha. Beta.",
        "third-speaker revisions preserve the stable prefix without duplication");
    auto ready = buffer.update(1.1, retag);
    check(ready.size() == 1, "only stable finals leave the revision buffer");
    std::vector<nemo_speech::subtitle::SpeakerTurn> printed;
    for (const auto& result : ready) {
        auto turns = presenter.present_final(result);
        printed.insert(printed.end(), turns.begin(), turns.end());
    }
    const auto tail = presenter.finish();
    printed.insert(printed.end(), tail.begin(), tail.end());
    check(
        printed.size() == 2 && printed[0].text == "Alpha." && printed[1].text == "Beta." &&
            printed[1].speaker == 3,
        "printed output uses corrected labels and emits every word once");
    check(buffer.update(2, retag).empty(), "stable results cannot be emitted twice");
}

void
test_preview_waits_for_confirmed_speaker_change() {
    nemo_speech::cli::LiveDiarizationBuffer buffer;
    LiveTranscriptPresenter presenter(true);
    presenter.present_final(final_result(alternative("Alpha.", {word("Alpha.", 0, 400, 1)})));
    buffer.push(final_result(alternative("Beta.", {word("Beta.", 500, 900, 2)}), 1.5));
    const auto partial = alternative("Gamma", {word("Gamma", 1000, 1300, 3)});
    auto view = buffer.preview(presenter, &partial, 0.45);
    check(
        view.turns.size() == 2 && view.turns[0].speaker == 1 && view.turns[1].speaker == 0 &&
            view.text == "Alpha. Beta. Gamma",
        "unconfirmed labels do not change the display speaker");
    view = buffer.preview(presenter, &partial, 1.0);
    check(
        view.turns.size() == 3 && view.turns[1].speaker == 2 && view.turns[2].speaker == 0,
        "confirmed speaker change shows before all pending words are stable");
    view = buffer.preview(presenter, &partial, 1.4);
    check(
        view.turns.size() == 3 && view.turns[2].speaker == 3,
        "the next confirmed speaker replaces the held preview label");
    const auto ready = buffer.update(2, [](Result&) {});
    check(
        ready.size() == 1 && ready[0].alternatives.front().words[0].speaker_tag == 2 &&
            partial.words[0].speaker_tag == 3 && presenter.current_turn()->speaker == 1,
        "preview does not mutate recognition tags or presenter state");
}

void
test_buffer_finish_and_zero_duration() {
    nemo_speech::cli::LiveDiarizationBuffer buffer;
    LiveTranscriptPresenter presenter(true);
    auto retag = [](Result&) {};
    buffer.push(final_result(alternative("Short.", {word("Short.", 1000, 1000, 1)}), 1.2));
    check(
        buffer.update(1.1, retag).empty(),
        "zero-duration word waits for its onset attribution window to stabilize");
    auto partial = alternative("Next", {word("Next", 1500, 1600, 2)});
    const auto preview = buffer.preview(presenter, &partial);
    check(
        preview.turns.size() == 2 && preview.text == "Short. Next",
        "active partial follows pending finals without replacing them");
    auto ready = buffer.update(std::numeric_limits<double>::infinity(), retag);
    check(ready.size() == 1, "end-of-stream releases the final revisable suffix");
}

void
test_late_punctuation_attaches_to_previous_final() {
    nemo_speech::cli::LiveDiarizationBuffer buffer;
    LiveTranscriptPresenter presenter(true);
    auto retag = [](Result&) {};
    check(!buffer.attach_late_punctuation("."), "late punctuation needs a buffered final");
    buffer.push(final_result(alternative("Ready", {word("Ready", 0, 300, 1)}), 1));
    check(buffer.attach_late_punctuation("?"), "late punctuation attaches to a buffered final");
    auto ready = buffer.update(2, retag);
    check(
        ready.size() == 1 && ready[0].alternatives[0].transcript == "Ready?" &&
            ready[0].alternatives[0].words[0].word == "Ready?",
        "late punctuation updates the buffered text and last word");

    for (const auto& result : ready) presenter.present_final(result);
    presenter.present_final(final_result(alternative("Go", {word("Go", 900, 1100, 1)}), 2));
    check(presenter.attach_late_punctuation("."), "late punctuation attaches to the open turn");
    const auto tail = presenter.finish();
    check(tail.size() == 1 && tail[0].text == "Ready? Go.", "open turn keeps attached punctuation");
    check(!presenter.attach_late_punctuation("."), "an emitted turn is not rewritten");
    check(
        !LiveTranscriptPresenter(false).attach_late_punctuation("."),
        "non-diarized finals have no open turn");
}

void
test_delayed_punctuation_in_revision_buffer() {
    for (const std::string leading : {"?", "  ? ", " ?  ” "}) {
        nemo_speech::cli::LiveDiarizationBuffer buffer;
        LiveTranscriptPresenter presenter(true);
        auto retag = [](Result&) {};
        buffer.push(final_result(alternative("Ready", {word("Ready", 0, 300, 1)}), 1));
        for (const auto& result : buffer.update(2, retag)) presenter.present_final(result);
        const std::string punctuation = leading.find("”") != std::string::npos ? "?”" : "?";
        buffer.push(final_result(alternative(leading + "Next", {}), 3));
        auto preview = buffer.preview(presenter);
        check(
            preview.turns.size() == 2 && preview.turns[0].text == "Ready" + punctuation &&
                preview.turns[1].text == "Next" && preview.turns[1].speaker == 0,
            "delayed punctuation crosses finals despite whitespace and unknown speaker");
        std::vector<nemo_speech::subtitle::SpeakerTurn> printed;
        for (const auto& result : buffer.update(4, retag)) {
            auto turns = presenter.present_final(result);
            printed.insert(printed.end(), turns.begin(), turns.end());
        }
        auto tail = presenter.finish();
        printed.insert(printed.end(), tail.begin(), tail.end());
        check(
            printed.size() == 2 && printed[0].text == "Ready" + punctuation &&
                printed[1].text == "Next",
            "next final does not start with the previous sentence's punctuation");
    }
    nemo_speech::cli::LiveDiarizationBuffer buffer;
    LiveTranscriptPresenter presenter(true);
    auto retag = [](Result&) {};
    buffer.push(final_result(alternative("Done", {word("Done", 0, 300, 1)}), 1));
    buffer.push(final_result(alternative(" . ", {word(".", 1000, 1100, 2)}), 2));
    buffer.update(0, retag);
    const auto preview = buffer.preview(presenter);
    check(
        preview.turns.size() == 1 && preview.turns[0].text == "Done." &&
            preview.turns[0].speaker == 1,
        "punctuation-only delayed final cannot create a phantom speaker turn");
    for (const auto& result : buffer.update(std::numeric_limits<double>::infinity(), retag))
        presenter.present_final(result);
    const auto tail = presenter.finish();
    check(
        tail.size() == 1 && tail[0].text == "Done.",
        "Ctrl-C emits delayed punctuation once on its lexical speaker");
}

}  // namespace

int
main() {
    test_canonical_single_speaker_text();
    test_silence_final_settles_without_more_speech();
    test_punctuation_stays_with_previous_speaker();
    test_delayed_punctuation_moves_to_previous_final();
    test_same_speaker_endpoints_form_one_turn();
    test_non_diarized_finals_release_at_each_endpoint();
    test_interim_revisions_never_consume_final_text();
    test_short_final_turns_are_preserved();
    test_pending_turn_and_partial_are_reconciled_once();
    test_canonical_multispeaker_subwords_and_unicode();
    test_changed_word_alignment_keeps_canonical_text();
    test_turns_without_spaces();
    test_unknown_does_not_inherit_neighbor();
    test_final_speaker_revisions();
    test_preview_waits_for_confirmed_speaker_change();
    test_buffer_finish_and_zero_duration();
    test_delayed_punctuation_in_revision_buffer();
    test_late_punctuation_attaches_to_previous_final();
    std::printf(failures ? "FAILED (%d)\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
