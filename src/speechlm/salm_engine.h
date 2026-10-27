// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Speech-LLM transcription (SALM): a speech encoder whose projected frames are
// spliced into an LLM chat prompt in place of an audio locator token, followed by
// greedy decoding. Both halves run from GGUF in one process:
//
//   wav -> log-mel -> PEE encoder (asr + speaker experts, fused) -> fusion.proj
//       -> LLM input embeddings (n_embd per 80 ms frame)
//   `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n` + text-before-locator
//       + [audio embeddings] + text-after-locator
//       + `<|im_end|>\n<|im_start|>assistant\n<think></think>`  -> llama.cpp decode
//
// Throughput comes from batching at every stage: encoder windows are batched by
// length across utterances, prompts are prefilled as multi-sequence batches, and
// decoding advances every in-flight sequence with one llama_decode per step,
// refilling slots as sequences finish (continuous batching).
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pee/ctc_timestamp.h"

namespace nemo_speech::speechlm {

struct SalmConfig {
    std::string encoder_path;  // PEE GGUF carrying fusion.proj (pee.fusion.proj_dim == LLM n_embd)
    std::string ctc_path;  // CTC timestamp head GGUF (the encoder file); empty: no word timestamps
    std::string llm_path;  // llama.cpp GGUF of the decoder
    int gpu = 0;           // GPU device index for encoder and decoder, -1 = CPU only
    int threads = 8;       // CPU threads for host-side decoder work
    // User turn; must contain exactly one audio locator.
    std::string prompt =
        "Produce a verbatim transcript of the audio. Preserve named entities, abbreviations, "
        "numbers, dates, "
        "measurements, acronyms, and technical terms as clearly as possible. Keep the same "
        "language as spoken "
        "and do not translate. <|audio|>";
    int max_new_tokens = 1024;
    // Decode slots: sequences decoded together in one llama_decode per step.
    // Memory scales linearly (recurrent state + attention KV per slot). The defaults fit a 32 GB
    // GPU.
    int max_sequences = 16;
    // Attention KV cells per slot, pooled: n_ctx_per_sequence x max_sequences cells shared by all
    // sequences, so one long request can use many slots' worth. The batched path admits a request
    // when its prompt and a first share of its output budget fit and grows its share as it
    // generates; when the pool cannot grow a sequence, the most recently admitted one is preempted
    // (re-encoded and prefilled again from its prompt and generated tokens later). Other paths
    // admit a request when its prompt and whole output budget fit.
    int n_ctx_per_sequence = 8192;
    // Longest sequence (prompt text + audio frames + generated tokens), within the pool.
    int max_context = 73728;
    int n_batch = 2048;  // tokens/frames per llama_decode during prefill
    // Multi-token prediction (speculative decoding with the model's own MTP head):
    // number of draft tokens per step, 0 = plain greedy. Requires a GGUF that
    // carries the NextN tensors.
    int mtp_draft = 7;
    // DFlash2 drafter GGUF (block drafter reading 6 target layers); when set it drafts the
    // mtp_draft tokens per step instead of the model's MTP head.
    std::string dflash_path;
    int encoder_max_batch = 16;  // windows per encoder graph launch
    // Audio seconds encoded but not yet admitted (ready queue + encoder calls in flight). Their
    // projection stays on the device (n_embd floats per 80 ms frame, about 134 KB per second), so
    // this bounds that memory when long requests wait for slots; 0 = unbounded.
    int encoder_ahead_seconds = 20000;
};

struct SalmRequest {
    std::string id;  // echoed in the result (e.g. the file path)
    const float* samples = nullptr;
    size_t count = 0;
    int sample_rate = 16000;  // resampled to the encoder rate if different
    // Duration of the audio before the caller resampled it to `sample_rate`; 0: count /
    // sample_rate.
    double seconds = 0;
    // User turn with exactly one audio locator; empty = SalmConfig::prompt. Requests with different
    // prompts are decoded together.
    std::string prompt;
    int max_new_tokens =
        0;  // > 0: generation budget of this request; 0 = min(max_new_tokens, 32 + 10 tokens/s)
    bool timestamps = false;  // word timestamps and speakers (needs SalmConfig::ctc_path)
    // Speaker turns alone (with `timestamps`): the request ends at its encoder output, with the
    // 10 ms diarization segments in SalmResult::details and no prefill, decode or transcript.
    bool diarization_only = false;
    float temperature = 0;         // below 1e-5: greedy; else sampled at this temperature
    std::optional<uint64_t> seed;  // sampling seed (same seed, same transcript); absent: random
    bool latin_script =
        false;  // a Latin-script language: no tokens of other scripts (batched path)
    // The generated text as it is committed: valid UTF-8 pieces that add up to SalmResult::text.
    // Called on the engine thread between decode steps, so it must return quickly and not throw.
    std::function<void(std::string_view)> on_text;
    // Non-empty (requests without timestamps): a response that is empty once whitespace is
    // stripped is decoded once more with this user turn, from the same encoder output; the result
    // is the second decode's and SalmResult::first_attempt keeps the first decode's counts.
    std::string fallback_prompt;
};

struct SalmResult {
    std::string
        timestamp_error;  // reason when the alignment is unavailable (transcript stays available)
    // Word-timestamp requests: the alignment part of the speech-LLM payload (words, speakers,
    // diarization, status).
    std::shared_ptr<asr::SpeechLmDetails> details;
    std::string id;
    std::string text;
    std::string error;  // non-empty when this request failed (others still complete)
    bool invalid_request =
        false;  // the error is the request's fault (bad audio length or prompt), not the engine's
    bool cancelled = false;  // dropped by SalmEngine::cancel (error says so; no text)
    double audio_seconds = 0;
    int audio_tokens = 0;      // encoder frames spliced into the prompt
    int prompt_tokens = 0;     // text tokens + audio tokens
    int generated_tokens = 0;  // text tokens (the end-of-turn token excluded)
    bool end_of_turn = false;  // generation ended at <|im_end|> (not at the budget or the context)
    // The first decode's counts when SalmRequest::fallback_prompt ran a second one.
    struct Attempt {
        int audio_tokens = 0, prompt_tokens = 0, generated_tokens = 0;
        bool end_of_turn = false;
    };
    std::optional<Attempt> first_attempt;
};

// Admission statistics of a serve() call.
struct SalmStats {
    size_t refills = 0;      // admission rounds
    int peak_sequences = 0;  // most sequences decoding at once
};

// Request source for SalmEngine::serve: append up to `max` requests to `out` and
// return false once no further requests will come. With `block`, wait until at
// least one request is available or the source closes.
using SalmPull = std::function<bool(std::vector<SalmRequest>& out, size_t max, bool block)>;
// Completion sink: `ticket` is the request's position in pull order.
using SalmDone = std::function<void(size_t ticket, SalmResult&& result)>;

class SalmEngine {
   public:
    explicit SalmEngine(SalmConfig cfg);
    ~SalmEngine();
    SalmEngine(const SalmEngine&) = delete;
    SalmEngine& operator=(const SalmEngine&) = delete;

    // In-flight batching over a live source: new requests are admitted into free
    // decode slots while others are still generating, and each result is handed
    // to `done` as soon as it completes. Returns once the source is closed and
    // every pulled request has completed. Borrowed samples must stay valid until
    // their result is delivered. Not thread-safe: one call at a time per engine (the
    // decoder context is shared).
    void serve(const SalmPull& pull, const SalmDone& done, SalmStats* stats = nullptr);

    // Drop the request with this ticket (its position in pull order in the running serve() call)
    // wherever it is: waiting to be encoded, inside an encoder call (the call completes; its output
    // for the request is discarded), encoded and waiting for a slot, decoding (its slot, KV cells,
    // recurrent and drafter state are released at the next step boundary) or aligning word
    // timestamps. Its result (SalmResult::cancelled) is delivered through `done` as usual; after
    // that the engine no longer reads the request's samples. Thread-safe, unlike the rest of the
    // class: the serve loop picks the message up between steps. Unknown or finished tickets are
    // ignored, and tickets sent between serve() calls are dropped when the next call starts.
    void cancel(size_t ticket);

    const SalmConfig& config() const;
    int sample_rate() const;
    // The batched serving path (patched ggml, CUDA, Nemotron-H MoE decoder) rather than upstream
    // llama.cpp behavior.
    bool batched() const;

    // The first special token of the decoder's tokenizer that `text` tokenizes into (the added
    // tokens: control or user-defined), other than the audio locator and the speaker, event and
    // style tags the model writes itself; empty when there is none. A client instruction is
    // spliced into the chat template, where such a token would forge or close a turn.
    // Thread-safe.
    std::string special_token_in(const std::string& text) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::speechlm
