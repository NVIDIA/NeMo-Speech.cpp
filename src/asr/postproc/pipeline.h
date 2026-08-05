// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// ASR text postprocessing pipeline. Runs the ordered chain on the final
// transcript: profanity filter -> ITN -> PnC -> output formatting. Filters/models
// are loaded once at construction; per-request AsrRequestOptions gate which passes run.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "batching.h"
#include "decoders/decoder.h"
#include "itn.h"
#include "parameter_parser.h"
#include "profanity.h"
#include "types.h"

namespace ggml_runtime {
class BackendManager;
}
namespace nemo_speech::asr::pnc {
class PncModel;
class PncRunner;
}  // namespace nemo_speech::asr::pnc

namespace nemo_speech::asr::postproc {

struct PostprocConfig {
    std::string profanity_list_path;  // empty = profanity filter unavailable
    // Direct grammar dir, or a parent with language children (en/, es/, ...).
    std::string itn_model_dir;   // empty = ITN unavailable
    std::string pnc_model_path;  // empty = PnC unavailable
    int cpu_workers = 2;         // bounded final-result CPU fan-out
    int max_queue_depth = 64;    // backpressure instead of unbounded work
    void Register(common::ParameterParser& p) {
        p.Register(
            "profanity_list_path", &profanity_list_path, "Profanity word list path",
            {"--profanity-list"});
        p.Register(
            "itn_model_dir", &itn_model_dir, "Sparrowhawk ITN grammar dir/root",
            {"--itn-model-dir"});
        p.Register("pnc_model_path", &pnc_model_path, "PnC BERT GGUF path", {"--pnc-model"});
        p.Register("cpu_workers", &cpu_workers, "Final-result postprocessing worker count");
        p.Register("max_queue_depth", &max_queue_depth, "Maximum queued postprocessing results");
    }
};

class Postprocessor {
   public:
    // `bm` is required to load the PnC model (BERT on the ggml runtime); pass
    // null when PnC isn't configured. `model_self_punctuates` true means the
    // acoustic model already emits its own capitalization + punctuation: the PnC
    // BERT (built for plain-text CTC output) is then skipped, since stacking it
    // doubles/garbles punctuation.
    explicit Postprocessor(
        const PostprocConfig& cfg, ggml_runtime::BackendManager* bm = nullptr,
        bool model_self_punctuates = false, const BatchingConfig& batching = {});
    ~Postprocessor();

    // Apply the chain to `transcript` for one request. If `words` is non-null,
    // it is kept consistent with the returned text: profanity masks per-word
    // strings and ITN remaps spans onto the rewritten words. PnC does not alter spans.
    std::string apply(
        const std::string& transcript, const AsrRequestOptions& opts,
        std::vector<WordTiming>* words = nullptr, const std::string& language_code = "") const;
    BatchMetrics pnc_batch_metrics() const;

   private:
    std::string apply_cpu(
        const std::string& transcript, const AsrRequestOptions& opts,
        std::vector<WordTiming>* words, const std::string& language_code) const;

    class Executor;
    struct ItnRegistry;
    Profanity profanity_;
    std::unique_ptr<ItnRegistry> itn_;
    std::unique_ptr<pnc::PncModel> pnc_model_;
    std::unique_ptr<pnc::PncRunner> pnc_runner_;
    mutable std::unique_ptr<Executor> executor_;
};

}  // namespace nemo_speech::asr::postproc
