// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Minimal source-compatibility layer for Sparrowhawk's Thrax GrmManager use.
//
// NeMo-Speech.cpp ships precompiled FAR grammars.  It never compiles Thrax
// grammars at run time, and Sparrowhawk only needs the small GrmManager surface
// below: load/enumerate named StdArc FSTs from an STTable FAR, look them up, and
// compose a rule with an input FST. Keeping that functionality here avoids
// linking the full Thrax and fstscript libraries while preserving the
// Sparrowhawk grammar artifact format.
#pragma once

#include <fst/arcsort.h>
#include <fst/compose.h>
#include <fst/extensions/far/far.h>
#include <fst/extensions/pdt/compose.h>
#include <fst/extensions/pdt/pdt.h>
#include <fst/fst.h>
#include <fst/fstlib.h>
#include <fst/matcher-fst.h>
#include <fst/vector-fst.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace thrax {

class GrmManager {
   public:
    using Arc = fst::StdArc;
    using Label = Arc::Label;
    using Transducer = fst::Fst<Arc>;
    using MutableTransducer = fst::VectorFst<Arc>;
    // Load every named graph from the same STTable FAR format used by Thrax.
    bool LoadArchive(const std::string& filename) {
        std::unique_ptr<fst::FarReader<Arc>> reader(fst::STTableFarReader<Arc>::Open(filename));
        if (!reader)
            return false;

        FstMap loaded;
        for (reader->Reset(); !reader->Done(); reader->Next()) {
            const Transducer* fst = reader->GetFst();
            if (fst == nullptr)
                return false;
            auto copy = std::make_unique<MutableTransducer>(*fst);
            if (copy->Properties(fst::kILabelSorted, false) != fst::kILabelSorted)
                fst::ArcSort(copy.get(), fst::ILabelCompare<Arc>());
            loaded.emplace(reader->GetKey(), std::move(copy));
        }
        if (loaded.empty())
            return false;

        fsts_ = std::move(loaded);
        return true;
    }

    const Transducer* GetFst(const std::string& name) const {
        const auto it = fsts_.find(name);
        return it == fsts_.end() ? nullptr : it->second.get();
    }

    // Compose a named rule with an input graph. Sparrowhawk's rule-order proto
    // optionally names a PDT-parentheses graph; support it directly with
    // OpenFST so existing/future FARs do not silently lose that behavior.
    bool Rewrite(
        const std::string& rule, const Transducer& input, MutableTransducer* output,
        const std::string& pdt_parens_rule = "") const {
        if (output == nullptr)
            return false;
        const Transducer* rule_fst = GetFst(rule);
        if (rule_fst == nullptr)
            return false;

        if (pdt_parens_rule.empty()) {
            const fst::ComposeOptions options(true, fst::ALT_SEQUENCE_FILTER);
            fst::Compose(input, *rule_fst, output, options);
            return true;
        }

        const Transducer* parens_fst = GetFst(pdt_parens_rule);
        if (parens_fst == nullptr)
            return false;
        std::vector<std::pair<Label, Label>> parens;
        if (!ReadParenPairs(*parens_fst, &parens))
            return false;
        const fst::PdtComposeOptions options(true, fst::PdtComposeFilter::EXPAND);
        fst::Compose(input, *rule_fst, parens, output, options);
        return true;
    }

   private:
    using FstMap = std::map<std::string, std::unique_ptr<MutableTransducer>>;

    static bool ReadParenPairs(
        const Transducer& fst, std::vector<std::pair<Label, Label>>* parens) {
        std::set<Label> seen;
        for (fst::StateIterator<Transducer> states(fst); !states.Done(); states.Next()) {
            for (fst::ArcIterator<Transducer> arcs(fst, states.Value()); !arcs.Done();
                 arcs.Next()) {
                const Arc& arc = arcs.Value();
                if (arc.ilabel == 0 && arc.olabel == 0)
                    continue;
                if (arc.ilabel == 0 || arc.olabel == 0 || arc.ilabel == arc.olabel ||
                    !seen.insert(arc.ilabel).second || !seen.insert(arc.olabel).second)
                    return false;
                parens->emplace_back(arc.ilabel, arc.olabel);
            }
        }
        return !parens->empty();
    }

    FstMap fsts_;
};

}  // namespace thrax
