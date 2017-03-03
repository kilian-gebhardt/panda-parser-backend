//
// Created by kilian on 03/03/17.
//

#ifndef STERMPARSER_TRAINERFACTORY_H
#define STERMPARSER_TRAINERFACTORY_H
#include "EMTrainer.h"
#include "SplitMergeTrainer.h"
#include "EMTrainerLA.h"

namespace Trainer {
    class TrainerFactory {
    public:
        template<typename Nonterminal, typename TraceID>
        EMTrainer<Nonterminal, TraceID> build_em_trainer(TraceManagerPtr<Nonterminal, TraceID> traceManager) {
            return EMTrainer<Nonterminal, TraceID>(traceManager);
        }

        template<typename Nonterminal, typename TraceID>
        SplitMergeTrainer <Nonterminal, TraceID>
        build_split_merge_trainer(
                TraceManagerPtr<Nonterminal, TraceID> traceManager
                , std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo
                , unsigned epochs
        ) {
            auto storageManager = std::make_shared<StorageManager>();
            auto expector = std::make_shared<SimpleExpector<Nonterminal, TraceID>>(traceManager, grammarInfo, storageManager);
            auto maximizer = std::make_shared<SimpleMaximizer<Nonterminal>>(grammarInfo);
            auto emTrainer = std::make_shared<EMTrainerLA> (epochs, expector, maximizer);
            auto splitter = std::make_shared<Splitter<Nonterminal>>(1.0, grammarInfo, storageManager);
            auto mergePreparator = std::make_shared<PercentMergePreparator<Nonterminal, TraceID>>(traceManager, storageManager, 50.0);
            auto merger = std::make_shared<Merger<Nonterminal>>(grammarInfo, storageManager);
            return SplitMergeTrainer<Nonterminal, TraceID>(emTrainer, splitter, mergePreparator, merger);
        }
    };
}

#endif //STERMPARSER_TRAINERFACTORY_H
