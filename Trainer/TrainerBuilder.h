//
// Created by kilian on 03/03/17.
//

#ifndef STERMPARSER_TRAINERBUILDER_H
#define STERMPARSER_TRAINERBUILDER_H

#include "EMTrainer.h"
#include "SplitMergeTrainer.h"
#include "EMTrainerLA.h"

namespace Trainer {
    class EMTrainerBuilder {
    public:
        template<typename Nonterminal, typename TraceID>
        EMTrainer<Nonterminal, TraceID> build_em_trainer(TraceManagerPtr<Nonterminal, TraceID> traceManager) {
            return EMTrainer<Nonterminal, TraceID>(traceManager);
        }
    };

    template<typename Nonterminal, typename TraceID>
    class SplitMergeTrainerBuilder {
        TraceManagerPtr<Nonterminal, TraceID> traceManager;
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;
        std::shared_ptr<Expector> expector;
        std::shared_ptr<Maximizer> maximizer;
        std::shared_ptr<EMTrainerLA> emTrainer;
        std::shared_ptr<Splitter> splitter;
        std::shared_ptr<MergePreparator> mergePreparator;
        std::shared_ptr<Merger> merger;
        unsigned em_epochs{20};
        unsigned THREADS{1};


    public:
        SplitMergeTrainerBuilder(
                TraceManagerPtr<Nonterminal, TraceID> traceManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::shared_ptr<StorageManager> storageManager = std::make_shared<StorageManager>()
        ) : traceManager(traceManager), grammarInfo(grammarInfo), storageManager(storageManager) {}

        SplitMergeTrainerBuilder &set_threads(unsigned THREADS) {
            this->THREADS = THREADS;
            return *this;
        }

        SplitMergeTrainerBuilder &set_simple_expector() {
            return set_simple_expector(THREADS);
        }

        SplitMergeTrainerBuilder &set_simple_expector(unsigned threads) {
            expector = std::make_shared<SimpleExpector<Nonterminal, TraceID>>(
                    traceManager
                    , grammarInfo
                    , storageManager
                    , threads
            );
            return *this;
        }

        SplitMergeTrainerBuilder &set_simple_maximizer() {
            return set_simple_maximizer(THREADS);
        }

        SplitMergeTrainerBuilder &set_simple_maximizer(unsigned threads) {
            maximizer = std::make_shared<SimpleMaximizer>(grammarInfo, threads);
            return *this;
        }

        SplitMergeTrainerBuilder &set_em_epochs(unsigned epochs) {
            em_epochs = epochs;
            return *this;
        }

        SplitMergeTrainerBuilder &set_merge_nothing() {
            mergePreparator = std::make_shared<MergeNothingMergePreparator>(grammarInfo);
            return *this;
        }

        SplitMergeTrainerBuilder &set_percent_merger(double percent = 50.0) {
            return set_percent_merger(percent, THREADS);
        }

        SplitMergeTrainerBuilder &set_percent_merger(double percent, unsigned threads) {
            mergePreparator = std::make_shared<PercentMergePreparator<Nonterminal, TraceID>>(
                    traceManager
                    , storageManager
                    , grammarInfo
                    , percent
                    , threads
            );
            return *this;
        }

        SplitMergeTrainerBuilder &set_threshold_merger(double threshold = 0.5) {
            return set_threshold_merger(threshold, THREADS);
        }
        SplitMergeTrainerBuilder &set_threshold_merger(double threshold, unsigned threads) {
            mergePreparator = std::make_shared<ThresholdMergePreparator<Nonterminal, TraceID>>(
                    traceManager
                    , storageManager
                    , grammarInfo
                    , threshold
                    , threads
            );
            return *this;
        }

        SplitMergeTrainerBuilder &set_split_randomization(double percent = 1.0) {
            splitter = std::make_shared<Splitter>(percent, grammarInfo, storageManager);
            return *this;
        }


        SplitMergeTrainer<Nonterminal, TraceID>
        build() {
            if (expector == nullptr)
                set_simple_expector();
            if (maximizer == nullptr)
                set_simple_maximizer();
            emTrainer = std::make_shared<EMTrainerLA>(em_epochs, expector, maximizer);

            if (splitter == nullptr)
                set_split_randomization();
            if (mergePreparator == nullptr)
                set_percent_merger();
            if (merger == nullptr)
                merger = std::make_shared<Merger>(grammarInfo, storageManager);


            return SplitMergeTrainer<Nonterminal, TraceID>(emTrainer, splitter, mergePreparator, merger);
        };
    };
}

#endif //STERMPARSER_TRAINERBUILDER_H
