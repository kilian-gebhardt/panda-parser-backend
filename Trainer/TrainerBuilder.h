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
        std::shared_ptr<CountsModifier> countsModifier;
        std::shared_ptr<EMTrainerLA> emTrainer;
        std::shared_ptr<Splitter> splitter;
        std::shared_ptr<MergePreparator> mergePreparator;
        std::shared_ptr<Merger> merger;
        std::shared_ptr<Smoother> smoother;
        std::shared_ptr<ValidationLA> validator;
        unsigned maxDrops{6};
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

        SplitMergeTrainerBuilder &set_discriminative_expector(
                TraceManagerPtr<Nonterminal, TraceID> discriminativeTraceManager
                , const double maxScale = std::numeric_limits<double>::infinity()
        ) {
            return set_discriminative_expector(discriminativeTraceManager, maxScale, THREADS);
        }

        SplitMergeTrainerBuilder &set_discriminative_expector(
                TraceManagerPtr<Nonterminal, TraceID> discriminativeTraceManager
                , const double maxScale
                , unsigned threads
        ) {
            if (traceManager->size() == discriminativeTraceManager->size()) {
                expector = std::make_shared<DiscriminativeExpector<Nonterminal, TraceID>>(
                        traceManager
                        , discriminativeTraceManager
                        , grammarInfo
                        , storageManager
                        , maxScale
                        , threads
                );
            } else {
                std::cerr << "Sizes of TraceManagers do not match." << std::endl << "primary: " << traceManager->size()
                          << std::endl << "discriminative: " << discriminativeTraceManager->size() << std::endl;
            }
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

        SplitMergeTrainerBuilder &set_no_count_modification() {
            countsModifier = std::make_shared<CountsModifier>();
            return *this;
        }

        SplitMergeTrainerBuilder &set_count_smoothing(std::vector<size_t> ruleIDs, double smoothValue) {
            countsModifier = std::make_shared<CountsSmoother>(ruleIDs, smoothValue);
            return *this;
        }

        SplitMergeTrainerBuilder &set_simple_validator(
                TraceManagerPtr<Nonterminal, TraceID> validationTraceManager
                , unsigned maxDrops = 6
        ) {
            return set_simple_validator(validationTraceManager, maxDrops, THREADS);
        }

        SplitMergeTrainerBuilder &set_simple_validator(
                TraceManagerPtr<Nonterminal, TraceID> validationTraceManager
                , unsigned maxDrops
                , unsigned threads
        ) {
            validator = std::make_shared<SimpleLikelihoodLA<Nonterminal, TraceID>>(
                    validationTraceManager
                    , grammarInfo
                    , storageManager
                    , threads
            );
            this->maxDrops = maxDrops;
            return *this;
        }

        SplitMergeTrainerBuilder &set_score_validator(
                std::shared_ptr<CandidateScoreValidator<Nonterminal, TraceID>> validator
                , unsigned maxDrops = 6
        ) {
            return set_score_validator(validator, maxDrops, THREADS);
        }

        SplitMergeTrainerBuilder &set_score_validator(
                std::shared_ptr<CandidateScoreValidator<Nonterminal, TraceID>> validator
                , unsigned maxDrops
                , unsigned threads
        ) {
            this->validator = validator;
            this->maxDrops = maxDrops;
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

        SplitMergeTrainerBuilder &set_threshold_merger(double threshold = std::log(0.5)) {
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

        SplitMergeTrainerBuilder &set_scc_merger(double threshold) {
            return set_scc_merger(threshold, THREADS);
        }

        SplitMergeTrainerBuilder &set_scc_merger(double threshold, unsigned threads) {
            std::vector<size_t> relevantNonterminals(grammarInfo->normalizationGroups.size());
            std::iota(std::begin(relevantNonterminals), std::end(relevantNonterminals), 0);
            return set_scc_merger(threshold, relevantNonterminals, threads);
        }

        SplitMergeTrainerBuilder &
        set_scc_merger(double threshold, const std::vector<size_t> &relevantNonterminals, const unsigned threads) {
            mergePreparator = std::make_shared<SCCMerger<Nonterminal, TraceID>>(
                    traceManager
                    , storageManager
                    , grammarInfo
                    , relevantNonterminals
                    , [threshold](const std::vector<double>&) {return threshold;}
                    , threads
            );
            return *this;
        }

        SplitMergeTrainerBuilder & set_scc_merge_threshold_function(ThresholdFunction thresholdFunction) {
            if (mergePreparator)
                mergePreparator->setMergeThresholdFunction(thresholdFunction);
            return *this;
        }

        SplitMergeTrainerBuilder &set_split_randomization(double percent = 1.0, unsigned seed = 0) {
            splitter = std::make_shared<Splitter>(percent, seed, grammarInfo, storageManager);
            return *this;
        }

        SplitMergeTrainerBuilder &set_smoothing_factor(double smoothingFactor = 0.01) {
            smoother = std::make_shared<Smoother>(grammarInfo, smoothingFactor);
            return *this;
        }

        SplitMergeTrainer<Nonterminal, TraceID>
        build() {
            if (expector == nullptr)
                set_simple_expector();
            if (maximizer == nullptr)
                set_simple_maximizer();
            if (countsModifier == nullptr)
                set_no_count_modification();

            if (validator == nullptr)
                emTrainer = std::make_shared<EMTrainerLA>(em_epochs, expector, maximizer, countsModifier);
            else
                emTrainer = std::make_shared<EMTrainerLAValidation>(
                        em_epochs
                        , expector
                        , maximizer
                        , validator
                        , countsModifier
                        , maxDrops
                );

            if (splitter == nullptr)
                set_split_randomization();
            if (mergePreparator == nullptr)
                set_percent_merger();
            if (merger == nullptr)
                merger = std::make_shared<Merger>(grammarInfo, storageManager);
            if (smoother == nullptr)
                set_smoothing_factor();

            return SplitMergeTrainer<Nonterminal, TraceID>(emTrainer, splitter, mergePreparator, merger, smoother);
        }

        const std::shared_ptr<EMTrainerLA> &getEmTrainer() const {
            return emTrainer;
        }
    };

    ThresholdFunction interpolate3rdQuartileMax(double factor) {
        return [factor](const std::vector<double> & quartiles)
            {return quartiles[3] * factor + quartiles[4] * (1.0 -factor);};
    }

}

#endif //STERMPARSER_TRAINERBUILDER_H
