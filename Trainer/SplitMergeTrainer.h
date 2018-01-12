//
// Created by kilian on 01/03/17.
//

#ifndef STERMPARSER_SPLITMERGETRAINER_H
#define STERMPARSER_SPLITMERGETRAINER_H

#include <cstddef>
#include <vector>
#include <functional>
#include <random>
#include "TrainingCommon.h"
#include "EMTrainerLA.h"
#include "GrammarInfo.h"
#include "MergePreparator.h"
#include "Smoother.h"

namespace Trainer {
    class Splitter {
        const double randPercent;
        std::mt19937 generator;
        std::uniform_real_distribution<double> distribution;
    public:
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;
        inline double rand_split() {
            return distribution(generator);
        }

        Splitter(
                double randPercent
                , unsigned seed
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::shared_ptr<StorageManager> storageManager
        )
                : randPercent(randPercent)
                , generator(seed)
                , distribution((100.0 - randPercent) / 100.0, (100.0 + randPercent) / 100.0)
                , grammarInfo(grammarInfo)
                , storageManager(storageManager) {}

        LatentAnnotation split(const LatentAnnotation &la) {
            std::vector<size_t> nonterminalSplits;
            // double nonterminal splits
            nonterminalSplits.reserve(la.nonterminalSplits.size());
            std::transform(
                    la.nonterminalSplits.cbegin()
                    , la.nonterminalSplits.cend()
                    , std::back_inserter(nonterminalSplits)
                    , [](auto x) { return x * 2; }
            );

//            std::cerr << "la root weights: " << std::endl << la.rootWeights << std::endl;
            // new root weights
            Eigen::Tensor<double, 1> rootWeights(la.rootWeights.dimension(0) * 2);
            rootWeights = la.rootWeights.broadcast(Eigen::array<long, 1>{2});
            rootWeights = rootWeights.unaryExpr([this](double x) { return x * rand_split(); });
            // normalization
            Eigen::Tensor<double, 0> total_root_weight = rootWeights.sum();
            rootWeights = rootWeights.unaryExpr([&total_root_weight](double x) { return x / total_root_weight(0); });
//            std::cerr << "split root weights: " << std::endl << rootWeights << std::endl;

            // new unnormalized rule weights
            auto ruleWeights = std::make_unique<std::vector<RuleTensor<double>>>();
            ruleWeights->reserve(la.ruleWeights->size());
            for (const RuleTensor<double> &rule_weight : *la.ruleWeights)
                ruleWeights->push_back(create_split_tensor(rule_weight));

            // normalization
            for (size_t nont = 0; nont < grammarInfo->normalizationGroups.size(); ++nont) {
                auto & group = grammarInfo->normalizationGroups[nont];
                Eigen::Tensor<double, 1> normalizationDivisor(nonterminalSplits[nont]);
                normalizationDivisor.setZero();
                for (size_t ruleId : group) {
                    compute_normalization_divisor(normalizationDivisor, (*ruleWeights)[ruleId]);

                }
                for (size_t ruleId : group) {
                    normalize((*ruleWeights)[ruleId], (*ruleWeights)[ruleId], normalizationDivisor);
                }
            }

            return LatentAnnotation(nonterminalSplits, std::move(rootWeights), std::move(ruleWeights), la.grammarInfo);
        };

    void reset_random_seed(unsigned seed) {
            generator = std::mt19937(seed);
        }

    private:
        RuleTensor<double> create_split_tensor(const RuleTensor<double> &wrappedTensor);
    };

    struct SplitTensorCreator : boost::static_visitor<RuleTensor<double>> {
            Splitter & splitter;
            SplitTensorCreator(Splitter & splitter) : splitter(splitter) {}
            template<int rule_rank>
            RuleTensor<double> operator()(const RuleTensorRaw<double, rule_rank> &tensorRaw) {
                Eigen::array<Eigen::DenseIndex, rule_rank> splitDimensions = tensorRaw.dimensions();
                Eigen::array<Eigen::DenseIndex, rule_rank> broadcast;
                std::fill(broadcast.begin(), broadcast.end(), 2);
                std::for_each(splitDimensions.begin(), splitDimensions.end(), [](auto &dim) { dim = 2 * dim; });
                auto tensorSplit = this->splitter.storageManager
                        ->create_uninitialized_tensor_ranked_typed<RuleTensorRaw<double, rule_rank>>(splitDimensions);
                tensorSplit = tensorRaw.broadcast(broadcast);
                tensorSplit = tensorSplit.unaryExpr([this](double x) { return x * this->splitter.rand_split(); });
                return tensorSplit;
            }
    };

    RuleTensor<double> Splitter::create_split_tensor(const RuleTensor<double> &wrappedTensor) {
        SplitTensorCreator splitTensorCreator(*this);
        return boost::apply_visitor(splitTensorCreator, wrappedTensor);
    }

    struct TensorMerger : boost::static_visitor<void> {
        const RuleTensor<double> &sourceTensor;
        const size_t ruleId;
        const MergeInfo &mergeInfo;
        const GrammarInfo2 & grammarInfo;
        TensorMerger( const RuleTensor<double> &sourceTensor
                      , const size_t ruleId
                      , const MergeInfo &mergeInfo
                      , const GrammarInfo2 & grammarInfo) :
                sourceTensor(sourceTensor), ruleId(ruleId), mergeInfo(mergeInfo), grammarInfo(grammarInfo) {}

        template<int rank>
        inline void operator()( RuleTensorRaw<double, rank> & mergedTensorRaw) {
            const auto &sourceTensorRaw = boost::get<RuleTensorRaw<double, rank>>(sourceTensor);

            for (TensorIteratorLowToHigh<rank> tensorIterator{&mergedTensorRaw};
                 tensorIterator != tensorIterator.end(); ++tensorIterator) {
                *tensorIterator = 0.0;
                for (MergeIterator<rank, true> mergeIterator(
                        &sourceTensorRaw
                        , ruleId
                        , &(tensorIterator.get_index())
                        , &mergeInfo
                        , &(grammarInfo.rule_to_nonterminals)
                ); mergeIterator != mergeIterator.end(); ++mergeIterator) {
                    /* (for debugging)
                    if (ruleId == 1417) {
                        std::cerr << ruleId << "goal [";
                        for (auto idx : tensorIterator.get_index())
                            std::cerr << idx << " ";
                        std::cerr << "] src: [";
                        for (auto idx : mergeIterator.get_source_index())
                            std::cerr << idx << " ";
                        std::cerr << "] value: " << *mergeIterator << "factor: " << mergeIterator.mergeFactor()
                                  << std::endl;
                    } */
                    *tensorIterator += *mergeIterator * mergeIterator.mergeFactor();
                }
            }
        }
    };

    class Merger {
        const GrammarInfo2& grammarInfo;
        std::shared_ptr<StorageManager> storageManager;
        const bool debug;

    public:
        Merger(
                const GrammarInfo2& grammarInfo
                , std::shared_ptr<StorageManager> storageManager
                , bool debug = false
        )
                : grammarInfo(grammarInfo), storageManager(storageManager), debug(debug) {}

        Merger(
                std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::shared_ptr<StorageManager> storageManager
                , bool debug = false
        )
                : grammarInfo(*grammarInfo), storageManager(storageManager), debug(debug) {}

        LatentAnnotation merge(const LatentAnnotation &la, const MergeInfo &mergeInfo) {
            // root weights
            Eigen::Tensor<double, 1> rootWeights(mergeInfo.nontSplitsAfterMerge[grammarInfo.start]);
            for (Eigen::DenseIndex idx = 0; idx < rootWeights.dimension(0); ++idx) {
                rootWeights(idx) = 0.0;
                for (size_t idx_origin : mergeInfo.mergeSources[grammarInfo.start][idx])
                    rootWeights(idx) += la.rootWeights(idx_origin);
            }

            if (debug) std::cerr << "root weights " << rootWeights << std::endl;

            // rule weights
            auto ruleWeights = std::make_unique<std::vector<RuleTensor<double>>>();
            for (size_t rule_id = 0; rule_id < grammarInfo.rule_to_nonterminals.size(); ++rule_id) {
                RuleTensor<double> merged_tensor = storageManager->create_uninitialized_tensor(
                        rule_id
                        , grammarInfo
                        , mergeInfo.nontSplitsAfterMerge
                );

                TensorMerger tensorMerger((*la.ruleWeights)[rule_id], rule_id, mergeInfo, grammarInfo);
                boost::apply_visitor(tensorMerger, merged_tensor);

                if (debug) std::cerr << rule_id << " " << merged_tensor << std::endl;
                ruleWeights->push_back(std::move(merged_tensor));
            }

            return LatentAnnotation(mergeInfo.nontSplitsAfterMerge
                                    , std::move(rootWeights)
                                    , std::move(ruleWeights)
                                    , la.grammarInfo);
        }
    };

    template<typename Nonterminal, typename TraceID>
    class SplitMergeTrainer {
        std::shared_ptr<EMTrainerLA> emTrainer;
    public:
        std::shared_ptr<Splitter> splitter;
    private:
        std::shared_ptr<MergePreparator> mergePreparator;
        std::shared_ptr<Merger> merger;
        std::shared_ptr<Smoother> smoother;
        std::shared_ptr<MergeInfo> mergeInfo;
        const bool debug;
    public:
        SplitMergeTrainer(
                std::shared_ptr<EMTrainerLA> emTrainer
                , std::shared_ptr<Splitter> splitter
                , std::shared_ptr<MergePreparator> mergePreparator
                , std::shared_ptr<Merger> merger
                , std::shared_ptr<Smoother> smoother
                , bool debug = false
        ) : emTrainer(emTrainer)
                , splitter(splitter)
                , mergePreparator(mergePreparator)
                , merger(merger)
                , smoother(smoother)
                , debug(debug) {}

        void em_train(LatentAnnotation & la) {
            emTrainer->setTrainingMode(Default);
            emTrainer->train(la);
        }

        LatentAnnotation split_merge_cycle(const LatentAnnotation &la) {
            if (not la.is_proper())
                if (debug)
                    abort();

            if (debug) {
                std::cerr << *(splitter->grammarInfo) << std::endl;

                std::cerr << "nonterminal splits: ";
                for (auto nont : la.nonterminalSplits)
                    std::cerr << nont << " ";
                std::cerr << std::endl;
                std::cerr << "rules weights at begin of SM cycle" << std::endl;
                size_t rule_id{0};
                for (const auto &ruleTensor : *la.ruleWeights) {
                    std::cerr << "rule " << rule_id << ":  ";
                    for (auto nont : splitter->grammarInfo->rule_to_nonterminals[rule_id])
                        std::cerr << nont << " ";
                    std::cerr << std::endl << ruleTensor << std::endl;

                    ++rule_id;
                }
                std::cerr << std::endl;
            }

            LatentAnnotation laSplit{splitter->split(la)};
            emTrainer->setTrainingMode(Splitting);
            emTrainer->train(laSplit);

            LatentAnnotation laMerged{merge(laSplit)};

            if (debug) {
                size_t rule_id{0};
                for (const RuleTensor<double> &ruleTensor : *laMerged.ruleWeights) {
                    std::cerr << "rule " << rule_id << std::endl << ruleTensor << std::endl;
                    ++rule_id;
                }
            }

            if (not laMerged.is_proper())
                if (debug)
                    abort();

            emTrainer->setTrainingMode(Merging);
            emTrainer->train(laMerged);

            if (not laMerged.is_proper())
                if (debug)
                    abort();

            // smoothing only if effective
            if (smoother->get_smoothing_factor() > 0.0 or smoother->get_smoothing_factor_unary() > 0.0) {
                smoother->smooth(laMerged);
                emTrainer->setTrainingMode(Smoothing);
                emTrainer->train(laMerged);
                if (not laMerged.is_proper())
                    if (debug)
                        abort();
            }

            return laMerged;
        }

        LatentAnnotation merge(const LatentAnnotation & la) {
            mergeInfo = std::make_shared<MergeInfo>(mergePreparator->merge_prepare(la));

            if (not mergeInfo->is_proper())
                if (debug)
                    abort();

            if (debug) {
                std::cerr << mergeInfo;
                std::cerr << "rules weights before merge" << std::endl;
                size_t rule_id{0};
                for (const auto &ruleTensor : *la.ruleWeights) {
                    std::cerr << "rule " << rule_id << std::endl << ruleTensor << std::endl;
                    ++rule_id;
                }
            }

            return merger->merge(la, *mergeInfo);
        }

        std::vector<std::vector<std::vector<size_t>>> get_current_merge_sources() const {
            if (mergeInfo) {
                return mergeInfo->mergeSources;
            } else {
                return std::vector<std::vector<std::vector<size_t>>>();
            }
        }

    };

}


#endif //STERMPARSER_SPLITMERGETRAINER_H
