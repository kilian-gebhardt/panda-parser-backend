//
// Created by kilian on 01/03/17.
//

#ifndef STERMPARSER_SPLITMERGETRAINER_H
#define STERMPARSER_SPLITMERGETRAINER_H

#include <cstddef>
#include <vector>
#include <functional>
#include "TrainingCommon.h"
#include "EMTrainerLA.h"

namespace Trainer {
    class Splitter {
        const double randPercent;
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;

        double rand_split() {
            return fRand((100 - randPercent) / 100.0, (100 + randPercent) / 100.0);
        }

    public:
        Splitter(
                double randPercent
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::shared_ptr<StorageManager> storageManager
        )
                : randPercent(randPercent), grammarInfo(grammarInfo), storageManager(storageManager) {}

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

            std::cerr << "la root weights: " << std::endl << la.rootWeights << std::endl;
            // new root weights
            Eigen::Tensor<double, 1> rootWeights(la.rootWeights.dimension(0) * 2);
            rootWeights = la.rootWeights.broadcast(Eigen::array<long, 1>{2});
            rootWeights = rootWeights.unaryExpr([this](double x) { return x * rand_split(); });
            // normalization
            Eigen::Tensor<double, 0> total_root_weight = rootWeights.sum();
            rootWeights = rootWeights.unaryExpr([&total_root_weight](double x) { return x / total_root_weight(0); });
            std::cerr << "split root weights: " << std::endl << rootWeights << std::endl;

            // new unnormalized rule weights
            std::vector<RuleTensor<double>> ruleWeights;
            ruleWeights.reserve(la.ruleWeights.size());
            for (const RuleTensor<double> &rule_weight : la.ruleWeights)
                ruleWeights.push_back(create_split_tensor(rule_weight));
            // normalization
            unsigned nont = 0;
            for (auto &group : grammarInfo->normalizationGroups) {
                Eigen::Tensor<double, 1> normalizationDivisor(nonterminalSplits[nont]);
                normalizationDivisor.setZero();
                for (size_t ruleId : group) {
                    compute_normalization_divisor(normalizationDivisor, ruleWeights[ruleId]);

                }
                for (size_t ruleId : group) {
                    normalize(ruleWeights[ruleId], ruleWeights[ruleId], normalizationDivisor);
                }
            }

            return LatentAnnotation(nonterminalSplits, rootWeights, ruleWeights);
        };

    private:
        RuleTensor<double> create_split_tensor(const RuleTensor<double> wrappedTensor) {
            switch (wrappedTensor.which() + 1) {
                case 1:
                    return create_split_tensor_ranked<1>(wrappedTensor);
                case 2:
                    return create_split_tensor_ranked<2>(wrappedTensor);
                case 3:
                    return create_split_tensor_ranked<3>(wrappedTensor);
                case 4:
                    return create_split_tensor_ranked<4>(wrappedTensor);
                default:
                    abort();
            }
        }

        template<long rule_rank>
        RuleTensor<double> create_split_tensor_ranked(const RuleTensor<double> tensorWrapped) {
            const auto &tensorRaw = boost::get<RuleTensorRaw<double, rule_rank>>(tensorWrapped);
            Eigen::array<Eigen::DenseIndex, rule_rank> splitDimenions = tensorRaw.dimensions();
            Eigen::array<Eigen::DenseIndex, rule_rank> broadcast;
            std::fill(broadcast.begin(), broadcast.end(), 2);
            std::for_each(splitDimenions.begin(), splitDimenions.end(), [](auto &dim) { dim = 2 * dim; });
            size_t memory = std::accumulate(
                    splitDimenions.cbegin()
                    , splitDimenions.cend()
                    , (size_t) 1
                    , std::multiplies<size_t>());
            auto tensorSplit = storageManager
                    ->create_uninitialized_tensor_ranked_typed<RuleTensorRaw<double, rule_rank>>(
                            memory
                            , splitDimenions
                    );
            tensorSplit = tensorRaw.broadcast(broadcast);
            tensorSplit = tensorSplit.unaryExpr([this](double x) { return x * rand_split(); });
            return tensorSplit;
        }

    };

    template<typename Nonterminal, typename TraceID>
    class MergePreparator;

    class Merger {
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;
        const bool debug;

    public:
        Merger(std::shared_ptr<const GrammarInfo2> grammarInfo
               , std::shared_ptr<StorageManager> storageManager
               , bool debug=false)
                : grammarInfo(grammarInfo), storageManager(storageManager), debug(debug) {}

        LatentAnnotation merge(const LatentAnnotation &la, const MergeInfo &mergeInfo) {
            // root weights
            Eigen::Tensor<double, 1> rootWeights(mergeInfo.nontSplitsAfterMerge[grammarInfo->start]);
            for (Eigen::DenseIndex idx = 0; idx < rootWeights.dimension(0); ++idx) {
                rootWeights(idx) = 0;
                for (size_t idx_origin : mergeInfo.mergeSources[grammarInfo->start][idx])
                    rootWeights(idx) += la.rootWeights(idx_origin);
            }

            if (debug) std::cerr << "root weights " << rootWeights << std::endl;

            // rule weights
            std::vector<RuleTensor<double>> ruleWeights;
            for (size_t rule_id = 0; rule_id < grammarInfo->rule_to_nonterminals.size(); ++rule_id) {
                RuleTensor<double> merged_tensor = storageManager->create_uninitialized_tensor(
                        rule_id
                        , *grammarInfo
                        , mergeInfo.nontSplitsAfterMerge
                );
                merge_tensor(merged_tensor, la.ruleWeights[rule_id], rule_id, mergeInfo);
                if (debug) std::cerr << rule_id << " " << merged_tensor << std::endl;
                ruleWeights.push_back(std::move(merged_tensor));
            }

            return LatentAnnotation(mergeInfo.nontSplitsAfterMerge, rootWeights, ruleWeights);
        }

    private:
        inline void merge_tensor(
                RuleTensor<double> &mergedTensor
                , const RuleTensor<double> &sourceTensor
                , const size_t ruleId
                , const MergeInfo &mergeInfo
        ) {
            switch (mergedTensor.which() + 1) {
                case 1:
                    merge_tensor_ranked<1>(mergedTensor, sourceTensor, ruleId, mergeInfo);
                    break;
                case 2:
                    merge_tensor_ranked<2>(mergedTensor, sourceTensor, ruleId, mergeInfo);
                    break;
                case 3:
                    merge_tensor_ranked<3>(mergedTensor, sourceTensor, ruleId, mergeInfo);
                    break;
                case 4:
                    merge_tensor_ranked<4>(mergedTensor, sourceTensor, ruleId, mergeInfo);
                    break;
                default:
                    abort();
            }
        }

        template<long rank>
        inline void merge_tensor_ranked(
                RuleTensor<double> &mergedTensor
                , const RuleTensor<double> &sourceTensor
                , const size_t ruleId
                , const MergeInfo &mergeInfo
        ) {
            auto &mergedTensorRaw = boost::get<RuleTensorRaw<double, rank>>(mergedTensor);
            const auto &sourceTensorRaw = boost::get<RuleTensorRaw<double, rank>>(sourceTensor);

            for (TensorIterator<rank> tensorIteraror{&mergedTensorRaw};
                 tensorIteraror != tensorIteraror.end(); ++tensorIteraror) {
                *tensorIteraror = 0;
                for (MergeIterator<rank, true> mergeIterator(
                        &sourceTensorRaw
                        , ruleId
                        , &(tensorIteraror.get_index())
                        , &mergeInfo
                        , &(grammarInfo->rule_to_nonterminals)
                ); mergeIterator != mergeIterator.end(); ++mergeIterator) {
                    *tensorIteraror += *mergeIterator * mergeIterator.mergeFactor();
                }
            }
        }
    };

    template<typename Nonterminal, typename TraceID>
    class SplitMergeTrainer {
        std::shared_ptr<EMTrainerLA> emTrainer;
        std::shared_ptr<Splitter> splitter;
        std::shared_ptr<MergePreparator<Nonterminal, TraceID>> mergePreparator;
        std::shared_ptr<Merger> merger;
        const bool debug;

    public:
        SplitMergeTrainer(
                std::shared_ptr<EMTrainerLA> emTrainer
                , std::shared_ptr<Splitter> splitter
                , std::shared_ptr<MergePreparator<Nonterminal, TraceID>> mergePreparator
                , std::shared_ptr<Merger> merger
                , bool debug = false
        ) : emTrainer(emTrainer), splitter(splitter), mergePreparator(mergePreparator), merger(merger), debug(debug) {}

        LatentAnnotation split_merge_cycle(LatentAnnotation la) {
            auto laSplit = splitter->split(la);
            emTrainer->train(laSplit);
            auto mergeInfo = mergePreparator->mergePrepare(laSplit);

            if (debug) {
                std::cerr << mergeInfo;
                std::cerr << "rules weights before merge" << std::endl;
                size_t rule_id{0};
                for (auto ruleTensor : laSplit.ruleWeights) {
                    std::cerr << "rule " << rule_id << std::endl << ruleTensor << std::endl;
                    ++rule_id;
                }
            }

            auto laMerged = merger->merge(laSplit, mergeInfo);

            if (debug) {
                size_t rule_id{0};
                for (const RuleTensor<double> ruleTensor : laMerged.ruleWeights) {
                    std::cerr << "rule " << rule_id << std::endl << ruleTensor << std::endl;
                    ++rule_id;
                }
            }
            emTrainer->train(laMerged);
            return laMerged;
        }

    };

    template<typename Nonterminal, typename TraceID>
    class MergePreparator {
        using TraceIterator = ConstManagerIterator<Trace<Nonterminal, TraceID>>;

        const TraceManagerPtr<Nonterminal, EdgeLabelT> traceManager;
        std::shared_ptr<StorageManager> storageManager;
        std::shared_ptr<const GrammarInfo2> grammarInfo;

        const bool debug;
        std::vector<MAPTYPE<Element<Node<Nonterminal>>, WeightVector>> tracesInsideWeights;
        std::vector<MAPTYPE<Element<Node<Nonterminal>>, WeightVector>> tracesOutsideWeights;

    public:
        MergePreparator(
                TraceManagerPtr<Nonterminal, EdgeLabelT> traceManager
                , std::shared_ptr<StorageManager> storageManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , bool debug = false
        )
                : traceManager(traceManager), storageManager(storageManager), grammarInfo(grammarInfo), debug(debug) {}

        MergeInfo mergePrepare(const LatentAnnotation latentAnnotation) {


            std::vector<WeightVector> nonterminalFrequencies;
            for (size_t nont = 0; nont < latentAnnotation.nonterminalSplits.size(); ++nont) {
                WeightVector mw
                        = storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[nont]);
                mw.setZero();
                nonterminalFrequencies.push_back(mw);
            }

            if (tracesInsideWeights.size() < traceManager->size())
                tracesInsideWeights.resize(traceManager->size());

            if (tracesOutsideWeights.size() < traceManager->size())
                tracesOutsideWeights.resize(traceManager->size());

            estimateNontFreqLA(
                    traceManager->cbegin()
                    , traceManager->cend()
                    , nonterminalFrequencies
                    , latentAnnotation
            );

            std::vector<std::vector<double>> mergeFactors = computeMergeFactors(nonterminalFrequencies);

            std::vector<std::vector<double>> mergeDelta;
            for (auto split : latentAnnotation.nonterminalSplits) {
                mergeDelta.emplace_back(split / 2, 1.0);
            }

            computeMergeDeltas(
                    traceManager->cbegin()
                    , traceManager->cend()
                    , mergeFactors
                    , latentAnnotation.nonterminalSplits
                    , mergeDelta
            );

            const double merge_threshold = computeMergeThreshold(mergeDelta);

            // clean up
            storageManager->free_weight_maps(tracesInsideWeights);
            storageManager->free_weight_maps(tracesOutsideWeights);
            for (WeightVector &weightVector : nonterminalFrequencies) {
                storageManager->free_weight_vector(weightVector);
            }
            nonterminalFrequencies.clear();

            return build_merge_info(
                    std::move(mergeFactors)
                    , merge_threshold
                    , mergeDelta
                    , latentAnnotation.nonterminalSplits
            );
        }

    private:
        inline void estimateNontFreqLA(
                const TraceIterator start
                , const TraceIterator stop
                , std::vector<WeightVector> &nonterminalFrequencies
                , const LatentAnnotation &latentAnnotation
        ) {
            // computing in(A_x) * out(A_x) for every A ∈ N and x ∈ X_A
            for (TraceIterator traceIterator = start; traceIterator < stop; ++traceIterator) {

                if (tracesInsideWeights[traceIterator - traceManager->cbegin()].size() !=
                    traceIterator->get_hypergraph()->size()) {
                    for (const auto &node : *(traceIterator->get_hypergraph())) {
                        tracesInsideWeights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                        tracesOutsideWeights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                    }
                }

                traceIterator->io_weights_la(
                        latentAnnotation.ruleWeights
                        , latentAnnotation.rootWeights
                        , tracesInsideWeights[traceIterator - traceManager->cbegin()]
                        , tracesOutsideWeights[traceIterator - traceManager->cbegin()]
                );

                const auto &insideWeights = tracesInsideWeights[traceIterator - traceManager->cbegin()];
                const auto &outsideWeights = tracesOutsideWeights[traceIterator - traceManager->cbegin()];

                for (const Element<Node<Nonterminal>> &node : *(traceIterator->get_hypergraph())) {

                    const auto &insideWeight = insideWeights.at(node);
                    const auto &outsideWeight = outsideWeights.at(node);

                    const auto vals = insideWeight * outsideWeight;
                    Eigen::Tensor<double, 0> denominator = vals.sum();
                    Eigen::Tensor<double, 1> fraction
                            = vals.unaryExpr([denominator](double x) { return x / denominator(0); });
                    Eigen::Tensor<bool, 0> nan = fraction.isnan().any();
                    Eigen::Tensor<bool, 0> inf = fraction.isinf().any();
                    if (not nan(0) and not inf(0)) {
                        auto &target = nonterminalFrequencies[node->get_label_id()];
                        target += fraction;
                    }
                }
            }
        }

        inline std::vector<std::vector<double>> computeMergeFactors(const std::vector<WeightVector> &mergeWeights) {
            std::cerr << "Computing merge factors." << std::endl;
            std::vector<std::vector<double>> p;
            for (auto las_weights : mergeWeights) {
                p.emplace_back(std::vector<double>());
                const size_t half_splits{las_weights.dimension(0) / 2};
                for (unsigned i = 0; i < half_splits; ++i) {
                    double combined_weight = las_weights(i) + las_weights(i + half_splits);
                    if ((not std::isnan(combined_weight)) and combined_weight > 0) {
                        p.back().push_back(las_weights(i) / combined_weight);
                        p.back().push_back(las_weights(i + half_splits) / combined_weight);
                    } else {
                        p.back().push_back(0.5);
                        p.back().push_back(0.5);
                    }
                }
            }
            return p;
        }

        inline void computeMergeDeltas(
                const TraceIterator start
                , const TraceIterator stop
                , const std::vector<std::vector<double>> &p
                , const std::vector<size_t> &nontDimensions
                , std::vector<std::vector<double>> &mergeDelta
        ) const {

            // prefix and postfix sums are used for efficient computation of
            // s(i) = sum_{j ∈ {0, …, i-1, i+1, …, n-1}} a_j
            // for each i ∈ {0, …, n-1}
            std::vector<double> prefixSums;
            std::vector<double> postfixSums;

            for (TraceIterator trace_id = start; trace_id < stop; ++trace_id) {
                const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &insideWeights = tracesInsideWeights[
                        trace_id - traceManager->cbegin()];
                const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &outsideWeights = tracesOutsideWeights[
                        trace_id - traceManager->cbegin()];

                for (const Element<Node<Nonterminal>> &node : *(trace_id->get_hypergraph())) {
                    const size_t nont = node->get_label_id();
                    const size_t nontDim = nontDimensions[nont];
                    const size_t halfDim = nontDim / 2;

                    const auto & insideWeight = insideWeights.at(node);
                    const auto & outsideWeight = outsideWeights.at(node);

                    prefixSums.resize(halfDim, 0.0);
                    postfixSums.resize(halfDim, 0.0);
                    double denominator = 0;
                    {
                        const size_t idx = halfDim - 1;
                        const double in1 = insideWeight(idx);
                        const double in2 = insideWeight(idx + halfDim);
                        const double out1 = outsideWeight(idx);
                        const double out2 = outsideWeight(idx + halfDim);
                        denominator += in1 * out1 + in2 * out2;
                    }
                    for (size_t idx = 0; idx < halfDim - 1; ++idx) {
                        const double in1 = insideWeight(idx);
                        const double in2 = insideWeight(idx + halfDim);
                        const double out1 = outsideWeight(idx);
                        const double out2 = outsideWeight(idx + halfDim);
                        prefixSums[idx + 1] = prefixSums[idx] + in1 * out1 + in2 * out2;
                        denominator += in1 * out1 + in2 * out2;
                    }

                    for (size_t idx = halfDim - 1; idx > 0; --idx) {
                        const double in1 = insideWeight(idx);
                        const double in2 = insideWeight(idx + halfDim);
                        const double out1 = outsideWeight(idx);
                        const double out2 = outsideWeight(idx + halfDim);
                        postfixSums[idx - 1] = postfixSums[idx] + in1 * out1 + in2 * out2;
                    }

                    // inside weight of some nodes can be zero in certain LA-dimensions
                    // since LA-rule weights may converge to zero
                    // we ignore those dimensions in Δ computation
                    if (denominator == 0)
                        continue;

                    for (unsigned idx = 0; idx < halfDim; ++idx) {
                        const double in1 = insideWeight(idx);
                        const double in2 = insideWeight(idx + halfDim);
                        const double out1 = outsideWeight(idx);
                        const double out2 = outsideWeight(idx + halfDim);
                        const double p1 = p[nont][idx];
                        const double p2 = p[nont][idx + halfDim];

                        const double inMerged = (p1 * in1) + (p2 * in2);
                        const double outMerged = out1 + out2;

                        const double Q = (prefixSums[idx] + postfixSums[idx] + inMerged * outMerged) / denominator;

                        if (std::isnan(Q)) {
                            std::cerr << "bad fraction " << Q << " where" << std::endl;
                            std::cerr << "prefix  " << prefixSums[idx] << std::endl;
                            std::cerr << "postfix " << postfixSums[idx] << std::endl;
                            std::cerr << "merged  " << inMerged * outMerged << std::endl;
                            std::cerr << "denom   " << denominator << std::endl;

                            assert(!std::isnan(Q));
                        }

                        double &delta = mergeDelta[nont][idx];
                        delta *= Q;
                    }

                    prefixSums.clear();
                    postfixSums.clear();
                }
            }
        }

        virtual double computeMergeThreshold(const std::vector<std::vector<double>> &mergeDelta) = 0;

        // evaluate Δ and build MergeInfo accordingly
        MergeInfo build_merge_info(
                const std::vector<std::vector<double>> &&merge_factors
                , const double merge_threshold
                , const std::vector<std::vector<double>> &merge_delta
                , const std::vector<size_t> &nontSplits
        ) {
            std::vector<std::vector<std::vector<size_t>>> mergeSelection;
            std::vector<size_t> nontSplitsAfterMerge;
            unsigned nont = 0;
            unsigned merges = 0;
            unsigned splits = 0;

            if (debug) std::cerr << "merge deltas: ";
            for (const auto &delta : merge_delta) {
                if (debug) std::cerr << " { ";
                mergeSelection.push_back(std::vector<std::vector<size_t >>());
                const size_t halfSplits = nontSplits[nont] / 2;
                for (size_t split = 0; split < halfSplits; ++split) {
                    if (debug) std::cerr << delta[split] << " ";
                    if (delta[split] >= merge_threshold - 0.00001
                        // always merge if Δ >= 1
                        || delta[split] >= 1 - 0.00001
                        // always merge initial symbol
                        || grammarInfo->start == nont) {
                        mergeSelection.back().emplace_back();
                        mergeSelection.back().back().push_back(split);
                        mergeSelection.back().back().push_back(split + halfSplits);
                        ++merges;
                    } else {
                        mergeSelection.back().emplace_back(1, split);
                        mergeSelection.back().emplace_back(1, split + halfSplits);
                        ++splits;
                    }
                }
                if (debug) std::cerr << " } ";
                ++nont;
                nontSplitsAfterMerge.push_back(mergeSelection.back().size());
            }
            if (debug) std::cerr << std::endl;

            std::cerr << "Merging " << merges << " of " << merges + splits << " splits. Merge threshold is "
                      << merge_threshold << std::endl;

            return MergeInfo(std::move(mergeSelection), std::move(nontSplitsAfterMerge), std::move(merge_factors));
        }
    };

    template<typename Nonterminal, typename TraceID>
    class ThresholdMergePreparator : public MergePreparator<Nonterminal, TraceID> {
        const double merge_threshold;

    public:
        ThresholdMergePreparator(
                TraceManagerPtr<Nonterminal, TraceID> traceManager
                , std::shared_ptr<StorageManager> storageManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , double merge_threshold
                , bool debug = false
        )
                : MergePreparator<Nonterminal, TraceID>(traceManager, storageManager, grammarInfo, debug),
                  merge_threshold(merge_threshold) {}

    protected:
        double computeMergeThreshold(const std::vector<std::vector<double>> &merge_delta) {
            std::cerr << "Selecting merges ";
            std::cerr << "above threshold " << merge_threshold;
            std::cerr << std::endl;
            return merge_threshold;
        }
    };

    template<typename Nonterminal, typename TraceID>
    class PercentMergePreparator : public MergePreparator<Nonterminal, TraceID> {
        const double mergePercent;

    public:
        PercentMergePreparator(
                TraceManagerPtr<Nonterminal, TraceID> traceManager
                , std::shared_ptr<StorageManager> storageManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , double mergePercent
                , bool debug = false
        ) : MergePreparator<Nonterminal, TraceID>(traceManager, storageManager, grammarInfo, debug)
                , mergePercent(mergePercent) {}

    protected:
        double computeMergeThreshold(const std::vector<std::vector<double>> &mergeDelta) {
            std::cerr << "Selecting merges ";
            std::cerr << "best " << mergePercent << " % ";
            std::cerr << std::endl;

            std::vector<double> orderedMergeWeights;

            // order merges according to likelihood_loss
            for (const auto &delta : mergeDelta) {
                orderedMergeWeights.insert(
                        std::end(orderedMergeWeights)
                        , std::begin(delta)
                        , std::end(delta));
            }

            std::sort(std::begin(orderedMergeWeights), std::end(orderedMergeWeights), std::greater<double>());

            // todo: option to skip over merge_weights >= 1

            size_t index = (size_t) (mergePercent / 100.0 * orderedMergeWeights.size());
            if (index > orderedMergeWeights.size())
                index = orderedMergeWeights.size() - 1;

            std::cerr << "index for ordered merges " << index << " / " << orderedMergeWeights.size() << std::endl;

            return orderedMergeWeights[index];
        }
    };

}


#endif //STERMPARSER_SPLITMERGETRAINER_H
