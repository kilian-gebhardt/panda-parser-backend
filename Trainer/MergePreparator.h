//
// Created by kilian on 10/03/17.
//

#ifndef STERMPARSER_MERGEPREPARATOR_H
#define STERMPARSER_MERGEPREPARATOR_H

#include <memory>
#include "GrammarInfo.h"
#include "LatentAnnotation.h"
#include "TrainingCommon.h"
#include <omp.h>

namespace Trainer {
    class MergePreparator {
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        const bool debug;

    protected:
        /**
         * Builds MergeInfo according to merge-Δs and threshold.
         * If (merge-Δ > 1 or the split of a start symbol is concerned)
         * then the split is always merged.
         */
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
                    if (delta[split] >= merge_threshold * 0.999
                        // always merge if Δ >= 1
                        || delta[split] >= 0.999
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

    public:
        MergePreparator(std::shared_ptr<const GrammarInfo2> grammarInfo, bool debug = false)
                : grammarInfo(grammarInfo), debug(debug) {}

        virtual MergeInfo merge_prepare(const LatentAnnotation &latentAnnotation) = 0;
    };

    /**
     * Merges none of the splits, expect for start symbol whose splits are always merged.
     */
    class MergeNothingMergePreparator : public MergePreparator {
    public:
        MergeNothingMergePreparator(std::shared_ptr<const GrammarInfo2> grammarInfo, bool debug = false)
                : MergePreparator(grammarInfo, debug) {};

        MergeInfo merge_prepare(const LatentAnnotation &latentAnnotation) {
            std::vector<std::vector<double>> mergeFactors;
            std::vector<std::vector<double>> mergeDelta;

            for (auto splits : latentAnnotation.nonterminalSplits) {
                mergeFactors.emplace_back(splits, 0.5);
                mergeDelta.emplace_back(splits / 2, 0.4);
            }

            double merge_threshold = 0.5;

            return build_merge_info(
                    std::move(mergeFactors)
                    , merge_threshold
                    , mergeDelta
                    , latentAnnotation.nonterminalSplits
            );
        }
    };

    template<typename Nonterminal, typename TraceID>
    class DefaultMergePreparator : public MergePreparator {
        using TraceIterator = ConstManagerIterator<Trace < Nonterminal, TraceID>>;

        const TraceManagerPtr <Nonterminal, EdgeLabelT> traceManager;
        std::shared_ptr<StorageManager> storageManager;
        const unsigned threads;

        std::vector<MAPTYPE<Element<Node<Nonterminal>>, WeightVector>> tracesInsideWeights;
        std::vector<MAPTYPE<Element<Node<Nonterminal>>, WeightVector>> tracesOutsideWeights;

    public:
        DefaultMergePreparator(
                TraceManagerPtr <Nonterminal, EdgeLabelT> traceManager
                , std::shared_ptr<StorageManager> storageManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , unsigned threads = 1
                , bool debug = false
        )
                : MergePreparator(grammarInfo, debug), traceManager(traceManager), storageManager(storageManager),
                  threads(threads) {}

        MergeInfo merge_prepare(const LatentAnnotation &latentAnnotation) {

            // setup temporary data structures
            if (tracesInsideWeights.size() < traceManager->size())
                tracesInsideWeights.resize(traceManager->size());
            if (tracesOutsideWeights.size() < traceManager->size())
                tracesOutsideWeights.resize(traceManager->size());

            std::vector<WeightVector> nonterminalFrequencies{estimateNontFreqLA(latentAnnotation)};
            std::vector<std::vector<double>> mergeFactors{computeMergeFactors(nonterminalFrequencies)};

            std::vector<std::vector<double>> mergeDelta;
            for (auto split : latentAnnotation.nonterminalSplits) {
                mergeDelta.emplace_back(split / 2, 1.0);
            }

            computeMergeDeltas(
                    mergeFactors
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
        inline std::vector<WeightVector> estimateNontFreqLA(const LatentAnnotation &latentAnnotation) {
            struct NontFreq {
                std::shared_ptr<StorageManager> storageManager;
                std::vector<WeightVector> nonterminalFrequencies;

                NontFreq(
                        std::shared_ptr<StorageManager> storageManager
                        , std::vector<WeightVector> &&nonterminalFrequencies
                ) : storageManager(storageManager), nonterminalFrequencies(nonterminalFrequencies) {};

                NontFreq(const NontFreq &other) : storageManager(other.storageManager) {
                    for (const WeightVector &vector : other.nonterminalFrequencies) {
                        nonterminalFrequencies.push_back(storageManager->create_weight_vector<WeightVector>(vector.size()));
                        nonterminalFrequencies.back() = vector;
                    }
                }

                NontFreq &operator+=(const NontFreq &other) {
                    std::transform(
                            other.nonterminalFrequencies.cbegin()
                            , other.nonterminalFrequencies.cend()
                            , nonterminalFrequencies.begin()
                            , nonterminalFrequencies.begin()
                            , [](const WeightVector &x, const WeightVector &y) { return x + y; }
                    );
                    return *this;
                }
            };

            NontFreq nonterminalFrequencies(storageManager, initialize_nonterminal_frequencies(latentAnnotation));

            // computing in(A_x) * out(A_x) for every A ∈ N and x ∈ X_A
            #ifdef _OPENMP
            omp_set_num_threads(threads);
            #endif

            #pragma omp declare reduction(+ : NontFreq : omp_out += omp_in) initializer (omp_priv = omp_orig)
            #pragma omp parallel for schedule(dynamic, 10) reduction(+:nonterminalFrequencies)
            for (TraceIterator traceIterator = traceManager->cbegin();
                 traceIterator < traceManager->cend(); ++traceIterator) {

                if (tracesInsideWeights[traceIterator - traceManager->cbegin()].size() !=
                    traceIterator->get_hypergraph()->size() or
                        tracesOutsideWeights[traceIterator - traceManager->cbegin()].size() !=
                        traceIterator->get_hypergraph()->size()) {

                    tracesInsideWeights[traceIterator - traceManager->cbegin()].clear();
                    tracesOutsideWeights[traceIterator - traceManager->cbegin()].clear();
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
                        *latentAnnotation.ruleWeights
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
                        auto &target = nonterminalFrequencies.nonterminalFrequencies[node->get_label_id()];
                        target += fraction;
                    }
                }
            }

            return nonterminalFrequencies.nonterminalFrequencies;
        }

        inline std::vector<WeightVector> initialize_nonterminal_frequencies(const LatentAnnotation &latentAnnotation) {
            std::vector<WeightVector> nonterminalFrequencies;
            for (size_t nont = 0; nont < latentAnnotation.nonterminalSplits.size(); ++nont) {
                WeightVector mw
                        = storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[nont]);
                mw.setZero();
                nonterminalFrequencies.push_back(mw);
            }
            return nonterminalFrequencies;
        }

        inline std::vector<std::vector<double>> computeMergeFactors(const std::vector<WeightVector> &mergeWeights) {
            std::cerr << "Computing merge factors." << std::endl;
            std::vector<std::vector<double>> p;
            for (auto las_weights : mergeWeights) {
                p.emplace_back(std::vector<double>(las_weights.dimension(0)));
                const size_t half_splits{las_weights.dimension(0) / 2};
                for (unsigned i = 0; i < half_splits; ++i) {
                    double combined_weight = las_weights(i) + las_weights(i + half_splits);
                    if ((not std::isnan(combined_weight)) and combined_weight > 0) {
                        p.back()[i]               = las_weights(i) / combined_weight;
                        p.back()[i + half_splits] = las_weights(i + half_splits) / combined_weight;
                    } else {
                        p.back()[i] = 0.5;
                        p.back()[i + half_splits] = 0.5;
                    }
                }
            }
            return p;
        }

        /**
         * Compute merge-Δ for each split. This is an approximation of likelihood after merge
         * divided by likelihood before merge.
         * Splits with high merge-Δ should be merged, splits with low merge-Δ should be kept.
         */
        inline void computeMergeDeltas(
                const std::vector<std::vector<double>> &p
                , const std::vector<size_t> &nontDimensions
                , std::vector<std::vector<double>> &mergeDelta
        ) const {

            // prefix and postfix sums are used for efficient computation of
            // s(i) = sum_{j ∈ {0, …, i-1, i+1, …, n-1}} a_j
            // for each i ∈ {0, …, n-1}
            std::vector<double> prefixSums;
            std::vector<double> postfixSums;

            for (TraceIterator trace_id = traceManager->cbegin(); trace_id < traceManager->cend(); ++trace_id) {
                const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &insideWeights = tracesInsideWeights[
                        trace_id - traceManager->cbegin()];
                const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &outsideWeights = tracesOutsideWeights[
                        trace_id - traceManager->cbegin()];

                for (const Element<Node<Nonterminal>> &node : *(trace_id->get_hypergraph())) {
                    const size_t nont = node->get_label_id();
                    const size_t nontDim = nontDimensions[nont];
                    const size_t halfDim = nontDim / 2;

                    const auto &insideWeight = insideWeights.at(node);
                    const auto &outsideWeight = outsideWeights.at(node);

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
    };

    /**
     * Merge all splits, where merge-Δ is above given threshold.
     */
    template<typename Nonterminal, typename TraceID>
    class ThresholdMergePreparator : public DefaultMergePreparator<Nonterminal, TraceID> {
        const double merge_threshold;

    public:
        ThresholdMergePreparator(
                TraceManagerPtr <Nonterminal, TraceID> traceManager
                , std::shared_ptr<StorageManager> storageManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , double merge_threshold
                , unsigned threads = 1
                , bool debug = false
        )
                : DefaultMergePreparator<Nonterminal, TraceID>(
                traceManager
                , storageManager
                , grammarInfo
                , threads
                , debug
        ),
                  merge_threshold(merge_threshold) {}

    protected:
        double computeMergeThreshold(const std::vector<std::vector<double>> &merge_delta) {
            std::cerr << "Selecting merges ";
            std::cerr << "above threshold " << merge_threshold;
            std::cerr << std::endl;
            return merge_threshold;
        }
    };

    /**
     * Merges the first mergePercent % of splits ordered by merge-Δ in descending order.
     */
    template<typename Nonterminal, typename TraceID>
    class PercentMergePreparator : public DefaultMergePreparator<Nonterminal, TraceID> {
        const double mergePercent;

    public:
        PercentMergePreparator(
                TraceManagerPtr <Nonterminal, TraceID> traceManager
                , std::shared_ptr<StorageManager> storageManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , double mergePercent
                , unsigned threads = 1
                , bool debug = false
        ) : DefaultMergePreparator<Nonterminal, TraceID>(traceManager, storageManager, grammarInfo, threads, debug),
            mergePercent(mergePercent) {}

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

            std::cerr << "ordered merge weights: ";
            for (auto weight : orderedMergeWeights)
                std::cerr << weight << " ";
            std::cerr << std::endl;

            // todo: option to skip over merge_weights >= 1

            size_t index = (size_t) (mergePercent / 100.0 * orderedMergeWeights.size());
            if (index > orderedMergeWeights.size())
                index = orderedMergeWeights.size() - 1;

            std::cerr << "index for ordered merges " << index << " / " << orderedMergeWeights.size() << std::endl;

            return orderedMergeWeights[index];
        }
    };
}

#endif //STERMPARSER_MERGEPREPARATOR_H
