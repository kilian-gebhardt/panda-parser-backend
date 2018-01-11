//
// Created by kilian on 10/03/17.
//

#ifndef STERMPARSER_MERGEPREPARATOR_H
#define STERMPARSER_MERGEPREPARATOR_H

#include <memory>
#include "GrammarInfo.h"
#include "LatentAnnotation.h"
#include "TrainingCommon.h"
#include <numeric>
#include <omp.h>

namespace Trainer {
    typedef std::function<double(const::std::vector<double>&)> ThresholdFunction;

    class MergePreparator {
    protected:

        std::shared_ptr<const GrammarInfo2> grammarInfo;
        const bool debug;

        /**
         * Builds MergeInfo according to merge-Δs and threshold.
         * If (merge-Δ > 1 or the split of a start symbol is concerned)
         * then the split is always merged.
         */
        MergeInfo build_merge_info(
                const std::vector<std::vector<double>> &&merge_factors
                , const double merge_threshold
                , const std::vector<std::vector<double>> &&merge_delta
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
                    // merge if Δ >= merge_thershold * 0.999, i.e. log(Δ) >= log(θ) + log(0.999)  (logarithmic)
                    if (delta[split] >= merge_threshold + std::log(0.999)
                        // always merge if Δ >= 1
                        // i.e. log(Δ) >= 0 + log(0.999)
                        || delta[split] >= std::log(0.999)
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

        virtual void setMergeThresholdFunction(ThresholdFunction /*thresholdFunction*/) {};
    };

    /**
     * Merges none of the splits, except for start symbol whose splits are always merged.
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
                mergeDelta.emplace_back(splits / 2, std::log(0.4));
            }

            double merge_threshold = std::log(0.5);

            return build_merge_info(
                      std::move(mergeFactors)
                    , merge_threshold
                    , std::move(mergeDelta)
                    , latentAnnotation.nonterminalSplits
            );
        }
    };

    template<typename Nonterminal, typename TraceID>
    class DefaultMergePreparator : public MergePreparator {
    protected:
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

        virtual MergeInfo merge_prepare(const LatentAnnotation &latentAnnotation) {

            // setup temporary data structures
            if (tracesInsideWeights.size() < traceManager->size())
                tracesInsideWeights.resize(traceManager->size());
            if (tracesOutsideWeights.size() < traceManager->size())
                tracesOutsideWeights.resize(traceManager->size());

            std::vector<WeightVector> nonterminalFrequencies{estimateNontFreqLA(latentAnnotation)};
            std::vector<std::vector<double>> mergeFactors{computeMergeFactors(nonterminalFrequencies)};

            std::vector<std::vector<double>> mergeDelta;
            for (auto split : latentAnnotation.nonterminalSplits) {
                mergeDelta.emplace_back(split / 2, std::log(1.0));
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
                    , std::move(mergeDelta)
                    , latentAnnotation.nonterminalSplits
            );
        }

    protected:
        /**
         * What this function computes corresponds to the mergeWeights of the Berkeley parser.
         * @param latentAnnotation
         * @return
         */
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
                          latentAnnotation
                        , tracesInsideWeights[traceIterator - traceManager->cbegin()]
                        , tracesOutsideWeights[traceIterator - traceManager->cbegin()]
                        , true
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
                        target += fraction * traceIterator->get_frequency();
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

        /**
         * @param nontFreqLA (== mergeWeight in Berkeley parser)
         * @return the p from the Berkeley parser
         */
        inline std::vector<std::vector<double>> computeMergeFactors(const std::vector<WeightVector> &nontFreqLA) {
            std::cerr << "Computing merge factors." << std::endl;
            std::vector<std::vector<double>> p;
            for (auto las_weights : nontFreqLA) {
                p.emplace_back(std::vector<double>(las_weights.dimension(0)));
                const long int half_splits{las_weights.dimension(0) / 2};
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
                        delta += std::log(Q);
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

            std::vector<double> orderedMergeDeltas;

            // order merges according to likelihood_loss
            for (const auto &delta : mergeDelta) {
                orderedMergeDeltas.insert(
                        std::end(orderedMergeDeltas)
                        , std::begin(delta)
                        , std::end(delta));
            }

            std::sort(std::begin(orderedMergeDeltas), std::end(orderedMergeDeltas), std::greater<double>());

            std::cerr << "ordered merge Δs: ";
            for (auto weight : orderedMergeDeltas)
                std::cerr << weight << " ";
            std::cerr << std::endl;

            // todo: option to skip over merge_weights >= 1

            size_t index = (size_t) (mergePercent / 100.0 * orderedMergeDeltas.size());
            if (index > orderedMergeDeltas.size())
                index = orderedMergeDeltas.size() - 1;

            std::cerr << "index for ordered merges " << index << " / " << orderedMergeDeltas.size() << std::endl;

            return orderedMergeDeltas[index];
        }
    };

    /**
     * Merges nonterminals according to the principle stated in www.aclweb.org/anthology/E14-1015
     *
     * Merge-Δs are computed for each pair {i,j} of latent annotations of some nonterminal.
     * Then a fully connected, undirected graph G with latent annotations as nodes is constructed.
     * Each edge {i,j} is weighted by w=Δ({i,j}) and edges with w <= threshold are removed.
     * The (strongly) connected components of G are the new latent annotations.
     * Merge weights are chosen proportional to the expected frequency of the annotations.
     *
     * @tparam Nonterminal
     * @tparam TraceID
     */
    template <typename Nonterminal, typename TraceID>
    class SCCMerger : public DefaultMergePreparator<Nonterminal, TraceID> {
        std::vector<size_t> relevantNonterminals;
        ThresholdFunction thresholdFunction;

    public:
        SCCMerger(
                TraceManagerPtr <Nonterminal, TraceID> traceManager
                , std::shared_ptr<StorageManager> storageManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::vector<size_t> relevantNonterminals
                , ThresholdFunction thresholdFunction
                , unsigned threads = 1
                , bool debug = false
        )
                : DefaultMergePreparator<Nonterminal, TraceID> (
                traceManager
                , storageManager
                , grammarInfo
                , threads
                , debug
        ), relevantNonterminals(relevantNonterminals), thresholdFunction(thresholdFunction) {};

        MergeInfo merge_prepare(const LatentAnnotation &latentAnnotation) {
            // setup temporary data structures
            if (this->tracesInsideWeights.size() < this->traceManager->size())
                this->tracesInsideWeights.resize(this->traceManager->size());
            if (this->tracesOutsideWeights.size() < this->traceManager->size())
                this->tracesOutsideWeights.resize(this->traceManager->size());

            std::vector<WeightVector> nonterminalFrequencies{this->estimateNontFreqLA(latentAnnotation)};

            // computing Δ per nont and pair of LAs j and i (where j > i)
            std::vector<std::vector<std::vector<double>>> merge_delta;
            computePairwiseMergeDeltas(nonterminalFrequencies, latentAnnotation.nonterminalSplits, merge_delta);
            auto stats = mergeWeightStatistics(merge_delta);
            const double merge_threshold = thresholdFunction(stats);
            std::cerr << "SCC merging with threshold: " << merge_threshold << std::endl;

            // ingredients for the MergeInfo
            std::vector<std::vector<std::vector<size_t>>> mergeSources;
            std::vector<size_t> nontSplitsAfterMerge;
            std::vector<std::vector<double>> mergeFactors;

            for (size_t nont = 0; nont < latentAnnotation.nonterminalSplits.size(); ++nont) {
                // check if nont ∈ relevantNonterminals
                bool relevant = false;
                for (size_t nont2 : relevantNonterminals) {
                    if (nont2 == nont) relevant = true;
                    if (nont2 >= nont) break;
                }
                if (relevant) {
                    // lazily build graph by pairwise connecting all LAs of nont (implicit)
                    // we only add an edge to the representation, if it is not removed in the next step
                    // the graph is represented by two maps encoding maximal SCCs,
                    // satisfying
                    // 1. j ∈ edges[i] if i < j and (i,j) are connected in graph
                    // 2. inSCC[i] = i or i ∈ edges[inSCC[i]]
                    MAPTYPE<size_t, std::vector<size_t >> edges;
                    MAPTYPE<size_t, size_t> inSCC;

                    // determine weight Δ for each edge (i,j) in graph and remove edge if Δ <= threshold
                    // i.e., we add i and j to the same SCC if Δ > threshold
                    for (size_t i = 0; i < latentAnnotation.nonterminalSplits[nont]; ++i) {
                        for (size_t j = i + 1; j < latentAnnotation.nonterminalSplits[nont]; ++j) {
                            if (merge_delta[nont][j][i] > merge_threshold) {
                                if (not(inSCC.count(i) or inSCC.count(j))) {
                                    edges[i].push_back(j);
                                    inSCC[i] = i;
                                    inSCC[j] = i;
                                } else if (not inSCC.count(j)) {
                                    inSCC[j] = inSCC[i];
                                    edges[inSCC[i]].push_back(j);
                                } else if (not inSCC.count(i)) {
                                    inSCC[i] = inSCC[j];
                                    edges[inSCC[j]].push_back(i);
                                } else {
                                    if (inSCC[i] == inSCC[j]) {
                                        // nothing needs to be done!
                                    } else if (inSCC[i] < inSCC[j]) {
                                        const size_t old_scc_j = inSCC[j];

                                        for (size_t k : edges[old_scc_j]) {
                                            edges[inSCC[i]].push_back(k);
                                            inSCC[k] = inSCC[i];
                                        }
                                        edges[inSCC[i]].push_back(old_scc_j);
                                        inSCC[old_scc_j] = inSCC[i];
                                        edges.erase(old_scc_j);
                                    } else {
                                        const size_t old_scc_i = inSCC[i];

                                        for (size_t k : edges[old_scc_i]) {
                                            edges[inSCC[j]].push_back(k);
                                            inSCC[k] = inSCC[j];
                                        }
                                        edges[inSCC[j]].push_back(old_scc_i);
                                        inSCC[old_scc_i] = inSCC[j];
                                        edges.erase(old_scc_i);
                                    }
                                }
                            }
                        }
                    }

                    // new LAs = maximal SCCs and
                    // set mergeFactor proportional to nontFreq
                    std::vector<std::vector<size_t>> mergeLists;
                    std::vector<double> laMergeFactors(latentAnnotation.nonterminalSplits[nont]);
                    size_t merged_splits = 0;
                    for (auto key_value_pair : edges) {
                        if (inSCC[key_value_pair.first] != key_value_pair.first)
                            continue;
                        mergeLists.push_back(key_value_pair.second);
                        mergeLists.back().push_back(key_value_pair.first);
                        std::sort(mergeLists.back().begin(), mergeLists.back().end());
                        merged_splits += mergeLists.back().size();
                        double normalizer = 0.0;
                        for (auto la : mergeLists.back())
                            normalizer += nonterminalFrequencies[nont](la);

                        if (normalizer > 0 and not std::isnan(normalizer) and not std::isnan(normalizer))
                            for (auto la : mergeLists.back()) {
                                /* for debugging
                                if (nont == 179)
                                    std::cerr << nont << " la: " << la << " freq: "
                                              << nonterminalFrequencies[nont](la) << " n: " << normalizer << std::endl;
                                */
                                laMergeFactors[la] = nonterminalFrequencies[nont](la) / normalizer;
                            }
                        else
                            for (auto la : mergeLists.back()) {
                                laMergeFactors[la] = 1 / mergeLists.back().size();
                            }
                    }
                    // add all singletons
                    for (size_t la = 0; la < latentAnnotation.nonterminalSplits[nont]; ++la) {
                        if (not inSCC.count(la)) {
                            mergeLists.emplace_back(1, la);
                            laMergeFactors[la] = 1.0;
                            ++merged_splits;
                        }

                    }

                    /*// for debugging
                    for (size_t i = 0; i < mergeLists.size(); ++i) {
                        std::cerr << nont << ": " << i << " [ ";
                        for (auto elem : mergeLists[i])
                            std::cerr << elem << ", ";
                        std::cerr << "]" << std::endl;
                    }
                     */

                    if (merged_splits != latentAnnotation.nonterminalSplits[nont]) {
                        for (size_t la = 0; la < latentAnnotation.nonterminalSplits[nont]; ++la) {
                            if (inSCC.count(la))
                                std::cerr << nont << "-" << la << " is in SCC " << inSCC[la] << std::endl;
                            else
                                std::cerr << nont << "-" << la << " is not in any SCC" << std::endl;
                            if (edges.count(la)) {
                                std::cerr << nont << "-" << la << " has edges to ";
                                for (auto e : edges[la])
                                    std::cerr << e << " ";
                                std::cerr << std::endl;
                            } else std::cerr << nont << "-" << la << " has no edges" << std::endl;
                        }
                        abort();
                    }

                    nontSplitsAfterMerge.push_back(mergeLists.size());
                    mergeSources.push_back(mergeLists);
                    mergeFactors.push_back(laMergeFactors);


                    // if nont not in relevant items
                } else {
                    size_t n = latentAnnotation.nonterminalSplits.at(nont);
                    nontSplitsAfterMerge.push_back(n);
                    mergeFactors.emplace_back(n, 1.0);
                    std::vector<std::vector<size_t>> mergeLists;
                    for (size_t la = 0; la < n; ++la) {
                        mergeLists.emplace_back(1, la);
                    }
                    mergeSources.push_back(mergeLists);

                    /*// for debugging
                    for (size_t i = 0; i < mergeLists.size(); ++i) {
                        std::cerr << nont << ": " << i << " [ ";
                        for (auto elem : mergeLists[i])
                            std::cerr << elem << ", ";
                        std::cerr << "]" << std::endl;
                    }
                    */
                }
            }

            // clean up
            this->storageManager->free_weight_maps(this->tracesInsideWeights);
            this->storageManager->free_weight_maps(this->tracesOutsideWeights);
            for (WeightVector &weightVector : nonterminalFrequencies) {
                this->storageManager->free_weight_vector(weightVector);
            }
            nonterminalFrequencies.clear();

            return MergeInfo(mergeSources, nontSplitsAfterMerge, mergeFactors);
        }

        void setMergeThresholdFunction(ThresholdFunction thresholdFunction) {
            this->thresholdFunction = thresholdFunction;
        }

    private:
        /**
             * Compute merge-Δ for each pair of latent annotation. This is an approximation of likelihood after merge
             * divided by likelihood before merge.
             * Splits with high merge-Δ should be merged, splits with low merge-Δ should be kept.
             */
        inline void computePairwiseMergeDeltas(
                const std::vector<WeightVector> & expectedFrequencies
                , const std::vector<size_t> &nontDimensions
                , std::vector<std::vector<std::vector<double>>> &mergeDelta
        ) const {
            mergeDelta.clear();
            for (size_t nont = 0; nont < nontDimensions.size(); ++nont){
                mergeDelta.emplace_back(0);
                for (size_t j = 0; j < nontDimensions[nont]; ++ j) {
                    mergeDelta.back().emplace_back(j, 0.0);
                }
            }

            for (typename DefaultMergePreparator<Nonterminal, TraceID>::TraceIterator trace_id = this->traceManager->cbegin()
                    ; trace_id < this->traceManager->cend()
                    ; ++trace_id) {
                const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &insideWeights
                        = this->tracesInsideWeights[trace_id - this->traceManager->cbegin()];
                const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &outsideWeights
                        = this->tracesOutsideWeights[trace_id - this->traceManager->cbegin()];

                for (const Element<Node<Nonterminal>> &node : *(trace_id->get_hypergraph())) {
                    const size_t nont = node->get_label_id();
                    const size_t nontDim = nontDimensions[nont];


                    const auto &insideWeight = insideWeights.at(node);
                    const auto &outsideWeight = outsideWeights.at(node);

                    double denominator = 0.0;
                    for (size_t i = 0; i < nontDim; ++i) {
                        const double in = insideWeight(i);
                        const double out = outsideWeight(i);
                        denominator += in * out;
                    }

                    if ( denominator <= 0 or std::isinf(denominator) or std::isnan(denominator))
                        continue;

                    double prefix_sum = 0;

                    for (size_t i = 0; i < nontDim; ++i) {
                        const double in1 = insideWeight(i);
                        const double out1 = outsideWeight(i);
                        double infix_sum = 0;

                        for (size_t j = i + 1; j < nontDim; ++j) {
                            const double in2 = insideWeight(j);
                            const double out2 = outsideWeight(j);
                            const double f_norm = expectedFrequencies[nont](i) + expectedFrequencies[nont](j);
                            const double p1 = expectedFrequencies[nont](i) / f_norm;
                            const double p2 = expectedFrequencies[nont](j) / f_norm;

                            const double inMerged = (p1 * in1) + (p2 * in2);
                            const double outMerged = out1 + out2;

                            double postfix_sum = 0;
                            for (size_t k = j + 1; k < nontDim; ++k) {
                                postfix_sum += insideWeight(k) * outsideWeight(k);
                            }
                            const double others = prefix_sum + infix_sum + postfix_sum;

                            const double Q = (others + inMerged * outMerged) / denominator;

                            if (std::isnan(Q)) {
                                std::cerr << "bad fraction " << Q << " where" << std::endl;
                                std::cerr << "merged  " << inMerged * outMerged << std::endl;
                                std::cerr << "denom   " << denominator << std::endl;

                                assert(!std::isnan(Q));
                            }

                            double &delta = mergeDelta[nont][j][i];
                            delta += std::log(Q);

                            infix_sum += in2 * out2;
                        }

                        prefix_sum += in1 * out1;
                    }
                }
            }


//            for (auto nont = 0; nont < nontDimensions.size(); ++nont) {
//                for (size_t j = 0; j < nontDimensions[nont]; ++j)
//                    for (size_t i = 0; i < j; ++i)
//                        std::cerr << "(" << nont << ": " << j << " vs. " << i << ": " << mergeDelta[nont][j][i] << ") ";
//            }
//            std::cerr << std::endl;
        }

        // not used in this class
        double computeMergeThreshold(const std::vector<std::vector<double>> &) { return 0.0; };

        // compute merge Δ statistics
        std::vector<double> mergeWeightStatistics(const std::vector<std::vector<std::vector<double>>>& mergeDeltas) {
            double min {std::numeric_limits<double>::max()};
            double max {std::numeric_limits<double>::min()};
            double sum {0.0};
            size_t count {0};
            for (auto nont_vec : mergeDeltas) {
                for (auto la_1 : nont_vec){
                    for (auto la_1_2_delta : la_1) {
                        if (la_1_2_delta > max) max = la_1_2_delta;
                        if (la_1_2_delta < min) min = la_1_2_delta;
                        sum += la_1_2_delta;
                        count++;
                    }
                }
            }
            const double mean {sum / count};
            double above_mean_sum {0.0};
            size_t above_mean_count {0};
            double below_mean_sum {0.0};
            size_t below_mean_count {0};
            for (auto nont_vec : mergeDeltas) {
                for (auto la_1 : nont_vec){
                    for (auto la_1_2_delta : la_1) {
                        if (la_1_2_delta > mean) {
                            above_mean_sum += la_1_2_delta;
                            above_mean_count++;
                        } else if (la_1_2_delta < mean) {
                            below_mean_sum += la_1_2_delta;
                            below_mean_count++;
                        }
                    }
                }
            }
            const double third_quartile {above_mean_count > 0 ? above_mean_sum / above_mean_count : mean};
            const double first_quartile {below_mean_count > 0 ? below_mean_sum / below_mean_count : mean};

            std::cerr << "SCC merge Δ statistics {";
            std::cerr << "min: " << min << " first quartile: " << first_quartile << " mean: " << mean
                                 << " third quartile: " << third_quartile << " max: " << max << " }" << std::endl;
            return {min, first_quartile, mean, third_quartile, max};
        }

    };


}

#endif //STERMPARSER_MERGEPREPARATOR_H
