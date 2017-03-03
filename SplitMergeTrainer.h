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
    template<typename Nonterminal>
    class Splitter {
        const double randPercent;
        std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;

        double rand_split() {
            return fRand((100 - randPercent) / 100.0, (100 + randPercent) / 100.0);
        }

    public:
        Splitter(double randPercent, std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo, std::shared_ptr<StorageManager> storageManager)
                : randPercent(randPercent), grammarInfo(grammarInfo), storageManager(storageManager) {}

        LatentAnnotation split(const LatentAnnotation & la) {
            std::vector<size_t> nonterminalSplits;
            // double nonterminal splits
            nonterminalSplits.reserve(la.nonterminalSplits.size());
            std::transform(
                    la.nonterminalSplits.cbegin()
                    , la.nonterminalSplits.cend()
                    , std::back_inserter(nonterminalSplits)
                    , [](auto x) { return x * 2; }
            );

            std::cerr << "la root weights: " << std::endl<< la.rootWeights << std::endl;
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
                Eigen::Tensor<double, 1> normalization_divisor(nonterminalSplits[nont]);
                normalization_divisor.setZero();
                for (size_t ruleId : group) {
                    compute_normalization_divisor(normalization_divisor, ruleWeights[ruleId]);

                }
                for (size_t ruleId : group) {
                    normalize(ruleWeights[ruleId], ruleWeights[ruleId], normalization_divisor);
                }
            }

            return LatentAnnotation(nonterminalSplits, rootWeights, ruleWeights);
        };

    private:
        RuleTensor<double> create_split_tensor(const RuleTensor<double> wrapped_tensor) {
            switch (wrapped_tensor.which() + 1) {
                case 1:
                    return create_split_tensor_ranked<1>(wrapped_tensor);
                case 2:
                    return create_split_tensor_ranked<2>(wrapped_tensor);
                case 3:
                    return create_split_tensor_ranked<3>(wrapped_tensor);
                case 4:
                    return create_split_tensor_ranked<4>(wrapped_tensor);
                default:
                    abort();
            }
        }

        template<long rule_rank>
        RuleTensor<double> create_split_tensor_ranked(const RuleTensor<double> wrapped_tensor) {
            const auto &raw_tensor = boost::get<RuleTensorRaw <double, rule_rank>>(wrapped_tensor);
            Eigen::array<Eigen::DenseIndex, rule_rank> split_dimenions = raw_tensor.dimensions();
            Eigen::array<Eigen::DenseIndex, rule_rank> broadcast;
            std::fill(broadcast.begin(), broadcast.end(), 2);
            std::for_each(split_dimenions.begin(), split_dimenions.end(), [](auto & dim) { dim = 2 * dim; });
            size_t memory = std::accumulate(
                            split_dimenions.cbegin()
                            , split_dimenions.cend()
                            , (size_t) 1
                            , std::multiplies<size_t>());
            auto split_tensor = storageManager
                    ->create_uninitialized_tensor_ranked_typed<RuleTensorRaw<double, rule_rank>>(memory, split_dimenions);
            split_tensor = raw_tensor.broadcast(broadcast);
            split_tensor = split_tensor.unaryExpr([this](double x) { return x * rand_split(); });
            return split_tensor;
        }

    };

    template<typename Nonterminal, typename TraceID>
    class MergePreparator;

    template<typename Nonterminal>
    class Merger;
    /*{
    public:
        Merger(const GrammarInfo2<Nonterminal> &, StorageManager &);
        LatentAnnotation merge(const LatentAnnotation, const MergeInfo);
    };*/

    template<typename Nonterminal, typename TraceID>
    class SplitMergeTrainer {
        std::shared_ptr<EMTrainerLA> emTrainer;
        std::shared_ptr<Splitter<Nonterminal>> splitter;
        std::shared_ptr<MergePreparator<Nonterminal, TraceID>> mergePreparator;
        std::shared_ptr<Merger<Nonterminal>> merger;

    public:
        SplitMergeTrainer(
                std::shared_ptr<EMTrainerLA >emTrainer
                , std::shared_ptr<Splitter<Nonterminal>> splitter
                , std::shared_ptr<MergePreparator <Nonterminal, TraceID>> mergePreparator
                , std::shared_ptr<Merger<Nonterminal>> merger
        ) :
                emTrainer(emTrainer), splitter(splitter), mergePreparator(mergePreparator), merger(merger) {}

        LatentAnnotation split_merge_cycle(LatentAnnotation la) {
            auto laSplit = splitter->split(la);
            emTrainer->train(laSplit);
            auto mergeInfo = mergePreparator->mergePrepare(laSplit);

            std::cerr << mergeInfo;
            std::cerr << "rules weights before merge" << std::endl;
            {
                size_t rule_id {0};
                for (auto ruleTensor : laSplit.ruleWeights) {
                    std::cerr << "rule " << rule_id << std::endl << ruleTensor << std::endl;
                    ++ rule_id;
                }
            }

            auto laMerged = merger->merge(laSplit, mergeInfo);

            {
                size_t rule_id{0};
                for (const RuleTensor<double> ruleTensor : laMerged.ruleWeights) {
                    std::cerr << "rule " << rule_id << std::endl << ruleTensor << std::endl;
                    ++ rule_id;
                }
            }
            emTrainer->train(laMerged);
            return laMerged;
        }

    };

    template<typename Nonterminal, typename TraceID>
    class MergePreparator {
        const TraceManagerPtr <Nonterminal, EdgeLabelT> traceManager;
        std::shared_ptr<StorageManager> storageManager;
        using TraceIterator = ConstManagerIterator<Trace<Nonterminal, TraceID>>;

        const bool debug;
        std::vector<MAPTYPE<Element<Node < Nonterminal>>, WeightVector>> traces_inside_weights;
        std::vector<MAPTYPE<Element<Node < Nonterminal>>, WeightVector>> traces_outside_weights;

    public:
        MergePreparator(TraceManagerPtr<Nonterminal, EdgeLabelT> traceManager, std::shared_ptr<StorageManager> storageManager, bool debug = false)
                : traceManager(traceManager), storageManager(storageManager), debug(debug) {}

        MergeInfo mergePrepare(const LatentAnnotation latentAnnotation) {


            std::vector<WeightVector> nonterminalFrequencies;
            for (size_t nont = 0; nont < latentAnnotation.nonterminalSplits.size(); ++nont) {
                WeightVector mw = storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[nont]);
                mw.setZero();
                nonterminalFrequencies.push_back(mw);
            }

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
            clean_up();
            for (WeightVector & weightVector : nonterminalFrequencies) {
                storageManager->free_weight_vector(weightVector);
            }
            nonterminalFrequencies.clear();

            return build_merge_info(std::move(mergeFactors), merge_threshold, mergeDelta, latentAnnotation.nonterminalSplits);
        }
    private:

        void clean_up(){
            for (auto traceIterator = traceManager->cbegin(); traceIterator != traceManager->cend(); ++ traceIterator) {
                if (traceIterator - traceManager->cbegin() < traces_inside_weights.size()) {
                    for (const auto &node : *(traceIterator->get_hypergraph())) {
                        storageManager->free_weight_vector(
                                traces_inside_weights[traceIterator - traceManager->cbegin()].at(node));
                        storageManager->free_weight_vector(
                                traces_outside_weights[traceIterator - traceManager->cbegin()].at(node));
                    }
                }
            }
            traces_inside_weights.clear();
            traces_outside_weights.clear();
        }

        inline void estimateNontFreqLA(
                const TraceIterator start
                , const TraceIterator stop
                , std::vector<WeightVector> &nonterminalFrequencies
                , const LatentAnnotation & latentAnnotation
        ) {
            // computing in(A_x) * out(A_x) for every A ∈ N and x ∈ X_A
            for (TraceIterator traceIterator = start; traceIterator < stop; ++traceIterator) {

                // todo: this could be problematic in a parallel setting
                if (traces_inside_weights.size() <= traceIterator - traceManager->cbegin()) {
                    traces_inside_weights.resize(1 + (traceIterator - traceManager->cbegin()));
                }
                if (traces_outside_weights.size() <= traceIterator - traceManager->cbegin()) {
                    traces_outside_weights.resize(1 + (traceIterator - traceManager->cbegin()));
                }
                if (traces_inside_weights[traceIterator - traceManager->cbegin()].size() != traceIterator->get_hypergraph()->size()) {
                    for (const auto &node : *(traceIterator->get_hypergraph())) {
                        traces_inside_weights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                        traces_outside_weights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                    }
                    // todo: free this eventually
                }

                traceIterator->io_weights_la(
                        latentAnnotation.ruleWeights
                        , latentAnnotation.rootWeights
                        , traces_inside_weights[traceIterator - traceManager->cbegin()]
                        , traces_outside_weights[traceIterator - traceManager->cbegin()]
                );

                const auto &insideWeights = traces_inside_weights[traceIterator - traceManager->cbegin()];
                const auto &outsideWeights = traces_outside_weights[traceIterator - traceManager->cbegin()];

                for (const Element<Node<Nonterminal>> &node : *(traceIterator->get_hypergraph())) {

                    const auto &inside_weight = insideWeights.at(node);
                    const auto &outside_weight = outsideWeights.at(node);

                    const auto vals = inside_weight * outside_weight;
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

        inline std::vector<std::vector<double>> computeMergeFactors(const std::vector<WeightVector> &merge_weights) {
            std::cerr << "Computing merge factors." << std::endl;
            std::vector<std::vector<double>> p;
            for (auto las_weights : merge_weights) {
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
                , const std::vector<size_t> &nont_dimensions
                , std::vector<std::vector<double>> &merge_delta
        ) const {
            std::vector<double> prefixes;
            std::vector<double> postfixes;
            for (TraceIterator trace_id = start; trace_id < stop; ++trace_id) {
                const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &inside_weights = traces_inside_weights[
                        trace_id - traceManager->cbegin()];
                const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &outside_weights = traces_outside_weights[
                        trace_id - traceManager->cbegin()];

                for (const Element<Node<Nonterminal>> &node : *(trace_id->get_hypergraph())) {

                    const size_t nont_dim = nont_dimensions[node->get_label_id()];
                    const size_t half_dim = nont_dim / 2;
                    prefixes.resize(half_dim, 0);
                    postfixes.resize(half_dim, 0);
                    double denominator = 0;
                    {
                        const size_t dim = half_dim - 1;
                        const double in1 = inside_weights.at(node).data()[dim];
                        const double in2 = inside_weights.at(node).data()[dim + half_dim];
                        const double out1 = outside_weights.at(node).data()[dim];
                        const double out2 = outside_weights.at(node).data()[dim + half_dim];
                        denominator += in1 * out1 + in2 * out2;
                    }
                    for (size_t dim = 0; dim < half_dim - 1; ++dim) {
                        const double in1 = inside_weights.at(node).data()[dim];
                        const double in2 = inside_weights.at(node).data()[dim + half_dim];
                        const double out1 = outside_weights.at(node).data()[dim];
                        const double out2 = outside_weights.at(node).data()[dim + half_dim];
                        prefixes[dim + 1] = prefixes[dim] + in1 * out1 + in2 * out2;
                        denominator += in1 * out1 + in2 * out2;
                    }

                    for (size_t dim_ = half_dim - 1; dim_ > 0; --dim_) {
                        const double in1 = inside_weights.at(node).data()[dim_];
                        const double in2 = inside_weights.at(node).data()[dim_ + half_dim];
                        const double out1 = outside_weights.at(node).data()[dim_];
                        const double out2 = outside_weights.at(node).data()[dim_ + half_dim];
                        postfixes[dim_ - 1] = postfixes[dim_] + in1 * out1 + in2 * out2;
                    }

                    // inside weight of some nodes can be zero in certain LA-dimensions
                    // since LA-rule weights may converge to zero
                    // we ignore those dimensions in Δ computation
                    if (denominator == 0)
                        continue;

                    for (unsigned dim = 0; dim < half_dim; ++dim) {
                        const double in1 = inside_weights.at(node).data()[dim];
                        const double in2 = inside_weights.at(node).data()[dim + half_dim];
                        const double out1 = outside_weights.at(node).data()[dim];
                        const double out2 = outside_weights.at(node).data()[dim + half_dim];
                        const size_t nont = node->get_label_id();
                        const double p1 = p[nont][dim];
                        const double p2 = p[nont][dim + half_dim];

                        const double out_merged = out1 + out2;
                        const double in_merged = (p1 * in1) + (p2 * in2);

                        const double Q = (prefixes[dim] + postfixes[dim] + in_merged * out_merged) / denominator;

                        if (std::isnan(Q)) {
                            std::cerr << "bad fraction " << Q << " where" << std::endl;
                            std::cerr << "prefix  " << prefixes[dim] << std::endl;
                            std::cerr << "postfix " << postfixes[dim] << std::endl;
                            std::cerr << "merged  " << in_merged * out_merged << std::endl;
                            std::cerr << "denom   " << denominator << std::endl;

                            assert(!std::isnan(Q));
                        }

                        double &delta = merge_delta[nont][dim];
                        delta *= Q;
                    }

                    prefixes.clear();
                    postfixes.clear();
                }
            }
        }

        virtual double computeMergeThreshold(const std::vector<std::vector<double>> & merge_delta) = 0;

        // evaluate Δ and build MergeInfo accordingly
        MergeInfo build_merge_info(
                const std::vector<std::vector<double>> && merge_factors
                , const double merge_threshold
                , const std::vector<std::vector<double>> & merge_delta
                , const std::vector<size_t> & nontSplits
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
                        || (traceManager->cbegin()->get_goal()->get_label_id()) ==
                           nont) { // todo: this info should be in GrammarInfo
                        mergeSelection.back().emplace_back();
                        mergeSelection.back().back().push_back(split);
                        mergeSelection.back().back().push_back(split +  halfSplits);
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
        ThresholdMergePreparator(TraceManagerPtr<Nonterminal, TraceID> traceManager, std::shared_ptr<StorageManager> storageManager, double merge_threshold, bool debug = false)
                : MergePreparator<Nonterminal, TraceID>(traceManager, storageManager, debug) , merge_threshold(merge_threshold) {}

    protected:
        double computeMergeThreshold(const std::vector<std::vector<double>> & merge_delta) {
            std::cerr << "Selecting merges ";
            std::cerr << "above threshold " << merge_threshold;
            std::cerr << std::endl;
            return merge_threshold;
        }
    };

    template<typename Nonterminal, typename TraceID>
    class PercentMergePreparator : public MergePreparator<Nonterminal, TraceID> {
        const double merge_percent;

    public:
        PercentMergePreparator(TraceManagerPtr<Nonterminal, TraceID> traceManager
                               , std::shared_ptr<StorageManager> storageManager
                               , double merge_percent
                               , bool debug = false
        ) : MergePreparator<Nonterminal, TraceID>(traceManager, storageManager, debug), merge_percent(merge_percent) {}

    protected:
        double computeMergeThreshold(const std::vector<std::vector<double>> & merge_delta){
            std::cerr << "Selecting merges ";
            std::cerr << "best " << merge_percent << " % ";
            std::cerr << std::endl;

            std::vector<double> ordered_merge_weights;

            // order merges according to likelihood_loss
            for (const auto &delta : merge_delta) {
                ordered_merge_weights.insert(
                        std::end(ordered_merge_weights)
                        , std::begin(delta)
                        , std::end(delta));
            }

            std::sort(std::begin(ordered_merge_weights), std::end(ordered_merge_weights), std::greater<double>());

            // todo: option to skip over merge_weights >= 1

            size_t index = (size_t) (merge_percent / 100.0 * ordered_merge_weights.size());
            if (index > ordered_merge_weights.size())
                index = ordered_merge_weights.size() - 1;

            std::cerr << "index for ordered merges " << index << " / " << ordered_merge_weights.size() << std::endl;

            return ordered_merge_weights[index];
        }
    };

    template<typename Nonterminal>
    class Merger {
        std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;

    public:
        Merger(std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo, std::shared_ptr<StorageManager> storageManager)
                : grammarInfo(grammarInfo), storageManager(storageManager) {}

        LatentAnnotation merge(const LatentAnnotation & la, const MergeInfo & mergeInfo) {
            // root weights
            Eigen::Tensor<double, 1> rootWeights (mergeInfo.nontSplitsAfterMerge[grammarInfo->start]);
            for (Eigen::DenseIndex idx = 0; idx < rootWeights.dimension(0); ++idx) {
                rootWeights(idx) = 0;
                for (size_t idx_origin : mergeInfo.mergeSources[grammarInfo->start][idx])
                    rootWeights(idx) += la.rootWeights(idx_origin);
            }
            std::cerr << "root weights " << rootWeights << std::endl;

            // rule weights
            std::vector<RuleTensor<double>> ruleWeights;
            for (size_t rule_id = 0; rule_id < grammarInfo->rule_to_nonterminals.size(); ++rule_id) {
                RuleTensor<double> merged_tensor = storageManager->create_uninitialized_tensor(
                        rule_id
                        , *grammarInfo
                        , mergeInfo.nontSplitsAfterMerge
                );
                merge_tensor(merged_tensor, la.ruleWeights[rule_id], rule_id, mergeInfo);
                std::cerr << rule_id << " " << merged_tensor << std::endl;
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
            auto &mergedTensorRaw = boost::get<RuleTensorRaw <double, rank>>(mergedTensor);
            const auto &sourceTensorRaw = boost::get<RuleTensorRaw <double, rank>>(sourceTensor);

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

}


#endif //STERMPARSER_SPLITMERGETRAINER_H
