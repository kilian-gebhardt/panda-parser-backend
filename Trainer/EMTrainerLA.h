//
// Created by kilian on 01/03/17.
//

#ifndef STERMPARSER_EMTRAINERLA_H
#define STERMPARSER_EMTRAINERLA_H

#include "../Names.h"
#include "TrainingCommon.h"
#include "StorageManager.h"
#include "TraceManager.h"
#include "../Legacy/Trace.h"
#include <boost/operators.hpp>

namespace Trainer {
    class Counts : boost::addable<Counts>{
    public:
        std::vector<RuleTensor<double>> ruleCounts;
        Eigen::Tensor<double, 1> rootCounts;
        double logLikelihood;

        Counts(LatentAnnotation latentAnnotation, const GrammarInfo2 &grammarInfo, StorageManager & storageManager)
                : rootCounts(latentAnnotation.rootWeights.dimension(0)), logLikelihood(0) {
            for (size_t ruleId = 0; ruleId < latentAnnotation.ruleWeights.size(); ++ruleId) {
                RuleTensor<double> count =
                        storageManager.create_uninitialized_tensor(
                                ruleId
                                , grammarInfo
                                , latentAnnotation.nonterminalSplits
                        );
                set_zero(count);
                ruleCounts.push_back(count);
            }
            rootCounts.setZero();
        }

        Counts &operator+=(const Counts &other) {
            logLikelihood += other.logLikelihood;

            for (unsigned rule = 0; rule < ruleCounts.size(); ++rule) {
                switch (ruleCounts[rule].which() + 1) {
                    case 1:
                        boost::get<RuleTensorRaw<double, 1>>(ruleCounts[rule])
                                += boost::get<RuleTensorRaw<double, 1>>(other.ruleCounts[rule]);
                        break;
                    case 2:
                        boost::get<RuleTensorRaw<double, 2>>(ruleCounts[rule])
                                += boost::get<RuleTensorRaw<double, 2>>(other.ruleCounts[rule]);
                        break;
                    case 3:
                        boost::get<RuleTensorRaw<double, 3>>(ruleCounts[rule])
                                += boost::get<RuleTensorRaw<double, 3>>(other.ruleCounts[rule]);
                        break;
                    case 4:
                        boost::get<RuleTensorRaw<double, 4>>(ruleCounts[rule])
                                += boost::get<RuleTensorRaw<double, 4>>(other.ruleCounts[rule]);
                        break;
                    default:
                        abort();
                }
            }

            rootCounts += other.rootCounts;
            return *this;
        }
    };

    class Expector {
    public:
        virtual Counts expect(const LatentAnnotation latentAnnotation) = 0;
        virtual void clean_up() = 0;
    };

    class Maximizer {
    public:
        virtual void maximize(LatentAnnotation &, const Counts&) = 0;
    };

    class EMTrainerLA {
        const unsigned epochs;
        std::shared_ptr<Expector> expector;
        std::shared_ptr<Maximizer> maximizer;
    public:

        EMTrainerLA(unsigned epochs, std::shared_ptr<Expector> expector, std::shared_ptr<Maximizer> maximizer)
                : epochs(epochs), expector(expector), maximizer(maximizer) {};

        void train(LatentAnnotation & latentAnnotation) {
            for (unsigned epoch = 0; epoch < epochs; ++epoch) {
                Counts counts = expector->expect(latentAnnotation);

                std::cerr <<"Epoch " << epoch << "/" << epochs << ": ";

                // output likelihood information based on old probability assignment
                Eigen::Tensor<double, 0>corpusProbSum = counts.rootCounts.sum();
                std::cerr << "corpus prob. sum " << corpusProbSum;
                std::cerr << " corpus likelihood " << counts.logLikelihood;
//                std::cerr << " root counts: " << counts.rootCounts << std::endl;

                maximizer->maximize(latentAnnotation, counts);

                std::cerr << " root weights: " << latentAnnotation.rootWeights << std::endl;
            }
            expector->clean_up();
        }
    };

    template<typename Nonterminal, typename TraceID>
    class SimpleExpector : public Expector {
        template<typename T1, typename T2>
        using MAPTYPE = typename std::unordered_map<T1, T2>;
        using TraceIterator = ConstManagerIterator<Trace<Nonterminal, TraceID>>;

        const TraceManagerPtr<Nonterminal, TraceID> traceManager;
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;
        const bool debug;

        std::vector<MAPTYPE<Element<Node < Nonterminal>>, WeightVector>> tracesInsideWeights;
        std::vector<MAPTYPE<Element<Node < Nonterminal>>, WeightVector>> tracesOutsideWeights;

    public:
        SimpleExpector(
                TraceManagerPtr<Nonterminal, TraceID> traceManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::shared_ptr<StorageManager> storageManager
                , bool debug = false
        )
                : traceManager(traceManager), grammarInfo(grammarInfo), storageManager(storageManager), debug(debug) {};

        Counts expect(const LatentAnnotation latentAnnotation) {
            if (tracesInsideWeights.size() < traceManager->size()) {
                tracesInsideWeights.resize(traceManager->size());
            }
            if (tracesOutsideWeights.size() < traceManager->size()) {
                tracesOutsideWeights.resize(traceManager->size());
            }
            return expectation_la(latentAnnotation, traceManager->cbegin(), traceManager->cend());
        }

        void clean_up(){
            storageManager->free_weight_maps(tracesInsideWeights);
            storageManager->free_weight_maps(tracesOutsideWeights);
        }

    private:
        inline Counts expectation_la(
                const LatentAnnotation latentAnnotation
                , const TraceIterator start
                , const TraceIterator end
        ) {
            Counts counts(latentAnnotation, *grammarInfo, *storageManager);
            TraceIterator traceIterator;
            
            for (traceIterator = start; traceIterator < end; ++traceIterator) {
                const auto &trace = *traceIterator;
                if (trace->get_hypergraph()->size() == 0)
                    continue;

                // create insert inside and outside weight for each node if necessary
                if (tracesInsideWeights[traceIterator - traceManager->cbegin()].size() != trace->get_hypergraph()->size()) {
                    tracesInsideWeights.clear();
                    tracesOutsideWeights.clear();
                    for (const auto &node : *(trace->get_hypergraph())) {
                        tracesInsideWeights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                        tracesOutsideWeights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                    }
                }

                trace->io_weights_la(
                        latentAnnotation.ruleWeights
                        , latentAnnotation.rootWeights
                        , tracesInsideWeights[traceIterator - traceManager->cbegin()]
                        , tracesOutsideWeights[traceIterator - traceManager->cbegin()]
                );

                const auto &insideWeights = tracesInsideWeights[traceIterator - traceManager->cbegin()];
                const auto &outsideWeights = tracesOutsideWeights[traceIterator - traceManager->cbegin()];

                const auto &rootInsideWeight = insideWeights.at(trace->get_goal());
                const auto &rootOutsideWeight = outsideWeights.at(trace->get_goal());

                Eigen::Tensor<double, 1> traceRootProbabilities = rootInsideWeight * rootOutsideWeight;
                Eigen::Tensor<double, 0> traceRootProbability = traceRootProbabilities.sum();

                if (not std::isnan(traceRootProbability(0))
                    and not std::isinf(traceRootProbability(0))
                    and traceRootProbability(0) > 0) {
                    counts.rootCounts += traceRootProbabilities;
                    counts.logLikelihood += log(traceRootProbability(0));
                } else {
                    counts.logLikelihood += minus_infinity;
                    continue;
                }

                if (debug)
                    std::cerr << "instance root probability: " << std::endl << traceRootProbabilities << std::endl;

                for (const Element<Node < Nonterminal>>
                    &node : *(trace->get_hypergraph())) {
                    const WeightVector &lhnOutsideWeight = outsideWeights.at(node);

                    if (debug) {
                        std::cerr << node << std::endl << "outside weight" << std::endl << lhnOutsideWeight
                                  << std::endl;
                        std::cerr << "inside weight" << std::endl;
                        const WeightVector &lhn_inside_weight = insideWeights.at(node);
                        std::cerr << lhn_inside_weight << std::endl;
                    }
                    for (const auto &edge : trace->get_hypergraph()->get_incoming_edges(node)) {
                        const int ruleId = edge->get_label_id();
                        const size_t rule_dim = edge->get_sources().size() + 1;

                        switch (rule_dim) {
                            case 1:
                                compute_rule_count1(
                                        latentAnnotation.ruleWeights[ruleId]
                                        , lhnOutsideWeight
                                        , traceRootProbability(0)
                                        , counts.ruleCounts[ruleId]
                                );
                                break;
                            case 2:
                                compute_rule_count2(
                                        latentAnnotation.ruleWeights[ruleId]
                                        , edge
                                        , lhnOutsideWeight
                                        , traceRootProbability(0)
                                        , insideWeights
                                        , counts.ruleCounts[ruleId]
                                );
                                break;
                            case 3:
                                compute_rule_count3(
                                        latentAnnotation.ruleWeights[ruleId]
                                        , edge
                                        , lhnOutsideWeight
                                        , traceRootProbability(0)
                                        , insideWeights
                                        , counts.ruleCounts[ruleId]
                                );
                                break;
                            case 4:
                                compute_rule_count<4>(
                                        latentAnnotation.ruleWeights[ruleId]
                                        , edge
                                        , lhnOutsideWeight
                                        , traceRootProbability(0)
                                        , insideWeights
                                        , counts.ruleCounts[ruleId]
                                );
                                break;
                            default:
                                std::cerr << "Rules with RHS > " << 3 << " are not implemented." << std::endl;
                                abort();
                        }
                    }
                }

            }
            return counts;
        }

        inline void compute_rule_count1(const RuleTensor<double> & ruleWeight, const RuleTensorRaw<double, 1> &lhnOutsideWeight,
                                        const double traceRootProbability, RuleTensor<double> & ruleCount
        ) {
            constexpr unsigned ruleRank {1};

            const auto & ruleWeightRaw = boost::get<RuleTensorRaw <double, ruleRank>>(ruleWeight);

            auto ruleValue = ruleWeightRaw * lhnOutsideWeight;

            auto & ruleCountRaw = boost::get<RuleTensorRaw <double, ruleRank>>(ruleCount);

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr([traceRootProbability] (double x) {return x / traceRootProbability;});
            }
        }


        inline void compute_rule_count2(
                const RuleTensor<double> & ruleWeight
                , const Element<HyperEdge<Nonterminal>>& edge
                , const RuleTensorRaw<double, 1>& lhnOutsideWeight
                , const double traceRootProbability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector>& insideWeights
                , RuleTensor<double> & ruleCount
        ) {
            constexpr unsigned ruleRank {2};

            const auto & ruleWeightRaw = boost::get<RuleTensorRaw <double, ruleRank>>(ruleWeight);

            const auto & rhsWeight = insideWeights.at(edge->get_sources()[0]);
            auto ruleValue = lhnOutsideWeight.reshape(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1})
                                    .broadcast(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1)})
                            * rhsWeight.reshape(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1)})
                                    .broadcast(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1}).eval()
                            * ruleWeightRaw
            ;

            auto & ruleCountRaw = boost::get<RuleTensorRaw <double, ruleRank>>(ruleCount);

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr([traceRootProbability] (double x) {return x / traceRootProbability;});
            }
        }


        inline void compute_rule_count3(
                const RuleTensor<double> & ruleWeight
                , const Element<HyperEdge<Nonterminal>>& edge
                , const RuleTensorRaw <double, 1>& lhnOutsideWeight
                , const double traceRootProbability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector>& insideWeights
                , RuleTensor<double> & ruleCount
        ) {
            constexpr unsigned ruleRank {3};

            const auto & ruleWeightRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleWeight);
            const auto & rhsWeight1 = insideWeights.at(edge->get_sources()[0]);
            const auto & rhsWeight2 = insideWeights.at(edge->get_sources()[1]);

            auto ruleValue = lhnOutsideWeight.reshape(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1, 1})
                                    .broadcast(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1), ruleWeightRaw.dimension(2)})
                            * rhsWeight1.reshape(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1), 1})
                                    .broadcast(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1, ruleWeightRaw.dimension(2)}).eval()
                            * rhsWeight2.reshape(Eigen::array<long, ruleRank>{1, 1, ruleWeightRaw.dimension(2)})
                                    .broadcast(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), ruleWeightRaw.dimension(1), 1}).eval()
                            * ruleWeightRaw;
            ;

            auto & ruleCountRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleCount);

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr([traceRootProbability] (double x) {return x / traceRootProbability;});
            }
        }

        template<long rank>
        inline void compute_rule_count(
                const RuleTensor<double> & ruleWeight
                , const Element<HyperEdge<Nonterminal>>& edge
                , const RuleTensorRaw <double, 1>& lhnOutsideWeight
                , const double traceRootProbability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector>& insideWeights
                , RuleTensor<double> & ruleCount
        ) {

            const auto & ruleWeightRaw = boost::get<RuleTensorRaw <double, rank>>(ruleWeight);
            const auto & ruleDimension = ruleWeightRaw.dimensions();

            Eigen::array<long, rank> reshapeDimensions;
            Eigen::array<long, rank> broadcastDimensions;
            for (unsigned i = 0; i < rank; ++i) {
                reshapeDimensions[i] = 1;
                broadcastDimensions[i] = ruleDimension[i];
            }

            Eigen::Tensor<double, rank> rule_val = ruleWeightRaw;
            for (unsigned nont = 0; nont < rank; ++nont) {
                const auto & itemWeight = (nont == 0)
                                           ? lhnOutsideWeight
                                           : insideWeights.at(edge->get_sources()[nont - 1]);
                reshapeDimensions[nont] = broadcastDimensions[nont];
                broadcastDimensions[nont] = 1;
                rule_val *= itemWeight.reshape(reshapeDimensions).broadcast(broadcastDimensions);
                broadcastDimensions[nont] = reshapeDimensions[nont];
                reshapeDimensions[nont] = 1;
            }

            auto & ruleCountRaw = boost::get<RuleTensorRaw <double, rank>>(ruleCount);
            if (traceRootProbability > 0) {
                ruleCountRaw += rule_val.unaryExpr([traceRootProbability] (double x) {return x / traceRootProbability;});
            }
        }
    };

    class SimpleMaximizer : public Maximizer {
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        const bool debug;

    public:
        SimpleMaximizer(std::shared_ptr<const GrammarInfo2> grammarInfo, bool debug = false)
                : grammarInfo(grammarInfo)
                , debug(debug) {};
        void maximize(LatentAnnotation &latentAnnotation, const Counts &counts) {
            unsigned nont = 0;
            for (const std::vector<std::size_t> &group : grammarInfo->normalizationGroups) {
                const std::size_t lhsSplits = latentAnnotation.nonterminalSplits[nont];
                maximization(lhsSplits, counts, latentAnnotation, group);
                ++nont;
            }

            // maximize root weights:
            Eigen::Tensor<double, 0> corpusProbSum = counts.rootCounts.sum();
            if (corpusProbSum(0) > 0)
                latentAnnotation.rootWeights = latentAnnotation.rootWeights.unaryExpr(
                        [corpusProbSum](double x) {
                            return x / corpusProbSum(0);
                        }
                );
        }

    private:
        inline void maximization(
                const size_t lhsSplits
                , const Counts &counts
                , LatentAnnotation &latentAnnotation
                , const std::vector<std::size_t> &group
        ) {
            Eigen::Tensor<double, 1> lhsCounts(lhsSplits);
            lhsCounts.setZero();
            for (const size_t ruleId : group) {
                compute_normalization_divisor(lhsCounts, counts.ruleCounts[ruleId]);
            }

            for (const size_t ruleId : group) {
                normalize(latentAnnotation.ruleWeights[ruleId], counts.ruleCounts[ruleId], lhsCounts);
            }

        }
    };
}

#endif //STERMPARSER_EMTRAINERLA_H
