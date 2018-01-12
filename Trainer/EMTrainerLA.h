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
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "LatentAnnotation.h"
#include "Validation.h"
#include <memory>
#include <algorithm>
#include <iostream>
#ifdef _OPENMP
# include <omp.h>
#endif

namespace Trainer {
    class Counts : boost::addable<Counts> {
    public:
        std::shared_ptr<StorageManager> storageManager;
        double logLikelihood;
        std::unique_ptr<std::vector<RuleTensor<double>>> ruleCounts;
        Eigen::Tensor<double, 1> rootCounts;

        Counts(
                const LatentAnnotation &latentAnnotation
                , const GrammarInfo2 &grammarInfo
                , std::shared_ptr<StorageManager> storageManager
        )
                : storageManager(storageManager), logLikelihood(0),
                  ruleCounts(std::make_unique<std::vector<RuleTensor<double>>>()),
                  rootCounts(latentAnnotation.rootWeights.dimension(0)) {
            for (size_t ruleId = 0; ruleId < latentAnnotation.ruleWeights->size(); ++ruleId) {
                ruleCounts->push_back(
                        storageManager->create_uninitialized_tensor(
                                ruleId
                                , grammarInfo
                                , latentAnnotation.nonterminalSplits
                        ));
                set_zero(ruleCounts->back());
            }
            rootCounts.setZero();
        }

        Counts(const Counts &other)
                : storageManager(other.storageManager), logLikelihood(other.logLikelihood),
                  ruleCounts(std::make_unique<std::vector<RuleTensor<double>>>()),
                  rootCounts(Eigen::Tensor<double, 1>(other.rootCounts.dimensions())) {
            for (const auto &ruleTensor : *other.ruleCounts) {
                ruleCounts->push_back(
                        storageManager->copy_tensor(ruleTensor)
                );
            }
            rootCounts = other.rootCounts;
        }

        Counts &operator+=(const Counts &other) {
            logLikelihood += other.logLikelihood;

            for (unsigned rule = 0; rule < ruleCounts->size(); ++rule) {
                TensorAdder va((*ruleCounts)[rule]);
                boost::apply_visitor(va, (*other.ruleCounts)[rule]);
            }

            rootCounts += other.rootCounts;
            return *this;
        }

        bool is_numerically_sane() {
            if (std::isnan(logLikelihood)) {
                std::cerr << " log likelihood is " << logLikelihood << std::endl;
                return false;
            }
            RuleTensorRaw<bool, 0> insane = rootCounts.isnan().any() || rootCounts.isinf().any();
            if (insane(0)) {
                std::cerr << " root counts " << std::endl << rootCounts << std::endl
                          << " are numerically problematic " << std::endl;
                return false;
            }
            auto visitor = ValidityChecker();
            for (auto ruleCount : *ruleCounts) {
                if (not ruleCount.apply_visitor(visitor))
                    return false;
            }

            return true;
        }

    };

    class Expector {
    public:
        virtual Counts expect(const LatentAnnotation &latentAnnotation) = 0;

        virtual void clean_up() = 0;
    };

    class Maximizer {
    public:
        virtual void maximize(LatentAnnotation &, const Counts &) = 0;
    };

    class CountsModifier {
    public:
        virtual void modifyCounts(Counts&) {};
    };

    enum TrainingMode { Default, Splitting, Merging, Smoothing };


    class EMTrainerLA {
    protected:
        std::map<TrainingMode, unsigned> modeEpochs;
        unsigned epochs;
        std::shared_ptr<Expector> expector;
        std::shared_ptr<Maximizer> maximizer;
        std::shared_ptr<CountsModifier> countsModifier;
        TrainingMode trainingMode { Default };

        virtual void updateSettings() {
            if (modeEpochs.count(trainingMode))
                epochs = modeEpochs[trainingMode];
            else
                epochs = modeEpochs[Default];
        }

    public:
        EMTrainerLA(unsigned epochs
                    , std::shared_ptr<Expector> expector
                    , std::shared_ptr<Maximizer> maximizer
                    , std::shared_ptr<CountsModifier> countsModifier = std::make_shared<CountsModifier>())
                : epochs(epochs), expector(expector), maximizer(maximizer), countsModifier(countsModifier) {
            modeEpochs[Default] = epochs;
        };

        void setTrainingMode(TrainingMode trainingMode) {
            EMTrainerLA::trainingMode = trainingMode;
            updateSettings();
        }

        void setEMepochs(unsigned epochs, TrainingMode mode=Default) {
            modeEpochs[mode] = epochs;
        }

        virtual void train(LatentAnnotation &latentAnnotation) {
            for (unsigned epoch = 0; epoch < epochs; ++epoch) {
                Counts counts {expector->expect(latentAnnotation)};
                countsModifier->modifyCounts(counts);
                counts.is_numerically_sane();

                std::cerr << "Epoch " << epoch << "/" << epochs << ": ";

                // output likelihood information based on old probability assignment
                Eigen::Tensor<double, 0> corpusProbSum = counts.rootCounts.sum();
                std::cerr << "training root counts " << corpusProbSum;
                std::cerr << " corpus likelihood " << counts.logLikelihood;

                maximizer->maximize(latentAnnotation, counts);

                std::cerr << " root weights: " << latentAnnotation.rootWeights << std::endl;
            }
            expector->clean_up();
        }

    };

    class EMTrainerLAValidation : public EMTrainerLA {
        std::map<TrainingMode, unsigned> modeMinEpochs;
        std::map<TrainingMode, unsigned> modeMaxDrops;
        std::shared_ptr<ValidationLA> validator;
        unsigned maxDrops {6};
        unsigned minEpochs {6};

        virtual void updateSettings() {
            EMTrainerLA::updateSettings();
            if (modeMaxDrops.count(trainingMode))
                maxDrops = modeMaxDrops[trainingMode];
            else
                maxDrops = modeMaxDrops[Default];
            if (modeMinEpochs.count(trainingMode))
                minEpochs = modeMinEpochs[trainingMode];
            else
                minEpochs = modeMinEpochs[Default];
        }

    public:
        EMTrainerLAValidation(unsigned epochs
                              , std::shared_ptr<Expector> expector
                              , std::shared_ptr<Maximizer> maximizer
                              , std::shared_ptr<ValidationLA> validator
                              , std::shared_ptr<CountsModifier> countsModifier
                              , unsigned minEpochs = 6
                              , unsigned maxDrops = 6)
                : EMTrainerLA(epochs, expector, maximizer, countsModifier) , validator(validator), maxDrops(maxDrops) {
            modeMaxDrops[Default] = maxDrops;
            modeMinEpochs[Default] = minEpochs;
        };

        void setMaxDrops(unsigned maxDrops, TrainingMode mode=Default) {
            modeMaxDrops[mode] = maxDrops;
        }

        void setMinEpochs(unsigned minEpochs, TrainingMode mode = Default) {
            modeMinEpochs[mode] = minEpochs;
        }

        virtual void train(LatentAnnotation &latentAnnotation) {
            double previousValidationScore {validator->minimum_score()};
            double bestValidationScore {previousValidationScore};
            double validationScore {previousValidationScore};
            LatentAnnotation bestAnnotation {latentAnnotation};
            unsigned drops {0};
            unsigned epoch {0};
            unsigned bestEpoch {epoch};
            for (; epoch < epochs; ++epoch) {
                std::cerr << "Epoch " << epoch << "/" << epochs << ": ";

                validationScore = validator->validation_score(latentAnnotation);
                if (validationScore < previousValidationScore)
                    ++drops;
                else {
                    drops = 0;
                }
                previousValidationScore = validationScore;

                if (epoch < minEpochs or validationScore >= bestValidationScore) {
                    bestValidationScore = validationScore;
                    bestAnnotation = latentAnnotation;
                    bestEpoch = epoch;
                }

                std::cerr << " validation corpus " << validator->quantity() << " " << validationScore
                          << " validation failures " << validator->getParseFailures();

                if (epoch >= minEpochs and drops >= maxDrops) {
                    std::cerr << std::endl;
                    break;
                }

                Counts counts {expector->expect(latentAnnotation)};
                countsModifier->modifyCounts(counts);

                // output likelihood information based on old probability assignment
                Eigen::Tensor<double, 0> corpusProbSum = counts.rootCounts.sum();
                std::cerr << " training root counts " << corpusProbSum;
                std::cerr << " training corpus likelihood " << counts.logLikelihood;

                maximizer->maximize(latentAnnotation, counts);

                std::cerr << " root weights: " << latentAnnotation.rootWeights << std::endl;
            }
            if (epoch == epochs) {
                validationScore = validator->validation_score(latentAnnotation);
                std::cerr << " validation corpus " << validator->quantity() << " " << validationScore
                          << " validation failures " << validator->getParseFailures() << std::endl;

                if (epoch < minEpochs or validationScore >= bestValidationScore) {
                    bestValidationScore = validationScore;
                    bestAnnotation = latentAnnotation;
                    bestEpoch = epoch;
                }
            }
            expector->clean_up();
            validator->clean_up();

            if (bestEpoch != epoch) {
                std::cerr << " resetting to annotation from epoch " << bestEpoch
                          << " with validation score " << bestValidationScore << std::endl;
                latentAnnotation = bestAnnotation;
            }
        }

    };

    template<typename Nonterminal>
    struct RuleCountComputer : boost::static_visitor<void> {
        const Element<HyperEdge<Nonterminal>> &edge;
        const RuleTensorRaw<double, 1> &lhnOutsideWeight;
        const double traceRootProbability;
        const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &insideWeights;
        RuleTensor<double> & ruleCount;
        const double scale;

        RuleCountComputer(
              const Element<HyperEdge<Nonterminal>> & edge
            , const RuleTensorRaw<double, 1> &lhnOutsideWeight
            , const double traceRootProbability
            , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> & insideWeights
            , RuleTensor<double> &ruleCount
            , const double scale = 1.0
        ) : edge(edge)
                , lhnOutsideWeight(lhnOutsideWeight)
                , traceRootProbability(traceRootProbability)
                , insideWeights(insideWeights)
                , ruleCount(ruleCount)
                , scale(scale) {};

        template<int rank>
        inline
        typename std::enable_if<(rank > 3), void>::type
        operator()(const RuleTensorRaw<double, rank> & ruleWeightRaw) {
            const auto &ruleDimension = ruleWeightRaw.dimensions();

            Eigen::array<long, rank> reshapeDimensions;
            Eigen::array<long, rank> broadcastDimensions;
            for (unsigned i = 0; i < rank; ++i) {
                reshapeDimensions[i] = 1;
                broadcastDimensions[i] = ruleDimension[i];
            }

            Eigen::Tensor<double, rank> rule_val = ruleWeightRaw;
            for (unsigned nont = 0; nont < rank; ++nont) {
                const auto &itemWeight = (nont == 0)
                                         ? lhnOutsideWeight
                                         : insideWeights.at(edge->get_sources()[nont - 1]);
                reshapeDimensions[nont] = broadcastDimensions[nont];
                broadcastDimensions[nont] = 1;
                rule_val *= itemWeight.reshape(reshapeDimensions).broadcast(broadcastDimensions);
                broadcastDimensions[nont] = reshapeDimensions[nont];
                reshapeDimensions[nont] = 1;
            }
            rule_val = rule_val * scale;

            auto &ruleCountRaw = boost::get<RuleTensorRaw<double, rank>>(ruleCount);
            const double trp {traceRootProbability};

            if (traceRootProbability > 0) {
                ruleCountRaw += rule_val.unaryExpr(
                        [trp](double x) {
                            return safe_division(x, trp);
                        }
                );
            }
        }

        inline void operator()(const RuleTensorRaw<double, 1> & ruleWeightRaw) {
            auto ruleValue = ruleWeightRaw * lhnOutsideWeight * scale;

            auto &ruleCountRaw = boost::get<RuleTensorRaw<double, 1>>(ruleCount);
            const double trp {traceRootProbability};

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr(
                        [trp](double x) {
                            return safe_division(x, trp);
                        }
                );
            }
        }

        inline void operator()(const RuleTensorRaw<double, 2> & ruleWeightRaw) {
            constexpr unsigned ruleRank{2};

            const auto &rhsWeight = insideWeights.at(edge->get_sources()[0]);
            auto ruleValue = lhnOutsideWeight.reshape(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1})
                                     .broadcast(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1)})
                             * rhsWeight.reshape(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1)})
                                     .broadcast(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1}).eval()
                             * ruleWeightRaw
                             * scale;

            auto &ruleCountRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleCount);
            const double trp {traceRootProbability};

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr(
                        [trp](double x) {
                            return safe_division(x, trp);
                        }
                );
            }
        }

        inline void operator()(const RuleTensorRaw<double, 3> & ruleWeightRaw) {
            constexpr unsigned ruleRank{3};

            const auto &rhsWeight1 = insideWeights.at(edge->get_sources()[0]);
            const auto &rhsWeight2 = insideWeights.at(edge->get_sources()[1]);

            auto ruleValue = lhnOutsideWeight.reshape(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1, 1})
                                     .broadcast(
                                             Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1),
                                                                          ruleWeightRaw.dimension(2)}
                                     )
                             * rhsWeight1.reshape(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1), 1})
                                     .broadcast(
                                             Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1,
                                                                          ruleWeightRaw.dimension(2)}
                                     ).eval()
                             * rhsWeight2.reshape(Eigen::array<long, ruleRank>{1, 1, ruleWeightRaw.dimension(2)})
                                     .broadcast(
                                             Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0),
                                                                          ruleWeightRaw.dimension(1), 1}
                                     ).eval()
                             * ruleWeightRaw
                             * scale;

            auto &ruleCountRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleCount);
            const double trp {traceRootProbability};

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr(
                        [trp](double x) {
                            return safe_division(x, trp);
                        }
                );
            }
        }

    };


    template<typename Nonterminal, typename TraceID>
    class SimpleExpector : public Expector {
    protected:
        template<typename T1, typename T2>
        using MAPTYPE = typename std::unordered_map<T1, T2>;
        using TraceIterator = ConstManagerIterator<Trace < Nonterminal, TraceID>>;
        const TraceManagerPtr <Nonterminal, TraceID> traceManager;
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;

    private:
        const unsigned threads;

    protected:
        const bool debug;
        std::vector<MAPTYPE<Element<Node<Nonterminal>>, WeightVector>> tracesInsideWeights;
        std::vector<MAPTYPE<Element<Node<Nonterminal>>, WeightVector>> tracesOutsideWeights;

    public:
        SimpleExpector(
                TraceManagerPtr <Nonterminal, TraceID> traceManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::shared_ptr<StorageManager> storageManager
                , unsigned threads = 1
                , bool debug = false
        )
                : traceManager(traceManager)
                , grammarInfo(grammarInfo)
                , storageManager(storageManager)
                , threads(threads)
                , debug(debug) {};

        Counts expect(const LatentAnnotation &latentAnnotation) {
            if (traceManager->cend() != traceManager->cbegin() + traceManager->size()) {
                std::cerr << "end - begin " << traceManager->cend() - traceManager->cbegin() << std::endl;
                std::cerr << "size: " << traceManager->size();
                std::abort();
            }
            if (tracesInsideWeights.size() < traceManager->size()) {
                tracesInsideWeights.resize(traceManager->size());
            }
            if (tracesOutsideWeights.size() < traceManager->size()) {
                tracesOutsideWeights.resize(traceManager->size());
            }
            return expectation_la(latentAnnotation);
        }

        void clean_up() {
            storageManager->free_weight_maps(tracesInsideWeights);
            storageManager->free_weight_maps(tracesOutsideWeights);
        }

    private:
        inline Counts expectation_la(
                const LatentAnnotation &latentAnnotation
        ) {
            Counts counts(latentAnnotation, *grammarInfo, storageManager);
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
            #pragma omp declare reduction (+ : Counts : omp_out += omp_in ) initializer (omp_priv = omp_orig)
            #pragma omp parallel for schedule(dynamic, 10) reduction (+:counts)
            for (TraceIterator traceIterator = traceManager->cbegin(); traceIterator < traceManager->cend(); ++traceIterator) {
                const auto &trace = *traceIterator;
                if (trace->get_hypergraph()->size() == 0)
                    continue;

                if (traceManager->cbegin() + tracesOutsideWeights.size()  <= traceIterator
                        or traceManager->cbegin() + tracesInsideWeights.size() <= traceIterator) {
                    std::cerr << "tried to access non-existent inside or outside weight map" << std::endl;
                    std::cerr << "it - begin " << traceIterator - traceManager->cbegin() << std::endl;
                    std::cerr << "out size: " << tracesOutsideWeights.size() << std::endl;
                    std::cerr << "in size: " << tracesInsideWeights.size() << std::endl;
                    abort();
                }
                // create insert inside and outside weight for each node if necessary
                if (tracesInsideWeights[traceIterator - traceManager->cbegin()].size() !=
                    trace->get_hypergraph()->size() or
                        tracesOutsideWeights[traceIterator - traceManager->cbegin()].size() !=
                        trace->get_hypergraph()->size()) {
                    tracesInsideWeights[traceIterator - traceManager->cbegin()].clear();
                    tracesOutsideWeights[traceIterator - traceManager->cbegin()].clear();
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
                          latentAnnotation
                        , tracesInsideWeights[traceIterator - traceManager->cbegin()]
                        , tracesOutsideWeights[traceIterator - traceManager->cbegin()]
                );

                const auto &insideWeights = tracesInsideWeights[traceIterator - traceManager->cbegin()];
                const auto &outsideWeights = tracesOutsideWeights[traceIterator - traceManager->cbegin()];

                Eigen::Tensor<double, 1> traceRootProbabilities {compute_trace_root_probabilities(traceIterator, latentAnnotation)};
                Eigen::Tensor<double, 0> traceRootProbability = traceRootProbabilities.sum();
                Eigen::Tensor<bool, 0> traceRootProbabilitiesImplausible
                        = (traceRootProbabilities.isinf().any() || traceRootProbabilities.isnan().any());
                const double scale = trace->get_frequency()
                                     * compute_counting_scalar(traceRootProbabilities, traceIterator, latentAnnotation);

                if ( not traceRootProbabilitiesImplausible(0)
                     and not std::isnan(traceRootProbability(0))
                     and not std::isinf(traceRootProbability(0))
                     and traceRootProbability(0) > 0
                    ) {
                    counts.rootCounts += scale * traceRootProbabilities / traceRootProbability(0);
                    counts.logLikelihood += log(traceRootProbability(0));

                    Eigen::Tensor<bool, 0> badCounts = counts.rootCounts.isinf().any() || counts.rootCounts.isnan().any();
                    if (badCounts(0)) {
                        std::cerr << "bad root counts " << std::endl << counts.rootCounts << std::endl
                                  << " after adding counts " << std::endl << traceRootProbabilities << std::endl
                                  << " the sum of the root counts is " << std::endl << traceRootProbability
                                  << " resp. " << traceRootProbability(0) << std::endl
                                  << " for trace position " << traceIterator - traceManager->cbegin() << std::endl;
                        abort();
                    }
                } else {
                    // std::cerr << "trace Root Probability " << traceRootProbability(0) << std::endl;
                    // Although formally correct, we do not do this to improve robustness.
                    // counts.logLikelihood += minus_infinity;
                    continue;
                }

                if (debug)
                    std::cerr << "instance root probability: " << std::endl << traceRootProbabilities << std::endl;

                for (const Element<Node<Nonterminal>>
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
//                        const size_t rule_dim = edge->get_sources().size() + 1;
                        RuleCountComputer<Nonterminal> ruleCountComputer(edge
                                                            , lhnOutsideWeight
                                                            , traceRootProbability(0)
                                                            , insideWeights
                                                            , (*counts.ruleCounts)[ruleId]
                                                            , scale);
                        boost::apply_visitor(ruleCountComputer, (*latentAnnotation.ruleWeights)[ruleId]);
                    }
                }

            }
            return counts;
        }

    protected:
        Eigen::Tensor<double, 1> compute_trace_root_probabilities(TraceIterator traceIterator
                                                                          , const LatentAnnotation &) {
            const auto &rootInsideWeight
                    = tracesInsideWeights[traceIterator - traceManager->cbegin()].at(traceIterator->get_goal());
            const auto &rootOutsideWeight
                    = tracesOutsideWeights[traceIterator - traceManager->cbegin()].at(traceIterator->get_goal());
            return Eigen::Tensor<double, 1> {rootOutsideWeight * rootInsideWeight};
        }

        virtual double compute_counting_scalar( const Eigen::Tensor<double, 1> & /*rootWeight*/
                                               , TraceIterator
                                               , const LatentAnnotation &) {
            return 1.0;
        }

    private:


    };


    template <typename Nonterminal, typename TraceID>
    class DiscriminativeExpector : public SimpleExpector<Nonterminal, TraceID> {
        using TraceIterator = ConstManagerIterator<Trace < Nonterminal, TraceID>>;
        TraceManagerPtr <Nonterminal, TraceID> conditionalTraceManager;
        std::shared_ptr<StorageManager> myStorageManager;
        const double maxScale;

        virtual double compute_counting_scalar( const Eigen::Tensor<double, 1> & rootWeight
                                                , TraceIterator traceIterator
                                                , const LatentAnnotation & latentAnnotation) {

            TraceIterator cit {conditionalTraceManager->cbegin() + (traceIterator - this->traceManager->cbegin())};
            const auto & trace = *cit;

            MAPTYPE<Element<Node<Nonterminal>>, WeightVector> insideWeights;
            for (const auto &node : *(trace->get_hypergraph())) {
                insideWeights[node] = myStorageManager
                        ->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]);
                insideWeights[node].setZero();
            }

            trace->inside_weights_la(
                      latentAnnotation
                    , insideWeights
            );

            if (false)
                for (auto & node : trace->get_topological_order()) {
                    std::cerr << node << " : " << insideWeights.at(node);
                    if (node == cit->get_goal())
                        std::cerr << " " << "goal " << std::endl;
                    else
                        std::cerr << std::endl;
                }

            Eigen::Tensor<double, 0> p_xy_joined = rootWeight.sum();
            Eigen::Tensor<double, 0> p_x {(latentAnnotation.rootWeights * insideWeights.at(cit->get_goal())).sum()};
            if (SimpleExpector<Nonterminal, TraceID>::debug or p_xy_joined(0) > p_x(0))
                std::cerr << "p(x,y)/p(x) = " << p_xy_joined(0) << '/' << p_x(0) << ", ";

            double scale = safe_division(p_x(0), p_xy_joined(0));
            if (scale < 1.0)
                return 1.0;
            else
                return std::min(scale, maxScale);
        }

    public:
        DiscriminativeExpector(TraceManagerPtr <Nonterminal, TraceID> traceManager
                               , TraceManagerPtr <Nonterminal, TraceID> conditionalTraceManager
                               , std::shared_ptr<const GrammarInfo2> grammarInfo
                               , std::shared_ptr<StorageManager> storageManager
                               , double maxScale = std::numeric_limits<double>::infinity()
                               , unsigned threads = 1
                               , bool debug = false
        ) : SimpleExpector<Nonterminal, TraceID>::SimpleExpector(
                traceManager
                , grammarInfo
                , storageManager
                , threads
                , debug
                )
            , conditionalTraceManager(conditionalTraceManager), myStorageManager(storageManager), maxScale(maxScale) {
            if (not traceManager->size() == conditionalTraceManager->size()) {
                std::cerr << "size missmatch of trace manager (" << traceManager->size()
                          <<") and conditional trace manager (" << conditionalTraceManager->size() << ")" << std::endl;
                abort();
            }
            // check that each hypergraph in traceManager is a subgraph of its correspondent in conditionalTraceManager
            // subgraph checking is NP-complete and our implementation is quite naive:
            // i.e., it takes forever in practical settings
            if (debug) {
                for (size_t idx = 0; idx < traceManager->size(); ++idx) {
                    if (not Manage::is_sub_hypergraph(
                            (conditionalTraceManager->cbegin() + idx)->get_hypergraph()
                            , (traceManager->cbegin() + idx)->get_hypergraph()
                            , (conditionalTraceManager->cbegin() + idx)->get_goal()
                            , (traceManager->cbegin() + idx)->get_goal())) {
                        std::cerr << "Hypergraph " << idx << " violates subhypergraph property." << std::endl;
                        abort();
                    }
                }
            }


        }
    };

    class SimpleMaximizer : public Maximizer {
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        const unsigned threads;
        const bool debug;

    public:
        SimpleMaximizer(std::shared_ptr<const GrammarInfo2> grammarInfo, unsigned threads = 1, bool debug = false)
                : grammarInfo(grammarInfo), threads(threads), debug(debug) {};

        void maximize(LatentAnnotation &latentAnnotation, const Counts &counts) {
#ifdef _OPENMP
            omp_set_num_threads(threads);
            #pragma omp parallel for schedule(dynamic, 20)
#endif
            for (size_t nont = 0; nont < grammarInfo->normalizationGroups.size(); ++nont) {
                const std::vector<std::size_t> &group = grammarInfo->normalizationGroups[nont];
                const std::size_t lhsSplits = latentAnnotation.nonterminalSplits[nont];
                maximization(lhsSplits, counts, latentAnnotation, group);
            }

            // maximize root weights:
            Eigen::Tensor<double, 0> corpusProbSum = counts.rootCounts.sum();
            if (corpusProbSum(0) > 0 and not std::isnan(corpusProbSum(0)) and not std::isinf(corpusProbSum(0)))
                latentAnnotation.rootWeights = counts.rootCounts.unaryExpr(
                        [corpusProbSum](double x) {
                            return x / corpusProbSum(0);
                        }
                );

            latentAnnotation.is_proper();
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
                compute_normalization_divisor(lhsCounts, (*counts.ruleCounts)[ruleId]);
            }

            for (const size_t ruleId : group) {
                normalize((*latentAnnotation.ruleWeights)[ruleId], (*counts.ruleCounts)[ruleId], lhsCounts);
            }

        }
    };

    struct CountSmoothVisitor : boost::static_visitor<void> {
        const double smoothValue;

        CountSmoothVisitor(double smoothValue) : smoothValue(smoothValue) {}

        template<int rank>
        void operator()(Eigen::Tensor<double, rank>& tensor) const {
            const long chip_size {tensor.size() / tensor.dimension(0)};
            const double summand { smoothValue / (1.0 * chip_size) };
            tensor = tensor.unaryExpr(
                    [summand](const double x) -> double {
                        return x + summand;
                    });
        }
    };

    class CountsSmoother : public CountsModifier {
        const std::vector<size_t> ruleIDs;
        const CountSmoothVisitor csv;
    public:
        CountsSmoother(const std::vector<size_t> ruleIDs, double smoothValue)
                : ruleIDs(ruleIDs), csv(CountSmoothVisitor(smoothValue)) {};

        virtual void modifyCount(Counts & counts) {
            for (auto ruleID : ruleIDs) {
                boost::apply_visitor(csv, (*counts.ruleCounts)[ruleID]);
            }
        };
    };
}

#endif //STERMPARSER_EMTRAINERLA_H
