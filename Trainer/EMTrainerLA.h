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
#ifdef _OPENMP
# include <omp.h>
#endif

namespace Trainer {
    struct ruleSanityVisitor : boost::static_visitor<bool>
    {
        template<class T>
        bool operator()(const T& ruleCountRaw)const
        {
            RuleTensorRaw<bool, 0> insane = ruleCountRaw.isnan().any() || ruleCountRaw.isinf().any();
            if (insane(0)) {
                std::cerr << " rule count " << std::endl << ruleCountRaw << std::endl << " is numerically problematic " << std::endl;
                return false;
            }
            return true;
        }
    };


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
                switch ((*ruleCounts)[rule].which() + 1) {
                    case 1:
                        boost::get<RuleTensorRaw<double, 1>>((*ruleCounts)[rule])
                                += boost::get<RuleTensorRaw<double, 1>>((*other.ruleCounts)[rule]);
                        break;
                    case 2:
                        boost::get<RuleTensorRaw<double, 2>>((*ruleCounts)[rule])
                                += boost::get<RuleTensorRaw<double, 2>>((*other.ruleCounts)[rule]);
                        break;
                    case 3:
                        boost::get<RuleTensorRaw<double, 3>>((*ruleCounts)[rule])
                                += boost::get<RuleTensorRaw<double, 3>>((*other.ruleCounts)[rule]);
                        break;
                    case 4:
                        boost::get<RuleTensorRaw<double, 4>>((*ruleCounts)[rule])
                                += boost::get<RuleTensorRaw<double, 4>>((*other.ruleCounts)[rule]);
                        break;
                    default:
                        abort();
                }
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
                std::cerr << " root counts " << std::endl << rootCounts << std::endl << " are numerically problematic " << std::endl;
                return false;
            }
            auto visitor = ruleSanityVisitor();
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

    enum TrainingMode { Default, Splitting, Merging, Smoothing };


    class EMTrainerLA {
    protected:
        std::map<TrainingMode, unsigned> modeEpochs;
        unsigned epochs;
        std::shared_ptr<Expector> expector;
        std::shared_ptr<Maximizer> maximizer;
        TrainingMode trainingMode { Default };

        virtual void updateSettings() {
            if (modeEpochs.count(trainingMode))
                epochs = modeEpochs[trainingMode];
            else
                epochs = modeEpochs[Default];
        }

    public:
        EMTrainerLA(unsigned epochs, std::shared_ptr<Expector> expector, std::shared_ptr<Maximizer> maximizer)
                : epochs(epochs), expector(expector), maximizer(maximizer) {
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
                counts.is_numerically_sane();

                std::cerr << "Epoch " << epoch << "/" << epochs << ": ";

                // output likelihood information based on old probability assignment
                Eigen::Tensor<double, 0> corpusProbSum = counts.rootCounts.sum();
                std::cerr << "corpus prob. sum " << corpusProbSum;
                std::cerr << " corpus likelihood " << counts.logLikelihood;

                maximizer->maximize(latentAnnotation, counts);

                std::cerr << " root weights: " << latentAnnotation.rootWeights << std::endl;
            }
            expector->clean_up();
        }

    };

    class EMTrainerLAValidation : public EMTrainerLA {
        std::map<TrainingMode, unsigned> modeMaxDrops;
        std::shared_ptr<ValidationLA> validator;
        unsigned maxDrops {6};

        virtual void updateSettings() {
            EMTrainerLA::updateSettings();
            if (modeMaxDrops.count(trainingMode))
                maxDrops = modeMaxDrops[trainingMode];
            else
                maxDrops = modeMaxDrops[Default];
        }

    public:
        EMTrainerLAValidation(unsigned epochs
                              , std::shared_ptr<Expector> expector
                              , std::shared_ptr<Maximizer> maximizer
                              , std::shared_ptr<ValidationLA> validator
                              , unsigned maxDrops = 6)
                : EMTrainerLA(epochs, expector, maximizer) , validator(validator), maxDrops(maxDrops) {
            modeMaxDrops[Default] = maxDrops;
        };

        void setMaxDrops(unsigned maxDrops, TrainingMode mode=Default) {
            modeMaxDrops[mode] = maxDrops;
        }

        virtual void train(LatentAnnotation &latentAnnotation) {
            double previousValidationLikelihood = validator->minimum_score();
            unsigned drops = 0;
            unsigned epoch = 0;
            for (; epoch < epochs; ++epoch) {
                std::cerr << "Epoch " << epoch << "/" << epochs << ": ";

                double validationLikelihood = validator->validation_score(latentAnnotation);
                if (validationLikelihood < previousValidationLikelihood)
                    ++drops;
                else
                    drops = 0;
                previousValidationLikelihood = validationLikelihood;
                std::cerr << " validation corpus " << validator->quantity() << " " << validationLikelihood;

                if (drops >= maxDrops) {
                    std::cerr << std::endl;
                    break;
                }

                Counts counts {expector->expect(latentAnnotation)};

                // output likelihood information based on old probability assignment
                Eigen::Tensor<double, 0> corpusProbSum = counts.rootCounts.sum();
                std::cerr << " training corpus prob. sum " << corpusProbSum;
                std::cerr << " training corpus likelihood " << counts.logLikelihood;

                maximizer->maximize(latentAnnotation, counts);

                std::cerr << " root weights: " << latentAnnotation.rootWeights << std::endl;
            }
            if (epoch == epochs) {
                double validationLikelihood = validator->validation_score(latentAnnotation);
                std::cerr << " validation corpus " << validator->quantity() << " " << validationLikelihood << std::endl;
            }
            expector->clean_up();
            validator->clean_up();
        }

    };

    inline double safe_division(double numerator, double denominator) {
        double quotient = numerator / denominator;
        if (not std::isnan(quotient) or std::isinf(quotient))
            return quotient;
        else {
            if (numerator >= denominator)
                return 1;
            else
                return 0;
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
                        *latentAnnotation.ruleWeights
                        , latentAnnotation.rootWeights
                        , tracesInsideWeights[traceIterator - traceManager->cbegin()]
                        , tracesOutsideWeights[traceIterator - traceManager->cbegin()]
                );

                const auto &insideWeights = tracesInsideWeights[traceIterator - traceManager->cbegin()];
                const auto &outsideWeights = tracesOutsideWeights[traceIterator - traceManager->cbegin()];

                Eigen::Tensor<double, 1> traceRootProbabilities {compute_trace_root_probabilities(traceIterator, latentAnnotation)};
                Eigen::Tensor<double, 0> traceRootProbability = traceRootProbabilities.sum();
                Eigen::Tensor<bool, 0> traceRootProbabilitiesImplausible
                        = (traceRootProbabilities.isinf().any() || traceRootProbabilities.isnan().any());
                const double scale = compute_counting_scalar(traceRootProbabilities, traceIterator, latentAnnotation);

                if ( not traceRootProbabilitiesImplausible(0)
                     and not std::isnan(traceRootProbability(0))
                     and not std::isinf(traceRootProbability(0))
                     and traceRootProbability(0) > 0
                    ) {
                    counts.rootCounts += traceRootProbabilities;
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
                    std::cerr << "trace Root Probability " << traceRootProbability(0) << std::endl;
                    counts.logLikelihood += minus_infinity;
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
                        const size_t rule_dim = edge->get_sources().size() + 1;

                        switch (rule_dim) {
                            case 1:
                                compute_rule_count1(
                                        (*latentAnnotation.ruleWeights)[ruleId]
                                        , lhnOutsideWeight
                                        , traceRootProbability(0)
                                        , (*counts.ruleCounts)[ruleId]
                                        , scale
                                );
                                break;
                            case 2:
                                compute_rule_count2(
                                        (*latentAnnotation.ruleWeights)[ruleId]
                                        , edge
                                        , lhnOutsideWeight
                                        , traceRootProbability(0)
                                        , insideWeights
                                        , (*counts.ruleCounts)[ruleId]
                                        , scale
                                );
                                break;
                            case 3:
                                compute_rule_count3(
                                        (*latentAnnotation.ruleWeights)[ruleId]
                                        , edge
                                        , lhnOutsideWeight
                                        , traceRootProbability(0)
                                        , insideWeights
                                        , (*counts.ruleCounts)[ruleId]
                                        , scale
                                );
                                break;
                            case 4:
                                compute_rule_count<4>(
                                        (*latentAnnotation.ruleWeights)[ruleId]
                                        , edge
                                        , lhnOutsideWeight
                                        , traceRootProbability(0)
                                        , insideWeights
                                        , (*counts.ruleCounts)[ruleId]
                                        , scale
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

        inline void compute_rule_count1(
                const RuleTensor<double> &ruleWeight
                , const RuleTensorRaw<double, 1> &lhnOutsideWeight
                , const double traceRootProbability
                , RuleTensor<double> &ruleCount
                , const double scale = 1.0
        ) {
            constexpr unsigned ruleRank{1};

            const auto &ruleWeightRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleWeight);

            auto ruleValue = ruleWeightRaw * lhnOutsideWeight * scale;

            auto &ruleCountRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleCount);

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr(
                        [traceRootProbability](double x) {
                            return safe_division(x, traceRootProbability);
                        }
                );
            }
        }


        inline void compute_rule_count2(
                const RuleTensor<double> &ruleWeight
                , const Element<HyperEdge<Nonterminal>> &edge
                , const RuleTensorRaw<double, 1> &lhnOutsideWeight
                , const double traceRootProbability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &insideWeights
                , RuleTensor<double> &ruleCount
                , const double scale = 1.0
        ) {
            constexpr unsigned ruleRank{2};

            const auto &ruleWeightRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleWeight);

            const auto &rhsWeight = insideWeights.at(edge->get_sources()[0]);
            auto ruleValue = lhnOutsideWeight.reshape(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1})
                                     .broadcast(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1)})
                             * rhsWeight.reshape(Eigen::array<long, ruleRank>{1, ruleWeightRaw.dimension(1)})
                                     .broadcast(Eigen::array<long, ruleRank>{ruleWeightRaw.dimension(0), 1}).eval()
                             * ruleWeightRaw
                             * scale;

            auto &ruleCountRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleCount);

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr(
                        [traceRootProbability](double x) {
                            return safe_division(x, traceRootProbability);
                        }
                );
            }
        }


        inline void compute_rule_count3(
                const RuleTensor<double> &ruleWeight
                , const Element<HyperEdge<Nonterminal>> &edge
                , const RuleTensorRaw<double, 1> &lhnOutsideWeight
                , const double traceRootProbability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &insideWeights
                , RuleTensor<double> &ruleCount
                , const double scale = 1.0
        ) {
            constexpr unsigned ruleRank{3};

            const auto &ruleWeightRaw = boost::get<RuleTensorRaw<double, ruleRank>>(ruleWeight);
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

            if (traceRootProbability > 0) {
                ruleCountRaw += ruleValue.unaryExpr(
                        [traceRootProbability](double x) {
                            return safe_division(x, traceRootProbability);
                        }
                );
            }
        }

        template<long rank>
        inline void compute_rule_count(
                const RuleTensor<double> &ruleWeight
                , const Element<HyperEdge<Nonterminal>> &edge
                , const RuleTensorRaw<double, 1> &lhnOutsideWeight
                , const double traceRootProbability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector> &insideWeights
                , RuleTensor<double> &ruleCount
                , const double scale = 1.0
        ) {

            const auto &ruleWeightRaw = boost::get<RuleTensorRaw<double, rank>>(ruleWeight);
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
            if (traceRootProbability > 0) {
                ruleCountRaw += rule_val.unaryExpr(
                        [traceRootProbability](double x) {
                            return safe_division(x, traceRootProbability);
                        }
                );
            }
        }
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
                    *latentAnnotation.ruleWeights
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

            double scale = p_x(0) / p_xy_joined(0);
            if (std::isinf(scale) or scale < 1.0 or std::isnan(scale))
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

            latentAnnotation.is_proper(grammarInfo);
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
}

#endif //STERMPARSER_EMTRAINERLA_H
