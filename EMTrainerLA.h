//
// Created by kilian on 01/03/17.
//

#ifndef STERMPARSER_EMTRAINERLA_H
#define STERMPARSER_EMTRAINERLA_H

#include "Names.h"
#include "TrainingCommon.h"
#include "StorageManager.h"
#include "TraceManager.h"
#include "Trace.h"

namespace Trainer {
    class Counts {
    public:
        std::vector<RuleTensor<double>> ruleCounts;
        Eigen::Tensor<double, 1> rootCounts;
        double logLikelihood;

        template<typename Nonterminal>
        Counts(LatentAnnotation latentAnnotation, const GrammarInfo2<Nonterminal> &grammarInfo, StorageManager & storageManager)
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
                Eigen::Tensor<double, 0>corpus_prob_sum = counts.rootCounts.sum();
                std::cerr << "corpus prob. sum " << corpus_prob_sum;
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
        std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;
        const bool debug;

        std::vector<MAPTYPE<Element<Node < Nonterminal>>, WeightVector>> traces_inside_weights;
        std::vector<MAPTYPE<Element<Node < Nonterminal>>, WeightVector>> traces_outside_weights;

    public:
        SimpleExpector(
                TraceManagerPtr<Nonterminal, TraceID> traceManager
                , std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo
                , std::shared_ptr<StorageManager> storageManager
                , bool debug = false
        )
                : traceManager(traceManager), grammarInfo(grammarInfo), storageManager(storageManager), debug(debug) {};

        Counts expect(const LatentAnnotation latentAnnotation) {
            return expectation_la(latentAnnotation, traceManager->cbegin(), traceManager->cend());
        }

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

    private:
        inline Counts expectation_la(
                const LatentAnnotation latentAnnotation
                , const TraceIterator start
                , const TraceIterator end
        ) {
            Counts counts(latentAnnotation, *grammarInfo, *storageManager);

            const std::vector<RuleTensor<double>> &rule_tensors = latentAnnotation.ruleWeights;
            const WeightVector &root_probability = latentAnnotation.rootWeights;

            std::vector<RuleTensor<double>> &rule_count_tensors = counts.ruleCounts;
            Eigen::Tensor<double, 1> &root_count = counts.rootCounts;
            double &corpus_likelihood = counts.logLikelihood;


            for (auto traceIterator = start; traceIterator < end; ++traceIterator) {
                const auto &trace = *traceIterator;
                if (trace->get_hypergraph()->size() == 0)
                    continue;

                // create maps for inside and outside maps if necessary
                // todo: make this thread-safe
                if (traces_inside_weights.size() <= traceIterator - traceManager->cbegin()) {
                    traces_inside_weights.resize(1 + (traceIterator - traceManager->cbegin()));
                }
                if (traces_outside_weights.size() <= traceIterator - traceManager->cbegin()) {
                    traces_outside_weights.resize(1 + (traceIterator - traceManager->cbegin()));
                }
                if (traces_inside_weights[traceIterator - traceManager->cbegin()].size() != trace->get_hypergraph()->size()) {
                    for (const auto &node : *(trace->get_hypergraph())) {
                        traces_inside_weights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                        traces_outside_weights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                    }
                }

                trace->io_weights_la(
                        rule_tensors
                        , root_probability
                        , traces_inside_weights[traceIterator - traceManager->cbegin()]
                        , traces_outside_weights[traceIterator - traceManager->cbegin()]
                );

                const auto &inside_weights = traces_inside_weights[traceIterator - traceManager->cbegin()];
                const auto &outside_weights = traces_outside_weights[traceIterator - traceManager->cbegin()];

                const auto &root_inside_weight = inside_weights.at(trace->get_goal());
                const auto &root_outside_weight = outside_weights.at(trace->get_goal());

                Eigen::Tensor<double, 1> trace_root_probabilities = root_inside_weight * root_outside_weight;
                Eigen::Tensor<double, 0> trace_root_probability = trace_root_probabilities.sum();

                if (not std::isnan(trace_root_probability(0))
                    and not std::isinf(trace_root_probability(0))
                    and trace_root_probability(0) > 0) {
                    root_count += trace_root_probabilities;
                    corpus_likelihood += log(trace_root_probability(0));
                } else {
                    corpus_likelihood += minus_infinity;
                    continue;
                }

                if (debug)
                    std::cerr << "instance root probability: " << std::endl << trace_root_probabilities << std::endl;

                for (const Element<Node < Nonterminal>>
                    &node : *(trace->get_hypergraph())) {
                    const WeightVector &lhn_outside_weight = outside_weights.at(node);

                    if (debug) {
                        std::cerr << node << std::endl << "outside weight" << std::endl << lhn_outside_weight
                                  << std::endl;
                        std::cerr << "inside weight" << std::endl;
                        const WeightVector &lhn_inside_weight = inside_weights.at(node);
                        std::cerr << lhn_inside_weight << std::endl;
                    }
                    for (const auto &edge : trace->get_hypergraph()->get_incoming_edges(node)) {
                        const int rule_id = edge->get_label_id();
                        const size_t rule_dim = edge->get_sources().size() + 1;

                        switch (rule_dim) {
                            case 1:
                                compute_rule_count1(
                                        rule_tensors[rule_id]
                                        , lhn_outside_weight
                                        , trace_root_probability(0)
                                        , rule_count_tensors[rule_id]
                                );
                                break;
                            case 2:
                                compute_rule_count2(
                                        rule_tensors[rule_id]
                                        , edge
                                        , lhn_outside_weight
                                        , trace_root_probability(0)
                                        , inside_weights
                                        , rule_count_tensors[rule_id]
                                );
                                break;
                            case 3:
                                compute_rule_count3(
                                        rule_tensors[rule_id]
                                        , edge
                                        , lhn_outside_weight
                                        , trace_root_probability(0)
                                        , inside_weights
                                        , rule_count_tensors[rule_id]
                                );
                                break;
                            case 4:
                                compute_rule_count<4>(
                                        rule_tensors[rule_id]
                                        , edge
                                        , lhn_outside_weight
                                        , trace_root_probability(0)
                                        , inside_weights
                                        , rule_count_tensors[rule_id]
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

        inline void compute_rule_count1(const RuleTensor<double> & rule_weight_tensor, const RuleTensorRaw<double, 1> &lhn_outside_weight,
                                        const double trace_root_probability, RuleTensor<double> & rule_count_tensor
        ) {
            constexpr unsigned rule_rank {1};

            const auto & rule_weight = boost::get<RuleTensorRaw <double, rule_rank>>(rule_weight_tensor);

            auto rule_val = rule_weight * lhn_outside_weight;

            auto & rule_count = boost::get<RuleTensorRaw <double, rule_rank>>(rule_count_tensor);

            if (trace_root_probability > 0) {
                rule_count += rule_val.unaryExpr([trace_root_probability] (double x) {return x / trace_root_probability;});
            }
        }


        inline void compute_rule_count2(
                const RuleTensor<double> & rule_weight_tensor
                , const Element<HyperEdge<Nonterminal>>& edge
                , const RuleTensorRaw<double, 1>& lhn_outside_weight
                , const double trace_root_probability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector>& inside_weights
                , RuleTensor<double> & rule_count_tensor
        ) {
            constexpr unsigned rule_rank {2};

            const auto & rule_weight = boost::get<RuleTensorRaw <double, rule_rank>>(rule_weight_tensor);

            const auto & rhs_weight = inside_weights.at(edge->get_sources()[0]);
            auto rule_val = lhn_outside_weight.reshape(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1})
                                    .broadcast(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1)})
                            * rhs_weight.reshape(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1)})
                                    .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1}).eval()
                            * rule_weight
            ;

            auto & rule_count = boost::get<RuleTensorRaw <double, rule_rank>>(rule_count_tensor);

            if (trace_root_probability > 0) {
                rule_count += rule_val.unaryExpr([trace_root_probability] (double x) {return x / trace_root_probability;});
            }
        }


        inline void compute_rule_count3(
                const RuleTensor<double> & rule_weight_tensor
                , const Element<HyperEdge<Nonterminal>>& edge
                , const RuleTensorRaw <double, 1>& lhn_outside_weight
                , const double trace_root_probability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector>& inside_weights
                , RuleTensor<double> & rule_count_tensor
        ) {
            constexpr unsigned rule_rank {3};

            const auto & rule_weight = boost::get<RuleTensorRaw<double, rule_rank>>(rule_weight_tensor);
            const auto & rhs_weight1 = inside_weights.at(edge->get_sources()[0]);
            const auto & rhs_weight2 = inside_weights.at(edge->get_sources()[1]);

            auto rule_val = lhn_outside_weight.reshape(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1, 1})
                                    .broadcast(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1), rule_weight.dimension(2)})
                            * rhs_weight1.reshape(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1), 1})
                                    .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1, rule_weight.dimension(2)}).eval()
                            * rhs_weight2.reshape(Eigen::array<long, rule_rank>{1, 1, rule_weight.dimension(2)})
                                    .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), rule_weight.dimension(1), 1}).eval()
                            * rule_weight;
            ;

            auto & rule_count = boost::get<RuleTensorRaw<double, rule_rank>>(rule_count_tensor);

            if (trace_root_probability > 0) {
                rule_count += rule_val.unaryExpr([trace_root_probability] (double x) {return x / trace_root_probability;});
            }
        }

        template<int rule_dim>
        inline void compute_rule_count(
                const RuleTensor<double> & rule_weight_tensor
                , const Element<HyperEdge<Nonterminal>>& edge
                , const RuleTensorRaw <double, 1>& lhn_outside_weight
                , const double trace_root_probability
                , const MAPTYPE<Element<Node<Nonterminal>>, WeightVector>& inside_weights
                , RuleTensor<double> & rule_count_tensor
        ) {

            const auto & rule_weight = boost::get<RuleTensorRaw <double, rule_dim>>(rule_weight_tensor);
            const auto & rule_dimension = rule_weight.dimensions();

            Eigen::array<long, rule_dim> rshape_dim;
            Eigen::array<long, rule_dim> broad_dim;
            for (unsigned i = 0; i < rule_dim; ++i) {
                rshape_dim[i] = 1;
                broad_dim[i] = rule_dimension[i];
            }

            Eigen::Tensor<double, rule_dim> rule_val = rule_weight;
            for (unsigned i = 0; i < rule_dim; ++i) {
                const auto & item_weight = (i == 0)
                                           ? lhn_outside_weight
                                           : inside_weights.at(edge->get_sources()[i - 1]);
                rshape_dim[i] = broad_dim[i];
                broad_dim[i] = 1;
                rule_val *= item_weight.reshape(rshape_dim).broadcast(broad_dim);
                broad_dim[i] = rshape_dim[i];
                rshape_dim[i] = 1;
            }

            auto & rule_count = boost::get<RuleTensorRaw <double, rule_dim>>(rule_count_tensor);
            if (trace_root_probability > 0) {
                rule_count += rule_val.unaryExpr([trace_root_probability] (double x) {return x / trace_root_probability;});
            }
        }
    };

    template <typename Nonterminal>
    class SimpleMaximizer : public Maximizer {
        std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo;
        const bool debug;

    public:
        SimpleMaximizer(std::shared_ptr<const GrammarInfo2<Nonterminal>> grammarInfo, bool debug = false)
                : grammarInfo(grammarInfo)
                , debug(debug) {};
        void maximize(LatentAnnotation &latentAnnotation, const Counts &counts) {
            unsigned nont = 0;
            for (const std::vector<std::size_t> &group : grammarInfo->normalizationGroups) {
                const std::size_t lhs_dim = latentAnnotation.nonterminalSplits[nont];
                maximization(lhs_dim, counts, latentAnnotation, group);
                ++nont;
            }

            // maximize root weights:
            Eigen::Tensor<double, 0> corpus_prob_sum = counts.rootCounts.sum();
            if (corpus_prob_sum(0) > 0)
                latentAnnotation.rootWeights = latentAnnotation.rootWeights.unaryExpr(
                        [corpus_prob_sum](double x) {
                            return x / corpus_prob_sum(0);
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
            Eigen::Tensor<double, 1> lhs_counts(lhsSplits);
            lhs_counts.setZero();
            for (const size_t ruleId : group) {
                compute_normalization_divisor(lhs_counts, counts.ruleCounts[ruleId]);
            }

            for (const size_t ruleId : group) {
                normalize(latentAnnotation.ruleWeights[ruleId], counts.ruleCounts[ruleId], lhs_counts);
            }

        }
    };
}

#endif //STERMPARSER_EMTRAINERLA_H
