//
// Created by kilian on 02/05/17.
//

#ifndef STERMPARSER_VALIDATION_H
#define STERMPARSER_VALIDATION_H

#include <iostream>
#include "LatentAnnotation.h"
#include "TraceManager.h"
#include "../util.h"
#include "HypergraphRanker.h"

namespace Trainer {

    class ValidationLA {
    public:
        virtual double validation_score(const LatentAnnotation &latentAnnotation) = 0;

        virtual double minimum_score() = 0;

        virtual std::string quantity() const = 0;

        virtual void clean_up() {};
    };

    class ValidationLikelihoodLA : public ValidationLA {
    public:
        virtual double validation_score(const LatentAnnotation &latentAnnotation) {
            return log_likelihood(latentAnnotation);
        }

        virtual double log_likelihood(const LatentAnnotation &latentAnnotation) = 0;

        virtual std::string quantity() const {
            return "likelihood";
        }

        virtual double minimum_score() {
            return minus_infinity;
        }
    };


    template<typename Nonterminal, typename TraceID>
    class SimpleLikelihoodLA : public ValidationLikelihoodLA {
    protected:
        template<typename T1, typename T2>
        using MAPTYPE = typename std::unordered_map<T1, T2>;
        using TraceIterator = ConstManagerIterator<Trace<Nonterminal, TraceID>>;
        const TraceManagerPtr<Nonterminal, TraceID> traceManager;
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;
        const bool debug;
    private:
        const unsigned threads;
    protected:
        std::vector<MAPTYPE<Element<Node<Nonterminal>>, WeightVector>> tracesInsideWeights;

    public:
        SimpleLikelihoodLA(
                TraceManagerPtr<Nonterminal, TraceID> traceManager
                , std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::shared_ptr<StorageManager> storageManager
                , unsigned threads = 1
                , bool debug = false
        )
                : traceManager(traceManager), grammarInfo(grammarInfo), storageManager(storageManager),
                  threads(threads), debug(debug) {};

        virtual double log_likelihood(const LatentAnnotation &latentAnnotation) {
            if (traceManager->cend() - traceManager->cbegin() != traceManager->size()) {
                std::cerr << "end - begin " << traceManager->cend() - traceManager->cbegin() << std::endl;
                std::cerr << "size: " << traceManager->size();
                std::abort();
            }
            if (tracesInsideWeights.size() < traceManager->size()) {
                tracesInsideWeights.resize(traceManager->size());
            }
            return log_likelihood_la(latentAnnotation);
        }

        void clean_up() {
            storageManager->free_weight_maps(tracesInsideWeights);
        }

    private:
        inline double log_likelihood_la(
                const LatentAnnotation &latentAnnotation
        ) {
            double logLikelihood{0.0};
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
// #pragma omp declare reduction (+ : omp_out += omp_in ) initializer (omp_priv = omp_orig)
#pragma omp parallel for schedule(dynamic, 10) reduction (+:logLikelihood)
            for (TraceIterator traceIterator = traceManager->cbegin();
                 traceIterator < traceManager->cend(); ++traceIterator) {
                const auto &trace = *traceIterator;
                if (trace->get_hypergraph()->size() == 0)
                    continue;

                if (tracesInsideWeights.size() <= traceIterator - traceManager->cbegin()) {
                    std::cerr << "tried to access non-existent inside or outside weight map" << std::endl;
                    std::cerr << "it - begin " << traceIterator - traceManager->cbegin() << std::endl;
                    std::cerr << "in size: " << tracesInsideWeights.size() << std::endl;
                    abort();
                }
                // create inside weight for each node if necessary
                if (tracesInsideWeights[traceIterator - traceManager->cbegin()].size() !=
                    trace->get_hypergraph()->size()) {
                    tracesInsideWeights[traceIterator - traceManager->cbegin()].clear();
                    for (const auto &node : *(trace->get_hypergraph())) {
                        tracesInsideWeights[traceIterator - traceManager->cbegin()].emplace(
                                node
                                , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                    }
                }

                MAPTYPE<Element<Node<Nonterminal>>, int> insideLogScales;

                trace->inside_weights_la(
                        *latentAnnotation.ruleWeights
                        , tracesInsideWeights[traceIterator - traceManager->cbegin()]
                        , insideLogScales
                );

                Eigen::Tensor<double, 1> traceRootProbabilities{
                        compute_trace_root_probabilities(traceIterator, latentAnnotation)};
                Eigen::Tensor<double, 0> traceRootProbability = traceRootProbabilities.sum();

                if (not std::isnan(traceRootProbability(0))
                    and not std::isinf(traceRootProbability(0))
                    and traceRootProbability(0) > 0) {
                    logLikelihood += log(traceRootProbability(0));
                } else {
                    //std::cerr << "trace Root Probability " << traceRootProbability(0) << std::endl;
                    // sentences with 0 probability are simply ignored, cf.
                    // https://github.com/slavpetrov/berkeleyparser/blob/release-1.0/src/edu/berkeley/nlp/PCFGLA/GrammarTrainer.java?ts=2#L558

                    // alternatively one could add a positive constant to each probability
                    // to get a useful validation measure

                    // logLikelihood += minus_infinity;
                    continue;
                }

                if (debug)
                    std::cerr << "instance root probability: " << std::endl << traceRootProbabilities << std::endl;
            }

            return logLikelihood;
        }

    protected:
        Eigen::Tensor<double, 1> compute_trace_root_probabilities(
                TraceIterator traceIterator
                , const LatentAnnotation &latentAnnotation
        ) {
            const auto &rootInsideWeight
                    = tracesInsideWeights[traceIterator - traceManager->cbegin()].at(traceIterator->get_goal());
            const auto &rootOutsideWeight = latentAnnotation.rootWeights;
            return Eigen::Tensor<double, 1> {rootOutsideWeight * rootInsideWeight};
        }
    };

    template <typename Nonterminal, typename TraceID>
    class CandidateScoreValidator : public ValidationLA {
    private:
        std::shared_ptr<const GrammarInfo2> grammarInfo;
        std::shared_ptr<StorageManager> storageManager;
        const std::string _quantity;
        std::vector<HypergraphRanker<Nonterminal, TraceID>> traces;
        std::vector<std::vector<double>> scores;
        std::vector<double> maxScores;
        double globalMaxScore {0};
        const double minimumScore;
    public:
        CandidateScoreValidator(
                std::shared_ptr<const GrammarInfo2> grammarInfo
                , std::shared_ptr<StorageManager> storageManager
                , std::string _quantity = "score"
                , double minimumScore = minus_infinity
        ) : grammarInfo(grammarInfo)
                , storageManager(storageManager)
                , _quantity(_quantity)
                , minimumScore(minimumScore) {};

        void add_scored_candidates(TraceManagerPtr<Nonterminal, TraceID> traces, std::vector<double> scores, double maxScore) {
            this->traces.emplace_back(traces, grammarInfo, storageManager);
            this->scores.push_back(scores);
            this->maxScores.push_back(maxScore);
            globalMaxScore += maxScore;
        }

        virtual double validation_score(const LatentAnnotation & latentAnnotation) {
            double globalScore = 0.0;
            for (size_t entry = 0; entry < scores.size(); ++entry) {
                auto ranking = traces[entry].rank(latentAnnotation);
                if (ranking.size() > 0) {
                    size_t selected {ranking[0].first};
                    globalScore += scores[entry][selected];
                }
            }

            if (globalMaxScore > 0.0)
                return globalScore / globalMaxScore;
            else {
                assert (globalScore == 0.0);
                return 0.0;
            }

        }

        virtual double minimum_score() {
            return minimumScore;
        }

        virtual std::string quantity() const {
            return _quantity;
        }
    };
}

#endif //STERMPARSER_VALIDATION_H
