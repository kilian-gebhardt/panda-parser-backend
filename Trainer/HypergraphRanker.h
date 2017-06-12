//
// Created by kilian on 27/04/17.
//

#ifndef STERMPARSER_HYPERGRAPHRANKER_H
#define STERMPARSER_HYPERGRAPHRANKER_H

#include <unordered_map>
#include <queue>
#include "TraceManager.h"
#include "TrainingCommon.h"
#include "StorageManager.h"
#include "LatentAnnotation.h"

namespace Trainer {
    template<typename Nonterminal, typename TraceID>
    class HypergraphRanker {
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
        MAPTYPE<Element<Node<Nonterminal>>, WeightVector> traceInsideWeights;
    public:

        using MyPair = typename std::pair<size_t, double>;

        HypergraphRanker(
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

        std::vector<std::pair<size_t, double>> rank(const LatentAnnotation &latentAnnotation) {
            if (traceManager->cend() != traceManager->cbegin() + traceManager->size()) {
                std::cerr << "end - begin " << traceManager->cend() - traceManager->cbegin() << std::endl;
                std::cerr << "size: " << traceManager->size();
                std::abort();
            }
            return rank_la(latentAnnotation);
        }

        void clean_up() {
//            storageManager->free_weight_maps(traceInsideWeights);
        }

    private:
        std::vector<MyPair> rank_la(
                const LatentAnnotation &latentAnnotation
        ) {
            auto compareFunc = [](MyPair a, MyPair b) { return a.second < b.second; };
            typedef std::priority_queue<MyPair, std::vector<MyPair>, decltype(compareFunc)> TraceWeights;
            TraceWeights traceWeights(compareFunc);

            for (TraceIterator traceIterator = traceManager->cbegin();
                 traceIterator < traceManager->cend(); ++traceIterator) {
                const auto &trace = *traceIterator;
                if (trace->get_hypergraph()->size() == 0)
                    continue;

                // create inside weight for each node
                traceInsideWeights.clear();
                for (const auto &node : *(trace->get_hypergraph())) {
                    traceInsideWeights.emplace(
                            node
                            , storageManager->create_weight_vector<WeightVector>(latentAnnotation.nonterminalSplits[node->get_label_id()]));
                }

                MAPTYPE<Element<Node<Nonterminal>>, int> insideLogScales;

                trace->inside_weights_la(
                        *latentAnnotation.ruleWeights
                        , traceInsideWeights
                        , insideLogScales
                );

                Eigen::Tensor<double, 1> traceRootProbabilities{
                        compute_trace_root_probabilities(traceIterator, latentAnnotation)};
                Eigen::Tensor<double, 0> traceRootProbability = traceRootProbabilities.sum();

                traceWeights.emplace(traceIterator - traceManager->cbegin(), traceRootProbability(0));

                if (debug)
                    std::cerr << "instance root probability: " << std::endl << traceRootProbabilities << std::endl;
            }
            std::vector<MyPair> rankedItems;
            rankedItems.reserve(traceWeights.size());
            while (not traceWeights.empty()) {
                rankedItems.push_back(traceWeights.top());
                traceWeights.pop();
            }
            return rankedItems;
        }

    protected:
        Eigen::Tensor<double, 1> compute_trace_root_probabilities(TraceIterator traceIterator
                                                                  , const LatentAnnotation & latentAnnotation) {
            const auto &rootInsideWeight
                    = traceInsideWeights.at(traceIterator->get_goal());
            const auto &rootOutsideWeight = latentAnnotation.rootWeights;
            return Eigen::Tensor<double, 1> {rootOutsideWeight * rootInsideWeight};
        }
    };
}

#endif //STERMPARSER_HYPERGRAPHRANKER_H
