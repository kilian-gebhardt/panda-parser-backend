//
// Created by kilian on 03/03/17.
//

#ifndef STERMPARSER_EMTRAINER_H
#define STERMPARSER_EMTRAINER_H
#include <vector>
#include <iostream>
#include "../Names.h"
#include "TraceManager.h"

namespace Trainer {

    template <typename Nonterminal, typename TraceID>
    class EMTrainer {
        TraceManagerPtr<Nonterminal, TraceID> traceManager;;

    public:
        template<typename Val>
        std::vector<double> do_em_training(
                const std::vector<double> &initialWeights
                , const std::vector <std::vector<unsigned>> &normalizationGroups
                , const unsigned noEpochs
        ) {

            std::vector <Val> ruleWeights;
            std::vector <Val> ruleCounts;

            unsigned epoch = 0;

            std::cerr << "Epoch " << epoch << "/" << noEpochs << ": ";

            // potential conversion to log semiring:
            for (auto i : initialWeights) {
                ruleWeights.push_back(Val::to(i));
            }
            std::cerr << std::endl;

            while (epoch < noEpochs) {
                // expectation
                ruleCounts = std::vector<Val>(ruleWeights.size(), Val::zero());
                for (const auto trace : *traceManager) {
                    const auto trIOweights = trace->io_weights(ruleWeights);

//                for (const auto &item : trace->get_topological_order()) {
//                    std::cerr << "T: " << item << " " << trIOweights.first.at(item) << " "
//                              << trIOweights.second.at(item) << std::endl;
//                }
//                std::cerr << std::endl;


                    const Val rootInsideWeight = trIOweights.first.at(trace->get_goal());
                    for (const auto &node : *(trace->get_hypergraph())) {
                        const Val lhnOutsideWeight = trIOweights.second.at(node);
//                    if(node->get_incoming().size() > 1)
//                        std::cerr << "Size is greater ";
                        for (const auto &edge : trace->get_hypergraph()->get_incoming_edges(node)) {
                            Val val = lhnOutsideWeight * ruleWeights[edge->get_label_id()] / rootInsideWeight;
                            for (const auto &sourceNode : edge->get_sources()) {
                                val = val * trIOweights.first.at(sourceNode);
                            }
                            if (not val.isNaN())
                                ruleCounts[edge->get_label_id()] += val;
                        }
                    }
                }

                // maximization
                for (auto group : normalizationGroups) {
                    Val groupCount = Val::zero();
                    for (auto member : group) {
                        groupCount = groupCount + ruleCounts[member];
                    }
                    if (not (groupCount == Val::zero() or groupCount.isNaN())) {
                        for (auto member : group) {
                            ruleWeights[member] = ruleCounts[member] / groupCount;
                        }
                    }
                }
                ++epoch;
                std::cerr << "Epoch " << epoch << "/" << noEpochs << ": ";
//                for (unsigned i = 0; i < ruleWeights.size(); ++i) {
//                    std::cerr << ruleWeights[i] << " ";
//                }
//            std::cerr << std::endl;
            }

            std::vector<double> result;

            // conversion from log semiring:
            for (auto i = ruleWeights.begin(); i != ruleWeights.end(); ++i) {
                result.push_back(i->from());
            }


            return result;
        }

        EMTrainer(TraceManagerPtr<Nonterminal, TraceID> & traceManager) : traceManager(traceManager) {}
    };

}


#endif //STERMPARSER_EMTRAINER_H
