//
// Created by Markus on 22.02.17.
//

#ifndef STERMPARSER_TRACEMANAGER_H
#define STERMPARSER_TRACEMANAGER_H

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "Manager.h"
#include "Hypergraph.h"


//// use partly specialized Hypergraph objects
//using oID = unsigned long;
////using Info = Manage::Info<oID>;
//template <typename oIDT> using Info = Manage::Info<oIDT>;
//template <template <typename oIDT> typename InfoT> using Manager = Manage::Manager<InfoT, oID>;
//template <template <typename oIDT> typename InfoT> using ManagerPtr = Manage::ManagerPtr<InfoT, oID>;
//using Hypergraph = Manage::Hypergraph<oID>;
//using HypergraphPtr = Manage::HypergraphPtr<oID>;
//template <typename oIDT> using Node = Manage::Node<oIDT>;
//template <typename oIDT> using HyperEdge = Manage::HyperEdge<oIDT>;
//template <template <typename oIDT> typename InfoT> using Element = Manage::Element<InfoT, oID>;

using namespace Manage;
template <typename T1, typename T2> using MAPTYPE = typename std::unordered_map<T1, T2>;

using WeightVector = Eigen::TensorMap<Eigen::Tensor<double, 1>>;



template <typename oID>
class TraceInfo : public Info<oID> {
private:
    ManagerPtr<TraceInfo, oID> manager;
    HypergraphPtr<oID> hypergraph;

    std::vector<Element<Node, oID>> topologicalOrder;
    std::vector<Element<Node, oID>> reverseTopologicalOrder;
    Element<Node, oID> goal;

public:
    TraceInfo(const ID aId
            , const oID oid
            , const ManagerPtr<TraceInfo, oID> aManager
            , HypergraphPtr<oID> aHypergraph
            , Element<Node, oID> aGoal)
            : Info<oID>(std::move(aId), std::move(oid))
            , manager(std::move(aManager))
            , hypergraph(std::move(aHypergraph))
            , goal(std::move(aGoal)){ }

    const Element<TraceInfo, oID> get_element() const noexcept {
        return Element<TraceInfo, oID>(Info<oID>::get_id(), manager);
    };

    const HypergraphPtr<oID>& get_hypergraph() const noexcept {
        return hypergraph;
    }

    const Element<Node, oID>& get_goal() const noexcept {
        return goal;
    }

    const std::vector<Element<Node, oID>>& get_topological_order(){
        if (topologicalOrder.size() == hypergraph->size())
            return topologicalOrder;

        std::vector<Element<Node, oID>> topOrder{};
        topOrder.reserve(hypergraph->size());
        std::set<Element<Node, oID>> visited{};
        bool changed = true;
        while (changed) {
            changed = false;

            // add item, if all its decendants were added
            for (const auto& node : *hypergraph) {
                if (visited.find(node) != visited.cend())
                    continue;
                bool violation = false;
                for (const auto& edge : node->get_incoming()) {
                    for (const auto sourceNode : edge->get_sources()) {
                        if (visited.find(sourceNode) == visited.cend()) {
                            violation = true;
                            break;
                        }
                    }
                    if (violation)
                        break;
                }
                if (!violation) {
                    changed = true;
                    visited.insert(node);
                    topOrder.push_back(node);
                }
            }
        }
        topologicalOrder = topOrder;
        return topologicalOrder;

    };

    template <typename Val>
    std::pair<MAPTYPE<Element<Node, oID>, Val>
            , MAPTYPE<Element<Node, oID>, Val>>
    io_weights(std::vector<Val>& ruleWeights){

        // calculate inside weigths
        // TODO: implement for general case (== no topological order) approximation of inside weights
        MAPTYPE<Element<Node, oID>, Val> inside{};
        for(const auto& node : get_topological_order()){
            inside[node] = Val::zero();
            for(const auto& incomingEdge : node->get_incoming()){
                Val val(ruleWeights[incomingEdge->get_original_id()]);
                for(const auto& sourceNode : incomingEdge->get_sources())
                    val *= inside.at(sourceNode);

                inside[node] += val;
            }
        }

        // calculate outside weights
        MAPTYPE<Element<Node, oID>, Val> outside{};
        for(auto nodeIterator = get_topological_order().crbegin(); nodeIterator != get_topological_order().crend(); ++nodeIterator){
            Val val = Val::zero();
            if(*nodeIterator == goal)
                val += Val::one();
            for(const auto outgoing : (*nodeIterator)->get_outgoing()){
                Val valOutgoing = outside.at(outgoing.first->get_target());
                valOutgoing *= ruleWeights[outgoing.first->get_original_id()];
                const auto& incomingNodes(outgoing.first->get_sources());
                for(unsigned int pos = 0; pos < incomingNodes.size(); ++pos){
                    if(pos != outgoing.second)
                        valOutgoing *= inside.at(incomingNodes[pos]);
                }
                val += valOutgoing;
            }
            outside[*nodeIterator] = val;
        }

        return std::make_pair(inside, outside);
    }




};




template <typename Nonterminal, typename Terminal>
class TraceManager2 : public Manager<TraceInfo, unsigned long> {
private:

public:

    template<typename Val>
    std::vector<double> do_em_training( const std::vector<double> & initialWeights
            , const std::vector<std::vector<unsigned>> & normalizationGroups
            , const unsigned noEpochs){

        std::vector<Val> ruleWeights;
        std::vector<Val> ruleCounts;

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
            for (const auto trace : *this) {

                // todo: do I need this test?
                if(trace->get_hypergraph()->size() == 0)
                    continue;

                const auto trIOweights = trace->io_weights(ruleWeights);

//                for (const auto &item : trace->get_topological_order()) {
//                    std::cerr << "T: " << item << " " << trIOweights.first.at(item) << " "
//                              << trIOweights.second.at(item) << std::endl;
//                }
//                std::cerr << std::endl;


                const Val rootInsideWeight = trIOweights.first.at(trace->get_goal());
                for (const auto & node : *(trace->get_hypergraph())) {
                    const Val lhnOutsideWeight = trIOweights.second.at(node);
//                    if(node->get_incoming().size() > 1)
//                        std::cerr << "Size is greater ";
                    for (const auto& edge : node->get_incoming()) {
                        Val val = lhnOutsideWeight * ruleWeights[edge->get_original_id()] / rootInsideWeight;
                        for (const auto& sourceNode : edge->get_sources()) {
                            val = val * trIOweights.first.at(sourceNode);
                        }
                        ruleCounts[edge->get_original_id()] += val;
                    }
                }
            }

            // maximization
            for (auto group : normalizationGroups) {
                Val groupCount = Val::zero();
                for (auto member : group) {
                    groupCount = groupCount + ruleCounts[member];
                }
                if (groupCount != Val::zero()) {
                    for (auto member : group) {
                        ruleWeights[member] = ruleCounts[member] / groupCount;
                    }
                }
            }
            epoch++;
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






};
template <typename Nonterminal, typename Terminal>
using TraceManagerPtr = std::shared_ptr<TraceManager2<Nonterminal, Terminal>>;





#endif //STERMPARSER_TRACEMANAGER_H
