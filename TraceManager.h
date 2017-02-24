//
// Created by Markus on 22.02.17.
//

#ifndef STERMPARSER_TRACEMANAGER_H
#define STERMPARSER_TRACEMANAGER_H

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "Manager.h"
#include "Hypergraph.h"


// use partly specialized Hypergraph objects
using NodeOriginalID = unsigned long;
using HyperedgeOriginalID = unsigned;
using Node = Manage::Node<NodeOriginalID>;

template <typename oID> using Info = Manage::Info<oID>;
template <typename InfoT> using Manager = Manage::Manager<InfoT>;
template <typename InfoT> using ManagerPtr = Manage::ManagerPtr<InfoT>;
template <typename InfoT> using Element = Manage::Element<InfoT>;


template <typename T1, typename T2> using MAPTYPE = typename std::unordered_map<T1, T2>;

using WeightVector = Eigen::TensorMap<Eigen::Tensor<double, 1>>;


template <typename Nonterminal>
class TraceNode : public Info<NodeOriginalID> {
private:
    ManagerPtr<TraceNode<Nonterminal>> manager;
    Nonterminal nonterminal;
public:
    TraceNode(Manage::ID aId
            , const ManagerPtr<TraceNode<Nonterminal>> aManager
            , const NodeOriginalID anOriginalID
            , const Nonterminal& nont)
            : Info(aId, anOriginalID)
            , manager(aManager)
            , nonterminal(nont) {}

    const Element<TraceNode<Nonterminal>> get_element() const noexcept {
        return Element<TraceNode<Nonterminal>>(Info::get_id(), manager);
    }

    const Nonterminal get_nonterminal() const noexcept { return nonterminal; }
};



template<typename Nonterminal> using Hypergraph = Manage::Hypergraph<TraceNode<Nonterminal>, HyperedgeOriginalID>;
template<typename Nonterminal> using HypergraphPtr = std::shared_ptr<Hypergraph<Nonterminal>>;
template<typename Nonterminal> using HyperEdge = Manage::HyperEdge<TraceNode<Nonterminal>, HyperedgeOriginalID>;




template <typename Nonterminal, typename oID>
class TraceInfo : public Info<oID> {
private:
    ManagerPtr<TraceInfo<Nonterminal, oID>> manager;
    HypergraphPtr<Nonterminal> hypergraph;

    std::vector<Element<TraceNode<Nonterminal>>> topologicalOrder;
    Element<TraceNode<Nonterminal>> goal;

public:
    TraceInfo(const Manage::ID aId
            , const ManagerPtr<TraceInfo<Nonterminal, oID>> aManager
            , const oID oid
            , HypergraphPtr<Nonterminal> aHypergraph
            , Element<TraceNode<Nonterminal>> aGoal)
            : Info<oID>(std::move(aId), std::move(oid))
            , manager(std::move(aManager))
            , hypergraph(std::move(aHypergraph))
            , goal(std::move(aGoal)){ }

    const Element<TraceInfo<Nonterminal, oID>> get_element() const noexcept {
        return Element<TraceInfo<Nonterminal, oID>>(Info<oID>::get_id(), manager);
    };

    const HypergraphPtr<Nonterminal>& get_hypergraph() const noexcept {
        return hypergraph;
    }

    const Element<TraceNode<Nonterminal>>& get_goal() const noexcept {
        return goal;
    }

    const std::vector<Element<TraceNode<Nonterminal>>>& get_topological_order(){
        if (topologicalOrder.size() == hypergraph->size())
            return topologicalOrder;

        std::vector<Element<TraceNode<Nonterminal>>> topOrder{};
        topOrder.reserve(hypergraph->size());
        std::set<Element<TraceNode<Nonterminal>>> visited{};
        bool changed = true;
        while (changed) {
            changed = false;

            // add item, if all its decendants were added
            for (const auto& node : *hypergraph) {
                if (visited.find(node) != visited.cend())
                    continue;
                bool violation = false;
                for (const auto& edge : hypergraph->get_incoming_edges(node)) {
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
    std::pair<MAPTYPE<Element<TraceNode<Nonterminal>>, Val>
            , MAPTYPE<Element<TraceNode<Nonterminal>>, Val>>
    io_weights(std::vector<Val>& ruleWeights){

        // calculate inside weigths
        // TODO: implement for general case (== no topological order) approximation of inside weights
        MAPTYPE<Element<TraceNode<Nonterminal>>, Val> inside{};
        for(const auto& node : get_topological_order()){
            inside[node] = Val::zero();
            for(const auto& incomingEdge : hypergraph->get_incoming_edges(node)){
                Val val(ruleWeights[incomingEdge->get_original_id()]);
                for(const auto& sourceNode : incomingEdge->get_sources())
                    val *= inside.at(sourceNode);

                inside[node] += val;
            }
        }

        // calculate outside weights
        MAPTYPE<Element<TraceNode<Nonterminal>>, Val> outside{};
        for(auto nodeIterator = get_topological_order().crbegin(); nodeIterator != get_topological_order().crend(); ++nodeIterator){
            Val val = Val::zero();
            if(*nodeIterator == goal)
                val += Val::one();
            for(const auto outgoing : hypergraph->get_outgoing_edges(*nodeIterator)){
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




template <typename Nonterminal, typename Terminal, typename TraceID>
class TraceManager2 : public Manager<TraceInfo<Nonterminal, TraceID>> {
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
                    for (const auto& edge : trace->get_hypergraph()->get_incoming_edges(node)) {
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
template <typename Nonterminal, typename Terminal, typename TraceID>
using TraceManagerPtr = std::shared_ptr<TraceManager2<Nonterminal, Terminal, TraceID>>;





#endif //STERMPARSER_TRACEMANAGER_H
