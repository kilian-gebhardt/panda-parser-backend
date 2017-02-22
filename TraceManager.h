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

using WeightVector = Eigen::TensorMap<Eigen::Tensor<double, 1>>;

template <typename oID>
class TraceInfo : Info<oID> {
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

    HypergraphPtr<oID>& get_hypergraph() const noexcept {
        return hypergraph;
    }

    const Element<Node, oID>& get_goal() const noexcept {
        return goal;
    }
};




template <typename Nonterminal, typename Terminal>
class TraceManager2 : public Manager<TraceInfo, unsigned long> {
private:

public:

};
template <typename Nonterminal, typename Terminal>
using TraceManagerPtr = std::shared_ptr<TraceManager2<Nonterminal, Terminal>>;





#endif //STERMPARSER_TRACEMANAGER_H
