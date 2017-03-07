//
// Created by Markus on 22.02.17.
//

#ifndef STERMPARSER_TRACEMANAGER_H
#define STERMPARSER_TRACEMANAGER_H

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "Manager.h"
#include "Hypergraph.h"
#include "GrammarInfo.h"
#include "Names.h"
#include "TrainingCommon.h"


template <typename Nonterminal, typename oID>
class Trace {
private:
    Manage::ID id;
    ManagerPtr<Trace<Nonterminal, oID>> manager;
    oID originalID;
    HypergraphPtr<Nonterminal> hypergraph;
    Element<Node<Nonterminal>> goal;

    mutable std::vector<Element<Node<Nonterminal>>> topologicalOrder;

public:
    Trace(const Manage::ID aId
            , const ManagerPtr<Trace<Nonterminal, oID>> aManager
            , const oID oid
            , HypergraphPtr<Nonterminal> aHypergraph
            , Element<Node<Nonterminal>> aGoal
    )
            : id(aId)
            , manager(std::move(aManager))
            , originalID(std::move(oid))
            , hypergraph(std::move(aHypergraph))
            , goal(std::move(aGoal))
    { }


    const Element<Trace<Nonterminal, oID>> get_element() const noexcept {
        return Element<Trace<Nonterminal, oID>>(get_id(), manager);
    };

    Manage::ID get_id() const noexcept {return id; }

    const oID get_original_id() const noexcept {return originalID; }

    const HypergraphPtr<Nonterminal>& get_hypergraph() const noexcept {
        return hypergraph;
    }

    const Element<Node<Nonterminal>>& get_goal() const noexcept {
        return goal;
    }

    const std::vector<Element<Node<Nonterminal>>>& get_topological_order() const {
        if (topologicalOrder.size() == hypergraph->size())
            return topologicalOrder;

        std::vector<Element<Node<Nonterminal>>> topOrder{};
        topOrder.reserve(hypergraph->size());
        std::set<Element<Node<Nonterminal>>> visited{};
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
    std::pair<MAPTYPE<Element<Node<Nonterminal>>, Val>
            , MAPTYPE<Element<Node<Nonterminal>>, Val>>
    io_weights(std::vector<Val>& ruleWeights) const {

        // calculate inside weigths
        // TODO: implement for general case (== no topological order) approximation of inside weights
        MAPTYPE<Element<Node<Nonterminal>>, Val> inside{};
        for(const auto& node : get_topological_order()){
            inside[node] = Val::zero();
            for(const auto& incomingEdge : hypergraph->get_incoming_edges(node)){
                Val val(ruleWeights[incomingEdge->get_label_id()]);
                for(const auto& sourceNode : incomingEdge->get_sources())
                    val *= inside.at(sourceNode);

                inside[node] += val;
            }
        }

        // calculate outside weights
        MAPTYPE<Element<Node<Nonterminal>>, Val> outside{};
        for(auto nodeIterator = get_topological_order().crbegin(); nodeIterator != get_topological_order().crend(); ++nodeIterator){
            Val val = Val::zero();
            if(*nodeIterator == goal)
                val += Val::one();
            for(const auto outgoing : hypergraph->get_outgoing_edges(*nodeIterator)){
                Val valOutgoing = outside.at(outgoing.first->get_target());
                valOutgoing *= ruleWeights[outgoing.first->get_label_id()];
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



    inline void io_weights_la(
            const std::vector<Trainer::RuleTensor<double>> & rules
            , const Eigen::TensorRef<Eigen::Tensor<double, 1>> & root
            , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& insideWeights
            , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outsideWeights
    ) const {

        // TODO implement for general case (== no topological order) approximation of inside weights
        // computation of inside weights

        for (const auto& node : get_topological_order()) {
            Trainer::WeightVector& targetWeight = insideWeights.at(node);

            targetWeight.setZero();

            for (const auto& edge : get_hypergraph()->get_incoming_edges(node)) {
                switch (edge->get_sources().size() + 1) {
                    case 1:
                        inside_weight_step1(targetWeight, edge, rules);
                        break;
                    case 2:
                        inside_weight_step2(insideWeights, targetWeight, edge, rules);
                        break;
                    case 3:
                        inside_weight_step3(insideWeights, targetWeight, edge, rules);
                        break;
                    case 4:
                        inside_weight_step<4>(insideWeights, targetWeight, edge, rules);
                        break;
                    default:
                        std::cerr<< "Rules with more than 3 RHS nonterminals are not implemented." << std::endl;
                        abort();
                }
            }

//                std::cerr << "inside weight " << node << std::endl;
//                std::cerr << targetWeight << std::endl;

        }

        // TODO implement for general case (== no topological order) solution by gauss jordan
        for (auto nodeIterator = get_topological_order().rbegin();  nodeIterator != get_topological_order().rend(); ++nodeIterator) {
            const Element<Node<Nonterminal>>& node = *nodeIterator;

            Trainer::WeightVector& outsideWeight = outsideWeights.at(node);

            if (node == get_goal())
                outsideWeight = root;
            else
                outsideWeight.setZero();


            for (const auto& outgoing : get_hypergraph()->get_outgoing_edges(node)) {
                switch (outgoing.first->get_sources().size() + 1) {
                    case 1:
                        // Cannot happen, since there is at least one source (namely 'node')
                        std::cerr << "The trace is inconsistent!";
                        abort();
                    case 2:
                        outside_weight_step2(rules, outsideWeights, outsideWeight, outgoing);
                        break;
                    case 3:
                        outside_weight_step3(rules, insideWeights, outsideWeights, outsideWeight, outgoing);
                        break;
                    case 4:
                        outside_weight_step<4>(rules, insideWeights, outsideWeights, outsideWeight, outgoing);
                        break;
                    default:
                        std::cerr<< "Rules with more than 3 RHS nonterminals are not implemented." << std::endl;
                        abort();
                }
            }
//            std::cerr << "outside weight " << node << std::endl << outsideWeight << std::endl;
        }
    }




    inline void inside_weight_step1(
            Trainer::WeightVector& targetWeight,
            const Element<HyperEdge<Nonterminal>>& edge,
            const std::vector<Trainer::RuleTensor<double>> &rules
    ) const {
        constexpr unsigned ruleRank {1};

//      std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const auto & ruleWeight  = boost::get<Trainer::RuleTensorRaw<double, ruleRank>>(rules[edge->get_label_id()]);

//        std::cerr << "rule tensor " << edge->get_label_id() << " address " << rules[edge->get_label_id()] << std::endl << ruleWeight << std::endl;
//        std::cerr << "target weight " << targetWeight << std::endl;
        targetWeight += ruleWeight;
    }


    inline void inside_weight_step2(
            const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& insideWeights
            , Trainer::WeightVector & targetWeight
            , const Element<HyperEdge<Nonterminal>>& edge
            , const std::vector<Trainer::RuleTensor<double>> &rules
    ) const {
        constexpr unsigned ruleRank {2};

//        std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const auto & ruleWeight = boost::get<Trainer::RuleTensorRaw <double, ruleRank>>(rules[edge->get_label_id()]);

//        std::cerr << "rule tensor " << edge->get_label_id() << " address " << rules[edge->get_label_id()] << std::endl << ruleWeight << std::endl;

        constexpr unsigned rhsPos = 1;
        const auto & itemWeight = insideWeights.at(edge->get_sources()[rhsPos-1]);
        auto tmpValue = ruleWeight.contract(itemWeight, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)});
        targetWeight += tmpValue;
    }

    inline void inside_weight_step3(
            const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& insideWeights
            , Trainer::WeightVector& targetWeight
            , const Element<HyperEdge<Nonterminal>>& edge
            , const std::vector<Trainer::RuleTensor<double>> &rules
    ) const {
        constexpr unsigned ruleRank{3};


//        std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const auto & ruleWeight = boost::get<Trainer::RuleTensorRaw<double, ruleRank>>(rules[edge->get_label_id()]);

//        std::cerr << "rule tensor " << edge->get_label_id() << " address " << rules[edge->get_label_id()] << std::endl << ruleWeight << std::endl;

        constexpr unsigned rhsPos1 = 1;
        const auto & rhsItemWeight1 = insideWeights.at(edge->get_sources()[rhsPos1-1]);

        constexpr unsigned rhsPos2 = 2;
        const auto & rhsItemWeight2 = insideWeights.at(edge->get_sources()[rhsPos2-1]);

        auto tmpValue1 = ruleWeight.contract(rhsItemWeight2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(2, 0)});
        auto tmpValue2 = tmpValue1.contract(rhsItemWeight1, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)});
        targetWeight += tmpValue2;
    }


    template<long ruleRank>
    void inside_weight_step(
            const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& insideWeights
            , Trainer::WeightVector& targetWeight
            , const Element<HyperEdge<Nonterminal>>& edge
            , const std::vector<Trainer::RuleTensor<double>> &rules
    ) const {

//        std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const auto & ruleWeight = boost::get<Trainer::RuleTensorRaw <double, ruleRank>>(rules[edge->get_label_id()]);
        const auto & ruleDimension = ruleWeight.dimensions();

//        std::cerr << "ruleWeight tensor " << edge->get_label_id() << " address " << rules[edge->get_label_id()] << std::endl << ruleWeight << std::endl;


        Eigen::Tensor<double, ruleRank> tmpValue = ruleWeight;

        Eigen::array<long, ruleRank> reshapeDimension;
        Eigen::array<long, ruleRank> broadcastDimension;
        for (unsigned i = 0; i < ruleRank; ++i) {
            reshapeDimension[i] = 1;
            broadcastDimension[i] = ruleDimension[i];
        }

        for (unsigned rhsPos = 1; rhsPos < ruleRank; ++rhsPos) {
            const auto & item_weight = insideWeights.at(edge->get_sources()[rhsPos-1]);
            reshapeDimension[rhsPos] = broadcastDimension[rhsPos];
            broadcastDimension[rhsPos] = 1;
            tmpValue *= item_weight.reshape(reshapeDimension).broadcast(broadcastDimension);
            broadcastDimension[rhsPos] = reshapeDimension[rhsPos];
            reshapeDimension[rhsPos] = 1;
        }

        targetWeight += tmpValue.sum(Eigen::array<long, 1>{1});
    }


    inline void outside_weight_step2(
            const std::vector<Trainer::RuleTensor<double>>&rules
            , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outsideWeights
            , Trainer::WeightVector & outsideWeight
            , const std::pair<Element<HyperEdge<Nonterminal>>, unsigned int>& outgoing
    ) const {

        const auto& parent = outgoing.first->get_target();
        constexpr unsigned ruleRank {2};

        const auto & ruleWeight = boost::get<Trainer::RuleTensorRaw <double, ruleRank>>(rules[outgoing.first->get_label_id()]);
        const auto & parentWeight = outsideWeights.at(parent);

        auto outsideWeightSummand = ruleWeight.contract(parentWeight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)});

        outsideWeight += outsideWeightSummand;
    }


    inline void outside_weight_step3(
            const std::vector<Trainer::RuleTensor<double>> &rules
            , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& insideWeights
            , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outsideWeights
            , Trainer::WeightVector & outsideWeight
            , const std::pair<Element<HyperEdge<Nonterminal>>, unsigned int>& outgoing
    ) const {
        const auto&siblings = outgoing.first->get_sources();
        const auto &parent = outgoing.first->get_target();
        const unsigned position = outgoing.second;
        constexpr unsigned ruleRank {3};


        const auto & ruleWeight = boost::get<Trainer::RuleTensorRaw<double, ruleRank>>(rules[outgoing.first->get_label_id()]);
        const auto & parentWeight = outsideWeights.at(parent);
        const auto & rhsWeight = insideWeights.at(siblings[position == 0 ? 1 : 0]);

        auto tmpValue1 = ruleWeight.contract(rhsWeight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(position == 0 ? 2 : 1, 0)});
        auto tmpValue2 = tmpValue1.contract(parentWeight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)});

        outsideWeight += tmpValue2;
    }


    template<int ruleRank>
    void outside_weight_step(
            const std::vector<Trainer::RuleTensor<double>> &rules
            , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& insideWeights
            , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outsideWeights
            , Trainer::WeightVector & outsideWeight
            , const std::pair<Element<HyperEdge<Nonterminal>>, unsigned int>& outgoing
    ) const {
        const auto &siblings = outgoing.first->get_sources();
        const auto &parent = outgoing.first->get_target();
        const unsigned position = outgoing.second;

        const auto & ruleWeight = boost::get<Trainer::RuleTensorRaw <double, ruleRank>>(rules[outgoing.first->get_label_id()]);
        const Eigen::array<long, ruleRank> & ruleDimension = ruleWeight.dimensions();
        Eigen::array<long, ruleRank> reshapeDimension;
        Eigen::array<long, ruleRank> broadcastDimension;

        for (unsigned idx = 0; idx < ruleRank; ++idx) {
            reshapeDimension[idx] = 1;
            broadcastDimension[idx] = ruleDimension[idx];
        }

        const auto & parent_weight = outsideWeights.at(parent);

        Eigen::Tensor<double, ruleRank> tmpValue = ruleWeight;
//        std::cerr << "init tmpValue" << std::endl<< tmpValue << std::endl << std::endl;

        reshapeDimension[0] = broadcastDimension[0];
        broadcastDimension[0] = 1;
        tmpValue *= parent_weight.reshape(reshapeDimension).broadcast(broadcastDimension);
//        std::cerr << "tmpValue" << 0 << std::endl<< tmpValue << std::endl << std::endl;
        broadcastDimension[0] = ruleDimension[0];
        reshapeDimension[0] = 1;

        for (unsigned rhsPos = 1; rhsPos < ruleRank; ++rhsPos) {
            if (rhsPos == position + 1)
                continue;

            const auto & item_weight = insideWeights.at(siblings[rhsPos - 1]);
//            std::cerr << "inside weight " << rhsPos << std::endl<< item_weight << std::endl << std::endl;
            reshapeDimension[rhsPos] = broadcastDimension[rhsPos];
            broadcastDimension[rhsPos] = 1;
            tmpValue *= item_weight.reshape(reshapeDimension).broadcast(broadcastDimension);
//            std::cerr << "int tmpValue" << rhsPos << std::endl<< tmpValue << std::endl << std::endl;
            broadcastDimension[rhsPos] = reshapeDimension[rhsPos];
            reshapeDimension[rhsPos] = 1;
        }

        Eigen::array<long, ruleRank - 1> sum_array;
        for (unsigned idx = 0; idx < ruleRank - 1; ++idx) {
            if (idx < position + 1)
                sum_array[idx] = idx;
            if (idx >= position + 1)
                sum_array[idx] = idx + 1;
        }
        Eigen::Tensor<double, 1> outsideWeightSummand = tmpValue.sum(sum_array);

//        std::cerr << "final tmpValue" << std::endl<< tmpValue << std::endl << std::endl;
//        std::cerr << "outside weight summand" << std::endl << outsideWeightSummand << std::endl << std::endl;

        outsideWeight += outsideWeightSummand;
    }






};




template <typename Nonterminal, typename TraceID>
class TraceManager2 : public Manager<Trace<Nonterminal, TraceID>> {
    using TraceIterator = ConstManagerIterator<Trace<Nonterminal, TraceID>>;

private:
    const bool debug;

    MAPTYPE<Nonterminal, unsigned int> noOfItemsPerNonterminal;


public:

    TraceManager2(const bool debug = false) : debug(debug) {}

    // todo: Hook on create?



    // Functions for Cython interface

    GrammarInfo2* grammar_info_id(const std::vector<std::vector<size_t>> &rule_to_nonterminals) {
        return new GrammarInfo2(rule_to_nonterminals, this->cbegin()->get_goal()->get_label_id());
    }

    std::vector<std::vector<double>> lift_doubles(const std::vector<double>& rule_weights) const {
        std::vector<std::vector< double >> rule_weights_la;
        for (const double &rule_weight : rule_weights) {
            rule_weights_la.emplace_back(1, rule_weight);
        }
        return rule_weights_la;
    }


    // ##############################
    // ##############################
    // ##############################


    template<typename Val>
    std::vector<std::vector<Val>> init_weights_la(const std::vector<double> &rule_weights) const {
        std::vector<std::vector< Val >> rule_weights_la;
        for (const double& rule_weight : rule_weights) {
            rule_weights_la.emplace_back(1, Val::to(rule_weight));
        }
        return rule_weights_la;
    }



    template<typename Val>
    unsigned long total_rule_sizes(const std::vector<std::vector<Val>> & rule_weights) const {
        unsigned long counter = 0;
        for(const auto & rule_weight : rule_weights) {
            counter += rule_weight.size();
        }
        return counter;
    }

    template<typename NontIdx>
    unsigned max_item_size(const std::vector<unsigned> & nont_dimensions, const NontIdx nont_idx) {
        if(noOfItemsPerNonterminal.size() == 0)
            calculate_no_of_items_per_nonterminal();
        unsigned counter = 0;
        for (const auto & pair : noOfItemsPerNonterminal) {
            counter += pair.second * nont_dimensions.at(nont_idx(pair.first));
        }
        return counter;
    }

    void calculate_no_of_items_per_nonterminal(){
        noOfItemsPerNonterminal.clear();
        for(const auto& trace : *this){
            for(const auto& node : *(trace->get_hypergraph())){
                ++noOfItemsPerNonterminal[node->get_label()];
            }
        }
    }


};
template <typename Nonterminal, typename TraceID>
using TraceManagerPtr = std::shared_ptr<TraceManager2<Nonterminal, TraceID>>;








#endif //STERMPARSER_TRACEMANAGER_H
