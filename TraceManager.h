//
// Created by Markus on 22.02.17.
//

#ifndef STERMPARSER_TRACEMANAGER_H
#define STERMPARSER_TRACEMANAGER_H

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "Manager.h"
#include "Hypergraph.h"


using WeightVector = Eigen::TensorMap<Eigen::Tensor<double, 1>>;
template <typename Scalar>
using RuleTensor = typename boost::variant<
        Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 5>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 6>>
>;

// use partly specialized Hypergraph objects
using NodeOriginalID = unsigned long;
using HyperedgeOriginalID = unsigned;
using Node = Manage::Node<NodeOriginalID>;

template <typename oID> using Info = Manage::Info<oID>;
template <typename InfoT> using Manager = Manage::Manager<InfoT>;
template <typename InfoT> using ManagerPtr = Manage::ManagerPtr<InfoT>;
template <typename InfoT> using ConstManagerIterator = Manage::ManagerIterator<InfoT, true>;
template <typename InfoT> using Element = Manage::Element<InfoT>;



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



    inline void io_weights_la(
            const std::vector<RuleTensor<double>> & rules
            , const WeightVector & root
            , MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& outside_weights
    ){

        // TODO implement for general case (== no topological order) approximation of inside weights
        // computation of inside weights

        for (const auto& node : get_topological_order()) {
            WeightVector& target_weight = inside_weights.at(node);

            target_weight.setZero();

            // witness is an incoming edge into item
            //    - first: edge label
            //    - second: list of source nodes
            for (const auto& edge : get_hypergraph()->get_incoming_edges(node)) {
                switch (edge->get_sources().size() + 1) {
                    case 1:
                        inside_weight_step1(target_weight, edge, rules);
                        break;
                    case 2:
                        inside_weight_step2(inside_weights, target_weight, edge, rules);
                        break;
                    case 3:
                        inside_weight_step3(inside_weights, target_weight, edge, rules);
                        break;
                    case 4:
                        inside_weight_step<4>(inside_weights, target_weight, edge, rules);
                        break;
                    default:
                        std::cerr<< "Rules with more than 3 RHS nonterminals are not implemented." << std::endl;
                        abort();
                }
            }

//                std::cerr << "inside weight " << node << std::endl;
//                std::cerr << target_weight << std::endl;

        }

        // TODO implement for general case (== no topological order) solution by gauss jordan
        for (auto node_iterator = get_topological_order().rbegin();  node_iterator != get_topological_order().rend(); ++node_iterator) {
            const Element<TraceNode<Nonterminal>>& node = *node_iterator;

            WeightVector& outside_weight = outside_weights.at(node);

            if (node == get_goal())
                outside_weight = root;
            else
                outside_weight.setZero();


            for (const auto& outgoing : get_hypergraph()->get_outgoing_edges(node)) {
                switch (outgoing.first->get_sources().size() + 1) {
                    case 1:
                        // Cannot happen, since there is at least one source (namely 'node')
                        std::cerr << "The trace is inconsistent!";
                        break;
                    case 2:
                        outside_weight_step2(rules, outside_weights, outside_weight, outgoing);
                        break;
                    case 3:
                        outside_weight_step3(rules, inside_weights, outside_weights, outside_weight, outgoing);
                        break;
                    case 4:
                        outside_weight_step<4>(rules, inside_weights, outside_weights, outside_weight, outgoing);
                        break;
                    default:
                        std::cerr<< "Rules with more than 3 RHS nonterminals are not implemented." << std::endl;
                        abort();
                }
            }
//            std::cerr << "outside weight " << node << std::endl << outside_weight << std::endl;
        }
    }




    inline void inside_weight_step1(
            WeightVector& target_weight,
            const Element<HyperEdge<Nonterminal>>& edge,
            const std::vector<RuleTensor<double>> &rules
    ) const {
        constexpr unsigned rule_rank {1};

//      std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[edge->get_original_id()]);

//        std::cerr << "rule tensor " << edge->get_original_id() << " address " << rules[edge->get_original_id()] << std::endl << rule_weight << std::endl;
        target_weight += rule_weight;
    }


    inline void inside_weight_step2(
            const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , WeightVector & target_weight
            , const Element<HyperEdge<Nonterminal>>& edge
            , const std::vector<RuleTensor<double>> &rules
    ) const {
        constexpr unsigned rule_rank {2};

//        std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[edge->get_original_id()]);

//        std::cerr << "rule tensor " << edge->get_original_id() << " address " << rules[edge->get_original_id()] << std::endl << rule_weight << std::endl;

        constexpr unsigned rhs_pos = 1;
        const WeightVector& item_weight = inside_weights.at(edge->get_sources()[rhs_pos-1]);
        auto c1 = rule_weight.contract(item_weight, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)});
        target_weight += c1;
    }

    template<typename Witness>
    inline void inside_weight_step3(
            const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , WeightVector& target_weight
            , const Element<HyperEdge<Nonterminal>>& edge
            , const std::vector<RuleTensor<double>> &rules
    ) const {
        constexpr unsigned rule_rank{3};


//        std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[edge->get_original_id()]);

//        std::cerr << "rule tensor " << edge->get_original_id() << " address " << rules[edge->get_original_id()] << std::endl << rule_weight << std::endl;

        constexpr unsigned rhs_pos1 = 1;
        const WeightVector & rhs_item_weight1 = inside_weights.at(edge->get_sources()[rhs_pos1-1]);

        constexpr unsigned rhs_pos2 = 2;
        const WeightVector & rhs_item_weight2 = inside_weights.at(edge->get_sources()[rhs_pos2-1]);

        auto c1 = rule_weight.contract(rhs_item_weight2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(2, 0)});
        auto c2 = c1.contract(rhs_item_weight1, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)});
        target_weight += c2;
    }


    template<int rule_rank>
    void inside_weight_step(
            const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , WeightVector& target_weight
            , const Element<HyperEdge<Nonterminal>>& edge
            , const std::vector<RuleTensor<double>> &rules
    ) const {

//        std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[edge->get_original_id()]);
        const auto & rule_dim = rule_weight.dimensions();

//        std::cerr << "rule tensor " << edge->get_original_id() << " address " << rules[edge->get_original_id()] << std::endl << rule_weight << std::endl;


        Eigen::Tensor<double, rule_rank> tmp_value = rule_weight;

        Eigen::array<long, rule_rank> rshape_dim;
        Eigen::array<long, rule_rank> broad_dim;
        for (unsigned i = 0; i < rule_rank; ++i) {
            rshape_dim[i] = 1;
            broad_dim[i] = rule_dim[i];
        }

        for (unsigned rhs_pos = 1; rhs_pos < rule_rank; ++rhs_pos) {
            const Eigen::TensorMap<Eigen::Tensor<double, 1>> & item_weight = inside_weights.at(edge->get_sources()[rhs_pos-1]);
            rshape_dim[rhs_pos] = broad_dim[rhs_pos];
            broad_dim[rhs_pos] = 1;
            tmp_value *= item_weight.reshape(rshape_dim).broadcast(broad_dim);
            broad_dim[rhs_pos] = rshape_dim[rhs_pos];
            rshape_dim[rhs_pos] = 1;
        }

        target_weight += tmp_value.sum(Eigen::array<long, 1>{1});
    }


    inline void outside_weight_step2(
            const std::vector<RuleTensor<double>>&rules
            , const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& outside_weights
            , WeightVector & outside_weight
            , const std::pair<Element<HyperEdge<Nonterminal>>, unsigned int>& outgoing
    ) const {

        const auto& parent = outgoing.first->get_target();
        constexpr unsigned rule_rank {2};

        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[outgoing.first->get_original_id()]);
        const Eigen::TensorMap<Eigen::Tensor<double, 1>> & parent_weight = outside_weights.at(parent);

        auto outside_weight_summand = rule_weight.contract(parent_weight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)});

        outside_weight += outside_weight_summand;
    }


    inline void outside_weight_step3(
            const std::vector<RuleTensor<double>> &rules
            , const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& outside_weights
            , WeightVector & outside_weight
            , const std::pair<Element<HyperEdge<Nonterminal>>, unsigned int>& outgoing
    ) const {
        const auto&siblings = outgoing.first->get_sources();
        const auto &parent = outgoing.first->get_target();
        const unsigned position = outgoing.second;
        constexpr unsigned rule_rank {3};


        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[outgoing.first->get_original_id()]);
        const Eigen::TensorMap<Eigen::Tensor<double, 1>> & parent_weight = outside_weights.at(parent);
        const Eigen::TensorMap<Eigen::Tensor<double, 1>> & rhs_weight = inside_weights.at(siblings[position == 0 ? 1 : 0]);

        auto c1 = rule_weight.contract(rhs_weight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(position == 0 ? 2 : 1, 0)});
        auto c2 = c1.contract(parent_weight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)});

        outside_weight += c2;
    }


    template<int rule_rank>
    void outside_weight_step(
            const std::vector<RuleTensor<double>> &rules
            , const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& outside_weights
            , WeightVector & outside_weight
            , const std::pair<Element<HyperEdge<Nonterminal>>, unsigned int>& outgoing
    ) const {
        const auto&siblings = outgoing.first->get_sources();
        const auto &parent = outgoing.first->get_target();
        const unsigned position = outgoing.second;

        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[outgoing.first->get_original_id()]);
        const Eigen::array<long, rule_rank> & rule_dim = rule_weight.dimensions();
        Eigen::array<long, rule_rank> rshape_dim;
        Eigen::array<long, rule_rank> broad_dim;

        for (unsigned i = 0; i < rule_rank; ++i) {
            rshape_dim[i] = 1;
            broad_dim[i] = rule_dim[i];
        }

        const auto & parent_weight = outside_weights.at(parent);

        Eigen::Tensor<double, rule_rank> rule_val = rule_weight;
//        std::cerr << "init rule_val" << std::endl<< rule_val << std::endl << std::endl;

        rshape_dim[0] = broad_dim[0];
        broad_dim[0] = 1;
        rule_val *= parent_weight.reshape(rshape_dim).broadcast(broad_dim);
//        std::cerr << "rule_val" << 0 << std::endl<< rule_val << std::endl << std::endl;
        broad_dim[0] = rule_dim[0];
        rshape_dim[0] = 1;

        for (unsigned rhs_pos = 1; rhs_pos < rule_rank; ++rhs_pos) {
            if (rhs_pos == position + 1)
                continue;

            const Eigen::TensorMap<Eigen::Tensor<double, 1>> & item_weight = inside_weights.at(siblings[rhs_pos - 1]);
//            std::cerr << "inside weight " << rhs_pos << std::endl<< item_weight << std::endl << std::endl;
            rshape_dim[rhs_pos] = broad_dim[rhs_pos];
            broad_dim[rhs_pos] = 1;
            rule_val *= item_weight.reshape(rshape_dim).broadcast(broad_dim);
//            std::cerr << "int rule_val" << rhs_pos << std::endl<< rule_val << std::endl << std::endl;
            broad_dim[rhs_pos] = rshape_dim[rhs_pos];
            rshape_dim[rhs_pos] = 1;
        }

        Eigen::array<long, rule_rank - 1> sum_array;
        for (unsigned i = 0; i < rule_rank - 1; ++i) {
            if (i < position + 1)
                sum_array[i] = i;
            if (i >= position + 1)
                sum_array[i] = i + 1;
        }
        Eigen::Tensor<double, 1> outside_weight_summand = rule_val.sum(sum_array);

//        std::cerr << "final rule_val" << std::endl<< rule_val << std::endl << std::endl;
//        std::cerr << "outside weight summand" << std::endl << outside_weight_summand << std::endl << std::endl;

        outside_weight += outside_weight_summand;
    }










};




template <typename Nonterminal>
class GrammarInfo2 {
public:
    typename std::function<unsigned(Nonterminal)> nont_idx;
    std::vector<std::vector<unsigned>> rule_to_nonterminals;
    std::vector<std::vector<unsigned>> normalization_groups;


    GrammarInfo2(
            const std::function<unsigned(Nonterminal)> nont_idx
            , const std::vector<std::vector<unsigned>> rule_to_nonterminals
    )
            : nont_idx(nont_idx)
            , rule_to_nonterminals(rule_to_nonterminals)
    {
        std::vector<std::vector<unsigned>> normGroups;
        for (unsigned rule_idx = 0; rule_idx < rule_to_nonterminals.size(); ++rule_idx) {
            if (rule_to_nonterminals[rule_idx].size() > 0) {
                if (normGroups.size() <= rule_to_nonterminals[rule_idx][0]) {
                    normGroups.resize(rule_to_nonterminals[rule_idx][0] + 1);
                }
                normGroups[rule_to_nonterminals[rule_idx][0]].push_back(rule_idx);
            }
        }

        normalization_groups = normGroups;

//        { // Debug Output:
//            unsigned i = 0;
//            for (auto rtn : rule_to_nonterminals) {
//                std::cerr << i << ": ";
//                unsigned j = 0;
//                for (auto n : rtn) {
//                    if (j == 1) {
//                        std::cerr << "-> ";
//                    }
//                    std::cerr << n << " ";
//                    ++j;
//                }
//                std::cerr << ";" << std::endl;
//                ++i;
//            }
//            for (unsigned i = 0; i < normalization_groups.size(); ++i) {
//                std::cerr << i << " : { ";
//                for (auto n : normalization_groups[i]) {
//                    std::cerr << n << " ";
//                }
//                std::cerr << "} ";
//            }
//            std::cerr << std::endl;
//        }

    }
};








template <typename Nonterminal, typename Terminal, typename TraceID>
class TraceManager2 : public Manager<TraceInfo<Nonterminal, TraceID>> {
private:
    const bool debug;

    const bool self_malloc = true;
    StorageManager storageManager {true};

    MAPTYPE<Nonterminal, unsigned int> noOfItemsPerNonterminal;

    std::vector<MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>> traces_inside_weights;
    std::vector<MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>> traces_outside_weights;


public:

    TraceManager2(const bool debug = false) : debug(debug) {}

    // Hook on create


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


    // ##############################
    // ##############################
    // ##############################


    template<typename Val>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>>
    split_merge(
            const unsigned N_THREADS, const unsigned BATCH_SIZE,
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>>& rule2nonterminals,
            const unsigned noEpochs, const std::map<Nonterminal, unsigned>& nont_idx, const unsigned split_merge_cycles, const double merge_threshold, const double merge_percentage=-1.0
    ) {
        std::function<unsigned(Nonterminal)> nont_idx_f = [&](const Nonterminal nont) -> unsigned { return nont_idx.at(nont);};
        GrammarInfo2<Nonterminal> grammarInfo(nont_idx_f, rule2nonterminals);
        const std::vector<std::vector<unsigned>> & normalization_groups = grammarInfo.normalization_groups;

        std::cerr << "starting split merge training" << std::endl;
        std::cerr << "# nonts: " << nont_idx.size() << std::endl;

        return split_merge<Val>(N_THREADS, BATCH_SIZE, rule_weights, rule2nonterminals, normalization_groups,
                                noEpochs, nont_idx_f,
                                split_merge_cycles, nont_idx.size(), merge_threshold, merge_percentage);
    };



    template<typename Val, typename NontToIdx>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>>
    split_merge(
            const unsigned N_THREADS, const unsigned BATCH_SIZE,
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const std::vector<std::vector<unsigned>> &normalization_groups, const unsigned n_epochs,
            const NontToIdx nont_idx, const unsigned split_merge_cycles, const unsigned long n_nonts, const double merge_threshold, const double merge_percentage=-1.0
    ){

        const double epsilon = 0.0001;

        // the next two structures hold split-dimensions and
        // rule weights for latent annotated rules before and after
        // each split/merge cycle
        std::vector<unsigned> nont_dimensions = std::vector<unsigned>(n_nonts, 1);
        std::vector<std::vector<Val>> rule_weights_la = init_weights_la<Val>(rule_weights);

        std::vector<Val> root_weights {Val::one()};

        for (unsigned cycle = 0; cycle < split_merge_cycles; ++cycle) {
            split_merge_cycle(N_THREADS, BATCH_SIZE, cycle, n_epochs, epsilon
                    , merge_threshold, merge_percentage, rule_to_nonterminals, normalization_groups, nont_idx, nont_dimensions
                    , rule_weights_la, root_weights);
        }

        std::vector<std::vector<double>> rule_weights_la_unlog = valToDouble(rule_weights_la);

        return std::make_pair(nont_dimensions, rule_weights_la_unlog);

    }



    template<typename Val, typename NontToIdx>
    void
    split_merge_cycle(const unsigned N_THREADS, const unsigned BATCH_SIZE, const unsigned cycle
            , const unsigned n_epochs, double epsilon, double merge_threshold, double merge_percentage
            , const std::vector<std::vector<unsigned>> &rule_to_nonterminals
            , const std::vector<std::vector<unsigned>> &normalization_groups, NontToIdx nont_idx
            , std::vector<unsigned> & nont_dimensions, std::vector<std::vector<Val>> & rule_weights_la
            , std::vector<Val> & root_weights
    ){
        std::vector<std::vector<Val>> rule_weights_splitted;
        rule_weights_splitted.reserve(rule_weights_la.size());
        std::vector<Val> root_weights_splitted;
        std::vector<std::vector<Val>> rule_weights_merged;
        std::vector<std::vector<unsigned>> rule_dimensions_splitted;
        rule_dimensions_splitted.reserve(rule_weights_la.size());

        std::vector<unsigned> split_dimensions;
        split_dimensions.reserve(nont_dimensions.size());

        if (debug) std::cerr << "prepare split" << std::endl;

        for (const unsigned dim : nont_dimensions)
            split_dimensions.push_back(dim * 2);

        // splitting
        for (unsigned i = 0; i < rule_weights_la.size(); ++i) {
            const std::vector<Val> &rule_weight = rule_weights_la[i];
            std::vector<unsigned> dimensions;
            dimensions.reserve(rule_to_nonterminals[i].size());
            for (auto nont : rule_to_nonterminals[i]) {
                dimensions.push_back(split_dimensions[nont]);
            }
            rule_weights_splitted.push_back(split_rule(rule_weight, dimensions));
            if (debug) {
                std::cerr << std::endl << "Rule prob " << i << " : { ";
                for (const auto val : rule_weight) std::cerr << val << " ";
                std::cerr << " } " << std::endl << "after split: { ";
                for (const auto val : rule_weights_splitted.back()) std::cerr << val << " ";
                std::cerr << " } " << std::endl << std::endl;
            }
            rule_dimensions_splitted.push_back(std::move(dimensions));
        }

        const double root_split = rand_split() * 0.5;
        root_weights_splitted = {Val::to(root_split) * Val::one(), Val::to(1 - root_split) * Val::one()};


        // todo: review: Why do you clear rule weights created in another function and given by reference?
        rule_weights_la.clear();

        std::cerr << "em training after " << cycle + 1 << ". split" << std::endl;

        // em training

        // memory allocation as prerequisite for EM training
        unsigned long required_memory = total_rule_sizes(rule_weights_splitted) * (1 + N_THREADS); // *2 for rule weights and rule counts
        required_memory += max_item_size(split_dimensions, nont_idx) * 2; // *2 for inside and outside weight

        std::cerr << " Tot rule size: " << total_rule_sizes(rule_weights_splitted) << " max item size " << max_item_size(split_dimensions, nont_idx) << std::endl;

        required_memory += 2 * 2; // for root weights and counts
        if (not storageManager.reserve_memory(required_memory)) {
            std::cerr << "Could not reserve required memory." << std::endl;
            abort();
        }

        const auto allocated_io_weights = allocate_io_weight_maps(nont_idx, split_dimensions);

        std::vector<double *> rule_weights_ptrs;
        std::vector<RuleTensor<double>> rule_weight_tensors;
        double * root_weights_ptrs;
        unsigned allocated = convert_to_eigen(rule_weights_ptrs, rule_weights_splitted, rule_weight_tensors, root_weights_ptrs, root_weights_splitted, rule_dimensions_splitted);

        // do actual EM training
        do_em_training_la(N_THREADS, BATCH_SIZE, rule_weight_tensors, rule_weights_ptrs, root_weights_ptrs, normalization_groups, n_epochs, split_dimensions,
                          rule_to_nonterminals, nont_idx);

        convert_from_eigen(rule_weights_ptrs, rule_weights_splitted, root_weights_ptrs, root_weights_splitted, rule_dimensions_splitted, allocated);
        rule_weight_tensors.clear();

        for (auto rule_weights_ : rule_weights_splitted) {
            for (auto rule_weight : rule_weights_) {
                if (rule_weight > (Val::one() + Val::to(epsilon))) {
                    std::cerr << "bad rule weight: " << rule_weight << std::endl;
                }
                assert(rule_weight <= (Val::one() + Val::to(epsilon)));
            }
        }

        // determine merges
        const auto merge_info = merge_prepare(N_THREADS, BATCH_SIZE, rule_weights_splitted, root_weights_splitted, split_dimensions,
                                              rule_to_nonterminals, nont_idx, Val::to(merge_threshold), merge_percentage);


        // nonterminal -> new_la -> contributing old_las
        const std::vector<std::vector<std::vector<unsigned>>> & merge_selection = std::get<0>(merge_info);
        const std::vector<unsigned> & new_nont_dimensions = std::get<1>(merge_info);
        const std::vector<std::vector<Val>> merge_factors = std::get<2>(merge_info);

        if (debug) {
            std::cerr << "merge factors ";
            for (auto factors : merge_factors) {
                std::cerr << "{ ";
                for (auto factor : factors) {
                    std::cerr << factor << " ";
                }
                std::cerr << " } ";
            }
            std::cerr << std::endl;
        }

        std::vector<std::vector<unsigned>> rule_dimensions_merged;

        // merging
        for (unsigned i = 0; i < rule_weights_splitted.size(); ++i) {
            std::vector<unsigned> old_dimensions;
            std::vector<unsigned> new_dimensions;
            //new_dimensions.reserve(rule_to_nonterminals[i].size());
            std::vector<std::vector<std::vector<unsigned>>> merges;
            //merges.reserve(rule_to_nonterminals[i].size());
            const std::vector<Val> & lhn_merge_factors = merge_factors[rule_to_nonterminals[i][0]];
            for (auto nont : rule_to_nonterminals[i]) {
                old_dimensions.push_back(split_dimensions[nont]);
                new_dimensions.push_back(merge_selection[nont].size());
                merges.push_back(merge_selection[nont]);
            }
            rule_weights_merged.push_back(
                    merge_rule(rule_weights_splitted[i], old_dimensions, new_dimensions, merges,
                               lhn_merge_factors));
            rule_dimensions_merged.emplace_back(std::move(new_dimensions));
        }

        rule_weights_splitted.clear();

        // em training
//        shrink_io_weight_maps(nont_idx, new_nont_dimensions);
        free_io_weight_maps(allocated_io_weights.first, allocated_io_weights.second);
        const auto allocated_io_weights2 = allocate_io_weight_maps(nont_idx, new_nont_dimensions);

        // conversion
        std::vector<double *> rule_weights_merged_ptrs;
        double * root_weights_merged_ptrs;
        allocated = convert_to_eigen(rule_weights_merged_ptrs, rule_weights_merged, rule_weight_tensors, root_weights_merged_ptrs, root_weights, rule_dimensions_merged);

        do_em_training_la(N_THREADS, BATCH_SIZE, rule_weight_tensors, rule_weights_merged_ptrs, root_weights_merged_ptrs, normalization_groups, n_epochs, new_nont_dimensions,
                          rule_to_nonterminals, nont_idx);

        convert_from_eigen(rule_weights_merged_ptrs, rule_weights_merged, root_weights_merged_ptrs, root_weights, rule_dimensions_merged, allocated);
        rule_weight_tensors.clear();


        for (auto rule_weights_ : rule_weights_merged) {
            for (auto rule_weight : rule_weights_) {
                if (rule_weight > Val::one() + Val::to(epsilon)) {
                    std::cerr << "bad rule weight: " << rule_weight << std::endl;
                }
                assert(rule_weight <= Val::one() + Val::to(epsilon));
            }
        }

        free_io_weight_maps(allocated_io_weights2.first, allocated_io_weights2.second);

        // create valid state after split/merge cycle
        nont_dimensions = new_nont_dimensions;
        rule_weights_la = rule_weights_merged;
    }










    template<typename Val>
    std::vector<std::vector<Val>> init_weights_la(const std::vector<double> &rule_weights) const {
        std::vector<std::vector< Val >> rule_weights_la;
        for (const double& rule_weight : rule_weights) {
            rule_weights_la.emplace_back(1, Val::to(rule_weight));
        }
        return rule_weights_la;
    }


    template <typename NontToIdx>
    void do_em_training_la(
            const unsigned N_THREADS
            , const unsigned BATCH_SIZE
            , std::vector<RuleTensor<double>> & rule_tensors
            , std::vector<double*> & rule_weights
            , double* const the_root_weights
            , const std::vector<std::vector<unsigned>> & normalization_groups
            , const unsigned n_epochs
            , const std::vector<unsigned> & nont_dimensions
            , const std::vector<std::vector<unsigned>> & rule_to_nont_ids
            , const NontToIdx & nont_idx
    ){

        feenableexcept(FE_DIVBYZERO | FE_INVALID| FE_OVERFLOW);
        unsigned epoch = 0;

        const unsigned root_dimension = nont_dimensions[nont_idx(begin()->get_goal())];
        unsigned rule_dimension_total = 0;

        std::vector<std::vector<double*>> rule_counts (N_THREADS);
        std::vector<std::vector<RuleTensor<double>>> rule_count_tensors (N_THREADS);
        std::vector<std::vector<unsigned>> rule_dimensions;
        for (auto nont_ids : rule_to_nont_ids) {
            unsigned size = 1;
            std::vector<unsigned> rule_dimension;
            for (auto nont_id : nont_ids) {
                rule_dimension.push_back(nont_dimensions[nont_id]);
                size *= nont_dimensions[nont_id];
            }
            rule_dimensions.push_back(rule_dimension);
            for (unsigned thread = 0; thread < N_THREADS; ++thread) {
                double *ptr = storageManager.get_region(size);
                rule_dimension_total += size;

                rule_counts[thread].push_back(ptr);

                switch (rule_dimension.size()) {
                    case 1:
                        rule_count_tensors[thread].push_back(createTensor<1>(ptr, rule_dimension));
                        break;
                    case 2:
                        rule_count_tensors[thread].push_back(createTensor<2>(ptr, rule_dimension));
                        break;
                    case 3:
                        rule_count_tensors[thread].push_back(createTensor<3>(ptr, rule_dimension));
                        break;
                    case 4:
                        rule_count_tensors[thread].push_back(createTensor<4>(ptr, rule_dimension));
                        break;
                    default:
                        std::cerr << "Rule Dimension out of Bounds:" << rule_dimension.size();
                        abort();
                }
            }
        }

        Eigen::TensorMap<Eigen::Tensor<double, 1>> root_probability (the_root_weights, root_dimension);

        // initialize root counts
        double* root_count_ptr = storageManager.get_region(root_dimension);
        Eigen::TensorMap<Eigen::Tensor<double, 1>> root_count (root_count_ptr, root_dimension);

        Eigen::Tensor<double, 1> trace_root_probabilities;
        Eigen::Tensor<double, 0> trace_root_probability;
        Eigen::Tensor<double, 0> corpus_prob_sum;

        while (epoch < n_epochs) {
            if (self_malloc) {
                // reset rule counts (all at once)
                Eigen::TensorMap<Eigen::Tensor<double, 1>> rule_count(rule_counts[0][0], rule_dimension_total);
                rule_count.setZero();
            } else {
                for (unsigned thread = 0; thread < N_THREADS; ++thread) {
                    unsigned rule = 0;
                    for (auto nont_ids : rule_to_nont_ids) {
                        const unsigned size = calc_size(rule_dimensions[rule]);
                        Eigen::TensorMap<Eigen::Tensor<double, 1>> rule_count(rule_counts[thread][rule], size);
                        rule_count.setZero();
                        ++rule;
                    }
                }
            }

            // reset root counts
            root_count.setZero();

            double corpus_likelihood(0.0);

            // expectation
            if (N_THREADS <= 1) {
                expectation_la(rule_tensors, rule_count_tensors[0], rule_dimensions, root_probability,
                               root_count, corpus_likelihood, 0, traces.size());
            } else {
                expectation_la_parallel(N_THREADS, BATCH_SIZE, rule_tensors, rule_count_tensors, rule_dimensions, root_probability,
                                        root_count, corpus_likelihood);
            }

            // maximization
            if (N_THREADS <= 1) {
                unsigned nont = 0;
                for (const std::vector<unsigned> & group : normalization_groups) {
                    const unsigned lhs_dim = rule_dimensions[group[0]][0];
                    maximization(lhs_dim, rule_dimensions, group, rule_counts[0], rule_weights);
                    if (debug) {
                        std::cerr << "rules for nonterminal " << nont << std::endl;
                        for (auto rule : group) {
                            std::cerr << "rule " << rule << " has probabilites: " << std::endl;
                            std::cerr << Eigen::TensorMap<Eigen::Tensor<double, 2>>(rule_weights[rule], lhs_dim, subdim(rule_dimensions[rule])) << std::endl;
                        }
                    }
                    ++nont;
                }
            } else {
                unsigned nont = 0;
                std::mutex nont_mutex;

                std::vector<std::thread> workers;
                for (unsigned thread = 0; thread < N_THREADS; ++thread) {
                    workers.push_back(std::thread([&]()
                                                  {
                                                      unsigned my_nont {0};
                                                      unsigned max_nont {0};

                                                      while (my_nont < normalization_groups.size()) {
                                                          {
                                                              std::lock_guard<std::mutex> lock(nont_mutex);
                                                              my_nont = nont;
                                                              max_nont = std::min<unsigned>(nont + BATCH_SIZE, normalization_groups.size());
                                                              nont = max_nont;
                                                          }

                                                          for ( ; my_nont < max_nont; ++my_nont) {
                                                              const std::vector<unsigned> & group = normalization_groups[my_nont];
                                                              const unsigned lhs_dim = rule_dimensions[group[0]][0];
                                                              maximization(lhs_dim, rule_dimensions, group, rule_counts[0], rule_weights);
                                                              if (debug) {
                                                                  std::cerr << "rules for nonterminal " << my_nont << std::endl;
                                                                  for (auto rule : group) {
                                                                      std::cerr << "rule " << rule << " has probabilites: " << std::endl;
                                                                      std::cerr << Eigen::TensorMap<Eigen::Tensor<double, 2>>(rule_weights[rule],
                                                                                                                              lhs_dim,
                                                                                                                              subdim(rule_dimensions[rule]))
                                                                                << std::endl;
                                                                  }
                                                              }
                                                          }
                                                      }
                                                  }));
                }
                std::for_each(workers.begin(), workers.end(), [](std::thread &t)
                {
                    t.join();
                });
            }
            if (debug) std::cerr << std::endl;

            epoch++;
            std::cerr <<"Epoch " << epoch << "/" << n_epochs << ": ";


            // maximize root weights:
            corpus_prob_sum = root_count.sum();

            // output likelihood information based on old probability assignment
            std::cerr << "corpus prob. sum " << corpus_prob_sum;
            std::cerr << " corpus likelihood " << corpus_likelihood;
            std::cerr << " root weights: " << root_probability << std::endl;

            // compute new root weights
            if (corpus_prob_sum(0) > 0)
                root_probability = root_count * (1 / corpus_prob_sum(0));
        }

        if (self_malloc) {
            if (not free_region(root_count_ptr, root_dimension))
                abort();
            if (not free_region(rule_counts[0][0], rule_dimension_total))
                abort();
        } else {
            free_region(root_count_ptr, root_dimension);
            for (unsigned thread = 0; thread < N_THREADS; ++ thread) {
                for (auto ptr : rule_counts[thread]) {
                    free_region(ptr, 0);
                }
            }
        }
        fedisableexcept(FE_DIVBYZERO | FE_INVALID| FE_OVERFLOW);

    }


    template<typename VECTOR>
    inline void expectation_la(const std::vector<RuleTensor<double>> &rule_tensors
            , std::vector<RuleTensor<double>> &rule_count_tensors
            , const std::vector<std::vector<unsigned int>> &rule_dimensions
            , const Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &root_probability
            , VECTOR &root_count
            , double &corpus_likelihood
            , const ConstManagerIterator<TraceInfo<Nonterminal, TraceID>> start
            , const ConstManagerIterator<TraceInfo<Nonterminal, TraceID>> end
    ) {
        // expectation
        for (auto traceIterator = start; traceIterator != end; ++traceIterator) {
            const auto& trace = *traceIterator;
            if (trace->get_hypergraph()->size() == 0)
                continue;

            trace->io_weights_la(rule_tensors
                    , root_probability
                    , traces_inside_weights[traceIterator - this->cbegin()]
                    , traces_outside_weights[traceIterator - this->cbegin()]
            );

            const auto & inside_weights = traces_inside_weights[traceIterator - this->cbegin()];
            const auto & outside_weights = traces_outside_weights[traceIterator - this->cbegin()];

            const Eigen::TensorMap<Eigen::Tensor<double, 1>> & root_inside_weight = inside_weights.at(trace->get_goal());
            const Eigen::TensorMap<Eigen::Tensor<double, 1>> & root_outside_weight = outside_weights.at(trace->get_goal());

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

            for (auto& node : *(trace->get_hypergraph())) {
                const WeightVector& lhn_outside_weight = outside_weights.at(node);

                if (debug) {
                    std::cerr << node << std::endl << "outside weight" << std::endl << lhn_outside_weight << std::endl;
                    std::cerr << "inside weight" << std::endl;
                    const WeightVector & lhn_inside_weight = inside_weights.at(node);
                    std::cerr << lhn_inside_weight << std::endl;
                }
                for (const auto& edge : node->get_incoming_edges()) {
                    const int rule_id = edge->get_original_id();
                    const size_t rule_dim = edge->get_sources().size() + 1;

                    switch (rule_dim) {
                        case 1:
                            compute_rule_count1(rule_tensors[rule_id],
                                                lhn_outside_weight, trace_root_probability(0),
                                                rule_count_tensors[rule_id]);
                            break;
                        case 2:
                            compute_rule_count2(rule_tensors[rule_id], edge,
                                                lhn_outside_weight, trace_root_probability(0),
                                                inside_weights,
                                                rule_count_tensors[rule_id]
                            );
                            break;
                        case 3:
                            compute_rule_count3(rule_tensors[rule_id], edge,
                                                lhn_outside_weight, trace_root_probability(0),
                                                inside_weights,
                                                rule_count_tensors[rule_id]
                            );
                            break;
                        case 4:
                            compute_rule_count<4>(rule_tensors[rule_id],edge,
                                                  lhn_outside_weight, trace_root_probability(0),
                                                  inside_weights,
                                                  rule_count_tensors[rule_id]
                            );
                            break;
                        default:
                            std::cerr << "Rules with RHS > " << 3 << " are not implemented." << std::endl;
                            abort();
                    }
                }
            }

        }
    }

    inline void expectation_la_parallel(const unsigned THREADS, const unsigned BATCH_SIZE,
                                        const std::vector<RuleTensor<double>> &rule_tensors,
                                        std::vector<std::vector<RuleTensor<double>>> &rule_count_tensors,
                                        const std::vector<std::vector<unsigned int>> &rule_dimensions,
                                        const Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &root_probability,
                                        Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &root_count,
                                        double & corpus_likelihood) {

        unsigned next_trace {0};
        std::mutex next_trace_mutex;

        std::vector<double> corpus_likelihoods(THREADS, corpus_likelihood);
        std::vector<Eigen::Tensor<double, 1>> root_counts;
        for (unsigned thread = 0; thread < THREADS; ++thread) {
            Eigen::Tensor<double, 1> root_count_thread(root_count.dimension(0));
            root_count_thread.setZero();
            root_counts.emplace_back(std::move(root_count_thread));
        }

        std::vector<std::thread> workers;
        for (unsigned thread = 0; thread < THREADS; ++thread) {
            workers.push_back(std::thread([thread,BATCH_SIZE,this,&next_trace_mutex,&next_trace,&rule_tensors,&root_probability,&root_counts,&rule_dimensions,&corpus_likelihoods,&rule_count_tensors]() {
                unsigned my_next_trace = 0;
                unsigned my_last_trace = 0;
                while (my_next_trace < traces.size()) {
                    {
                        std::lock_guard<std::mutex> lock(next_trace_mutex);
                        my_next_trace = next_trace;
                        next_trace += BATCH_SIZE;
                        my_last_trace = std::min<unsigned>(next_trace, traces.size());
                    }

                    expectation_la(rule_tensors, rule_count_tensors[thread], rule_dimensions, root_probability,
                                   root_counts[thread], corpus_likelihoods[thread], my_next_trace, my_last_trace);
                }
            }));
        }

        std::for_each(workers.begin(), workers.end(), [](std::thread &t)
        {
            t.join();
        });

        // collect counts, etc.
        corpus_likelihood
                = std::accumulate(corpus_likelihoods.begin(), corpus_likelihoods.end(), corpus_likelihood, std::plus<double>());
        for (unsigned rule = 0; rule < rule_count_tensors[0].size(); ++ rule) {
            for (unsigned thread = 1; thread < THREADS; ++thread) {
                switch (rule_dimensions[rule].size()) {
                    case 1:
                        boost::get<Eigen::TensorMap<Eigen::Tensor<double, 1>>>(rule_count_tensors[0][rule])
                                += boost::get<Eigen::TensorMap<Eigen::Tensor<double, 1>>>(rule_count_tensors[thread][rule]);
                        break;
                    case 2:
                        boost::get<Eigen::TensorMap<Eigen::Tensor<double, 2>>>(rule_count_tensors[0][rule])
                                += boost::get<Eigen::TensorMap<Eigen::Tensor<double, 2>>>(rule_count_tensors[thread][rule]);
                        break;
                    case 3:
                        boost::get<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(rule_count_tensors[0][rule])
                                += boost::get<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(rule_count_tensors[thread][rule]);
                        break;
                    case 4:
                        boost::get<Eigen::TensorMap<Eigen::Tensor<double, 4>>>(rule_count_tensors[0][rule])
                                += boost::get<Eigen::TensorMap<Eigen::Tensor<double, 4>>>(rule_count_tensors[thread][rule]);
                        break;
                    default:
                        abort();
                }
            }
        }
        for (unsigned thread = 0; thread < THREADS; ++thread) {
            root_count += root_counts[thread];
        }
    }






    inline void compute_rule_count1(const RuleTensor<double> & rule_weight_tensor, const Eigen::TensorMap<Eigen::Tensor<double, 1>> &lhn_outside_weight,
                                    const double trace_root_probability, RuleTensor<double> & rule_count_tensor
    ) {
        constexpr unsigned rule_rank {1};

        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_weight_tensor);

        auto rule_val = rule_weight * lhn_outside_weight;

        Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_count = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_count_tensor);

        if (trace_root_probability > 0) {
            rule_count += rule_val * (1 / trace_root_probability);
        }
    }


    inline void compute_rule_count2(
            const RuleTensor<double> & rule_weight_tensor
            , Element<HyperEdge<Nonterminal>>& edge
            , const Eigen::TensorMap<Eigen::Tensor<double, 1>>& lhn_outside_weight
            , const double trace_root_probability
            , const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , RuleTensor<double> & rule_count_tensor
    ) {
        constexpr unsigned rule_rank {2};

        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_weight_tensor);

        const Eigen::TensorMap<Eigen::Tensor<double, 1>> & rhs_weight = inside_weights.at(edge->get_sources()[0]);
        auto rule_val = lhn_outside_weight.reshape(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1})
                                .broadcast(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1)})
                        * rhs_weight.reshape(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1)})
                                .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1}).eval()
                        * rule_weight
        ;

        Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_count = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_count_tensor);

        if (trace_root_probability > 0) {
            rule_count += rule_val * (1 / trace_root_probability);
        }
    }


    inline void compute_rule_count3(
            const RuleTensor<double> & rule_weight_tensor
            , Element<HyperEdge<Nonterminal>>& edge
            , const Eigen::TensorMap<Eigen::Tensor<double, 1>>& lhn_outside_weight
            , const double trace_root_probability
            , const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , RuleTensor<double> & rule_count_tensor
    ) {
        constexpr unsigned rule_rank {3};

        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_weight_tensor);
        const Eigen::TensorMap<Eigen::Tensor<double, 1>> & rhs_weight1 = inside_weights.at(edge->get_sources()[0]);
        const Eigen::TensorMap<Eigen::Tensor<double, 1>> & rhs_weight2 = inside_weights.at(edge->get_sources()[1]);

        auto rule_val = lhn_outside_weight.reshape(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1, 1})
                                .broadcast(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1), rule_weight.dimension(2)})
                        * rhs_weight1.reshape(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1), 1})
                                .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1, rule_weight.dimension(2)}).eval()
                        * rhs_weight2.reshape(Eigen::array<long, rule_rank>{1, 1, rule_weight.dimension(2)})
                                .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), rule_weight.dimension(1), 1}).eval()
                        * rule_weight;
        ;

        Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_count = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_count_tensor);

        if (trace_root_probability > 0) {
            rule_count += rule_val * (1 / trace_root_probability);
        }
    }

    template<int rule_dim>
    inline void compute_rule_count(
            const RuleTensor<double> & rule_weight_tensor
            , Element<HyperEdge<Nonterminal>>& edge
            , const Eigen::TensorMap<Eigen::Tensor<double, 1>>& lhn_outside_weight
            , const double trace_root_probability
            , const MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector>& inside_weights
            , RuleTensor<double> & rule_count_tensor
    ) {

        const Eigen::TensorMap<Eigen::Tensor<double, rule_dim>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_dim>>>(rule_weight_tensor);
        const Eigen::array<long, rule_dim> & rule_dimension = rule_weight.dimensions();

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

        Eigen::TensorMap<Eigen::Tensor<double, rule_dim>> & rule_count = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_dim>>>(rule_count_tensor);
        if (trace_root_probability > 0) {
            rule_count += rule_val * (1 / trace_root_probability);
        }
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
                ++noOfItemsPerNonterminal[node->get_nonterminal()];
            }
        }
    }


    template<typename NontToIdx>
    std::pair<double*, unsigned>
    allocate_io_weight_maps(const NontToIdx nont_idx, const std::vector<unsigned> & nont_dimensions){
        double * start(nullptr);
        unsigned allocated(0);

        traces_inside_weights.clear();
        traces_inside_weights.reserve(size());
        traces_outside_weights.clear();
        traces_outside_weights.reserve(size());

        for (const auto& trace : *this) {
            MAPTYPE<Element<TraceNode<Nonterminal>>, WeightVector> inside_weights, outside_weights;

            for (const auto& node : *(trace->get_hypergraph()) {
                const unsigned item_dimension = nont_dimensions[nont_idx(node->get_nonterminal())];
                double *const inside_weight_ptr = storageManager.get_region(item_dimension);
                double *const outside_weight_ptr = storageManager.get_region(item_dimension);
                if (start == nullptr)
                    start = inside_weight_ptr;
                allocated += item_dimension * 2;

                Eigen::TensorMap<Eigen::Tensor<double, 1>> inside_weight(inside_weight_ptr, item_dimension);
                Eigen::TensorMap<Eigen::Tensor<double, 1>> outside_weight(outside_weight_ptr, item_dimension);
                inside_weights.emplace(node, std::move(inside_weight));
                outside_weights.emplace(node, std::move(outside_weight));
            }
            traces_inside_weights.push_back(std::move(inside_weights));
            traces_outside_weights.push_back(std::move(outside_weights));
        }
        return std::make_pair(start, allocated);
    }



    template <typename Val>
    unsigned convert_to_eigen(std::vector<double*> & rule_weights_ptrs, const std::vector<std::vector<Val>> & rule_weights, std::vector<RuleTensor<double>> & rule_tensors,
                              double* & root_weights_ptrs, const std::vector<Val> & root_weights, const std::vector<std::vector<unsigned>> & rule_dimensions) {
        unsigned allocated(0);
        unsigned rule = 0;
        for(auto rule_weight : rule_weights) {
            const std::vector<unsigned> & rule_dim = rule_dimensions[rule];
            double * rule_weight_ptr = storageManager.get_region(rule_weight.size());
            const unsigned dims = rule_dim.size();

            switch (dims) {
                case 1:
                    convert_format<1>(rule_weight_ptr, rule_dim, rule_weight, rule_tensors);
                    break;
                case 2:
                    convert_format<2>(rule_weight_ptr, rule_dim, rule_weight, rule_tensors);
                    break;
                case 3:
                    convert_format<3>(rule_weight_ptr, rule_dim, rule_weight, rule_tensors);
                    break;
                case 4:
                    convert_format<4>(rule_weight_ptr, rule_dim, rule_weight, rule_tensors);
                    break;
                default:
                    assert(false && "Rules with more than 3 RHS nonterminals are not implemented.");
                    abort();
            }
            rule_weights_ptrs.push_back(rule_weight_ptr);
            allocated += rule_weight.size();
            ++rule;
        }
        root_weights_ptrs = storageManager.get_region(root_weights.size());
        allocated += root_weights.size();
        for (unsigned i = 0; i < root_weights.size(); ++i) {
            root_weights_ptrs[i] = root_weights[i].from();
        }
        return allocated;
    }



    template<typename Val>
    void convert_from_eigen(const std::vector<double*> rule_weight_ptr, std::vector<std::vector<Val>> & rule_weights,
                            const double* root_weight_ptr, std::vector<Val> & root_weights, const std::vector<std::vector<unsigned>> & rule_dimensions, const unsigned allocated) {


        for(unsigned rule = 0; rule < rule_weights.size(); ++rule) {
            auto & rule_weight = rule_weights[rule];
            const std::vector<unsigned> & rule_dim = rule_dimensions[rule];
            const unsigned dims = rule_dim.size();

            switch (dims) {
                case 1:
                    de_convert_format<1>(rule_weight_ptr[rule], rule_dim, rule_weight);
                    break;
                case 2:
                    de_convert_format<2>(rule_weight_ptr[rule], rule_dim, rule_weight);
                    break;
                case 3:
                    de_convert_format<3>(rule_weight_ptr[rule], rule_dim, rule_weight);
                    break;
                case 4:
                    de_convert_format<4>(rule_weight_ptr[rule], rule_dim, rule_weight);
                    break;
                default:
                    assert(false && "Rules with more than 3 RHS nonterminals are not implemented.");
                    abort();
            }
        }
        for (unsigned i = 0; i < root_weights.size(); ++i) {
            root_weights[i] = Val::to(root_weight_ptr[i]);
        }
        if (self_malloc) {
            if (not storageManager.free_region(rule_weight_ptr[0], allocated))
                abort();
        } else {
            for (auto ptr : rule_weight_ptr) {
                storageManager.free_region(ptr, 0);
            }
        }
    }




};
template <typename Nonterminal, typename Terminal, typename TraceID>
using TraceManagerPtr = std::shared_ptr<TraceManager2<Nonterminal, Terminal, TraceID>>;






class StorageManager {
private:
    bool self_malloc;
    double * start = nullptr;
    double * next = nullptr;
    double * max_mem = nullptr;
    unsigned the_size = 0; // 625000; // 5MB


public:
    StorageManager(bool selfmal = false): self_malloc(selfmal) {}


    bool reserve_memory(unsigned size) {
        if(!self_malloc)
            return true;

        std::cerr << "reserving " << size << std::endl;
        if (start == next) {
            if (start != nullptr and max_mem - start < size) {
                free(start);
            }
            if (start == nullptr or max_mem - start < size) {
                unsigned allocate = the_size > size ? the_size : size;
                std::cerr << "allocating " << allocate << std::endl;
                start = (double *) malloc(sizeof(double) * allocate);
                max_mem = start + allocate;
                next = start;
            }
            return true;
        } else
            return false;
    }

    double * get_region(unsigned size) {
        if (not self_malloc)
            return (double*) malloc(sizeof(double) * size);
        else {
            if (start == nullptr) {
                start = (double *) malloc(sizeof(double) * the_size);
                max_mem = start + the_size;
                next = start;
            }
            if (max_mem - next < size) {
                std::cerr << "Maximum size of double storage exceeded" << std::endl;
                std::cerr << "Required  " << size << std::endl;
                std::cerr << "Available " << (max_mem - next) / sizeof(double) << std::endl;
                abort();
            }
            double *return_ = next;
            next = return_ + size;
            return return_;
        }
    }

    bool free_region(double* const ptr, const unsigned size) {
        if (not self_malloc) {
            free(ptr);
            return true;
        } else {
            if (ptr + size == next) {
                assert(start <= ptr);
                next = ptr;
                return true;
            }
            return false;
        }
    }

};










#endif //STERMPARSER_TRACEMANAGER_H
