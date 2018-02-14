//
// Created by Markus on 22.02.17.
//

#ifndef STERMPARSER_TRACEMANAGER_H
#define STERMPARSER_TRACEMANAGER_H

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "../Names.h"
#include "../Manage/Manager.h"
#include "../Manage/Manager_util.h"
#include "GrammarInfo.h"
#include "TrainingCommon.h"
#include <set>
#include <iostream>
#include <fstream>
#include "../util.h"
#include "LatentAnnotation.h"
#include <queue>
// custom specialization of std::hash can be injected in namespace std
namespace std
{
    template<typename Nonterminal>
    struct hash<pair<Element<Node<Nonterminal>>, size_t>>
    {
        typedef std::pair<Element<Node<Nonterminal>>, size_t> argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& p) const noexcept
        {
            result_type const h1 ( std::hash<Element<Node<Nonterminal>>>{}(p.first) );
            result_type const h2 ( std::hash<size_t>{}(p.second) );
            return h1 ^ (h2 << 1); // or use boost::hash_combine (see Discussion)
        }
    };
}


namespace Trainer {
    template<typename Nonterminal, typename TraceID>
    class TraceManager2;

    template<typename Nonterminal, typename TraceID>
    using TraceManagerPtr = std::shared_ptr<TraceManager2<Nonterminal, TraceID>>;
    template<typename Nonterminal, typename TraceID>
    using TraceManagerWeakPtr = std::weak_ptr<TraceManager2<Nonterminal, TraceID>>;

    void scaledIncrement(WeightVector & targetWeight
                         , int & targetScale
                         , const WeightVector & incrementWeight
                         , const int incrementScale) {
        if (incrementScale > targetScale) {
            targetWeight = targetWeight * calcScaleFactor(targetScale - incrementScale) + incrementWeight;
            targetScale = incrementScale;
        } else {
            targetWeight = targetWeight + incrementWeight * calcScaleFactor(incrementScale - targetScale);
        }
    }

    template <typename Nonterminal>
    struct InsideWeightComputation : boost::static_visitor<void> {
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights;
        Trainer::WeightVector &targetWeight;
        const Element<HyperEdge<Nonterminal>> &edge;
        int & targetLogScale;
        const MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales;
        const bool debug;

        InsideWeightComputation(
                const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , Trainer::WeightVector &targetWeight
                , const Element<HyperEdge<Nonterminal>> &edge
                , int & targetLogScale
                , const MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales
                , bool debug = false
        ) :  insideWeights(insideWeights)
                , targetWeight(targetWeight)
                , edge(edge)
                , targetLogScale(targetLogScale)
                , insideLogScales(insideLogScales)
                , debug(debug)
        {}


        inline void operator()(const RuleTensorRaw<double, 1> & ruleWeight) {
            if (debug) {
                std::cerr << std::endl << "Computing inside weight summand" << std::endl;
                std::cerr << "rule tensor " << edge->get_label_id() << std::endl << ruleWeight << std::endl;
            }
            targetWeight += ruleWeight * calcScaleFactor(-targetLogScale);
        }


        inline void operator()(const RuleTensorRaw<double, 2> & ruleWeight) {
            if (debug) {
                std::cerr << std::endl << "Computing inside weight summand" << std::endl;
                std::cerr << "rule tensor " << edge->get_label_id() << std::endl << ruleWeight << std::endl;
            }

            constexpr unsigned rhsPos = 1;
            const auto &itemWeight = insideWeights.at(edge->get_sources()[rhsPos - 1]);
            auto tmpValue = ruleWeight.contract(
                    itemWeight, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}
            );

            int rhs1LogScale = insideLogScales.at(edge->get_sources()[rhsPos - 1]);

            WeightVector summand = tmpValue;
            rhs1LogScale = scaleTensor(summand, rhs1LogScale);

            scaledIncrement(targetWeight, targetLogScale, summand, rhs1LogScale);
        }

        inline void operator()(
                const RuleTensorRaw<double, 3> & ruleWeight
        ) {
            if (debug) {
                std::cerr << std::endl << "Computing inside weight summand" << std::endl;
                std::cerr << "rule tensor " << edge->get_label_id() << std::endl << ruleWeight << std::endl;
            }
            constexpr unsigned rhsPos1 = 1;
            const auto &rhsItemWeight1 = insideWeights.at(edge->get_sources()[rhsPos1 - 1]);

            constexpr unsigned rhsPos2 = 2;
            const auto &rhsItemWeight2 = insideWeights.at(edge->get_sources()[rhsPos2 - 1]);

            auto tmpValue1 = ruleWeight.contract(
                    rhsItemWeight2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(2, 0)}
            );
            WeightVector summand = tmpValue1.contract(
                    rhsItemWeight1, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}
            );
            int rhs_scales = insideLogScales.at(edge->get_sources()[rhsPos1 - 1])
                                     + insideLogScales.at(edge->get_sources()[rhsPos2 - 1]);

            rhs_scales = scaleTensor(summand, rhs_scales);

            scaledIncrement(targetWeight, targetLogScale, summand, rhs_scales);
        }

        template<int ruleRank>
        inline
        void operator()(
                const Trainer::RuleTensorRaw <double, ruleRank> & ruleWeight
        ) {
            if (debug) {
                std::cerr << std::endl << "Computing inside weight summand" << std::endl;
                std::cerr << "ruleWeight tensor " << edge->get_label_id() << std::endl << ruleWeight << std::endl;
            }
            const auto &ruleDimension = ruleWeight.dimensions();


            Eigen::Tensor<double, ruleRank> tmpValue = ruleWeight;

            Eigen::array<long, ruleRank> reshapeDimension;
            Eigen::array<long, ruleRank> broadcastDimension;
            for (unsigned i = 0; i < ruleRank; ++i) {
                reshapeDimension[i] = 1;
                broadcastDimension[i] = ruleDimension[i];
            }

            int tmpScale = 0;
            for (unsigned rhsPos = 1; rhsPos < ruleRank; ++rhsPos) {
                const auto &item_weight = insideWeights.at(edge->get_sources()[rhsPos - 1]);
                reshapeDimension[rhsPos] = broadcastDimension[rhsPos];
                broadcastDimension[rhsPos] = 1;
                tmpValue *= item_weight.reshape(reshapeDimension).broadcast(broadcastDimension);
                broadcastDimension[rhsPos] = reshapeDimension[rhsPos];
                reshapeDimension[rhsPos] = 1;

                tmpScale += insideLogScales.at(edge->get_sources()[rhsPos - 1]);
                tmpScale = scaleTensor(tmpValue, tmpScale);
            }

            Eigen::array<long, ruleRank - 1> sum_array;
            for (unsigned idx = 0; idx < ruleRank - 1; ++idx) {
                sum_array[idx] = idx + 1;
            }
            WeightVector summand = tmpValue.sum(sum_array);

            scaledIncrement(targetWeight, targetLogScale, summand, tmpScale);
        }
    };

    template <typename Nonterminal>
    struct OutsideWeightComputation : boost::static_visitor<void> {
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &outsideWeights;
        Trainer::WeightVector &outsideWeight;
        const std::pair<Element<HyperEdge<Nonterminal>>, unsigned int> &outgoing;
        int & targetLogScale;
        const MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales;
        const MAPTYPE<Element<Node<Nonterminal>>, int> & outsideLogScales;
        const LatentAnnotation & latentAnnotation;
        const bool debug;

        OutsideWeightComputation(
                const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &outsideWeights
                , Trainer::WeightVector &outsideWeight
                , const std::pair<Element<HyperEdge<Nonterminal>>, unsigned int> &outgoing
                , int & targetLogScale
                , const MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales
                , const MAPTYPE<Element<Node<Nonterminal>>, int> & outsideLogScales
                , const LatentAnnotation & latentAnnotation
                , const bool debug = false
        ) : insideWeights(insideWeights)
                , outsideWeights(outsideWeights)
                , outsideWeight(outsideWeight)
                , outgoing(outgoing)
                , targetLogScale(targetLogScale)
                , insideLogScales(insideLogScales)
                , outsideLogScales(outsideLogScales)
                , latentAnnotation(latentAnnotation)
                , debug(debug) {}

        inline void operator()(const Trainer::RuleTensorRaw<double, 1>& /* ruleWeight */) {
            // Cannot happen, since there is at least one source (namely 'node')
            std::cerr << "The trace is inconsistent!";
            abort();
        }

        inline void operator()(const Trainer::RuleTensorRaw<double, 2>& ruleWeight) {
            if (debug) {
                std::cerr << std::endl << "Computing outside weight summand" << std::endl;
                std::cerr << "ruleWeight tensor " << outgoing.first->get_label_id() << std::endl << ruleWeight << std::endl;
            }
            const auto &parent = outgoing.first->get_target();
//            constexpr unsigned ruleRank{2};

            const auto &parentWeight = outsideWeights.at(parent);

            if (debug) std::cerr << "parent Weight" << parentWeight << std::endl;

            if (parentWeight.dimension(0) != ruleWeight.dimension(0)) {
                std::cerr << "parent " << parent << " label " << parent->get_label_id()
                          << " weight dim " << parentWeight.dimension(0)
                          << " la label dim: " << latentAnnotation.nonterminalSplits[parent->get_label_id()]
                          << " weight: " << std::endl << parentWeight << std::endl
                          << "rule idx " << outgoing.first->get_label_id()
                          << " dims " << ruleWeight.dimension(0) << " " << ruleWeight.dimension(1)
                          << " weight: " << ruleWeight << std::endl
                          << " rule nonterminals ";
                operator<<(std::cerr, latentAnnotation.grammarInfo.rule_to_nonterminals[outgoing.first->get_label_id()]);
                std::cerr << std::endl;
                std::cerr << latentAnnotation.ruleWeights[outgoing.first->get_label_id()];
                std::cerr << std::endl;
            }

            WeightVector outsideWeightSummand = ruleWeight.contract(
                    parentWeight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
            );

            const int parentLogScale = outsideLogScales.at(parent);

            scaledIncrement(outsideWeight, targetLogScale, outsideWeightSummand, parentLogScale);
        }

        inline void operator()(const Trainer::RuleTensorRaw<double, 3>& ruleWeight) {
            if (debug) {
                std::cerr << std::endl << "Computing outside weight summand" << std::endl;
                std::cerr << "ruleWeight tensor " << outgoing.first->get_label_id() << std::endl << ruleWeight << std::endl;
            }
            const auto &siblings = outgoing.first->get_sources();
            const auto &parent = outgoing.first->get_target();
            const unsigned position = outgoing.second;
//            constexpr unsigned ruleRank{3};


            const auto &parentWeight = outsideWeights.at(parent);
            const auto &rhsWeight = insideWeights.at(siblings[position == 0 ? 1 : 0]);

            auto tmpValue1 = ruleWeight.contract(
                    rhsWeight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(position == 0 ? 2 : 1, 0)}
            );
            WeightVector tmpValue2 = tmpValue1.contract(
                    parentWeight, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
            );

            const int tmpScale = outsideLogScales.at(parent) + insideLogScales.at(siblings[position == 0 ? 1 : 0]);

            scaledIncrement(outsideWeight, targetLogScale, tmpValue2, tmpScale);
        }

        template<int ruleRank>
        inline void operator()(const Trainer::RuleTensorRaw<double, ruleRank>& ruleWeight) {
            if (debug) {
                std::cerr << std::endl << "Computing outside weight summand" << std::endl;
                std::cerr << "ruleWeight tensor " << outgoing.first->get_label_id() << std::endl << ruleWeight << std::endl;
            }

            const auto &siblings = outgoing.first->get_sources();
            const auto &parent = outgoing.first->get_target();
            const unsigned position = outgoing.second;

//            const auto &ruleWeight = boost::get<Trainer::RuleTensorRaw<double, ruleRank>>(rules[outgoing.first->get_label_id()]);
            const Eigen::array<long, ruleRank> &ruleDimension = ruleWeight.dimensions();
            Eigen::array<long, ruleRank> reshapeDimension;
            Eigen::array<long, ruleRank> broadcastDimension;

            for (unsigned idx = 0; idx < ruleRank; ++idx) {
                reshapeDimension[idx] = 1;
                broadcastDimension[idx] = ruleDimension[idx];
            }

            const auto &parent_weight = outsideWeights.at(parent);
            int tmpScale = outsideLogScales.at(parent);

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

                const auto &item_weight = insideWeights.at(siblings[rhsPos - 1]);
                tmpScale += insideLogScales.at(siblings[rhsPos - 1]);
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


            scaledIncrement(outsideWeight, targetLogScale, outsideWeightSummand, tmpScale);
        }
    };

    template<typename Nonterminal, typename oID>
    class Trace {
    private:
        Manage::ID id;
        TraceManagerWeakPtr<Nonterminal, oID> manager;
        oID originalID;
        HypergraphPtr<Nonterminal> hypergraph;
        Element<Node<Nonterminal>> goal;
        double frequency;


        mutable std::vector<Element<Node<Nonterminal>>> topologicalOrder;

    public:
        Trace(
                const Manage::ID aId
                , const TraceManagerWeakPtr<Nonterminal, oID> aManager
                , const oID oid
                , HypergraphPtr<Nonterminal> aHypergraph
                , Element<Node<Nonterminal>> aGoal
                , double aFrequency = 1.0
        )
                : id(aId), manager(std::move(aManager)), originalID(std::move(oid)), hypergraph(std::move(aHypergraph)),
                  goal(std::move(aGoal)), frequency(aFrequency) {}


        const Element<Trace<Nonterminal, oID>> get_element() const noexcept {
            return Element<Trace<Nonterminal, oID>>(get_id(), manager);
        };

        Manage::ID get_id() const noexcept { return id; }

        const oID get_original_id() const noexcept { return originalID; }

        const HypergraphPtr<Nonterminal> &get_hypergraph() const noexcept {
            return hypergraph;
        }

        bool is_consistent_with_grammar(const GrammarInfo2& grammarInfo) {
            if (goal->get_label_id() != grammarInfo.start) {
                std::cerr << "Inconsistent trace: root label is different from start nonterminal "
                          << goal->get_label_id() << " vs. " << grammarInfo.start << std::endl;
                return false;
            }
            for (const Element<HyperEdge<Nonterminal>> & edge : *(get_hypergraph()->get_edges().lock())) {
                std::vector<size_t> nonterminals;
                nonterminals.push_back(edge->get_target()->get_label_id());
                for (const Element<Node<Nonterminal>>& source : edge->get_sources()){
                    nonterminals.push_back(source->get_label_id());
                }
                size_t edge_idx {edge->get_label_id()};
                if (nonterminals != grammarInfo.rule_to_nonterminals[edge_idx]) {
                    std::cerr << "Inconsistent trace: edge label " << edge_idx
                              << " and grammar info mismatch " << std::endl
                              << " edge: ";
                    operator<<(std::cerr, nonterminals);
                    std::cerr << std::endl << " rule: ";
                    operator<<(std::cerr, grammarInfo.rule_to_nonterminals[edge_idx]);
                            std::cerr << std::endl;

                    return false;
                }
            }
            return true;
        }

        const Element<Node<Nonterminal>> &get_goal() const noexcept {
            return goal;
        }

        double get_frequency() const {
            return frequency;
        }

        bool has_topological_order() const {
            return get_topological_order().size() == hypergraph->size();
        }

        // TODO: this is not a const function, but there is some hassle if it is not declared as such
        const std::vector<Element<Node<Nonterminal>>> &get_topological_order() const {
            if (topologicalOrder.size() == hypergraph->size())
                return topologicalOrder;

            std::vector<Element<Node<Nonterminal>>> topOrder{};
            topOrder.reserve(hypergraph->size());
            std::set<Element<Node<Nonterminal>>> visited{};
            bool changed = true;
            while (changed) {
                changed = false;

                // add item, if all its decendants were added
                for (const auto &node : *hypergraph) {
                    if (visited.find(node) != visited.cend())
                        continue;
                    bool violation = false;
                    for (const auto &edge : hypergraph->get_incoming_edges(node)) {
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


        template<typename Val>
        std::pair<MAPTYPE<Element<Node<Nonterminal>>, Val>
                  , MAPTYPE<Element<Node<Nonterminal>>, Val>>
        io_weights(std::vector<Val> &ruleWeights) const {

            if(has_topological_order()) {
//                std::cerr << "Calculate IO-weights using topological order" << std::endl;
                return io_weights_topological(ruleWeights);
            }
            else {
//                std::cerr << "Calculate IO-weights using fixpoint approximation" << std::endl;
                return io_weights_fixpoint(ruleWeights);
            }

        };


        template<typename Val>
        std::pair<MAPTYPE<Element<Node<Nonterminal>>, Val>
                  , MAPTYPE<Element<Node<Nonterminal>>, Val>>
        io_weights_topological(std::vector<Val> &ruleWeights) const {

            // calculate inside weigths
            MAPTYPE<Element<Node<Nonterminal>>, Val> inside{};
            for (const auto &node : get_topological_order()) {
                inside[node] = Val::zero();
                for (const auto &incomingEdge : hypergraph->get_incoming_edges(node)) {
                    Val val(ruleWeights[incomingEdge->get_label_id()]);
                    for (const auto &sourceNode : incomingEdge->get_sources())
                        val *= inside.at(sourceNode);

                    inside[node] += val;
                }
            }

            // calculate outside weights
            MAPTYPE<Element<Node<Nonterminal>>, Val> outside{};
            for (auto nodeIterator = get_topological_order().crbegin();
                 nodeIterator != get_topological_order().crend(); ++nodeIterator) {
                Val val = Val::zero();
                if (*nodeIterator == goal)
                    val += Val::one();
                for (const auto outgoing : hypergraph->get_outgoing_edges(*nodeIterator)) {
                    Val valOutgoing = outside.at(outgoing.first->get_target());
                    valOutgoing *= ruleWeights[outgoing.first->get_label_id()];
                    const auto &incomingNodes(outgoing.first->get_sources());
                    for (unsigned int pos = 0; pos < incomingNodes.size(); ++pos) {
                        if (pos != outgoing.second)
                            valOutgoing *= inside.at(incomingNodes[pos]);
                    }
                    val += valOutgoing;
                }
                outside[*nodeIterator] = val;
            }

            return std::make_pair(inside, outside);
        }



        inline void inside_weights_la(
                const LatentAnnotation & latentAnnotation
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales
                , bool scaling = false
                , bool debug = false
        ) const {
            if(has_topological_order()) {
                if (debug) std::cerr << "Calculate Inside-weights using topological order" << std::endl;
                inside_weights_topological_la(latentAnnotation, insideWeights, insideLogScales, scaling, debug);
            }
            else {
                if (debug) std::cerr << "Calculate Inside-weights using fixpoint approximation" << std::endl;
                inside_weights_fixpoint_la(latentAnnotation, insideWeights, insideLogScales, scaling, debug);
            }
        }

        inline void inside_weights_topological_la(
                const LatentAnnotation & latentAnnotation
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales
                , bool scaling = false
                , bool debug = false
        ) const {
            if (!has_topological_order()) {
                std::cerr << "Hypergraph " << id << " cannot be ordered topologically." << std::endl;
                abort();
            }

            const std::vector<Trainer::RuleTensor<double>> & rules {latentAnnotation.ruleWeights};
            for (const auto &node : get_topological_order()) {
                Trainer::WeightVector &targetWeight = insideWeights[node];

                targetWeight.setZero();
                insideLogScales[node] = 0;
                int & targetScale = insideLogScales[node];

                for (const auto &edge : get_hypergraph()->get_incoming_edges(node)) {
                    InsideWeightComputation<Nonterminal> iwc(insideWeights
                                                             , targetWeight
                                                             , edge
                                                             , targetScale
                                                             , insideLogScales
                                                             , debug);
                    boost::apply_visitor(iwc, rules[edge->get_label_id()]);
                }

                assert(targetWeight.dimension(0)
                       == (long long int) latentAnnotation.nonterminalSplits[node->get_label_id()]);

                if (scaling)
                    targetScale = scaleTensor(targetWeight, targetScale);
                if (debug) {
                    std::cerr << "inside weight " << node << std::endl;
                    std::cerr << targetWeight << " " << targetScale << std::endl;
                }
            }
        }



        inline void inside_weights_la(
                const LatentAnnotation & latentAnnotation
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , bool scaling = false
                , bool debug = false
        ) const {
            MAPTYPE<Element<Node<Nonterminal>>, int> insideLogScales;
            if(has_topological_order())
                inside_weights_la(latentAnnotation, insideWeights, insideLogScales, scaling, debug);
            else {
                for(auto n : *hypergraph)
                    insideLogScales[n] = 0;
                inside_weights_la(latentAnnotation, insideWeights, insideLogScales, scaling, debug);
            }
        }

        inline void io_weights_la(
                  const LatentAnnotation & latentAnnotation
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &outsideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales
                , MAPTYPE<Element<Node<Nonterminal>>, int> & outsideLogScales
                , bool scaling = false
                , bool debug = false
        ) const {
            if(has_topological_order()) {
//                std::cerr << "Calculate IO-weights using topological order" << std::endl;
                return io_weights_topological_la(  latentAnnotation
                                                 , insideWeights
                                                 , outsideWeights
                                                 , insideLogScales
                                                 , outsideLogScales
                                                 , scaling
                                                 , debug);
            }
            else {
//                std::cerr << "Calculate IO-weights using fixpoint approximation" << std::endl;
                return io_weights_fixpoint_la( latentAnnotation
                                              , insideWeights
                                              , outsideWeights
                                              , insideLogScales
                                              , outsideLogScales
                                              , scaling
                                              , debug);
            }
        }

        inline void io_weights_la(
                  const LatentAnnotation & latentAnnotation
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &outsideWeights
                , bool scaling = false
                , bool debug = false
        ) const {
            MAPTYPE<Element<Node<Nonterminal>>, int> insideLogScales;
            MAPTYPE<Element<Node<Nonterminal>>, int> outsideLogScales;
            if(has_topological_order()) {
//                std::cerr << "Calculate IO-weights using topological order" << std::endl;
                return io_weights_topological_la(latentAnnotation
                                                 , insideWeights
                                                 , outsideWeights
                                                 , insideLogScales
                                                 , outsideLogScales
                                                 , scaling);
            }
            else {
//                std::cerr << "Calculate IO-weights using fixpoint approximation" << std::endl;
                return io_weights_fixpoint_la(latentAnnotation
                                              , insideWeights
                                              , outsideWeights
                                              , insideLogScales
                                              , outsideLogScales
                                              , scaling
                                              , debug);
            }
        }

        inline void io_weights_topological_la(
                const LatentAnnotation & latentAnnotation
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &outsideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, int> &insideLogScales
                , MAPTYPE<Element<Node<Nonterminal>>, int> &outsideLogScales
                , bool scaling = false
                , bool debug = false
        ) const {

            inside_weights_topological_la(latentAnnotation, insideWeights, insideLogScales, scaling, debug);
            const std::vector<Trainer::RuleTensor<double>> &rules {latentAnnotation.ruleWeights};
            const Eigen::TensorRef<Eigen::Tensor<double, 1>> &root {latentAnnotation.rootWeights};

            for (auto nodeIterator = get_topological_order().rbegin();
                 nodeIterator != get_topological_order().rend(); ++nodeIterator) {
                const Element<Node<Nonterminal>> &node = *nodeIterator;

                Trainer::WeightVector & outsideWeight {outsideWeights[node]};
                outsideLogScales[node] = 0;
                int & targetLogScale {outsideLogScales[node]};

                if (node == get_goal())
                    outsideWeight = root;
                else
                    outsideWeight.setZero();

                assert(outsideWeight.dimension(0) ==
                       (long long int) latentAnnotation.nonterminalSplits[node->get_label_id()]);

                for (const auto &outgoing : get_hypergraph()->get_outgoing_edges(node)) {
                    OutsideWeightComputation<Nonterminal>
                            outsideWeightComputation(  insideWeights
                                                     , outsideWeights
                                                     , outsideWeight
                                                     , outgoing
                                                     , targetLogScale
                                                     , insideLogScales
                                                     , outsideLogScales
                                                     , latentAnnotation
                                                     , debug
                            );
                    const Element<HyperEdge<Nonterminal>> & edge = outgoing.first;
                    if (rules[edge->get_label_id()].which() + 1 !=
                            latentAnnotation.grammarInfo.rule_to_nonterminals[edge->get_label_id()].size()) {
                        std::cerr << "LA / grammarInfo mismatch " << std::endl;
                        std::cerr << "node " << node->get_label() << " / " << node->get_label_id() << std::endl;
                        std::cerr << "rule " << edge->get_label() << " / " << edge->get_label_id() << std::endl;
                        std::cerr << "edge weight " << rules[edge->get_label_id()] << std::endl << std::endl;
                    }

                    boost::apply_visitor(outsideWeightComputation, rules[edge->get_label_id()]);
                }

                assert(outsideWeight.dimension(0)
                       == (long long int) latentAnnotation.nonterminalSplits[node->get_label_id()]);

                if (scaling)
                    targetLogScale = scaleTensor(outsideWeight, targetLogScale);
                if (debug)
                    std::cerr << "outside weight " << node << std::endl
                              << outsideWeight << " " << targetLogScale << std::endl;
            }
        }

        template<typename Val>
        std::pair<MAPTYPE<Element<Node<Nonterminal>>, Val>
                  , MAPTYPE<Element<Node<Nonterminal>>, Val>>
        io_weights_fixpoint(std::vector<Val> &ruleWeights) const {

            MAPTYPE<Element<Node<Nonterminal>>, Val> inside{};
            MAPTYPE<Element<Node<Nonterminal>>, Val> outside{};

            // initialize values to zero
            for (const auto &node : *hypergraph) {
                inside[node] = Val::zero();
                outside[node] = Val::zero();
            }


            // inside weights
            Val maxChange;
            unsigned int cycleCount = 0;
            while(true) {
                ++cycleCount;
                maxChange = Val::zero();
                for (const auto &node : *hypergraph) {
                    Val old = inside[node];

                    Val target = Val::zero();
                    for (const auto &incomingEdge : hypergraph->get_incoming_edges(node)) {
                        Val val(ruleWeights[incomingEdge->get_label_id()]);
                        for (const auto &sourceNode : incomingEdge->get_sources())
                            val *= inside.at(sourceNode);

                        target += val;
                    }

                    inside[node] = target;

                    // calculate change
                    Val delta = inside[node] - old;
                    if(delta < 0)
                        delta *= -1;
                    // update max change
                    maxChange = std::max(maxChange, delta);
                }

                if(cycleCount > manager.lock()->get_io_cycle_limit() || maxChange < Val::to(manager.lock()->get_io_precision())){
                    break;
                }
            }

            // compute outside values

            cycleCount = 0;
            while(true) {
                maxChange = Val::zero();
                for (const auto &node : *hypergraph) {
                    ++cycleCount;
                    Val old = outside[node];
                    if(node == goal)
                        outside[node] = Val::one();
                    else
                        outside[node] = Val::zero();
                    for (const auto &outgoingEdge : hypergraph->get_outgoing_edges(node)) {
                        Val val(ruleWeights[outgoingEdge.first->get_label_id()]);
                        val *= outside[outgoingEdge.first->get_target()];
                        for (size_t i = 0; i < outgoingEdge.first->get_sources().size(); i++) {
                            if (i != outgoingEdge.second)
                                val *= inside.at(outgoingEdge.first->get_sources()[i]);
                        }

                        outside[node] += val;
                    }


                    // calculate change
                    Val delta = outside[node] - old;
                    if (delta < 0)
                        delta *= -1;
                    // update max change
                    maxChange = std::max(maxChange, delta);
                }

                if(cycleCount > manager.lock()->get_io_cycle_limit() || maxChange < Val::to(manager.lock()->get_io_precision())){
                    break;
                }

            }
            return std::make_pair(inside, outside);
        };




        inline void io_weights_fixpoint_la(
                  const LatentAnnotation & latentAnnotation
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &outsideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales
                , MAPTYPE<Element<Node<Nonterminal>>, int> & outsideLogScales
                , bool scaling = false
                , bool debug = false
        ) const {
            for(auto n : *hypergraph){
                insideWeights[n].setZero();
                outsideWeights[n].setZero();
                insideLogScales[n] = 0;
                outsideLogScales[n] = 0;
            }

            const std::vector<Trainer::RuleTensor<double>> &rules {latentAnnotation.ruleWeights};
            const Eigen::TensorRef<Eigen::Tensor<double, 1>> &root {latentAnnotation.rootWeights};

            inside_weights_fixpoint_la(latentAnnotation, insideWeights, insideLogScales, scaling);

            unsigned int cycle_count {0};
            while(true) {
                double maxChange {0.0};
                ++cycle_count;
                if (debug) std::cerr << "cycle count " << cycle_count << std::endl;

                for (const Element<Node<Nonterminal>> &node : *hypergraph){

                    Trainer::WeightVector outsideWeight = outsideWeights.at(node);
                    Trainer::WeightVector oldWeight = outsideWeights.at(node);

                    if (node == get_goal())
                        outsideWeight = root;
                    else
                        outsideWeight.setZero();

                    int targetLogScale {0};

                    if (debug) std::cerr << "node " << node << " outsideWeight " << outsideWeight << " oldWeight "
                                         << oldWeight << std::endl;

                    for (const auto &outgoing : get_hypergraph()->get_outgoing_edges(node)) {
                        OutsideWeightComputation<Nonterminal> outsideWeightComputation(
                                insideWeights
                                , outsideWeights
                                , outsideWeight
                                , outgoing
                                , targetLogScale
                                , insideLogScales
                                , outsideLogScales
                                , latentAnnotation
                                , debug);
                        boost::apply_visitor(outsideWeightComputation, rules[outgoing.first->get_label_id()]);
                        if (scaling)
                            targetLogScale = scaleTensor(outsideWeight, targetLogScale);
                    }

                    outsideWeights[node] = outsideWeight;

                    if (scaling)
                        outsideLogScales[node] = targetLogScale;

                    if (debug) std::cerr << "outsideWeight " << outsideWeight << " log scales " << outsideLogScales[node] << std::endl;

                    Trainer::WeightVector diffVector(outsideWeight.dimensions());
                    diffVector = oldWeight - outsideWeight;
                    diffVector = diffVector.abs();
                    Eigen::Tensor<double, 0> diffP = diffVector.sum();
                    double diff = diffP(0);
                    maxChange = std::max(maxChange, diff);
                }

                if (debug) std::cerr << "maxChange " << maxChange << std::endl;

                // stop the iteration:
                if(cycle_count > manager.lock()->get_io_cycle_limit() or (not scaling and maxChange < manager.lock()->get_io_precision()))
                    break;

            }
        }


        inline void inside_weights_fixpoint_la(
                const LatentAnnotation & latentAnnotation
                , MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
                , MAPTYPE<Element<Node<Nonterminal>>, int> & insideLogScales
                , bool scaling = false
                , bool debug = false
        ) const {
            // computation of inside weights

            const std::vector<Trainer::RuleTensor<double>> &rules {latentAnnotation.ruleWeights};
            unsigned int cycle_count {0};
            while(true) {
                double maxChange {0.0};
                ++cycle_count;

                for (const auto &node : *hypergraph) {
                    Trainer::WeightVector targetWeight {insideWeights.at(node)};
                    Trainer::WeightVector oldInside {targetWeight};

                    targetWeight.setZero();
                    int targetScale {0};

                    for (const auto &edge : get_hypergraph()->get_incoming_edges(node)) {
                        // the cases are not dependent on the topological order!
                        InsideWeightComputation<Nonterminal>
                                iwc(insideWeights, targetWeight, edge, targetScale, insideLogScales);
                        boost::apply_visitor(iwc, rules[edge->get_label_id()]);
                        if (scaling)
                            targetScale = scaleTensor(targetWeight, targetScale);
                    }

                    insideWeights.at(node) = targetWeight;
                    if (scaling)
                        insideLogScales[node] = targetScale;

                    Trainer::WeightVector diffVector(targetWeight.dimensions());
                    diffVector = oldInside - targetWeight;
                    diffVector = diffVector.abs();
                    Eigen::Tensor<double, 0> diffP = diffVector.sum();
                    double diff = diffP(0);
                    maxChange = std::max(maxChange, diff);
                }


                if(cycle_count > manager.lock()->get_io_cycle_limit() or (not scaling and maxChange < manager.lock()->get_io_precision()))
                    break;
            }
        }



        /**
         * Computes latent Viterbi derivation (using Knuth's generalization of Dijkstra's algorithm.
         *
         * Returns a pair consisting of
         *  - the latent index of the goal node for which the viterbi derivation was found
         *  - a map with witnesses / backtraces.
         * If the latent index equals std::numeric_limits<size_t>::max(), then no best derivation was found
         * and the map with backtraces might be inconsistent.
         */
        std::pair< size_t
                 , MAPTYPE< std::pair<Element<Node<Nonterminal>>, size_t>
                          , std::pair<Element<HyperEdge<Nonterminal>>, std::vector<size_t>>
                          >
                 > computeViterbiPath(const LatentAnnotation & latentAnnotation, bool debug=false) const {
            MAPTYPE<Element<Node<Nonterminal>>, std::vector<double>> maxIncomingWeight;
            MAPTYPE<std::pair<Element<Node<Nonterminal>>, size_t>, std::pair<Element<HyperEdge<Nonterminal>>, std::vector<size_t>>> witness;

            // the Semiring operations
            /*
            // normal probabilities
            const double zero_weight {0.0};
            const double one_weight {1.0};
            auto convert = [](const double x) -> double { return x; };
            auto multiply = [](const double x, const double y) -> double {return x * y; };
            */
            // log probabilities
            const double zero_weight {-std::numeric_limits<double>::infinity()};
            const double one_weight {0.0};
            auto convert = [](const double x) -> double { return std::log(x); };
            auto multiply = [](const double x, const double y) -> double {return x + y; };

            // initialize maxIncomingWeight
            for (const Element<Node<Nonterminal>> & node : *hypergraph) {
                size_t splits {latentAnnotation.nonterminalSplits[node->get_label_id()]};
                maxIncomingWeight[node] =
                        std::vector<double>(splits, zero_weight);
            }


            typedef std::pair<Element<Node<Nonterminal>>, size_t> SubNode;

            auto cmp = [&maxIncomingWeight](const SubNode & left, const SubNode & right) {
                return maxIncomingWeight[left.first][left.second] < maxIncomingWeight[right.first][right.second];
            };
            std::priority_queue<SubNode, std::vector<SubNode>, decltype(cmp)> queue(cmp);

            std::set<SubNode> processed;

            // process edges empty sources
            for (Element<HyperEdge<Nonterminal>> edge : *(get_hypergraph()->get_edges().lock())) {
                if (edge->get_sources().size() == 0) {
                    const auto target = edge->get_target();
                    size_t target_nont = target->get_label_id();
                    size_t rule_id = edge->get_label_id();
                    for (size_t dim {0}; dim < latentAnnotation.nonterminalSplits[target_nont]; ++dim) {
                        if (maxIncomingWeight[target][dim] < convert(latentAnnotation.get_weight(rule_id, {dim}))) {
                            maxIncomingWeight[target][dim] = convert(latentAnnotation.get_weight(rule_id, {dim}));
                            std::vector<size_t> index(1, dim);
                            std::pair<Element<HyperEdge <Nonterminal>>, std::vector<size_t>> new_witness(edge, index);
                            witness.emplace(std::pair<Element<Node<Nonterminal>>, size_t >(target, dim), new_witness);
                            queue.push(std::make_pair(target, dim));
                        }
                    }
                }
            }

            size_t goal_index {std::numeric_limits<size_t>::max()};

            // process until viterbi parse was found or queue is empty
            while (queue.size() > 0) {
                SubNode sn {queue.top()};
                queue.pop();
                auto p {processed.insert(sn)};

                // continue with next element, if sn already in processed
                if (not p.second)
                    continue;

                if (debug)
                    std::cerr << "processing " << sn.first << " " << sn.first->get_label_id() << " " << sn.second
                          << " weight " << maxIncomingWeight[sn.first][sn.second] << std::endl;

                if (sn.first == goal) {
                    goal_index = sn.second;
                    break;
                }

                for (const std::pair<Element<HyperEdge<Nonterminal>>, size_t>& pair : hypergraph->get_outgoing_edges(sn.first)) {
                    auto edge_element = pair.first;
                    const size_t source_position = pair.second;
                    const auto target = edge_element->get_target();

                    std::vector<size_t> index {0};
                    std::vector<double> partial_products {one_weight};
                    size_t j {0};
                    size_t jp1 {1};

                    while (jp1 > 0 and j <= edge_element->get_sources().size()) {
                        assert (jp1 == j + 1);
//                        assert (j + 1 == index.size() or j == index.size());

                        if (j == edge_element->get_sources().size()) {
                            for(index[0] = 0;
                                index[0] < latentAnnotation.nonterminalSplits[target->get_label_id()];
                                ++index[0]) {
                                double weight = multiply(
                                        partial_products.back(),
                                        convert(latentAnnotation.get_weight(edge_element->get_label_id(), index)));

                                // we assume that the goal is
                                if (target == goal)
                                    weight = multiply(weight, convert(latentAnnotation.rootWeights(index[0])));

                                if (weight > maxIncomingWeight[target][index[0]]) {
                                    maxIncomingWeight[target][index[0]] = weight;
                                    witness.emplace(std::make_pair(target, index[0]), std::make_pair(edge_element, index));

                                    // push item again to fix the queue
                                    queue.push(std::make_pair(target, index[0]));
                                }
                            }
                            jp1--;
                            j--;
                            continue;
                        }

//                        const size_t jp1 {j + 1};

                        if (index.size() < jp1 + 1) {
                            if (source_position == j) {
                                index.push_back(sn.second);
                            } else {
                                index.push_back(0);
                            }
                            assert(index.size() == jp1 + 1);
                        } else {
                            if (source_position == j) {
                                index.pop_back();
                                partial_products.pop_back();
                                j--;
                                jp1--;
                                continue;
                            } else {
                                index[jp1] = index[jp1] + 1;
                            }
                        }

                        const Element<Node<Nonterminal>> source {edge_element->get_sources()[j]};

                        if (index[jp1] >= latentAnnotation.nonterminalSplits[source->get_label_id()]) {
                            index.pop_back();
                            partial_products.pop_back();
                            j--;
                            jp1--;
                            continue;
                        }

                        if (processed.count(std::make_pair(source, index[jp1]))) {
                            partial_products.push_back(
                                    multiply(
                                            partial_products.back()
                                            , maxIncomingWeight[source][index[jp1]]) );
                            j++;
                            jp1++;
                        } else {
                            continue;
                        }
                    }
                }
            }

            return std::make_pair(goal_index, witness);
        }





        // Serialization works for LabelT of std::string and size_t
        template<typename T = oID>
        typename std::enable_if_t<
                std::is_same<T, std::string>::value || std::is_same<T, size_t>::value
                , void
                                 >
        serialize(std::ostream &out) const {
            Manage::serialize_string_or_size_t(out, originalID);
            out << ";" << goal;
            out << ";" << frequency << std::endl;
            hypergraph->serialize(out);
        }


        template<typename T = oID>
        static
        typename std::enable_if_t<
                std::is_same<T, std::string>::value || std::is_same<T, size_t>::value
                , Trace<Nonterminal, oID>
                                 >
        deserialize(std::istream &in, Manage::ID id, TraceManagerPtr<Nonterminal, oID> man) {
            oID l;
            char sep;
            Manage::ID goalID;
            Manage::deserialize_string_or_size_t(in, l);
            double freq;
            in >> sep;
            in >> goalID;
            in >> sep;
            in >> freq;
            HypergraphPtr<Nonterminal> hg = Hypergraph<Nonterminal>::deserialize(
                    in
                    , man->get_node_labels()
                    , man->get_edge_labels()
            );
            Element<Node<Nonterminal>> goalElement(goalID, hg);

            return Trace<Nonterminal, oID>(id, man, l, hg, goalElement, freq);
        }

    };


    template<typename Nonterminal, typename TraceID>
    class TraceManager2 : public Manager<Trace<Nonterminal, TraceID>> {
        using TraceIterator = ConstManagerIterator<Trace<Nonterminal, TraceID>>;

    private:
        MAPTYPE<Nonterminal, unsigned int> noOfItemsPerNonterminal;

        std::shared_ptr<const std::vector<Nonterminal>> nodeLabels;
        std::shared_ptr<const std::vector<EdgeLabelT>> edgeLabels;

        const bool debug;

        unsigned int io_cycle_limit {20};
        double io_precision {0.0000001};
    public:
        TraceManager2(
                std::shared_ptr<const std::vector<Nonterminal>> nodeLs
                , std::shared_ptr<const std::vector<EdgeLabelT>> edgeLs
                , const bool debug = false
        ) :
                nodeLabels(nodeLs), edgeLabels(edgeLs), debug(debug) {}



        std::shared_ptr<TraceManager2<Nonterminal, TraceID>> get_shared_ptr() {
            return std::static_pointer_cast<TraceManager2<Nonterminal, TraceID>>(this->shared_from_this());
        };

        Element<Trace<Nonterminal, TraceID>>
        create(TraceID oid, HypergraphPtr<Nonterminal> hg, Element<Node<Nonterminal>> goal, double frequency = 1.0){
            const Manage::ID id = Manager<Trace<Nonterminal, TraceID>>::infos.size();
            Manager<Trace<Nonterminal, TraceID>>::infos.emplace_back(id, this->get_shared_ptr(), oid, hg, goal, frequency);
            return Manager<Trace<Nonterminal, TraceID>>::infos[id].get_element();
        }



        const std::shared_ptr<const std::vector<Nonterminal>> &get_node_labels() const {
            return nodeLabels;
        }

        const std::shared_ptr<const std::vector<EdgeLabelT>> &get_edge_labels() const {
            return edgeLabels;
        }



        void set_io_cycle_limit(unsigned int io_cycle_limit) {
            TraceManager2::io_cycle_limit = io_cycle_limit;
        }

        void set_io_precision(double io_precision) {
            TraceManager2::io_precision = io_precision;
        }

        unsigned int get_io_cycle_limit() const {
            return io_cycle_limit;
        }

        double get_io_precision() const {
            return io_precision;
        }


        void serialize(std::ostream &o) {
            o << "TraceManager Version 1.1" << std::endl;

            o << "NodeLabels:" << std::endl;
            Manage::serialize_labels<Nonterminal>(o, *nodeLabels);
            o << std::endl;
            o << "EdgeLabels:" << std::endl;
            Manage::serialize_labels<EdgeLabelT>(o, *edgeLabels);
            o << std::endl;

            o << Manager<Trace<Nonterminal, TraceID>>::size() << " Traces:" << std::endl;
            for (const auto &trace : *this) {
                trace->serialize(o);
            }
        }

        static TraceManagerPtr<Nonterminal, TraceID> deserialize(std::istream &in) {
            std::string line;
            std::getline(in, line);
            if (line != "TraceManager Version 1.1")
                throw std::string("Version Mismatch for TraceManager");

            std::getline(in, line); // read: "NodeLabels:"
            if (line != "NodeLabels:")
                throw std::string("Unexpected line '" + line + "' expected 'NodeLabels:'");
            std::vector<Nonterminal> nodeLs;
            nodeLs = Manage::deserialize_labels<Nonterminal>(in);

            std::getline(in, line); // end the current line
            std::getline(in, line); // read: "EdgeLabels:"
            if (line != "EdgeLabels:")
                throw std::string("Unexpected line '" + line + "' expected 'EdgeLabels:'");
            std::vector<EdgeLabelT> edgeLs;
            edgeLs = Manage::deserialize_labels<EdgeLabelT>(in);

            int noItems;
            in >> noItems;
            std::getline(in, line);
            if (line != " Traces:")
                throw std::string("Unexpected line '" + line + "' expected ' Traces'");

            TraceManagerPtr<Nonterminal, TraceID> traceManager = std::make_shared<TraceManager2<Nonterminal, TraceID>>(
                    std::make_shared<std::vector<Nonterminal>>(nodeLs)
                    , std::make_shared<std::vector<EdgeLabelT >>(edgeLs)
            );

            for (int i = 0; i < noItems; ++i) {
                Trace<Nonterminal, TraceID> trace{Trace<Nonterminal, TraceID>::deserialize(in, i, traceManager)};
                traceManager->create(trace.get_original_id(), trace.get_hypergraph(), trace.get_goal(), trace.get_frequency());
            }

            return traceManager;
        }


        // Functions for Cython interface
        // todo: remove this
        Trainer::GrammarInfo2 *grammar_info_id(const std::vector<std::vector<size_t>> &rule_to_nonterminals) {
            return new Trainer::GrammarInfo2(rule_to_nonterminals, this->cbegin()->get_goal()->get_label_id());
        }

        // ##############################
        // ##############################
        // ##############################


        template<typename Val>
        std::vector<std::vector<Val>> init_weights_la(const std::vector<double> &rule_weights) const {
            std::vector<std::vector<Val >> rule_weights_la;
            for (const double &rule_weight : rule_weights) {
                rule_weights_la.emplace_back(1, Val::to(rule_weight));
            }
            return rule_weights_la;
        }


        template<typename Val>
        unsigned long total_rule_sizes(const std::vector<std::vector<Val>> &rule_weights) const {
            unsigned long counter = 0;
            for (const auto &rule_weight : rule_weights) {
                counter += rule_weight.size();
            }
            return counter;
        }

        template<typename NontIdx>
        unsigned max_item_size(const std::vector<unsigned> &nont_dimensions, const NontIdx nont_idx) {
            if (noOfItemsPerNonterminal.size() == 0)
                calculate_no_of_items_per_nonterminal();
            unsigned counter = 0;
            for (const auto &pair : noOfItemsPerNonterminal) {
                counter += pair.second * nont_dimensions.at(nont_idx(pair.first));
            }
            return counter;
        }

        void calculate_no_of_items_per_nonterminal() {
            noOfItemsPerNonterminal.clear();
            for (const auto &trace : *this) {
                for (const auto &node : *(trace->get_hypergraph())) {
                    ++noOfItemsPerNonterminal[node->get_label()];
                }
            }
        }


    };


    template<typename Nonterminal, typename TraceID>
    TraceManagerPtr<Nonterminal, TraceID> build_trace_manager_ptr(
            std::shared_ptr<std::vector<Nonterminal>> nLabels
            , std::shared_ptr<std::vector<EdgeLabelT>> eLabels
            , bool debug = false
    ) {
        return std::make_shared<TraceManager2<Nonterminal, TraceID>>(nLabels, eLabels, debug);
    }

    template <typename Nonterminal, typename TraceID>
    void serialize_trace(TraceManagerPtr<Nonterminal, TraceID> traceManager, const std::string & path) {
        std::filebuf fb;
        if (fb.open(path, std::ios::out)) {
            std::ostream out(&fb);
            traceManager->serialize(out);
            fb.close();
        } else {
            std::clog << "Error writing file: \'" << path << "\'" << std::endl;
        }
    }

    template <typename Nonterminal, typename TraceID>
    TraceManagerPtr<Nonterminal, TraceID> load_trace_manager(const std::string & path) {
        std::filebuf fb;
        if (fb.open(path, std::ios::in)) {
            std::istream in(&fb);
            auto tm = TraceManager2<Nonterminal, TraceID>::deserialize(in);
            std::clog << "Read in TraceManager from \'" << path << "\'" << std::endl
                      << "with Trace#: " << tm->size() << std::endl;
            fb.close();
            return tm;
        } else {
            std::clog << "No such file: \'" << path << "\'" << std::endl;
            return std::shared_ptr<TraceManager2<Nonterminal, TraceID>>();
        }
    }

    template<typename Nonterminal, typename TraceID>
    void add_hypergraph_to_trace(TraceManagerPtr<Nonterminal, TraceID> traceManager
                                 , HypergraphPtr<Nonterminal> hypergraph
                                 , Element<Node<Nonterminal>> root
                                 , double frequency = 1.0) {
        traceManager->create(0L, hypergraph, root, frequency);
    };

    template<typename Nonterminal, typename TraceID>
    std::shared_ptr<TraceManager2<Nonterminal, TraceID>> fool_cython_unwrap(TraceManagerPtr<Nonterminal, TraceID> tmp) {
        return tmp;
    };
}


#endif //STERMPARSER_TRACEMANAGER_H
