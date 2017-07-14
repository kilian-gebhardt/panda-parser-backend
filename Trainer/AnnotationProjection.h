//
// Created by Markus on 06.06.17.
//

#ifndef STERMPARSER_ANNOTATIONPROJECTION_H
#define STERMPARSER_ANNOTATIONPROJECTION_H

#include "TrainingCommon.h"
#include "LatentAnnotation.h"

namespace Trainer {

    static const double IO_PRECISION_DEFAULT = 0.000001;
    static const unsigned int IO_CYCLE_LIMIT_DEFAULT = 200;


    void check_rule_weight_for_consistency(
            const RuleTensor<double> &res
            , const std::vector<Trainer::WeightVector> &insides
            , const Trainer::WeightVector &outside
            , const Eigen::Tensor<double, 0> &normalization
            , std::ostringstream &rTstr
            , std::vector<double> dimens
            , double calc
    ) {
        double sum = 0;
        switch (res.which() + 1) {
            case 1: {
                const unsigned int ruleRank = 1;
                const RuleTensorRaw<double, ruleRank>& ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                     , ruleRank>>(
                        res
                );
                RuleTensorRaw<double, 0> sumvec = ruleTensor.sum();
                sum = sumvec(0);
                break;
            }
            case 2: {
                const unsigned int ruleRank = 2;
                const RuleTensorRaw<double, ruleRank>& ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                     , ruleRank>>(
                        res
                );
                RuleTensorRaw<double, 0> sumvec = ruleTensor.sum();
                sum = sumvec(0);
                break;
            }
            case 3: {
                const unsigned int ruleRank = 3;
                const RuleTensorRaw<double, ruleRank>& ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                     , ruleRank>>(
                        res
                );
                RuleTensorRaw<double, 0> sumvec = ruleTensor.sum();
                sum = sumvec(0);
                break;
            }
            default: {
                std::cerr << "Rules with such a fanout are not supported to be checked!";
            }
        }
        if (sum > 1) {
            std::cerr << "The sum of a rule was larger than 1: " << sum
                      << "  inside weights: \n";
            for (auto &inside : insides)
                std::cerr << " / " << inside << "(dim: " << inside.dimension(0) << ")";
            std::cerr << "\n  outside weights: " << outside
                      << "  rule tensor:\n" << rTstr.str() << "\n"
                      << "  normalization: " << normalization
                      << "  calc: " << calc
                      << "  dimensions: ";

            for (const auto d : dimens)
                std::cerr << " " << d;
            std::cerr << std::endl;
        }
    }


    template<typename Nonterminal>
    HypergraphPtr<Nonterminal> hypergraph_from_grammar(GrammarInfo2 grammarInfo) {
        // build HG
        MAPTYPE<size_t, bool> nodes;
        std::vector<size_t> nLabels{};

        size_t labelCounter = 0;
        std::vector<size_t> eLabels{};


        for (const auto rule : grammarInfo.rule_to_nonterminals) {
            eLabels.push_back(labelCounter++);
            for (size_t nont : rule) {
                if (nodes.count(nont) == 0) {
                    nodes[nont] = true;
                    nLabels.push_back(nont);
                }
            }
        }

        const auto nLabelsPtr = std::make_shared<const std::vector<size_t>>(nLabels);
        const auto eLabelsPtr = std::make_shared<const std::vector<size_t>>(eLabels);

        HypergraphPtr<Nonterminal> hg = std::make_shared<Hypergraph<Nonterminal>>(nLabelsPtr, eLabelsPtr);

        MAPTYPE<size_t, Element<Node<size_t>>> nodeElements;
        for (size_t ruleNr = 0; ruleNr < grammarInfo.rule_to_nonterminals.size(); ++ruleNr) {
            std::vector<size_t> rule = grammarInfo.rule_to_nonterminals[ruleNr];
            std::vector<Element<Node<Nonterminal>>> rhs{};
            for (size_t i = 0; i < rule.size(); ++i) {
                size_t nont = rule[i];
                if (nodeElements.count(nont) == 0) {
                    nodeElements.insert(MAPTYPE<size_t, Element<Node<size_t>>>::value_type(nont, hg->create(nont)));
                }
                if (i != 0)
                    rhs.push_back(nodeElements.at(nont));
            }
            hg->add_hyperedge(ruleNr, nodeElements.at(rule[0]), rhs);
        }

        return hg;
    }


    template<typename Nonterminal>
    void io_weights_for_grammar(
            const HypergraphPtr<Nonterminal> hg
            , const LatentAnnotation &annotation
            , const Element<Node<Nonterminal>> initialNode
            , MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> &insideWeights
            , MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> &outsideWeights
            , const double ioPrecision = IO_PRECISION_DEFAULT
            , const unsigned int ioCycleLimit = IO_CYCLE_LIMIT_DEFAULT
    ) {

        std::shared_ptr<Trainer::TraceManager2<Nonterminal, size_t>> tMPtr = std::make_shared<Trainer::TraceManager2<
                Nonterminal
                , size_t>>(hg->get_node_labels(), hg->get_edge_labels());
        tMPtr->create(1, hg, initialNode);

        for (const auto n : *hg) {
            insideWeights[n] = Trainer::WeightVector{annotation.nonterminalSplits.at(n->get_label_id())};
            outsideWeights[n] = Trainer::WeightVector{annotation.nonterminalSplits.at(n->get_label_id())};
        }

        tMPtr->set_io_precision(ioPrecision);
        tMPtr->set_io_cycle_limit(ioCycleLimit);
        (*tMPtr)[0].io_weights_fixpoint_la(
                *(annotation.ruleWeights)
                , annotation.rootWeights
                , insideWeights
                , outsideWeights
        );

    }


    template<typename Nonterminal>
    LatentAnnotation project_annotation(
            const LatentAnnotation &annotation
            , const GrammarInfo2 &grammarInfo
            , const double ioPrecision = IO_PRECISION_DEFAULT
            , const size_t ioCycleLimit = IO_CYCLE_LIMIT_DEFAULT
    ) {

        HypergraphPtr<Nonterminal> hg = hypergraph_from_grammar<Nonterminal>(grammarInfo);

        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> insideWeights;
        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> outsideWeights;
        io_weights_for_grammar<Nonterminal>(
                hg
                , annotation
                , hg->get_node_by_label(grammarInfo.start)
                , insideWeights
                , outsideWeights
                , ioPrecision
                , ioCycleLimit
        );


        // do projection
        auto projRuleWeights = std::vector<RuleTensor<double>>();

        for (const auto &edge : *hg->get_edges()) {
            size_t ruleId = edge->get_label();
            const std::vector<size_t> &rule = grammarInfo.rule_to_nonterminals[ruleId];
            const auto &ruleVariant = (*(annotation.ruleWeights))[ruleId];

            const auto normalisationCalc = insideWeights[edge->get_target()].contract(
                    outsideWeights[edge->get_target()]
                    , Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
            );
            Eigen::Tensor<double, 0> normalisationVector = normalisationCalc;

            if (normalisationVector(0) == 0) { // either in(A)=0 or out(A)=0
                // weight of rule is defined to be equally distributed
                size_t norm = grammarInfo.normalizationGroups[rule[0]].size();
                switch (rule.size()) {
                    case 1: {
                        RuleTensorRaw<double, 1> res(1);
                        res.setValues({1.0 / (double) norm});

//                        std::ostringstream rTstr;
//                        std::vector<double> dimens{res.dimension(0)};
//                        std::vector<Trainer::WeightVector> insides{};
//                        check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, 1);

                        projRuleWeights.push_back(res);
                        continue;
                    }
                    case 2: {
                        RuleTensorRaw<double, 2> res(1, 1);
                        res.setValues({{1.0 / (double) norm}});

//                        std::ostringstream rTstr;
//                        std::vector<double> dimens{res.dimension(0), res.dimension(1)};
//                        std::vector<Trainer::WeightVector> insides{};
//                        check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, 1);

                        projRuleWeights.push_back(res);
                        continue;
                    }
                    case 3: {
                        RuleTensorRaw<double, 3> res(1, 1, 1);
                        res.setValues({{{1.0 / (double) norm}}});

//                        std::ostringstream rTstr;
//                        std::vector<double> dimens{res.dimension(0), res.dimension(1), res.dimension(2)};
//                        std::vector<Trainer::WeightVector> insides{};
//                        check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, 1);

                        projRuleWeights.push_back(res);
                        continue;
                    }
                    case 4: {
                        RuleTensorRaw<double, 4> res(1, 1, 1, 1);
                        res.setValues({{{{1.0 / (double) norm}}}});

//                        std::ostringstream rTstr;
//                        std::vector<double> dimens{res.dimension(0), res.dimension(1), res.dimension(2), res.dimension(3)};
//                        std::vector<Trainer::WeightVector> insides{};
//                        check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, 1);

                        projRuleWeights.push_back(res);
                        continue;
                    }
                    default: {
                        std::cerr << "Rules with " << rule.size() - 1 << " RHS nonterminals are not supported.";
                        abort();
                    }
                }

            }

            switch (rule.size()) {
                case 1: {
                    constexpr unsigned ruleRank{1};
                    const RuleTensorRaw<double, ruleRank>& ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                         , ruleRank>>(
                            ruleVariant
                    );
                    auto sumWeight = ruleTensor.contract(
                            outsideWeights.at(edge->get_target())
                            , Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
                    );
                    RuleTensorRaw<double, 0> calc = sumWeight;
                    RuleTensorRaw<double, 1> res(1);
                    res.setValues({calc(0) / normalisationVector(0)});

//                    std::ostringstream rTstr;
//                    rTstr << ruleTensor;
//                    std::vector<double> dimens{ruleTensor.dimension(0)};
//                    std::vector<Trainer::WeightVector> insides{};
//                    check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, calc(0));

                    projRuleWeights.push_back(res);
                    break;
                }
                case 2: {
                    constexpr unsigned ruleRank{2};
                    const RuleTensorRaw<double, ruleRank>& ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                         , ruleRank>>(
                            ruleVariant
                    );
                    auto sumWeight = ruleTensor.contract(
                                    insideWeights.at(edge->get_sources()[0])
                                    , Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(1, 0)}
                            )
                            .contract(
                                    outsideWeights.at(edge->get_target()), Eigen::array<Eigen::IndexPair<long>, 1>{
                                            Eigen::IndexPair<long>(0, 0)}
                            );
                    RuleTensorRaw<double, 0> calc = sumWeight;
                    RuleTensorRaw<double, 2> res(1, 1);
                    res.setValues({{calc(0) / normalisationVector(0)}});

//                    std::ostringstream rTstr;
//                    rTstr << ruleTensor;
//                    std::vector<double> dimens{ruleTensor.dimension(0), ruleTensor.dimension(1)};
//                    std::vector<Trainer::WeightVector> insides{insideWeights[nodeElements.at(rule.at(1))]};
//                    check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, calc(0));

                    projRuleWeights.push_back(res);
                    break;
                }
                case 3: {
                    constexpr unsigned ruleRank{3};
                    const RuleTensorRaw<double, ruleRank>& ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                         , ruleRank>>(
                            ruleVariant
                    );
                    auto sumWeight = ruleTensor.contract(
                                    insideWeights.at(edge->get_sources()[1])
                                    , Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(2, 0)}
                            )
                            .contract(
                                    insideWeights.at(edge->get_sources()[0]), Eigen::array<Eigen::IndexPair<long>, 1>{
                                            Eigen::IndexPair<long>(1, 0)}
                            )
                            .contract(
                                    outsideWeights.at(edge->get_target()), Eigen::array<Eigen::IndexPair<long>, 1>{
                                            Eigen::IndexPair<long>(0, 0)}
                            );
                    RuleTensorRaw<double, 0> calc = sumWeight;
                    RuleTensorRaw<double, 3> res(1, 1, 1);
                    res.setValues({{{calc(0) / normalisationVector(0)}}});

//                    std::ostringstream rTstr;
//                    rTstr << ruleTensor;
//                    std::vector<double> dimens{ruleTensor.dimension(0), ruleTensor.dimension(1), ruleTensor.dimension(2)};
//                    std::vector<Trainer::WeightVector> insides{insideWeights[nodeElements.at(rule.at(1))], insideWeights[nodeElements.at(rule.at(2))]};
//                    check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, calc(0));

                    projRuleWeights.push_back(res);
                    break;
                }
                case 4: {
                    constexpr unsigned ruleRank{4};
                    const RuleTensorRaw<double, ruleRank>& ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                         , ruleRank>>(
                            ruleVariant
                    );
                    auto sumWeight = ruleTensor.contract(
                                    insideWeights.at(edge->get_sources()[2])
                                    , Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(3, 0)}
                            )
                            .contract(
                                    insideWeights.at(edge->get_sources()[1]), Eigen::array<Eigen::IndexPair<long>, 1>{
                                            Eigen::IndexPair<long>(2, 0)}
                            )
                            .contract(
                                    insideWeights.at(edge->get_sources()[0]), Eigen::array<Eigen::IndexPair<long>, 1>{
                                            Eigen::IndexPair<long>(1, 0)}
                            )
                            .contract(
                                    outsideWeights.at(edge->get_target()), Eigen::array<Eigen::IndexPair<long>, 1>{
                                            Eigen::IndexPair<long>(0, 0)}
                            );
                    RuleTensorRaw<double, 0> calc = sumWeight;
                    RuleTensorRaw<double, 4> res(1, 1, 1, 1);
                    res.setValues({{{{calc(0) / normalisationVector(0)}}}});

//                    std::ostringstream rTstr;
//                    rTstr << ruleTensor;
//                    std::vector<double> dimens{ruleTensor.dimension(0), ruleTensor.dimension(1), ruleTensor.dimension(2), ruleTensor.dimension(3)};
//                    std::vector<Trainer::WeightVector> insides{insideWeights[nodeElements.at(rule.at(1))], insideWeights[nodeElements.at(rule.at(2))], insideWeights[nodeElements.at(rule.at(3))]};
//                    check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, calc(0));

                    projRuleWeights.push_back(res);
                    break;
                }
                default:
                    std::cerr << "Rules with " << rule.size() - 1 << " RHS nonterminals are not supported";
                    abort();
            }
        }

        // calculate root weights
        WeightVector root(1);
        Eigen::Tensor<double, 0> rootval = annotation.rootWeights.sum();
        root.setValues({rootval(0)});


        return LatentAnnotation(
                std::vector<size_t>(annotation.nonterminalSplits.size(), 1)
                , std::move(root)
                , std::make_unique<std::vector<RuleTensor<double>>>(projRuleWeights));
    }




    template <typename Nonterminal>
    struct TensorMultiplyer : boost::static_visitor<RuleTensor<double>> {
        const RuleTensor<double>& factor;

        TensorMultiplyer(
            const RuleTensor<double>& factor
        )
        :
            factor(factor)
        {};

        template<int rank>
        typename std::enable_if<rank != 1, RuleTensorRaw<double, rank-1>>::type
        operator()(const RuleTensorRaw<double, rank>& tensor) const {
            const Eigen::Tensor<double, 1>& fac = boost::get<Eigen::Tensor<double, 1>>(factor);
            RuleTensorRaw<double, rank-1> result = tensor.contract(
                    fac, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(rank - 1, 0)});
            return result;
        }

        RuleTensorRaw<double, 1>
        operator()(const RuleTensorRaw<double, 1>& tensor) const {
            std::cerr << "TensorMultiplyer can only handle Tensors of at least dimension 2!";
            abort();
        }
    };


    template <typename Nonterminal>
    struct RuleSummer : boost::static_visitor<double> {

        const Element<HyperEdge<Nonterminal>> edge;
        const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> inside;
        const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> outside;

        RuleSummer(
            const Element<HyperEdge<Nonterminal>> edge
            , const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> &inside
            , const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> &outside
        )
        :
            edge(edge)
            , inside(inside)
            , outside(outside)
        {};

        template<int rank>
        double operator()(const RuleTensorRaw<double, rank> &weight) const {

            RuleTensor<double> intermediate = weight;
            for(int dim = rank - 1; dim > 0 ; --dim) {
                TensorMultiplyer<Nonterminal> tmult(inside.at(edge->get_sources()[dim - 1]));
                intermediate = boost::apply_visitor(tmult, intermediate);
            }

            const RuleTensorRaw<double, 1>& withoutInside = boost::get<RuleTensorRaw<double, 1>>(intermediate);
            RuleTensorRaw<double, 0> sum = withoutInside
                        .contract(outside.at(edge->get_target())
                                , Eigen::array<Eigen::IndexPair<long>, 1>{
                                        Eigen::IndexPair<long>(0, 0)}
                        );

            return sum(0);
        }
    };





    template <typename Nonterminal, int numberInOne>
    struct GeneticCrosser : boost::static_visitor<RuleTensor<double>> {

        const Element<HyperEdge<Nonterminal>> edge;
        const std::vector<bool>& keepFromOne;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> inside2;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> outside2;
        double ruleSum;
        const RuleTensor<double> ruleWeight2;

        /*
         * Assumes that left-hand side of rule belongs to weight1!
         */
        GeneticCrosser(const Element<HyperEdge<Nonterminal>> edge
            , const std::vector<bool>& keepFromOne
            , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &inside2
            , const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> &outside2
            , double rS
            , RuleTensor<double> ruleWeight2
        )
        : edge(edge)
        , keepFromOne(keepFromOne)
        , inside2(inside2)
        , outside2(outside2)
        , ruleSum(rS)
        , ruleWeight2(ruleWeight2) {};


        /*
         * Assumes that left-hand side of rule belongs to weight1!
         */
        template<int rank>
        typename std::enable_if<rank >= numberInOne, Eigen::Tensor<double, rank>>::type
        operator()(const Eigen::Tensor<double, rank> &weight1) const {

            const Eigen::Tensor<double, rank>& weight2 {boost::get<Eigen::Tensor<double, rank>>(ruleWeight2)};

            if (numberInOne == 0){
                RuleTensorRaw<double, rank> result;
                result = weight2;
                return result;
            }
            if (rank - numberInOne == 0){
                RuleTensorRaw<double, rank> result;
                result = weight1;
                return result;
            }

            
            Eigen::array<int, numberInOne> sumDimensions1;
            Eigen::array<int, rank - numberInOne> sumDimensions2;
            size_t dimIndex1 = 0;
            size_t dimIndex2 = 0;

            // GeneticCrosser requires LHS to belong to LA1.
            sumDimensions1[dimIndex1++] = 0;

            bool lhsIsFirst{keepFromOne[edge->get_target()->get_label()]};
            if(rank != edge->get_sources().size() + 1) {
                std::cerr << "Rank does not correspond to edge! " << rank << " vs. " << edge->get_sources().size();
                abort();
            }
            for (int dim = 0; dim < edge->get_sources().size(); ++dim) {
                if (keepFromOne[edge->get_sources()[dim]->get_label()] ^ (!lhsIsFirst))
                    sumDimensions1[dimIndex1++] = dim + 1;
                else
                    sumDimensions2[dimIndex2++] = dim + 1;


            }

            // calculate the probability to be distributed
            RuleTensorRaw<double, numberInOne> probabilityMass = weight1.sum(sumDimensions2);

            // calculate the factor

            Eigen::array<int, rank> reshapeDimensions;
            Eigen::array<int, rank> broadcastDimensions;
            reshapeDimensions[0] = weight2.dimension(0);
            broadcastDimensions[0] = 1;
            for (unsigned int i = 1; i < rank; ++i) {
                reshapeDimensions[i] = 1;
                broadcastDimensions[i] = weight2.dimension(i);
            }

            RuleTensorRaw<double, rank> weightDistribution(weight2.dimensions());

            weightDistribution
                    = outside2.at(edge->get_target()).reshape(reshapeDimensions)
                              .broadcast(broadcastDimensions);

            weightDistribution = weightDistribution * weight2;

            for (unsigned int lhsNumber = 1; lhsNumber < rank; ++lhsNumber) {
                reshapeDimensions[0] = 1;
                broadcastDimensions[0] = weight2.dimension(0);
                for (unsigned int i = 1; i < rank; ++i) {
                    if (i == lhsNumber) {
                        reshapeDimensions[i] = weight2.dimension(i);
                        broadcastDimensions[i] = 1;
                    } else {
                        reshapeDimensions[i] = 1;
                        broadcastDimensions[i] = weight2.dimension(i);
                    }
                }

                weightDistribution
                        = weightDistribution
                          * inside2.at(edge->get_sources()[lhsNumber-1]).reshape(reshapeDimensions)
                                  .broadcast(broadcastDimensions);
            }

            RuleTensorRaw<double, rank - numberInOne> weightSum
                    = weightDistribution.sum(sumDimensions1) / ruleSum;



            // extend the tensors and multiply pointwise
            Eigen::array<int, rank> reshape1;
            Eigen::array<int, rank> reshape2;
            Eigen::array<int, rank> broadcast1;
            Eigen::array<int, rank> broadcast2;
            for (unsigned int i = 0; i < rank; ++i) {
                reshape1[i] = weight2.dimension(i);
                reshape2[i] = weight1.dimension(i);
                broadcast1[i] = 1;
                broadcast2[i] = 1;
            }

            for (int i : sumDimensions2) {
                reshape1[i] = 1;
                broadcast1[i] = weight2.dimension(i);
            }
            for (int i : sumDimensions1) {
                reshape2[i] = 1;
                broadcast2[i] = weight1.dimension(i);
            }

            return probabilityMass.reshape(reshape1).broadcast(broadcast1)
                    *
                    weightSum.reshape(reshape2).broadcast(broadcast2);

        }


        /*
         * This function is here to suit the implementation, but is never called.
         */
        template<int rank>
        typename std::enable_if<rank < numberInOne, Eigen::Tensor<double, rank>>::type
        operator()(const Eigen::Tensor<double, rank> &weight1) const {
            std::cerr << "Tried to genetically cross a tensor of rank " << rank << " with " << numberInOne << " entries! Aborting!";
            abort();
        };

    };







// Genetic algorithms: Mix 2 latent annotations
    template <typename Nonterminal>
    LatentAnnotation mix_annotations(
            const LatentAnnotation &la1
            , const LatentAnnotation &la2
            , const GrammarInfo2 &info
            , const std::vector<bool>& keepFromOne
            , const double ioPrecision = IO_PRECISION_DEFAULT
            , const unsigned int ioCycleLimit = IO_CYCLE_LIMIT_DEFAULT
    ) {

        // check that la1 and la2 are compatible
        assert(la1.nonterminalSplits.size() == la2.nonterminalSplits.size());
        assert(la1.ruleWeights->size() == la2.ruleWeights->size());


        HypergraphPtr<Nonterminal> hg = hypergraph_from_grammar<Nonterminal>(info);

        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> inside1;
        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> outside1;
        io_weights_for_grammar<Nonterminal>(
                hg
                , la1
                , hg->get_node_by_label(info.start)
                , inside1
                , outside1
                , ioPrecision
                , ioCycleLimit
        );

        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> inside2;
        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> outside2;
        io_weights_for_grammar<Nonterminal>(
                hg
                , la2
                , hg->get_node_by_label(info.start)
                , inside2
                , outside2
                , ioPrecision
                , ioCycleLimit
        );


        // adapting the new splits:
        std::vector<size_t> nonterminalSplits(la1.nonterminalSplits.size());
        for(size_t i = 0; i < la1.nonterminalSplits.size(); ++i){
            nonterminalSplits[i] = keepFromOne[i]? la1.nonterminalSplits[i] : la2.nonterminalSplits[i];
        }

        std::vector <RuleTensor<double>> ruleWeights(la1.ruleWeights->size());
        // calculate new weight for each rule
        for (auto edge : *hg->get_edges() ){

            // ensure assumption: weight1 is the weight of the LHS
            bool lhsIsFirst{keepFromOne[edge->get_target()->get_label()]};

            const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& inside {lhsIsFirst ? inside2 : inside1};
            const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outside {lhsIsFirst ? outside2 : outside1};
            const RuleTensor<double>& weight1 {lhsIsFirst ? (*la1.ruleWeights)[edge->get_label()] : (*la2.ruleWeights)[edge->get_label()]};
            const RuleTensor<double>& weight2 {lhsIsFirst ? (*la2.ruleWeights)[edge->get_label()] : (*la1.ruleWeights)[edge->get_label()]};


            // TODO! calculate the sum of the whole rule:
            RuleSummer<Nonterminal> ruleSummer(edge, inside, outside);
            double ruleSum = boost::apply_visitor(ruleSummer, weight2);


            unsigned int countFirsts{1}; // LHS always belongs to first
//            if(keepFromOne[edge->get_target()->get_label()])
//                ++countFirsts;
            for (Element<Node<Nonterminal>> choice : edge->get_sources())
                if (keepFromOne[choice->get_label()] ^ (!lhsIsFirst)) // XOR ensures first being the one with LHS
                    ++countFirsts;

            switch (countFirsts){
                case 0:
                    if (lhsIsFirst) {
                        ruleWeights[edge->get_label()] = (*la2.ruleWeights)[edge->get_label()];
                    } else {
                        ruleWeights[edge->get_label()] = (*la1.ruleWeights)[edge->get_label()];
                    }
                    break;
                case 1: {
                    GeneticCrosser<Nonterminal, 1> crosser(
                            edge
                            , keepFromOne
                            , inside
                            , outside
                            , ruleSum
                            , weight2
                    );
                    ruleWeights[edge->get_label()]
                            = boost::apply_visitor(
                            crosser
                            , weight1
                    );
                    break;
                }
                case 2: {
                    GeneticCrosser<Nonterminal, 2> crosser(
                            edge
                            , keepFromOne
                            , inside
                            , outside
                            , ruleSum
                            , weight2
                    );
                    ruleWeights[edge->get_label()]
                            = boost::apply_visitor(
                            crosser
                            , weight1
                    );
                    break;
                }
                case 3: {
                    GeneticCrosser<Nonterminal, 3> crosser(
                            edge
                            , keepFromOne
                            , inside
                            , outside
                            , ruleSum
                            , weight2
                    );
                    ruleWeights[edge->get_label()]
                            = boost::apply_visitor(
                            crosser
                            , weight1
                    );
                    break;
                }
                case 4: {
                    GeneticCrosser<Nonterminal, 4> crosser(
                            edge
                            , keepFromOne
                            , inside
                            , outside
                            , ruleSum
                            , weight2
                    );
                    ruleWeights[edge->get_label()]
                            = boost::apply_visitor(
                            crosser
                            , weight1
                    );
                    break;
                }
                default:
                    std::cerr << "Genetic crosser can only handle RHS up to size 3!";
                    abort();
            }

        }

        WeightVector rootWeights;
        if(keepFromOne[info.start])
            rootWeights = la1.rootWeights;
        else
            rootWeights = la2.rootWeights;


        return LatentAnnotation(nonterminalSplits
                                , std::move(rootWeights)
                                , std::make_unique<std::vector <RuleTensor<double>>>(ruleWeights)
        );

    }


}


#endif //STERMPARSER_ANNOTATIONPROJECTION_H
