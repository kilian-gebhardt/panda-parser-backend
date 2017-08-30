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
    HypergraphPtr<Nonterminal> hypergraph_from_grammar(const GrammarInfo2& grammarInfo) {
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
            const HypergraphPtr<Nonterminal>& hg
            , const LatentAnnotation &annotation
            , const Element<Node<Nonterminal>>& initialNode
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
            insideWeights[n] = Trainer::WeightVector(annotation.nonterminalSplits.at(n->get_label_id()));
            outsideWeights[n] = Trainer::WeightVector(annotation.nonterminalSplits.at(n->get_label_id()));
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



        auto projRuleWeights = std::vector<RuleTensor<double>>();
        projRuleWeights.reserve(hg->get_edges().lock()->size());
        OneDimensionalVectorCreator odc(0.0);
        for(size_t ruleId = 0; ruleId < (*annotation.ruleWeights).size(); ++ruleId)
            projRuleWeights.push_back(boost::apply_visitor(odc, (*annotation.ruleWeights)[ruleId]));



        // do projection
        for (const auto &edge : *hg->get_edges().lock()) {
            size_t ruleId = edge->get_label();
            const std::vector<size_t> &rule = grammarInfo.rule_to_nonterminals[ruleId];
            const auto &ruleVariant = (*(annotation.ruleWeights))[ruleId];

            const auto normalisationCalc = insideWeights[edge->get_target()].contract(
                    outsideWeights[edge->get_target()]
                    , Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
            );
            Eigen::Tensor<double, 0> normalisationVector = normalisationCalc;

            if (std::abs(normalisationVector(0)) < std::exp(-50)) { // normalization is 0, apply a defalut value
                size_t norm = grammarInfo.normalizationGroups[rule[0]].size();
                OneDimensionalVectorCreator nvc(1.0/(double)norm);

                projRuleWeights[ruleId]  = boost::apply_visitor(nvc, ruleVariant);
                continue;

            }

            InsideOutsideMultiplierAndNormalizer<Nonterminal> ioMultiplier(edge, insideWeights, outsideWeights, normalisationVector(0));
            projRuleWeights[ruleId] = boost::apply_visitor(ioMultiplier, ruleVariant);
        }

        // calculate root weights
        WeightVector root(1);
        Eigen::Tensor<double, 0> rootval = annotation.rootWeights.sum();
        root.setValues({rootval(0)});


        LatentAnnotation result(
                std::vector<size_t>(annotation.nonterminalSplits.size(), 1)
                , std::move(root)
                , std::make_unique<std::vector<RuleTensor<double>>>(std::move(projRuleWeights)));


        // make the LA proper
        // (this is needed, since inside or outside values of some nonterminals might be 0)
        result.make_proper(grammarInfo);

        return result;
    }


    template <typename Nonterminal, int numberInOne>
    struct GeneticCrosser : boost::static_visitor<RuleTensor<double>> {

        const Element<HyperEdge<Nonterminal>> &edge;
        const std::vector<bool> &keepFromOne;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &inside2;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &outside2;
        double ruleSum;
        const RuleTensor<double> &ruleWeight2;

        /*
         * Assumes that left-hand side of rule belongs to weight1!
         */
        GeneticCrosser(const Element<HyperEdge<Nonterminal>> &edge
            , const std::vector<bool> &keepFromOne
            , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &inside2
            , const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector> &outside2
            , const double rS
            , const RuleTensor<double> &ruleWeight2
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
        typename std::enable_if<(rank > numberInOne), RuleTensorRaw<double, rank>>::type
        operator()(const Eigen::Tensor<double, rank> &weight1) const {

            const Eigen::Tensor<double, rank>& weight2 {boost::get<Eigen::Tensor<double, rank>>(ruleWeight2)};

            if (numberInOne == 0){
                RuleTensorRaw<double, rank> result(weight2.dimensions());
                result = weight2;
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
            Eigen::Tensor<double, numberInOne> probabilityMass = weight1.sum(sumDimensions2);


            // calculate the factor
            Eigen::array<int, rank> reshapeDimensions;
            Eigen::array<int, rank> broadcastDimensions;
            reshapeDimensions[0] = weight2.dimension(0);
            broadcastDimensions[0] = 1;
            for (unsigned int i = 1; i < rank; ++i) {
                reshapeDimensions[i] = 1;
                broadcastDimensions[i] = weight2.dimension(i);
            }

            RuleTensor<double> intermediate{RuleTensorRaw<double, rank>(weight2.dimensions())};
            intermediate = weight2;
            int sumDimCount = 0;
            for(int dim : sumDimensions1) {
                const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& inout {dim == 0 ? outside2 : inside2};
                const Element<Node<Nonterminal>> nt {dim == 0 ? edge->get_target() : edge->get_sources()[dim - 1]};
                RuleTensorContractor tmult(inout.at(nt), dim - sumDimCount);
                intermediate = boost::apply_visitor(tmult, intermediate);
                ++sumDimCount;
            }

            Eigen::Tensor<double, rank - numberInOne> &weightDistribution
                    = boost::get<RuleTensorRaw<double, rank - numberInOne>>(intermediate);


            // extend the tensors and multiply pointwise
            Eigen::array<int, rank> reshape1;
            Eigen::array<int, rank> reshape2;
            Eigen::array<int, rank> broadcast1;
            Eigen::array<int, rank> broadcast2;
            for (unsigned int i = 0; i < rank; ++i) {
                reshape1[i] = weight1.dimension(i);
                reshape2[i] = weight2.dimension(i);
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


            // in case inside- or outside-values are 0, the ruleSum is equally distributed to weightdistribution
            Eigen::Tensor<double, rank - numberInOne> equalDistribution(weightDistribution.dimensions());
            equalDistribution.setConstant(ruleSum);
            // determine how many entries there are
            size_t weightCount = 1;
            for(int i : sumDimensions1)
                weightCount *= weight2.dimension(i);

            equalDistribution = equalDistribution / ((double)weightCount);

            Eigen::Tensor<bool, rank - numberInOne> isWeightNonZero = weightDistribution.unaryExpr([](double x){return x > std::exp(-50);});

            weightDistribution = isWeightNonZero.select(weightDistribution, equalDistribution);

            return probabilityMass.reshape(reshape1).broadcast(broadcast1)
                   *
                   (weightDistribution / ruleSum).reshape(reshape2).broadcast(broadcast2);

        }

        /*
         * Handling of the special case where all LAs from 1 are chosen. Just returns weight1.
         */
        template<int rank>
        typename std::enable_if<rank == numberInOne, RuleTensorRaw<double, rank>>::type
        operator()(const Eigen::Tensor<double, rank> &weight1) const {

            RuleTensorRaw<double, rank> result(weight1.dimensions());
            result = weight1;
            return result;
        }


        /*
         * This function is here to suit the implementation, but is never called.
         */
        template<int rank>
        typename std::enable_if<(rank < numberInOne), RuleTensorRaw<double, rank>>::type
        operator()(const Eigen::Tensor<double, rank>) const {
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


        // prepare the new ruleWeight vector by initializing with the correct ranks
        std::vector<RuleTensor<double>> ruleWeights;
        ruleWeights.reserve(la1.ruleWeights->size());
        for(size_t ruleId = 0; ruleId < la1.ruleWeights->size(); ++ ruleId){
            switch (info.rule_to_nonterminals[ruleId].size()) {
                case 1:
                    ruleWeights.push_back(RuleTensorRaw<double, 1>(1));
                    break;
                case 2:
                    ruleWeights.push_back(RuleTensorRaw<double, 2>(1,1));
                    break;
                case 3:
                    ruleWeights.push_back(RuleTensorRaw<double, 3>(1,1,1));
                    break;
                case 4:
                    ruleWeights.push_back(RuleTensorRaw<double, 4>(1,1,1,1));
                    break;
                case 5:
                    ruleWeights.push_back(RuleTensorRaw<double, 5>(1,1,1,1,1));
                    break;
                default:
                    std::cerr << "Cannot create empty rule tensor for rules with LHS > 4! (Requested: " << info.rule_to_nonterminals[ruleId].size() << ")\n";
                    abort();
            }
        }


        // calculate new weight for each rule
        for (auto edge : *hg->get_edges().lock()){

            // ensure assumption: weight1 is the weight of the LHS
            bool lhsIsFirst{keepFromOne[edge->get_target()->get_label()]};

            const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& inside {lhsIsFirst ? inside2 : inside1};
            const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outside {lhsIsFirst ? outside2 : outside1};
            const RuleTensor<double>& weight1 {lhsIsFirst ? (*la1.ruleWeights)[edge->get_label()] : (*la2.ruleWeights)[edge->get_label()]};
            const RuleTensor<double>& weight2 {lhsIsFirst ? (*la2.ruleWeights)[edge->get_label()] : (*la1.ruleWeights)[edge->get_label()]};


            // calculate the sum of the whole rule:
            RuleSummerIO<Nonterminal> ruleSummer(edge, inside, outside);
            double ruleSum = boost::apply_visitor(ruleSummer, weight2);


            unsigned int countFirsts{1}; // LHS always belongs to first
//            if(keepFromOne[edge->get_target()->get_label()])
//                ++countFirsts;
            for (Element<Node<Nonterminal>> choice : edge->get_sources())
                if (keepFromOne[choice->get_label()] ^ (!lhsIsFirst)) // XOR ensures first being the one with LHS
                    ++countFirsts;

            switch (countFirsts){
                case 0:
                    ruleWeights[edge->get_label()] = weight2;
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


        long noRootWeights = 0;
        if(keepFromOne[info.start])
            noRootWeights = la1.rootWeights.dimension(0);
        else
            noRootWeights = la2.rootWeights.dimension(0);
        WeightVector rootWeights(noRootWeights);
        if(keepFromOne[info.start])
            for(int i = 0; i < noRootWeights; ++i)
                rootWeights[i] = la1.rootWeights[i];
        else
            for(int i = 0; i < noRootWeights; ++i)
                rootWeights[i] = la2.rootWeights[i];


//        // Debug: compute the size of the LA in doubles:
//        unsigned long laSize {0};
//        for (const RuleTensor<double>& rw : ruleWeights) {
//            SizeVisitor sizeVisitor;
//            laSize += boost::apply_visitor(sizeVisitor, rw);
//        }
//        std::cerr << "Genetic[debug]: Size of crossed LA: " << laSize << std::endl;

        return LatentAnnotation(nonterminalSplits
                                , std::move(rootWeights)
                                , std::make_unique<std::vector <RuleTensor<double>>>(ruleWeights)
        );

    }


}


#endif //STERMPARSER_ANNOTATIONPROJECTION_H
