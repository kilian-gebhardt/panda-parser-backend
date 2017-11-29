//
// Created by kilian on 29/11/17.
//

#ifndef STERMPARSER_GENETICCROSSER_H
#define STERMPARSER_GENETICCROSSER_H
#include "TrainingCommon.h"
#include "LatentAnnotation.h"
#include "AnnotationProjection.h"

namespace Trainer {
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
            sumDimensions1.fill(0);
            sumDimensions2.fill(0);
            size_t dimIndex1 = 0;
            size_t dimIndex2 = 0;

            // GeneticCrosser requires LHS to belong to LA1.
            sumDimensions1[dimIndex1++] = 0;

            bool lhsIsFirst{keepFromOne[edge->get_target()->get_label()]};
            if(rank != edge->get_sources().size() + 1) {
                std::cerr << "Rank does not correspond to edge! " << rank << " vs. " << edge->get_sources().size();
                abort();
            }

            // The following loop guarantees the initialization of sumDimension1/2, because
            // a) edge->get_sources().size() == rank - 1
            // b) rank > numberInOne > 0  ==> rank >= 2
            for (size_t dim {0}; dim < edge->get_sources().size(); ++dim) {
//                #pragma GCC diagnostic push
//                #pragma GCC diagnostic ignored "-W"
                if (keepFromOne[edge->get_sources()[dim]->get_label()] ^ (!lhsIsFirst))
                    sumDimensions1[dimIndex1++] = (int) (dim + 1);
                else
                    sumDimensions2[dimIndex2++] = (int) (dim + 1);
//                #pragma GCC diagnostic pop
            }

            // calculate the probability to be distributed
            Eigen::Tensor<double, numberInOne> probabilityMass = weight1.sum(sumDimensions2);



            // Calculate the weightDistribution tensor
            Eigen::Tensor<double, rank - numberInOne> weightDistribution;

            if (std::abs(ruleSum) > std::exp(-30)) { // the rule has probability > 0

                RuleTensor<double> intermediate{RuleTensorRaw<double, rank>(weight2.dimensions())};
                intermediate = weight2;
                int sumDimCount = 0;
                for (int dim : sumDimensions1) {
                    const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> &inout{
                            dim == 0 ? outside2 : inside2};
                    const Element<Node<Nonterminal>> nt{dim == 0 ? edge->get_target() : edge->get_sources()[dim - 1]};
                    RuleTensorContractor tmult(inout.at(nt), dim - sumDimCount);
                    intermediate = boost::apply_visitor(tmult, intermediate);
                    ++sumDimCount;
                }

                weightDistribution = boost::get<RuleTensorRaw<double, rank - numberInOne>>(intermediate) / ruleSum;
            } else { // rule has probability 0
                // weightdistribution is an equal distribution over all annotations

                // determine how many entries there are
                size_t weightCount = 1;
                for (int i : sumDimensions2)
                    weightCount *= weight2.dimension(i);

                Eigen::array<Eigen::Index, rank - numberInOne> dimensions;
                for (size_t i = 0; i < sumDimensions2.size(); ++i)
                    dimensions[i] = weight2.dimension(sumDimensions2[i]);
                Eigen::Tensor<double, rank - numberInOne> equalDistribution(dimensions);
                equalDistribution.setConstant(1.0 / (double) weightCount);

                weightDistribution = equalDistribution;
            }

            // extend the tensors
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


            return probabilityMass.reshape(reshape1).broadcast(broadcast1)
                   *
                   (weightDistribution).reshape(reshape2).broadcast(broadcast2);

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
        OneDimensionalVectorCreator odvc(0);
        for(size_t ruleId = 0; ruleId < la1.ruleWeights->size(); ++ ruleId){
            ruleWeights.push_back(boost::apply_visitor(odvc, (*la1.ruleWeights)[ruleId]));
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


        return LatentAnnotation(nonterminalSplits
                                , std::move(rootWeights)
                                , std::make_unique<std::vector <RuleTensor<double>>>(std::move(ruleWeights))
        );

    }

}

#endif //STERMPARSER_GENETICCROSSER_H
