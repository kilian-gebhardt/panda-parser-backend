//
// Created by Markus on 06.06.17.
//

#ifndef STERMPARSER_ANNOTATIONPROJECTION_H
#define STERMPARSER_ANNOTATIONPROJECTION_H

#include "TrainingCommon.h"
#include "LatentAnnotation.h"

namespace Trainer {


    void check_rule_weight_for_consistency(
            const RuleTensor<double>& res
            , const std::vector<Trainer::WeightVector>& insides
            , const Trainer::WeightVector& outside
            , const Eigen::Tensor<double, 0>& normalization
            , std::ostringstream& rTstr
            , std::vector<double> dimens
            , double calc
    ){
        double sum = 0;
        switch (res.which()+1){
            case 1: {
                const unsigned int ruleRank = 1;
                const RuleTensorRaw<double, ruleRank> ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                     , ruleRank>>(
                        res
                );
                RuleTensorRaw<double, 0> sumvec = ruleTensor.sum();
                sum = sumvec(0);
                break;
            }
            case 2: {
                const unsigned int ruleRank = 2;
                const RuleTensorRaw<double, ruleRank> ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                     , ruleRank>>(
                        res
                );
                RuleTensorRaw<double, 0> sumvec = ruleTensor.sum();
                sum = sumvec(0);
                break;
            }
            case 3: {
                const unsigned int ruleRank = 3;
                const RuleTensorRaw<double, ruleRank> ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                     , ruleRank>>(
                        res
                );
                RuleTensorRaw<double, 0> sumvec = ruleTensor.sum();
                sum = sumvec(0);
                break;
            }
            default:{
                std::cerr << "Rules with such a fanout are not supported to be checked!";
            }
        }
        if(sum > 1) {
            std::cerr << "The sum of a rule was larger than 1: " << sum
                      << "  inside weights: \n";
            for(auto& inside : insides)
                std::cerr << " / " << inside << "(dim: " << inside.dimension(0) << ")";
            std::cerr << "\n  outside weights: " << outside
                      << "  rule tensor:\n" << rTstr.str() << "\n"
                      << "  normalization: " << normalization
                      << "  calc: " << calc
                      << "  dimensions: ";

            for(const auto d : dimens)
                std::cerr << " "<< d;
            std::cerr << std::endl;
        }
    }



    template <typename Nonterminal>
    LatentAnnotation project_annotation(const LatentAnnotation & annotation, const GrammarInfo2 & grammarInfo) {
        // build HG
        MAPTYPE<size_t, bool> nodes;
        std::vector<size_t> nLabels {};

        size_t labelCounter = 0;
        std::vector<size_t> eLabels {};


        for(const auto rule : grammarInfo.rule_to_nonterminals){
            eLabels.push_back(labelCounter++);
            for(size_t nont : rule) {
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
        for(size_t ruleNr = 0; ruleNr < grammarInfo.rule_to_nonterminals.size(); ++ruleNr){
            std::vector<size_t> rule = grammarInfo.rule_to_nonterminals[ruleNr];
            std::vector<Element<Node<Nonterminal>>> rhs{};
            for(size_t i = 0; i < rule.size(); ++i) {
                size_t nont = rule[i];
                if (nodeElements.count(nont) == 0) {
                    nodeElements.insert(MAPTYPE<size_t, Element<Node<size_t>>>::value_type (nont, hg->create(nont)));
                }
                if(i != 0)
                    rhs.push_back(nodeElements.at(nont));
            }
            hg->add_hyperedge(ruleNr, nodeElements.at(rule[0]), rhs);
        }



        std::shared_ptr<Trainer::TraceManager2<Nonterminal, size_t>> tMPtr = std::make_shared<Trainer::TraceManager2<Nonterminal, size_t>>(nLabelsPtr, eLabelsPtr);
        tMPtr->create(1, hg, nodeElements.at(grammarInfo.start));



        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> insideWeights;
        MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector> outsideWeights;
        for(const auto n : *hg){
            insideWeights[n] = Trainer::WeightVector{annotation.nonterminalSplits.at(n->get_label_id())};
            outsideWeights[n] = Trainer::WeightVector{annotation.nonterminalSplits.at(n->get_label_id())};
        }

        // TODO: precision needs to be set from the outside. Change interface?
        tMPtr->set_io_precision(0.0001);
        tMPtr->set_io_cycle_limit(200);
        (*tMPtr)[0].io_weights_fixpoint_la(*(annotation.ruleWeights), annotation.rootWeights, insideWeights, outsideWeights);

        // do projection
        auto projRuleWeights = std::vector<RuleTensor<double>>();

        for(size_t ruleId = 0; ruleId < grammarInfo.rule_to_nonterminals.size(); ++ruleId){
            const std::vector<size_t>& rule = grammarInfo.rule_to_nonterminals[ruleId];
            const auto& ruleVariant = (*(annotation.ruleWeights))[ruleId];

            const auto normalisationCalc = insideWeights[nodeElements.at(rule.at(0))].contract(outsideWeights[nodeElements.at(rule.at(0))], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(0, 0)});
            Eigen::Tensor<double, 0> normalisationVector = normalisationCalc;

            if(normalisationVector(0) == 0) { // either in(A)=0 or out(A)=0
                // weight of rule is defined to be equally distributed
                size_t norm = grammarInfo.normalizationGroups[rule[0]].size();
                switch(rule.size()){
                    case 1: {
                        RuleTensorRaw<double, 1> res(1);
                        res.setValues({1.0/(double)norm});

//                        std::ostringstream rTstr;
//                        std::vector<double> dimens{res.dimension(0)};
//                        std::vector<Trainer::WeightVector> insides{};
//                        check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, 1);

                        projRuleWeights.push_back(res);
                        continue;
                    }
                    case 2: {
                        RuleTensorRaw<double, 2> res(1, 1);
                        res.setValues({{1.0/(double)norm}});

//                        std::ostringstream rTstr;
//                        std::vector<double> dimens{res.dimension(0), res.dimension(1)};
//                        std::vector<Trainer::WeightVector> insides{};
//                        check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, 1);

                        projRuleWeights.push_back(res);
                        continue;
                    }
                    case 3: {
                        RuleTensorRaw<double, 3> res(1,1,1);
                        res.setValues({{{1.0/(double)norm}}});

//                        std::ostringstream rTstr;
//                        std::vector<double> dimens{res.dimension(0), res.dimension(1), res.dimension(2)};
//                        std::vector<Trainer::WeightVector> insides{};
//                        check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, 1);

                        projRuleWeights.push_back(res);
                        continue;
                    }
                    case 4: {
                        RuleTensorRaw<double, 4> res(1,1,1,1);
                        res.setValues({{{{1.0/(double)norm}}}});

//                        std::ostringstream rTstr;
//                        std::vector<double> dimens{res.dimension(0), res.dimension(1), res.dimension(2), res.dimension(3)};
//                        std::vector<Trainer::WeightVector> insides{};
//                        check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, 1);

                        projRuleWeights.push_back(res);
                        continue;
                    }
                    default: {
                        std::cerr << "Rules with " << rule.size()-1 << " RHS nonterminals are not supported.";
                        abort();
                    }
                }

            }

            switch(rule.size()){
                case 1: {
                    constexpr unsigned ruleRank{1};
                    const RuleTensorRaw<double, ruleRank> ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                         , ruleRank>>(
                            ruleVariant
                    );
                    auto sumWeight = ruleTensor.contract(outsideWeights[nodeElements.at(rule[0])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(0, 0)});
                    RuleTensorRaw<double, 0> calc = sumWeight;
                    RuleTensorRaw<double, 1> res(1);
                    res.setValues({calc(0)/normalisationVector(0)});

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
                    const RuleTensorRaw<double, ruleRank> ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                         , ruleRank>>(
                            ruleVariant
                    );
                    auto sumWeight = ruleTensor.contract(insideWeights[nodeElements.at(rule[1])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(1, 0)})
                                                .contract(outsideWeights[nodeElements.at(rule[0])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(0, 0)});
                    RuleTensorRaw<double, 0> calc = sumWeight;
                    RuleTensorRaw<double, 2> res(1,1);
                    res.setValues({{calc(0)/normalisationVector(0)}});

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
                    const RuleTensorRaw<double, ruleRank> ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                         , ruleRank>>(
                            ruleVariant
                    );
                    auto sumWeight = ruleTensor.contract(insideWeights[nodeElements.at(rule[2])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(2, 0)})
                            .contract(insideWeights[nodeElements.at(rule[1])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(1, 0)})
                            .contract(outsideWeights[nodeElements.at(rule[0])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(0, 0)});
                    RuleTensorRaw<double, 0> calc = sumWeight;
                    RuleTensorRaw<double, 3> res(1,1,1);
                    res.setValues({{{calc(0)/normalisationVector(0)}}});

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
                    const RuleTensorRaw<double, ruleRank> ruleTensor = boost::get<Trainer::RuleTensorRaw<double
                                                                                                         , ruleRank>>(
                            ruleVariant
                    );
                    auto sumWeight = ruleTensor.contract(insideWeights[nodeElements.at(rule[3])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(3, 0)})
                            .contract(insideWeights[nodeElements.at(rule[2])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(2, 0)})
                            .contract(insideWeights[nodeElements.at(rule[1])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(1, 0)})
                            .contract(outsideWeights[nodeElements.at(rule[0])], Eigen::array<Eigen::IndexPair<long>,1>{Eigen::IndexPair<long>(0, 0)});
                    RuleTensorRaw<double, 0> calc = sumWeight;
                    RuleTensorRaw<double, 4> res(1,1,1,1);
                    res.setValues({{{{calc(0)/normalisationVector(0)}}}});

//                    std::ostringstream rTstr;
//                    rTstr << ruleTensor;
//                    std::vector<double> dimens{ruleTensor.dimension(0), ruleTensor.dimension(1), ruleTensor.dimension(2), ruleTensor.dimension(3)};
//                    std::vector<Trainer::WeightVector> insides{insideWeights[nodeElements.at(rule.at(1))], insideWeights[nodeElements.at(rule.at(2))], insideWeights[nodeElements.at(rule.at(3))]};
//                    check_rule_weight_for_consistency(res, insides, outsideWeights[nodeElements.at(rule.at(0))], normalisationVector, rTstr, dimens, calc(0));

                    projRuleWeights.push_back(res);
                    break;
                }
                default:
                    std::cerr << "Rules with " << rule.size()-1 << " RHS nonterminals are not supported";
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
                , std::make_unique<std::vector <RuleTensor<double>>>(projRuleWeights));
    }

}




#endif //STERMPARSER_ANNOTATIONPROJECTION_H
