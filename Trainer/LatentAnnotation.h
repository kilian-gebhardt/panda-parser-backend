//
// Created by kilian on 08/03/17.
//

#ifndef STERMPARSER_LATENTANNOTATION_H
#define STERMPARSER_LATENTANNOTATION_H

#include "StorageManager.h"
#include "TrainingCommon.h"
#include <memory>


namespace Trainer {
    class LatentAnnotation {
    public:
        const std::vector <std::size_t> nonterminalSplits;
        WeightVector rootWeights;
        std::unique_ptr<std::vector <RuleTensor<double>>> ruleWeights;

//        LatentAnnotation(LatentAnnotation && other) :
//                nonterminalSplits(std::move(other.nonterminalSplits))
//            ,   rootWeights(std::move(other.rootWeights))
//            ,   ruleWeights(std::move(other.ruleWeights))
//        {}

        LatentAnnotation(
                const std::vector <size_t> nonterminalSplits
                , const WeightVector && rootWeights
                , std::unique_ptr<std::vector <RuleTensor<double>>> && ruleWeights
        ) : nonterminalSplits(nonterminalSplits), rootWeights(std::move(rootWeights)), ruleWeights(std::move(ruleWeights)) {
        };

        LatentAnnotation(
                const std::vector <size_t> nonterminalSplits
                , const std::vector<double> &rootWeights
                , const std::vector <std::vector<double>> &ruleWeights
                , const GrammarInfo2 &grammarInfo
                , StorageManager &storageManager
        ) : nonterminalSplits(nonterminalSplits)
            , rootWeights(storageManager.create_weight_vector<WeightVector>(rootWeights.size()))
            , ruleWeights(std::make_unique<std::vector<RuleTensor <double>>>()) {
            convert_to_eigen(
                    ruleWeights
                    , *(this->ruleWeights)
                    , nonterminalSplits
                    , storageManager
                    , grammarInfo
            );
            for (unsigned i = 0; i < rootWeights.size(); ++i) {
                this->rootWeights(i) = rootWeights[i];
            }
        }

        // if there are no splits
        LatentAnnotation(
                const std::vector<double> &ruleWeights
                , const GrammarInfo2 &grammarInfo
                , StorageManager &storageManager
        ) : LatentAnnotation(
                std::vector<size_t>(grammarInfo.normalizationGroups.size(), 1)
                , std::vector<double>(1, 1.0)
                , LatentAnnotation::lift_doubles(ruleWeights)
                , grammarInfo
                , storageManager
        ) {}

    private:
        unsigned convert_to_eigen(
                const std::vector <std::vector<double>> &rule_weights
                , std::vector <Trainer::RuleTensor<double>> &rule_tensors
                , const std::vector <size_t> &nonterminal_splits
                , StorageManager &storageManager
                , const GrammarInfo2 &grammarInfo
        ) {
            unsigned allocated(0);
            unsigned rule = 0;
            for (auto rule_weight : rule_weights) {
                rule_tensors.push_back(
                        storageManager.create_uninitialized_tensor(rule, grammarInfo, nonterminal_splits)
                    );
                auto & rule_tensor = rule_tensors.back();
                const size_t dims = grammarInfo.rule_to_nonterminals[rule].size();

                switch (dims) {
                    case 1:
                        convert_format_no_creation<1>(rule_weight, rule_tensor);
                        break;
                    case 2:
                        convert_format_no_creation<2>(rule_weight, rule_tensor);
                        break;
                    case 3:
                        convert_format_no_creation<3>(rule_weight, rule_tensor);
                        break;
                    case 4:
                        convert_format_no_creation<4>(rule_weight, rule_tensor);
                        break;
                    default:
                        assert(false && "Rules with more than 3 RHS nonterminals are not implemented.");
                        abort();
                }
                allocated += rule_weight.size();
                ++rule;
            }
            return allocated;
        }

        template<long rank>
        inline void convert_format_no_creation(
                const std::vector<double> &weights, Trainer::RuleTensor<double> &tensor
        ) {
            RuleTensorRaw<double, rank>& tensor_raw = boost::get < Trainer::RuleTensorRaw < double, rank>>(tensor);
            TensorIteratorHighToLow<rank> tensorIterator(&tensor_raw);
            std::copy(weights.begin(), weights.end(), tensorIterator);
        }

        static std::vector<std::vector<double>> lift_doubles(const std::vector<double> &ruleWeights) {
            std::vector<std::vector<double >> ruleWeightsLa;
            for (const double &rule_weight : ruleWeights) {
                ruleWeightsLa.emplace_back(1, rule_weight);
            }
            return ruleWeightsLa;
        }

    };

}
#endif //STERMPARSER_LATENTANNOTATION_H
