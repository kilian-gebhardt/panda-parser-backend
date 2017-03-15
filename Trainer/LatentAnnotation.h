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

        double get_weight(const size_t ruleID, const std::vector<size_t> & index) const {
            switch ((*ruleWeights)[ruleID].which() + 1) {
                case 1: return get_weight_ranked<1>(ruleID, index);
                case 2: return get_weight_ranked<2>(ruleID, index);
                case 3: return get_weight_ranked<3>(ruleID, index);
                case 4: return get_weight_ranked<4>(ruleID, index);
                default:
                    std::cerr << "Rule of rank " << (*ruleWeights)[ruleID].which() + 1 << " unsupported." << std::endl;
                    abort();
            }
        }

        std::vector<std::vector<double>> get_rule_weights() const {
            std::vector<std::vector<double>> weights;
            for (const auto & tensor : *ruleWeights) {
                weights.emplace_back();
                switch(tensor.which() + 1) {
                    case 1: de_convert_format<1>(weights.back(), tensor); break;
                    case 2: de_convert_format<2>(weights.back(), tensor); break;
                    case 3: de_convert_format<3>(weights.back(), tensor); break;
                    case 4: de_convert_format<4>(weights.back(), tensor); break;
                    default:
                        std::cerr << "Rule of rank " << tensor.which() + 1 << " unsupported." << std::endl;
                        abort();
                }
            }
            return weights;
        }

        std::vector<double> get_root_weights() const {
            std::vector<double> root;
            for (Eigen::Index idx = 0; idx < rootWeights.dimension(0); ++idx)
                root.push_back(rootWeights(idx));
            return root;
        }

        bool is_proper(std::shared_ptr<const GrammarInfo2> grammarInfo) const {
            for (size_t nont = 0; nont < grammarInfo->normalizationGroups.size(); ++nont) {
                auto & group = grammarInfo->normalizationGroups[nont];
                Eigen::Tensor<double, 1> normalizationDivisor(nonterminalSplits[nont]);
                normalizationDivisor.setZero();
                for (size_t ruleId : group) {
                    compute_normalization_divisor(normalizationDivisor, (*ruleWeights)[ruleId]);
                }
                for (auto idx = 0; idx < normalizationDivisor.dimension(0); ++idx) {
                    if (std::abs(normalizationDivisor(idx) - 1) > 0.01) {
                        std::cerr << "non proper LA at idx " << idx << ": " << normalizationDivisor << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }

    private:

        template <long rank>
        double get_weight_ranked(const size_t ruleID, const std::vector<size_t> & index) const  {
            RuleTensorRaw<double, rank>& tensor_raw
                    = boost::get < Trainer::RuleTensorRaw < double, rank>>((*ruleWeights)[ruleID]);
            Eigen::array<size_t, rank> index_array;
            std::copy(index.cbegin(), index.cend(), index_array.begin());
            return tensor_raw(index_array);
        }

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
        static inline void convert_format_no_creation(
                const std::vector<double> &weights, Trainer::RuleTensor<double> &tensor
        ) {
            RuleTensorRaw<double, rank>& tensor_raw = boost::get < Trainer::RuleTensorRaw < double, rank>>(tensor);
            TensorIteratorHighToLow<rank> tensorIterator(&tensor_raw);
            std::copy(weights.begin(), weights.end(), tensorIterator);
        }

        template<int rank>
        static inline void de_convert_format(std::vector<double> & weight, const RuleTensor<double> & tensor) {
            const RuleTensorRaw<double, rank> & tensor_raw = boost::get < Trainer::RuleTensorRaw < double, rank>>(tensor);
            TensorIteratorHighToLow<rank, true> tensorIterator(&tensor_raw);
            std::copy(tensorIterator, tensorIterator.end(), std::back_inserter(weight));
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
