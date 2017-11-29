//
// Created by kilian on 08/03/17.
//

#ifndef STERMPARSER_LATENTANNOTATION_H
#define STERMPARSER_LATENTANNOTATION_H

#include "StorageManager.h"
#include "TrainingCommon.h"
#include <memory>


namespace Trainer {


    struct ValidityChecker : boost::static_visitor<bool> {

        template<int rank>
        bool operator()(const RuleTensorRaw<double, rank> &tensor) {
            bool validity = true;

            auto isnan = tensor.isnan().any();
            Eigen::Tensor<bool,0> evalIsNan = isnan;
            if(evalIsNan(0)){
                std::clog << "[LA] A rule contains a NaN-value!";
                validity = false;
            }

            auto isinf = tensor.isinf().any();
            Eigen::Tensor<bool,0> evalIsInf = isinf;
            if(evalIsInf(0)){
                std::clog << "[LA] A rule contains an inf-value!";
                validity = false;
            }

            return validity;
        }
    };

    struct WeightAccessVisitor : boost::static_visitor<double> {
        const std::vector<size_t> & index;
        WeightAccessVisitor(const std::vector<size_t> & index) : index(index) {};

        template <int rank>
        double operator()(RuleTensorRaw<double, rank>& tensor_raw) const  {
            Eigen::array<size_t, rank> index_array;
            std::copy(index.cbegin(), index.cend(), index_array.begin());
            return tensor_raw(index_array);
        }
    };

    struct FormatDeconversionVisitor : boost::static_visitor<std::vector<double>> {
        template<int rank>
        inline std::vector<double> operator()(const RuleTensorRaw<double, rank> & tensor) {
            std::vector<double> weight;
            TensorIteratorHighToLow<rank, true> tensorIterator(&tensor);
            std::copy(tensorIterator, tensorIterator.end(), std::back_inserter(weight));
            return weight;
        }
    };

    struct FormatConversionVisitor : boost::static_visitor<void> {
        const std::vector<double> &weights;
        FormatConversionVisitor(const std::vector<double> &weights) : weights(weights) {};
        template<int rank>
        inline void operator()(RuleTensorRaw<double, rank>& tensor) {
            TensorIteratorHighToLow<rank> tensorIterator(&tensor);
            std::copy(weights.begin(), weights.end(), tensorIterator);
        }
    };

    struct TensorRandomizer : boost::static_visitor<void> {
        std::mt19937 & generator;
        std::uniform_real_distribution<double> & distribution;
        const double bias;

        TensorRandomizer(
                std::mt19937 & generator
                , std::uniform_real_distribution<double> & distribution
                , const double bias
        ) : generator(generator), distribution(distribution), bias(bias) {}

        template<int rule_rank>
        inline void operator()(RuleTensorRaw<double, rule_rank> &tensor) {
            tensor = tensor.unaryExpr([&] (double x) -> double {
                return x * distribution(generator) + bias * distribution(generator);
            });
        }
    };

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

        LatentAnnotation(const LatentAnnotation & latentAnnotation) :
                nonterminalSplits(latentAnnotation.nonterminalSplits)
        , rootWeights(latentAnnotation.rootWeights.size())
        , ruleWeights(std::make_unique<std::vector<RuleTensor <double>>>()) {
            rootWeights = latentAnnotation.rootWeights;
            StorageManager sm;
            for (auto tensor : *latentAnnotation.ruleWeights)
                (*ruleWeights).push_back(sm.copy_tensor(tensor));
        }

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
            WeightAccessVisitor weightAccessVisitor(index);
            return boost::apply_visitor(weightAccessVisitor, (*ruleWeights)[ruleID]);
        }

        std::vector<std::vector<double>> get_rule_weights() const {
            FormatDeconversionVisitor formatConversionVisitor;
            std::vector<std::vector<double>> weights;
            for (const auto & tensor : *ruleWeights) {
                weights.push_back(boost::apply_visitor(formatConversionVisitor, tensor));
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
            bool proper = true;
            for (size_t nont = 0; nont < grammarInfo->normalizationGroups.size(); ++nont) {
                auto & group = grammarInfo->normalizationGroups[nont];
                Eigen::Tensor<double, 1> normalizationDivisor(nonterminalSplits[nont]);
                normalizationDivisor.setZero();
                for (size_t ruleId : group) {
                    compute_normalization_divisor(normalizationDivisor, (*ruleWeights)[ruleId]);
                }
                for (auto idx = 0; idx < normalizationDivisor.dimension(0); ++idx) {
                    if (std::abs(normalizationDivisor(idx) - 1.0) > std::exp(-5)) {
                        std::cerr << "nont " << nont << " has non-proper LA at LA-Index " << idx << ": "
                                  << normalizationDivisor[idx] << std::endl;
                        proper = false;
                    }
                }
            }
            return proper;
        }

        void make_proper(const GrammarInfo2& grammarInfo) {
            size_t nont_id {0};
            #pragma GCC diagnostic push
            for (const auto ruleSet : grammarInfo.normalizationGroups) {
                #pragma GCC diagnostic ignored "-Wsign-compare"
                for (int index {0}; index < nonterminalSplits[nont_id]; ++index) {
                    VectorSummer vectorSummer(index);
                    double sum {0.0};
                    for (auto ruleID : ruleSet) {
                        sum += boost::apply_visitor(vectorSummer, (*ruleWeights)[ruleID]);
                    }

                    if (std::abs(sum) < std::exp(-30) or std::isnan(sum)) { // The sum is 0 or nan
                        for (auto ruleID : ruleSet) {
                            TensorChipValueSetter tvs(1.0, index);
                            boost::apply_visitor(tvs, (*ruleWeights)[ruleID]);
                        }
                        double sum2 {0.0};
                        for (auto ruleID : ruleSet) {
                            sum2 += boost::apply_visitor(vectorSummer, (*ruleWeights)[ruleID]);
                        }
                        RuleTensorMultiplier rtd(1.0 / sum2, index);
                        for (auto ruleID : ruleSet) {
                            boost::apply_visitor(rtd, (*ruleWeights)[ruleID]);
                        }
                    } else if (std::abs(sum - 1.0) > std::exp(-30)) { // does not sum to 1
                        RuleTensorMultiplier rtd(1.0 / sum, index);
                        for (auto ruleID : ruleSet) {
                            boost::apply_visitor(rtd, (*ruleWeights)[ruleID]);
                        }
                    }
                }
                ++nont_id;
            }
            #pragma GCC diagnostic pop
        }


        void make_proper(std::shared_ptr<GrammarInfo2>& grammarInfo) {
            make_proper(*grammarInfo);
        }


        void add_random_noise(std::shared_ptr<const GrammarInfo2> grammarInfo
                              , double randPercent=1.0
                              , size_t seed=0
                              , double bias=0.01
        ) {
            std::mt19937 generator(seed);
            std::uniform_real_distribution<double> distribution
                    ((100.0 - randPercent) / 100.0, (100.0 + randPercent) / 100.0);

            // add noise to root weights
            rootWeights = rootWeights.unaryExpr([&](double x) {
                return x * distribution(generator) + bias * distribution(generator);
            });
            // normalize root weights
            Eigen::Tensor<double, 0> total_root_weight = rootWeights.sum();
            rootWeights = rootWeights.unaryExpr([&total_root_weight](double x) { return x / total_root_weight(0); });

            TensorRandomizer tensorRandomizer(generator, distribution, bias);

            for (size_t nont = 0; nont < grammarInfo->normalizationGroups.size(); ++nont) {
                auto & group = grammarInfo->normalizationGroups[nont];

                // add noise to rule weights

                for (size_t ruleID : group)
                    boost::apply_visitor(tensorRandomizer, (*ruleWeights)[ruleID]);

                // normalization
                Eigen::Tensor<double, 1> normalizationDivisor(nonterminalSplits[nont]);
                normalizationDivisor.setZero();
                for (size_t ruleId : group) {
                    compute_normalization_divisor(normalizationDivisor, (*ruleWeights)[ruleId]);
                }
                for (size_t ruleId : group) {
                    normalize((*ruleWeights)[ruleId], (*ruleWeights)[ruleId], normalizationDivisor);
                }
            }
        }

        LatentAnnotation& operator= (const LatentAnnotation& other) {
            if (this->nonterminalSplits == other.nonterminalSplits) {
                this->rootWeights = other.rootWeights;
                for (size_t rule = 0; rule < (*ruleWeights).size(); ++rule){
                    (*ruleWeights)[rule] = (*other.ruleWeights)[rule];
                }
            } else {
                std::cerr << "Latent annotation assignment is only supported, if the nonterminal splits match.";
                abort();
            }
            return *this;
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

                FormatConversionVisitor formatConversionVisitor(rule_weight);
                boost::apply_visitor(formatConversionVisitor, rule_tensor);

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

    public:
        bool check_for_validity(double delta = 0.0005){
            bool valid = true;

            // check for root weights:
            Eigen::Tensor<double, 0> rootSum = rootWeights.sum();
            if(std::abs(rootSum(0)-1) > delta){
                std::clog << "[LA] invalid root sum: " << rootSum(0) << std::endl;
                valid = false;
            }

            // check that there is at least one annotation for each nonterminal
            if(! std::all_of(nonterminalSplits.cbegin(), nonterminalSplits.cend(), [](size_t i){return i > 0;})){
                std::clog << "[LA] invalid nonterminal splits (zero-split found)\n";
                valid = false;
            }

            // check the weights for nan and inf
            for(RuleTensor<double> rule : *ruleWeights){
                ValidityChecker validityChecker;
                bool check = boost::apply_visitor(validityChecker, rule);
                if(!check)
                    valid = false;

            }

            return valid;
        }

    };

}
#endif //STERMPARSER_LATENTANNOTATION_H
