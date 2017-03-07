//
// Created by kilian on 01/03/17.
//

#ifndef STERMPARSER_TRAININGCOMMON_H
#define STERMPARSER_TRAININGCOMMON_H

#include "TraceManager.h"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <boost/variant.hpp>
#include <vector>
#include <cstddef>

namespace Trainer {
//    using WeightVector = Eigen::TensorMap<Eigen::Tensor<double, 1>>;
    using WeightVector = Eigen::Tensor<double, 1>;

    template <typename Scalar, long rank>
    //using RuleTensorRaw = Eigen::TensorMap<Eigen::Tensor<Scalar, rank>>;
    using RuleTensorRaw = Eigen::Tensor<Scalar, rank>;

    template<typename Scalar>
    using RuleTensor = typename boost::variant<
            RuleTensorRaw<Scalar, 1>
            , RuleTensorRaw<Scalar, 2>
            , RuleTensorRaw<Scalar, 3>
            , RuleTensorRaw<Scalar, 4>
            , RuleTensorRaw<Scalar, 5>
            , RuleTensorRaw<Scalar, 6>
            , RuleTensorRaw<Scalar, 7>
           >;

    struct LatentAnnotation {
        const std::vector<std::size_t> nonterminalSplits;
        WeightVector rootWeights;
        std::vector<RuleTensor<double>> ruleWeights;
        LatentAnnotation(
                const std::vector<size_t> nonterminalSplits
                , WeightVector rootWeights
                , std::vector<RuleTensor<double>> ruleWeights
        ) : nonterminalSplits(nonterminalSplits) , rootWeights(rootWeights), ruleWeights(ruleWeights) {};
    };

    struct MergeInfo {
        const std::vector<std::vector<std::vector<std::size_t >>> mergeSources;
        const std::vector<std::size_t> nontSplitsAfterMerge;
        const std::vector<std::vector<double >> mergeFactors;

        MergeInfo(
                const std::vector<std::vector<std::vector<std::size_t >>> mergeSources
                , const std::vector<std::size_t> nontSplitsAfterMerge
                , const std::vector<std::vector<double>> mergeFactors
        ) : mergeSources(mergeSources), nontSplitsAfterMerge(nontSplitsAfterMerge), mergeFactors(mergeFactors) {}
    };

    std::ostream &operator<<(std::ostream &os, const MergeInfo & mergeInfo) {
        os << "Merge Info: " << std::endl << "Merge factors: " << std::endl;
        {
            size_t nont = 0;
            for (auto vector : mergeInfo.mergeFactors) {
                os << nont << ": ";
                for (auto la_factor : vector)
                    os << la_factor << ", ";
                os << std::endl;
                ++nont;
            }
        }
        os<< std::endl << "Merge sources: " << std::endl;
        {
            size_t nont = 0;
            for (auto nont_split : mergeInfo.mergeSources) {
                size_t la = 0;
                for (auto sources : nont_split) {
                    os << " " << nont << "-" << la << "<- ";
                    for (auto source : sources)
                        os << source << " ";
                    os << std::endl;
                    ++la;
                }
                ++nont;
                os << std::endl;
            }
        }
        return os;
    }

    template<long rank, typename VECTOR>
    inline void compute_normalization_divisor_ranked(VECTOR &goal, const RuleTensor<double> &tensor) {
        Eigen::array<long, rank - 1> sum_dimensions;
        for (size_t index = 0; index < rank - 1; ++index) {
            sum_dimensions[index] = index + 1;
        }
        goal += boost::get<RuleTensorRaw<double, rank>>(tensor).sum(sum_dimensions);
    };

    template<typename VECTOR>
    inline void compute_normalization_divisor(VECTOR &goal, const RuleTensor<double> &tensor) {
        switch (tensor.which() + 1) {
            case 1:
                compute_normalization_divisor_ranked<1>(goal, tensor);
                break;
            case 2:
                compute_normalization_divisor_ranked<2>(goal, tensor);
                break;
            case 3:
                compute_normalization_divisor_ranked<3>(goal, tensor);
                break;
            case 4:
                compute_normalization_divisor_ranked<4>(goal, tensor);
                break;
            default:
                abort();
        }
    }

    template<long rank, typename VECTOR>
    inline void normalize_ranked(RuleTensor<double> &normalized, const RuleTensor<double> &unnormalized, const VECTOR &normalizer) {
        auto &raw_normalized = boost::get<RuleTensorRaw <double, rank>>(normalized);
        auto &raw_unnormalized = boost::get<RuleTensorRaw <double, rank>>(unnormalized);

        for (unsigned idx = 0; idx < normalizer.dimension(0); ++idx)
            if (not std::isnan(normalizer(idx))
                and not std::isinf(normalizer(idx))
                and normalizer(idx) > 0) {
                raw_normalized.chip(idx, 0)
                        = raw_unnormalized.chip(idx, 0).unaryExpr(
                        [normalizer, idx](const double x) -> double {
                            return x / normalizer(idx);
                        }
                );
            }
    };

    template<typename VECTOR>
    inline void normalize(RuleTensor<double> &normalized, const RuleTensor<double> &unnormalized, const VECTOR &normalizer) {
        switch (unnormalized.which() + 1) {
            case 1:
                normalize_ranked<1>(normalized, unnormalized, normalizer);
                break;
            case 2:
                normalize_ranked<2>(normalized, unnormalized, normalizer);
                break;
            case 3:
                normalize_ranked<3>(normalized, unnormalized, normalizer);
                break;
            case 4:
                normalize_ranked<4>(normalized, unnormalized, normalizer);
                break;
            default:
                abort();
        }
    }

    template<int rule_rank>
    inline void set_zero_ranked(
            RuleTensor<double> & ruleTensor
    ) {
        boost::get<RuleTensorRaw<double, rule_rank>>(ruleTensor).setZero();
    }

    inline void set_zero(RuleTensor<double> & ruleTensor) {
        switch (ruleTensor.which() + 1) {
            case 1: set_zero_ranked<1>(ruleTensor); break;
            case 2: set_zero_ranked<2>(ruleTensor); break;
            case 3: set_zero_ranked<3>(ruleTensor); break;
            case 4: set_zero_ranked<4>(ruleTensor); break;
            default: abort();
        }
    }

    template<unsigned rank, bool isConst = false>
    class TensorIterator {
    private:
        Eigen::array<Eigen::DenseIndex, rank> index;
        RuleTensorRaw <double, rank> *tensor;
    public:
        using value_type = typename std::conditional<isConst, const double, double>::type;
        using pointer = typename std::conditional<isConst, const double *, double *>::type;
        using reference = typename std::conditional<isConst, const double &, double &>::type;
        using iterator_category = std::forward_iterator_tag;

        TensorIterator() : tensor(nullptr) { std::fill(index.begin(), index.end(), 0); }

        TensorIterator(RuleTensorRaw <double, rank> *tensor) : tensor(tensor) {
            std::fill(index.begin(), index.end(), 0);
        }

        TensorIterator(
                RuleTensorRaw<double, rank> *tensor
                , Eigen::array<Eigen::DenseIndex, rank> index
        ) : index(index), tensor(tensor) {}

        bool operator==(const TensorIterator<rank, isConst> &other) const { return index == other.index; }

        bool operator!=(const TensorIterator<rank, isConst> &other) const { return !(*this == other); }

        reference operator*() { return (*tensor)(index); }

        pointer operator->() { return &((*tensor)(index)); }

        TensorIterator<rank, isConst> &operator++() {
            size_t idx = 0;
            const auto &dimensions = tensor->dimensions();
            while (idx < rank) {
                if (index[idx] + 1 < dimensions[idx]) {
                    ++index[idx];
                    break;
                } else {
                    index[idx] = 0;
                    ++idx;
                }
            }
            if (idx == rank)
                index = dimensions;
            return *this;
        }

        TensorIterator<rank, isConst> end() const {
            return TensorIterator<rank, isConst>(tensor, tensor->dimensions());
        };

        const Eigen::array<Eigen::DenseIndex, rank> &get_index() {
            return index;
        };
//
//        friend class TensorIterator<rank, true>;
//        friend class TensorIterator<rank, false>;
    };


    template<unsigned rank, bool isConst = false>
    class MergeIterator {
    private:
        const RuleTensorRaw <double, rank> *tensor;
        const MergeInfo *mergeInfo;
        const std::vector<std::vector<size_t>> * rule_to_nonterminals;
        size_t ruleId;
        const Eigen::array<Eigen::DenseIndex, rank> *goalIndex;
        std::array<unsigned, rank> hopIndex;
        Eigen::array<Eigen::DenseIndex, rank> sourceIndex;


    public:
        using value_type = typename std::conditional<isConst, const double, double>::type;
        using pointer = typename std::conditional<isConst, const double *, double *>::type;
        using reference = typename std::conditional<isConst, const double &, double &>::type;
        using iterator_category = std::forward_iterator_tag;

        MergeIterator()
                : tensor(nullptr), mergeInfo(nullptr), rule_to_nonterminals(nullptr), ruleId(0),
                  goalIndex(nullptr) {
            std::fill(hopIndex.begin(), hopIndex.end(), 0);
            std::fill(sourceIndex.begin(), sourceIndex.end(), 0);
        }

        MergeIterator(
                const RuleTensorRaw<double, rank> *tensor
                , size_t rule_id
                , const Eigen::array<Eigen::DenseIndex, rank> *goal_index
                , const MergeInfo *mergeInfo
                , const std::vector<std::vector<size_t>> *rule_to_nonterminals
        ) : tensor(tensor), mergeInfo(mergeInfo), rule_to_nonterminals(rule_to_nonterminals), ruleId(rule_id), goalIndex(goal_index) {
            std::fill(hopIndex.begin(), hopIndex.end(), 0);
            for (size_t idx = 0; idx < rank; ++idx) {
                sourceIndex[idx] = (mergeInfo->mergeSources)
                [(*rule_to_nonterminals)[rule_id][idx]] // nont
                [(*goal_index)[idx]]                              // goal split
                        .front();                                         // source split
            }
        }

        MergeIterator(
                const RuleTensorRaw<double, rank> *tensor
                , size_t rule_id
                , const Eigen::array<Eigen::DenseIndex, rank> *goal_index
                , Eigen::array<Eigen::DenseIndex, rank> index
                , const MergeInfo *mergeInfo
                , const std::vector<std::vector<size_t>> *rule_to_nonterminals
        )
                : tensor(tensor), mergeInfo(mergeInfo), rule_to_nonterminals(rule_to_nonterminals), ruleId(rule_id),
                  goalIndex(goal_index), sourceIndex(index) {}

        bool operator==(const MergeIterator<rank, isConst> &other) const { return sourceIndex == other.sourceIndex; }

        bool operator!=(const MergeIterator<rank, isConst> &other) const { return not (*this == other); }

        reference operator*() { return (*tensor)(sourceIndex); }

        pointer operator->() { return &((*tensor)(sourceIndex)); }

        MergeIterator<rank, isConst> &operator++() {
            size_t idx = 0;
            while (idx < rank) {
                if (hopIndex[idx] + 1 < (mergeInfo->mergeSources)
                [(*rule_to_nonterminals)[ruleId][idx]] // nont
                [(*goalIndex)[idx]]                    // goal split
                        .size()) {                     // source split
                    ++hopIndex[idx];
                    sourceIndex[idx] = (mergeInfo->mergeSources)
                    [(*rule_to_nonterminals)[ruleId][idx]] // nont
                    [(*goalIndex)[idx]]                    // goal split
                    [hopIndex[idx]];                       // source split
                    break;
                } else {
                    hopIndex[idx] = 0;
                    sourceIndex[idx] = (mergeInfo->mergeSources)
                    [(*rule_to_nonterminals)[ruleId][idx]]          // nont
                    [(*goalIndex)[idx]]                             // goal split
                    [hopIndex[idx]];                                // source split
                    ++idx;
                }
            }
            if (idx == rank) {
                sourceIndex = tensor->dimensions();
            }

            return *this;
        }

        MergeIterator<rank, isConst> end() const {
            return MergeIterator<rank, isConst>(
                    tensor
                    , ruleId
                    , goalIndex
                    , tensor->dimensions()
                    , mergeInfo
                    , rule_to_nonterminals
            );
        };

        double mergeFactor() {
            if ((mergeInfo->mergeSources)
                [(*rule_to_nonterminals)[ruleId][0]] // nont
                [(*goalIndex)[0]]                    // goal split
                        .size() == 1) {              // source split
                return 1.0;
            } else
                return mergeInfo->mergeFactors[(*rule_to_nonterminals)[ruleId][0]][sourceIndex[0]];
        }

    };


}

#endif //STERMPARSER_TRAININGCOMMON_H
