//
// Created by kilian on 01/03/17.
//

#ifndef STERMPARSER_TRAININGCOMMON_H
#define STERMPARSER_TRAININGCOMMON_H

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <boost/variant.hpp>
#include <vector>
#include <cstddef>
#include <unordered_map>
#include <iostream>


namespace Trainer {
    template<typename T1, typename T2>
    using MAPTYPE = typename std::unordered_map<T1, T2>;

//    using WeightVector = Eigen::TensorMap<Eigen::Tensor<double, 1>>;
    using WeightVector = Eigen::Tensor<double, 1>;

    template<typename Scalar, int rank>
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
            , RuleTensorRaw<Scalar, 8>
    >;

    struct MergeInfo {
        const std::vector<std::vector<std::vector<std::size_t >>> mergeSources;
        const std::vector<std::size_t> nontSplitsAfterMerge;
        const std::vector<std::vector<double >> mergeFactors;

        MergeInfo(
                const std::vector<std::vector<std::vector<std::size_t>>> mergeSources
                , const std::vector<std::size_t> nontSplitsAfterMerge
                , const std::vector<std::vector<double>> mergeFactors
        ) : mergeSources(mergeSources), nontSplitsAfterMerge(nontSplitsAfterMerge), mergeFactors(mergeFactors) {}

        bool is_proper() const {
            bool proper = true;
            for (size_t nont = 0; nont < nontSplitsAfterMerge.size(); ++nont) {
                for (size_t split = 0; split < nontSplitsAfterMerge[nont]; ++split) {
                    double factorsum = 0.0;
                    for (double mergeSource : mergeSources[nont][split])
                        factorsum += mergeFactors[nont][mergeSource];
                    if (mergeSources[nont][split].size() > 1 and std::abs(factorsum - 1.0) > std::exp(-5)) {
                        std::cerr << "merge factor sum != 1: " << nont << " " << split << " sum: " << factorsum
                                  << std::endl;
                        proper = false;
                    }
                }
            }
            return proper;
        }
    };

    constexpr int LOGSCALE = 100;
    constexpr double SCALE = std::exp(LOGSCALE);

    double calcScaleFactor(double logScale, double scale) {
        if (logScale == std::numeric_limits<int>::min()) {
            return 0.0;// System.out.println("give me a break!");
        }
        if (logScale == 0.0)
            return 1.0;
        if (logScale == 1.0)
            return scale;
        if (logScale == 2.0)
            return scale * scale;
        if (logScale == 3.0)
            return scale * scale * scale;
        if (logScale == -1.0)
            return 1.0 / scale;
        if (logScale == -2.0)
            return 1.0 / scale / scale;
        if (logScale == -3.0)
            return 1.0 / scale / scale / scale;
        return std::pow(scale, logScale);
    }

    double calcScaleFactor(double logScale) {
		return calcScaleFactor(logScale, SCALE);
	}

    /**
     * Inspired by Berkeley parser's edu.berkeley.nlp.util.ScalingTools.scaleArray function
     * @param scalar
     */
    int scaleScalar(double & scalar, int previousScale) {
        int logScale = 0;
        double scale = 1.0;
        double max {scalar};
        if (std::isinf(max) or max == 0)
            return previousScale;
        while (max > SCALE) {
            max = max / SCALE;
            scale /= SCALE;
            logScale += 1;
        }
        while (max > 0.0 and max < 1.0 / SCALE) {
            max = max * SCALE;
            scale *= SCALE;
            logScale -= 1;
        }
        if (logScale != 0)
            scalar = scalar * scale;
        return previousScale + logScale;
    }

    /**
     * Inspired by Berkeley parser's edu.berkeley.nlp.util.ScalingTools.scaleArray function
     * @param vector
     */
    template<int rank>
    int scaleTensor(Eigen::Tensor<double, rank> & vector, int previousScale) {
        double max {0};
        if (vector.size()) {
            Eigen::Tensor<double, 0> max_tensor{vector.maximum()};
            max = max_tensor(0);
        }
        int logScale = 0;
        double scale = 1.0;
        if (std::isinf(max) or max == 0)
            return previousScale;
        while (max > SCALE) {
            max = max / SCALE;
            scale /= SCALE;
            logScale += 1;
        }
        while (max > 0.0 and max < 1.0 / SCALE) {
            max = max * SCALE;
            scale *= SCALE;
            logScale -= 1;
        }
        if (logScale != 0)
            vector = vector * scale;
        return previousScale + logScale;
    }


    // Boost static visitors
    struct TensorScaleVisitor : boost::static_visitor<int> {
        int previousScale;
        TensorScaleVisitor(int previousScale) : previousScale(previousScale) {}

        template<int rank>
        int
        operator()(RuleTensorRaw<double, rank>& tensor) {
            return scaleTensor(tensor, previousScale);
        }
    };

    int scaleTensor(RuleTensor<double> & tensor, int previousScale) {
        TensorScaleVisitor tsv(previousScale);
        return boost::apply_visitor(tsv, tensor);
    }


    std::ostream &operator<<(std::ostream &os, const MergeInfo &mergeInfo) {
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
        os << std::endl << "Merge sources: " << std::endl;
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

    template<typename VECTOR>
    struct NormalizationDivisor : boost::static_visitor<void> {
        VECTOR &goal;
        NormalizationDivisor(VECTOR & goal) : goal(goal) {};
        template<int rank>
        inline void operator()(const RuleTensorRaw<double, rank> &tensor) {
            Eigen::array<Eigen::Index, rank - 1> sum_dimensions;
            for (Eigen::Index index = 0; index < rank - 1; ++index) {
                sum_dimensions[index] = index + 1;
            }
            goal += tensor.sum(sum_dimensions);
        };
    };


    template<typename VECTOR>
    inline void compute_normalization_divisor(VECTOR &goal, const RuleTensor<double> &tensor) {
        NormalizationDivisor<VECTOR> nd(goal);
        boost::apply_visitor(nd, tensor);
    }

    template<typename VECTOR>
    struct NormalizationVisitor : boost::static_visitor<void> {
        RuleTensor<double> &normalized;
        const VECTOR &normalizer;

        NormalizationVisitor(RuleTensor<double> &normalized, const VECTOR &normalizer)
                : normalized(normalized), normalizer(normalizer) {};

        template<int rank>
        inline void operator()(const RuleTensorRaw<double, rank> &raw_unnormalized) {
            auto &raw_normalized = boost::get<RuleTensorRaw<double, rank>>(normalized);
            for (unsigned idx = 0; idx < normalizer.dimension(0); ++idx)
                if (not std::isnan(normalizer(idx))
                    and not std::isinf(normalizer(idx))
                    and normalizer(idx) > 0) {
                    const double normalizationValue {normalizer(idx)};
                    raw_normalized.chip(idx, 0)
                            = raw_unnormalized.chip(idx, 0).unaryExpr(
                            [normalizationValue](const double x) -> double {
                                return x / normalizationValue;
                            }
                    );
                }
        };
    };

    template<typename VECTOR>
    inline void
    normalize(RuleTensor<double> &normalized, const RuleTensor<double> &unnormalized, const VECTOR &normalizer) {
        NormalizationVisitor<VECTOR> normalizationVisitor(normalized, normalizer);
        boost::apply_visitor(normalizationVisitor, unnormalized);
    }

    struct SetZeroVisitor : boost::static_visitor<void> {
        template<int rule_rank>
        inline void operator()(RuleTensorRaw<double, rule_rank> & ruleTensor) {
            ruleTensor.setZero();
        }
    };

    inline void set_zero(RuleTensor<double> &ruleTensor) {
        SetZeroVisitor setZeroVisitor;
        boost::apply_visitor(setZeroVisitor, ruleTensor);
    }

    template<unsigned rank, bool isConst = false>
    class TensorIteratorLowToHigh {
    protected:
        Eigen::array<Eigen::DenseIndex, rank> index;
        RuleTensorRaw<double, rank> *tensor;
    public:
        using value_type = typename std::conditional<isConst, const double, double>::type;
        using pointer = typename std::conditional<isConst, const double *, double *>::type;
        using reference = typename std::conditional<isConst, const double &, double &>::type;
        using iterator_category = std::forward_iterator_tag;

        TensorIteratorLowToHigh() : tensor(nullptr) { std::fill(index.begin(), index.end(), 0); }

        TensorIteratorLowToHigh(RuleTensorRaw<double, rank> *tensor) : tensor(tensor) {
            std::fill(index.begin(), index.end(), 0);
        }

        TensorIteratorLowToHigh(
                RuleTensorRaw<double, rank> *tensor
                , Eigen::array<Eigen::DenseIndex, rank> index
        ) : index(index), tensor(tensor) {}

        bool operator==(const TensorIteratorLowToHigh<rank, isConst> &other) const { return index == other.index; }

        bool operator!=(const TensorIteratorLowToHigh<rank, isConst> &other) const { return !(*this == other); }

        reference operator*() { return (*tensor)(index); }

        pointer operator->() { return &((*tensor)(index)); }

        TensorIteratorLowToHigh<rank, isConst> &operator++() {
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

        TensorIteratorLowToHigh<rank, isConst> end() const {
            return TensorIteratorLowToHigh<rank, isConst>(tensor, tensor->dimensions());
        };

        const Eigen::array<Eigen::DenseIndex, rank> &get_index() {
            return index;
        };
    };

    template<unsigned rank, bool isConst = false>
    class TensorIteratorHighToLow {
    public:
        using value_type = typename std::conditional<isConst, const double, double>::type;
        using difference_type = void;
        using pointer = typename std::conditional<isConst, const double *, double *>::type;
        using reference = typename std::conditional<isConst, const double &, double &>::type;
        using iterator_category = std::forward_iterator_tag;
        using tensor_type = typename std::conditional<isConst
                                                      , const RuleTensorRaw <double, rank>
                                                      , RuleTensorRaw <double, rank>
                                                     >::type;
    protected:
        Eigen::array<Eigen::DenseIndex, rank> index;
        tensor_type *tensor;

    public:

        TensorIteratorHighToLow() : tensor(nullptr) { std::fill(index.begin(), index.end(), 0); }

        TensorIteratorHighToLow(tensor_type * tensor) : tensor(tensor) {
            std::fill(index.begin(), index.end(), 0);
        }

        TensorIteratorHighToLow(
                tensor_type *tensor
                , const Eigen::array<Eigen::DenseIndex, rank> & index
        ) : index(index), tensor(tensor) {}

        bool operator==(const TensorIteratorHighToLow<rank, isConst> &other) const { return index == other.index; }

        bool operator!=(const TensorIteratorHighToLow<rank, isConst> &other) const { return !(*this == other); }

        reference operator*() { return (*tensor)(index); }

        pointer operator->() { return &((*tensor)(index)); }

        TensorIteratorHighToLow<rank, isConst> &operator++() {
            size_t idx {rank - 1};
            const auto &dimensions = tensor->dimensions();
            while (true) {
                if (index[idx] + 1 < dimensions[idx]) {
                    ++index[idx];
                    break;
                } else {
                    index[idx] = 0;
                    if (idx == 0) {
                        index = dimensions;
                        return *this;
                    }
                    else
                        --idx;
                }
            }
            return *this;
        }

        TensorIteratorHighToLow<rank, isConst> end() const {
            return TensorIteratorHighToLow<rank, isConst>(tensor, tensor->dimensions());
        };

        const Eigen::array<Eigen::DenseIndex, rank> &get_index() {
            return index;
        };
    };

    template<unsigned rank, bool isConst = false>
    class MergeIterator {
    private:
        const RuleTensorRaw<double, rank> *tensor;
        const MergeInfo *mergeInfo;
        const std::vector<std::vector<size_t>> *rule_to_nonterminals;
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
        )
                : tensor(tensor), mergeInfo(mergeInfo), rule_to_nonterminals(rule_to_nonterminals), ruleId(rule_id),
                  goalIndex(goal_index) {
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

        bool operator!=(const MergeIterator<rank, isConst> &other) const { return not(*this == other); }

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

        const Eigen::array<Eigen::DenseIndex, rank> & get_source_index() {
            return sourceIndex;
        }
    };





    // Boost static visitors


    struct SizeOneTensorCreator : boost::static_visitor<RuleTensor<double>> {
        double value;

        SizeOneTensorCreator(
                double value
        )
                :
                value(value)
        {};


        template<int rank>
        RuleTensorRaw<double, rank>
        operator()(const RuleTensorRaw<double, rank>& /*tensor*/) {
            Eigen::array<Eigen::Index, rank> dim;
            std::fill(dim.begin(), dim.end(), 1);
            RuleTensorRaw<double, rank> result(dim);
            result.setConstant(value);

            return result;
        }
    };

    struct SizeOneTensorAccessor : boost::static_visitor<double> {
        template<int rank>
        double
        operator()(const RuleTensorRaw<double, rank>& tensor) {
            Eigen::array<Eigen::Index, rank> dim;
            std::fill(dim.begin(), dim.end(), 0);
            return tensor(dim);
        }
    };

    struct TensorChipValueSetter : boost::static_visitor<void> {
        const double value;
        const int index;

        TensorChipValueSetter(double value, int index) : value(value), index(index) {};

        template<int rank>
        void
        operator()(RuleTensorRaw<double, rank>& tensor) {
            tensor.chip(index, 0).setConstant(value);
        }

    };



    struct TensorChipMultiplier : boost::static_visitor<void> {
        double value;
        const int index;

        TensorChipMultiplier(double value, int index=0)
                : value(value), index(index)
        {};


        template<int rank>
        void
        operator()(RuleTensorRaw<double, rank>& tensor) {
            tensor.chip(index, 0) = tensor.chip(index, 0) * (value);
        }
    };


    struct RuleTensorContractor : boost::static_visitor<RuleTensor<double>> {
        const WeightVector& factor;
        int multiplyDimension;

        RuleTensorContractor(
                const WeightVector& f
                , int multiplyDimension = -1
        )
                :
                factor(f)
                , multiplyDimension(multiplyDimension)
        {};

        template<int rank>
        typename std::enable_if<(rank > 1), RuleTensorRaw<double, rank-1>>::type
        operator()(const RuleTensorRaw<double, rank>& tensor) {
            if (multiplyDimension < 0 || multiplyDimension > rank-1)
                multiplyDimension = rank -1;
            RuleTensorRaw<double, rank-1> result = tensor.contract(
                    factor, Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(multiplyDimension, 0)});
            return result;
        }

        template <int rank>
        typename std::enable_if<(rank <= 1), RuleTensorRaw<double, 1>>::type
        operator()(const RuleTensorRaw<double, rank>& /*tensor*/) const {
            std::cerr << "RuleTensorContractor can only handle Tensors of at least dimension 2!";
            abort();
        }
    };


    inline double safe_division(double numerator, double denominator) {
        double quotient {numerator / denominator};
        if (not std::isnan(quotient) or std::isinf(quotient))
            return quotient;
        else {
            if (numerator >= denominator)
                return 1.0;
            else
                return 0.0;
        }
    };


    template <typename Nonterminal>
    struct InsideOutsideMultiplierAndNormalizer : boost::static_visitor<RuleTensor<double>> {
        const Element<HyperEdge<Nonterminal>> edge;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& insideWeights;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outsideWeights;
        const double normalize;

        InsideOutsideMultiplierAndNormalizer(
                const Element<HyperEdge<Nonterminal>> e,
                const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& inside,
                const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outside,
                const double norm
        )
                :
                edge(e), insideWeights(inside), outsideWeights(outside), normalize(norm)
        {}

        template<int rank>
        RuleTensor<double>
        operator()(const RuleTensorRaw<double, rank> &tensor){
            RuleTensor<double> intermediate {tensor};
            // Multiply all inside weights
            for(int dim = 0; dim < rank - 1 ; ++dim) {
                RuleTensorContractor tmult(insideWeights.at(edge->get_sources()[dim]), 1);
                intermediate = boost::apply_visitor(tmult, intermediate);
            }

            // multiply outside weight
            Eigen::Tensor<double, 0> ioSum = boost::get<RuleTensorRaw<double, 1>>(intermediate).contract(
                    outsideWeights.at(edge->get_target()), Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
            );

            Eigen::array<Eigen::Index, rank> dim;
            for(Eigen::Index i = 0; i < rank; ++i)
                dim[i] = 1;

            RuleTensorRaw<double, rank> res(dim);
            res.setConstant(safe_division(ioSum(0), normalize));

            return res;
        }
    };


    template <typename Nonterminal>
    struct InsideOutsideMultiplierAndNormalizerScale : boost::static_visitor<RuleTensor<double>> {
        const Element<HyperEdge<Nonterminal>> edge;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& insideWeights;
        const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outsideWeights;
        const MAPTYPE<Element<Node<Nonterminal>>, int>& insideLogScales;
        const MAPTYPE<Element<Node<Nonterminal>>, int>& outsideLogScales;
        const double normalize;
        const int normalizeLogScale;
        int & resultLogScale;

        InsideOutsideMultiplierAndNormalizerScale(
                const Element<HyperEdge<Nonterminal>> e
                , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& inside
                , const MAPTYPE<Element<Node<Nonterminal>>, Trainer::WeightVector>& outside
                , const MAPTYPE<Element<Node<Nonterminal>>, int>& insideLogScales
                , const MAPTYPE<Element<Node<Nonterminal>>, int>& outsideLogScales
                , const double normalize
                , const int normalizeLogScale
                , int & resultLogScale
        )
                : edge(e)
                , insideWeights(inside)
                , outsideWeights(outside)
                , insideLogScales(insideLogScales)
                , outsideLogScales(outsideLogScales)
                , normalize(normalize)
                , normalizeLogScale(normalizeLogScale)
                , resultLogScale(resultLogScale)
        {}

        template<int rank>
        RuleTensor<double>
        operator()(const RuleTensorRaw<double, rank> &tensor){
            resultLogScale = 0;
            RuleTensor<double> intermediate {tensor};
            // Multiply all inside weights
            for(int dim = 0; dim < rank - 1 ; ++dim) {
                RuleTensorContractor tmult(insideWeights.at(edge->get_sources()[dim]), 1);
                intermediate = boost::apply_visitor(tmult, intermediate);
                resultLogScale += insideLogScales.at(edge->get_sources()[dim]);
                resultLogScale = scaleTensor(intermediate, resultLogScale);
            }

            // multiply outside weight
            Eigen::Tensor<double, 0> ioSum = boost::get<RuleTensorRaw<double, 1>>(intermediate).contract(
                    outsideWeights.at(edge->get_target()), Eigen::array<Eigen::IndexPair<long>, 1>{Eigen::IndexPair<long>(0, 0)}
            );
            resultLogScale += outsideLogScales.at(edge->get_target());
            resultLogScale = scaleTensor(ioSum, resultLogScale);

            Eigen::array<Eigen::Index, rank> dim;
            for(Eigen::Index i = 0; i < rank; ++i)
                dim[i] = 1;

            RuleTensorRaw<double, rank> res(dim);

            double result = ioSum(0) / normalize;
            resultLogScale -= normalizeLogScale;

            resultLogScale = scaleScalar(result, resultLogScale);

            res.setConstant(result);

            return res;
        }
    };



    struct VectorSummer : boost::static_visitor<double> {
        const int index;

        VectorSummer(int index=0) : index(index) {}

        template<int rank>
        double operator()(const Eigen::Tensor<double, rank>& tensor) const {

            Eigen::Tensor<double, 0> sum = tensor.chip(index, 0).sum();
            return sum(0);
        }
    };



    template <typename Nonterminal>
    struct RuleSummerIO : boost::static_visitor<double> {

        const Element<HyperEdge<Nonterminal>> edge;
        const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector>& inside;
        const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector>& outside;

        RuleSummerIO(
                const Element<HyperEdge<Nonterminal>> edge
                , const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector>& inside
                , const MAPTYPE <Element<Node<Nonterminal>>, Trainer::WeightVector>& outside
        )
                :
                edge(edge)
                , inside(inside)
                , outside(outside)
        {};

        template<int rank>
        double operator()(const RuleTensorRaw<double, rank>& weight) const {

            RuleTensor<double> intermediate{RuleTensorRaw<double, rank>(weight.dimensions())};
            intermediate = weight;
            for(int dim = rank - 1; dim > 0 ; --dim) {
                RuleTensorContractor tmult(inside.at(edge->get_sources()[dim - 1]));
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

    struct TensorAdder : boost::static_visitor<void> {
        RuleTensor<double> & goal;

        TensorAdder(RuleTensor<double> & goal) : goal(goal) {}

        template<int rank>
        inline void operator()(const RuleTensorRaw<double, rank>& tensor) const {
            RuleTensorRaw<double, rank>& goalRaw = boost::get<RuleTensorRaw<double, rank>>(goal);
            goalRaw += tensor;
        }
    };




}

#endif //STERMPARSER_TRAININGCOMMON_H
