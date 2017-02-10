//
// Created by kilian on 23/01/17.
//

#ifndef STERMPARSER_EIGENUTIL_H
#define STERMPARSER_EIGENUTIL_H
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "SplitMergeUtil.h"
#include <boost/variant.hpp>

template<typename Val, typename TENSORTYPE>
inline Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 1> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    assert(rhs_lass.size() == 0);
    return rule_las;
};
template<typename Val, typename TENSORTYPE>
inline const Eigen::TensorMap<Eigen::Tensor<Val, 1>> & rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, 1>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    return rule_las;
};

template<typename Val, typename TENSORTYPE>
inline Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 2> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto contraction = rule_las.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return contraction;
}
template<typename Val, typename TENSORTYPE>
inline Eigen::Tensor<Val, 1> rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, 2>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto contraction = rule_las.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return contraction;
}


template<typename Val, typename TENSORTYPE>
inline Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 3> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c2 = c1.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c2;
}
template<typename Val, typename TENSORTYPE>
inline Eigen::Tensor<Val, 1> rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, 3>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c2 = c1.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c2;
}

template<typename Val, typename TENSORTYPE>
inline Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 4> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[2], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(3, 0)}));
    auto c2 = c1.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c3 = c2.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c3;
}
template<typename Val, typename TENSORTYPE>
inline Eigen::Tensor<Val, 1> rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, 4>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[2], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(3, 0)}));
    auto c2 = c1.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c3 = c2.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c3;
}


template <typename Scalar>
using RuleTensor = typename boost::variant<
          Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 5>>
        , Eigen::TensorMap<Eigen::Tensor<Scalar, 6>>
        >;

inline void compute_rule_count1(const RuleTensor<double> & rule_weight_tensor, const Eigen::TensorMap<Eigen::Tensor<double, 1>> &lhn_outside_weight,
                                const double trace_root_probability, RuleTensor<double> & rule_count_tensor
) {
    constexpr unsigned rule_rank {1};

    const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_weight_tensor);

    auto rule_val = rule_weight * lhn_outside_weight;

    Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_count = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_count_tensor);

    if (trace_root_probability > 0) {
        rule_count += rule_val * (1 / trace_root_probability);
    }
}

template<typename Witness, typename MAP>
inline void compute_rule_count2(const RuleTensor<double> & rule_weight_tensor, Witness &witness,
                               const Eigen::TensorMap<Eigen::Tensor<double, 1>> &lhn_outside_weight,
                               const double trace_root_probability, const MAP &inside_weights,
                               RuleTensor<double> & rule_count_tensor
) {
    constexpr unsigned rule_rank {2};

    const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_weight_tensor);

    const Eigen::TensorMap<Eigen::Tensor<double, 1>> & rhs_weight = inside_weights.at(*witness.second[0]);
    auto rule_val = lhn_outside_weight.reshape(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1})
                                                  .broadcast(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1)})
                    * rhs_weight.reshape(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1)})
                                                                        .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1}).eval()
                    * rule_weight
                    ;

    Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_count = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_count_tensor);

    if (trace_root_probability > 0) {
        rule_count += rule_val * (1 / trace_root_probability);
    }
}

template<typename Witness, typename MAP>
inline void compute_rule_count3(const RuleTensor<double> & rule_weight_tensor, Witness &witness,
                                const Eigen::TensorMap<Eigen::Tensor<double, 1>> &lhn_outside_weight,
                                const double trace_root_probability, const MAP &inside_weights,
                                RuleTensor<double> & rule_count_tensor
) {
    constexpr unsigned rule_rank {3};

    const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_weight_tensor);
    const Eigen::TensorMap<Eigen::Tensor<double, 1>> & rhs_weight1 = inside_weights.at(*witness.second[0]);
    const Eigen::TensorMap<Eigen::Tensor<double, 1>> & rhs_weight2 = inside_weights.at(*witness.second[1]);

    auto rule_val = lhn_outside_weight.reshape(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1, 1})
                            .broadcast(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1), rule_weight.dimension(2)})
                    * rhs_weight1.reshape(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1), 1})
                            .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1, rule_weight.dimension(2)}).eval()
                    * rhs_weight2.reshape(Eigen::array<long, rule_rank>{1, 1, rule_weight.dimension(2)})
                            .broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), rule_weight.dimension(1), 1}).eval()
                    * rule_weight;
    ;

    Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_count = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rule_count_tensor);

    if (trace_root_probability > 0) {
        rule_count += rule_val * (1 / trace_root_probability);
    }
}

template<int rule_dim, typename Witness, typename MAP>
inline void compute_rule_count(
                        const RuleTensor<double> & rule_weight_tensor,
                        const Witness &witness,
                        const Eigen::TensorMap<Eigen::Tensor<double, 1>> &lhn_outside_weight,
                        const double trace_root_probability, const MAP &inside_weights,
                        RuleTensor<double> & rule_count_tensor
//                        double * const rule_count_ptr
) {

    const Eigen::TensorMap<Eigen::Tensor<double, rule_dim>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_dim>>>(rule_weight_tensor);
    const Eigen::array<long, rule_dim> & rule_dimension = rule_weight.dimensions();

    Eigen::array<long, rule_dim> rshape_dim;
    Eigen::array<long, rule_dim> broad_dim;
    for (unsigned i = 0; i < rule_dim; ++i) {
        rshape_dim[i] = 1;
        broad_dim[i] = rule_dimension[i];
    }

    Eigen::Tensor<double, rule_dim> rule_val = rule_weight;
    for (unsigned i = 0; i < rule_dim; ++i) {
        const auto & item_weight = (i == 0)
                        ? lhn_outside_weight
                        : inside_weights.at(*witness.second[i - 1]);
        rshape_dim[i] = broad_dim[i];
        broad_dim[i] = 1;
        rule_val *= item_weight.reshape(rshape_dim).broadcast(broad_dim);
        broad_dim[i] = rshape_dim[i];
        rshape_dim[i] = 1;
    }


    Eigen::TensorMap<Eigen::Tensor<double, rule_dim>> & rule_count = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_dim>>>(rule_count_tensor);

//    Eigen::TensorMap<Eigen::Tensor<double, rule_dim>> rule_count (rule_count_ptr, rule_dimension);

    if (trace_root_probability > 0) {
        rule_count += rule_val * (1 / trace_root_probability);
    }
}

inline void maximization(const unsigned lhs_dim, const std::vector<std::vector<unsigned>>& rule_dimensions, const std::vector<unsigned> & group, const std::vector<double *> & rule_counts, std::vector<double*> & rule_probabilites) {
    Eigen::Tensor<double, 1> lhs_counts (lhs_dim);
    lhs_counts.setZero();

    for (const auto rule : group) {
        const unsigned block_size = subdim(rule_dimensions[rule]);
        Eigen::TensorMap<Eigen::Tensor<double, 2>> rule_count(rule_counts[rule], lhs_dim, block_size);
        lhs_counts += rule_count.sum(Eigen::array<long, 1>({1}));
    }
    for (const auto & rule : group) {
        const unsigned block_size = subdim(rule_dimensions[rule]);
        const Eigen::TensorMap<Eigen::Tensor<double, 2>> rule_count(rule_counts[rule], lhs_dim, block_size);
        Eigen::TensorMap<Eigen::Tensor<double, 2>> rule_probability(rule_probabilites[rule], lhs_dim, block_size);

        for (unsigned dim = 0; dim < lhs_dim; ++dim)
            if (lhs_counts(dim) > 0) {
                rule_probability.chip(dim, 0) = rule_count.chip(dim, 0) * (1 / lhs_counts(dim));
            }
    }
}

template<int rule_rank>
inline RuleTensor<double> createTensor(double * const storage, const std::vector<unsigned> & rule_dimension) {
    Eigen::array<long, rule_rank> rule_dim_a;
    for(unsigned dim = 0; dim < rule_rank; ++dim) {
        rule_dim_a[dim] = rule_dimension[dim];
    }
    return Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>(storage, rule_dim_a);
}

template<int max_dim, typename Val>
inline void convert_format(double * const rule_ptr, const std::vector<unsigned> & rule_dim, const std::vector<Val> & weights, std::vector<RuleTensor<double>> & rule_tensors) {
    Eigen::array<long, max_dim> rule_dim_a;
    for(unsigned dim = 0; dim < max_dim; ++dim) {
        rule_dim_a[dim] = rule_dim[dim];
    }
    Eigen::array<long, max_dim> weight_position;
    Eigen::TensorMap<Eigen::Tensor<double, max_dim>> rule_weight_tensor(rule_ptr, rule_dim_a);

    for(unsigned dim = 0; dim < max_dim; ++dim) {
        weight_position[dim] = -1;
    }

    unsigned dim = 0;
    auto weight_it = weights.begin();

    while (dim < max_dim) {
        if (dim == max_dim - 1 and weight_position[dim] + 1 < rule_dim[dim]) {
            ++weight_position[dim];
            assert(weight_it != weights.end());
            rule_weight_tensor(weight_position) =  weight_it->from();
            ++weight_it;
        } else if (weight_position[dim] + 1 == rule_dim[dim]) {
            if (dim > 0) {
                for (unsigned dim_ = dim; dim_ < max_dim; ++dim_) {
                    weight_position[dim_] = -1;
                }
                --dim;
            }
            else
                break;
        } else if (weight_position[dim] + 1 < rule_dim[dim]) {
            ++weight_position[dim];
            ++dim;
        }
    }

    if(weight_it != weights.end()) {
        std::cerr << "conversion error.";
        abort();
    }

    rule_tensors.emplace_back(std::move(rule_weight_tensor));
}

template<int max_dim, typename Val>
inline void de_convert_format(double * const rule_ptr, const std::vector<unsigned> & rule_dim, std::vector<Val> & weights) {
    Eigen::array<long, max_dim> rule_dim_a;
    for(unsigned dim = 0; dim < max_dim; ++dim) {
        rule_dim_a[dim] = rule_dim[dim];
    }
    Eigen::array<long, max_dim> weight_position;
    Eigen::TensorMap<Eigen::Tensor<double, max_dim>> rule_weight_tensor(rule_ptr, rule_dim_a);


    for(unsigned dim = 0; dim < max_dim; ++dim) {
        weight_position[dim] = -1;
    }

    unsigned dim = 0;
    auto weight_it = weights.begin();

    while (dim < max_dim) {
        if (dim == max_dim - 1 and weight_position[dim] + 1 < rule_dim[dim]) {
            ++weight_position[dim];
            assert(weight_it != weights.end());
            *weight_it = Val::to(rule_weight_tensor(weight_position));

            ++weight_it;
        } else if (weight_position[dim] + 1 == rule_dim[dim]) {
            if (dim > 0) {
                for (unsigned dim_ = dim; dim_ < max_dim; ++dim_) {
                    weight_position[dim_] = -1;
                }
                --dim;
            }
            else
                break;
        } else if (weight_position[dim] + 1 < rule_dim[dim]) {
            ++weight_position[dim];
            ++dim;
        }
    }

    if(weight_it != weights.end()) {
        std::cerr << "conversion error.";
        abort();
    }
}

#endif //STERMPARSER_EIGENUTIL_H
