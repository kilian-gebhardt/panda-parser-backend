//
// Created by kilian on 23/01/17.
//

#ifndef STERMPARSER_EIGENUTIL_H
#define STERMPARSER_EIGENUTIL_H
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "SplitMergeUtil.h"

template<typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 1> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    return rule_las;
};
template<typename Val, typename TENSORTYPE>
const Eigen::TensorMap<Eigen::Tensor<Val, 1>> & rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, 1>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    return rule_las;
};

template<typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 2> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto contraction = rule_las.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return contraction;
}
template<typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, 2>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto contraction = rule_las.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return contraction;
}


template<typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 3> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c2 = c1.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c2;
}
template<typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, 3>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c2 = c1.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c2;
}

template<typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 4> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[2], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(3, 0)}));
    auto c2 = c1.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c3 = c2.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c3;
}
template<typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, 4>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[2], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(3, 0)}));
    auto c2 = c1.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c3 = c2.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c3;
}


template <int N, typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, N> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    assert((N > 1) && (rhs_lass.size() > N-2));
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(N - 1, 0)};
    Eigen::Tensor<Val, N-1> contraction = rule_las.contract(rhs_lass[N - 2], product_dims);

    return rule_probs(contraction, rhs_lass);
};
template <int N, typename Val, typename TENSORTYPE>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::TensorMap<Eigen::Tensor<Val, N>> & rule_las, const std::vector<TENSORTYPE> & rhs_lass) {
    assert((N > 1) && (rhs_lass.size() > N-2));
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(N - 1, 0)};
    Eigen::Tensor<Val, N-1> contraction = rule_las.contract(rhs_lass[N - 2], product_dims);

    return rule_probs(contraction, rhs_lass);
};

template<int rule_dim, typename Witness, typename NontToIdx, typename MAP>
void foo(const std::vector<std::vector<unsigned>> & rule_dimensions
        , const int rule_id
        , const std::vector<double*> & rule_weights
        , Witness & witness
        , const std::vector<unsigned> & nont_dimensions
        , const Eigen::TensorMap<Eigen::Tensor<double, 1>> & lhn_outside_weight
        , Eigen::Tensor<double, 1> & trace_root_probability
        , NontToIdx nont_idx
        , const MAP & tr_io_weight
        , std::vector<double*> & rule_counts
) {
    Eigen::array<long, rule_dim> rule_dimension;
    for (unsigned i = 0; i < rule_dimensions[rule_id].size(); ++i) {
        rule_dimension[i] = rule_dimensions[rule_id][i];
    }

    const Eigen::TensorMap<Eigen::Tensor<double, rule_dim>> rule_weight (rule_weights[rule_id], rule_dimension);

    Eigen::array<long, rule_dim> rshape_dim;
    Eigen::array<long, rule_dim> broad_dim;
    for (unsigned i = 0; i < rule_dim; ++i) {
        rshape_dim[i] = 1;
        broad_dim[i] = rule_dimension[i];
    }

    Eigen::Tensor<double, rule_dim> rule_val = rule_weight;
    for (unsigned i = 0; i < rule_dim; ++i) {
        const unsigned item_dim = rule_dimension[i];
        const auto item_weight = (i == 0)
                        ? lhn_outside_weight
                        : Eigen::TensorMap<Eigen::Tensor<double, 1>>(std::get<0>(tr_io_weight).at(*witness.second[i - 1]), item_dim);
        rshape_dim[i] = broad_dim[i];
        broad_dim[i] = 1;
        rule_val *= item_weight.reshape(rshape_dim).broadcast(broad_dim);
        broad_dim[i] = rshape_dim[i];
        rshape_dim[i] = 1;
    }

    rshape_dim[0] = broad_dim[0];
    broad_dim[0] = 1;

    rule_val /= (trace_root_probability.reshape(rshape_dim).broadcast(broad_dim));

    // std::cerr << "rule val: " << std::endl << rule_val << std::endl;

    Eigen::TensorMap<Eigen::Tensor<double, rule_dim>> rule_count(rule_counts[rule_id], rule_dimension);
    rule_count += rule_val;
}

void maximization(const unsigned lhs_dim, const std::vector<std::vector<unsigned>>& rule_dimensions, const std::vector<unsigned> & group, const std::vector<double *> & rule_counts, std::vector<double *> & rule_probabilites) {
    Eigen::Tensor<double, 1> lhs_counts (lhs_dim);
    lhs_counts.setZero();

    for (const auto rule : group) {
        const unsigned block_size = subdim(rule_dimensions[rule]);
        const Eigen::TensorMap<Eigen::Tensor<double, 2>> rule_count(rule_counts[rule], lhs_dim, block_size);
        lhs_counts += rule_count.sum(Eigen::array<long, 1>({1}));
    }
    for (const auto rule : group) {
        const unsigned block_size = subdim(rule_dimensions[rule]);
        const Eigen::TensorMap<Eigen::Tensor<double, 2>> rule_count(rule_counts[rule], lhs_dim, block_size);
        Eigen::TensorMap<Eigen::Tensor<double, 2>> rule_probability(rule_probabilites[rule], lhs_dim, block_size);

        for (unsigned dim = 0; dim < lhs_dim; ++dim)
            if (lhs_counts(dim) > 0) {
                rule_probability.chip(dim, 0) = rule_count.chip(dim, 0).unaryExpr([&](const double x) -> double { return x / lhs_counts(dim); });
            }
    }
}

template<int max_dim, typename Val>
void convert_format(double * const rule_ptr, const std::vector<unsigned> & rule_dim, const std::vector<Val> & weights) {
    Eigen::array<long, max_dim> rule_dim_a;
    for(unsigned dim = 0; dim < max_dim; ++dim) {
        rule_dim_a[dim] = rule_dim[dim];
    }
    Eigen::array<long, max_dim> weight_position;
    Eigen::TensorMap<Eigen::Tensor<double, max_dim>> rule_weight_tensor(rule_ptr, rule_dim_a);

//    std::cerr << rule_weight_tensor << std::endl;

    for(unsigned dim = 0; dim < max_dim; ++dim) {
        weight_position[dim] = -1;
    }

    unsigned dim = 0;
    auto weight_it = weights.begin();
//    unsigned counter = 0;

    while (dim < max_dim) {
        if (dim == max_dim - 1 and weight_position[dim] + 1 < rule_dim[dim]) {
            ++weight_position[dim];
            assert(weight_it != weights.end());
            rule_weight_tensor(weight_position) =  weight_it->from();
//            std::cerr << counter << " ";
//            for (auto val : weight_position) std::cerr << val << " ";
//            std::cerr << weight_it->from() << std::endl;
//            ++counter;
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

//    std::cerr << rule_weight_tensor;

    assert(weight_it == weights.end());
}

template<int max_dim, typename Val>
void de_convert_format(double * const rule_ptr, const std::vector<unsigned> & rule_dim, std::vector<Val> & weights) {
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


    assert(weight_it == weights.end());
}

#endif //STERMPARSER_EIGENUTIL_H
