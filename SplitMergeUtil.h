//
// Created by kilian on 12/12/16.
//

#ifndef STERMPARSER_SPLITMERGEUTIL_H
#define STERMPARSER_SPLITMERGEUTIL_H

#include <boost/ptr_container/ptr_vector.hpp>

// new representation
unsigned indexation (const std::vector<unsigned> & positions, const std::vector<unsigned> & dimensions, bool half=false) {
    unsigned index = 0;
    assert(positions.size() == dimensions.size());
    for (unsigned i = 0; i < positions.size(); ++i){
        unsigned offset = 1;
        for (unsigned j = i + 1; j < dimensions.size(); ++j)
            offset *= dimensions[j];
        if (half)
            index += positions[i] / 2 * offset;
        else
            index += positions[i] * offset;
    }
    return index;
}

double weight(const double * const weights, const std::vector<unsigned> & positions, const std::vector<unsigned> & dimensions) {
    unsigned index = indexation(positions, dimensions);
    return weights[index];
}


double rand_split() {
    return 0.55;
}

void fill_split(const double * const old_weights, double * const new_weights, const std::vector<unsigned> & dimensions
        , std::vector<unsigned> & selection, const unsigned dim) {
    if (dimensions.size() == dim) {
        unsigned origin_index = indexation(selection, dimensions, true);
        unsigned index = indexation(selection, dimensions);
        // TODO log likelihoods
        double split_weight = old_weights[origin_index] * rand_split();
        new_weights[index] = split_weight;
    }
    else {
        selection.push_back(0);
        for (unsigned i = 0; i < dimensions[dim]; ++i) {
            selection[dim] = 2 * i;
            fill_split(old_weights, new_weights, dimensions, selection, dim + 1);
            selection[dim] = 2 * i + 1;
            fill_split(old_weights, new_weights, dimensions, selection, dim + 1);
        }
        selection.pop_back();
    }
}

double * split_rule(const double * const weights, const std::vector<unsigned> & dimensions) {
    unsigned new_size = 1;
    for (auto dim : dimensions) {
        new_size *= dim * 2;
    }
    double * splits = (double *) malloc(sizeof(double) * new_size);
    std::vector<unsigned> selection;
    fill_split(weights, splits, dimensions, selection, 0);
    return splits;
}

void accumulate_probabilites( double * const goal_value
                              , const double * const old_weights
        , const std::vector<unsigned> & old_dimensions
        , std::vector<unsigned> & selection, const unsigned dim
        , const boost::ptr_vector<std::vector<unsigned>> & merges
        , const double * const lhn_merge_weights) {
    if (dim == old_dimensions.size()) {
        *goal_value += weight(old_weights, selection, old_dimensions) * lhn_merge_weights[selection[0]];
    }
    else {
        assert(merges[dim].size() > 0);
        for (auto value : merges[dim]) {
            selection.push_back(value);
            accumulate_probabilites(goal_value, old_weights, old_dimensions, selection, dim+1, merges, lhn_merge_weights);
            selection.pop_back();
        }
    }
}

void fill_merge(const double * const weights, double * merged_weights, const std::vector<unsigned> & old_dimensions
        , const std::vector<unsigned> & new_dimensions, std::vector<unsigned> & selection, const unsigned dim,
                const boost::ptr_vector<std::vector<std::vector<unsigned>>> & merges, const double * const lhn_merge_weights
) {
    if (new_dimensions.size() == dim) {
        unsigned index = indexation(selection, new_dimensions);
        double * goal_value = merged_weights + index;
        std::vector<unsigned> witnesses;
        boost::ptr_vector<std::vector<unsigned>> the_merges;
        the_merges.reserve(selection.size());
        for (unsigned i; i < selection.size(); ++i) {
            the_merges[i] = merges[i][selection[i]];
        }
        accumulate_probabilites(goal_value, weights, old_dimensions, witnesses, 0, the_merges, lhn_merge_weights);
    }
    else {
        assert(new_dimensions[dim] > 0);
        for (unsigned i = 0; i < new_dimensions[dim]; ++i){
            selection.push_back(i);
            fill_merge(weights, merged_weights, old_dimensions, new_dimensions, selection, dim + 1, merges, lhn_merge_weights);
            selection.pop_back();
        }
    }
}

double * merge_rule(  const double * const weights
                    , const std::vector<unsigned> & old_dimensions
                    , const std::vector<unsigned> & new_dimensions
                    , const boost::ptr_vector<std::vector<std::vector<unsigned>>> & merges
                    , const double * const lhn_merge_weights
                    ) {
    unsigned new_size = 1;
    for (auto dim : new_dimensions) {
        new_size *= dim * 2;
    }

    double * merged_weights = (double *) malloc(sizeof(double) * new_size);
    std::vector<unsigned> selection;
    fill_merge(weights, merged_weights, old_dimensions, new_dimensions, selection, 0, merges, lhn_merge_weights);
}

unsigned subdim(const std::vector<unsigned> & dims, const unsigned target = 0) {
    unsigned dim = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        if (i != target)
            dim += dims[i];
    }
    return dim;
}

template <typename Accum1, typename Accum2>
std::vector<double> compute_inside_weights(const double * const rule_weight_tensor
        , const boost::ptr_vector<std::vector<unsigned>> & nont_weight_vectors
        , const std::vector<unsigned> & dim_rule, const double zero
        , const Accum1 sum , Accum2 prod) {
    std::vector<double> result = std::vector<double>(dim_rule[0], zero);
        const double * next_la_weight = rule_weight_tensor;
        for (unsigned lhs = 0; lhs < dim_rule[0]; ++ lhs) {
            assert(next_la_weight - rule_weight_tensor == lhs * subdim(dim_rule));
            double & target = result[lhs];

            unsigned rhs = 0;
            std::vector<unsigned> selection = std::vector<unsigned>(dim_rule.size() - 1, 0);
            std::vector<double> factors;
            while (rhs >= 0) {
                if (rhs == dim_rule.size() - 1) {
                    double val = prod(next_la_weight, factors[rhs - 1]);
                    target = sum(target, val);
                    ++next_la_weight;
                    rhs--;
                } else if (selection[rhs] == nont_weight_vectors[rhs].size()) {
                        selection[rhs] = 0;
                        rhs--;
                } else {
                    if (rhs > 0)
                        factors[rhs] = prod(factors[rhs-1], nont_weight_vectors[rhs][selection[rhs]]);
                    else
                        factors[rhs] = nont_weight_vectors[rhs][selection[rhs]];
                    ++selection[rhs];
                    ++rhs;
                }
            }
        }
    return result;
}


template <typename Accum1, typename Accum2>
std::vector<double> compute_outside_weights(const double * const rule_weight_tensor
        , const std::vector<unsigned> & lhn_outside_weight_vector
        , const boost::ptr_vector<std::vector<unsigned>> & inside_weight_vectors
        , const std::vector<unsigned> & dim_rule, const double zero, const double one
        , const Accum1 sum , Accum2 prod, const unsigned target_pos) {

    std::vector<double> result = std::vector<double>(dim_rule[target_pos], zero);

    const double * next_la_weight = rule_weight_tensor;
    for (unsigned lhs = 0; lhs < dim_rule[0]; ++lhs) {
        assert(next_la_weight - rule_weight_tensor == lhs * subdim(dim_rule, 0));

        unsigned rhs = 0;
        std::vector<unsigned> selection = std::vector<unsigned>(dim_rule.size() - 1, 0);
        std::vector<double> factors;
        while (rhs >= 0) {
            if (rhs == dim_rule.size() - 1) {
                double val = prod(next_la_weight, factors[rhs - 1]);
                result[target_pos] = sum(result[target_pos], val);
                ++next_la_weight;
                rhs--;
            } else if (selection[rhs] == inside_weight_vectors[rhs].size()) {
                selection[rhs] = 0;
                rhs--;
            } else {
                if (rhs == target_pos) {
                    factors[rhs] = prod(factors[rhs-1], one);
                }
                if (rhs > 0)
                    factors[rhs] = prod(factors[rhs-1], inside_weight_vectors[rhs][selection[rhs]]);
                else
                    factors[rhs] = inside_weight_vectors[rhs][selection[rhs]];
                ++selection[rhs];
                ++rhs;
            }
        }
    }
    return result;
}

template<typename Accum1, typename Val>
std::vector<Val> dot_product(const Accum1 point_product, const std::vector<Val> & arg1, const std::vector<Val> &arg2) {
    assert(arg1.size() == arg2.size());
    std::vector<Val> result;
    for (unsigned i = 0; i < arg1.size(); ++i) {
        result.push_back(point_product(arg1[i], arg2[i]));
    }
    return result;
};

template<typename Accum1, typename Val>
std::vector<Val> scalar_product(const Accum1 point_scalar, const std::vector<Val> &goal, const Val scalar) {
    std::vector<Val> result;
    for (auto i = goal.begin(); i != goal.end(); ++i) {
        result.push_back(point_scalar(*i, scalar));
    }
    return result;
};

template<typename Accum, typename Val>
Val reduce(const Accum accum, const std::vector<Val> vec, Val init, const unsigned start = 0) {
    for (unsigned i = start; i < vec.size(); ++i)
        init = accum(vec[i], init);
    return init;
};

#endif //STERMPARSER_SPLITMERGEUTIL_H
