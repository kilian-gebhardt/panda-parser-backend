//
// Created by kilian on 12/12/16.
//

#ifndef STERMPARSER_SPLITMERGEUTIL_H
#define STERMPARSER_SPLITMERGEUTIL_H

#include <boost/ptr_container/ptr_vector.hpp>
#include <stdlib.h>
#include <iostream>
#include "util.h"
#include <cmath>

/*
#ifdef NDEBUG
# define NDEBUG_DISABLED
# undef NDEBUG
#endif
#include <cassert>
*/

// new representation
unsigned indexation (const std::vector<unsigned> & positions, const std::vector<unsigned> & dimensions, bool half=false) {
    unsigned index = 0;
    assert(positions.size() == dimensions.size());
    for (unsigned i = 0; i < positions.size(); ++i){
        unsigned offset = 1;
        for (unsigned j = i + 1; j < dimensions.size(); ++j) {
            if (half)
                offset *= dimensions[j] / 2;
            else
                offset *= dimensions[j];
        }
        if (half)
            index += (positions[i] / 2) * offset;
        else
            index += positions[i] * offset;
    }
    return index;
}

template<typename Val>
Val weight(const std::vector<Val> & weights, const std::vector<unsigned> & positions, const std::vector<unsigned> & dimensions) {
    unsigned index = indexation(positions, dimensions);
    assert (index < weights.size());
    return weights[index];
}

double fRand(double fMin, double fMax)
{
    double f = (double) rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double rand_split() {
    return fRand(0.95, 1.05);
//    if (id % 2)
//        return 0.55;
//    else
//        return 0.45;
}

template<typename Val>
void fill_split(const std::vector<Val> & old_weights, std::vector<Val> & new_weights, const std::vector<unsigned> & dimensions
        , std::vector<unsigned> & selection, const unsigned dim) {
    if (dimensions.size() == dim) {
        unsigned origin_index = indexation(selection, dimensions, true);
        unsigned index = indexation(selection, dimensions);
        assert (origin_index < old_weights.size());
        assert (index < new_weights.size());
        Val split_weight = old_weights[origin_index] * Val::to(rand_split() * std::pow<double>(0.5, dim - 1));
        new_weights[index] = split_weight;
    } else {
        selection.push_back(0);
        for (unsigned i = 0; i < dimensions[dim]; i = i + 2) {
            selection[dim] = i;
            fill_split(old_weights, new_weights, dimensions, selection, dim + 1);
            selection[dim] = i + 1;
            fill_split(old_weights, new_weights, dimensions, selection, dim + 1);
        }
        selection.pop_back();
    }
}


unsigned calc_size(const std::vector<unsigned> & dims) {
    unsigned size = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        size *= dims[i];
    }
    return size;
}

template<typename Val>
std::vector<Val> split_rule(const std::vector<Val> & weights, const std::vector<unsigned> & dimensions) {
    unsigned new_size = calc_size(dimensions);
    std::vector<Val> splits = std::vector<Val>(new_size, Val::one());
    std::vector<unsigned> selection;
    fill_split(weights, splits, dimensions, selection, 0);
    return splits;
}

template <typename Val>
void accumulate_probabilites(const typename std::vector<Val>::iterator goal_value
                              , const std::vector<Val> & split_weights
        , const std::vector<unsigned> & old_dimensions
        , std::vector<unsigned> & selection, const unsigned dim
        , const std::vector<std::reference_wrapper<const std::vector<unsigned>>> & merges
        , const std::vector<Val> & lhn_merge_weights) {
    assert(dim == selection.size());
    if (dim == old_dimensions.size()) {
        /*
        std::cerr << "witness la: ";
        for (auto i : selection) {
            std::cerr << i <<" : ";
        }*/
        const auto val = (weight(split_weights, selection, old_dimensions) * lhn_merge_weights[selection[0]]);
        // std::cerr << val << std::endl;
        *goal_value = (*goal_value) + (val);
    }
    else {
        assert(merges.at(dim).get().size() > 0);
        for (auto value : merges[dim].get()) {
            selection.push_back(value);
            accumulate_probabilites(goal_value, split_weights, old_dimensions, selection, dim+1, merges, lhn_merge_weights);
            selection.pop_back();
        }
    }
}

/*
template <typename Val>
void fill_merge(const std::vector<Val> & split_weights, std::vector<Val> & merged_weights, const std::vector<unsigned> & old_dimensions
        , const std::vector<unsigned> & new_dimensions, std::vector<unsigned> & selection, const unsigned dim,
                const std::vector<std::vector<std::vector<unsigned>>> & merges, const std::vector<Val> & lhn_merge_weights
) {
    assert(dim == selection.size());
    if (new_dimensions.size() == dim) {
        unsigned index = indexation(selection, new_dimensions);
        std::vector<double>::iterator goal_value = merged_weights.begin() + index;
        std::vector<unsigned> witnesses;
        std::vector<std::vector<unsigned>> the_merges;
        for (unsigned i = 0; i < selection.size(); ++i) {
            the_merges.push_back(merges[i][selection[i]]);
        }

//        std::cerr << "merge   la: ";
//        for (auto i : selection) {
//            std::cerr << i <<" : ";
//        }
//        std::cerr << std::endl;
        accumulate_probabilites(goal_value, split_weights, old_dimensions, witnesses, 0, the_merges, lhn_merge_weights);
    }
    else {
        assert(new_dimensions[dim] > 0);
        for (unsigned i = 0; i < new_dimensions[dim]; ++i){
            selection.push_back(i);
            fill_merge(split_weights, merged_weights, old_dimensions, new_dimensions, selection, dim + 1, merges, lhn_merge_weights);
            selection.pop_back();
        }
    }
}
*/


template <typename Val>
void fill_merge2(const std::vector<Val> & split_weights, std::vector<Val> & merged_weights, const std::vector<unsigned> & old_dimensions
        , const std::vector<unsigned> & new_dimensions, std::vector<unsigned> & selection,
                const std::vector<std::vector<std::vector<unsigned>>> & merges, const std::vector<Val> & lhn_merge_weights
) {
    unsigned dim = 0;
    selection = std::vector<unsigned>(new_dimensions.size(), 0);
    while (true) {
        if (dim == new_dimensions.size()) {
            unsigned index = indexation(selection, new_dimensions);
            typename std::vector<Val>::iterator goal_value = merged_weights.begin() + index;
            assert (goal_value < merged_weights.end());
            std::vector<unsigned> witnesses;
            std::vector<std::reference_wrapper<const std::vector<unsigned>>> the_merges;
            for (unsigned i = 0; i < selection.size(); ++i) {
                the_merges.push_back(
                        (std::reference_wrapper<const std::vector<unsigned int>> &&) merges[i][selection[i]]);
            }

            /*
            std::cerr << "merge   la: ";
            for (auto i : selection) {
                std::cerr << i <<" : ";
            }
            std::cerr << std::endl;
            */
            accumulate_probabilites(goal_value, split_weights, old_dimensions, witnesses, 0, the_merges, lhn_merge_weights);
            if (dim == 0)
                break;
            dim--;
            selection[dim]++;
        } else if (selection[dim] == new_dimensions[dim]) {
            selection[dim] = 0;
            if (dim == 0)
                break;
            dim--;
            selection[dim]++;
        } else {
            ++dim;
        }
    }
    selection.clear();
}

template <typename Val>
std::vector<Val> merge_rule(  const std::vector<Val> & split_weights
                    , const std::vector<unsigned> & old_dimensions
                    , const std::vector<unsigned> & new_dimensions
                    , const std::vector<std::vector<std::vector<unsigned>>> & merges
                    , const std::vector<Val> & lhn_merge_factors
                    ) {
    unsigned new_size = calc_size(new_dimensions);
    /*
    std::cerr << "sw: ";
    for (auto sw : split_weights) {
        std::cerr << sw << " ";
    }
    std::cerr << std::endl;
    std::cerr << "mf: ";
    for (auto mf : lhn_merge_factors) {
        std::cerr << mf << " ";
    }
    std::cerr << std::endl;*/
    std::vector<Val> merged_weights = std::vector<Val>(new_size, Val::zero());
    std::vector<unsigned> selection;
    fill_merge2(split_weights, merged_weights, old_dimensions, new_dimensions, selection, merges, lhn_merge_factors);

    /*std::cerr << "mw: ";
    for (auto mw : merged_weights) {
        std::cerr << mw << " ";
    }
    std::cerr << std::endl;*/
    return merged_weights;
}


unsigned subdim(const std::vector<unsigned> & dims, const unsigned target = 0) {
    unsigned dim = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        if (i != target)
            dim *= dims[i];
    }
    return dim;
}

template <typename T>
unsigned subdim_from(const std::vector<std::vector<T>> & dims, const unsigned start) {
    unsigned dim = 1;
    for (auto i = dims.begin() + start; i != dims.end(); ++i) {
        dim *= i->size();
    }
    return dim;
}

template <typename Val>
std::vector<Val> compute_inside_weights(const std::vector<Val> & rule_weight_tensor
        , const std::vector<std::vector<Val>> & nont_weight_vectors
        , const std::vector<unsigned> & dim_rule) {
    std::vector<Val> result = std::vector<Val>(dim_rule[0], Val::zero());
        typename std::vector<Val>::const_iterator next_la_weight = rule_weight_tensor.begin();
        for (unsigned lhs = 0; lhs < dim_rule[0]; ++ lhs) {
            const unsigned sd = subdim(dim_rule);
            assert(next_la_weight == rule_weight_tensor.begin() + lhs * sd);
            Val & target = result[lhs];

            unsigned rhs = 0;
            std::vector<unsigned> selection = std::vector<unsigned>(dim_rule.size() - 1, 0);
            std::vector<Val> factors;
            factors.reserve(dim_rule.size() - 1);
            while (rhs >= 0) {
                if (rhs == dim_rule.size() - 1) {
                    assert(next_la_weight != rule_weight_tensor.end());
                    const Val val = (*next_la_weight) * (rhs > 0 ? factors[rhs - 1] : Val::one());
                    target = target + val;
                    ++next_la_weight;
                    if (rhs == 0)
                        break;
                    rhs--;
                } else if (selection[rhs] == nont_weight_vectors[rhs].size()) {
                        selection[rhs] = 0;
                        if (rhs == 0)
                            break;
                        rhs--;
                } else {
                    if (rhs > 0)
                        factors[rhs] = (factors[rhs-1]) * (nont_weight_vectors[rhs][selection[rhs]]);
                    else
                        factors[rhs] = nont_weight_vectors[rhs][selection[rhs]];
                    ++selection[rhs];
                    // if a factor is zero already, then we can abort early
                    // but we have to increase next_la_weight accordingly
                    if (factors[rhs] == Val::zero()) {
                        unsigned skip = subdim_from(nont_weight_vectors, rhs+1);
                        next_la_weight += skip;
                    } else {
                        ++rhs;
                    }
                }
            }
        }
    assert(next_la_weight == rule_weight_tensor.end());
    return result;
}


template <typename Val>
std::vector<Val> compute_rule_frequency(const std::vector<Val> & rule_weight_tensor
        , const std::vector<Val> & lhs_outside_weights
        , const std::vector<std::vector<Val>> & rhs_inside_weights
        , const std::vector<unsigned> & dim_rule) {
    std::vector<Val> result = std::vector<Val>(calc_size(dim_rule), Val::zero());
    typename std::vector<Val>::const_iterator next_la_weight = rule_weight_tensor.begin();
    typename std::vector<Val>::iterator target = result.begin();
    for (unsigned lhs = 0; lhs < dim_rule[0]; ++ lhs) {
        const unsigned sd = subdim(dim_rule);
        assert(next_la_weight == rule_weight_tensor.begin() + lhs * sd);
        assert(target == result.begin() + lhs * sd);

        unsigned rhs = 0;
        std::vector<unsigned> selection = std::vector<unsigned>(dim_rule.size() - 1, 0);
        std::vector<Val> factors = std::vector<Val>(dim_rule.size() - 1);
        while (rhs >= 0) {
            if (rhs == dim_rule.size() - 1) {
                assert (target != result.end());
                assert (next_la_weight != rule_weight_tensor.end());
                const Val val = (lhs_outside_weights[lhs]) * ((*next_la_weight) * (rhs > 0 ? factors[rhs - 1] : Val::one()));
                *target = val;
                ++next_la_weight;
                ++target;
                if (rhs == 0)
                    break;
                rhs--;
            } else if (selection[rhs] == rhs_inside_weights[rhs].size()) {
                selection[rhs] = 0;
                if (rhs == 0)
                    break;
                rhs--;
            } else {
                if (rhs > 0)
                    factors[rhs] = (factors[rhs-1]) * (rhs_inside_weights[rhs][selection[rhs]]);
                else
                    factors[rhs] = rhs_inside_weights[rhs][selection[rhs]];
                ++selection[rhs];
                // if a factor is zero already, then we can abort early
                // but we have to increase next_la_weight/ target accordingly
                if (factors[rhs] == Val::zero()) {
                    const unsigned skip = subdim_from(rhs_inside_weights, rhs+1);
                    next_la_weight += skip;
                    target += skip;
                } else {
                    ++rhs;
                }
            }
        }
    }
    assert (target == result.end());
    assert (next_la_weight == rule_weight_tensor.end());
    return result;
}


template <typename Val>
std::vector<Val> compute_outside_weights(const std::vector<Val> & rule_weight_tensor
        , const std::vector<Val> & lhn_outside_weight_vector
        , const std::vector<std::vector<Val>> & inside_weight_vectors
        , const std::vector<unsigned> & dim_rule
        , const unsigned target_pos) {

    std::vector<Val> result = std::vector<Val>(dim_rule[target_pos + 1], Val::zero());

    typename std::vector<Val>::const_iterator next_la_weight = rule_weight_tensor.begin();
    unsigned steps = 0;
    for (unsigned lhs = 0; lhs < dim_rule[0]; ++lhs) {
        const unsigned sd = subdim(dim_rule);
        assert(next_la_weight == rule_weight_tensor.begin() + lhs * sd);

        // we can skip rules with zero probability
        if (lhn_outside_weight_vector[lhs] == Val::zero()) {
            next_la_weight += sd;
            continue;
        }

        unsigned rhs = 0;
        std::vector<unsigned> selection = std::vector<unsigned>(dim_rule.size() - 1, 0);
        std::vector<Val> factors = std::vector<Val>(dim_rule.size() -1, Val::one());
        while (true) {
            if (rhs == dim_rule.size() - 1) {
                assert (next_la_weight != rule_weight_tensor.end());
                Val val = (*next_la_weight) * ((rhs > 0) ? factors[rhs - 1] : Val::one());
                val *= lhn_outside_weight_vector[lhs];
                result[selection[target_pos] - 1] = result[selection[target_pos] - 1] + val;
                ++next_la_weight;
                ++steps;
                if (rhs == 0)
                    break;
                rhs--;
            } else if (selection[rhs] == inside_weight_vectors[rhs].size()) {
                selection[rhs] = 0;
                if (rhs == 0)
                    break;
                rhs--;
            } else {
                if (rhs > 0)
                    factors[rhs] = (factors[rhs-1]) * inside_weight_vectors[rhs][selection[rhs]];
                else
                    factors[rhs] = inside_weight_vectors[rhs][selection[rhs]];
                ++selection[rhs];
                // if a factor is zero already, then we can abort early
                // but we have to increase next_la_weight/ target accordingly
                if (factors[rhs] == Val::zero()) {
                    const unsigned skip = subdim_from(inside_weight_vectors, rhs+1);
                    next_la_weight += skip;
                } else {
                    ++rhs;
                }
            }
        }
    }
    assert (next_la_weight == rule_weight_tensor.end());
    return result;
}



#endif //STERMPARSER_SPLITMERGEUTIL_H
