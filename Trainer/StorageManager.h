//
// Created by jamb on 28.02.17.
//

#ifndef STERMPARSER_STORAGEMANAGER_H
#define STERMPARSER_STORAGEMANAGER_H

#include <iostream>
#include <cassert>
#include "GrammarInfo.h"
#include "../Names.h"
#include "TraceManager.h"
#include "TrainingCommon.h"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>


namespace Trainer {

    class StorageManager {
    private:
        bool selfMalloc;
        double *start = nullptr;
        double *next = nullptr;
        double *maxMem = nullptr;
        size_t theSize = 0; // 625000; // 5MB


    public:
        StorageManager(bool selfmal = false) : selfMalloc(selfmal) {}


        bool reserve_memory(unsigned size) {
            if (!selfMalloc)
                return true;

            std::cerr << "reserving " << size << std::endl;
            if (start == next) {
                if (start != nullptr and maxMem - start < size) {
                    free(start);
                }
                if (start == nullptr or maxMem - start < size) {
                    unsigned allocate = theSize > size ? theSize : size;
                    std::cerr << "allocating " << allocate << std::endl;
                    start = (double *) malloc(sizeof(double) * allocate);
                    maxMem = start + allocate;
                    next = start;
                }
                return true;
            } else
                return false;
        }

        double *get_region(const size_t size) {
            if (not selfMalloc)
                return (double *) malloc(sizeof(double) * size);
            else {
                if (start == nullptr) {
                    start = (double *) malloc(sizeof(double) * theSize);
                    maxMem = start + theSize;
                    next = start;
                }
                if (maxMem < size + next) {
                    std::cerr << "Maximum size of double storage exceeded" << std::endl;
                    std::cerr << "Required  " << size << std::endl;
                    std::cerr << "Available " << (maxMem - next) / sizeof(double) << std::endl;
                    abort();
                }
                double *return_ = next;
                next = return_ + size;
                return return_;
            }
        }

        bool free_region(double *const ptr, const size_t size) {
            if (not selfMalloc) {
                free(ptr);
                return true;
            } else {
                if (ptr + size == next) {
                    assert(start <= ptr);
                    next = ptr;
                    return true;
                }
                return false;
            }
        }

        template<typename T>
        typename std::enable_if_t<std::is_same<T, Eigen::TensorMap<Eigen::Tensor<double, 1>>>::value, T>
        create_weight_vector(size_t size) {
            double *ptr = get_region(size);
            return Eigen::TensorMap<Eigen::Tensor<double, 1>>(ptr, size);
        }

        template<typename T>
        typename std::enable_if_t<std::is_same<T, Eigen::Tensor<double, 1>>::value, T>
        create_weight_vector(size_t size) {
            return Eigen::Tensor<double, 1>(size);
        };

        inline void free_weight_vector(Eigen::Tensor<double, 1> &) {
            // nothing needs to be done
        }

        inline void free_weight_vector(Eigen::TensorMap<Eigen::Tensor<double, 1>> &weightVector) {
            if (not selfMalloc) {
                free_region(weightVector.data(), (size_t) weightVector.size());
            } else {
                // todo: not implemented
                abort();
            }
        }

        template<unsigned long rank>
        inline RuleTensor<double> create_uninitialized_tensor_ranked(
                const size_t ruleId
                , const GrammarInfo2 &grammarInfo
                , const std::vector<size_t> &nontSplits
        ) {
            Eigen::array<Eigen::DenseIndex, rank> ruleDimensions;
            for (unsigned dim = 0; dim < rank; ++dim) {
                ruleDimensions[dim] = nontSplits[grammarInfo.rule_to_nonterminals[ruleId][dim]];
            }
            return create_uninitialized_tensor_ranked_typed<Trainer::RuleTensorRaw<double, rank>>(ruleDimensions);
        }

        template<typename T, unsigned long rank>
        inline
        typename std::enable_if_t<std::is_same<T, Eigen::Tensor<double, rank>>::value, T>
        create_uninitialized_tensor_ranked_typed(const Eigen::array<long, rank> &ruleDimensions) {
            return Eigen::Tensor<double, rank>(ruleDimensions);
        };

        template<typename T, unsigned long rank>
        inline
        typename std::enable_if_t<std::is_same<T, Eigen::TensorMap<Eigen::Tensor<double, rank>>>::value, T>
        create_uninitialized_tensor_ranked_typed(const Eigen::array<long, rank> &ruleDimensions) {
            size_t memory = std::accumulate(
                    ruleDimensions.cbegin()
                    , ruleDimensions.cend()
                    , (size_t) 1
                    , std::multiplies<size_t>());
            double *storage = get_region(memory);
            return Eigen::TensorMap<Eigen::Tensor<double, rank>>(storage, ruleDimensions);
        };


        inline Trainer::RuleTensor<double> create_uninitialized_tensor(
                const size_t ruleId
                , const GrammarInfo2 &grammarInfo
                , const std::vector<size_t> &nontSplits
        ) {
            switch (grammarInfo.rule_to_nonterminals[ruleId].size()) {
                case 1:
                    return create_uninitialized_tensor_ranked<1>(ruleId, grammarInfo, nontSplits);
                case 2:
                    return create_uninitialized_tensor_ranked<2>(ruleId, grammarInfo, nontSplits);
                case 3:
                    return create_uninitialized_tensor_ranked<3>(ruleId, grammarInfo, nontSplits);
                case 4:
                    return create_uninitialized_tensor_ranked<4>(ruleId, grammarInfo, nontSplits);
                default:
                    abort();
            }
        }


        template<typename T, unsigned long rank>
        inline
        typename std::enable_if_t<std::is_same<T, Eigen::Tensor<double, rank>>::value, T>
        copy_tensor_ranked(const RuleTensor<double> & ruleTensor) {
            const auto & ruleTensorRaw = boost::get<RuleTensorRaw<double, rank>>(ruleTensor);
            Eigen::Tensor<double, rank> copyTensor(ruleTensorRaw.dimensions());
            copyTensor = ruleTensorRaw;
            return copyTensor;
        };

        template<typename T, unsigned long rank>
        inline
        typename std::enable_if_t<std::is_same<T, Eigen::TensorMap<Eigen::Tensor<double, rank>>>::value, T>
        copy_tensor_ranked(const RuleTensor<double> & ruleTensor) {
            const auto & ruleTensorRaw = boost::get<RuleTensorRaw<double, rank>>(ruleTensor);
            double *storage = get_region((size_t) ruleTensorRaw.size());
            Eigen::TensorMap<Eigen::Tensor<double, rank>> copyTensor(storage, ruleTensorRaw.dimensions());
            copyTensor = ruleTensorRaw;
            return copyTensor;
        };

        inline Trainer::RuleTensor<double> copy_tensor(
               const RuleTensor<double> & ruleTensor
        ) {
            switch (ruleTensor.which() + 1) {
                case 1:
                    return copy_tensor_ranked<RuleTensorRaw<double, 1>, 1>(ruleTensor);
                case 2:
                    return copy_tensor_ranked<RuleTensorRaw<double, 2>, 2>(ruleTensor);
                case 3:
                    return copy_tensor_ranked<RuleTensorRaw<double, 3>, 3>(ruleTensor);
                case 4:
                    return copy_tensor_ranked<RuleTensorRaw<double, 4>, 4>(ruleTensor);
                default:
                    abort();
            }
        }

        /*
        template<template<typename T1, typename T2> typename MapType, typename Nonterminal, typename TraceID>
        std::pair<double *, unsigned>
        allocate_io_weight_maps(
                const std::vector<unsigned> &nont_dimensions
                , std::vector<MapType<Element<Node<Nonterminal>>, WeightVector>> &traces_inside_weights
                , std::vector<MapType<Element<Node<Nonterminal>>, WeightVector>> &traces_outside_weights
                , TraceManagerPtr<Nonterminal, TraceID> traceManager
        ) {
            double *start(nullptr);
            unsigned allocated(0);

            traces_inside_weights.clear();
            traces_inside_weights.reserve(traceManager->size());
            traces_outside_weights.clear();
            traces_outside_weights.reserve(traceManager->size());

            for (const auto &trace : *traceManager) {
                MapType<Element<Node<Nonterminal>>, WeightVector> inside_weights, outside_weights;

                for (const auto &node : *(trace->get_hypergraph())) {
                    const unsigned item_dimension = nont_dimensions[node->get_label_id()];
                    double *const inside_weight_ptr = get_region(item_dimension);
                    double *const outside_weight_ptr = get_region(item_dimension);
                    if (start == nullptr)
                        start = inside_weight_ptr;
                    allocated += item_dimension * 2;

                    Eigen::TensorMap<Eigen::Tensor<double, 1>> inside_weight(inside_weight_ptr, item_dimension);
                    Eigen::TensorMap<Eigen::Tensor<double, 1>> outside_weight(outside_weight_ptr, item_dimension);
                    inside_weights.emplace(node, std::move(inside_weight));
                    outside_weights.emplace(node, std::move(outside_weight));
                }
                traces_inside_weights.push_back(std::move(inside_weights));
                traces_outside_weights.push_back(std::move(outside_weights));
            }
            return std::make_pair(start, allocated);
        }
        */

        template<typename Key>
        void free_weight_maps(std::vector<MAPTYPE<Key, Eigen::Tensor<double, 1>>> &maps) {
            maps.clear();
        };

        template<typename Key>
        void free_weight_maps(std::vector<MAPTYPE<Key, Eigen::TensorMap<Eigen::Tensor<double, 1>>>> &maps) {
            for (auto map : maps) {
                for (auto entry : map) {
                    this->free_weight_vector(entry.second);
                }
            }
            maps.clear();
        };

        /*template<template<typename T1, typename T2> typename MapType, typename Nonterminal, typename TraceID>
        void free_io_weight_maps(
                double *const start
                , unsigned allocated
                , std::vector<MapType<Node<Nonterminal>, WeightVector>> &traces_inside_weights
                , std::vector<MapType<Node<Nonterminal>, WeightVector>> &traces_outside_weights
                , TraceManager2<Nonterminal, TraceID> &traceManager
        ) {
            if (selfMalloc) {
                if (not free_region(start, allocated))
                    abort();
                traces_inside_weights.clear();
                traces_outside_weights.clear();
            } else {
                for (ConstManagerIterator<Trace<Nonterminal, TraceID>> traceIterator = traceManager.cbegin();
                     traceIterator < traceManager.cend(); ++traceIterator) {
                    MapType<Element<Node<Nonterminal>>, WeightVector> &inside_weights = traces_inside_weights[
                            traceIterator - traceManager.cbegin()];
                    MapType<Element<Node<Nonterminal>>, WeightVector> &outside_weights = traces_outside_weights[
                            traceIterator - traceManager.cbegin()];
                    for (const auto &pair : inside_weights) {
                        free(pair.second.data());
                    }
                    for (const auto &pair : outside_weights) {
                        free(pair.second.data());
                    }
                }
            }
            traces_inside_weights.clear();
            traces_outside_weights.clear();
        }*/

    };

}





#endif //STERMPARSER_STORAGEMANAGER_H
