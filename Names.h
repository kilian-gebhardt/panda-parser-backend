//
// Created by kilian on 03/03/17.
//

#ifndef STERMPARSER_NAMES_H
#define STERMPARSER_NAMES_H
#include "Manage/Manager.h"
#include "Manage/Hypergraph.h"

/*
using WeightVector = Eigen::TensorMap<Eigen::Tensor<double, 1>>;
template <typename Scalar>
using RuleTensor = typename boost::variant<
        Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>
, Eigen::TensorMap<Eigen::Tensor<Scalar, 2>>
, Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>
, Eigen::TensorMap<Eigen::Tensor<Scalar, 4>>
, Eigen::TensorMap<Eigen::Tensor<Scalar, 5>>
, Eigen::TensorMap<Eigen::Tensor<Scalar, 6>>
>;
*/

// use partly specialized Hypergraph objects


//using NodeOriginalID = unsigned long;
using EdgeLabelT = size_t;


template <typename NodeLabelT> using Node = Manage::Node<NodeLabelT>;
template <typename NodeLabelT> using HyperEdge = Manage::HyperEdge<Node<NodeLabelT>, EdgeLabelT>;
template <typename InfoT> using Manager = Manage::Manager<InfoT>;
template <typename InfoT> using ManagerPtr = Manage::ManagerPtr<InfoT>;
template <typename InfoT> using ConstManagerIterator = Manage::ManagerIterator<InfoT, true>;
template <typename InfoT> using Element = Manage::Element<InfoT>;


template<typename Nonterminal> using Hypergraph = Manage::Hypergraph<Nonterminal, EdgeLabelT>;
template<typename Nonterminal> using HypergraphPtr = std::shared_ptr<Hypergraph<Nonterminal>>;


#endif //STERMPARSER_NAMES_H
