//
// Created by kilian on 03/02/17.
//

template<typename Witness>
inline void outside_weight_step2(const std::vector<RuleTensor<double>> &rules,
                          MAPTYPE<ParseItem<Nonterminal, Position>, double *> & outside_weights,
                          Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &
                          outside_weight, Witness witness) const {
    const auto &rule = *(std::get<0>(witness));
    const auto &parent = *(std::get<2>(witness));
    constexpr unsigned rule_rank {2};

    const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[rule.id]);
    const Eigen::array<long, rule_rank> & rule_dim = rule_weight.dimensions();
    const auto parent_weight = Eigen::TensorMap<Eigen::Tensor<double, 1>>(outside_weights.at(parent), rule_dim[0]);

    /*

    auto rule_val = rule_weight * parent_weight.reshape(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1}).broadcast(Eigen::array<long, 2>{1, rule_weight.dimension(1)});
    auto outside_weight_summand = rule_val.sum(Eigen::array<long, 1>{0});
    */

    auto outside_weight_summand = rule_weight.contract(parent_weight, Eigen::array<Eigen::IndexPair<long>, 1>({Eigen::IndexPair<long>(0, 0)}));

    outside_weight += outside_weight_summand;
}

template<typename Witness>
inline void outside_weight_step3(const std::vector<RuleTensor<double>> &rules,
                                 const MAPTYPE<ParseItem < Nonterminal, Position>, double *> & inside_weights,
                          MAPTYPE<ParseItem<Nonterminal, Position>, double *> & outside_weights,
                          Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &
                          outside_weight, Witness witness) const {
    const auto &rule = *(std::get<0>(witness));
    const auto &siblings = *(std::get<1>(witness));
    const auto &parent = *(std::get<2>(witness));
    const unsigned position = std::get<3>(witness);
    constexpr unsigned rule_rank {3};


    const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[rule.id]);
    const Eigen::array<long, rule_rank> & rule_dim = rule_weight.dimensions();
    const auto parent_weight = Eigen::TensorMap<Eigen::Tensor<double, 1>>(outside_weights.at(parent), rule_dim[0]);

    double *const rhs_ptr = inside_weights.at(*siblings[position == 0 ? 1 : 0]);

    Eigen::TensorMap<Eigen::Tensor<double, 1>>rhs_weight(rhs_ptr, rule_dim[position == 0 ? 2 : 1]);

//    auto rule_val = rule_weight
//                    * parent_weight.reshape(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1, 1}).broadcast(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1), rule_weight.dimension(2)})
//                    * (position == 1 ? rhs_weight.reshape(Eigen::array<long, rule_rank>{1, rule_weight.dimension(1), 1}).broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), 1, rule_weight.dimension(2)}) :
//                       rhs_weight.reshape(Eigen::array<long, rule_rank>{1, 1, rule_weight.dimension(2)}).broadcast(Eigen::array<long, rule_rank>{rule_weight.dimension(0), rule_weight.dimension(1), 1}));
//    auto outside_weight_summand = rule_val.sum(Eigen::array<long, 2>{0, position == 0 ? 2 : 1});

    auto c1 = rule_weight.contract(rhs_weight, Eigen::array<Eigen::IndexPair<long>, 1>({Eigen::IndexPair<long>(position == 0 ? 2 : 1, 0)}));
    auto c2 = c1.contract(parent_weight, Eigen::array<Eigen::IndexPair<long>, 1>({Eigen::IndexPair<long>(0, 0)}));

    outside_weight += c2;
}