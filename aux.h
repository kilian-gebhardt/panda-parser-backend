//
// Created by kilian on 03/02/17.
//

template<typename Witness>
inline void outside_weight_step2(
        const std::vector<RuleTensor<double>> &rules,
        const MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector> & outside_weights,
        WeightVector & outside_weight,
        const Witness & witness) const {
    const auto &rule = *(std::get<0>(witness));
    const auto &parent = *(std::get<2>(witness));
    constexpr unsigned rule_rank {2};

    const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[rule.id]);
    const Eigen::TensorMap<Eigen::Tensor<double, 1>> & parent_weight = outside_weights.at(parent);

    auto outside_weight_summand = rule_weight.contract(parent_weight, Eigen::array<Eigen::IndexPair<long>, 1>({Eigen::IndexPair<long>(0, 0)}));

    outside_weight += outside_weight_summand;
}

template<typename Witness>
inline void outside_weight_step3(
        const std::vector<RuleTensor<double>> &rules,
        const MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> & inside_weights,
        const MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector> & outside_weights,
        WeightVector & outside_weight,
        const Witness & witness) const {
    const auto &rule = *(std::get<0>(witness));
    const auto &siblings = *(std::get<1>(witness));
    const auto &parent = *(std::get<2>(witness));
    const unsigned position = std::get<3>(witness);
    constexpr unsigned rule_rank {3};


    const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[rule.id]);
    const Eigen::TensorMap<Eigen::Tensor<double, 1>> & parent_weight = outside_weights.at(parent);
    const Eigen::TensorMap<Eigen::Tensor<double, 1>> & rhs_weight = inside_weights.at(*siblings[position == 0 ? 1 : 0]);

    auto c1 = rule_weight.contract(rhs_weight, Eigen::array<Eigen::IndexPair<long>, 1>({Eigen::IndexPair<long>(position == 0 ? 2 : 1, 0)}));
    auto c2 = c1.contract(parent_weight, Eigen::array<Eigen::IndexPair<long>, 1>({Eigen::IndexPair<long>(0, 0)}));

    outside_weight += c2;
}