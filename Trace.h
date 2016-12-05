//
// Created by kilian on 05/12/16.
//

#ifndef STERMPARSER_TRACE_H
#define STERMPARSER_TRACE_H

#include "SDCP_Parser.h"

template <typename Nonterminal, typename Terminal, typename Position>
class TraceManager {
private:
    std::vector<
            std::map<
                      ParseItem<Nonterminal, Position>
                    , std::vector<
                            std::pair<
                                      std::shared_ptr<Rule<Nonterminal, Terminal>>
                                    , std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>
                                    >
                            >
                    >
            >
    > traces;
    std::vector<
            std::vector<
                ParseItem<Nonterminal, Position>
            >
            > topological_orders;
    std::vector<ParseItem<Nonterminal, Position>> goals;


    // empty default values
    const std::map<
    ParseItem<Nonterminal, Position>
    , std::vector<
            std::pair<
                    std::shared_ptr<Rule<Nonterminal, Terminal>>
                    , std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>
                    >
            >
    >> empty_trace;
    const std::vector<ParseItem<Nonterminal, Position>> empty_order;

    // auxiliary structures
    std::set<ParseItem<Nonterminal, Position>> inserted_items;

public:
    void add_trace_entry(
            const std::map<
                ParseItem<Nonterminal, Position>
                , std::vector<
                        std::pair<
                                std::shared_ptr<Rule<Nonterminal, Terminal>>
                                , std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>
                                >
                        >
                >
            > trace, ParseItem<Nonterminal, Position> goal, unsigned i) {

        if (traces.size() <= i) {
            traces.resize(i + 1);
            topological_orders.resize(i + 1);
            goals.resize(i+1);
        }
        traces[i] = trace;
        goals[i] = goal;

        inserted_items.clear();

        // compute topological order of trace items
        std::vector<ParseItem<Nonterminal, Position>> topological_order;
        bool changed = true;
        while (changed) {
            changed = false;

            // add item, if all its decendants were added
            for (const auto &entry : trace) {
                if (inserted_items.count(entry.first))
                    continue;
                bool violation = false;
                for (const auto &witness : entry.second) {
                    for (const auto item : witness.second) {
                        if (!inserted_items.count(*item)) {
                            violation = true;
                            break;
                        }
                    }
                    if (violation)
                        break;
                }
                if (!violation) {
                    changed = true;
                    inserted_items.insert(entry.first);
                    topological_order.push_back(entry.first);
                }
            }
        }

        inserted_items.clear();

        if (topological_order.size() == trace.size()) {
            topological_orders[i] = topological_order;
        }

    }

    const std::map<
            ParseItem<Nonterminal, Position>
            , std::vector<
                    std::pair<
                            std::shared_ptr<Rule<Nonterminal, Terminal>>
                            , std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>
                            >
                    >
            >
    > & query_trace_entry(unsigned i){
        if (traces.size() <= i) {
            return empty_trace;
        } else {
            return traces[i];
        }
    };

    template<typename Val>
    std::pair<std::map<ParseItem<Nonterminal, Position>, Val>,
              std::map<ParseItem<Nonterminal, Position>, Val>>
            io_weights(const std::vector<Val> & rules, const Val leaf, const Val root, const Val zero, const unsigned i) {
        // TODO implement for general case (== no topological order) approximation of inside weights
        assert (topological_orders.size() > i && topological_orders[i].size() > 0);

        const auto & topological_order = topological_orders[i];

        std::map<ParseItem<Nonterminal, Position>, Val> inside_weights;

        for (const auto & item : topological_orders[i]) {
            inside_weights[item] = zero;
            for (const auto & witness : traces[i][item]) {
                Val val = rules[witness.first->id];
                for (const auto & dep_item : witness.second) {
                    val *= inside_weights[*dep_item];
                }
                inside_weights[item] += val;
            }
        }

        // TODO implement for general case (== no topological order) solution by gauss jordan
        std::map<ParseItem<Nonterminal, Position>, Val> outside_weights;
        for (int j = topological_order.size() - 1; j >= 0; --j) {
            const ParseItem<Nonterminal, Position> & item = topological_order[j];
            Val val = zero;
            if (item == goals[i])
                val += root;
            for (int k = topological_order.size() - 1; k > j; --k){
                const ParseItem<Nonterminal, Position> & parent = topological_order[k];
                for (const auto & witness : traces[i][parent]) {
                    Val val_witness = outside_weights[parent] * rules[witness.first->id];
                    bool item_found = false;
                    for (const auto & rhs_item : witness.second) {
                        if (*rhs_item == item)
                            item_found = true;
                        else
                            val_witness *= inside_weights[*rhs_item];
                    }
                    if (item_found)
                        val += val_witness;
                }
            }
            outside_weights[item] = val;
        }

        return std::make_pair(inside_weights, outside_weights);
    }

    const std::vector<ParseItem<Nonterminal, Position>> & get_order(unsigned i) {
        if (topological_orders.size() <= i)
            return empty_order;
        else
            return topological_orders[i];
    };

    std::vector<double> do_em_training( const std::vector<double> & initial_weights
                       , const std::vector<std::vector<unsigned>> & normalization_groups
                       , const unsigned n_epochs){
        std::vector<double> rule_weights = initial_weights;
        std::vector<double> rule_counts;

        unsigned epoch = 0;

        std::cerr <<"Epoch " << epoch << "/" << n_epochs << ": ";
        for (auto i = 0; i < rule_weights.size(); ++i) {
            std::cerr << rule_weights[i] << " ";
        }
        std::cerr << std::endl;

        while (epoch < n_epochs) {
            // expectation
            rule_counts = std::vector<double>(rule_weights.size(), 0.0);
            for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
                auto trace = traces[trace_id];
                if (trace.size() == 0)
                    continue;
                const auto tr_io_weight = io_weights(rule_weights, 1.0, 1.0, 0.0, trace_id);
                for (const auto & item : get_order(trace_id)) {
                    std::cerr << "T: " << item << " " << tr_io_weight.first.at(item) << " " << tr_io_weight.second.at(item) << std::endl;
                }
                std::cerr << std::endl;
                const double root_weight = tr_io_weight.first.at(goals[trace_id]);
                for (auto & pair : trace) {
                    const double lhn_outside_weight = tr_io_weight.second.at(pair.first);
                    for (const auto & witness : pair.second) {
                        const int rule_id = witness.first->id;
                        double val = lhn_outside_weight * rule_weights[rule_id] / root_weight;
                        for (const auto & rhs_item : witness.second) {
                            val *= tr_io_weight.first.at(*rhs_item);
                        }
                        rule_counts[rule_id] += val;
                    }
                }
            }

            // maximization
            for (auto group : normalization_groups) {
                double group_count = 0.0;
                for (auto member : group) {
                    group_count += rule_counts[member];
                }
                if (group_count != 0) {
                    for (auto member : group) {
                        rule_weights[member] = rule_counts[member] / group_count;
                    }
                }
            }
            epoch++;
            std::cerr <<"Epoch " << epoch << "/" << n_epochs << ": ";
            for (auto i = 0; i < rule_weights.size(); ++i) {
                std::cerr << rule_weights[i] << " ";
            }
            std::cerr << std::endl;
        }
        return rule_weights;
    }

};

#endif //STERMPARSER_TRACE_H
