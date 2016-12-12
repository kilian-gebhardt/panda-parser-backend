//
// Created by kilian on 05/12/16.
//

#ifndef STERMPARSER_TRACE_H
#define STERMPARSER_TRACE_H

#include "SDCP_Parser.h"
#include <limits>
#include <math.h>
#include <malloc.h>
#include <random>
#include "SplitMergeUtil.h"


class Chance {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
public:
    double get_chance() {
        return distribution(generator);
    }
};

//class RuleLA {
//    const int rule_id;
//    unsigned lhs;
//    unsigned * const rhs;
//public:
//    bool valid = true;
//    const unsigned rhs_size;
//    double weight;
//    RuleLA(int rule_id, unsigned lhs, const std::vector<unsigned> & rhs, double weight) :
//              rule_id(rule_id)
//            , lhs(lhs)
//            , weight(weight)
//            , rhs_size(rhs.size())
//            , rhs((unsigned * const) malloc(sizeof(unsigned) * rhs_size)) {
//        for (auto i = 0; i < rhs.size(); ++i)
//            this->rhs[i] = rhs[i];
//    }
//
//    const unsigned & get_lhs_la() {
//        return lhs;
//    }
//
//    const int & get_rule_id() {
//        return rule_id;
//    }
//
//    const unsigned & get_rhs_la(unsigned i) const {
//        return rhs[i];
//    }
//
//    void expand() {
//        lhs *= 2;
//        for (unsigned i = 0; i < rhs_size; ++i) {
//            rhs[i] *= 2;
//        }
//    }
//
//    void shift_la(const unsigned from, const unsigned to) {
//        if (lhs == from)
//            lhs = to;
//        for (unsigned i = 0; i < rhs_size; ++i) {
//            unsigned & la = rhs[i];
//            if (la == from)
//                la = to;
//        }
//    }
//
//    const std::vector<RuleLA> splits(Chance & chance) const {
//        std::vector<RuleLA> transport;
//        std::vector<unsigned> selection;
//
//        for (unsigned lhs_ : {lhs, lhs + 1})
//            selections(lhs_, transport, selection);
//
//        // assuming log likelihood
//        double base_weight = weight - log(pow(2, rhs_size + 1));
//        double chance_sum = 0;
//        for (auto & la : transport) {
//            la.weight = chance.get_chance();
//            chance_sum += la.weight;
//        }
//        for (auto & la : transport) {
//            la.weight = la.weight / chance_sum * base_weight;
//        }
//
//        return transport;
//    }
//
//    void selections(const unsigned lhs, std::vector<RuleLA> & transport, std::vector<unsigned> & selection) const {
//        if (selection.size() == rhs_size) {
//            transport.push_back(RuleLA(rule_id, lhs, selection, 0.0));
//        } else {
//            const unsigned rhs_la = rhs[selection.size() - 1];
//            for (unsigned var_ : {rhs_la, rhs_la + 1}) {
//                selection.push_back(var_);
//                selections(lhs, transport, selection);
//                selection.pop_back();
//            }
//        }
//    }
//
//};


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

    const bool debug = false;

public:
    TraceManager(bool debug=false) : debug(debug) {}

    void add_trace_from_parser(const SDCPParser<Nonterminal, Terminal, Position> & parser, unsigned i){
        add_trace_entry(parser.get_trace(), *parser.goal, i);
    }

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

    const double plus_infinity = std::numeric_limits<double>::max();
    const double minus_infinity = std::numeric_limits<double>::infinity();

    double log_sum(double x, double y) {
        if (x == minus_infinity)
            return y;
        else if (y == minus_infinity)
            return x;
        return log(exp(x) + exp(y));
    }

    static double log_prod(double x, double y){
        return x + y;
    }

    static double log_div(double x, double y){
        return x - y;
    }


    template<typename Val, typename Accum, typename Accum2>
    std::pair<std::map<ParseItem<Nonterminal, Position>, Val>,
              std::map<ParseItem<Nonterminal, Position>, Val>>
            io_weights(const std::vector<Val> & rules, const Val leaf, const Val root, const Val zero, Accum sum, Accum2 prod, const unsigned i) const {
        // TODO implement for general case (== no topological order) approximation of inside weights
        assert (topological_orders.size() > i && topological_orders[i].size() > 0);

        const auto & topological_order = topological_orders[i];

        std::map<ParseItem<Nonterminal, Position>, Val> inside_weights;

        for (const auto & item : topological_order) {
            inside_weights[item] = zero;
            for (const auto & witness : traces[i].at(item)) {
                Val val = rules[witness.first->id];
                for (const auto & dep_item : witness.second) {
                    val = prod(val, inside_weights.at(*dep_item));
                }
                if (debug && (val == zero || val == -zero)) {
                    std::cerr << "rule weight: " << rules[witness.first->id] <<std::endl;
                    for (const auto & dep_item : witness.second) {
                        std::cerr << *dep_item << " " << inside_weights.at(*dep_item) << std::endl;
                    }
                }
                inside_weights[item] = sum(inside_weights.at(item), val);
            }
        }

        // TODO implement for general case (== no topological order) solution by gauss jordan
        std::map<ParseItem<Nonterminal, Position>, Val> outside_weights;
        for (int j = topological_order.size() - 1; j >= 0; --j) {
            const ParseItem<Nonterminal, Position> & item = topological_order[j];
            Val val = zero;
            if (item == goals[i])
                val = sum(val, root);
            for (int k = topological_order.size() - 1; k > j; --k){
                const ParseItem<Nonterminal, Position> & parent = topological_order[k];
                for (const auto & witness : traces[i].at(parent)) {
                    Val val_witness = prod(outside_weights.at(parent), rules[witness.first->id]);
                    bool item_found = false;
                    for (const auto & rhs_item : witness.second) {
                        if (*rhs_item == item)
                            item_found = true;
                        else
                            val_witness = prod(val_witness, inside_weights.at(*rhs_item));
                    }
                    if (item_found)
                        val = sum(val, val_witness);
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

        bool log_semiring = true;
//        auto sum =  [] (double x, double y) -> double {return x + y;};
//        auto prod = [] (double x, double y) -> double {return x * y;};
//        auto division = [] (double x, double y) -> double {return x / y;};
//        double root = 1.0;
//        double leaf = 1.0;
//        double zero = 0.0;

//        if (log_semiring) {
            auto sum = [] (double x, double y) -> double {
                const double minus_infinity = std::numeric_limits<double>::infinity();
                if (x == minus_infinity)
                    return y;
                else if (y == minus_infinity)
                    return x;
                return log(exp(x) + exp(y));};
            auto prod = [] (double x, double y) -> double {return x + y;};;
            auto division = [] (double x, double y) -> double {return x - y;};;
            double root = 0.0;
            double leaf = 0.0;
            double zero = minus_infinity;

//        }

        unsigned epoch = 0;


        std::cerr <<"Epoch " << epoch << "/" << n_epochs << ": ";
/*
        for (auto i = rule_weights.begin(); i != rule_weights.end(); ++i) {
            std::cerr << *i << " ";
        }
        std::cerr << std::endl;
*/

        // conversion to log semiring:
        if (log_semiring) {
            for (auto i = rule_weights.begin(); i != rule_weights.end(); ++i) {
                *i = log(*i);
                std::cerr << *i << " ";
            }
        }
        std::cerr << std::endl;

        while (epoch < n_epochs) {
            // expectation
            rule_counts = std::vector<double>(rule_weights.size(), zero);
            for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
                auto trace = traces[trace_id];
                if (trace.size() == 0)
                    continue;

                const auto tr_io_weight = io_weights(rule_weights, leaf, root, zero, sum, prod, trace_id);
                if (debug) {
                    for (const auto &item : get_order(trace_id)) {
                        std::cerr << "T: " << item << " " << tr_io_weight.first.at(item) << " "
                                  << tr_io_weight.second.at(item) << std::endl;
                    }
                    std::cerr << std::endl;
                }
                const double root_weight = tr_io_weight.first.at(goals[trace_id]);
                for (auto & pair : trace) {
                    const double lhn_outside_weight = tr_io_weight.second.at(pair.first);
                    for (const auto & witness : pair.second) {
                        const int rule_id = witness.first->id;
                        double val = division(prod(lhn_outside_weight, rule_weights[rule_id]), root_weight);
                        for (const auto & rhs_item : witness.second) {
                            val = prod(val, tr_io_weight.first.at(*rhs_item));
                        }
                        rule_counts[rule_id] = sum(rule_counts[rule_id], val);
                    }
                }
            }

            // maximization
            for (auto group : normalization_groups) {
                double group_count = zero;
                for (auto member : group) {
                    group_count = sum(group_count, rule_counts[member]);
                }
                if (group_count != zero) {
                    for (auto member : group) {
                        rule_weights[member] = division(rule_counts[member], group_count);
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

        // conversion from log semiring:
        if (log_semiring) {
            for (auto i = rule_weights.begin(); i != rule_weights.end(); ++i) {
                *i = exp(*i);
            }
        }

        return rule_weights;
    }

//    void split_merge(std::vector<std::vector<RuleLA>> rule_las, std::vector<unsigned> & nont_las) {
//        Chance chance;
//
//        // splitting
//        std::vector<std::vector<RuleLA>> rule_las_split;
//
//        for (unsigned nont_idx = 0; nont_idx < rule_las.size(); ++nont_idx) {
//            std::vector<RuleLA> splits;
//            for (const auto & rule_la : rule_las[nont_idx]) {
//                for (const auto & rule_la_ : rule_la.splits(chance)) {
//                    splits.push_back(rule_la_);
//                }
//            }
//            rule_las_split.push_back(splits);
//        }
//
//        // em-training
//
//        // merging
//
//
//        // em-training
//
//    }

    void split_merge(std::vector<unsigned> nont_dimensions, std::vector<double*> rule_weights_la, const std::vector<std::vector<unsigned>> & rule_to_nonterminals) {
        // splitting
        std::vector<double *> rule_weights_splitted;
        for (auto i = 0; i < rule_weights_la.size(); ++i) {
            const double * rule_weight = rule_weights_la[i];
            std::vector<unsigned> dimensions;
            for (auto nont : rule_to_nonterminals[i]) {
                dimensions.push_back(nont_dimensions[nont]);
            }
            rule_weights_splitted.push_back(split_rule(rule_weight, dimensions));
        }

        // em training

        // TODO determine merges

        // nonterminal -> new_la -> contributing old_las
        std::vector<std::vector<std::vector<unsigned>>> merge_selection;

        // merging
        // TODO continue here!
        // for (auto i = 0; i < rule)


        // em training
    }

};


//void merge_lhs(std::vector<RuleLA> & rule_las, const unsigned lhs);
//void merge_rhs(std::vector<RuleLA> & rule_las, const unsigned var_pos, const unsigned la);
//
//void apply_merges_to_rule(std::vector<RuleLA> & rule_las, const std::vector<std::pair<unsigned, unsigned>> & merges) {
//
//    // TODO perhaps reverse order neccessary?!
//    for (const auto tmp : merges) {
//        if (tmp.first == 0) {
//            merge_lhs(rule_las, tmp.second);
//        }
//        else {
//            merge_rhs(rule_las, tmp.first - 1, tmp.second);
//        }
//    }
//
//    // TODO we still need to alpha-convert the names
//
//
//    // now collapse everything
//    unsigned next_free = 0;
//    unsigned current_item = 0;
//    while (current_item < rule_las.size()) {
//        while (current_item < rule_las.size() && rule_las[current_item].valid)
//            // TODO noch nicht fertig
//            ++current_item;
//    }
//}
//
//void merge_rhs(std::vector<RuleLA> & rule_las, const unsigned var_pos, const unsigned la) {
//    unsigned i = 0;
//    while (i < rule_las.size()) {
//        // search items
//        while (rule_las[i].get_rhs_la(var_pos) != la && i < rule_las.size())
//            ++i;
//        if (i == rule_las.size())
//            break;
//        unsigned j = i;
//        while (j < rule_las.size() && rule_las[j].get_rhs_la(var_pos) == la)
//            ++j;
//        assert (rule_las[j].get_rhs_la(var_pos) == la + 1);
//        while (j < rule_las.size() && rule_las[i].get_rhs_la(var_pos) == la) {
//            assert (rule_las[i].get_rhs_la(var_pos) + 1 == rule_las[j].get_rhs_la(var_pos));
//            assert (rule_las[i].get_lhs_la() == rule_las[j].get_lhs_la());
//            for (unsigned k = 0; k < rule_las[i].rhs_size; ++k) {
//                if (k != var_pos)
//                    assert(rule_las[i].get_rhs_la(k) == rule_las[j].get_rhs_la(j));
//            }
//
//            rule_las[i].weight += rule_las[j].weight;
//            rule_las[j].valid = false;
//            ++i;
//            ++j;
//        }
//    }
//}
//
//void merge_lhs(std::vector<RuleLA> & rule_las, const unsigned lhs) {
//    const unsigned block_size = pow(2, rule_las[0].rhs_size);
//    unsigned i = block_size * lhs;
//    unsigned j = block_size * lhs + rule_las.size() / 2;
//
//    // we assume a particular order on of the las in rule_las,
//    // i.e., the first half has annotiton 0, the second one annotation j
//    for (; i < (lhs + 1) * block_size; ++i) {
//        assert (rule_las[i].get_lhs_la() + 1 == rule_las[j].get_lhs_la());
//        for (unsigned k = 0; k < rule_las[i].rhs_size; ++k) {
//            assert(rule_las[i].get_rhs_la(k) == rule_las[j].get_rhs_la(j));
//        }
//        rule_las[i].weight += rule_las[j].weight;
//        rule_las[j].valid = false;
//        ++j;
//    }
//}




#endif //STERMPARSER_TRACE_H
