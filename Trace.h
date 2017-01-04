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
#include "util.h"
#include "SplitMergeUtil.h"
#include <functional>
#include <boost/range/irange.hpp>
#include <cmath>


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

    const unsigned traces_size() {
        return traces.size();
    }

    const std::pair<
              std::vector<std::pair<Nonterminal
                    , std::pair<std::vector<std::pair<Position, Position>>
                    , std::pair<std::vector<std::pair<Position, Position>>
                    , std::vector<std::pair<Position, Position>>
                    >>>>
            , std::pair<
                std::vector<std::vector<std::pair<unsigned, std::vector<unsigned>>>>
                    , unsigned
            >> serialize(unsigned trace_id) {
            std::vector<std::pair<Nonterminal
                , std::pair<std::vector<std::pair<Position, Position>>
                        , std::pair<std::vector<std::pair<Position, Position>>
                                , std::vector<std::pair<Position, Position>>
                        >>>> the_items;
        unsigned counter = 0;
        std::map<ParseItem<Nonterminal, Position>, unsigned> item_map;

        for (const auto entry : traces.at(trace_id)) {
            const ParseItem<Nonterminal, Position> item = entry.first;
            item_map[item] = the_items.size();
            the_items.push_back(std::make_pair(item.nonterminal, std::make_pair(item.spans_inh, std::make_pair(item.spans_syn, item.spans_lcfrs))));

        }

        std::vector<std::vector<std::pair<unsigned, std::vector<unsigned>>>> the_trace;

        for (const auto entry : traces.at(trace_id)) {
            const ParseItem<Nonterminal, Position> item = entry.first;
            unsigned lhs_id = item_map.at(item);
            std::vector<std::pair<unsigned, std::vector<unsigned>>> the_witnesses;
            for (auto witness : entry.second) {
                unsigned rule_id = witness.first->id;
                std::vector<unsigned> rhs_items;
                for (auto item : witness.second) {
                    rhs_items.push_back(item_map.at(*item));
                }
                the_witnesses.push_back(std::make_pair(rule_id, rhs_items));
            }

            the_trace.push_back(the_witnesses);
        }

        return std::make_pair(the_items, std::make_pair(the_trace, item_map.at(goals[trace_id])));
    };

    void deserialize(const std::pair<
            std::vector<std::pair<Nonterminal
                    , std::pair<std::vector<std::pair<Position, Position>>
                            , std::pair<std::vector<std::pair<Position, Position>>
                                    , std::vector<std::pair<Position, Position>>
                            >>>>
            , std::pair<
                    std::vector<std::vector<std::pair<unsigned, std::vector<unsigned>>>>
                    , unsigned
            >> & serial_info
            , SDCP<Nonterminal, Terminal> & sDCP
            ) {
        const std::vector<std::pair<Nonterminal
                , std::pair<std::vector<std::pair<Position, Position>>
                        , std::pair<std::vector<std::pair<Position, Position>>
                                , std::vector<std::pair<Position, Position>>
                        >>>> & items = serial_info.first;
        const std::vector<std::vector<std::pair<unsigned, std::vector<unsigned>>>> & trace_info = serial_info.second.first;
        const unsigned goal = serial_info.second.second;

        std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> parse_items;
        for (auto enc : items) {
            std::shared_ptr<ParseItem<Nonterminal, Position>> parse_item = std::make_shared<ParseItem<Nonterminal, Position>>(
                    ParseItem<Nonterminal, Position>()
            );
            parse_item->nonterminal = enc.first;
            parse_item->spans_inh = enc.second.first;
            parse_item->spans_syn = enc.second.second.first;
            parse_item->spans_lcfrs = enc.second.second.second;
            parse_items.push_back(parse_item);
        }

        std::map<
                ParseItem<Nonterminal, Position>
                , std::vector<
                        std::pair<
                                std::shared_ptr<Rule<Nonterminal, Terminal>>
                                , std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>
                                >
                        >
                >
        > trace;
        for (unsigned item = 0; item < trace_info.size(); ++item){
            auto & entry = trace[*parse_items[item]];
            for (const std::pair<unsigned, std::vector<unsigned>> & witness : trace_info[item]) {
                std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> rhss;
                for (unsigned rhs : witness.second) {
                    rhss.push_back(parse_items[rhs]);
                }
                entry.push_back(std::make_pair(sDCP.get_rule_by_id(witness.first), rhss));
            }
        }
        add_trace_entry(trace, *parse_items[goal], traces_size());
    }

    // const double plus_infinity = std::numeric_limits<double>::max();
    const double minus_infinity = -std::numeric_limits<double>::infinity();

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
            const auto sum = [](const double x, const double y) -> double {
                const double minus_infinity = -std::numeric_limits<double>::infinity();
                if (x == minus_infinity)
                    return y;
                else if (y == minus_infinity)
                    return x;
                // return log(exp(x) + exp(y));
                // cf. wiki, better accuracy with very small probabilites
                if (x >= y)
                    return x + log1p(exp(y - x));
                else
                    return y + log1p(exp(x - y));
            };
            const auto prod = [] (double x, double y) -> double {return x + y;};;
            const auto division = [] (double x, double y) -> double {return x - y;};;
            const double root = 0.0;
            const double leaf = 0.0;
            const double zero = minus_infinity;

//        }

        unsigned epoch = 0;


        std::cerr << "Epoch " << epoch << "/" << n_epochs << ": ";
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
                if (debug) std::cerr << *i << " ";
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
            std::cerr << "Epoch " << epoch << "/" << n_epochs << ": ";
            if (debug) {
                for (unsigned i = 0; i < rule_weights.size(); ++i) {
                    std::cerr << rule_weights[i] << " ";
                }
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

    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> split_merge_id(
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const unsigned n_epochs, const unsigned n_nonts, const unsigned split_merge_cycles, const double merge_threshold
    ) {
        auto nont_idx_f = [](const unsigned nont) -> unsigned { return nont; };

        std::cerr << "building normalization groups" << std::endl;

        std::vector<std::vector<unsigned>> normalization_groups;
        for (unsigned rule_idx = 0; rule_idx < rule_to_nonterminals.size(); ++rule_idx) {
            if (rule_to_nonterminals[rule_idx].size() > 0) {
                if (normalization_groups.size() <= rule_to_nonterminals[rule_idx][0]) {
                    normalization_groups.resize(rule_to_nonterminals[rule_idx][0] + 1);
                }
                normalization_groups[rule_to_nonterminals[rule_idx][0]].push_back(rule_idx);
            }
        }
        if (debug) {
            unsigned i = 0;
            for (auto rtn : rule_to_nonterminals) {
                std::cerr << i << ": ";
                unsigned j = 0;
                for (auto n : rtn) {
                    if (j == 1) {
                        std::cerr << "-> ";
                    }
                    std::cerr << n << " ";
                    ++j;
                }
                std::cerr << ";" << std::endl;
                ++i;
            }
            for (unsigned i = 0; i < normalization_groups.size(); ++i) {
                std::cerr << i << " : { ";
                for (auto n : normalization_groups[i]) {
                    std::cerr << n << " ";
                }
                std::cerr << "} " ;
            }
            std::cerr << std::endl;
        }

        std::cerr << "starting split merge training" << std::endl;
        std::cerr << "# nonts: " << n_nonts << std::endl;

        return split_merge(rule_weights, rule_to_nonterminals, normalization_groups, n_epochs, nont_idx_f,
                           split_merge_cycles, n_nonts, merge_threshold);
    };

    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> split_merge(
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const unsigned n_epochs, const std::map<Nonterminal, unsigned> &nont_idx, const unsigned split_merge_cycles, const double merge_threshold
    ) {
        auto nont_idx_f = [&](const Nonterminal nont) -> unsigned { return nont_idx.at(nont); };

        std::cerr << "building normalization groups" << std::endl;

        std::vector<std::vector<unsigned>> normalization_groups;
        for (unsigned rule_idx = 0; rule_idx < rule_to_nonterminals.size(); ++rule_idx) {
            if (rule_to_nonterminals[rule_idx].size() > 0) {
                if (normalization_groups.size() <= rule_to_nonterminals[rule_idx][0]) {
                    normalization_groups.resize(rule_to_nonterminals[rule_idx][0] + 1);
                }
                normalization_groups[rule_to_nonterminals[rule_idx][0]].push_back(rule_idx);
            }
        }
        if (debug) {
            unsigned i = 0;
            for (auto rtn : rule_to_nonterminals) {
                std::cerr << i << ": ";
                unsigned j = 0;
                for (auto n : rtn) {
                    if (j == 1) {
                        std::cerr << "-> ";
                    }
                    std::cerr << n << " ";
                    ++j;
                }
                std::cerr << ";" << std::endl;
                ++i;
            }
            for (unsigned i = 0; i < normalization_groups.size(); ++i) {
                std::cerr << i << " : { ";
                for (auto n : normalization_groups[i]) {
                    std::cerr << n << " ";
                }
                std::cerr << "} " ;
            }
            std::cerr << std::endl;
        }

        std::cerr << "starting split merge training" << std::endl;
        std::cerr << "# nonts: " << nont_idx.size() << std::endl;

        return split_merge(rule_weights, rule_to_nonterminals, normalization_groups, n_epochs, nont_idx_f,
                           split_merge_cycles, nont_idx.size(), merge_threshold);
    };

    template<typename NontToIdx>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> split_merge(
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const std::vector<std::vector<unsigned>> &normalization_groups, const unsigned n_epochs,
            const NontToIdx nont_idx, const unsigned split_merge_cycles, const unsigned n_nonts, const double merge_threshold
    ) {

        // TODO implement flexibility for semiring!
        const auto sum = [](const double x, const double y) -> double {
            const double minus_infinity = -std::numeric_limits<double>::infinity();
            if (x == minus_infinity)
                return y;
            else if (y == minus_infinity)
                return x;
            // return log(exp(x) + exp(y));
            // cf. wiki, better accuracy with very small probabilites
            if (x >= y)
                return x + log1p(exp(y - x));
            else
                return y + log1p(exp(x - y));
        };
        const auto difference = [](const double x, const double y) -> double {
            // const double minus_infinity = std::numeric_limits<double>::infinity();
            if (x >= y)
                return x + log(1 - exp(y - x));
            else
                return log(exp(x) - exp(y));
        };
        const auto prod = [](const double x, const double y) -> double { return x + y; };
        const auto division = [](const double x, const double y) -> double { return x - y; };
        const double root = 0.0;
        const double one = 0.0;
        const double leaf = 0.0;
        const double zero = minus_infinity;
        const double epsilon = log(0.0001);


        // the next two structures hold split-dimensions and
        // rule weights for latent annotated rules before and after
        // each split/merge cycle
        std::vector<unsigned> nont_dimensions = std::vector<unsigned>(n_nonts, 1);
        std::vector<std::vector<double>> rule_weights_la;
        for (const double & rule_weight : rule_weights) {
            // TODO log
            rule_weights_la.emplace_back(std::vector<double>(1, log(rule_weight)));
        }
        std::vector<double> root_weights = {root};


        std::vector<std::vector<double>> rule_weights_splitted;
        std::vector<double> root_weights_splitted;
        std::vector<std::vector<double>> rule_weights_merged;

        for (unsigned cycle = 0; cycle < split_merge_cycles; ++cycle) {
            std::vector<unsigned> split_dimensions;

            if (debug) std::cerr << "prepare split" << std::endl;

            for (const unsigned dim : nont_dimensions)
                split_dimensions.push_back(dim * 2);
            // splitting
            for (unsigned i = 0; i < rule_weights_la.size(); ++i) {
                const std::vector<double> &rule_weight = rule_weights_la[i];
                std::vector<unsigned> dimensions;
                for (auto nont : rule_to_nonterminals[i]) {
                    dimensions.push_back(split_dimensions[nont]);
                }
                const std::vector<double> split_probabilities = split_rule(rule_weight, dimensions);
                rule_weights_splitted.push_back(split_probabilities);
            }

            const double root_split = rand_split();
            root_weights_splitted = {prod(log(root_split), root), prod(log(1 - root_split), root)};

            rule_weights_la.clear();

            std::cerr << "em training after " << cycle + 1 << ". split" << std::endl;

            // em training
            do_em_training_la(rule_weights_splitted, root_weights_splitted, normalization_groups, n_epochs, split_dimensions,
                              rule_to_nonterminals, nont_idx, zero, one, leaf, prod, sum, division);

            for (auto rule_weights_ : rule_weights_splitted) {
                for (auto rule_weight : rule_weights_) {
                    if (rule_weight > sum(one, epsilon)) {
                        std::cerr << "bad rule weight: " << rule_weight << std::endl;
                    }
                    assert(rule_weight <= sum(one, epsilon));
                }
            }

            // determine merges
            const auto merge_info = merge_prepare(rule_weights_splitted, root_weights_splitted, split_dimensions,
                                                  rule_to_nonterminals, nont_idx, leaf, zero, one, sum, prod, division,
                                                  difference, log(merge_threshold));


            // nonterminal -> new_la -> contributing old_las
            const std::vector<std::vector<std::vector<unsigned>>> & merge_selection = std::get<0>(merge_info);
            const std::vector<unsigned> & new_nont_dimensions = std::get<1>(merge_info);
            const std::vector<std::vector<double>> merge_factors = std::get<2>(merge_info);

            if (debug) {
                std::cerr << "merge factors ";
                for (auto factors : merge_factors) {
                    std::cerr << "{ ";
                    for (auto factor : factors) {
                        std::cerr << factor << " ";
                    }
                    std::cerr << " } ";
                }
                std::cerr << std::endl;
            }

            // merging
            std::vector<std::vector<double>> rule_weights_merged;
            for (unsigned i = 0; i < rule_weights_splitted.size(); ++i) {
                std::vector<unsigned> old_dimensions;
                std::vector<unsigned> new_dimensions;
                //new_dimensions.reserve(rule_to_nonterminals[i].size());
                std::vector<std::vector<std::vector<unsigned>>> merges;
                //merges.reserve(rule_to_nonterminals[i].size());
                const std::vector<double> & lhn_merge_factors = merge_factors[rule_to_nonterminals[i][0]];
                for (auto nont : rule_to_nonterminals[i]) {
                    old_dimensions.push_back(split_dimensions[nont]);
                    new_dimensions.push_back(merge_selection[nont].size());
                    merges.push_back(merge_selection[nont]);
                }
                rule_weights_merged.push_back(
                        merge_rule(rule_weights_splitted[i], old_dimensions, new_dimensions, merges,
                                   lhn_merge_factors, sum, prod, zero));
            }

            rule_weights_splitted.clear();

            // em training
            do_em_training_la(rule_weights_merged, root_weights, normalization_groups, n_epochs, new_nont_dimensions,
                              rule_to_nonterminals, nont_idx, zero, one, leaf, prod, sum, division);

            for (auto rule_weights_ : rule_weights_merged) {
                for (auto rule_weight : rule_weights_) {
                    if (rule_weight > sum(one, epsilon)) {
                        std::cerr << "bad rule weight: " << rule_weight << std::endl;
                    }
                    assert(rule_weight <= sum(one, epsilon));
                }
            }

            // create valid state after split/merge cycle
            nont_dimensions = new_nont_dimensions;
            rule_weights_la = rule_weights_merged;
        }

        // undo log conversion
        for (auto & weights : rule_weights_la) {
            for (double & weight : weights) {
                weight = exp(weight);
            }
        }

        return std::make_pair(nont_dimensions, rule_weights_la);
    }

    template<typename Val, typename Accum, typename Accum2, typename NontToIdx>
    std::pair<std::map<ParseItem < Nonterminal, Position>, std::vector<Val>>,
    std::map<ParseItem < Nonterminal, Position>, std::vector<Val>>>
    io_weights_la(const std::vector<std::vector<Val>> &rules, const std::vector<unsigned> &nont_dimensions,
                  const std::vector<std::vector<unsigned>> rule_id_to_nont_ids, const NontToIdx nont_idx,
                  const Val leaf, const std::vector<Val> &root, const Val zero, const Val one, const Accum sum, const Accum2 prod,
                  const unsigned i) const {


        // TODO implement for general case (== no topological order) approximation of inside weights
        assert (topological_orders.size() > i && topological_orders[i].size() > 0);

        const auto & topological_order = topological_orders[i];

        // computation of inside weights
        std::map<ParseItem<Nonterminal, Position>, std::vector<Val>> inside_weights;
        for (const auto & item : topological_order) {
            inside_weights[item] = std::vector<Val>(nont_dimensions[nont_idx(item.nonterminal)], zero);
            std::vector<Val> & inside_weight = inside_weights[item];
            for (const auto & witness : traces[i].at(item)) {
                std::vector<std::vector<Val>> nont_vectors;
                // nont_vectors.reserve(witness.second.size());
                std::vector<unsigned> rule_dim;
                for (auto nont : rule_id_to_nont_ids[witness.first->id]) {
                    rule_dim.push_back(nont_dimensions[nont]);
                }
                for (const auto & dep_item : witness.second) {
                    nont_vectors.push_back(inside_weights.at(*dep_item));
                }
                inside_weight = dot_product(sum, inside_weight, compute_inside_weights(rules[witness.first->id], nont_vectors,
                                                                       rule_dim, zero, one, sum, prod));
            }
        }

        // TODO implement for general case (== no topological order) solution by gauss jordan
        std::map<ParseItem<Nonterminal, Position>, std::vector<Val>> outside_weights;
        std::vector<Val> empty = std::vector<Val>(0,0);
        for (int j = topological_order.size() - 1; j >= 0; --j) {
            const ParseItem<Nonterminal, Position> & item = topological_order[j];
            outside_weights[item] = std::vector<Val>(nont_dimensions[nont_idx(item.nonterminal)], zero);
            std::vector<Val> & outside_weight = outside_weights[item];

            if (item == goals[i])
                outside_weight = dot_product(sum, outside_weight, root);

            for (int k = topological_order.size() - 1; k > j; --k){
                const ParseItem<Nonterminal, Position> & parent = topological_order[k];

                for (const auto & witness : traces[i].at(parent)) {
                    bool item_found = false;
                    std::vector<std::vector<Val>> relevant_inside_weights;
                    std::vector<unsigned> rule_dim;
                    rule_dim.push_back(nont_dimensions[nont_idx(parent.nonterminal)]);

                    unsigned item_pos = 0;
                    for (const auto & rhs_item : witness.second) {
                        rule_dim.push_back(nont_dimensions[nont_idx(rhs_item->nonterminal)]);
                        if (*rhs_item == item) {
                            item_found = true;
                            relevant_inside_weights.push_back(std::vector<double>(rule_dim.back(), one));
                            continue;
                        } else if (!item_found)
                            ++item_pos;
                        relevant_inside_weights.push_back(inside_weights[*rhs_item]);
                    }
                    if (!item_found)
                        continue;

                    const std::vector<Val> new_weights
                            = compute_outside_weights(
                              rules[witness.first->id]
                            , outside_weights[parent]
                            , relevant_inside_weights
                            , rule_dim
                            , zero
                            , one
                            , sum
                            , prod
                            , item_pos);

                    outside_weight = dot_product( sum
                               , outside_weight
                               , new_weights);
                }
            }
        }

        return std::make_pair(inside_weights, outside_weights);
    }


    template <typename NontToIdx, typename Val, typename Accum1, typename Accum2, typename Accum3>
    void do_em_training_la(
            std::vector<std::vector<double>> & rule_weights
            , std::vector<Val> & the_root_weights
            , const std::vector<std::vector<unsigned>> & normalization_groups
            , const unsigned n_epochs
            , const std::vector<unsigned> & nont_dimensions
            , const std::vector<std::vector<unsigned>> & rule_to_nont_ids
            , const NontToIdx nont_idx
            , const Val zero, const Val one, const Val leaf, const Accum1 prod, const Accum2 sum, const Accum3 division
    ){

        unsigned epoch = 0;

        std::cerr <<"Epoch " << epoch << "/" << n_epochs << ": ";
        if (debug) {
            for (unsigned i = 0; i < rule_weights.size(); ++i) {
                std::cerr << i << " { ";
                for (double elem : rule_weights[i])
                    std::cerr << exp(elem) << " ";
                std::cerr << " } , ";
            }
        }
        std::cerr << std::endl;
        if (debug) {
            bool first = true;
            std::cerr << "Root weights { ";
            for (auto weight : the_root_weights) {
                if (!first)
                    std::cerr << " , ";
                first = false;
                std::cerr << exp(weight);
            }
            std::cerr << " } " << std::endl;
        }

        std::vector<std::vector<Val>> rule_counts;
        std::vector<std::vector<unsigned>> rule_dimensions;
        for (auto nont_ids : rule_to_nont_ids) {
            unsigned size = 1;
            std::vector<unsigned> rule_dimension;
            for (auto nont_id : nont_ids) {
                rule_dimension.push_back(nont_dimensions[nont_id]);
                size *= nont_dimensions[nont_id];
            }
            rule_dimensions.push_back(rule_dimension);
        }

        while (epoch < n_epochs) {

            // expectation
            assert (rule_counts.size() == 0);

            // initialize rule counts
            for (const auto & dims : rule_dimensions) {
                rule_counts.push_back(std::vector<Val>(calc_size(dims), zero));
            }

            std::vector<Val> root_counts = std::vector<Val>(nont_dimensions[nont_idx(goals[0].nonterminal)], zero);


            for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
                auto trace = traces[trace_id];
                if (trace.size() == 0)
                    continue;

                const auto tr_io_weight = io_weights_la(rule_weights, nont_dimensions, rule_to_nont_ids, nont_idx, leaf, the_root_weights, zero, one, sum, prod, trace_id);
                if (debug) {
                    for (const auto &item : get_order(trace_id)) {
                        std::cerr << "T: " << item << " (idx: " << nont_idx(item.nonterminal) << " )" << std::endl  ;

                        for (unsigned offset = 0; offset < nont_dimensions[nont_idx(item.nonterminal)]; ++offset) {
                            std::cerr << "    " << tr_io_weight.first.at(item)[offset] << " "
                                      << tr_io_weight.second.at(item)[offset] << std::endl;

                        }
                    }
                    std::cerr << std::endl;
                }
//
                root_counts = dot_product(sum, root_counts, dot_product(prod, tr_io_weight.first.at(goals[trace_id]), tr_io_weight.second.at(goals[trace_id])));

                const auto instance_root_weights = tr_io_weight.first.at(goals[trace_id]);
                const Val instance_root_weight = reduce(sum, dot_product(prod, instance_root_weights, the_root_weights), zero);
                if (debug)
                    std::cerr << "instance root weight: " << instance_root_weight << std::endl;

                for (auto & pair : trace) {
                    const std::vector<Val> & lhn_outside_weights = tr_io_weight.second.at(pair.first);
                    if (debug) {
                        std::cerr << pair.first << std::endl;
                    }
                    for (const auto & witness : pair.second) {
                        const int rule_id = witness.first->id;

                        std::vector<std::vector<Val>> nont_weight_vectors;
                        nont_weight_vectors.reserve(witness.second.size());
                        for (const auto & rhs_item : witness.second) {
                            nont_weight_vectors.push_back(tr_io_weight.first.at(*rhs_item));
                        }

                        std::vector<Val> rule_val;
                        rule_val = compute_rule_frequency(rule_weights[rule_id], lhn_outside_weights, nont_weight_vectors, rule_dimensions[rule_id], zero, one, sum, prod);
                        rule_val = scalar_product(division, rule_val, instance_root_weight);

                        if (debug) {
                            std::cerr << rule_id << ": {";
                            for (auto val : rule_val)
                                std::cerr << val << " ";
                            std::cerr << " }" << std::endl;
                        }
                        rule_counts[rule_id] = dot_product(sum, rule_counts[rule_id], rule_val);
                    }
                }
            }

            if (debug) {
                unsigned i = 0;
                for (auto rule : rule_counts) {
                    std::cerr << i << ": { ";
                    for (auto count : rule) {
                        std::cerr << count << " ";
                    }
                    std::cerr<<" } " << std::endl;
                    ++i;
                }
            }

            // maximization
            for (const std::vector<unsigned> & group : normalization_groups) {
                const unsigned group_dim = rule_dimensions[group[0]][0];
                std::vector<Val> group_counts = std::vector<Val>(group_dim, zero);
                for (auto member : group) {
                    const unsigned block_size = subdim(rule_dimensions[member]);
                    // reduce([] (const unsigned x, const unsigned y) -> unsigned {return x * y;}, rule_dimensions[member], (unsigned) 1, (unsigned) 1);
                    for (unsigned dim : boost::irange((unsigned) 0, group_dim)) {
                        const std::vector<double>::iterator block_start = rule_counts[member].begin() + block_size * dim;
                        for (auto it = block_start; it != block_start + block_size; ++it) {
                            group_counts[dim] = sum(*it, group_counts[dim]);
                        }
                    }
                }
                if (debug) {
                    std::cerr << " { ";
                    for (auto count : group_counts) {
                        std::cerr << count << " ";
                    }
                    std::cerr << " } ";
                }
                for (auto member : group) {
                    const unsigned block_size = subdim(rule_dimensions[member]);
                    for (unsigned dim : boost::irange((unsigned) 0, group_dim)) {
                        if (group_counts[dim] > zero) {
                            const unsigned block_start = block_size * dim;
                            for (unsigned offset = block_start; offset < block_start + block_size; ++offset) {
                                *(rule_weights[member].begin() + offset) = division(*(rule_counts[member].begin() + offset), group_counts[dim]);
                            }
                        }
                    }
                }
            }
            if (debug) std::cerr << std::endl;

            epoch++;
            std::cerr <<"Epoch " << epoch << "/" << n_epochs << ": ";
            if (debug)
                for (unsigned i = 0; i < rule_weights.size(); ++i) {
                    std::cerr << i << " { ";
                    for (double elem : rule_weights[i])
                        std::cerr << exp(elem) << " ";
                    std::cerr << " } , ";
                }
            if (debug) std::cerr << std::endl;

            // maximize root weights:
            const Val likelihood = reduce(sum, root_counts, zero);
            std::cerr << "corpus prob. sum " << exp(likelihood) << std::endl;
            const unsigned len = the_root_weights.size();
            the_root_weights.clear();
            if (debug) std::cerr << "single root weights: ";
            for (const auto weight : root_counts) {
                if (debug) std::cerr << exp(weight) << "/" << exp(division(weight, likelihood)) << " ";
                the_root_weights.push_back(division(weight, likelihood));
            }
            assert(len == the_root_weights.size());
            std::cerr << std::endl;

            if (debug) {
                std::cerr << "Nont sums: ";
                unsigned i = 0;
                for (auto group : normalization_groups) {
                    std::vector<double> dim_weights = std::vector<double>(nont_dimensions[i], zero);
                    for (auto rule_id : group) {
                        unsigned size = rule_weights[rule_id].size();
                        for (unsigned weight_id = 0; weight_id < size; ++weight_id) {
                            const unsigned index = weight_id / (size / nont_dimensions[i]);
                            dim_weights[index] = sum(dim_weights[index], rule_weights[rule_id][weight_id]);
                        }
                    }
                    std::cerr << " { ";
                    for (auto weight_sum : dim_weights) {
                        std::cerr << exp(weight_sum) << " ";
                    }
                    std::cerr << " } ";
                    ++i;
                }
                std::cerr << std::endl;
            }
            if (debug) {
                bool first = true;
                std::cerr << "Root weights { ";
                for (auto weight : the_root_weights) {
                    if (!first)
                        std::cerr << " , ";
                    first = false;
                    std::cerr << exp(weight);
                }
                std::cerr << " } " << std::endl;
            }

            rule_counts.clear();
            root_counts.clear();
        }

        // clear memory for rule counts
//        for (auto ptr : rule_counts) {
//            free(ptr);
//        }

        // return rule_weights;
    }

    template <typename Val, typename Accum1, typename Accum2, typename Accum3, typename Accum4, typename NontToIdx>
    std::tuple< std::vector<std::vector<std::vector<unsigned>>>
              , std::vector<unsigned>
              , std::vector<std::vector<Val>>
              >
    merge_prepare(const std::vector<std::vector<Val>> & rule_weights
            , const std::vector<Val> & root_weights
            , const std::vector<unsigned> & nont_dimensions
            , const std::vector<std::vector<unsigned>> & rule_ids_to_nont_ids
                       , const NontToIdx nont_idx
                       , const Val leaf
                       , const Val zero
                       , const Val one
                       , const Accum1 sum
                       , const Accum2 prod
                       , const Accum3 quotient
                       , const Accum4 difference
                       , const Val merge_threshold
            ) {


        // first we compute the fractions p_1, p_2
        // with which the probabality mass is shared between merged latent states

        // this is prepared with computing globally averaged outside weights
        std::vector<std::vector<Val>> global_nont_outside_weights;
        for (auto dim : nont_dimensions) {
            global_nont_outside_weights.emplace_back(std::vector<Val>(dim, zero));
        }

        // computing out(A_x) for every A ∈ N and x ∈ X_A
        for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
            std::map<Nonterminal, std::pair<std::vector<Val>, unsigned>> nonterminal_count;
            const auto io_weight = io_weights_la(rule_weights, nont_dimensions, rule_ids_to_nont_ids, nont_idx, leaf, root_weights, zero, one, sum, prod, trace_id);

            for (const auto & pair : traces[trace_id]) {
                const ParseItem<Nonterminal, Position> & item = pair.first;

                const std::vector<Val> & outside_weight = io_weight.second.at(item);

                if (nonterminal_count.count(item.nonterminal)) {
                    std::pair<std::vector<Val>, unsigned> &entry = nonterminal_count.at(item.nonterminal);
                    entry.first = dot_product(sum, entry.first, outside_weight);
                    ++entry.second;
                } else {
                    nonterminal_count[item.nonterminal] = std::make_pair(outside_weight, 1);
                }
            }

            for (const auto pair : nonterminal_count) {
                std::vector<Val> & gow = global_nont_outside_weights[nont_idx(pair.first)];
                gow = dot_product(sum, gow,
                                  scalar_product(quotient, pair.second.first, (Val) pair.second.second));
            }
        }

        // finally we compute the fractions
        std::vector<std::vector<Val>> p;
        for (auto las_weights : global_nont_outside_weights) {
            p.emplace_back(std::vector<Val>());
            for (unsigned i = 0; i < las_weights.size(); i = i + 2) {
                const Val combined_weight = sum(las_weights[i], las_weights[i+1]);
                p.back().push_back(quotient(las_weights[i], combined_weight));
                p.back().push_back(quotient(las_weights[i+1], combined_weight));
            }
        }

        // now we approximate the likelihood Δ of merging two latent states
        std::vector<std::vector<Val>> merge_delta;
        for (auto dim : nont_dimensions) {
            merge_delta.push_back(std::vector<Val>(dim / 2, one));
        }

        for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
            const auto io_weight = io_weights_la(rule_weights, nont_dimensions, rule_ids_to_nont_ids, nont_idx, leaf, root_weights, zero, one, sum, prod, trace_id);
            for (const auto & pair : traces[trace_id]) {
                const ParseItem<Nonterminal, Position> &item = pair.first;

                // compute Q( item )
                Val denominator = zero;
                for (unsigned dim : boost::irange((unsigned) 0, nont_dimensions[nont_idx(item.nonterminal)])) {
                    const Val in = io_weight.first.at(item)[dim];
                    const Val out = io_weight.second.at(item)[dim];
                    denominator = sum(denominator, prod(in, out));
                    assert(! isnan(denominator));
                }

                // in of some item can be zero in certain LA-dimensions
                // since LA-rule weights may converge to zero
                // we ignore those dimensions in Δ computation
                if (denominator == zero)
                    continue;

                for (unsigned dim = 0; dim < nont_dimensions[nont_idx(item.nonterminal)]; dim = dim+2) {
                    Val nominator = denominator;
                    const Val in1 = io_weight.first.at(item)[dim];
                    const Val in2 = io_weight.first.at(item)[dim+1];
                    const Val out1 = io_weight.second.at(item)[dim];
                    const Val out2 = io_weight.second.at(item)[dim+1];
                    const unsigned nont = nont_idx(item.nonterminal);
                    const Val p1 = p[nont][dim];
                    const Val p2 = p[nont][dim+1];

                    const Val out_merged = sum(out1, out2);
                    const Val in_merged = sum(prod(p1, in1), prod(p2, in2));

                    nominator = sum(nominator, prod(in_merged, out_merged));
                    nominator = difference(nominator, prod(in1, out1));
                    nominator = difference(nominator, prod(in2, out2));

                    const Val Q2 = quotient(nominator, denominator);

                    // TODO: find a more generic solution to floating point accurcay problems
                    const Val Q = log(exp(denominator)
                                      + exp(in_merged + out_merged)
                                      - exp(in1 + out1)
                                      - exp(in2 + out2))
                                   - denominator;

                    double & delta = merge_delta[nont][dim / 2];

                    delta = prod(delta, Q);


                    if (isnan(delta)) {
                        std::cerr << "bad fraction " << nominator << " / " << denominator << " = " << Q << std::endl;
                        std::cerr << "prod(in_merged, out_merged) = " << prod(in_merged, out_merged) << std::endl;
                        std::cerr << "prod(in1, out1) = " << prod(in1, out1) << std::endl;
                        std::cerr << "prod(in2, out2) = " << prod(in2, out2) << std::endl;
                        assert(!isnan(delta));
                    }

                }
            }
        }

        // evaluate Δ and build merge table accordingly
        std::vector<std::vector<std::vector<unsigned>>> merge_selection;
        std::vector<unsigned> new_nont_dimensions;
        unsigned nont = 0;

        if (debug) std::cerr << "merge deltas: ";
        for (auto delta : merge_delta) {
            if (debug) std::cerr << " { ";
            merge_selection.push_back(std::vector<std::vector<unsigned>>());
            for (unsigned dim = 0; dim < nont_dimensions[nont] / 2; ++dim) {
                if (debug) std::cerr << exp(delta[dim]) << " ";
                if (delta[dim] >= merge_threshold
                    // always merge initial symbol
                    || nont_idx(goals[0].nonterminal) == nont) {
                    merge_selection.back().push_back(std::vector<unsigned>());
                    merge_selection.back().back().push_back(dim * 2);
                    merge_selection.back().back().push_back(dim * 2 + 1);
                } else {
                    merge_selection.back().push_back(std::vector<unsigned>(1, dim * 2 ));
                    merge_selection.back().push_back(std::vector<unsigned>(1, dim * 2 + 1));
                }
            }
            if (debug) std::cerr << " } ";
            ++nont;
            new_nont_dimensions.push_back(merge_selection.back().size());
        }
        if (debug) std::cerr << std::endl;


        return std::make_tuple(merge_selection, new_nont_dimensions, p);
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
