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
#include <boost/operators.hpp>
#include <functional>

class Chance {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
public:
    double get_chance() {
        return distribution(generator);
    }
};

const double minus_infinity = -std::numeric_limits<double>::infinity();

class LogDouble : boost::operators<LogDouble> {
private:
    double x;
public:
    const double & get_Value() const {
        return x;
    };

    LogDouble() : x(minus_infinity) {} ;

    LogDouble(const double x) : x(x) {};
    bool operator<(const LogDouble& y) const {
        return x < y.get_Value();
    }

    LogDouble& operator= (const LogDouble & y) {
        x = y.get_Value();
        return *this;
    }

    bool operator==(const LogDouble & y) const {
        return x == y.get_Value();
    }

    LogDouble& operator+=(const LogDouble& y_){
        const double y = y_.get_Value();

        if (x == minus_infinity)
            x = y;
        else if (y == minus_infinity)
            ;
        // return log(exp(x) + exp(y));
        // cf. wiki, better accuracy with very small probabilites
        else if (x >= y)
            x = x + log1p(exp(y - x));
        else
            x = y + log1p(exp(x - y));
        return *this;
    }

    LogDouble& operator-= (const LogDouble & y_) {
            // const double minus_infinity = std::numeric_limits<double>::infinity();
            if (x >= y_.get_Value())
                x += log(1 - exp(y_.get_Value() - x));
            else
                x = log(exp(x) - exp(y_.get_Value()));
            return *this;
        };

    LogDouble& operator*=(const LogDouble& y_){
        x = x + y_.get_Value();
        return *this;
    }

    LogDouble& operator/=(const LogDouble& y_) {
        x = x - y_.get_Value();
        return *this;
    }

    static const LogDouble one()  {
        return LogDouble(0);
    }

    static const LogDouble zero() {
        return LogDouble(-std::numeric_limits<double>::infinity());
    }

    static const LogDouble to(const double x) {
        return LogDouble(log(x));
    }

    double from() const {
        return exp(x);
    }

    static const LogDouble add_subtract2_divide(const LogDouble base, const LogDouble add, const LogDouble sub1, const LogDouble sub2, const LogDouble div) {
        return LogDouble(log(exp(base.get_Value())
                          + exp(add.get_Value())
                          - exp(sub1.get_Value())
                          - exp(sub2.get_Value()))
                      - div.get_Value());
    }

};


class Double : boost::operators<Double> {
private:
    double x;
    const double minus_infinity = -std::numeric_limits<double>::infinity();
public:
    const double & get_Value() const {
        return x;
    };
    Double(const double x) : x(x) {};
    bool operator<(const Double& y) const {
        return x < y.get_Value();
    }

    Double() : x(0) {};

    Double& operator=(const Double& y) {
        x = y.get_Value();
        return *this;
    }

    bool operator==(const Double & y) const {
        return x == y.get_Value();
    }

    Double& operator+=(const Double& y_){
        x += y_.get_Value();
        return *this;
    }

    Double& operator-=(const Double& y_){
        x -= y_.get_Value();
        return *this;
    }

    Double operator-() const {
        return Double(-x);
    }

    Double operator*=(const Double& y_){
        x *= y_.get_Value();
        return *this;
    }

    Double operator/=(const Double& y_) {
        x = x / y_.get_Value();
        return *this;
    }

    static const Double one()  {
        return Double(1.0);
    }

    static const Double zero() {
        return Double(0.0);
    }

    static const Double to(const double x) {
        return Double(x);
    }

    double from() const {
        return x;
    }

    static Double add_subtract2_divide(const Double base, const Double add, const Double sub1, const Double sub2, const Double div) {
        return Double(((base + add - sub1) - sub2) / div);
    }

};



std::ostream &operator<<(std::ostream &os, const LogDouble &log_double){
    os << " L" << log_double.get_Value();
    return os;
}

std::ostream &operator<<(std::ostream &os, const Double &x){
    os << x.get_Value();
    return os;
}

template <typename Nonterminal>
class GrammarInfo {
public:
    typename std::function<unsigned(Nonterminal)> nont_idx;
    std::vector<std::vector<unsigned>> normalization_groups;
    std::vector<std::vector<unsigned>> rule_to_nonterminals;

    GrammarInfo() {}
    GrammarInfo(
              const std::function<unsigned(Nonterminal)> nont_idx
            , const std::vector<std::vector<unsigned>> normalization_groups
            , const std::vector<std::vector<unsigned>> rule_to_nonterminals
            )
            : nont_idx(nont_idx)
            , normalization_groups(normalization_groups)
            , rule_to_nonterminals(rule_to_nonterminals
            ) {}
};



template <typename Nonterminal, typename Terminal, typename Position>
class TraceManager {
private:
    std::vector<
            MAPTYPE<
                      ParseItem<Nonterminal, Position>
                    , std::vector<
                            std::pair<
                                      std::shared_ptr<Rule<Nonterminal, Terminal>>
                                    , std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>>
                            >
                    >
            >
    > traces;
    std::vector<
            MAPTYPE<
                    ParseItem<Nonterminal, Position>
                    , std::vector<
                            std::tuple<
                                      std::shared_ptr<Rule<Nonterminal, Terminal>>
                                    , std::shared_ptr<std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>>>
                                    , std::shared_ptr<ParseItem<Nonterminal, Position>>
                                    , unsigned
                                    >
                            >
                    >
            > traces_reverse;

    std::vector<
            std::vector<
                ParseItem<Nonterminal, Position>
            >
            > topological_orders;
    std::vector<ParseItem<Nonterminal, Position>> goals;


    // empty default values
    const MAPTYPE<
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
    std::unordered_set<ParseItem<Nonterminal, Position>> inserted_items;

    const bool debug = false;

public:
    TraceManager(bool debug=false) : debug(debug) {}

    void add_trace_from_parser(const SDCPParser<Nonterminal, Terminal, Position> & parser, unsigned i){
        add_trace_entry(parser.get_trace(), *parser.goal, i);
    }

    void add_trace_entry(
            const MAPTYPE<
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

        // compute outgoing hyperedges for each item
        if (traces_reverse.size() <= i)
            traces_reverse.resize(i + 1);
        MAPTYPE<ParseItem<Nonterminal, Position>
                , std::vector<
                        std::tuple<
                                std::shared_ptr<Rule<Nonterminal, Terminal>>
                                , std::shared_ptr<std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>>>
                                , std::shared_ptr<ParseItem<Nonterminal, Position>>
                                , unsigned
                        >
                >
        > & trace_reverse = traces_reverse[i];
        trace_reverse.clear();
        for (auto entry: trace) {
            auto parent_ptr = std::make_shared<ParseItem<Nonterminal, Position>>(entry.first);
            for (auto &witness : entry.second) {
                auto rule = witness.first;
                auto witness_ptr = std::make_shared<std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>>>(
                        witness.second);
                for (unsigned i = 0; i < witness.second.size(); ++i) {
                    auto &item = witness.second[i];
                    trace_reverse[*item].push_back(std::make_tuple(rule, witness_ptr, parent_ptr, i));
                }
            }
        }
    }

    const MAPTYPE<
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

    unsigned traces_size() {
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
        MAPTYPE<ParseItem<Nonterminal, Position>, unsigned> item_map;

        for (const auto entry : traces.at(trace_id)) {
            const ParseItem<Nonterminal, Position> item = entry.first;
            item_map[item] = the_items.size();
            the_items.push_back(std::make_pair(item.nonterminal, std::make_pair(item.spans_inh, std::make_pair(item.spans_syn, item.spans_lcfrs))));

        }

        std::vector<std::vector<std::pair<unsigned, std::vector<unsigned>>>> the_trace;

        for (const auto entry : traces.at(trace_id)) {
            const ParseItem<Nonterminal, Position> item = entry.first;
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

        MAPTYPE<
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

    template<typename Val>
    std::pair<MAPTYPE<ParseItem<Nonterminal, Position>, Val>,
              MAPTYPE<ParseItem<Nonterminal, Position>, Val>>
            io_weights(const std::vector<Val> & rules, const unsigned i) const {
        // TODO implement for general case (== no topological order) approximation of inside weights
        assert (topological_orders.size() > i && topological_orders[i].size() > 0);

        const auto & topological_order = topological_orders[i];

        MAPTYPE<ParseItem<Nonterminal, Position>, Val> inside_weights;

        for (const auto & item : topological_order) {
            inside_weights[item] = Val::zero();
            for (const auto & witness : traces[i].at(item)) {
                Val val = rules[witness.first->id];
                for (const auto & dep_item : witness.second) {
                    val = val * inside_weights.at(*dep_item);
                }
                if (debug && (val == Val::zero())) { //|| val == -Val::zero())) {
                    std::cerr << "rule weight: " << rules[witness.first->id] <<std::endl;
                    for (const auto & dep_item : witness.second) {
                        std::cerr << *dep_item << " " << inside_weights.at(*dep_item) << std::endl;
                    }
                }
                inside_weights[item] = (inside_weights.at(item) + val);
            }
        }

        // TODO implement for general case (== no topological order) solution by gauss jordan
        MAPTYPE<ParseItem<Nonterminal, Position>, Val> outside_weights;
        for (int j = topological_order.size() - 1; j >= 0; --j) {
            const ParseItem<Nonterminal, Position> & item = topological_order[j];
            Val val = Val::zero();
            if (item == goals[i])
                val = val + Val::one();
            if (traces_reverse[i].count(item)) {
                for (const auto &witness : traces_reverse[i].at(item)) {
                    const auto &rule = *(std::get<0>(witness));
                    const auto &siblings = *(std::get<1>(witness));
                    const auto &parent = *(std::get<2>(witness));
                    const unsigned position = std::get<3>(witness);
                    Val val_witness = outside_weights.at(parent);
                    val_witness *= rules[rule.id];
                    for (unsigned sib_position = 0; sib_position < siblings.size(); ++sib_position) {
                        if (sib_position != position)
                            val_witness *= inside_weights.at(*(siblings[sib_position]));
                    }
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

    template<typename Val>
    std::vector<double> do_em_training( const std::vector<double> & initial_weights
                       , const std::vector<std::vector<unsigned>> & normalization_groups
                       , const unsigned n_epochs){
        std::vector<Val> rule_weights;
        std::vector<Val> rule_counts;

        unsigned epoch = 0;


        std::cerr << "Epoch " << epoch << "/" << n_epochs << ": ";

        // potential conversion to log semiring:

        for (auto i = initial_weights.begin(); i != initial_weights.end(); ++i) {
            rule_weights.push_back(Val::to(*i));
            if (debug) std::cerr << *i << " / " << rule_weights.back() << " ";
        }

        std::cerr << std::endl;

        while (epoch < n_epochs) {
            // expectation
            rule_counts = std::vector<Val>(rule_weights.size(), Val::zero());
            for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
                auto trace = traces[trace_id];
                if (trace.size() == 0)
                    continue;

                const auto tr_io_weight = io_weights(rule_weights, trace_id);
                if (debug) {
                    for (const auto &item : get_order(trace_id)) {
                        std::cerr << "T: " << item << " " << tr_io_weight.first.at(item) << " "
                                  << tr_io_weight.second.at(item) << std::endl;
                    }
                    std::cerr << std::endl;
                }
                const Val root_weight = tr_io_weight.first.at(goals[trace_id]);
                for (auto & pair : trace) {
                    const Val lhn_outside_weight = tr_io_weight.second.at(pair.first);
                    for (const auto & witness : pair.second) {
                        const int rule_id = witness.first->id;
                        Val val = lhn_outside_weight * rule_weights[rule_id] / root_weight;
                        for (const auto & rhs_item : witness.second) {
                            val = val * tr_io_weight.first.at(*rhs_item);
                        }
                        rule_counts[rule_id] = rule_counts[rule_id] + val;
                    }
                }
            }

            // maximization
            for (auto group : normalization_groups) {
                Val group_count = Val::zero();
                for (auto member : group) {
                    group_count = group_count + rule_counts[member];
                }
                if (group_count != Val::zero()) {
                    for (auto member : group) {
                        rule_weights[member] = rule_counts[member] / group_count;
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

        std::vector<double> result;

        // conversion from log semiring:
        for (auto i = rule_weights.begin(); i != rule_weights.end(); ++i) {
            result.push_back(i->from());
        }


        return result;
    }


    GrammarInfo<unsigned> grammar_info_id(const std::vector<std::vector<unsigned>> &rule_to_nonterminals) {
        auto nont_idx_f = [](const unsigned nont) -> unsigned { return nont; };
        return grammar_info(rule_to_nonterminals, nont_idx_f);
    }

    template<typename NontIdx>
    GrammarInfo<Nonterminal> grammar_info(const std::vector<std::vector<unsigned>> &rule_to_nonterminals, NontIdx nont_idx_f) {
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
        return GrammarInfo<Nonterminal>(nont_idx_f, normalization_groups, rule_to_nonterminals);
    };


    template<typename Val>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> split_merge_id(
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const unsigned n_epochs, const unsigned n_nonts, const unsigned split_merge_cycles, const double merge_threshold, const double merge_percentage=-1.0
    ) {

        GrammarInfo<unsigned> grammarInfo = grammar_info_id(rule_to_nonterminals);
        const auto normalization_groups = grammarInfo.normalization_groups;
        const auto nont_idx_f = grammarInfo.nont_idx;



        std::cerr << "starting split merge training" << std::endl;
        std::cerr << "# nonts: " << n_nonts << std::endl;

        return split_merge<Val>(rule_weights, rule_to_nonterminals, normalization_groups, n_epochs, nont_idx_f,
                           split_merge_cycles, n_nonts, merge_threshold, merge_percentage);
    };

    template<typename Val>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> split_merge(
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const unsigned n_epochs, const std::map<Nonterminal, unsigned> &nont_idx, const unsigned split_merge_cycles, const double merge_threshold, const double merge_percentage=-1.0
    ) {
        auto nont_idx_f = [&](const Nonterminal nont) -> unsigned { return nont_idx.at(nont);};
        GrammarInfo<Nonterminal> grammarInfo = grammar_info(rule_to_nonterminals, nont_idx_f);
        const std::vector<std::vector<unsigned>> & normalization_groups = grammarInfo.normalization_groups;

        std::cerr << "starting split merge training" << std::endl;
        std::cerr << "# nonts: " << nont_idx.size() << std::endl;

        return split_merge<Val>(rule_weights, rule_to_nonterminals, normalization_groups, n_epochs, nont_idx_f,
                           split_merge_cycles, nont_idx.size(), merge_threshold, merge_percentage);
    };

    template<typename Val, typename NontToIdx>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> split_merge(
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const std::vector<std::vector<unsigned>> &normalization_groups, const unsigned n_epochs,
            const NontToIdx nont_idx, const unsigned split_merge_cycles, const unsigned n_nonts, const double merge_threshold, const double merge_percentage=-1.0
    ) {

        const double epsilon = 0.0001;


        // the next two structures hold split-dimensions and
        // rule weights for latent annotated rules before and after
        // each split/merge cycle
        std::vector<unsigned> nont_dimensions = std::vector<unsigned>(n_nonts, 1);
        std::vector<std::vector<Val>> rule_weights_la = init_weights_la<Val>(rule_weights);

        std::vector<Val> root_weights = {Val::one()};

        for (unsigned cycle = 0; cycle < split_merge_cycles; ++cycle) {
            split_merge_cycle(cycle, n_epochs, epsilon
                    , merge_threshold, merge_percentage, rule_to_nonterminals, normalization_groups, nont_idx, nont_dimensions
                    , rule_weights_la, root_weights);
        }

        std::vector<std::vector<double>> rule_weights_la_unlog = valToDouble(rule_weights_la);

        return std::make_pair(nont_dimensions, rule_weights_la_unlog);

    }

    template<typename Val>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> run_split_merge_cycle(
            const GrammarInfo<Nonterminal> & grammar_Info,
            std::vector<unsigned> nont_dimensions,
            const std::vector<std::vector<double>> &rule_weights_la,
            const unsigned n_epochs,
            const unsigned n_nonts, const double merge_threshold, const double merge_percentage,
            const unsigned cycle
    ) {
        const double epsilon = 0.0001;
        auto rule_weights_val = doubleToVal<Val>(rule_weights_la);
        std::vector<Val> root_weights = {Val::one()};
        split_merge_cycle(cycle, n_epochs, epsilon, merge_threshold, merge_percentage, grammar_Info.rule_to_nonterminals, grammar_Info.normalization_groups, grammar_Info.nont_idx, nont_dimensions, rule_weights_val, root_weights);
        auto rule_weights_la_after = valToDouble(rule_weights_val);

        return std::make_pair(nont_dimensions, rule_weights_la_after);
    };

    std::vector<std::vector<double>> lift_doubles(const std::vector<double> &rule_weights) const {
        std::vector<std::vector< double >> rule_weights_la;
        for (const double &rule_weight : rule_weights) {
            rule_weights_la.emplace_back(std::vector<double>(1, rule_weight));
        }
        return rule_weights_la;
    }

    template<typename Val>
    std::vector<std::vector<Val>> init_weights_la(const std::vector<double> &rule_weights) const {
        std::vector<std::vector< Val >> rule_weights_la;
        for (const double &rule_weight : rule_weights) {
            rule_weights_la.emplace_back(std::vector<Val>(1, Val::to(rule_weight)));
        }
        return rule_weights_la;
    }

    template<typename Val>
    std::vector<std::vector<Val>> doubleToVal(const std::vector<std::vector<double>> &rule_weights_double) const {
        std::vector<std::vector<Val>> rule_weights_val;
        for (const auto & weights : rule_weights_double) {
            rule_weights_val.push_back(std::vector<Val>());
            for (const double & weight : weights) {
                rule_weights_val.back().push_back(Val::to(weight));
            }
        }
        return rule_weights_val;
    }

    template<typename Val>
    std::vector<std::vector<double>> valToDouble(const std::vector<std::vector<Val>> &rule_weights_la) const {
        std::vector<std::vector<double>> rule_weights_la_unlog;
        for (const auto & weights : rule_weights_la) {
            rule_weights_la_unlog.push_back(std::vector<double>());
            for (const Val & weight : weights) {
                rule_weights_la_unlog.back().push_back(weight.from());
            }
        }
        return rule_weights_la_unlog;
    }

    template<typename Val, typename NontToIdx>
    void split_merge_cycle(const unsigned cycle, const unsigned n_epochs, double epsilon, double merge_threshold, double merge_percentage,
                           const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
                           const std::vector<std::vector<unsigned>> &normalization_groups, NontToIdx nont_idx,
                           std::vector<unsigned> & nont_dimensions, std::vector<std::vector<Val>> & rule_weights_la,
                           std::vector<Val> & root_weights) {

        std::vector<std::vector<Val>> rule_weights_splitted;
        std::vector<Val> root_weights_splitted;
        std::vector<std::vector<Val>> rule_weights_merged;

        std::vector<unsigned> split_dimensions;

        if (debug) std::cerr << "prepare split" << std::endl;

        for (const unsigned dim : nont_dimensions)
            split_dimensions.push_back(dim * 2);

        // splitting
        for (unsigned i = 0; i < rule_weights_la.size(); ++i) {
            const std::vector<Val> &rule_weight = rule_weights_la[i];
            std::vector<unsigned> dimensions;
            for (auto nont : rule_to_nonterminals[i]) {
                dimensions.push_back(split_dimensions[nont]);
            }
            const std::vector<Val> split_probabilities = split_rule(rule_weight, dimensions);
            rule_weights_splitted.push_back(split_probabilities);
        }

        const double root_split = rand_split();
        root_weights_splitted = {Val::to(root_split) * Val::one(), Val::to(1 - root_split) * Val::one()};

        rule_weights_la.clear();

        std::cerr << "em training after " << cycle + 1 << ". split" << std::endl;

        // em training
        do_em_training_la(rule_weights_splitted, root_weights_splitted, normalization_groups, n_epochs, split_dimensions,
                          rule_to_nonterminals, nont_idx);

        for (auto rule_weights_ : rule_weights_splitted) {
            for (auto rule_weight : rule_weights_) {
                if (rule_weight > (Val::one() + Val::to(epsilon))) {
                    std::cerr << "bad rule weight: " << rule_weight << std::endl;
                }
                assert(rule_weight <= (Val::one() + Val::to(epsilon)));
            }
        }

        // determine merges
        const auto merge_info = merge_prepare(rule_weights_splitted, root_weights_splitted, split_dimensions,
                                              rule_to_nonterminals, nont_idx, Val::to(merge_threshold), merge_percentage);


        // nonterminal -> new_la -> contributing old_las
        const std::vector<std::vector<std::vector<unsigned>>> & merge_selection = std::get<0>(merge_info);
        const std::vector<unsigned> & new_nont_dimensions = std::get<1>(merge_info);
        const std::vector<std::vector<Val>> merge_factors = std::get<2>(merge_info);

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
        for (unsigned i = 0; i < rule_weights_splitted.size(); ++i) {
            std::vector<unsigned> old_dimensions;
            std::vector<unsigned> new_dimensions;
            //new_dimensions.reserve(rule_to_nonterminals[i].size());
            std::vector<std::vector<std::vector<unsigned>>> merges;
            //merges.reserve(rule_to_nonterminals[i].size());
            const std::vector<Val> & lhn_merge_factors = merge_factors[rule_to_nonterminals[i][0]];
            for (auto nont : rule_to_nonterminals[i]) {
                old_dimensions.push_back(split_dimensions[nont]);
                new_dimensions.push_back(merge_selection[nont].size());
                merges.push_back(merge_selection[nont]);
            }
            rule_weights_merged.push_back(
                    merge_rule(rule_weights_splitted[i], old_dimensions, new_dimensions, merges,
                               lhn_merge_factors));
        }

        rule_weights_splitted.clear();

        // em training
        do_em_training_la(rule_weights_merged, root_weights, normalization_groups, n_epochs, new_nont_dimensions,
                          rule_to_nonterminals, nont_idx);

        for (auto rule_weights_ : rule_weights_merged) {
            for (auto rule_weight : rule_weights_) {
                if (rule_weight > Val::one() + Val::to(epsilon)) {
                    std::cerr << "bad rule weight: " << rule_weight << std::endl;
                }
                assert(rule_weight <= Val::one() + Val::to(epsilon));
            }
        }

        // create valid state after split/merge cycle
        nont_dimensions = new_nont_dimensions;
        rule_weights_la = rule_weights_merged;
    }

    template<typename Val, typename NontToIdx>
    std::pair<MAPTYPE<ParseItem < Nonterminal, Position>, std::vector<Val>>,
    MAPTYPE<ParseItem < Nonterminal, Position>, std::vector<Val>>>
    io_weights_la(const std::vector<std::vector<Val>> &rules, const std::vector<unsigned> &nont_dimensions,
                  const std::vector<std::vector<unsigned>> rule_id_to_nont_ids, const NontToIdx nont_idx,
                  const std::vector<Val> & root,
                  const unsigned i) const {


        // TODO implement for general case (== no topological order) approximation of inside weights
        assert (topological_orders.size() > i && topological_orders[i].size() > 0);

        const auto & topological_order = topological_orders[i];

        // computation of inside weights
        MAPTYPE<ParseItem<Nonterminal, Position>, std::vector<Val>> inside_weights;
        std::vector<unsigned> rule_dim;
        std::vector<std::vector<Val>> nont_vectors;
        for (const auto & item : topological_order) {
            inside_weights[item] = std::vector<Val>(nont_dimensions[nont_idx(item.nonterminal)], Val::zero());
            std::vector<Val> & inside_weight = inside_weights[item];
            for (const auto & witness : traces[i].at(item)) {
                // nont_vectors.reserve(witness.second.size());
                for (auto nont : rule_id_to_nont_ids[witness.first->id]) {
                    rule_dim.push_back(nont_dimensions[nont]);
                }
                for (const auto & dep_item : witness.second) {
                    nont_vectors.push_back(inside_weights.at(*dep_item));
                }
//                inside_weight = zipWith<Val>(std::plus<Val>(), inside_weight,
//                                        compute_inside_weights(rules[witness.first->id], nont_vectors,
//                                                               rule_dim));
                std::transform(
                          std::begin(inside_weight)
                        , std::end(inside_weight)
                        , std::begin(compute_inside_weights(rules[witness.first->id], nont_vectors, rule_dim))
                        , std::begin(inside_weight)
                        , std::plus<Val>()
                        );

                rule_dim.clear();
                nont_vectors.clear();
            }
        }

        // TODO implement for general case (== no topological order) solution by gauss jordan
        MAPTYPE<ParseItem<Nonterminal, Position>, std::vector<Val>> outside_weights;
        std::vector<std::vector<Val>> relevant_inside_weights;
        for (int j = topological_order.size() - 1; j >= 0; --j) {
            const ParseItem<Nonterminal, Position> & item = topological_order[j];
            outside_weights[item] = std::vector<Val>(nont_dimensions[nont_idx(item.nonterminal)], Val::zero());
            std::vector<Val> & outside_weight = outside_weights[item];

            if (item == goals[i])
//                outside_weight = zipWith<Val>(std::plus<Val>(), outside_weight, root);
                std::transform(
                          std::begin(outside_weight)
                        , std::end(outside_weight)
                        , std::begin(root)
                        , std::begin(outside_weight)
                        , std::plus<Val>());

            if (traces_reverse[i].count(item)) {
                for (const auto &witness : traces_reverse[i].at(item)) {
                    const auto &rule = *(std::get<0>(witness));
                    const auto &siblings = *(std::get<1>(witness));
                    const auto &parent = *(std::get<2>(witness));
                    const unsigned position = std::get<3>(witness);

                    rule_dim.push_back(nont_dimensions[nont_idx(parent.nonterminal)]);

                    for (unsigned sib_position = 0; sib_position < siblings.size(); ++sib_position) {
                        const auto & rhs_item = siblings[sib_position];

                        rule_dim.push_back(nont_dimensions[nont_idx(rhs_item->nonterminal)]);
                        if (sib_position == position)
                            relevant_inside_weights.push_back(std::vector<Val>(rule_dim.back(), Val::one()));
                        else
                            relevant_inside_weights.push_back(inside_weights[*rhs_item]);
                    }

                    const std::vector<Val> new_weights
                            = compute_outside_weights(
                                    rules[rule.id]
                                    , outside_weights[parent]
                                    , relevant_inside_weights
                                    , rule_dim
                                    , position);

                    std::transform(outside_weight.begin(), outside_weight.end(), new_weights.begin(), outside_weight.begin(), std::plus<Val>());
                    //                  outside_weight = zipWith<Val>( std::plus<Val>()
                    //                               , outside_weight
                    //                               , new_weights);
                    rule_dim.clear();
                    relevant_inside_weights.clear();
                }
            }
        }

        return std::make_pair(inside_weights, outside_weights);
    }


    template <typename NontToIdx, typename Val>
    void do_em_training_la(
            std::vector<std::vector<Val>> & rule_weights
            , std::vector<Val> & the_root_weights
            , const std::vector<std::vector<unsigned>> & normalization_groups
            , const unsigned n_epochs
            , const std::vector<unsigned> & nont_dimensions
            , const std::vector<std::vector<unsigned>> & rule_to_nont_ids
            , const NontToIdx nont_idx
//            , const Val zero, const Val one, const Val leaf, const Accum1 prod, const Accum2 sum, const Accum3 division
    ){

        unsigned epoch = 0;

        std::cerr <<"Epoch " << epoch << "/" << n_epochs << ": ";
        if (debug) {
            for (unsigned i = 0; i < rule_weights.size(); ++i) {
                std::cerr << i << " { ";
                for (Val elem : rule_weights[i])
                    std::cerr << elem.from() << " ";
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
                std::cerr << weight.from();
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
                rule_counts.push_back(std::vector<Val>(calc_size(dims), Val::zero()));
            }

            std::vector<Val> root_counts = std::vector<Val>(nont_dimensions[nont_idx(goals[0].nonterminal)], Val::zero());
            Val corpus_likelihood = Val::one();

            for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
                auto trace = traces[trace_id];
                if (trace.size() == 0)
                    continue;

                const auto tr_io_weight = io_weights_la(rule_weights, nont_dimensions, rule_to_nont_ids, nont_idx, the_root_weights, trace_id);
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

                const auto trace_root_weights = zipWith<Val>(std::multiplies<Val>(), tr_io_weight.first.at(goals[trace_id]),
                                                       tr_io_weight.second.at(goals[trace_id]));
//                root_counts = zipWith<Val>(std::plus<Val>(), root_counts, trace_root_weights);
                std::transform(
                          std::begin(root_counts)
                        , std::end(root_counts)
                        , std::begin(trace_root_weights)
                        , std::begin(root_counts)
                        , std::plus<Val>());
                corpus_likelihood *= (reduce(std::plus<Val>(), trace_root_weights, Val::zero()));

                const auto instance_root_weights = tr_io_weight.first.at(goals[trace_id]);
                const Val instance_root_weight = reduce(std::plus<Val>(),
                                                        zipWith<Val>(std::multiplies<Val>(), instance_root_weights,
                                                                the_root_weights), Val::zero());
                // const auto tmpxyz = zipWith<Val>(std::multiplies<Val>(), instance_root_weights, the_root_weights);
                // const Val instance_root_weight = std::accumulate(tmpxyz.begin(), tmpxyz.end(), Val::zero(), std::plus<Val>());
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
                        rule_val = compute_rule_frequency(rule_weights[rule_id], lhn_outside_weights, nont_weight_vectors, rule_dimensions[rule_id]);
//                        rule_val = zipWithConstant<Val>(std::divides<Val>(), rule_val, instance_root_weight);
                        std::for_each(rule_val.begin(), rule_val.end(), [&](Val& x) {x /= instance_root_weight;});

                        if (debug) {
                            std::cerr << rule_id << ": {";
                            for (auto val : rule_val)
                                std::cerr << val << " ";
                            std::cerr << " }" << std::endl;
                        }
//                        rule_counts[rule_id] = zipWith<Val>(std::plus<Val>(), rule_counts[rule_id], rule_val);
                        std::transform(
                                  std::begin(rule_counts[rule_id])
                                , std::end(rule_counts[rule_id])
                                , std::begin(rule_val)
                                , std::begin(rule_counts[rule_id])
                                , std::plus<Val>()
                                );
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
                std::vector<Val> group_counts = std::vector<Val>(group_dim, Val::zero());
                for (auto member : group) {
                    const unsigned block_size = subdim(rule_dimensions[member]);
                    // reduce([] (const unsigned x, const unsigned y) -> unsigned {return x * y;}, rule_dimensions[member], (unsigned) 1, (unsigned) 1);
                    for (unsigned dim : boost::irange((unsigned) 0, group_dim)) {
                        const typename std::vector<Val>::iterator block_start = rule_counts[member].begin() + block_size * dim;
                        for (auto it = block_start; it != block_start + block_size; ++it) {
                            group_counts[dim] += (*it);
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
                        if (group_counts[dim] > Val::zero()) {
                            const unsigned block_start = block_size * dim;
                            for (unsigned offset = block_start; offset < block_start + block_size; ++offset) {
                                *(rule_weights[member].begin() + offset) = (*(rule_counts[member].begin() + offset)) / group_counts[dim];
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
                    for (Val elem : rule_weights[i])
                        std::cerr << elem.from() << " ";
                    std::cerr << " } , ";
                }
            if (debug) std::cerr << std::endl;

            // maximize root weights:
            const Val corpus_prob_sum = std::accumulate(root_counts.begin(), root_counts.end(), Val::zero(), std::plus<Val>());
            std::cerr << "corpus prob. sum " << corpus_prob_sum.from();
            std::cerr << " corpus likelihood " << corpus_likelihood;
            for (auto root_weight : the_root_weights) {
                std::cerr << " " << root_weight << " ";
            }
            std::cerr << std::endl;
            const unsigned len = the_root_weights.size();
            the_root_weights.clear();
            if (debug) std::cerr << "single root weights: ";
            for (const auto weight : root_counts) {
                if (debug) std::cerr << weight.from() << "/" << (weight / corpus_prob_sum).from() << " ";
                the_root_weights.push_back(weight / corpus_prob_sum);
            }
            assert(len == the_root_weights.size());
            std::cerr << std::endl;

            if (debug) {
                std::cerr << "Nont sums: ";
                unsigned i = 0;
                for (auto group : normalization_groups) {
                    std::vector<Val> dim_weights = std::vector<Val>(nont_dimensions[i], Val::zero());
                    for (auto rule_id : group) {
                        const unsigned size = rule_weights[rule_id].size();
                        for (unsigned weight_id = 0; weight_id < size; ++weight_id) {
                            const unsigned index = weight_id / (size / nont_dimensions[i]);
                            dim_weights[index] = dim_weights[index] + rule_weights[rule_id][weight_id];
                        }
                    }
                    std::cerr << " { ";
                    for (auto weight_sum : dim_weights) {
                        std::cerr << weight_sum.from() << " ";
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
                    std::cerr << weight.from();
                }
                std::cerr << " } " << std::endl;
            }

            rule_counts.clear();
            root_counts.clear();
        }

    }

    template <typename Val, typename NontToIdx>
    std::tuple< std::vector<std::vector<std::vector<unsigned>>>
              , std::vector<unsigned>
              , std::vector<std::vector<Val>>
              >
    merge_prepare(const std::vector<std::vector<Val>> & rule_weights
            , const std::vector<Val> & root_weights
            , const std::vector<unsigned> & nont_dimensions
            , const std::vector<std::vector<unsigned>> & rule_ids_to_nont_ids
                       , const NontToIdx nont_idx
                       , const Val merge_threshold_
                       , const double merge_percent = -1.0
            ) {


        // first we compute the fractions p_1, p_2
        // with which the probabality mass is shared between merged latent states

        // this is prepared with computing globally averaged outside weights
        std::vector<std::vector<Val>> merge_weights_partial;
        for (auto dim : nont_dimensions) {
            merge_weights_partial.emplace_back(std::vector<Val>(dim, Val::zero()));
        }

        std::cerr << "Estimating relative frequency of annotated nonterminals." << std::endl;
        // computing in(A_x) * out(A_x) for every A ∈ N and x ∈ X_A
        for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
            const auto io_weight = io_weights_la(rule_weights, nont_dimensions, rule_ids_to_nont_ids, nont_idx, root_weights, trace_id);

            for (const auto & pair : traces[trace_id]) {
                const ParseItem<Nonterminal, Position> & item = pair.first;

                const std::vector<Val> & inside_weight = io_weight.second.at(item);
                const std::vector<Val> & outside_weight = io_weight.second.at(item);

                const auto vals = zipWith<Val>(std::multiplies<Val>(), inside_weight, outside_weight);
                // denominator cancels out later
                // const auto denominator = reduce<Val>(std::plus<Val>(), vals, Val::zero());
                auto & target =  merge_weights_partial[nont_idx(item.nonterminal)];

                // denominator cancels out later
                //target = zipWith<Val>(std::plus<Val>(), target, zipWithConstant<Val>(std::divides<Val>(), vals, denominator));
//                target = zipWith<Val>(std::plus<Val>(), target, vals);
                std::transform(std::begin(target), std::end(target), std::begin(vals), std::begin(target), std::plus<Val>());
            }

        }

        std::cerr << "Computing merge factors." << std::endl;
        // finally we compute the fractions
        std::vector<std::vector<Val>> p;
        for (auto las_weights : merge_weights_partial) {
            p.emplace_back(std::vector<Val>());
            for (unsigned i = 0; i < las_weights.size(); i = i + 2) {
                const Val combined_weight = las_weights[i] + las_weights[i+1];
                if (combined_weight != Val::zero()) {
                    p.back().push_back(las_weights[i] / combined_weight);
                    p.back().push_back(las_weights[i + 1] / combined_weight);
                } else {
                    p.back().push_back(Val::to(0.5));
                    p.back().push_back(Val::to(0.5));
                }
            }
        }

        std::cerr << "Computing likelihood deltas of merges." << std::endl;
        // now we approximate the likelihood Δ of merging two latent states
        std::vector<std::vector<Val>> merge_delta;
        for (auto dim : nont_dimensions) {
            merge_delta.push_back(std::vector<Val>(dim / 2, Val::one()));
        }

        std::vector<Val> prefixes;
        std::vector<Val> postfixes;
        for (unsigned trace_id = 0; trace_id < traces.size(); ++trace_id) {
            const auto io_weight = io_weights_la(rule_weights, nont_dimensions, rule_ids_to_nont_ids, nont_idx, root_weights, trace_id);
            for (const auto & pair : traces[trace_id]) {
                const ParseItem<Nonterminal, Position> &item = pair.first;

                // compute Q( item )
//                Val denominator = Val::zero();
//                for (unsigned dim : boost::irange((unsigned) 0, nont_dimensions[nont_idx(item.nonterminal)])) {
//                    const Val in = io_weight.first.at(item)[dim];
//                    const Val out = io_weight.second.at(item)[dim];
//                    denominator += (in * out);
//                    assert(! isnan(denominator.get_Value()));
//                }
                const auto nont_dim = nont_dimensions[nont_idx(item.nonterminal)];
                prefixes.resize(nont_dim / 2, Val::zero());
                postfixes.resize(nont_dim / 2, Val::zero());
                Val denominator = Val::zero();
                {
                    const unsigned dim = nont_dim - 2;
                    const Val in1 = io_weight.first.at(item)[dim];
                    const Val in2 = io_weight.first.at(item)[dim + 1];
                    const Val out1 = io_weight.second.at(item)[dim];
                    const Val out2 = io_weight.second.at(item)[dim + 1];
                    denominator += in1 * out1 + in2 * out2;
                }
                for (unsigned dim = 0; dim < nont_dim - 2; dim = dim + 2) {
                    {
                        const Val in1 = io_weight.first.at(item)[dim];
                        const Val in2 = io_weight.first.at(item)[dim + 1];
                        const Val out1 = io_weight.second.at(item)[dim];
                        const Val out2 = io_weight.second.at(item)[dim + 1];
                        prefixes[dim / 2 + 1] = prefixes[dim / 2] + in1 * out1 + in2 * out2;
                        denominator += in1 * out1 + in2 * out2;
                    }
                    {
                        const unsigned dim_ = nont_dim - dim - 2;
                        const Val in1 = io_weight.first.at(item)[nont_dim - dim_];
                        const Val in2 = io_weight.first.at(item)[nont_dim - dim_ + 1];
                        const Val out1 = io_weight.second.at(item)[nont_dim - dim_];
                        const Val out2 = io_weight.second.at(item)[nont_dim - dim_ + 1];
                        postfixes[(nont_dim - dim) / 2 - 2] = postfixes[(nont_dim - dim)/ 2] + in1 * out1 + in2 * out2;
                    }
                }

                // in of some item can be zero in certain LA-dimensions
                // since LA-rule weights may converge to zero
                // we ignore those dimensions in Δ computation
                if (denominator == Val::zero())
                    continue;

                for (unsigned dim = 0; dim < nont_dimensions[nont_idx(item.nonterminal)]; dim = dim+2) {
                    const Val in1 = io_weight.first.at(item)[dim];
                    const Val in2 = io_weight.first.at(item)[dim+1];
                    const Val out1 = io_weight.second.at(item)[dim];
                    const Val out2 = io_weight.second.at(item)[dim+1];
                    const unsigned nont = nont_idx(item.nonterminal);
                    const Val p1 = p[nont][dim];
                    const Val p2 = p[nont][dim+1];

                    const Val out_merged = out1 + out2;
                    const Val in_merged = (p1 * in1) + (p2 * in2);

//                    const Val Q = Val::add_subtract2_divide(denominator, in_merged * out_merged, in1 * out1, in2 * out2, denominator);
                    const Val Q = (prefixes[dim / 2] + postfixes[dim / 2] + in_merged * out_merged) / denominator;

                    if (isnan(Q.get_Value())) {
                        std::cerr << "bad fraction " << Q << " where" << std::endl;
                        std::cerr << "prefix  " << prefixes[dim/2] << std::endl;
                        std::cerr << "postfix " << postfixes[dim/2] << std::endl;
                        std::cerr << "merged  " << in_merged * out_merged << std::endl;
                        std::cerr << "denom   " << denominator << std::endl;

//                        Val nominator = denominator;
//                        nominator = nominator + (in_merged * out_merged);
//                        nominator = nominator - (in1 * out1);
//                        nominator = nominator - (in2 * out2);
//                        // const Val Q2 = nominator / denominator;
//                        std::cerr << "bad fraction " << nominator << " / " << denominator << " = " << Q << std::endl;
//                        std::cerr << "prod(in_merged, out_merged) = " << in_merged * out_merged << std::endl;
//                        std::cerr << "prod(in1, out1) = " << in1 * out1 << std::endl;
//                        std::cerr << "prod(in2, out2) = " << in2 * out2 << std::endl;
                        assert(!isnan(Q.get_Value()));
                    }

                    Val & delta = merge_delta[nont][dim / 2];

                    delta *= Q;
                }
                prefixes.clear();
                postfixes.clear();
            }
        }

        const bool merge_perc = merge_percent >= 0.0 && merge_percent <= 100.0;

        std::cerr << "Selecting merges ";
        if (merge_perc)
            std::cerr << "best " << merge_percent << " % " ;
        else
            std::cerr << "above threshold " << merge_threshold_;
        std::cerr << std::endl;
        std::vector<Val> ordered_merge_weights;
        Val threshold = Val::zero();
        if (merge_perc) {
            // order merges according to likelihood_loss
            for (const auto & delta : merge_delta) {
                ordered_merge_weights.insert(std::end(ordered_merge_weights), std::begin(delta), std::end(delta));
            }
            std::sort(std::begin(ordered_merge_weights), std::end(ordered_merge_weights), std::greater<Val>());
            unsigned index = (unsigned) merge_percent / 100.0 * ordered_merge_weights.size();
            if (index > ordered_merge_weights.size())
                index = ordered_merge_weights.size() - 1;

            if (true || debug) std::cerr << "index for ordered merges " << index << " / " << ordered_merge_weights.size() << std::endl;

            threshold = ordered_merge_weights[index];
        }

        const Val merge_threshold = ! merge_perc ? merge_threshold_ : threshold;
        // evaluate Δ and build merge table accordingly
        std::vector<std::vector<std::vector<unsigned>>> merge_selection;
        std::vector<unsigned> new_nont_dimensions;
        unsigned nont = 0;
        unsigned merges = 0;
        unsigned splits = 0;

        if (debug) std::cerr << "merge deltas: ";
        for (const auto & delta : merge_delta) {
            if (debug) std::cerr << " { ";
            merge_selection.push_back(std::vector<std::vector<unsigned>>());
            for (unsigned dim = 0; dim < nont_dimensions[nont] / 2; ++dim) {
                if (debug) std::cerr << delta[dim].from() << " ";
                if (delta[dim] >= merge_threshold - Val::to(0.00001)
                    // always merge if Δ >= 1
                    || delta[dim] >= Val::one() - Val::to(0.00001)
                    // always merge initial symbol
                    || nont_idx(goals[0].nonterminal) == nont) {
                    merge_selection.back().push_back(std::vector<unsigned>());
                    merge_selection.back().back().push_back(dim * 2);
                    merge_selection.back().back().push_back(dim * 2 + 1);
                    ++merges;
                } else {
                    merge_selection.back().push_back(std::vector<unsigned>(1, dim * 2 ));
                    merge_selection.back().push_back(std::vector<unsigned>(1, dim * 2 + 1));
                    ++splits;
                }
            }
            if (debug) std::cerr << " } ";
            ++nont;
            new_nont_dimensions.push_back(merge_selection.back().size());
        }
        if (debug) std::cerr << std::endl;

        std::cerr << "Merging " << merges << " of " << merges + splits << " splits. Merge threshold is " << merge_threshold << std::endl;

        return std::make_tuple(merge_selection, new_nont_dimensions, p);
    }

};


#endif //STERMPARSER_TRACE_H
