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
#include <functional>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "EigenUtil.h"
#include <fenv.h>
#include <algorithm>
#include <thread>
#include <mutex>

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

typedef Eigen::TensorMap<Eigen::Tensor<double, 1>> WeightVector;

template <typename Nonterminal, typename Terminal, typename Position>
class TraceManager {
private:
    double * start = nullptr;
    double * next = nullptr;
    double * max_mem = nullptr;
    unsigned the_size = 0; // 625000; // 5MB

    bool reserve_memory(unsigned size) {
        std::cerr << "reserving " << size << std::endl;
        if (self_malloc) {
            if (start == next) {
                if (start != nullptr and max_mem - start < size) {
                    free(start);
                }
                if (start == nullptr or max_mem - start < size) {
                    unsigned allocate = the_size > size ? the_size : size;
                    std::cerr << "allocating " << allocate << std::endl;
                    start = (double *) malloc(sizeof(double) * allocate);
                    max_mem = start + allocate;
                    next = start;
                }
                return true;
            } else
                return false;

        } else
            return true;
    }

    double * get_region(unsigned size) {
        if (not self_malloc)
            return (double*) malloc(sizeof(double) * size);
        else {
            if (start == nullptr) {
                start = (double *) malloc(sizeof(double) * the_size);
                max_mem = start + the_size;
                next = start;
            }
            if (max_mem - next < size) {
                std::cerr << "Maximum size of double storage exceeded" << std::endl;
                std::cerr << "Required  " << size << std::endl;
                std::cerr << "Available " << (max_mem - next) / sizeof(double) << std::endl;
                abort();
            }
            double *return_ = next;
            next = return_ + size;
            return return_;
        }
    }

    bool free_region(double* const ptr, const unsigned size) {
        if (not self_malloc) {
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
    const bool self_malloc = true;

    MAPTYPE<Nonterminal, unsigned> max_items_with_nonterminal;
    std::vector<MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector>> traces_inside_weights, traces_outside_weights;

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

        MAPTYPE<Nonterminal, unsigned> number_of_items;
        for (auto item : topological_order) {
            if (number_of_items.count(item.nonterminal))
                ++number_of_items[item.nonterminal];
            else
                number_of_items[item.nonterminal] = 1;
        }
        for (auto pair : number_of_items) {
            if (max_items_with_nonterminal.count(pair.first))
                // changed from max to plus
                max_items_with_nonterminal[pair.first] = max_items_with_nonterminal.at(pair.first) + pair.second;
            else
                max_items_with_nonterminal[pair.first] = pair.second;
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
            const unsigned N_THREADS, const unsigned BATCH_SIZE,
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const unsigned n_epochs, const unsigned n_nonts, const unsigned split_merge_cycles, const double merge_threshold, const double merge_percentage=-1.0
    ) {

        GrammarInfo<unsigned> grammarInfo = grammar_info_id(rule_to_nonterminals);
        const auto normalization_groups = grammarInfo.normalization_groups;
        const auto nont_idx_f = grammarInfo.nont_idx;



        std::cerr << "starting split merge training" << std::endl;
        std::cerr << "# nonts: " << n_nonts << std::endl;

        return split_merge<Val>(N_THREADS, BATCH_SIZE, rule_weights, rule_to_nonterminals, normalization_groups, n_epochs, nont_idx_f,
                           split_merge_cycles, n_nonts, merge_threshold, merge_percentage);
    };

    template<typename Val>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> split_merge(
            const unsigned N_THREADS, const unsigned BATCH_SIZE,
            const std::vector<double> &rule_weights, const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
            const unsigned n_epochs, const std::map<Nonterminal, unsigned> &nont_idx, const unsigned split_merge_cycles, const double merge_threshold, const double merge_percentage=-1.0
    ) {
        auto nont_idx_f = [&](const Nonterminal nont) -> unsigned { return nont_idx.at(nont);};
        GrammarInfo<Nonterminal> grammarInfo = grammar_info(rule_to_nonterminals, nont_idx_f);
        const std::vector<std::vector<unsigned>> & normalization_groups = grammarInfo.normalization_groups;

        std::cerr << "starting split merge training" << std::endl;
        std::cerr << "# nonts: " << nont_idx.size() << std::endl;

        return split_merge<Val>(N_THREADS, BATCH_SIZE, rule_weights, rule_to_nonterminals, normalization_groups,
                                n_epochs, nont_idx_f,
                           split_merge_cycles, nont_idx.size(), merge_threshold, merge_percentage);
    };

    template<typename Val, typename NontToIdx>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> split_merge(
            const unsigned N_THREADS, const unsigned BATCH_SIZE,
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
            split_merge_cycle(N_THREADS, BATCH_SIZE, cycle, n_epochs, epsilon
                    , merge_threshold, merge_percentage, rule_to_nonterminals, normalization_groups, nont_idx, nont_dimensions
                    , rule_weights_la, root_weights);
        }

        std::vector<std::vector<double>> rule_weights_la_unlog = valToDouble(rule_weights_la);

        return std::make_pair(nont_dimensions, rule_weights_la_unlog);

    }

    template<typename Val>
    std::pair<std::vector<unsigned>, std::vector<std::vector<double>>> run_split_merge_cycle(
            const unsigned N_THREADS, const unsigned BATCH_SIZE,
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
        split_merge_cycle(N_THREADS, BATCH_SIZE, cycle, n_epochs, epsilon, merge_threshold, merge_percentage, grammar_Info.rule_to_nonterminals, grammar_Info.normalization_groups, grammar_Info.nont_idx, nont_dimensions, rule_weights_val, root_weights);
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

    template<typename Val>
    unsigned total_rule_sizes(std::vector<std::vector<Val>> & rule_weights) const {
        unsigned counter = 0;
        for(const auto & rule_weight : rule_weights) {
            counter += rule_weight.size();
        }
        return counter;
    }

    template<typename NontIdx>
    unsigned max_item_size(const std::vector<unsigned> & nont_dimensions, const NontIdx nont_idx) const {
        unsigned counter = 0;
        for (const auto & pair : max_items_with_nonterminal) {
            counter += pair.second * nont_dimensions.at(nont_idx(pair.first));
        }
        return counter;
    }

    template <typename Val>
    unsigned convert_to_eigen(std::vector<double*> & rule_weights_ptrs, const std::vector<std::vector<Val>> & rule_weights, std::vector<RuleTensor<double>> & rule_tensors,
    double* & root_weights_ptrs, const std::vector<Val> & root_weights, const std::vector<std::vector<unsigned>> & rule_dimensions) {
        unsigned allocated(0);
        unsigned rule = 0;
        for(auto rule_weight : rule_weights) {
            const std::vector<unsigned> & rule_dim = rule_dimensions[rule];
            double * rule_weight_ptr = get_region(rule_weight.size());
            const unsigned dims = rule_dim.size();

            switch (dims) {
                case 1:
                    convert_format<1>(rule_weight_ptr, rule_dim, rule_weight, rule_tensors);
                    break;
                case 2:
                    convert_format<2>(rule_weight_ptr, rule_dim, rule_weight, rule_tensors);
                    break;
                case 3:
                    convert_format<3>(rule_weight_ptr, rule_dim, rule_weight, rule_tensors);
                    break;
                case 4:
                    convert_format<4>(rule_weight_ptr, rule_dim, rule_weight, rule_tensors);
                    break;
                default:
                    assert(false && "Rules with more than 3 RHS nonterminals are not implemented.");
                    abort();
            }
            rule_weights_ptrs.push_back(rule_weight_ptr);
            allocated += rule_weight.size();
            ++rule;
        }
        root_weights_ptrs = get_region(root_weights.size());
        allocated += root_weights.size();
        for (unsigned i = 0; i < root_weights.size(); ++i) {
            root_weights_ptrs[i] = root_weights[i].from();
        }
        return allocated;
    }

    template<typename Val>
    void convert_from_eigen(const std::vector<double*> rule_weight_ptr, std::vector<std::vector<Val>> & rule_weights,
    const double* root_weight_ptr, std::vector<Val> & root_weights, const std::vector<std::vector<unsigned>> & rule_dimensions, const unsigned allocated) {


        for(unsigned rule = 0; rule < rule_weights.size(); ++rule) {
            auto & rule_weight = rule_weights[rule];
            const std::vector<unsigned> & rule_dim = rule_dimensions[rule];
            const unsigned dims = rule_dim.size();

            switch (dims) {
                case 1:
                    de_convert_format<1>(rule_weight_ptr[rule], rule_dim, rule_weight);
                    break;
                case 2:
                    de_convert_format<2>(rule_weight_ptr[rule], rule_dim, rule_weight);
                    break;
                case 3:
                    de_convert_format<3>(rule_weight_ptr[rule], rule_dim, rule_weight);
                    break;
                case 4:
                    de_convert_format<4>(rule_weight_ptr[rule], rule_dim, rule_weight);
                    break;
                default:
                    assert(false && "Rules with more than 3 RHS nonterminals are not implemented.");
                    abort();
            }
        }
        for (unsigned i = 0; i < root_weights.size(); ++i) {
            root_weights[i] = Val::to(root_weight_ptr[i]);
        }
        if (self_malloc) {
            if (not free_region(rule_weight_ptr[0], allocated))
                abort();
        } else {
            for (auto ptr : rule_weight_ptr) {
                free_region(ptr, 0);
            }
        }
    }


    template<typename Val, typename NontToIdx>
    void split_merge_cycle(const unsigned N_THREADS, const unsigned BATCH_SIZE, const unsigned cycle,
                           const unsigned n_epochs, double epsilon, double merge_threshold, double merge_percentage,
                           const std::vector<std::vector<unsigned>> &rule_to_nonterminals,
                           const std::vector<std::vector<unsigned>> &normalization_groups, NontToIdx nont_idx,
                           std::vector<unsigned> & nont_dimensions, std::vector<std::vector<Val>> & rule_weights_la,
                           std::vector<Val> & root_weights) {

        std::vector<std::vector<Val>> rule_weights_splitted;
        std::vector<Val> root_weights_splitted;
        std::vector<std::vector<Val>> rule_weights_merged;
        std::vector<std::vector<unsigned>> rule_dimensions_splitted;

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
            rule_weights_splitted.emplace_back(std::move(split_probabilities));
            if (debug) {
                std::cerr << std::endl << "Rule prob " << i << " : { ";
                for (const auto val : rule_weight) std::cerr << val << " ";
                std::cerr << " } " << std::endl << "after split: { ";
                for (const auto val : rule_weights_splitted.back()) std::cerr << val << " ";
                std::cerr << " } " << std::endl << std::endl;
            }
            rule_dimensions_splitted.emplace_back(std::move(dimensions));
        }

        const double root_split = rand_split() * 0.5;
        root_weights_splitted = {Val::to(root_split) * Val::one(), Val::to(1 - root_split) * Val::one()};

        rule_weights_la.clear();

        std::cerr << "em training after " << cycle + 1 << ". split" << std::endl;

        // em training

        // memory allocation as prerequisite for EM training
        unsigned required_memory = total_rule_sizes(rule_weights_splitted) * (1 + N_THREADS); // *2 for rule weights and rule counts
        required_memory += max_item_size(split_dimensions, nont_idx) * 2; // *2 for inside and outside weight

        std::cerr << " Tot rule size: " << total_rule_sizes(rule_weights_splitted) << " max item size " << max_item_size(split_dimensions, nont_idx) << std::endl;

        required_memory += 2 * 2; // for root weights and counts
        if (not reserve_memory(required_memory)) {
            std::cerr << "Could not reserve required memory." << std::endl;
            abort();
        }

        const auto allocated_io_weights = allocate_io_weight_maps(nont_idx, split_dimensions);

        std::vector<double *> rule_weights_ptrs;
        std::vector<RuleTensor<double>> rule_weight_tensors;
        double * root_weights_ptrs;
        unsigned allocated = convert_to_eigen(rule_weights_ptrs, rule_weights_splitted, rule_weight_tensors, root_weights_ptrs, root_weights_splitted, rule_dimensions_splitted);

        // do actual EM training
        do_em_training_la(N_THREADS, BATCH_SIZE, rule_weight_tensors, rule_weights_ptrs, root_weights_ptrs, normalization_groups, n_epochs, split_dimensions,
                          rule_to_nonterminals, nont_idx);

        convert_from_eigen(rule_weights_ptrs, rule_weights_splitted, root_weights_ptrs, root_weights_splitted, rule_dimensions_splitted, allocated);
        rule_weight_tensors.clear();

        for (auto rule_weights_ : rule_weights_splitted) {
            for (auto rule_weight : rule_weights_) {
                if (rule_weight > (Val::one() + Val::to(epsilon))) {
                    std::cerr << "bad rule weight: " << rule_weight << std::endl;
                }
                assert(rule_weight <= (Val::one() + Val::to(epsilon)));
            }
        }

        // determine merges
        const auto merge_info = merge_prepare(N_THREADS, BATCH_SIZE, rule_weights_splitted, root_weights_splitted, split_dimensions,
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

        std::vector<std::vector<unsigned>> rule_dimensions_merged;

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
            rule_dimensions_merged.emplace_back(std::move(new_dimensions));
        }

        rule_weights_splitted.clear();

        // em training
//        shrink_io_weight_maps(nont_idx, new_nont_dimensions);
        free_io_weight_maps(allocated_io_weights.first, allocated_io_weights.second);
        const auto allocated_io_weights2 = allocate_io_weight_maps(nont_idx, new_nont_dimensions);

        // conversion
        std::vector<double *> rule_weights_merged_ptrs;
        double * root_weights_merged_ptrs;
        allocated = convert_to_eigen(rule_weights_merged_ptrs, rule_weights_merged, rule_weight_tensors, root_weights_merged_ptrs, root_weights, rule_dimensions_merged);

        do_em_training_la(N_THREADS, BATCH_SIZE, rule_weight_tensors, rule_weights_merged_ptrs, root_weights_merged_ptrs, normalization_groups, n_epochs, new_nont_dimensions,
                          rule_to_nonterminals, nont_idx);

        convert_from_eigen(rule_weights_merged_ptrs, rule_weights_merged, root_weights_merged_ptrs, root_weights, rule_dimensions_merged, allocated);
        rule_weight_tensors.clear();


        for (auto rule_weights_ : rule_weights_merged) {
            for (auto rule_weight : rule_weights_) {
                if (rule_weight > Val::one() + Val::to(epsilon)) {
                    std::cerr << "bad rule weight: " << rule_weight << std::endl;
                }
                assert(rule_weight <= Val::one() + Val::to(epsilon));
            }
        }

        free_io_weight_maps(allocated_io_weights2.first, allocated_io_weights2.second);

        // create valid state after split/merge cycle
        nont_dimensions = new_nont_dimensions;
        rule_weights_la = rule_weights_merged;
    }

    template<typename NontToIdx>
    std::pair<double*, unsigned> allocate_io_weight_maps(const NontToIdx nont_idx, const std::vector<unsigned> & nont_dimensions) {
        double * start(nullptr);
        unsigned allocated(0);

        for (unsigned i = 0; i < traces_size(); ++i) {
            MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> inside_weights, outside_weights;

            const auto &topological_order = topological_orders[i];
            for (const auto &item : topological_order) {
                const unsigned item_dimension = nont_dimensions[nont_idx(item.nonterminal)];
                double *const inside_weight_ptr = get_region(item_dimension);
                double *const outside_weight_ptr = get_region(item_dimension);
                if (start == nullptr)
                    start = inside_weight_ptr;
                allocated += item_dimension * 2;

                Eigen::TensorMap<Eigen::Tensor<double, 1>> inside_weight(inside_weight_ptr, item_dimension);
                Eigen::TensorMap<Eigen::Tensor<double, 1>> outside_weight(outside_weight_ptr, item_dimension);
                inside_weights.emplace(std::make_pair(item, std::move(inside_weight)));
                outside_weights.emplace(std::make_pair(item, std::move(outside_weight)));
            }
            if (traces_inside_weights.size() != i or traces_outside_weights.size() != i)
                abort();
            traces_inside_weights.emplace_back(std::move(inside_weights));
            traces_outside_weights.emplace_back(std::move(outside_weights));
        }
        return std::make_pair(start, allocated);
    }

    template<typename NontToIdx>
    void shrink_io_weight_maps(const NontToIdx nont_idx, const std::vector<unsigned> & nont_dimensions) {
        for (unsigned i = 0; i < traces_size(); ++i) {
            MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> & inside_weights = traces_inside_weights[i];
            MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> & outside_weights = traces_outside_weights[i];
            MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> inside_weights_new, outside_weights_new;
            const auto &topological_order = topological_orders[i];
            for (const auto &item : topological_order) {
                const unsigned item_dimension = nont_dimensions[nont_idx(item.nonterminal)];
                double *const inside_weight_ptr = inside_weights.at(item).data();
                double *const outside_weight_ptr = outside_weights.at(item).data();

                Eigen::TensorMap<Eigen::Tensor<double, 1>> inside_weight(inside_weight_ptr, item_dimension);
                Eigen::TensorMap<Eigen::Tensor<double, 1>> outside_weight(outside_weight_ptr, item_dimension);
                inside_weights_new.emplace(std::make_pair(item, std::move(inside_weight)));
                outside_weights_new.emplace(std::make_pair(item, std::move(outside_weight)));
            }
            traces_inside_weights[i] = inside_weights_new;
            traces_outside_weights[i] = outside_weights_new;
        }
    }

    void free_io_weight_maps(double * const start, unsigned allocated) {
        if (self_malloc) {
            if (not free_region(start, allocated))
                abort();
            traces_inside_weights.clear();
            traces_outside_weights.clear();
        } else {
            for (unsigned i = 0; i < traces_size(); ++i) {
                MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> & inside_weights = traces_inside_weights[i];
                MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> & outside_weights = traces_outside_weights[i];
                const auto &topological_order = topological_orders[i];
                for (const auto &item : topological_order) {
                    free(inside_weights.at(item).data());
                    free(outside_weights.at(item).data());
                }
            }
        }
        traces_inside_weights.clear();
        traces_outside_weights.clear();
    }

    inline void io_weights_la(
            const std::vector<RuleTensor<double>> & rules,
            const WeightVector & root,
            const unsigned i) {


        // TODO implement for general case (== no topological order) approximation of inside weights
        assert (topological_orders.size() > i && topological_orders[i].size() > 0);

        const auto & topological_order = topological_orders[i];

        // computation of inside weights
        MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> & inside_weights = traces_inside_weights[i];

        for (const auto & item : topological_order) {
            Eigen::TensorMap<Eigen::Tensor<double, 1>> & inside_weight = inside_weights.at(item);

            inside_weight.setZero();

            // witness is an incoming edge into item
            //    - first: edge label
            //    - second: list of source nodes
            for (const auto & witness : traces[i].at(item)) {
                switch (witness.second.size() + 1) {
                    case 1:
                        inside_weight_step1(inside_weight, witness, rules);
                        break;
                    case 2:
                        inside_weight_step2(inside_weights, inside_weight, witness, rules);
                        break;
                    case 3:
                        inside_weight_step3(inside_weights, inside_weight, witness, rules);
                        break;
                    case 4:
                        inside_weight_step<4>(inside_weights, inside_weight, witness, rules);
                        break;
                    default:
                        std::cerr<< "Rules with more than 3 RHS nonterminals are not implemented." << std::endl;
                        abort();
                }
            }
            if (debug) {
                std::cerr << "inside weight " << item << std::endl;
                std::cerr << inside_weight << std::endl;
            }
        }

        // TODO implement for general case (== no topological order) solution by gauss jordan
        MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> & outside_weights = traces_outside_weights[i];

        for (auto i_ptr = topological_order.rbegin();  i_ptr != topological_order.rend(); ++i_ptr) {
            const ParseItem<Nonterminal, Position> & item = *i_ptr;

            Eigen::TensorMap<Eigen::Tensor<double, 1>> & outside_weight = outside_weights.at(item);

            if (item == goals[i])
                outside_weight = root;
            else
                outside_weight.setZero();

            if (traces_reverse[i].count(item)) {
                for (const auto &witness : traces_reverse[i].at(item)) {
                    switch (std::get<1>(witness)->size() + 1) {
                        /*case 1:
                            outside_weight_step<1>(rules, nont_dimensions, nont_idx, inside_weights, outside_weights, outside_weight, witness);
                            break;*/
                        case 2:
                            outside_weight_step2(rules, outside_weights, outside_weight, witness);
                            break;
                        case 3:
                            outside_weight_step3(rules, inside_weights, outside_weights, outside_weight, witness);
                            break;
                        case 4:
                            outside_weight_step<4>(rules, inside_weights, outside_weights, outside_weight, witness);
                            break;
                        default:
                            std::cerr<< "Rules with more than 3 RHS nonterminals are not implemented." << std::endl;
                            abort();
                    }
                }
            }
            if (debug && false) std::cerr << "outside weight " << item << std::endl << outside_weight << std::endl;
        }
    }

#include "aux.h"


    template<int rule_rank, typename Witness>
    void outside_weight_step(const std::vector<RuleTensor<double>> &rules,
                             const MAPTYPE<ParseItem < Nonterminal, Position>, WeightVector> & inside_weights,
                             const MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector> & outside_weights,
                             WeightVector & outside_weight,
                             const Witness & witness) const {
        const auto &rule = *(std::get<0>(witness));
        const auto &siblings = *(std::get<1>(witness));
        const auto &parent = *(std::get<2>(witness));
        const unsigned position = std::get<3>(witness);


        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight
                = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[rule.id]);
        const Eigen::array<long, rule_rank> & rule_dim = rule_weight.dimensions();
        Eigen::array<long, rule_rank> rshape_dim;
        Eigen::array<long, rule_rank> broad_dim;

        for (unsigned i = 0; i < rule_rank; ++i) {
            rshape_dim[i] = 1;
            broad_dim[i] = rule_dim[i];
        }

        const auto & parent_weight = outside_weights.at(parent);

        Eigen::Tensor<double, rule_rank> rule_val = rule_weight;
        if (debug) std::cerr << "init rule_val" << std::endl<< rule_val << std::endl << std::endl;

        rshape_dim[0] = broad_dim[0];
        broad_dim[0] = 1;
        rule_val *= parent_weight.reshape(rshape_dim).broadcast(broad_dim);
        if (debug) std::cerr << "rule_val" << 0 << std::endl<< rule_val << std::endl << std::endl;
        broad_dim[0] = rule_dim[0];
        rshape_dim[0] = 1;

        for (unsigned rhs_pos = 1; rhs_pos < rule_rank; ++rhs_pos) {
            if (rhs_pos == position + 1)
                continue;

            const Eigen::TensorMap<Eigen::Tensor<double, 1>> & item_weight = inside_weights.at(*siblings[rhs_pos - 1]);
            if (debug) std::cerr << "inside weight " << rhs_pos << std::endl<< item_weight << std::endl << std::endl;
            rshape_dim[rhs_pos] = broad_dim[rhs_pos];
            broad_dim[rhs_pos] = 1;
            rule_val *= item_weight.reshape(rshape_dim).broadcast(broad_dim);
            if (debug) std::cerr << "int rule_val" << rhs_pos << std::endl<< rule_val << std::endl << std::endl;
            broad_dim[rhs_pos] = rshape_dim[rhs_pos];
            rshape_dim[rhs_pos] = 1;
        }

        Eigen::array<long, rule_rank - 1> sum_array;
        for (unsigned i = 0; i < rule_rank - 1; ++i) {
            if (i < position + 1)
                sum_array[i] = i;
            if (i >= position + 1)
                sum_array[i] = i + 1;
        }
        Eigen::Tensor<double, 1> outside_weight_summand = rule_val.sum(sum_array);

        if (debug) std::cerr << "final rule_val" << std::endl<< rule_val << std::endl << std::endl;
        if (debug) std::cerr << "outside weight summand" << std::endl << outside_weight_summand << std::endl << std::endl;

        outside_weight += outside_weight_summand;

    }

    template<typename Witness>
    void inside_weight_step1(
            WeightVector & inside_weight,
            const Witness & witness,
            const std::vector<RuleTensor<double>> &rules) const {
        constexpr unsigned rule_rank {1};


        if (debug)
            std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[witness.first->id]);
        if (debug)
            std::cerr << "rule tensor " << witness.first->id << " address " << rules[witness.first->id] << std::endl << rule_weight << std::endl;

        inside_weight += rule_weight;
    }

    template<typename Witness>
    inline void inside_weight_step2(
                            const MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector> & inside_weights,
                            WeightVector & inside_weight,
                            const Witness & witness,
                            const std::vector<RuleTensor<double>> &rules) const {
        constexpr unsigned rule_rank {2};

        if (debug)
            std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[witness.first->id]);
        if (debug)
            std::cerr << "rule tensor " << witness.first->id << " address " << rules[witness.first->id] << std::endl << rule_weight << std::endl;

        constexpr unsigned rhs_pos = 1;
        const WeightVector & item_weight = inside_weights.at(*(witness.second[rhs_pos - 1]));
        auto c1 = rule_weight.contract(item_weight, Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
        inside_weight += c1;
    }

    template<typename Witness>
    inline void inside_weight_step3(
                            const MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector> & inside_weights,
                            Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &
                            inside_weight, Witness & witness, const std::vector<RuleTensor<double>> &rules) const {
        constexpr unsigned rule_rank{3};

        if (debug)
            std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[witness.first->id]);
        if (debug)
            std::cerr << "rule tensor " << witness.first->id << " address " << rules[witness.first->id] << std::endl << rule_weight << std::endl;

        constexpr unsigned rhs_pos1 = 1;
        const WeightVector & rhs_item_weight1 = inside_weights.at(*(witness.second[rhs_pos1 - 1]));

        constexpr unsigned rhs_pos2 = 2;
        const WeightVector & rhs_item_weight2 = inside_weights.at(*(witness.second[rhs_pos2 - 1]));

        auto c1 = rule_weight.contract(rhs_item_weight2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(2, 0)});
        auto c2 = c1.contract(rhs_item_weight1, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)});
        inside_weight += c2;
    }


    template<int rule_rank, typename Witness>
    void inside_weight_step(const MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector> & inside_weights,
                                WeightVector &
                                inside_weight, Witness & witness, const std::vector<RuleTensor<double>> &rules) const {
        if (debug)
            std::cerr << std::endl << "Computing inside weight summand" << std::endl;
        const Eigen::TensorMap<Eigen::Tensor<double, rule_rank>> & rule_weight = boost::get<Eigen::TensorMap<Eigen::Tensor<double, rule_rank>>>(rules[witness.first->id]);
        const auto & rule_dim = rule_weight.dimensions();
        if (debug)
            std::cerr << "rule tensor " << witness.first->id << " address " << rules[witness.first->id] << std::endl << rule_weight << std::endl;


        Eigen::Tensor<double, rule_rank> tmp_value = rule_weight;

        Eigen::array<long, rule_rank> rshape_dim;
        Eigen::array<long, rule_rank> broad_dim;
        for (unsigned i = 0; i < rule_rank; ++i) {
            rshape_dim[i] = 1;
            broad_dim[i] = rule_dim[i];
        }

        for (unsigned rhs_pos = 1; rhs_pos < rule_rank; ++rhs_pos) {
            const Eigen::TensorMap<Eigen::Tensor<double, 1>> & item_weight = inside_weights.at(*(witness.second[rhs_pos - 1]));
            rshape_dim[rhs_pos] = broad_dim[rhs_pos];
            broad_dim[rhs_pos] = 1;
            tmp_value *= item_weight.reshape(rshape_dim).broadcast(broad_dim);
            broad_dim[rhs_pos] = rshape_dim[rhs_pos];
            rshape_dim[rhs_pos] = 1;
        }

        inside_weight += tmp_value.sum(Eigen::array<long, 1>({1}));

        /*
        nont_pos = 1;
        for (const auto & dep_item : witness.second) {
                    double * ptr = inside_weights.at(*dep_item);
                    Eigen::TensorMap<Eigen::Tensor<double, 1>> rhs_weight(ptr, rule_dim[nont_pos]);
                    if (debug)
                        std::cerr << "rhs " << nont_pos << " inside weight " << std::endl << rhs_weight << std::endl;
                    rhs_weights.push_back(rhs_weight);
                    ++nont_pos;
                }
        if (debug)
            std::cerr << "resulting inside weight summand " << std::endl <<  rule_probs(rule_weight, rhs_weights) << std::endl << std::endl;

        inside_weight += rule_probs(rule_weight, rhs_weights);
         */
    }

    template <typename NontToIdx>
    void do_em_training_la(
            const unsigned N_THREADS
            , const unsigned BATCH_SIZE
            , std::vector<RuleTensor<double>> & rule_tensors
            , std::vector<double*> & rule_weights
            , double* const the_root_weights
            , const std::vector<std::vector<unsigned>> & normalization_groups
            , const unsigned n_epochs
            , const std::vector<unsigned> & nont_dimensions
            , const std::vector<std::vector<unsigned>> & rule_to_nont_ids
            , const NontToIdx & nont_idx
    ){

        feenableexcept(FE_DIVBYZERO | FE_INVALID| FE_OVERFLOW);
        unsigned epoch = 0;

        const unsigned root_dimension = nont_dimensions[nont_idx(goals[0].nonterminal)];
        unsigned rule_dimension_total = 0;

        std::vector<std::vector<double*>> rule_counts (N_THREADS);
        std::vector<std::vector<RuleTensor<double>>> rule_count_tensors (N_THREADS);
        std::vector<std::vector<unsigned>> rule_dimensions;
        for (auto nont_ids : rule_to_nont_ids) {
            unsigned size = 1;
            std::vector<unsigned> rule_dimension;
            for (auto nont_id : nont_ids) {
                rule_dimension.push_back(nont_dimensions[nont_id]);
                size *= nont_dimensions[nont_id];
            }
            rule_dimensions.push_back(rule_dimension);
            for (unsigned thread = 0; thread < N_THREADS; ++thread) {
                double *ptr = get_region(size);
                rule_dimension_total += size;

                rule_counts[thread].push_back(ptr);

                switch (rule_dimension.size()) {
                    case 1:
                        rule_count_tensors[thread].emplace_back(createTensor<1>(ptr, rule_dimension));
                        break;
                    case 2:
                        rule_count_tensors[thread].emplace_back(createTensor<2>(ptr, rule_dimension));
                        break;
                    case 3:
                        rule_count_tensors[thread].emplace_back(createTensor<3>(ptr, rule_dimension));
                        break;
                    case 4:
                        rule_count_tensors[thread].emplace_back(createTensor<4>(ptr, rule_dimension));
                        break;
                    default:
                        abort();
                }
            }

        }

        Eigen::TensorMap<Eigen::Tensor<double, 1>> root_probability (the_root_weights, root_dimension);

        // initialize root counts
        double* root_count_ptr = get_region(root_dimension);
        Eigen::TensorMap<Eigen::Tensor<double, 1>> root_count (root_count_ptr, root_dimension);

        Eigen::Tensor<double, 1> trace_root_probabilities;
        Eigen::Tensor<double, 0> trace_root_probability;
        Eigen::Tensor<double, 0> corpus_prob_sum;

        while (epoch < n_epochs) {

            if (self_malloc) {
                // reset rule counts (all at once)
                Eigen::TensorMap<Eigen::Tensor<double, 1>> rule_count(rule_counts[0][0], rule_dimension_total);
                rule_count.setZero();
            } else {
                for (unsigned thread = 0; thread < N_THREADS; ++thread) {
                    unsigned rule = 0;
                    for (auto nont_ids : rule_to_nont_ids) {
                        const unsigned size = calc_size(rule_dimensions[rule]);
                        Eigen::TensorMap<Eigen::Tensor<double, 1>> rule_count(rule_counts[thread][rule], size);
                        rule_count.setZero();
                        ++rule;
                    }
                }
            }

            // reset root counts
            root_count.setZero();

            double corpus_likelihood(0.0);

            // expectation
            if (N_THREADS <= 1) {
                expectation_la(rule_tensors, rule_count_tensors[0], rule_dimensions, root_probability,
                               root_count, corpus_likelihood, 0, traces.size());
            } else {
                expectation_la_parallel(N_THREADS, BATCH_SIZE, rule_tensors, rule_count_tensors, rule_dimensions, root_probability,
                                        root_count, corpus_likelihood);
            }

            // maximization
            if (N_THREADS <= 1) {
                unsigned nont = 0;
                for (const std::vector<unsigned> & group : normalization_groups) {
                    const unsigned lhs_dim = rule_dimensions[group[0]][0];
                    maximization(lhs_dim, rule_dimensions, group, rule_counts[0], rule_weights);
                    if (debug) {
                        std::cerr << "rules for nonterminal " << nont << std::endl;
                        for (auto rule : group) {
                            std::cerr << "rule " << rule << " has probabilites: " << std::endl;
                            std::cerr << Eigen::TensorMap<Eigen::Tensor<double, 2>>(rule_weights[rule], lhs_dim, subdim(rule_dimensions[rule])) << std::endl;
                        }
                    }
                    ++nont;
                }
            } else {
                unsigned nont = 0;
                std::mutex nont_mutex;

                std::vector<std::thread> workers;
                for (unsigned thread = 0; thread < N_THREADS; ++thread) {
                    workers.push_back(std::thread([&]()
                    {
                        unsigned my_nont {0};
                        unsigned max_nont {0};

                        while (my_nont < normalization_groups.size()) {
                            {
                                std::lock_guard<std::mutex> lock(nont_mutex);
                                my_nont = nont;
                                max_nont = std::min<unsigned>(nont + BATCH_SIZE, normalization_groups.size());
                                nont = max_nont;
                            }

                            for ( ; my_nont < max_nont; ++my_nont) {
                                const std::vector<unsigned> & group = normalization_groups[my_nont];
                                const unsigned lhs_dim = rule_dimensions[group[0]][0];
                                maximization(lhs_dim, rule_dimensions, group, rule_counts[0], rule_weights);
                                if (debug) {
                                    std::cerr << "rules for nonterminal " << my_nont << std::endl;
                                    for (auto rule : group) {
                                        std::cerr << "rule " << rule << " has probabilites: " << std::endl;
                                        std::cerr << Eigen::TensorMap<Eigen::Tensor<double, 2>>(rule_weights[rule],
                                                                                                lhs_dim,
                                                                                                subdim(rule_dimensions[rule]))
                                                  << std::endl;
                                    }
                                }
                            }
                        }
                    }));
                }
                std::for_each(workers.begin(), workers.end(), [](std::thread &t)
                {
                    t.join();
                });
            }
            if (debug) std::cerr << std::endl;

            epoch++;
            std::cerr <<"Epoch " << epoch << "/" << n_epochs << ": ";


            // maximize root weights:
            corpus_prob_sum = root_count.sum();

            // output likelihood information based on old probability assignment
            std::cerr << "corpus prob. sum " << corpus_prob_sum;
            std::cerr << " corpus likelihood " << corpus_likelihood;
            std::cerr << " root weights: " << root_probability << std::endl;

            // compute new root weights
            if (corpus_prob_sum(0) > 0)
                root_probability = root_count * (1 / corpus_prob_sum(0));
        }

        if (self_malloc) {
            if (not free_region(root_count_ptr, root_dimension))
                abort();
            if (not free_region(rule_counts[0][0], rule_dimension_total))
                abort();
        } else {
            free_region(root_count_ptr, root_dimension);
            for (unsigned thread = 0; thread < N_THREADS; ++ thread) {
                for (auto ptr : rule_counts[thread]) {
                    free_region(ptr, 0);
                }
            }
        }
        fedisableexcept(FE_DIVBYZERO | FE_INVALID| FE_OVERFLOW);

    }

    template<typename VECTOR>
    inline void expectation_la(const std::vector<RuleTensor<double>> &rule_tensors,
                        std::vector<RuleTensor<double>> &rule_count_tensors,
                        const std::vector<std::vector<unsigned int>> &rule_dimensions,
                        const Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &root_probability,
                        VECTOR &root_count,
                        double &corpus_likelihood,
                        const unsigned start,
                        const unsigned end
    ) {
        // expectation
        for (unsigned trace_id = start; trace_id < end; ++trace_id) {
                auto trace = traces[trace_id];
                if (trace.size() == 0)
                    continue;

                io_weights_la(rule_tensors, root_probability, trace_id);

                const auto & inside_weights = traces_inside_weights[trace_id];
                const auto & outside_weights = traces_outside_weights[trace_id];

                const Eigen::TensorMap<Eigen::Tensor<double, 1>> & root_inside_weight = inside_weights.at(goals[trace_id]);
                const Eigen::TensorMap<Eigen::Tensor<double, 1>> & root_outside_weight = outside_weights.at(goals[trace_id]);

                Eigen::Tensor<double, 1> trace_root_probabilities = root_inside_weight * root_outside_weight;
                Eigen::Tensor<double, 0> trace_root_probability = trace_root_probabilities.sum();

                if (not std::isnan(trace_root_probability(0))
                    and not std::isinf(trace_root_probability(0))
                    and trace_root_probability(0) > 0) {
                    root_count += trace_root_probabilities;
                    corpus_likelihood += log(trace_root_probability(0));
                } else {
                    corpus_likelihood += minus_infinity;
                    continue;
                }

                if (debug)
                    std::cerr << "instance root probability: " << std::endl << trace_root_probabilities << std::endl;

                for (auto & pair : trace) {
                    const WeightVector & lhn_outside_weight = outside_weights.at(pair.first);

                    if (debug) {
                        std::cerr << pair.first << std::endl << "outside weight" << std::endl << lhn_outside_weight << std::endl;
                        std::cerr << "inside weight" << std::endl;
                        const WeightVector & lhn_inside_weight = inside_weights.at(pair.first);
                        std::cerr << lhn_inside_weight << std::endl;
                    }
                    for (const auto & witness : pair.second) {
                        const int rule_id = witness.first->id;
                        const unsigned rule_dim = rule_dimensions[rule_id].size();

                        switch (rule_dim) {
                            case 1:
                                compute_rule_count1(rule_tensors[rule_id],
                                                    lhn_outside_weight, trace_root_probability(0),
                                                    rule_count_tensors[rule_id]);
                                break;
                            case 2:
                                compute_rule_count2(rule_tensors[rule_id], witness,
                                                      lhn_outside_weight, trace_root_probability(0),
                                                      inside_weights,
                                                      rule_count_tensors[rule_id]
//                                                           rule_counts[rule_id]
                                );
                                break;
                            case 3:
                                compute_rule_count3(rule_tensors[rule_id], witness,
                                                      lhn_outside_weight, trace_root_probability(0),
                                                      inside_weights,
                                                      rule_count_tensors[rule_id]
//                                                          rule_counts[rule_id]
                                );
                                break;
                            case 4:
                                compute_rule_count<4>(rule_tensors[rule_id], witness,
                                                      lhn_outside_weight, trace_root_probability(0),
                                                      inside_weights,
                                                      rule_count_tensors[rule_id]
//                                                          rule_counts[rule_id]
                                );
                                break;
                            default:
                                std::cerr << "Rules with RHS > " << 3 << " are not implemented." << std::endl;
                                abort();
                        }
                    }
                }

            }
    }

    inline void expectation_la_parallel(const unsigned THREADS, const unsigned BATCH_SIZE,
                               const std::vector<RuleTensor<double>> &rule_tensors,
                               std::vector<std::vector<RuleTensor<double>>> &rule_count_tensors,
                               const std::vector<std::vector<unsigned int>> &rule_dimensions,
                               const Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &root_probability,
                               Eigen::TensorMap<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>, 0, Eigen::MakePointer> &root_count,
                               double & corpus_likelihood) {

        unsigned next_trace {0};
        std::mutex next_trace_mutex;

        std::vector<double> corpus_likelihoods(THREADS, corpus_likelihood);
        std::vector<Eigen::Tensor<double, 1>> root_counts;
        for (unsigned thread = 0; thread < THREADS; ++thread) {
            Eigen::Tensor<double, 1> root_count_thread(root_count.dimension(0));
            root_count_thread.setZero();
            root_counts.emplace_back(std::move(root_count_thread));
        }

        std::vector<std::thread> workers;
        for (unsigned thread = 0; thread < THREADS; ++thread) {
            workers.push_back(std::thread([thread,BATCH_SIZE,this,&next_trace_mutex,&next_trace,&rule_tensors,&root_probability,&root_counts,&rule_dimensions,&corpus_likelihoods,&rule_count_tensors]() {
                unsigned my_next_trace = 0;
                unsigned my_last_trace = 0;
                while (my_next_trace < traces.size()) {
                    {
                        std::lock_guard<std::mutex> lock(next_trace_mutex);
                        my_next_trace = next_trace;
                        next_trace += BATCH_SIZE;
                        my_last_trace = std::min<unsigned>(next_trace, traces.size());
                    }

                    expectation_la(rule_tensors, rule_count_tensors[thread], rule_dimensions, root_probability,
                                   root_counts[thread], corpus_likelihoods[thread], my_next_trace, my_last_trace);
                }
            }));
        }

        std::for_each(workers.begin(), workers.end(), [](std::thread &t)
        {
            t.join();
        });

        // collect counts, etc.
        corpus_likelihood
                = std::accumulate(corpus_likelihoods.begin(), corpus_likelihoods.end(), corpus_likelihood, std::plus<double>());
        for (unsigned rule = 0; rule < rule_count_tensors[0].size(); ++ rule) {
            for (unsigned thread = 1; thread < THREADS; ++thread) {
                switch (rule_dimensions[rule].size()) {
                    case 1:
                        boost::get<Eigen::TensorMap<Eigen::Tensor<double, 1>>>(rule_count_tensors[0][rule])
                                += boost::get<Eigen::TensorMap<Eigen::Tensor<double, 1>>>(rule_count_tensors[thread][rule]);
                        break;
                    case 2:
                        boost::get<Eigen::TensorMap<Eigen::Tensor<double, 2>>>(rule_count_tensors[0][rule])
                                += boost::get<Eigen::TensorMap<Eigen::Tensor<double, 2>>>(rule_count_tensors[thread][rule]);
                        break;
                    case 3:
                        boost::get<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(rule_count_tensors[0][rule])
                                += boost::get<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(rule_count_tensors[thread][rule]);
                        break;
                    case 4:
                        boost::get<Eigen::TensorMap<Eigen::Tensor<double, 4>>>(rule_count_tensors[0][rule])
                                += boost::get<Eigen::TensorMap<Eigen::Tensor<double, 4>>>(rule_count_tensors[thread][rule]);
                        break;
                    default:
                        abort();
                }
            }
        }
        for (unsigned thread = 0; thread < THREADS; ++thread) {
            root_count += root_counts[thread];
        }
    }

    template <typename NontToIdx>
    inline void estimateNontFreqLA(const unsigned start,
                            const unsigned stop,
                            const NontToIdx nont_idx,
                        std::vector<Eigen::Tensor<double, 1, 0, Eigen::DenseIndex>> &merge_weights_partial,
                        const std::vector<RuleTensor<double>> &rule_weight_tensors,
                        const WeightVector &root_weight_tensor) {
        // computing in(A_x) * out(A_x) for every A ∈ N and x ∈ X_A
        for (unsigned trace_id = start; trace_id < stop; ++trace_id) {
            io_weights_la(rule_weight_tensors, root_weight_tensor, trace_id);
            const auto & inside_weights = traces_inside_weights[trace_id];
            const auto & outside_weights = traces_outside_weights[trace_id];

            for (const auto & pair : traces[trace_id]) {
                const ParseItem<Nonterminal, Position> & item = pair.first;

                const Eigen::TensorMap<Eigen::Tensor<double, 1>> & inside_weight = inside_weights.at(item);
                const Eigen::TensorMap<Eigen::Tensor<double, 1>> & outside_weight = outside_weights.at(item);

                const auto vals = inside_weight * outside_weight;
                Eigen::Tensor<double, 0> denominator = vals.sum();
                Eigen::Tensor<double, 1> fraction = vals * (1 / denominator(0));
                Eigen::Tensor<bool, 0> nan = fraction.isnan().any();
                Eigen::Tensor<bool, 0> inf = fraction.isinf().any();
                if (not nan(0) and not inf(0)){
                    auto & target =  merge_weights_partial[nont_idx(item.nonterminal)];
                    target += fraction;
                }
            }
        }
    }

    template <typename NontToIdx, typename Val>
    inline void computeMergeDeltas(const unsigned start, const unsigned stop, const NontToIdx nont_idx, const std::vector<std::vector<Val>> &p, std::vector<Val> &prefixes,
                   std::vector<Val> &postfixes, const std::vector<unsigned int> &nont_dimensions,
                   std::vector<std::vector<Val>> &merge_delta) const {
        for (unsigned trace_id = start; trace_id < stop; ++trace_id) {
            const MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector> & inside_weights = traces_inside_weights[trace_id];
            const MAPTYPE<ParseItem<Nonterminal, Position>, WeightVector> & outside_weights = traces_outside_weights[trace_id];

            for (const auto & pair : traces[trace_id]) {
                const ParseItem<Nonterminal, Position> &item = pair.first;

                // compute Q( item )
//                Val denominator = Val::zero();
//                for (unsigned dim : boost::irange((unsigned) 0, nont_dimensions[nont_idx(item.nonterminal)])) {
//                    const Val in = io_weight.first.at(item)[dim];
//                    const Val out = io_weight.second.at(item)[dim];
//                    denominator += (in * out);
//                    assert(! std::isnan(denominator.get_Value()));
//                }
                const auto nont_dim = nont_dimensions[nont_idx(item.nonterminal)];
                prefixes.resize(nont_dim / 2, Val::zero());
                postfixes.resize(nont_dim / 2, Val::zero());
                Val denominator = Val::zero();
                {
                    const unsigned dim = nont_dim - 2;
                    const Val in1 = Val::to(inside_weights.at(item).data()[dim]);
                    const Val in2 = Val::to(inside_weights.at(item).data()[dim + 1]);
                    const Val out1 = Val::to(outside_weights.at(item).data()[dim]);
                    const Val out2 = Val::to(outside_weights.at(item).data()[dim + 1]);
                    denominator += in1 * out1 + in2 * out2;
                }
                for (unsigned dim = 0; dim < nont_dim - 2; dim = dim + 2) {
                    {
                        const Val in1 = Val::to(inside_weights.at(item).data()[dim]);
                        const Val in2 = Val::to(inside_weights.at(item).data()[dim + 1]);
                        const Val out1 = Val::to(outside_weights.at(item).data()[dim]);
                        const Val out2 = Val::to(outside_weights.at(item).data()[dim + 1]);
                        prefixes[dim / 2 + 1] = prefixes[dim / 2] + in1 * out1 + in2 * out2;
                        denominator += in1 * out1 + in2 * out2;
                    }
                    {
                        const unsigned dim_ = nont_dim - dim - 2;
                        const Val in1 = Val::to(inside_weights.at(item).data()[nont_dim - dim_]);
                        const Val in2 = Val::to(inside_weights.at(item).data()[nont_dim - dim_ + 1]);
                        const Val out1 = Val::to(outside_weights.at(item).data()[nont_dim - dim_]);
                        const Val out2 = Val::to(outside_weights.at(item).data()[nont_dim - dim_ + 1]);
                        postfixes[(nont_dim - dim) / 2 - 2] = postfixes[(nont_dim - dim)/ 2] + in1 * out1 + in2 * out2;
                    }
                }

                // in of some item can be zero in certain LA-dimensions
                // since LA-rule weights may converge to zero
                // we ignore those dimensions in Δ computation
                if (denominator == Val::zero())
                    continue;

                for (unsigned dim = 0; dim < nont_dimensions[nont_idx(item.nonterminal)]; dim = dim+2) {
                    const Val in1 = Val::to(inside_weights.at(item).data()[dim]);
                    const Val in2 = Val::to(inside_weights.at(item).data()[dim + 1]);
                    const Val out1 = Val::to(outside_weights.at(item).data()[dim]);
                    const Val out2 = Val::to(outside_weights.at(item).data()[dim + 1]);
                    const unsigned nont = nont_idx(item.nonterminal);
                    const Val p1 = p[nont][dim];
                    const Val p2 = p[nont][dim+1];

                    const Val out_merged = out1 + out2;
                    const Val in_merged = (p1 * in1) + (p2 * in2);

//                    const Val Q = Val::add_subtract2_divide(denominator, in_merged * out_merged, in1 * out1, in2 * out2, denominator);
                    const Val Q = (prefixes[dim / 2] + postfixes[dim / 2] + in_merged * out_merged) / denominator;

                    if (std::isnan(Q.get_Value())) {
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
                        assert(!std::isnan(Q.get_Value()));
                    }

                    Val & delta = merge_delta[nont][dim / 2];
                    delta *= Q;

                }
                prefixes.clear();
                postfixes.clear();
            }
        }
    }


    template <typename Val, typename NontToIdx>
    std::tuple< std::vector<std::vector<std::vector<unsigned>>>
              , std::vector<unsigned>
              , std::vector<std::vector<Val>>
              >
    merge_prepare(
            const unsigned N_THREADS
            , const unsigned BATCH_SIZE
            , const std::vector<std::vector<Val>> & rule_weights
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
        // TODO allocate via get_region
        std::vector<std::vector<Eigen::Tensor<double, 1>>> merge_weights_partial(N_THREADS);
        for (auto dim : nont_dimensions) {
            Eigen::Tensor<double, 1> merge_weight(dim);
            merge_weight.setZero();
            for (unsigned thread = 0; thread < N_THREADS; ++thread)
                merge_weights_partial[thread].emplace_back(merge_weight);
        }

        // conversion
        std::vector<double *> rule_weights_ptrs;
        std::vector<RuleTensor<double>> rule_weight_tensors;
        double * root_weights_ptrs;

        std::vector<std::vector<unsigned>> rule_dimensions;
        for (const auto rule : rule_ids_to_nont_ids) {
            std::vector<unsigned> rule_dimension;
            for (unsigned nont : rule) {
                rule_dimension.emplace_back(nont_dimensions[nont]);
            }
            rule_dimensions.emplace_back(std::move(rule_dimension));
        }

        unsigned allocated = convert_to_eigen(rule_weights_ptrs, rule_weights, rule_weight_tensors, root_weights_ptrs, root_weights, rule_dimensions);

        WeightVector root_weight_tensor(root_weights_ptrs, root_weights.size());

        std::cerr << "Estimating relative frequency of annotated nonterminals." << std::endl;
        if (N_THREADS <= 1)
            estimateNontFreqLA(0, traces.size(), nont_idx, merge_weights_partial[0], rule_weight_tensors, root_weight_tensor);
        else {
            std::mutex next_trace_mutex;
            unsigned next_trace {0};
            std::vector<std::thread> workers;
            for (unsigned thread = 0; thread < N_THREADS; ++thread) {
                workers.push_back(std::thread([thread, this, &BATCH_SIZE, &next_trace, &next_trace_mutex, &nont_idx,
                                                      &merge_weights_partial, &rule_weight_tensors,
                                                      &root_weight_tensor](){
                    unsigned my_next_trace {0};
                    unsigned my_last_trace {0};
                    while (my_next_trace < traces.size()) {
                        {
                            std::lock_guard<std::mutex> lock(next_trace_mutex);
                            my_next_trace = next_trace;
                            // since nonterminal freq expectation is cheaper than rule freq expectation,
                            // we doubled the batch_size here
                            next_trace += BATCH_SIZE * 2;
                            my_last_trace = std::min<unsigned>(next_trace, traces.size());
                        }
                        estimateNontFreqLA(my_next_trace, my_last_trace, nont_idx, merge_weights_partial[thread],
                                           rule_weight_tensors, root_weight_tensor);
                    }
                }));
            }
            std::for_each(workers.begin(), workers.end(), [](std::thread &t)
            {
                t.join();
            });
            for (unsigned nont = 0; nont < merge_weights_partial[0].size(); ++nont) {
                for (unsigned thread = 1; thread < N_THREADS; ++thread) {
                    merge_weights_partial[0][nont] += merge_weights_partial[0][thread];
                }
            }
        }

        std::cerr << "Computing merge factors." << std::endl;
        // finally we compute the fractions
        std::vector<std::vector<Val>> p;
        for (auto las_weights : merge_weights_partial[0]) {
            p.emplace_back(std::vector<Val>());
            for (unsigned i = 0; i < las_weights.dimension(0); i = i + 2) {
                double combined_weight = las_weights(i) + las_weights(i+1);
                if ((not std::isnan(combined_weight)) and combined_weight > 0) {
                    p.back().emplace_back(Val::to(las_weights(i) / combined_weight));
                    p.back().emplace_back(Val::to(las_weights(i + 1) / combined_weight));
                } else {
                    p.back().emplace_back(Val::to(0.5));
                    p.back().emplace_back(Val::to(0.5));
                }
            }
        }

        std::cerr << "Computing likelihood deltas of merges." << std::endl;
        // now we approximate the likelihood Δ of merging two latent states
        std::vector<std::vector<std::vector<Val>>> merge_delta(N_THREADS);
        for (auto dim : nont_dimensions) {
            for (unsigned thread {0}; thread < N_THREADS; ++thread)
                merge_delta[thread].push_back(std::vector<Val>(dim / 2, Val::one()));
        }

        if (N_THREADS <= 1) {
            std::vector<Val> prefixes;
            std::vector<Val> postfixes;
            unsigned start = 0;
            unsigned stop = traces.size();
            computeMergeDeltas(start, stop, nont_idx, p, prefixes, postfixes, nont_dimensions, merge_delta[0]);
        } else {
            std::mutex next_trace_mutex;
            unsigned next_trace {0};
            std::vector<std::thread> workers;
            for (unsigned thread = 0; thread < N_THREADS; ++thread) {
                workers.push_back(std::thread([thread, this, &next_trace_mutex, &next_trace, &BATCH_SIZE, &nont_idx, &p, &nont_dimensions, &merge_delta](){
                    std::vector<Val> prefixes;
                    std::vector<Val> postfixes;
                    unsigned my_next_trace {0};
                    unsigned my_last_trace {0};
                    while (my_next_trace < traces.size()) {
                        {
                            std::lock_guard<std::mutex> lock(next_trace_mutex);
                            my_next_trace = next_trace;
                            next_trace += BATCH_SIZE;
                            my_last_trace = std::min<unsigned>(next_trace, traces.size());
                        }
                        computeMergeDeltas(my_next_trace, my_last_trace, nont_idx, p, prefixes, postfixes, nont_dimensions, merge_delta[thread]);
                    }
                }));
            }
            std::for_each(workers.begin(), workers.end(), [](std::thread &t)
            {
                t.join();
            });
            for (unsigned nont = 0; nont < merge_delta[0].size(); ++nont) {
                for (unsigned thread = 1; thread < N_THREADS; ++thread) {
                    std::transform(merge_delta[0][nont].begin(), merge_delta[0][nont].end(),
                                   merge_delta[thread][nont].begin(), merge_delta[0][nont].begin(), std::multiplies<Val>());
                }
            }
        }

        if (self_malloc) {
            if (not free_region(rule_weights_ptrs[0], allocated))
                abort();
        } else {
            for (auto ptr : rule_weights_ptrs) {
                free_region(ptr, 0);
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
            for (const auto & delta : merge_delta[0]) {
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
        for (const auto & delta : merge_delta[0]) {
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
