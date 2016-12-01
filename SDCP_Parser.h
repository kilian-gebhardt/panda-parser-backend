//
// Created by kilian on 18/11/16.
//

#ifndef STERMPARSER_SDCP_PARSER_H
#define STERMPARSER_SDCP_PARSER_H

#include <memory>
#include <vector>
#include <tuple>
#include <queue>
#include "HybridTree.h"
#include "SDCP.h"
#include <type_traits>
#include <iostream>
#include <assert.h>
#include <set>

template<size_t N>
struct print_tuple{
    template<typename... T>static typename std::enable_if<(N<sizeof...(T))>::type
    print(std::ostream& os, const std::tuple<T...>& t) {
        char quote = (std::is_convertible<decltype(std::get<N>(t)), std::string>::value) ? '"' : 0;
        os << ", " << quote << std::get<N>(t) << quote;
        print_tuple<N+1>::print(os,t);
    }
    template<typename... T>static typename std::enable_if<!(N<sizeof...(T))>::type
    print(std::ostream&, const std::tuple<T...>&) {
    }
};
std::ostream& operator<< (std::ostream& os, const std::tuple<>&) {
    return os << "()";
}
template<typename T0, typename ...T> std::ostream&
operator<<(std::ostream& os, const std::tuple<T0, T...>& t){
    char quote = (std::is_convertible<T0, std::string>::value) ? '"' : 0;
    os << '(' << quote << std::get<0>(t) << quote;
    print_tuple<1>::print(os,t);
    return os << ')';
}

template <typename Nonterminal, typename Position>
class ParseItem {
public:
    Nonterminal nonterminal;
    std::vector<std::pair<Position, Position>> spans_inh, spans_syn;
    std::vector<std::pair<int, int>> spans_lcfrs;
};

template <typename Nonterminal, typename Position>
bool operator==(const ParseItem<Nonterminal, Position>& lhs, const ParseItem<Nonterminal, Position>& rhs)
{
    return lhs.nonterminal == rhs.nonterminal
           && lhs.spans_inh == rhs.spans_inh
           && lhs.spans_syn == rhs.spans_syn
           && lhs.spans_lcfrs == rhs.spans_lcfrs;
}

template <typename Nonterminal, typename Position>
bool operator<(const ParseItem<Nonterminal, Position>& lhs, const ParseItem<Nonterminal, Position>& rhs)
{
    if (lhs.nonterminal < rhs.nonterminal)
        return true;
    if (lhs.nonterminal > rhs.nonterminal)
        return false;
    int i = 0;
    while (i < lhs.spans_inh.size()) {
        if (lhs.spans_inh[i] < rhs.spans_inh[i])
            return true;
        if (lhs.spans_inh[i] > rhs.spans_inh[i])
            return false;
        i++;
    }
    i = 0;
    while (i < lhs.spans_syn.size()) {
        if (lhs.spans_syn[i] < rhs.spans_syn[i])
            return true;
        if (lhs.spans_syn[i] > rhs.spans_syn[i])
            return false;
        i++;
    }
    i = 0;
    while (i < lhs.spans_lcfrs.size()) {
        if (lhs.spans_lcfrs[i] < rhs.spans_lcfrs[i])
            return true;
        if (lhs.spans_lcfrs[i] > rhs.spans_lcfrs[i])
            return false;
        i++;
    }
    return false;
}



template <typename Nonterminal, typename Position>
std::ostream &operator<<(std::ostream &os, ParseItem<Nonterminal, Position> const &item) {
    os << item.nonterminal << " ( ";
    for(auto range : item.spans_inh)
        os << " <" << range.first << "-" << range.second << ">";
    os << " ; ";
    for(auto range : item.spans_syn)
        os << " <" << range.first << "-" << range.second << ">";
    if (item.spans_lcfrs.size()) {
        os << " | ";
        for(auto range : item.spans_lcfrs)
            os << " <" << range.first << "-" << range.second << ">";
    }
    os << " ) ";
    return os;
}


template <typename T>
bool pairwise_different(const std::vector<T> & vec){
    auto i = 0;
    for (auto s1 : vec) {
        for(auto j = ++i; j < vec.size(); ++j) {
            if (s1 == vec[j]) {
                return false;
            }
        }
    }
    return true;
}

template <typename Nonterminal, typename Terminal, typename Position>
class SDCPParser{
private:
    bool match_and_retrieve_vars(
              Position position
            , const STerm<Terminal> & sterm
            , std::map<Variable, std::pair<Position, Position>> & var_assignment
            , std::pair<Position, Position> & span_assignment
            , std::vector<Position> & lcfrs_terminals) {
        span_assignment = std::pair<Position, Position> (position, position);
        bool var_flag = false;
        Variable * var = nullptr;
        for (TermOrVariable<Terminal> obj : sterm) {
            try {
                Term<Terminal> term = boost::get<Term<Terminal>>(obj);

                if (var_flag) {
                    // TODO greedy matching is incomplete / requires normal form
                    while (term.head != input.get_tree_label(position) && ! input.is_final(position))
                        position = input.get_next(position);
                    var_flag = false;
                    var_assignment[*var].second = input.get_previous(position);
                    var = nullptr;
                } else {
                    position = input.get_next(position);
                }

                if (term.head != input.get_tree_label(position))
                    return false;
                if (term.children.size() > 0 != input.get_children(position).size() > 0)
                    return false;
                if (term.children.size() > 0) {
                    std::pair<Position, Position> dummy;
                    if (!match_and_retrieve_vars(input.get_children(position).front(), term.children, var_assignment,
                                                 dummy, lcfrs_terminals))
                        return false;
                }
                if (term.is_ordered()) {
                    if (lcfrs_terminals.size() < term.order + 1)
                        lcfrs_terminals.resize(term.order + 1);
                    lcfrs_terminals[term.order] = position;
                }

            } catch (boost::bad_get&) {
                assert (!var_flag);
                Variable & var_ = boost::get<Variable>(obj);
                var = &var_;
                var_assignment[*var] = std::pair<Position, Position> (position, position);
                var_flag = true;
            }

        }
        if (var_flag) {
            while (! input.is_final(position))
                position = input.get_next(position);
            var_assignment[*var].second = position;
        }
        span_assignment.second = position;
        return true;
    }

    bool match_lcfrs(const Rule<Nonterminal, Terminal> & rule, const std::vector<Position> & lcfrs_terminals, const std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> & items, std::vector<std::pair<int, int>> & lcfrs_spans) {
        unsigned i = 0;
        int span_start;
        int pos;
        for (auto argument : rule.word_function) {
            span_start = -1;
            bool begin = true;
            for (auto obj : argument) {
                try {
                    const Variable & var = boost::get<Variable>(obj);

                    assert (0 < var.member && var.member <= items.size());
                    assert (var.argument <= items[var.member - 1]->spans_lcfrs.size());

                    const std::pair<int,int> & var_pos = items[var.member - 1]->spans_lcfrs[var.argument - 1];

                    if (begin) {
                        span_start = var_pos.first;
                        pos = var_pos.second;
                        begin = false;
                    } else if (pos == var_pos.first) {
                        pos = var_pos.second;
                    }
                    else
                        return false;
                } catch (boost::bad_get&) {
                    const Terminal & term = boost::get<Terminal>(obj);

                    assert (i < lcfrs_terminals.size());

                    const Position & tree_pos = lcfrs_terminals[i++];

                    if (term != input.get_string_label(tree_pos))
                        return false;

                    unsigned pos2 = 0;
                    for ( ; pos2 < input.get_linearization().size(); ++pos2) {
                        if (input.get_linearization()[pos2] == tree_pos) {
                            if (begin) {
                                span_start = pos2;
                                pos = pos2 + 1;
                                begin = false;
                            } else if (pos == pos2) {
                                pos = pos2 + 1;
                            } else
                                return false;
                            break;
                        }
                    }
                    if (pos2 == input.get_linearization().size())
                        return false;
                }
            }
            assert (span_start != - 1);
            lcfrs_spans.push_back(std::make_pair(span_start, pos));
        }
        return true;
    }

    void match_lexical_rules(){
        const std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> no_rhs_items;
        for (auto pair : input.terminals()){
            Position & position = pair.first;
            Terminal & terminal = pair.second;
            for (Rule<Nonterminal, Terminal> rule : sDCP.axioms[terminal]) {
                // std::cerr << rule.lhn << std::endl;
                assert ("Assume normal form that " && rule.outside_attributes.size() == 1);
                assert ("Assume normal form that " && rule.outside_attributes.front().size() == 1);
                Term<Terminal> term = boost::get<Term<Terminal>>(rule.outside_attributes[0][0][0]);
                assert (term.head == terminal);
                std::map<Variable, std::pair<Position, Position>> var_assignment;
                std::pair<Position, Position> span_assignment;
                std::vector<Position> lcfrs_terminals;
                if (match_and_retrieve_vars(input.get_previous(position), rule.outside_attributes[0][0], var_assignment, span_assignment, lcfrs_terminals)){
                    if (debug)
                        std::cerr << "matched sdcp" << std::endl;

                    if (debug) {
                        std::cerr << "lcfrs terminals: [";
                        for (auto position : lcfrs_terminals)
                            std::cerr << position << ", ";
                        std::cerr << "]" << std::endl;
                    }

                    std::vector<std::pair<int, int>> lcfrs_spans;
                    if (debug)
                        std::cerr << "created vector" << std::endl;

                    if (parse_lcfrs && !match_lcfrs(rule, lcfrs_terminals, no_rhs_items, lcfrs_spans)) {
                        if (debug)
                            std::cerr << "did not match lcfrs" << std::endl;
                        continue;
                    }
                    if (debug)
                        std::cerr << "matched lcfrs" << std::endl;

                    std::shared_ptr<ParseItem<Nonterminal, Position>> item = std::make_shared<ParseItem<Nonterminal, Position>>();
                    item->nonterminal = rule.lhn;
                    for (int j = 1; j <= rule.irank(0); ++j) {
                        auto p = var_assignment.at(Variable(0, j));
                        item->spans_inh.emplace_back(p);
                    }
                    item->spans_syn.emplace_back(span_assignment);
                    item->spans_lcfrs = lcfrs_spans;

                    if (no_parallel) {
                        if (   !pairwise_different(item->spans_syn)
                            || !pairwise_different(item->spans_inh)
                            || !pairwise_different(item->spans_lcfrs)) {
                            if (debug)
                                std::cerr << "skipped (no parallel) " << *item << std::endl;
                            continue;
                        }
                    }
                    if (debug)
                        std::cerr << "Agenda : added " << *item << std::endl;
                    agenda.push(item);
                    trace[*item].push_back(std::make_pair(rule, std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> ()));//std::pair<Rule<Nonterminal, Terminal>, std::vector<ParseItem<Nonterminal, Position>>>(rule, std::vector<ParseItem<Nonterminal, Position>> ()));
                }
            }
        }
    }

    bool match_sterm_rec(STerm<Terminal> sterm, Position pos
            , const bool search_goal
            , Position & goal
            , std::vector<std::pair<Position,Position>> & inherited
            , std::vector<std::pair<Position,Position>> & synthesized
            , const std::vector<std::shared_ptr<ParseItem<Nonterminal
            , Position>>> & items
            , std::vector<Position> & lcfrs_terminals){
        bool lhn_var = false;
        Variable var(0,0);
        for (TermOrVariable<Terminal> obj : sterm) {
            try {
                Term<Terminal> &term = boost::get<Term<Terminal>>(obj);
                if (lhn_var) {
                    // TODO greedy matching is incomplete / requires normal form
                    inherited[var.argument - 1].second = pos;
                    lhn_var = false;
                }
                pos = input.get_next(pos);
                if (term.head != input.get_tree_label(pos))
                    return false;
                else {
                    if (input.get_children(pos).size() > 0 && term.children.size() > 0) {
                        Position first_child = input.get_children(pos).front();
                        Position last_child = input.get_children(pos).back();
                        if (!match_sterm_rec(term.children, first_child, false, last_child, inherited, synthesized, items, lcfrs_terminals))
                            return false;
//                        if (child != input.get_children(pos).back())
//                            return false;
                    } else if (input.get_children(pos).size())
                        return false;
                }
                if (term.is_ordered()) {
                    if (lcfrs_terminals.size() < term.order + 1)
                        lcfrs_terminals.resize(term.order + 1);
                    lcfrs_terminals[term.order] = pos;
                }

            } catch (boost::bad_get&) {
                Variable & var_ = boost::get<Variable>(obj);
                if (var_.member > 0) {
                    if (lhn_var) {
                        // TODO greedy matching is incomplete / requires normal form
                        pos = items[var_.member - 1]->spans_syn[var_.argument-1].first;
                        inherited[var.argument - 1].second = pos;
                        if (debug)
                            std::cerr << " match var " << var << " <" << inherited[var.argument-1].first << "-" << pos << ">" << std::endl;
                        lhn_var = false;
                    }
                    else if (pos != items[var_.member - 1]->spans_syn[var_.argument-1].first)
                        return false;
                    pos = items[var_.member - 1]->spans_syn[var_.argument-1].second;
                }
                else {
                    assert (!lhn_var);
                    lhn_var = true;
                    var = var_;
                    inherited[var.argument - 1] = std::make_pair(pos, pos);
                }
            }
        }
        if (lhn_var) {
            // TODO greedy matching is incomplete / requires normal form
            while (!(input.is_final(pos) || (!search_goal && pos == goal)))
                pos = input.get_next(pos);
            inherited[var.argument - 1].second = pos;
        }
        if (search_goal)
            goal = pos;
        if (pos != goal)
            if (debug)
                std::cerr << "pos/goal mismatch where pos=" << pos << " and goal=" << goal << std::endl;
        return pos == goal;
    }

    bool find_start(const STerm<Terminal> & sterm, Position & pos, int level, std::vector<std::pair<Position,Position>> & inherited, std::vector<std::pair<Position,Position>> & synthesized, const std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> & items) {
        int steps = 0;
        for (TermOrVariable<Terminal> obj : sterm) {

            try {
                Term<Terminal> &term = boost::get<Term<Terminal>>(obj);
                steps++;
                Position child;
                if (term.children.size() > 0) {
                    if (find_start(term.children, child, level + 1, inherited, synthesized, items)) {
                        pos = input.get_parent(child);
                        while (steps > 0) {
                            pos = input.get_previous(pos);
                            steps--;
                        }
                        return true;
                    }
                }
            } catch (boost::bad_get&)  {
                Variable var = boost::get<Variable>(obj);
                if (var.member > 0) {
                    pos = items[var.member - 1]->spans_syn[var.argument - 1].first;
                    while (steps > 0)
                        pos = input.get_previous(pos);
                    return true;
                } else {
                    // TODO The case should not occur at top level
                    // by normal form assumptions
                    assert(level > 0);
                }
            }
        }
        return false;
    }

    void match_rule(const Rule<Nonterminal, Terminal> & rule, std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> & transport, const std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> items) {

        if (debug) {
            std::cerr << "match: ";
            for (auto item : items) std::cerr << *item;
            std::cerr << "with rule " << rule << std::endl;
        }

        if (rule.rhs.size() != items.size()) {
            if (debug)
                std::cerr << "size mismatch" << std::endl;
            assert(0);
        }


        std::vector<std::pair<Position,Position>> inherited, synthesized;
        inherited.resize(rule.irank(0));
        std::vector<Position> lcfrs_terminals;
        int mem = 0;
        int arg = 1;
        for (auto attributes : rule.outside_attributes){
            // first, check compatibility of rhs inherited attributes
            // and determine lhs inherited attributes
            arg = 1;
            if (mem > 0) {
                for (STerm<Terminal> sterm : attributes) {
                    // obtain predicted span start
                    Position & pos = items[mem-1]->spans_inh[arg-1].first;
                    Position & goal = items[mem-1]->spans_inh[arg-1].second;

                    if (!match_sterm_rec(sterm, pos, false, goal, inherited, synthesized, items, lcfrs_terminals)) {
                        if (debug)
                            std::cerr << "match error for mem " << mem << " and arg " << arg << std::endl;
                        return;
                    }

                    // check whether predicted span end
//                    if (pos != items[mem-1]->spans_inh[arg-1].second) {
//                        std::cerr << "pos mismatch for mem " << mem << " and arg " << arg << " where pos=" << pos << " vs. " << items[mem - 1]->spans_inh[arg - 1].second
//                                  << std::endl;
//                        return;
//                    }
                    arg++;
                }
            }
            mem++;
        }
        // now check compatibility of lhs synthesized attributes
        arg = 1;
        for (STerm<Terminal> sterm : rule.outside_attributes[0]) {
            Position start;
            if (!find_start(sterm, start, 0, inherited, synthesized, items))
                return;
            Position goal = start;
            if (!match_sterm_rec(sterm, start, true, goal, inherited, synthesized, items, lcfrs_terminals))
                return;
            synthesized.push_back(std::make_pair(start, goal));
            arg++;
        }

        // finally, check lcfrs component
        std::vector<std::pair<int, int>> spans_lcfrs;
        if (parse_lcfrs && !match_lcfrs(rule, lcfrs_terminals, items, spans_lcfrs))
            return;

        auto new_item = std::make_shared<ParseItem<Nonterminal, Position>>();
        new_item->nonterminal = rule.lhn;
        new_item->spans_inh = inherited;
        new_item->spans_syn = synthesized;
        new_item->spans_lcfrs = spans_lcfrs;

        if (no_parallel) {
            if (   !pairwise_different(new_item->spans_syn)
                   || !pairwise_different(new_item->spans_inh)
                   || !pairwise_different(new_item->spans_lcfrs)) {
                if (debug)
                    std::cerr << "skipped (no parallel) " << *new_item << std::endl;
                return;
            }
        }

        if (inh_strict_successor) {
            for (auto s1 : new_item->spans_inh) {
                for (auto s2 : new_item->spans_syn) {
                    if (s1 == s2) {
                        if (debug)
                            std::cerr << "skipped (inherent strict successor) " << *new_item << std::endl;
                        return;
                    }
                }
            }
        }


//        std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> predecessors;
//        for (auto item : items)
//            predecessors.push_back(item);
        trace[*new_item].push_back( // std::pair<Rule<Nonterminal, Terminal>, std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>>>
                                   std::make_pair(rule, items));

        if (debug)
            std::cerr << *new_item << std::endl;
        transport.push_back(new_item);
    }
//    void match_structural_rules(Rule<Nonterminal, Terminal> & rule, ParseItem<Nonterminal, Position> item, int mem){
//        std::vector<std::vector<std::pair<Position,Position>>> inherit, synthesized;
//        inherit[mem] = item.spans_syn;
//        synthesized[mem] = item.spans_inh;
//        mem = 0;
//        for (auto attributes : rule.outside_attributes){
//            for (STerm<Terminal> sterm : attributes){
//
//            }
//
//            mem++;
//        }
//    }
//
//    std::vector<ParseItem<Nonterminal, Position>> get_by(Nonterminal nonterminal, int arg, Position position, bool inh=false, bool start=true){
//        return chart[std::tuple<Nonterminal, bool, int, bool, Position>(nonterminal, inh, arg, start, position)];
//    };
//
//    const std::vector<ParseItem<Nonterminal, Position>> get_by_inh_start(Nonterminal nonterminal, int arg, Position position) {
//        return get_by(nonterminal, arg, position, true, true);
//    };
//    const std::vector<ParseItem<Nonterminal, Position>> get_by_inh_end(Nonterminal nonterminal, int arg, Position position) {
//        return get_by(nonterminal, arg, position, true, false);
//    };
//    const std::vector<ParseItem<Nonterminal, Position>> get_by_syn_start(Nonterminal nonterminal, int arg, Position position) {
//        return get_by(nonterminal, arg, position, false, true);
//    };
//    const std::vector<ParseItem<Nonterminal, Position>> get_by_syn_end(Nonterminal nonterminal, int arg, Position position) {
//        return get_by(nonterminal, arg, position, false, false);
//    };
//


    bool addToChart(std::shared_ptr<ParseItem<Nonterminal, Position>> item) {
        if (!chart.count(item->nonterminal))
            chart[item->nonterminal];
        else {
            for (auto item2 : chart[item->nonterminal])
                if (*item == *item2)
                    return false;
        }
        chart[item->nonterminal].push_back(item);
        return true;
//        int arg = 1;
//        bool first = true;
//        for(auto range : item->spans_syn) {
//            auto key1 = std::tuple<Nonterminal, bool, int, bool, Position>(item->nonterminal, false, arg, true, range.first);
//            auto key2 = std::tuple<Nonterminal, bool, int, bool, Position>(item->nonterminal, false, arg, false, range.second);
//            if (!chart.count(key1))
//                chart[key1];
//            else if (first) {
//                for (auto item2 : chart[key1])
//                    if (*item == *item2)
//                        return false;
//                first = false;
//            }
//            chart[key1].push_back(item);
//            if (!chart.count(key2))
//                chart[key2];
//            chart[key2].push_back(item);
//            arg++;
//        }
//        arg = 1;
//        for(auto range : item->spans_inh){
//            auto key1 = std::tuple<Nonterminal, bool, int, bool, Position>(item->nonterminal, true, arg, true, range.first);
//            auto key2 = std::tuple<Nonterminal, bool, int, bool, Position>(item->nonterminal, true, arg, false, range.second);
//            if (!chart.count(key1))
//                chart[key1];
//            chart[key1].push_back(item);
//            if (!chart.count(key2))
//                chart[key2];
//            chart[key2].push_back(item);
//            arg++;
//        }
//        return true;
    }
public:
    std::queue<std::shared_ptr<ParseItem<Nonterminal, Position>>> agenda;
    std::map<ParseItem<Nonterminal, Position>, std::vector<std::pair<Rule<Nonterminal, Terminal>, std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>> >>>> trace;
    std::map<Nonterminal, std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>>> chart;

    SDCP<Nonterminal, Terminal> sDCP;
    ParseItem<Nonterminal, Position> * goal = nullptr;
    HybridTree<Terminal, Position> input;
    const bool no_parallel;
    const bool inh_strict_successor;
    const bool debug;
    const bool parse_lcfrs;

    SDCPParser(bool parse_lcfrs=false, bool debug=false, bool no_parallel=true, bool inh_strict_successor=true) : no_parallel(no_parallel), inh_strict_successor(inh_strict_successor), debug(debug), parse_lcfrs(parse_lcfrs) {};


    void set_input(HybridTree<Terminal, Position>& tree) {
        input = tree;
    }

    void set_sDCP(SDCP<Nonterminal, Terminal> & sDCP) {
        this->sDCP = sDCP;
    }

    void do_parse() {
        match_lexical_rules();

        std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> transport;

        std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>>> candidates;
        std::vector<int> selection;

        while (! agenda.empty()) {
            auto item_ = agenda.front();
            agenda.pop();
            if (debug)
                std::cerr << *item_ << std::endl;

            if (! addToChart(item_))
                continue;

            for (auto p : sDCP.nont_corner[item_->nonterminal]) {
                Rule<Nonterminal, Terminal> & rule = p.first;
                int & j_ = p.second;
                int j = 0;
                selection.resize(rule.rhs.size(), 0);

                while (0 <= j) {
                    if (j == rule.rhs.size()) {
                        match_rule(rule, transport, candidates);
                        j --;
                        continue;
                    }
                    if (j == j_) {
                        if (selection[j]) {
                            selection[j] = 0;
                            j--;
                            candidates.pop_back();
                        }
                        else {
                            candidates.push_back(item_);
                            selection[j]++;
                            j++;
                        }
                        continue;
                    }
                    else if (selection[j] < chart[rule.rhs[j]].size()) {
                        if (selection[j] == 0)
                            candidates.push_back(chart[rule.rhs[j]][selection[j]]);
                        else
                            candidates[j] = chart[rule.rhs[j]][selection[j]];
                        selection[j]++;
                        j++;
                        continue;
                    }
                    else {
                        if (selection[j] > 0) {
                            candidates.pop_back();
                            selection[j] = 0;
                        }
                        j--;
                        continue;
                    }
                }
                candidates.clear();
            }

            candidates.clear();
            selection.clear();

            for (auto new_item : transport) {
                bool flag = true;
                for (auto chart_item : chart[new_item->nonterminal]) {
                    if (*new_item == *chart_item) {
                        flag = false;
                        break;
                    }
                }
                if (flag)
                    agenda.push(new_item);
            }
            transport.clear();

            if (debug) {
                print_chart();
                std::cerr << "#########################" << std::endl;
            }
        }
    }

    void add_recursively(std::set<ParseItem<Nonterminal, Position>> & reachable, ParseItem<Nonterminal, Position>& start) {
        for (auto list : trace[start]) {
            for (std::shared_ptr<ParseItem<Nonterminal, Position>> item : list.second) {
                if (! reachable.count(*item)) {
                    reachable.insert(*item);
                    add_recursively(reachable, *item);
                }
            }
        }
    }

    void reachability_simplification() {
        std::set<ParseItem<Nonterminal, Position>> reachable;
        if (this->goal) {
            if (debug)
                std::cerr << "goal: " << *(this->goal) << std::endl;
            std::map<ParseItem<Nonterminal, Position>, std::vector<std::pair<Rule<Nonterminal, Terminal>, std::vector<std::shared_ptr<ParseItem<Nonterminal, Position>> >>>> trace_;

            reachable.insert(*(this->goal));
            add_recursively(reachable, *(this->goal));
            for (auto p : trace) {
                if (reachable.count(p.first)) {
                    trace_.insert(p);
                }
            }
            trace = trace_;
        }
    }

    void set_goal() {
        if (goal)
            delete goal;
        goal = new ParseItem<Nonterminal, Position>();
        goal->nonterminal = sDCP.initial;
        goal->spans_syn.emplace_back(std::make_pair(input.get_entry(), input.get_exit()));
        if (parse_lcfrs)
            goal->spans_lcfrs.push_back(std::make_pair(0, input.get_linearization().size()));
        //else goal->spans_lcfrs.push_back(std::make_pair(0, 0));
    }


    void print_chart(){
        for (auto pairs : chart) {
                std::cerr << pairs.first << " : " << std::endl;
                for (auto item : pairs.second) {
                    if (trace[*item].size() > 0) {
                        std::cerr << "  " << *item << " [ ";
                        for (auto trace_entry : trace[*item]) {
                            std::cerr << " [ ";
                            for (auto item_ : trace_entry.second)
                                std::cerr << *item_;
                            std::cerr << " ] ";
                        }
                        std::cerr << "]" << std::endl;
                    }

                }
                std::cerr << std::endl;
        }
    };

    void print_trace(){
            for (auto item : trace) {
                std::cerr << "  " << item.first << " [ ";
                for (auto trace_entry : item.second) {
                    std::cerr << trace_entry.first << " using ";
                    std::cerr << " [ ";
                    for (auto item_ : trace_entry.second)
                        std::cerr << *item_;
                    std::cerr << " ] ";
                }
                std::cerr << "]" << std::endl;

            }
    };

    void clear() {
        std::cerr << "clear" << std::endl;
        agenda = std::queue<std::shared_ptr<ParseItem<Nonterminal, Position>>>();
        chart.clear();
        trace.clear();
    }

    bool recognized() {
        return trace[*goal].size() > 0;
    }

    std::vector<std::pair<Rule<Nonterminal, Terminal>, std::vector<ParseItem<Nonterminal,Position>>>>
        query_trace(ParseItem<Nonterminal, Position> start) {
        std::vector<std::pair<Rule<Nonterminal, Terminal>, std::vector<ParseItem<Nonterminal,Position>>>> result;
        for(auto item : trace[start]) {
            Rule<Nonterminal, Terminal> & rule = item.first;
            std::vector<ParseItem<Nonterminal, Position>> child_items;
            for (auto child : item.second) {
                child_items.push_back(*child);
            }

            result.emplace_back(std::make_pair(item.first, child_items));
        }
        return result;
    }



    // std::
};

#endif //STERMPARSER_SDCP_PARSER_H
