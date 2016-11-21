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

template <typename Nonterminal, typename Position>
class ParseItem {
public:
    Nonterminal nonterminal;
    std::vector<std::pair<Position, Position>> spans_inh, spans_syn;
};

template <typename Nonterminal, typename Terminal, typename Position>
class SDCPParser{
private:
    bool match_and_retrieve_vars(
              Position position
            , const STerm<Terminal> & sterm
            , std::map<Variable, std::pair<Position, Position>> & var_assignment
            , std::pair<Position, Position> & span_assignment) {
        span_assignment = std::pair<Position, Position> (position, position);
        bool var_flag = false;
        Variable * var = nullptr;
        for (TermOrVariable<Terminal> obj : sterm) {
            try {
                Term<Terminal> term = boost::get<Term<Terminal>>(obj);

                if (var_flag) {
                    //TODO actually one needs to search from the end with the current normal form assumptions
                    while (term.head != input.get_label(position) && ! input.is_final(position))
                        position = input.get_next(position);
                    var_flag = false;
                    var_assignment[*var].second = input.get_previous(position);
                    var = nullptr;
                } else {
                    position = input.get_next(position);
                }

                if (term.head != input.get_label(position))
                    return false;
                if (term.children.size() > 0 != input.get_children(position).size() > 0)
                    return false;
                if (term.children.size() > 0) {
                    std::pair<Position, Position> dummy;
                    if (! match_and_retrieve_vars(input.get_children(position)[0]
                            , term.children
                            , var_assignment
                            , dummy))
                        return false;
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
    void match_lexical_rules(){
        for (auto pair : input.terminals()){
            Position & position = pair.first;
            Terminal & terminal = pair.second;
            for (Rule<Nonterminal, Terminal> rule : sDCP.axioms[terminal]) {
                std::cout << rule.lhn << std::endl;
                assert ("Assume normal form that " && rule.outside_attributes.size() == 1);
                assert ("Assume normal form that " && rule.outside_attributes[0].size() == 1);
                Term<Terminal> term = boost::get<Term<Terminal>>(rule.outside_attributes[0][0][0]);
                assert (term.head == terminal);
                std::map<Variable, std::pair<Position, Position>> var_assignment;
                std::pair<Position, Position> span_assignment;
                if (match_and_retrieve_vars(input.get_previous(position), rule.outside_attributes[0][0], var_assignment, span_assignment)){
                    std::shared_ptr<ParseItem<Nonterminal, Position>> item = std::make_shared<ParseItem<Nonterminal, Position>>();
                    item->nonterminal = rule.lhn;
                    for (int j = 1; j <= rule.irank(0); ++j) {
                        auto p = var_assignment.at(Variable(0, j));
                        item->spans_inh.emplace_back(p);
                    }
                    item->spans_syn.emplace_back(span_assignment);
                    agenda.push(item);
                    trace[item].emplace_back(std::pair<Rule<Nonterminal, Terminal>, std::vector<ParseItem<Nonterminal, Position>>>(rule, std::vector<ParseItem<Nonterminal, Position>> ()));
                }
            }
        }
    }
public:
    std::queue<std::shared_ptr<ParseItem<Nonterminal, Position>>> agenda;
    std::map<std::shared_ptr<ParseItem<Nonterminal, Position>>, std::vector<std::pair<Rule<Nonterminal, Terminal>, std::vector<ParseItem<Nonterminal, Position>> > > > trace;


    void do_parse() {
        match_lexical_rules();

        std::vector<std::shared_ptr<ParseItem<Nonterminal, Terminal>>> transport;

        while (! agenda.empty()) {
            auto item_ = agenda.front();
            ParseItem<Nonterminal, Position> & item = *item_;
            agenda.pop();
            std::cout << item.nonterminal << " ( ";
            for(auto range : item.spans_inh)
                std::cout << " <" << range.first << "-" << range.second << ">";
            std::cout << " ; ";
            for(auto range : item.spans_syn)
                std::cout << " <" << range.first << "-" << range.second << ">";
            std::cout << " ) " << std::endl;
        }
    }

    bool addToChart(std::shared_ptr<ParseItem<Nonterminal, Position>>) {
        return false;
    }

    ParseItem<Nonterminal, Position> * goal = nullptr;
    HybridTree<Terminal, Position> input;


    SDCP<Nonterminal, Terminal> sDCP;
    // std::vector
};

#endif //STERMPARSER_SDCP_PARSER_H
