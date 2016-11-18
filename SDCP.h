//
// Created by kilian on 18/11/16.
//

#ifndef STERMPARSER_SDCP_H
#define STERMPARSER_SDCP_H

#include <vector>
#include <boost/variant.hpp>
#include <map>

class Variable {
public:
    int member;
    int argument;
    Variable(int member, int argument) : member(member), argument(argument) {};
};

template <typename Terminal>
class Term;

template <typename Terminal>
using TermOrVariable = typename boost::variant<Variable, Term<Terminal>>;

template <typename Terminal>
using STerm = typename std::vector<TermOrVariable<Terminal>>;

template <typename Terminal>
class Term{
public:
    Terminal head;
    // Position in linear order
    int order;
    STerm<Terminal> children;
    Term(Terminal head) : head(head) {};
    Term() {};
};

template <typename Nonterminal, typename Terminal>
class Rule {
public:
    Nonterminal lhn;
    std::vector<Nonterminal> rhs;
    std::vector<std::vector<STerm<Terminal>>> outside_attributes;
    int irank(int nont_idx) const;
    int srank(int nont_idx) const;

    std::pair<bool, Terminal> first_terminal() {
        for (auto attributes: outside_attributes) {
            for (auto attribute : attributes) {
                for (TermOrVariable<Terminal> obj : attribute) {
                    try {
                        Term<Terminal> t = boost::get<Term<Terminal>>(obj);
                        return std::pair<bool, Terminal>(true, t.head);
                    }
                    catch (boost::bad_get&) {}
                }
            }
        }
        return std::make_pair<bool, Terminal>(false, nullptr);
    }
};

template <typename Nonterminal, typename Terminal>
class SDCP {
public:
    Nonterminal initial;
    std::map<Nonterminal, std::vector<Rule<Nonterminal, Terminal>>> lhn_to_rule;
    std::map<Nonterminal, std::vector<Rule<Nonterminal, Terminal>>> left_nont_corner;
    std::map<Terminal, std::vector<Rule<Nonterminal, Terminal>>> axioms;
    std::vector<Rule<Nonterminal, Terminal>> epsilon_axioms;
    std::map<Nonterminal, int> irank, srank;

    bool add_rule(Rule<Nonterminal, Terminal> rule);

    bool set_initial(Nonterminal nonterminal);
};

template <typename Nonterminal, typename Terminal>
int countVariable(int idx, const Rule<Nonterminal, Terminal> & rule) {
    int counter = 0;
    for (auto member_attributes: rule.outside_attributes) {
        for (auto attribute : member_attributes) {
            // TODO terms need to be traversed recursively
            for (TermOrVariable<Terminal> obj : attribute){
                try {
                    Variable v = boost::get<Variable>(obj);
                    if (v.member == idx)
                        counter++;
                    std::cout << v.member << " " << v.argument << std::endl;
                }
                catch (boost::bad_get&) {}
            }
        }
    }
    return counter;
}

template <typename Nonterminal, typename Terminal>
int Rule<Nonterminal, Terminal>::irank(int nont_idx) const {
    if (nont_idx == 0)
        return countVariable<Nonterminal, Terminal>(0, *this);
    else
        return outside_attributes[nont_idx].size();
}

template <typename Nonterminal, typename Terminal>
int Rule<Nonterminal, Terminal>::srank(int nont_idx) const {
    if (nont_idx == 0)
        return outside_attributes[0].size();
    return countVariable<Nonterminal, Terminal>(nont_idx, *this);
}

template <typename Nonterminal, typename Terminal>
bool SDCP<Nonterminal, Terminal>::add_rule(Rule<Nonterminal, Terminal> rule) {
    // Checking that iranks and sranks match
    try {
        if (irank.at(rule.lhn) != rule.irank(0) ||
            srank.at(rule.lhn) != rule.srank(0) )
            return false;

    }
        // new nonterminal
    catch  (const std::out_of_range&){
        irank[rule.lhn] = rule.irank(0);
        srank[rule.lhn] = rule.srank(0);
        lhn_to_rule[rule.lhn] = std::vector<Rule<Nonterminal, Terminal>> ();
    }
    auto i = 1;
    for (Nonterminal nonterminal : rule.rhs) {
        try {
            if (irank.at(nonterminal) != rule.irank(i) ||
                srank.at(nonterminal) != rule.srank(i))
                return false;

        }
            // new nonterminal
        catch  (const std::out_of_range&){
            irank[nonterminal] = rule.irank(i);
            srank[nonterminal] = rule.srank(i);
            lhn_to_rule[nonterminal] = std::vector<Rule<Nonterminal, Terminal>> ();
        }
        ++i;
    }

    // finally adding rule
    lhn_to_rule[rule.lhn].push_back(rule);
    if (rule.rhs.size() > 0)
        left_nont_corner[rule.rhs[0]].push_back(rule);
    else {
        auto term = rule.first_terminal();
        if (term.first)
            axioms[term.second].push_back(rule);
        else
            epsilon_axioms.push_back(rule);
    }

    return true;
}

template <typename Nonterminal, typename Terminal>
bool SDCP<Nonterminal, Terminal>::set_initial(Nonterminal nonterminal) {
    // Checking that iranks and sranks match
    try {
        if (irank.at(nonterminal) != 0 ||
            srank.at(nonterminal) != 1 )
            return false;
    }
        // new nonterminal
    catch  (const std::out_of_range&){
        irank[nonterminal] = 0;
        srank[nonterminal] = 1;
        lhn_to_rule[nonterminal] = std::vector<Rule<Nonterminal, Terminal>> ();
    }
    initial = nonterminal;
    return true;
}

#endif //STERMPARSER_SDCP_H
