//
// Created by kilian on 18/11/16.
//

#ifndef STERMPARSER_SDCP_H
#define STERMPARSER_SDCP_H

#include <vector>
#include <boost/variant.hpp>
#include <map>
#include <iostream>
#include <term.h>
#include <assert.h>

class Variable {
public:
    int member;
    int argument;
    Variable(int member, int argument) : member(member), argument(argument) {};
    bool operator<(const Variable otherVar) const {
        return (member < otherVar.member || (member == otherVar.member && argument < otherVar.argument));
    }
    bool operator== (const Variable otherVar) const {
        return member == otherVar.member && argument == otherVar.argument;
    }
};

std::ostream &operator<<(std::ostream &os, Variable &var) {
    os << " X-" << var.member << "-" << var.argument << " ";
    return os;
}

template <typename Terminal>
class Term;

template <typename Terminal>
using TermOrVariable = typename boost::variant<Variable, Term<Terminal>>;

template <typename Terminal>
using STerm = typename std::vector<TermOrVariable<Terminal>>;

template <typename Nonterminal, typename Terminal>
class Rule;

template <typename Terminal>
class Term{
public:
    Terminal head;
    // Position in linear order
    int order;
    STerm<Terminal> children;
    Term(Terminal head) : head(head) {};
    Term() {};

    void add_variable(Variable v) {
        children.push_back(v);
    }

    void add_term(Term<Terminal> t) {
        children.push_back(t);
    }
};


template <typename Terminal>
std::ostream &operator<<(std::ostream &os, STerm<Terminal> &sterm) {
    os << " [ ";
    unsigned i = 0;
    while (i < sterm.size()) {
        os << sterm[i];
        if (i < sterm.size() - 1)
            os << " , ";
        i++;
    }
    os << " ] ";
    return os;
}

template <typename Terminal>
std::ostream &operator<<(std::ostream &os, Term<Terminal> &term) {
    os << " " << term.head;
    if (term.children.size()) {
        os << term.children;
    }
    return os;
}

template <typename Terminal>
std::ostream &operator<<(std::ostream &os, TermOrVariable<Terminal> &obj) {
    try {
        auto term = boost::get<Term<Terminal>> (obj);
        os << term;
    } catch (boost::bad_get &) {
        auto var = boost::get<Variable> (obj);
        os << var;
    }
    return os;
}


template <typename Nonterminal, typename Terminal>
class STermBuilder {
public:
    STerm<Terminal> sterm;
    STerm<Terminal> * current_position = &sterm;
    std::vector<STerm<Terminal>*> history;

    void add_var(int mem, int arg){
        if (current_position) {
//            std::cerr << "add var " << mem << " " << arg << std::endl;
            current_position->emplace_back(Variable(mem, arg));
//            std::cerr << "added var " << current_position->back() << std::endl;
        } else {
//            std::cerr << "pointer invalid" << std::endl;
        }
    }
    void add_terminal(Terminal terminal) {
        current_position->emplace_back(Term<Terminal>(terminal));
//        std::cerr << "added term " << terminal << " " << current_position->back() << std::endl;
//        std::cerr << "output sterm " << sterm << std::endl;
    }

    bool add_children() {
//        std::cerr << "added children " << std::endl;
        if (current_position->size()) {
            try {
                Term<Terminal> & term = boost::get<Term<Terminal>>(current_position->back());
                history.push_back(current_position);
                current_position = & (term.children);
                return true;
            } catch (boost::bad_get &) {
                return false;
            }
        }
        return false;
    }

    bool move_up() {
//        std::cerr << "moved up " << std::endl;
        if (history.size()) {
            current_position = history.back();
            history.pop_back();
            return true;
        }
        return false;
    }

    STerm<Terminal> get_sTerm() {
//        std::cerr << "output sterm " << sterm << std::endl;
        return sterm;
    }

    void clear() {
        sterm.clear();
        history.clear();
        current_position =  & sterm;
    }

    void add_to_rule(Rule<Nonterminal, Terminal> * rule) {
//        std::cerr << "adding sterm to rule " << sterm << std::endl;
        rule->add_outside_attribute(sterm);
    }
};

template <typename Nonterminal, typename Terminal>
class Rule {
public:
    Nonterminal lhn;
    std::vector<Nonterminal> rhs;
    std::vector<std::vector<STerm<Terminal>>> outside_attributes;
    int id;
    int irank(int nont_idx) const;
    int srank(int nont_idx) const;

    std::pair<bool, Terminal> first_terminal() {
        for (auto attributes: outside_attributes) {
            for (STerm<Terminal> sterm : attributes) {
                for (TermOrVariable<Terminal> obj : sterm) {
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

    Rule() {}
    Rule(Nonterminal lhn) : lhn(lhn) {}

    void add_nonterminal(Nonterminal nonterminal){
        rhs.push_back(nonterminal);
    }

    void add_outside_attribute(STerm<Terminal> sterm){
        outside_attributes.back().push_back(sterm);
    }

    void add_sterm_from_builder(STermBuilder<Nonterminal, Terminal> & builder) {
        STerm<Terminal> sterm = builder.get_sTerm();
        std::cerr << sterm << std::endl;
        add_outside_attribute(sterm);
    }

    void next_outside_attribute(){
        outside_attributes.push_back(std::vector<STerm<Terminal>> ());
    }

    void set_id(int id) {
        this->id = id;
    }

    int get_id() {
       return id;
    }
};


template <typename Nonterminal, typename Terminal>
std::ostream &operator<<(std::ostream &os, Rule<Nonterminal, Terminal> & rule) {
    os << rule.lhn;
    int i = 0;
    for (auto attributes : rule.outside_attributes) {
        if (i > 0)
            os << " " << rule.rhs[i - 1] << " ";
        os << " ⟨ ";
        int j = 0;
        for (auto sterm : attributes) {
            if (j++)
                os << " , ";
            os << sterm;
        }
        os << " ⟩ ";
        if (i == 0)
            os << " -> ";
        i++;
    }
    return os;
}


template <typename Nonterminal, typename Terminal>
class SDCP {
public:
    Nonterminal initial;
    std::map<Nonterminal, std::vector<Rule<Nonterminal, Terminal>>> lhn_to_rule;
    std::map<Nonterminal, std::vector<Rule<Nonterminal, Terminal>>> left_nont_corner;
    std::map<Nonterminal, std::vector<std::pair<Rule<Nonterminal, Terminal>, int>>> nont_corner;
    std::map<Terminal, std::vector<Rule<Nonterminal, Terminal>>> axioms;
    std::vector<Rule<Nonterminal, Terminal>> epsilon_axioms;
    std::map<Nonterminal, int> irank, srank;

    bool add_rule(Rule<Nonterminal, Terminal> rule);

     bool set_initial(Nonterminal nonterminal);
    void output(){
        std::cerr << *this << std::endl;
    }
};

template <typename Nonterminal, typename Terminal>
std::ostream &operator<<(std::ostream &os, SDCP<Nonterminal, Terminal> & sdcp) {
    os << "initial " << sdcp.initial << std::endl;
    for (auto p : sdcp.lhn_to_rule) {
        for (auto rule : p.second)
            os << rule << std::endl;
    }
    return os;
}



template <typename Terminal>
void countVariableRecursive(int idx, int & counter, STerm<Terminal> sterm) {
    for (TermOrVariable<Terminal> obj : sterm) {
        try {
            Variable v = boost::get<Variable>(obj);
            if (v.member == idx)
                counter++;
            // std::cout << v.member << " " << v.argument << std::endl;
        }
        catch (boost::bad_get &) {
            Term<Terminal> t = boost::get<Term<Terminal>>(obj);
            countVariableRecursive(idx, counter, t.children);
        }
    }
}

template <typename Nonterminal, typename Terminal>
int countVariable(int idx, const Rule<Nonterminal, Terminal> & rule) {
    int counter = 0;
    for (auto member_attributes: rule.outside_attributes) {
        for (auto attribute : member_attributes) {
            countVariableRecursive<Terminal>(idx, counter, attribute);
        }
    }
    return counter;
}

template <typename Nonterminal, typename Terminal>
int Rule<Nonterminal, Terminal>::irank(int nont_idx) const {
    if (nont_idx == 0)
        return countVariable<Nonterminal, Terminal>(0, *this);
    else {
        if (nont_idx >= outside_attributes.size())
            return 0;
        return outside_attributes[nont_idx].size();
    }
}

template <typename Nonterminal, typename Terminal>
int Rule<Nonterminal, Terminal>::srank(int nont_idx) const {
    if (nont_idx == 0)
        return outside_attributes[0].size();
    return countVariable<Nonterminal, Terminal>(nont_idx, *this);
}

template <typename Nonterminal, typename Terminal>
bool SDCP<Nonterminal, Terminal>::add_rule(Rule<Nonterminal, Terminal> rule) {
    assert (rule.rhs.size() == rule.outside_attributes.size() - 1);
    // Checking that iranks and sranks match
    try {
        if (irank.at(rule.lhn) != rule.irank(0) ||
            srank.at(rule.lhn) != rule.srank(0) ) {
//            std::cerr << rule.lhn << " " << irank.at(rule.lhn) << " " << rule.irank(0) << std::endl;
//            std::cerr << rule.lhn << " " << srank.at(rule.lhn) << " " << rule.srank(0) << std::endl;
            return false;
        }
    }
        // new nonterminal
    catch  (const std::out_of_range&){
        irank[rule.lhn] = rule.irank(0);
        srank[rule.lhn] = rule.srank(0);
//        std::cerr << rule.lhn << " " << irank.at(rule.lhn) << " " << rule.irank(0) << std::endl;
//        std::cerr << rule.lhn << " " << srank.at(rule.lhn) << " " << rule.srank(0) << std::endl;
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
//            std::cerr << nonterminal << " " << irank.at(nonterminal) << " " << rule.irank(i) << std::endl;
//            std::cerr << nonterminal << " " << srank.at(nonterminal) << " " << rule.srank(i) << std::endl;
            lhn_to_rule[nonterminal] = std::vector<Rule<Nonterminal, Terminal>> ();
        }
        ++i;
    }

    // finally adding rule
    lhn_to_rule[rule.lhn].push_back(rule);
    if (rule.rhs.size() > 0) {
        left_nont_corner[rule.rhs[0]].push_back(rule);
        int j = 0;
        for (Nonterminal nont : rule.rhs)
            nont_corner[nont].push_back(std::make_pair(rule, j++));
    }
    else {
        auto term = rule.first_terminal();
        if (term.first)
            axioms[term.second].push_back(rule);
        else
            epsilon_axioms.push_back(rule);
    }

//    std::cerr << "added rule " << rule << std::endl;

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
