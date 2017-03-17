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
#include "../util.h"

namespace DCP {
    class Variable {
    public:
        int member;
        int argument;

        Variable(int member, int argument) : member(member), argument(argument) {};

        bool operator<(const Variable otherVar) const {
            return (member < otherVar.member || (member == otherVar.member && argument < otherVar.argument));
        }

        bool operator==(const Variable otherVar) const {
            return member == otherVar.member && argument == otherVar.argument;
        }
    };

    std::ostream &operator<<(std::ostream &os, Variable &var) {
        os << " X-" << var.member << "-" << var.argument << " ";
        return os;
    }

    template<typename Terminal>
    class Term;

    template<typename Terminal>
    using TermOrVariable = typename boost::variant<Variable, Term<Terminal>>;

    template<typename Terminal>
    using STerm = typename std::vector<TermOrVariable<Terminal>>;

    template<typename Nonterminal, typename Terminal>
    class Rule;

    template<typename Terminal>
    class Term {
    public:
        Terminal head;
        // Position in linear order
        int order = -1;
        STerm<Terminal> children;

        Term(Terminal head) : head(head), order(-1) {};

        Term(Terminal head, int order) : head(head), order(order) {};

        Term() {};

        void add_variable(Variable v) {
            children.push_back(v);
        }

        void add_term(Term<Terminal> t) {
            children.push_back(t);
        }

        bool is_ordered() const {
            return order > -1;
        }
    };

    template<typename Terminal>
    std::ostream &operator<<(std::ostream &os, std::vector<boost::variant<Terminal, Variable>> &word_function) {
        os << " [ ";
        int i = 0;
        for (auto obj : word_function) {
            if (i++)
                os << " , ";
            try {
                auto term = boost::get<Terminal>(obj);
                os << term;
            } catch (boost::bad_get &) {
                auto var = boost::get<Variable>(obj);
                os << var;
            }
        }
        os << " ] ";
        return os;
    }

    template<typename Terminal>
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

    template<typename Terminal>
    std::ostream &operator<<(std::ostream &os, Term<Terminal> &term) {
        os << " " << term.head;
        if (term.children.size()) {
            os << term.children;
        }
        return os;
    }

    template<typename Terminal>
    std::ostream &operator<<(std::ostream &os, TermOrVariable<Terminal> &obj) {
        try {
            auto term = boost::get<Term<Terminal>>(obj);
            os << term;
        } catch (boost::bad_get &) {
            auto var = boost::get<Variable>(obj);
            os << var;
        }
        return os;
    }


    template<typename Nonterminal, typename Terminal>
    class STermBuilder {
    public:
        STerm<Terminal> sterm;
        STerm<Terminal> *current_position = &sterm;
        std::vector<STerm<Terminal> *> history;

        void add_var(int mem, int arg) {
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

        void add_terminal(Terminal terminal, int position) {
            current_position->emplace_back(Term<Terminal>(terminal, position));
//        std::cerr << "added term " << terminal << " " << current_position->back() << std::endl;
//        std::cerr << "output sterm " << sterm << std::endl;
        }

        bool add_children() {
//        std::cerr << "added children " << std::endl;
            if (current_position->size()) {
                try {
                    Term<Terminal> &term = boost::get<Term<Terminal>>(current_position->back());
                    history.push_back(current_position);
                    current_position = &(term.children);
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

        STerm<Terminal> &get_sTerm() {
//        std::cerr << "output sterm " << sterm << std::endl;
            return sterm;
        }

        void clear() {
            sterm.clear();
            history.clear();
            current_position = &sterm;
        }

        void add_to_rule(Rule<Nonterminal, Terminal> *rule) {
//        std::cerr << "adding sterm to rule " << sterm << std::endl;
            rule->add_inside_attribute(sterm);
        }
    };

    template<typename Nonterminal, typename Terminal>
    class Rule {
    public:
        Nonterminal lhn;
        std::vector<Nonterminal> rhs;
        std::vector<std::vector<STerm<Terminal>>> inside_attributes;
        std::vector<std::vector<boost::variant<Terminal, Variable>>> word_function;

        int id = 0;

        int irank(int nont_idx) const;

        int srank(int nont_idx) const;

        int fanout(int nont_idx) const;

        std::pair<bool, Terminal> first_terminal() {
            for (auto attributes: inside_attributes) {
                for (STerm<Terminal> sterm : attributes) {
                    for (TermOrVariable<Terminal> obj : sterm) {
                        try {
                            Term<Terminal> t = boost::get<Term<Terminal>>(obj);
                            return std::pair<bool, Terminal>(true, t.head);
                        }
                        catch (boost::bad_get &) {}
                    }
                }
            }
            Terminal terminal;
            return std::pair<bool, Terminal>(false, terminal);
        }

        Rule() {}

        Rule(Nonterminal lhn) : lhn(lhn) {}

        void add_nonterminal(Nonterminal nonterminal) {
            rhs.push_back(nonterminal);
        }

        void add_inside_attribute(STerm<Terminal> sterm) {
            inside_attributes.back().push_back(sterm);
        }

        void add_sterm_from_builder(STermBuilder<Nonterminal, Terminal> &builder) {
            STerm<Terminal> &sterm = builder.get_sTerm();
            // std::cerr << sterm << std::endl;
            add_inside_attribute(sterm);
        }

        void next_inside_attribute() {
            inside_attributes.push_back(std::vector<STerm<Terminal>>());
        }

        void next_word_function_argument() {
            word_function.push_back(std::vector<boost::variant<Terminal, Variable>>());
        }

        void add_var_to_word_function(int mem, int arg) {
            word_function.back().emplace_back(Variable(mem, arg));
        }

        void add_terminal_to_word_function(Terminal terminal) {
            word_function.back().emplace_back(terminal);
        }

        void set_id(int id) {
            this->id = id;
        }

        int get_id() {
            return id;
        }

        void collect_variables(const STerm<Terminal> &sterm, std::vector<Variable> &sdcp_vars) {
            for (TermOrVariable<Terminal> obj : sterm) {
                try {
                    Variable v = boost::get<Variable>(obj);
                    sdcp_vars.push_back(v);
                }
                catch (boost::bad_get &) {
                    Term<Terminal> t = boost::get<Term<Terminal>>(obj);
                    collect_variables(t.children, sdcp_vars);
                }
            }
        }

        bool single_syntactic_use() {
            std::vector<Variable> sdcp_vars, lcfrs_vars;

            for (auto attribute : inside_attributes)
                for (const STerm<Terminal> &sterm : attribute)
                    collect_variables(sterm, sdcp_vars);

            if (!pairwise_different(sdcp_vars))
                return false;

            for (auto argument : word_function)
                for (const auto &obj : argument)
                    try {
                        Variable v = boost::get<Variable>(obj);
                        lcfrs_vars.push_back(v);
                    }
                    catch (boost::bad_get &) {
                        // Terminal t = boost::get<Terminal>(obj);
                    }

            return pairwise_different(lcfrs_vars);
        }


        // attributes of the sDCP may not be empty_trace
        bool verify_sdcp_restrictions_recursive(
                const STerm<Terminal> &sterm
                , std::vector<bool> &lcfrs_terminals
                , unsigned mem
                , bool root
        ) {
            bool lhn_var = false;
            for (const TermOrVariable<Terminal> &obj : sterm) {
                try {
                    const Variable &v = boost::get<Variable>(obj);
                    if (v.member == 0) {
                        // a lhn_var must be followed by a non_lhn var
                        if (lhn_var)
                            return false;
                        lhn_var = true;
                        // a lhn_var may occur at root level only in the inherent attribute of an rhs nonterminal
                        // TODO could be softened a bit to "a lhn_var can occur at root level,
                        // TODO if it is followed and succeeded by a rhs_var which sufficiently constrain it"
                        if (root && mem == 0)
                            return false;
                    } else
                        lhn_var = false;
                }
                catch (boost::bad_get &) {
                    const Term<Terminal> &t = boost::get<Term<Terminal>>(obj);
                    if (!verify_sdcp_restrictions_recursive(t.children, lcfrs_terminals, mem, false))
                        return false;

                    if (t.is_ordered()) {
                        if (lcfrs_terminals.size() < t.order + 1)
                            lcfrs_terminals.resize(t.order + 1, false);
                        if (lcfrs_terminals[t.order])
                            // only one sDCP symbol may link to the same string position
                            return false;
                        lcfrs_terminals[t.order] = true;
                    }
                }
            }
            return true;
        }

        bool verify_grammar_restrictions() {
            std::vector<bool> lcfrs_terminals;

            unsigned mem = 0;
            for (auto attributes : inside_attributes) {
                // exactly one synthesized attribute for lhn, if rhs is empty_trace
                if (rhs.size() == 0 && mem == 0 && attributes.size() != 1)
                    return false;
                for (auto sterm : attributes) {
                    if (!verify_sdcp_restrictions_recursive(sterm, lcfrs_terminals, mem, true))
                        return false;
                }
                ++mem;
            }

            unsigned i = 0;
            for (auto argument : word_function) {

                // the LCFRS may not have ε arguments
                if (!argument.size())
                    return false;

                for (auto obj : argument)
                    try {
                        boost::get<Variable>(obj);
                    }
                    catch (boost::bad_get &) {
                        boost::get<Terminal>(obj);

                        // all terminals in the LCFRS may be linked
                        if (lcfrs_terminals.size() <= i || !lcfrs_terminals[i])
                            return false;
                        ++i;
                    }
            }
            return true;
        }

    };


    template<typename Nonterminal, typename Terminal>
    std::ostream &operator<<(std::ostream &os, const Rule<Nonterminal, Terminal> &rule) {
        os << "id="<< rule.id << ": " << rule.lhn;
        int i = 0;
        for (auto attributes : rule.inside_attributes) {
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
        if (rule.word_function.size() > 0) {
            os << " | ⟨ ";
            i = 0;
            for (auto arg : rule.word_function) {
                if (i++)
                    os << " , ";
                os << arg;
            }
            os << " ⟩ ";
        }
        return os;
    }


    template<typename Nonterminal, typename Terminal>
    class SDCP {
    private:
        std::map<Nonterminal, std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>>> left_nont_corner;
        std::map<Nonterminal, std::vector<std::pair<std::shared_ptr<Rule<Nonterminal, Terminal>>, int>>> nont_corner;
        std::map<Terminal, std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>>> axioms;
        std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>> epsilon_axioms;
        std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>> rules;
    public:
        std::map<Nonterminal, std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>>> lhn_to_rule;
        Nonterminal initial;
        std::map<Nonterminal, int> irank, srank, fanout;

        bool add_rule(Rule<Nonterminal, Terminal> rule);

        bool set_initial(Nonterminal nonterminal);

        std::shared_ptr<Rule<Nonterminal, Terminal>> &get_rule_by_id(unsigned id) {
            return rules[id];
        };

        void output() {
            std::cerr << *this << std::endl;
        }

        const std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>> &get_lhn_to_rule(Nonterminal nonterminal) {
            return lhn_to_rule[nonterminal];
        };

        const std::vector<std::pair<std::shared_ptr<Rule<Nonterminal, Terminal>>, int>> &
        get_nont_corner(Nonterminal nonterminal) {
            return nont_corner[nonterminal];
        };

        const std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>> &get_axioms(Terminal terminal) {
            return axioms[terminal];
        };
    };

    template<typename Nonterminal, typename Terminal>
    std::ostream &operator<<(std::ostream &os, SDCP<Nonterminal, Terminal> &sdcp) {
        os << "initial " << sdcp.initial << std::endl;
        for (auto p : sdcp.lhn_to_rule) {
            for (auto rule : p.second)
                os << *rule << std::endl;
        }
        return os;
    }


    template<typename Terminal>
    void countVariableRecursive(int idx, int &counter, STerm<Terminal> sterm) {
        for (TermOrVariable<Terminal> obj : sterm) {
            if (obj.type() == typeid(Variable)) {
                Variable v = boost::get<Variable>(obj);
                if (v.member == idx)
                    counter++;
                // std::cout << v.member << " " << v.argument << std::endl;
            } else {
                Term<Terminal> t = boost::get<Term<Terminal>>(obj);
                countVariableRecursive(idx, counter, t.children);
            }
        }
    }

    template<typename Nonterminal, typename Terminal>
    int countVariable(int idx, const Rule<Nonterminal, Terminal> &rule) {
        int counter = 0;
        for (auto member_attributes: rule.inside_attributes) {
            for (auto attribute : member_attributes) {
                countVariableRecursive<Terminal>(idx, counter, attribute);
            }
        }
        return counter;
    }

    template<typename Nonterminal, typename Terminal>
    int Rule<Nonterminal, Terminal>::irank(int nont_idx) const {
        if (nont_idx == 0)
            return countVariable<Nonterminal, Terminal>(0, *this);
        else {
            if (nont_idx >= inside_attributes.size())
                return 0;
            return inside_attributes[nont_idx].size();
        }
    }

    template<typename Nonterminal, typename Terminal>
    int Rule<Nonterminal, Terminal>::srank(int nont_idx) const {
        if (nont_idx == 0)
            return inside_attributes[0].size();
        return countVariable<Nonterminal, Terminal>(nont_idx, *this);
    }

    template<typename Nonterminal, typename Terminal>
    int Rule<Nonterminal, Terminal>::fanout(int nont_idx) const {
        if (nont_idx == 0)
            return (int) word_function.size();
        int counter = 0;
        for (auto arg : word_function) {
            for (auto obj : arg) {
                if (obj.type() == typeid(Variable)) {
                    Variable v = boost::get<Variable>(obj);
                    if (v.member == nont_idx)
                        counter++;
                }
            }
        }
        return counter;
    };

    template<typename Nonterminal, typename Terminal>
    bool SDCP<Nonterminal, Terminal>::add_rule(Rule<Nonterminal, Terminal> rule) {
        // basic sanitiy checks
        if (!rule.rhs.size() == rule.inside_attributes.size() - 1)
            return false;

        if (!rule.single_syntactic_use())
            return false;

        if (!rule.verify_grammar_restrictions())
            return false;

        // Checking that iranks, sranks, and fanouts match globally
        try {
            if (irank.at(rule.lhn) != rule.irank(0) ||
                srank.at(rule.lhn) != rule.srank(0) ||
                fanout.at(rule.lhn) != rule.fanout(0)) {
//            std::cerr << rule.lhn << " " << irank.at(rule.lhn) << " " << rule.irank(0) << std::endl;
//            std::cerr << rule.lhn << " " << srank.at(rule.lhn) << " " << rule.srank(0) << std::endl;
//            std::cerr << rule.lhn << " " << fanout.at(rule.lhn) << " " << rule.fanout(0) << std::endl;
                return false;
            }
        }
            // new nonterminal
        catch (const std::out_of_range &) {
            irank[rule.lhn] = rule.irank(0);
            srank[rule.lhn] = rule.srank(0);
            fanout[rule.lhn] = rule.fanout(0);
//        std::cerr << rule.lhn << " " << irank.at(rule.lhn) << " " << rule.irank(0) << std::endl;
//        std::cerr << rule.lhn << " " << srank.at(rule.lhn) << " " << rule.srank(0) << std::endl;
//        std::cerr << rule.lhn << " " << fanout.at(rule.lhn) << " " << rule.fanout(0) << std::endl;
            lhn_to_rule[rule.lhn] = std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>>();
        }
        auto i = 1;
        for (Nonterminal nonterminal : rule.rhs) {
            try {
                if (irank.at(nonterminal) != rule.irank(i) ||
                    srank.at(nonterminal) != rule.srank(i) ||
                    fanout.at(nonterminal) != rule.fanout(i))
                    return false;

            }
                // new nonterminal
            catch (const std::out_of_range &) {
                irank[nonterminal] = rule.irank(i);
                srank[nonterminal] = rule.srank(i);
                fanout[nonterminal] = rule.fanout(i);
//            std::cerr << nonterminal << " " << irank.at(nonterminal) << " " << rule.irank(i) << std::endl;
//            std::cerr << nonterminal << " " << srank.at(nonterminal) << " " << rule.srank(i) << std::endl;
//            std::cerr << nonterminal << " " << fanout.at(nonterminal) << " " << rule.fanout(i) << std::endl;
                lhn_to_rule[nonterminal] = std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>>();
            }
            ++i;
        }

        auto rule_ptr = std::make_shared<Rule<Nonterminal, Terminal>>(rule);
        // finally adding rule
        lhn_to_rule[rule.lhn].push_back(rule_ptr);
        if (rule.rhs.size() > 0) {
            left_nont_corner[rule.rhs[0]].push_back(rule_ptr);
            int j = 0;
            for (Nonterminal nont : rule.rhs)
                nont_corner[nont].push_back(std::make_pair(rule_ptr, j++));
        } else {
            auto term = rule.first_terminal();
            if (term.first)
                axioms[term.second].push_back(rule_ptr);
            else
                epsilon_axioms.push_back(rule_ptr);
        }
        if (rules.size() <= rule.id) {
            rules.resize(rule.id + 1);
            rules[rule.id] = rule_ptr;
        }

//    std::cerr << "added rule " << rule << std::endl;

        return true;
    }

    template<typename Nonterminal, typename Terminal>
    bool SDCP<Nonterminal, Terminal>::set_initial(Nonterminal nonterminal) {
        // Checking that iranks and sranks match
        try {
            if (irank.at(nonterminal) != 0 ||
                srank.at(nonterminal) != 1)
                return false;
        }
            // new nonterminal
        catch (const std::out_of_range &) {
            irank[nonterminal] = 0;
            srank[nonterminal] = 1;
            lhn_to_rule[nonterminal] = std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>>();
        }
        initial = nonterminal;
        return true;
    }
} // end namespace DCP

#endif //STERMPARSER_SDCP_H
