//
// Created by Markus on 19.01.17.
//

#ifndef STERM_PARSER_LCFRS_H
#define STERM_PARSER_LCFRS_H

#include <vector>
#include <boost/variant.hpp>
#include <map>
#include "LCFRS_util.h"


namespace LCFR {

    class Variable {
    private:
        unsigned long index, arg;
    public:
        Variable(unsigned long ind, unsigned long arg) : index{ind}, arg{arg} {};

        unsigned long get_index() const {
            return index;
        }

        unsigned long get_arg() const {
            return arg;
        };

        bool operator ==(const Variable& v) const{
            return(v.index == index && v.arg == arg);
        }

        template <typename Nonterminal1, typename Terminal1>
        friend std::ostream& operator <<(std::ostream &, const Variable&);
    };

    std::ostream& operator <<(std::ostream& o, const Variable& v) {
        o << "x<" << v.get_index() << "," << v.get_arg() << ">";
        return o;
    }



    template<typename Terminal>
    using TerminalOrVariable = typename boost::variant<Terminal, Variable>;


    template<typename Nonterminal, typename Terminal>
    class LHS {
    private:
        Nonterminal nont;
        std::vector<std::vector<TerminalOrVariable<Terminal>>> args;
    public:
        LHS(Nonterminal n) : nont(n) {
            args = std::vector<std::vector<TerminalOrVariable<Terminal>>>();
        };

        Nonterminal get_nont() const {
            return nont;
        };

        std::vector<std::vector<TerminalOrVariable<Terminal>>> get_args() const {
            return args;
        }

        void addArgument(std::vector<TerminalOrVariable<Terminal>> arg) {
            args.push_back(arg);
        };
    };


    template<typename Nonterminal, typename Terminal>
    class Rule {
    private:
        LHS<Nonterminal, Terminal> lhs;
        std::vector<Nonterminal> rhs;
    public:

        Rule(LHS<Nonterminal, Terminal> l, std::vector<Nonterminal> r) : lhs(l), rhs(r) {};

        LHS<Nonterminal, Terminal> get_lhs() const {
            return lhs;
        };

        std::vector<Nonterminal> get_rhs() const {
            return rhs;
        }

        friend std::ostream& operator <<(std::ostream& o, const Rule<Nonterminal, Terminal>& r) {
            LHS<Nonterminal, Terminal> lhs{r.get_lhs()};
            o << lhs.get_nont();
            o << "( ";
            bool first = true;
            for (auto arg : lhs.get_args()) {
                if(!first){
                    o << ", ";
                }
                first=false;
                for (auto a : arg) {
                    o << a << " ";
                }
            }
            o << ")";
            o << " -> ";
            for (auto rhs : r.get_rhs()){
                o << rhs;
            }
            return o;
        }
    };




    template<typename Nonterminal, typename Terminal>
    class LCFRS {
    private:
        std::string name;
        Nonterminal initial_nont;
        std::map<Nonterminal, std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>>> rules;
    public:
        LCFRS(Nonterminal initial): initial_nont(initial) {};

        LCFRS(Nonterminal initial, std::string gr) : name(gr), initial_nont(initial) {};

        const std::map<Nonterminal, std::vector<std::shared_ptr<Rule<Nonterminal, Terminal>>>>& get_rules() const {
            return rules;
        }

        const Nonterminal get_initial_nont() const {
            return initial_nont;
        }

        void add_rule(Rule<Nonterminal, Terminal>&& r) {
            Nonterminal nont = r.get_lhs().get_nont();
            rules[nont].emplace_back(std::make_shared<Rule<Nonterminal,Terminal>>(std::move(r)));
        }

        friend std::ostream& operator <<(std::ostream& o, const LCFRS<Nonterminal, Terminal>& grammar) {
            o << "Grammar: " << grammar.name  << " (initial: " << grammar.initial_nont << ")" << std::endl;
            for (auto rs : grammar.get_rules()) {
                for (auto r : rs.second)
                    o << "    " << *r << std::endl;
            }
            return o;
        }

    };




    // Helper funtions

    template <typename  Nonterminal, typename Terminal>
    Rule<Nonterminal, Terminal> constructRule(
            const Nonterminal nont
            , const std::vector<std::vector<TerminalOrVariable<Terminal>>> args
            , const std::vector<Nonterminal> rhs
    ){
        LHS<Nonterminal,Terminal> lhs(nont);
        for(auto const& arg : args)
            lhs.addArgument(arg);
        return Rule<Nonterminal, Terminal>(lhs, rhs);
    }

    Rule<std::string, std::string> constructRule(
            const std::string nont
            , const std::vector<std::string> args
            , const std::string rhs
    ){
        LHS<std::string,std::string> lhs(nont);
        for (auto const& arg : args) {
            std::vector<std::string> tokens;
            tokenize<std::vector<std::string>>(arg, tokens);
            std::vector<TerminalOrVariable<std::string>> argument;
            for (std::string s : tokens){
                if(s[0] == 'x')
                    argument.emplace_back(Variable{(unsigned long)s[2]-'0',(unsigned long)s[4]-'0'});
                else
                    argument.emplace_back(s);
            }
            lhs.addArgument(std::move(argument));
        }

        std::vector<std::string> nonterminals;
        tokenize(rhs, nonterminals);

        return Rule<std::string, std::string>(lhs, nonterminals);
    };
}



#endif //STERM_PARSER_LCFRS_H
