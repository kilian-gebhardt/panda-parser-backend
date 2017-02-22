//
// Created by Markus on 19.01.17.
//

#ifndef STERM_PARSER_LCFRS_H
#define STERM_PARSER_LCFRS_H

#include <vector>
#include <boost/variant.hpp>
#include <map>


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

        const Nonterminal& get_nont() const {
            return nont;
        };

        const std::vector<std::vector<TerminalOrVariable<Terminal>>>& get_args() const {
            return args;
        }

        void add_argument(std::vector<TerminalOrVariable<Terminal>>& arg) {
            args.push_back(arg);
        };

        void add_argument(std::vector<TerminalOrVariable<Terminal>>&& arg) {
            args.emplace_back(std::move(arg));
        };
    };


    template<typename Nonterminal, typename Terminal>
    class Rule {
    private:
        LHS<Nonterminal, Terminal> lhs;
        std::vector<Nonterminal> rhs;
        unsigned long ruleId;
    public:

        Rule(LHS<Nonterminal, Terminal> l
                , std::vector<Nonterminal> r
                , unsigned long rId = 0)
        : lhs(l), rhs(r), ruleId(rId) {};

        const LHS<Nonterminal, Terminal>& get_lhs() const {
            return lhs;
        };

        const std::vector<Nonterminal>& get_rhs() const {
            return rhs;
        }

        unsigned long get_rule_id() const {
            return ruleId;
        }

        friend std::ostream& operator <<(std::ostream& o, const Rule<Nonterminal, Terminal>& r) {
            LHS<Nonterminal, Terminal> lhs{r.get_lhs()};
            o << r.get_rule_id() << ": ";
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

        const Nonterminal& get_initial_nont() const {
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

}



#endif //STERM_PARSER_LCFRS_H
