//
// Created by Markus on 24.01.17.
//

#ifndef STERM_PARSER_LCFRS_UTIL_H
#define STERM_PARSER_LCFRS_UTIL_H

#include "LCFRS.h"
#include <vector>

// https://stackoverflow.com/questions/236129/split-a-string-in-c
template <typename ContainerT>
void tokenize(const std::string& str, ContainerT& tokens,
              const std::string& delimiters = " ", bool trimEmpty = false)
{
    std::string::size_type pos, lastPos = 0, length = str.length();

    using value_type = typename ContainerT::value_type;
    using size_type  = typename ContainerT::size_type;

    while(lastPos < length + 1)
    {
        pos = str.find_first_of(delimiters, lastPos);
        if(pos == std::string::npos)
        {
            pos = length;
        }

        if(pos != lastPos || !trimEmpty)
            tokens.push_back(value_type(str.data()+lastPos,
                                        (size_type)pos-lastPos ));

        lastPos = pos + 1;
    }
}



namespace LCFR {


    template<typename Nonterminal, typename Terminal>
    class RuleFactory{
    private:
        LHS<Nonterminal, Terminal> currentLHS;
        std::vector<TerminalOrVariable<Terminal>> argument;
    public:
        void new_rule(const Nonterminal lhsNont){
            currentLHS = LHS<Nonterminal,Terminal>(lhsNont);
            argument.reset();
        }

        void add_terminal(const Terminal term){
            argument.push_back(term);
        }

        void add_variable(const unsigned long index, const unsigned long arg){
            argument.push_back(Variable(index,arg));
        }

        void complete_argument(){
            currentLHS.add_argument(argument);
            argument.reset();

        }

        Rule<Nonterminal,Terminal>&& get_rule(std::vector<Nonterminal> rhs){
            return Rule<Nonterminal,Terminal>(currentLHS,rhs);
        };
    };



    template<typename Nonterminal, typename Terminal>
    Rule<Nonterminal, Terminal> construct_rule(const Nonterminal nont, const std::vector<std::vector<TerminalOrVariable<Terminal>>> args, const std::vector<Nonterminal> rhs) {
        LHS<Nonterminal, Terminal> lhs(nont);
        for(auto const &arg : args)
            lhs.add_argument(arg);
        return Rule<Nonterminal, Terminal>(lhs, rhs);
    }


    LCFR::Rule<std::string, std::string> construct_rule(
            const std::string nont, const std::vector<std::string> args, const std::string rhs
    ) {
        LCFR::LHS <std::__cxx11::string, std::__cxx11::string> lhs(nont);
        for (auto const &arg : args) {
            std::vector<std::__cxx11::string> tokens;
            tokenize<std::vector<std::__cxx11::string>>(arg, tokens, " ", true);
            std::vector<LCFR::TerminalOrVariable < std::string>>
            argument;
            for (std::string s : tokens) {
                if (s[0] == 'x')
                    argument.emplace_back(LCFR::Variable{(unsigned long) s[2] - '0', (unsigned long) s[4] - '0'});
                else
                    argument.emplace_back(s);
            }
            lhs.add_argument(std::move(argument));
        }

        std::vector<std::__cxx11::string> nonterminals;
        tokenize(rhs, nonterminals, " ", true);

        return Rule<std::string, std::string>(lhs, nonterminals);
    }

} // namespace LCFR



#endif //STERM_PARSER_LCFRS_UTIL_H
