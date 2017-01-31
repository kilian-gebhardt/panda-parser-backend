//
// Created by Markus on 24.01.17.
//

#ifndef STERM_PARSER_LCFRS_UTIL_H
#define STERM_PARSER_LCFRS_UTIL_H

#include "LCFRS.h"
#include "LCFRS_Parser.h"
#include <vector>
#include <iostream>

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
    class LCFRSFactory{
    private:
        LHS<Nonterminal, Terminal> currentLHS;
        std::vector<TerminalOrVariable<Terminal>> argument;
        LCFRS<Nonterminal,Terminal> grammar;
        std::unique_ptr<LCFRS_Parser<Nonterminal,Terminal>> parser;
        std::map<PassiveItem<Nonterminal>,TraceItem<Nonterminal,Terminal>> trace;
        std::map<PassiveItem<Nonterminal>, unsigned long> passiveItemMap;
        std::vector<Terminal> word;

    public:
        LCFRSFactory(const Nonterminal initial):
            currentLHS{LHS<Nonterminal,Terminal>(initial)},
            argument{std::vector<TerminalOrVariable<Terminal>>()},
            grammar{LCFRS<Nonterminal,Terminal>(initial)}
        {}

        void new_rule(const Nonterminal lhsNont){
            currentLHS = LHS<Nonterminal,Terminal>(lhsNont);
            argument.clear();
        }

        void add_terminal(const Terminal term){
            argument.push_back(term);
        }

        void add_variable(const unsigned long index, const unsigned long arg){
            argument.push_back(Variable(index,arg));
        }

        void complete_argument(){
            currentLHS.add_argument(argument);
            argument.clear();

        }

        void add_rule_to_grammar(std::vector<Nonterminal> rhs, const unsigned long ruleId){
            grammar.add_rule(Rule<Nonterminal,Terminal>(currentLHS,rhs, ruleId));
        };

        void do_parse(std::vector<Terminal> w){
//std::cerr << grammar;
            word = w;
            parser = std::unique_ptr<LCFRS_Parser<Nonterminal,Terminal>>(
                new LCFRS_Parser<Nonterminal,Terminal>(grammar, word));
//std::clog << "Doing the parse";
            parser->do_parse();
            trace = parser->get_trace();
//print_top_trace(grammar, trace, word);
        }

        std::map<unsigned long
                ,std::pair<Nonterminal
                        , std::vector<std::pair<unsigned long, unsigned long>>>>
        get_passive_items_map(){
            auto result = std::map<unsigned long,std::pair<Nonterminal
                                                      , std::vector<std::pair<unsigned long, unsigned long>>>>();
            passiveItemMap = std::map<PassiveItem<Nonterminal>, unsigned long>();

            unsigned long pId{0};
            for (auto tEntry : trace){
                Nonterminal nont = tEntry.first.get_nont();
                std::vector<std::pair<unsigned long, unsigned long>> ranges{};
                for (auto range : tEntry.first.get_ranges()){
                    ranges.push_back(std::pair<unsigned long, unsigned long>(range.first, range.second));
                }
                result[pId] = std::pair<Nonterminal, std::vector<std::pair<unsigned long, unsigned long>>>
                        (nont, ranges);
                passiveItemMap[tEntry.first] = pId;

                ++pId;
            }

            return result;
        }


        std::map<unsigned long, std::vector<std::pair<unsigned long, std::vector<unsigned long>>>>
                convert_trace(){
            auto result = std::map<unsigned long, std::vector<std::pair<unsigned long, std::vector<unsigned long>>>>();

            for (auto tEntry : trace){
                const unsigned long pId = passiveItemMap[tEntry.first];
                auto newTrace = std::vector<std::pair<unsigned long, std::vector<unsigned long>>>();
                for (auto parse : tEntry.second.parses){
                    const unsigned long ruleId = parse.first->get_rule_id();
                    auto pItems = std::vector<unsigned long>();
                    for (auto ptrPassiveItem : parse.second){
                        pItems.push_back(passiveItemMap[*ptrPassiveItem]);
                    }
                    newTrace.push_back(std::pair<unsigned long, std::vector<unsigned long>>(ruleId, pItems));
                }
                result[pId] = newTrace;
            }

            return result;
        }

        Nonterminal get_initial_nont() const {
            return grammar.get_initial_nont();
        }

        std::vector<Terminal> get_word() const {
            return word;
        }

        std::pair<Nonterminal, std::vector<std::pair<unsigned long, unsigned long>>> get_initial_passive_item() const {
            return std::pair<Nonterminal, std::vector<std::pair<unsigned long, unsigned long>>>
                (grammar.get_initial_nont()
                , std::vector<std::pair<unsigned long, unsigned long>>
                    {std::pair<unsigned long, unsigned long>(0,word.size())});
        }



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

    template <typename Nonterminal, typename Terminal>
    void print_top_trace(LCFRS<Nonterminal, Terminal> grammar
            , std::map<PassiveItem<Nonterminal>,TraceItem<Nonterminal,Terminal>> trace
            , std::vector<Terminal> word)
    {
        for (auto const& parse : trace[PassiveItem<Nonterminal>(grammar.get_initial_nont()
                                        , std::vector<Range>{Range(0,word.size())})].parses){
            std::clog << "    " << *(parse.first) << ": " ;
            for(auto const& ppitem : parse.second){
                std::clog << *ppitem << ", ";
            }
            std::clog << std::endl;
        }
    }


} // namespace LCFR



#endif //STERM_PARSER_LCFRS_UTIL_H
