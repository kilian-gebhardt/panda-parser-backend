//
// Created by Markus on 19.01.17.
//

#ifndef STERM_PARSER_LCFRS_PARSER_H
#define STERM_PARSER_LCFRS_PARSER_H

#include <map>
#include "LCFRS.h"

namespace LCFR{

    typedef std::pair<unsigned int, unsigned int> Range;
    std::ostream& operator <<(std::ostream& o, const Range& r){
        o << "(" << r.first << "," << r.second << ")";
        return o;
    }

    template <typename Nonterminal, typename Terminal>
    class PassiveItem {
    private:
        Nonterminal nont;
        std::vector<Range> ranges;
    public:
        PassiveItem(PassiveItem<Nonterminal, Terminal>&& pItem){
            nont = pItem.get_nont();
            ranges = pItem.get_ranges();
        }

        PassiveItem(Nonterminal n, std::vector<Range> rs): nont(n), ranges(rs){}

        Nonterminal get_nont() const {
            return nont;
        }
        std::vector<Range> get_ranges() const {
                return ranges;
        }

        template <typename Nonterminal1, typename Terminal1>
        friend std::ostream& operator <<(std::ostream&, const PassiveItem<Nonterminal1, Terminal1>&);
    };

    template <typename Nonterminal, typename Terminal>
    std::ostream& operator <<(std::ostream& o, const PassiveItem<Nonterminal, Terminal>& item){
        o << "<" << item.get_nont() << ", [";
        for(auto range : item.get_ranges()){
            o << range;
        }
        o << "]>";
        return o;
    };




    template <typename Nonterminal, typename Terminal>
    class ActiveItem {
    private:
        std::shared_ptr<Rule<Nonterminal, Terminal>> rule;
        unsigned long fanout;
        std::vector<Range> pre_ranges;
        unsigned long k; // currently active argument of rule (0-based)
        unsigned int posInK; // position of the "dot"
        Range currentRange;
        std::map<unsigned long, std::shared_ptr<PassiveItem<Nonterminal, Terminal>>> records;


    public:
        ActiveItem(std::shared_ptr<Rule<Nonterminal,Terminal>> r): rule(r) {
            fanout = rule->get_lhs().get_args().size();
            pre_ranges.reserve(fanout);
            k = 0;
            posInK = 0;
            currentRange = Range{0,0};
        }


        bool isArgumentCompleted() const{
            return posInK >= rule->get_lhs().get_args().at(k).size();
        }

        void complete(){
            assert(isArgumentCompleted());
            pre_ranges.push_back(std::move(currentRange));
            currentRange = Range{0,0};
            ++k;
            posInK = 0;
        }


        bool isFinished() const {
            return k >= fanout && posInK == 0;
        }

        PassiveItem<Nonterminal, Terminal> convert() const {
            assert(isFinished());

            return PassiveItem<Nonterminal, Terminal>(rule->get_lhs().get_nont(), pre_ranges);

        }


        bool isAtWildcardPosition() const {
            return currentRange.first == 0 && currentRange.second == 0;
        }

        void setCurrentPosition(unsigned int pos) {
            assert(isAtWildcardPosition());

            currentRange.first = pos;
            currentRange.second = pos;
        }



        bool scanTerminal(Terminal t) {
            const auto arg = rule->get_lhs().get_args().at(k);
            if(arg.at(posInK).which() == 1) // next item is a variable
                return false;
            Terminal w = boost::get<Terminal>(arg.at(posInK));
            if(w == t) {
                ++currentRange.second;
                ++posInK;
                return true;
            }
            return false; // symbol does not match
        }

        bool scanVariable() {
            const auto arg = rule->get_lhs().get_args().at(k);
            if(arg.at(posInK).which() == 0) // next item is a terminal
                return false;
            Variable var = boost::get<Variable>(arg.at(posInK));
            Nonterminal b = rule->get_rhs().at(var.get_index());
            if(records.count(var.get_index()) == 0) // no record yet
                return false;
            std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pitem = records.at(var.get_index());
            Range rangeOfRec = pitem->get_ranges().at(var.get_arg());

            if(currentRange.second != rangeOfRec.first) // record does not match!
                return false;

            currentRange.second = rangeOfRec.second;
            ++posInK;
            return true;
        }

        bool addRecord(const unsigned long arg, const std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pitem){
            if(records.count(arg)==1)
                return false; // record already exist
            records[arg] = pitem;
            return true;
        }

        template <typename Nonterminal1, typename Terminal1>
        friend std::ostream& operator <<(std::ostream&, const ActiveItem<Nonterminal1, Terminal1>&);
    };

    template <typename Nonterminal, typename Terminal>
    std::ostream& operator <<(std::ostream& o, const ActiveItem<Nonterminal, Terminal>& item){
        o << "<" << *item.rule << ", [";
        for(auto range : item.pre_ranges){
            o << range;
        }
        o << "], ";
        o << item.currentRange;
        o << " -" << item.k << "/" << item.posInK << "- ";
        if(item.k < item.fanout){
            auto comp = item.rule->get_lhs().get_args().at(item.k);
            for (unsigned long i= item.posInK; i < comp.size(); ++i){
                o << comp.at(i) << " ";
            }
        }
        o << ">";
        return o;
    };



    template <typename Nonterminal, typename Terminal>
    class LCFRS_Parser {

    private:
        LCFRS<Nonterminal, Terminal> grammar;
        std::vector<Terminal> word;

    public:
        LCFRS_Parser(const LCFRS<Nonterminal, Terminal> &grammar, std::vector<Terminal> word) : grammar(grammar), word(word) {};


        void do_parse(){


        }
    };













}

#endif //STERM_PARSER_LCFRS_PARSER_H
