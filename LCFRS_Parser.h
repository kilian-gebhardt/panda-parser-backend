//
// Created by Markus on 19.01.17.
//

#ifndef STERM_PARSER_LCFRS_PARSER_H
#define STERM_PARSER_LCFRS_PARSER_H

#include <map>
#include <deque>
#include <stack>
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
        ActiveItem(std::shared_ptr<Rule<Nonterminal,Terminal>> r) : rule(r) {
            fanout = rule->get_lhs().get_args().size();
            pre_ranges.reserve(fanout);
            k = 0;
            posInK = 0;
            currentRange = Range{0,0};
        }


        unsigned long get_argument() const {
            return k;
        }

        std::shared_ptr<Rule<Nonterminal, Terminal>> get_rule() const {
            return rule;
        };

        bool isArgumentCompleted() const{
            return posInK >= rule->get_lhs().get_args().at(k).size();
        }

        void completeArgument(){
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

        Range get_current_Position() const {
            return currentRange;
        }

        bool setCurrentPosition(unsigned int pos) {
            assert(isAtWildcardPosition());

            // check whether the position is not already in a known range:
            for(Range range : pre_ranges){
                if(range.first < pos && pos < range.second)
                    return false;
            }

            currentRange.first = pos;
            currentRange.second = pos;
        }


        TerminalOrVariable<Terminal> afterDot() const {
            return rule->get_lhs().get_args().at(k).at(posInK);
        }



        bool isRecordPlausible() const {
            TerminalOrVariable<Terminal> adot = afterDot();

            if(adot.which() == 0) // next item is a terminal, so we don't know
                return true;
            Variable var = boost::get<Variable>(adot);

            if(records.count(var.get_index()) == 0) // no record yet
                return true;

            return currentRange.second == records.at(var.get_index())->get_ranges().at(var.get_arg()).first;
        }


        bool isTerminalPlausible(const std::vector<Terminal> word) const {
            TerminalOrVariable<Terminal> adot = afterDot();

            if(adot.which() == 1) // next item is a variable, so we don't know
                return true;

            if(get_current_Position().second >= word.size())
                return false; // there is no word left to be scanned

            Terminal t = boost::get<Terminal>(adot);

            return word.at(get_current_Position().second) == t;
        }

        /**
         * Scans a terminal, assumes that isTerminalPlausible(word) holds!
         * @param word The word to scan from
         */
        void scanTerminal(const std::vector<Terminal> word) {

            assert(isTerminalPlausible(word));

            ++currentRange.second;
            ++posInK;
        }

        bool scanVariable() {
            if ( ! isRecordPlausible())
                return false;

            const auto arg = rule->get_lhs().get_args().at(k);
            if(arg.at(posInK).which() == 0) // next item is a terminal
                return false;
            Variable var = boost::get<Variable>(arg.at(posInK));

            std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pitem = records.at(var.get_index());
            Range rangeOfRec = pitem->get_ranges().at(var.get_arg());

            Nonterminal b = rule->get_rhs().at(var.get_index());

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



    template  <typename Nonterminal>
    class ItemIndex {
    public:
        Nonterminal nont;
        unsigned long arg;
        unsigned long startingPos;

        ItemIndex(const Nonterminal n, const unsigned long a, unsigned long sp):
                nont(n), arg(a), startingPos(sp)
                {}

        ItemIndex(ItemIndex&& i): nont(i.nont), arg(i.arg), startingPos(i.startingPos) {}

    };

    template <typename Nonterminal, typename Terminal>
    class LCFRS_Parser {

    private:
        LCFRS<Nonterminal, Terminal> grammar;
        std::vector<Terminal> word;
        std::map<ItemIndex<Nonterminal>, std::deque<std::shared_ptr<ActiveItem<Nonterminal, Terminal>>>> agenda;
        std::map<ItemIndex<Nonterminal>, std::deque<std::shared_ptr<ActiveItem<Nonterminal, Terminal>>>> history;
        std::map<ItemIndex<Nonterminal>, std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>>> passiveItems;
        std::stack<ItemIndex<Nonterminal>> goalList;

    public:
        LCFRS_Parser(const LCFRS<Nonterminal, Terminal> &grammar, std::vector<Terminal> word)
                : grammar(grammar), word(word) {};


        /**
         * Parse word with grammar
         *
         * Assumption:
         *   - each reachable nonterminal is productive!
         */
        void do_parse(){

            initialize(); // fill up the agenda

            goalList.push_back(ItemIndex<Nonterminal>(grammar.get_initial_nont(), 0, 0));

            while( ! goalList.empty()){
                ItemIndex<Nonterminal> currentGoal{goalList.pop()};

                if(! agenda.count(currentGoal)) { // there is no item to be processed
                    continue;
                }

                // Get current item from current goal
                std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem{agenda.at(currentGoal).pop_front()};

                if(history.at(currentGoal)
                           .find(history.at(currentGoal).begin(), history.at(currentGoal).end(), currentItem)
                   != history.at(currentGoal).end()){
                    // item is already in history
                    continue;
                }

                // Handle ε-arguments
                if(currentItem->isArgumentCompleted()){
                    writeHistory(currentGoal,currentItem); // copy this item into history

                    goalList.push(currentGoal); // The goal stays the same after handling of ε-rule
                    currentItem->completeArgument();
                    if(currentItem->isFinished())
                        transformToPassive(currentGoal, std::move(currentItem));
                    else
                        // Continue scanning this item for the next components (there are, since it is not finished)
                        completeArgumentAndAddToAgenda(currentGoal, std::move(currentItem));
                    continue;
                }

                // All items have something to be scanned!
                assert(currentItem->get_current_Position().second < word.size());

                // Try to scan a terminal
                if (tryToScanTerminal(currentGoal, currentItem))
                    continue;

                // Is assigned record plausible?
                if (! currentItem->isRecordPlausible()) {
                    goalList.push(currentGoal); // Goal stays the same, item can be forgotten
                    continue;
                }


                //TODO: adapt to scheme of variable: unify plausability and check
                // Try to scan a variable with existing record
                if (tryToScanVariable(currentGoal, currentItem))
                    continue;


                // Try to find the needed record
                assert(currentItem->afterDot().which() == 1) // The next thing is a variable
                const Variable var = boost::get<Variable>(currentItem->afterDot());
                const Nonterminal nont = currentItem->get_rule()->get_rhs().at(var.get_index());
                const ItemIndex goal(nont, var.get_arg(), currentItem->get_current_Position().second);
                if(passiveItems.count(goal) == 0){
                    // there are no suitable passive items, try to build one
                    goalList.push(std::move(currentGoal));
                    goalList.push(std::move(goal));
                    agenda[currentGoal].push_back(std::move(currentItem)); // check for this item later
                    continue;
                }

                for (Passiveitem pItem : passiveItems.at(goal)){

                }


            }

        }


        void initialize() {

        }

        bool tryToScanTerminal(
                const ItemIndex<Nonterminal>& currentGoal
                , std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            if(currentItem->afterDot().which() != 0){
                return false; // next item is a variable, nothing to be done here
            }

            goalList.push(currentGoal); // Goal stays the same
            writeHistory(currentGoal,currentItem); // item will be processed in the following

            // Is a terminal plausible?
            if (! currentItem->isTerminalPlausible(word))
                // There is a terminal, but it is not plausible. Hence, signal to abort this item
                return true;

            currentItem->scanTerminal(word);

            if(currentItem->isArgumentCompleted()){
                currentItem->completeArgument();
                if(currentItem->isFinished())
                    transformToPassive(currentGoal, std::move(currentItem));
                else
                    // Continue scanning this item for the next components (there are, since it is not finished)
                    completeArgumentAndAddToAgenda(currentGoal, std::move(currentItem));
            } else { // Argument is not complete
                // Push the item to the front, since it worked out!
                agenda[currentGoal].push_front(std::move(currentItem));
            }
            return true; // the appropriate action has been taken
        }


        bool tryToScanVariable(
                const ItemIndex<Nonterminal>& currentGoal
                , std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            goalList.push(currentGoal); // The goal stays the same after scanning

            if(currentItem->scanVariable()){
                if(currentItem->isArgumentCompleted()){
                    currentItem->completeArgument();
                    if(currentItem->isFinished())
                        transformToPassive(currentGoal, std::move(currentItem));
                    else
                        // Continue scanning this item for the next components (there are, since it is not finished)
                        completeArgumentAndAddToAgenda(currentGoal, std::move(currentItem));
                } else { // Argument is not complete
                    // Push the item to the front, since it worked out!
                    agenda[currentGoal].push_front(std::move(currentItem));
                }
                return true;
            }
            return false; // Could not scan variable
        }


        void transformToPassive(
                const ItemIndex<Nonterminal>& currentGoal
                , std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            passiveItems.at(currentGoal).push_back(std::make_shared(currentItem->convert()));
        }

        void completeArgumentAndAddToAgenda(
                const ItemIndex<Nonterminal>& currentGoal
                , std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            currentItem->completeArgument();
            for (unsigned long pos=0; pos <= word.size(); ++pos) {
                std::shared_ptr<ActiveItem<Nonterminal, Terminal>> copyItem
                        = std::make_shared(*currentItem);
                if (copyItem->setCurrentPosition(pos)) {
                    // only add, if current position can be set (its still free)
                    ItemIndex<Nonterminal> copyGoal(currentGoal.nont, copyItem->get_argument(), pos);
                    agenda[copyGoal].push_front(std::move(copyItem));
                    goalList.push(copyGoal);
                }
            }
        }

        void writeHistory(
                const ItemIndex<Nonterminal>& currentGoal
                , const std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            history[currentGoal].push_back(std::make_shared<ActiveItem>(*currentItem));
        }
    };













}

#endif //STERM_PARSER_LCFRS_PARSER_H
