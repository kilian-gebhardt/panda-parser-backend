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

        ActiveItem(std::shared_ptr<Rule<Nonterminal,Terminal>> r, Range range): rule(r), currentRange(range) {
            fanout = rule->get_lhs().get_args().size();
            pre_ranges.reserve(fanout);
            k = 0;
            posInK = 0;
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

            return true;
        }


        TerminalOrVariable<Terminal> afterDot() const {
            return rule->get_lhs().get_args().at(k).at(posInK);
        }



        bool isTerminalPlausible(const std::vector<Terminal> word) const {
            TerminalOrVariable<Terminal> adot = afterDot();

            if(adot.which() == 1) // next item is a variable
                return false;

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


        bool hasRecordFor(unsigned long index) const {
            return records.count(index) > 0;
        }


        bool isRecordPlausible() const {
            TerminalOrVariable<Terminal> adot = afterDot();

            if(adot.which() == 0) // next item is a terminal
                return false;
            Variable var = boost::get<Variable>(adot);

            if(records.count(var.get_index()) == 0) // no record yet
                return true;

            return currentRange.second == records.at(var.get_index())->get_ranges().at(var.get_arg()).first;
        }

        /**
         * Scans a variable from the record. Assumes that isRecordPlausible() holds!
         */
        void scanVariable() {
            assert(isRecordPlausible());

            Variable var = boost::get<Variable>(rule->get_lhs().get_args().at(k).at(posInK));

            Range rangeOfRec = records.at(var.get_index())->get_ranges().at(var.get_arg());

            currentRange.second = rangeOfRec.second;
            ++posInK;
        }

        bool addRecord(const unsigned long index, const std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pitem){
            if(records.count(index)>0)
                return false; // record already exist
            records[index] = pitem;
            return true;
        }


        bool operator ==(const ActiveItem<Nonterminal, Terminal>& item) const {
            return
                rule == item.rule
                && pre_ranges == item.pre_ranges
                && k == item.k
                && posInK == item.posInK
                && currentRange == item.currentRange
                && records == item.records;
        }



        friend std::ostream& operator <<(std::ostream& o, const ActiveItem<Nonterminal, Terminal>& item){
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

        ItemIndex(const ItemIndex& i): nont(i.nont), arg(i.arg), startingPos(i.startingPos) {}

        friend bool operator<(const ItemIndex<Nonterminal>& l, const ItemIndex<Nonterminal>& r){
            return std::tie(l.nont, l.arg, l.startingPos) < std::tie(r.nont, r.arg, r.startingPos);
        }

        friend std::ostream& operator <<(std::ostream& o, const ItemIndex<Nonterminal>& item) {
            o << "(" << item.nont << "," << item.arg << "," << item.startingPos << ")";
            return o;
        }

    };

    template <typename Nonterminal, typename Terminal>
    class LCFRS_Parser {

    private:
        LCFRS<Nonterminal, Terminal> grammar;
        std::vector<Terminal> word;
        std::map<ItemIndex<Nonterminal>, std::deque<std::shared_ptr<ActiveItem<Nonterminal, Terminal>>>> agenda;
        std::map<ItemIndex<Nonterminal>, std::deque<std::shared_ptr<ActiveItem<Nonterminal, Terminal>>>> history;
        std::map<ItemIndex<Nonterminal>, std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>>> passiveItems;
        std::deque<std::shared_ptr<ActiveItem<Nonterminal, Terminal>>> queue;
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

            goalList.push(ItemIndex<Nonterminal>(grammar.get_initial_nont(), 0, 0));

            while( ! goalList.empty()){

                ItemIndex<Nonterminal> currentGoal(std::move(goalList.top()));
                goalList.pop();

std::clog << "Goal: " << currentGoal << std::endl;
//std::clog << "Passive Items:" << std::endl;
//for (auto const& passive : passiveItems) {
//    for(auto const& pItem : passive.second)
//        std::clog << "    " << *pItem << std::endl;
//}

                // add all items from the agenda to the queue
                queue.insert(std::end(queue), std::begin(agenda[currentGoal]), std::end(agenda[currentGoal]));
                agenda[currentGoal].clear();

                workQueue(std::move(currentGoal));
            }

std::clog << "Passive Items:" << std::endl;
for (auto const& passive : passiveItems) {
    for(auto const& pItem : passive.second)
        std::clog << "    " << *pItem << std::endl;
}

        }


        void workQueue(const ItemIndex<Nonterminal>&& currentGoal){
            while( ! queue.empty()) {

                std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem = queue.front();
                queue.pop_front();

std::clog << "    Item: " << *currentItem << std::endl;



                if (isInHistory(currentGoal, currentItem)) {
                    // item is already in history
//std::clog << "      - skipped" << std::endl;
                    continue;
                }

                // Handle ε-arguments
                if (currentItem->isArgumentCompleted()) {
                    writeHistory(currentGoal, currentItem); // copy this item into history

                    currentItem->completeArgument();
                    if (currentItem->isFinished())
                        transformToPassive(currentGoal, std::move(currentItem));
                    else
                        // Continue scanning this item for the next components (there are, since it is not finished)
                        generatePositionsAndAddToQueue(std::move(currentItem));
                    continue;
                }

                // All items have something to be scanned!
                assert(currentItem->afterDot().which() >= 0);

                // Try to scan a terminal
                if (tryToScanTerminal(currentGoal, currentItem))
                    continue;

                // Try to scan a variable with existing record
                if (tryToScanVariable(currentGoal, currentItem))
                    continue;


                // Try to find the needed record
                assert(currentItem->afterDot().which() == 1); // The next thing is a variable

                const Variable var = boost::get<Variable>(currentItem->afterDot());
                const Nonterminal nont = currentItem->get_rule()->get_rhs().at(var.get_index());
                const ItemIndex<Nonterminal> goal(nont, var.get_arg() , currentItem->get_current_Position().second);
                if (passiveItems.count(goal) == 0) {
                    if(var.get_arg() == 0) { // at argument position 0, only the relevant position needs to be considered
                        // There are no suitable passive items, try to build one
                        passiveItems[goal]; // add an empty passive item list to never use this case again!
                        goalList.push(currentGoal); // Revisit this goal later
                        goalList.push(std::move(goal)); // But before try to evaluate this
                        agenda[currentGoal].push_back(std::move(currentItem)); // Check for this item later

                    } else { // If argument > 0, no optimization possible. Try all positions starting from argument 0
                        agenda[currentGoal].push_back(std::move(currentItem)); // Check for this item later
                        goalList.push(std::move(currentGoal)); // Revisit this goal later
                        for (unsigned long pos = 0; pos <= word.size(); ++pos) {
                            const ItemIndex<Nonterminal> goal2(nont, 0, pos);
                            passiveItems[goal2]; // add an empty passive item list to never use this case again!
                            goalList.push(std::move(goal2)); // But before try to evaluate all other
                        }
                    }
                    continue;
                }

                bool somethingWasPushed{false};
                for (std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pItem : passiveItems.at(goal)) {
                    std::shared_ptr<ActiveItem<Nonterminal, Terminal>> newItem
                            {std::make_shared<ActiveItem<Nonterminal, Terminal>>(*currentItem)};
                    if (newItem->addRecord(var.get_index(), pItem) && ! isInHistory(currentGoal, newItem)) {
                        somethingWasPushed = true;
                        queue.push_front(std::move(newItem));
                    }
                }
                if(somethingWasPushed) {
                    // no history entry cause we want to check again!
                    // add the item to the agenda, to check again later
                    agenda[currentGoal].push_back(std::move(currentItem));
                    goalList.push(std::move(currentGoal));
std::clog << "           yeah! :-)" << std::endl;
                }else{
std::clog << "           nothing was pushed :-(" << std::endl;
                    writeHistory(currentGoal, currentItem);
                }
            }
        }


        void initialize() {
            for(std::shared_ptr<Rule<Nonterminal, Terminal>> r : grammar.get_rules()){
                for(unsigned long pos=0; pos <= word.size(); ++pos){
                    agenda[ItemIndex<Nonterminal>(r->get_lhs().get_nont(), 0, pos)].push_back(
                            std::make_shared<ActiveItem<Nonterminal, Terminal>>(
                                    ActiveItem<Nonterminal, Terminal>{r,Range{pos,pos}}
                            )
                    );
                }
            }
        }

        bool tryToScanTerminal(
                const ItemIndex<Nonterminal>& currentGoal
                , std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            if(currentItem->afterDot().which() != 0){
                return false; // next item is a variable, nothing to be done here
            }

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
                else {
                    // Continue scanning this item for the next components (there are, since it is not finished)
                    generatePositionsAndAddToQueue(std::move(currentItem));
                }
            } else { // Argument is not complete
                // Push the item to the front, since it worked out!
                queue.push_front(std::move(currentItem));
            }
            return true; // the appropriate action has been taken
        }


        bool tryToScanVariable(
                const ItemIndex<Nonterminal>& currentGoal
                , std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            if(currentItem->afterDot().which() != 1){
                return false; // next item is a terminal, nothing to be done here
            }

            if( ! currentItem->hasRecordFor(boost::get<Variable>(currentItem->afterDot()).get_index())){
                return false; // there is no record yet, nothing to be done here
            }

            writeHistory(currentGoal, currentItem); // in any case, the item will be processed


            if( ! currentItem->isRecordPlausible()) // record cannot be applied. Hence signal to abort
                return true;

            currentItem->scanVariable();

            if(currentItem->isArgumentCompleted()){
                currentItem->completeArgument();
                if(currentItem->isFinished())
                    transformToPassive(currentGoal, std::move(currentItem));
                else
                    // Continue scanning this item for the next components (there are, since it is not finished)
                    generatePositionsAndAddToQueue(std::move(currentItem));
            } else { // Argument is not complete
                // Push the item to the front, since it worked out!
                queue.push_front(std::move(currentItem));
            }
            return true;
        }


        void transformToPassive(
                const ItemIndex<Nonterminal>& currentGoal
                , std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pItem
                    (std::make_shared<PassiveItem<Nonterminal, Terminal>>(currentItem->convert()));

            Nonterminal nont{pItem->get_nont()};
            std::vector<Range> ranges{pItem->get_ranges()};
            for(int argument = 0; argument < ranges.size(); ++argument){
                passiveItems[ItemIndex<Nonterminal>{nont, argument, ranges.at(argument).first}].push_back(pItem);
            }
        }

        void generatePositionsAndAddToQueue(
                std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            for (unsigned long pos=0; pos <= word.size(); ++pos) {
                std::shared_ptr<ActiveItem<Nonterminal, Terminal>> copyItem
                        {std::make_shared<ActiveItem<Nonterminal, Terminal>>(*currentItem)};
                if (copyItem->setCurrentPosition(pos)) {
                    queue.push_front(std::move(copyItem));
                }
            }
        }


        bool isInHistory
                (const ItemIndex<Nonterminal>& currentGoal
                , const std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
                ) const {
            if(history.count(currentGoal) == 0)
                return false;
            auto it = std::find_if
                    (std::begin(history.at(currentGoal))
                            , std::end(history.at(currentGoal))
                            , [&](std::shared_ptr<ActiveItem<Nonterminal, Terminal>> const& itemIndex) {
                         return *currentItem == *itemIndex;
                     });
            return (it != std::end(history.at(currentGoal)));
        }


        void writeHistory(
                const ItemIndex<Nonterminal>& currentGoal
                , const std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem
        ){
            history[currentGoal].push_back(std::make_shared<ActiveItem<Nonterminal, Terminal>>(*currentItem));
        }
    };













}

#endif //STERM_PARSER_LCFRS_PARSER_H
