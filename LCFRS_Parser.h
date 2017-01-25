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
    class PassiveItem {
    private:
        Nonterminal nont;
        std::vector<Range> ranges;
    public:
        PassiveItem(PassiveItem<Nonterminal, Terminal>&& pItem)
                : nont(pItem.get_nont()), ranges(pItem.get_ranges()){}

        PassiveItem(const PassiveItem<Nonterminal, Terminal>& pItem)
                : nont(pItem.nont), ranges(pItem.ranges){}

        PassiveItem(Nonterminal n, std::vector<Range> rs): nont(n), ranges(rs){}

        Nonterminal get_nont() const {
            return nont;
        }
        std::vector<Range> get_ranges() const {
                return ranges;
        }


        friend bool operator<(const PassiveItem<Nonterminal, Terminal>& l
                , const PassiveItem<Nonterminal, Terminal>& r) {
            return std::tie(l.nont, l.ranges) < std::tie(r.nont, r.ranges);
        }

        friend std::ostream& operator <<(std::ostream& o, const PassiveItem<Nonterminal, Terminal>& item){
            o << "<" << item.get_nont() << ", [";
            for(auto range : item.get_ranges()){
                o << range;
            }
            o << "]>";
            return o;
        };
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
        std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>> records;


    public:

        ActiveItem(std::shared_ptr<Rule<Nonterminal,Terminal>> r, Range range) : rule(r), currentRange(range) {
            fanout = rule->get_lhs().get_args().size();
            pre_ranges.reserve(fanout);
            k = 0;
            posInK = 0;
            records = std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>>(rule->get_rhs().size());
        }

        ActiveItem(const ActiveItem& a):
                rule(a.rule)
                , fanout(a.fanout)
                , pre_ranges(a.pre_ranges)
                , k(a.k)
                , posInK(a.posInK)
                , currentRange(a.currentRange)
                , records(a.records)
        {}


        unsigned long get_argument() const {
            return k;
        }

        std::shared_ptr<Rule<Nonterminal, Terminal>> get_rule() const {
            return rule;
        };

        const std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>>& getRecords() const {
            return records;
        };

        const std::shared_ptr<Rule<Nonterminal,Terminal>>& getRule() const{
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
            return records[index] != nullptr;
        }


        bool isRecordPlausible() const {
            TerminalOrVariable<Terminal> adot = afterDot();

            if(adot.which() == 0) // next item is a terminal
                return false;
            Variable var = boost::get<Variable>(adot);

            if(! hasRecordFor(var.get_index())) // no record yet
                return true;

            return currentRange.second == records[var.get_index()]->get_ranges().at(var.get_arg()).first;
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

        bool addRecord(const std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pitem){
            if(afterDot().which()==0)
                return false; // there is no variable
            Variable var{boost::get<Variable>(afterDot())};
            if(hasRecordFor(var.get_index()))
                return false; // record already exist
            records.at(var.get_index()) = pitem;
            return true;
        }


        ItemIndex<Nonterminal> getItemIndex() const {
            return ItemIndex<Nonterminal>(rule->get_lhs().get_nont(), k, posInK);
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





    template <typename Nonterminal, typename Terminal>
    class TraceItem{
    public:
        std::shared_ptr<PassiveItem<Nonterminal,Terminal>> uniquePtr;
        std::vector
                <
                std::pair
                        <
                        std::shared_ptr<Rule<Nonterminal, Terminal>>
                        , std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>>
                        >
                > parses;
    };










    template <typename Nonterminal, typename Terminal>
    class LCFRS_Parser {

    private:
        LCFRS<Nonterminal, Terminal> grammar;
        std::vector<Terminal> word;
        std::map<ItemIndex<Nonterminal>, std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>>> passiveItems;
        std::deque<std::shared_ptr<ActiveItem<Nonterminal, Terminal>>> queue;
        std::map<ItemIndex<Nonterminal>, std::vector<std::shared_ptr<ActiveItem<Nonterminal, Terminal>>>> waiting;

        std::map<PassiveItem<Nonterminal,Terminal>
                , TraceItem<Nonterminal,Terminal>
                > trace;

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

            queueRules(grammar.get_initial_nont(), 0);

            workQueue();

//std::clog << "Passive Items:" << std::endl;
//for (auto const& passive : passiveItems) {
//    for(auto const& pItem : passive.second)
//        std::clog << "    " << *pItem << std::endl;
//}

        }


        void workQueue(){
            while( ! queue.empty()) {

                std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem = queue.front();
                queue.pop_front();

//std::clog << "    Item: " << *currentItem << std::endl;



                // Handle ε-arguments
                if (currentItem->isArgumentCompleted()) {

                    currentItem->completeArgument();
                    if (currentItem->isFinished())
                        transformToPassive(std::move(currentItem));
                    else
                        // Continue scanning this item for the next components (there are, since it is not finished)
                        generatePositionsAndAddToQueue(std::move(currentItem));
                    continue;
                }

                // All items have something to be scanned!
                assert(currentItem->afterDot().which() >= 0);

                // Try to scan a terminal
                if (tryToScanTerminal(currentItem))
                    continue;

                // Try to scan a variable with existing record
                if (tryToScanVariable(currentItem))
                    continue;


                // Try to find the needed record
                assert(currentItem->afterDot().which() == 1); // The next thing is a variable

                const Variable var = boost::get<Variable>(currentItem->afterDot());
                const Nonterminal nont = currentItem->get_rule()->get_rhs().at(var.get_index());
                const ItemIndex<Nonterminal> ind(nont, var.get_arg() , currentItem->get_current_Position().second);


                ItemIndex<Nonterminal> representative(nont, 0, ind.startingPos);
                if (passiveItems.count(representative) == 0) { // Question has not been asked
                    passiveItems[representative]; // Indicate that the question has been asked
                    if (var.get_arg() == 0) { // At argument position 0, only the relevant position needs to be considered
                        queueRules(nont, ind.startingPos);

                    } else { // If argument > 0, no optimization possible. Try all positions starting from argument 0
                        for (unsigned long pos = 0; pos <= word.size(); ++pos) {
                            queueRules(nont, pos);
                            passiveItems[ItemIndex<Nonterminal>{nont, 0, pos}];
                        }
                    }
                } else {
                    // If question has been asked, then there may be answers:
                    for (std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pItem : passiveItems[ind]) {
                        std::shared_ptr<ActiveItem<Nonterminal, Terminal>> aItem
                                = std::make_shared<ActiveItem<Nonterminal, Terminal>>
                                        (ActiveItem<Nonterminal, Terminal>{*currentItem});
                        if (aItem->addRecord(pItem))
                            queue.push_front(aItem);
                    }
                }


                // this item is now waiting for further answers
                waiting[ind].emplace_back(std::move(currentItem));
//std::clog << "         - waiting..." << std::endl;

            }

        }

        void queueRules(const Nonterminal& nont, const unsigned long pos){
//std::clog << "Queueing items for " << nont
//          << " at " << pos
//          << " (size: " << grammar.get_rules().at(nont).size()  << ")" << std::endl;

            for(std::shared_ptr<Rule<Nonterminal, Terminal>> r : grammar.get_rules().at(nont)){
                queue.push_back(
                        std::make_shared<ActiveItem<Nonterminal, Terminal>>(
                                ActiveItem<Nonterminal, Terminal>(r,Range{pos,pos})
                        )
                );
            }
            return;
        }

        bool tryToScanTerminal(std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem){
            if(currentItem->afterDot().which() != 0){
                return false; // next item is a variable, nothing to be done here
            }

            // Is a terminal plausible?
            if (! currentItem->isTerminalPlausible(word))
                // There is a terminal, but it is not plausible. Hence, signal to abort this item
                return true;

            currentItem->scanTerminal(word);

            if(currentItem->isArgumentCompleted()){
                currentItem->completeArgument();
                if(currentItem->isFinished())
                    transformToPassive(std::move(currentItem));
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


        bool tryToScanVariable(std::shared_ptr<ActiveItem<Nonterminal, Terminal>> currentItem){
            if(currentItem->afterDot().which() != 1){
                return false; // next item is a terminal, nothing to be done here
            }

            if( ! currentItem->hasRecordFor(boost::get<Variable>(currentItem->afterDot()).get_index())){
                return false; // there is no record yet, nothing to be done here
            }


            if( ! currentItem->isRecordPlausible()) // record cannot be applied. Hence signal to abort
                return true;

            currentItem->scanVariable();

            if(currentItem->isArgumentCompleted()){
                currentItem->completeArgument();
                if(currentItem->isFinished())
                    transformToPassive(std::move(currentItem));
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
                std::shared_ptr<ActiveItem<Nonterminal, Terminal>>&& currentItem
        ){
            std::shared_ptr<PassiveItem<Nonterminal, Terminal>> pItem
                    (std::make_shared<PassiveItem<Nonterminal, Terminal>>(currentItem->convert()));


            // prepare the parse
            std::pair
                    <std::shared_ptr<Rule<Nonterminal, Terminal>>
                            , std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>>
                    > parse
                    {
                            currentItem->getRule()
                            , currentItem->getRecords()
                    };




            Nonterminal nont{pItem->get_nont()};
            if(trace.count(*pItem) == 0) { // observed for first time
                std::vector<Range> ranges{pItem->get_ranges()};
                // notify and store
                for(unsigned long argument = 0; argument < ranges.size(); ++argument){
                    ItemIndex<Nonterminal> ind(nont, argument, ranges.at(argument).first);
                    passiveItems[ind].push_back(pItem);

                    for (std::shared_ptr<ActiveItem<Nonterminal, Terminal>> aItem : waiting[ind]) {
                        std::shared_ptr<ActiveItem<Nonterminal, Terminal>> copyItem
                                {std::make_shared<ActiveItem<Nonterminal, Terminal>>
                                         (ActiveItem<Nonterminal, Terminal>{*aItem})};
                        if (copyItem->addRecord(pItem))
                            queue.push_back(copyItem);
                    }
                }
                // put into trace
                TraceItem<Nonterminal,Terminal> traceItem
                        {
                                pItem,
                                std::vector<
                                        std::pair<
                                                std::shared_ptr<Rule<Nonterminal, Terminal>>
                                                , std::vector<std::shared_ptr<PassiveItem<Nonterminal, Terminal>>>
                                        >
                                >
                                        {parse} // only one parse item
                        };
                trace[*pItem]
                        = traceItem;
            } else { // pItem was already observed
                // No notification and no storage.
                // Add parse to existing trace
                trace[*pItem].parses.push_back(parse);
            }
        }

        void generatePositionsAndAddToQueue(
                std::shared_ptr<ActiveItem<Nonterminal, Terminal>>&& currentItem
        ){
            for (unsigned long pos=0; pos <= word.size(); ++pos) {
                std::shared_ptr<ActiveItem<Nonterminal, Terminal>> copyItem
                        {std::make_shared<ActiveItem<Nonterminal, Terminal>>(*currentItem)};
                if (copyItem->setCurrentPosition(pos)) {
                    queue.push_front(std::move(copyItem));
                }
            }
        }



        const std::map<PassiveItem<Nonterminal, Terminal>,TraceItem<Nonterminal,Terminal>>& getTrace() const {
            return trace;
        };

    };













}

#endif //STERM_PARSER_LCFRS_PARSER_H
