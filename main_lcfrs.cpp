//
// Created by Markus on 19.01.17.
//
#include <iostream>

#include "LCFRS.h"
#include "LCFRS_Parser.h"

using namespace LCFR;
using namespace std;

int main(){
    LCFRS<string, string> grammar("Test");

    LHS<string,string> lhs1("S");
    Variable var1(0,1);
    vector<TerminalOrVariable<string>> arg1{"a","a", var1};
    lhs1.addArgument(arg1);
    grammar.add_rule(Rule<string,string>(lhs1, vector<string>{"S"}));
    clog << grammar << endl;

    ActiveItem<string, string> activeItem{grammar.get_rules().front()};
    clog << activeItem.isFinished() << endl;
    clog << activeItem << endl;
    activeItem.scanTerminal("a");
    clog << activeItem << endl;
    activeItem.scanTerminal("a");
    clog << activeItem << endl;



    vector<Range> rangeVector{Range(2,5), Range(2,6)};
    shared_ptr<PassiveItem<string,string>> pItem = make_shared<PassiveItem<string,string>>("A", rangeVector);
    activeItem.addRecord(0, pItem);
    activeItem.scanVariable();
    clog << activeItem << endl;
    clog << activeItem.isArgumentCompleted() << endl;
    activeItem.complete();
    clog << activeItem.isFinished() << endl;
    clog << activeItem.isAtWildcardPosition() << endl;
    PassiveItem<string,string> convItem{activeItem.convert()};
    clog << convItem << endl;

    return 0;
}