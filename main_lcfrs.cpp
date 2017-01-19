//
// Created by Markus on 19.01.17.
//
#include <iostream>

#include "LCFRS.h"

using namespace LCFR;
using namespace std;

int main(){
    LCFRS<string, string> grammar("Test");

    LHS<string,string> lhs1("S");
    Variable var1(1,1);
    vector<TerminalOrVariable<string>> arg1{"a","a", var1};
    lhs1.addArgument(arg1);
    grammar.add_rule(Rule<string,string>(lhs1, vector<string>{"S"}));
    clog << grammar;


    return 0;
}