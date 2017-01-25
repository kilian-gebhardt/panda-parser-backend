//
// Created by Markus on 19.01.17.
//
#include <iostream>

#include "LCFRS.h"
#include "LCFRS_Parser.h"

using namespace std;
using namespace LCFR;

void manualParse(const LCFRS<string, string> &grammar, const vector<string> &word);



int main(){
    LCFRS<string, string> grammar("S","Test");

//    grammar.add_rule(constructRule("S", vector<string>{"a a x{0,0}"}, "S"));
//    grammar.add_rule(constructRule("S", vector<string>{"a"}, ""));
//    vector<string> word{"a", "a", "a", "a"};
//    manualParse(grammar, word);

//    grammar.add_rule(constructRule("S", vector<string>{"x{0,0} x{0,1}"}, "A"));
//    grammar.add_rule(constructRule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A"));
//    grammar.add_rule(constructRule("A", vector<string>{"a", "b"}, "A"));
//    vector<string> word;
//    tokenize<vector<string>>("a a a b b b", word);


//    grammar.add_rule(constructRule("S", vector<string>{"x{0,0} x{0,1}"}, "A"));
//    grammar.add_rule(constructRule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A"));
//    grammar.add_rule(constructRule("A", vector<string>{"a", "b"}, ""));
//    vector<string> word;
//    tokenize<vector<string>>("a a a a a a a a a b b b b b b b b b", word);

    grammar.add_rule(constructRule("S", vector<string>{"x{0,0} x{1,0} x{0,1} x{1,1}"}, "A B"));
    grammar.add_rule(constructRule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A"));
    grammar.add_rule(constructRule("A", vector<string>{"a", "b"}, ""));
    grammar.add_rule(constructRule("B", vector<string>{"a", "b"}, ""));
    grammar.add_rule(constructRule("B", vector<string>{"x{0,0} a", "x{0,1} b"}, "B"));
    vector<string> word;
    tokenize<vector<string>>("a a a b b b", word);

//    grammar.add_rule(constructRule("S", vector<string>{"x{0,1} x{1,1} x{0,0} x{1,0}"}, "A B"));
//    grammar.add_rule(constructRule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A"));
//    grammar.add_rule(constructRule("A", vector<string>{"a", "b"}, ""));
//    grammar.add_rule(constructRule("B", vector<string>{"a", "b"}, ""));
//    grammar.add_rule(constructRule("B", vector<string>{"x{0,0} a", "x{0,1} b"}, "B"));
//    vector<string> word;
//    tokenize<vector<string>>("b b b b b a a a a a", word);




    clog << grammar << endl;


    LCFRS_Parser<string, string> parser(grammar, word);
    parser.do_parse();

    return 0;
}

void manualParse(const LCFRS<string, string> &grammar, const vector<string> &word) {
    ActiveItem<string, string> activeItem{grammar.get_rules().at("S").front(), Range{0, 0}};
    clog << activeItem.isFinished() << endl;
    clog << activeItem << endl;
    activeItem.scanTerminal(word);
    clog << activeItem << endl;
    activeItem.scanTerminal(word);
    clog << activeItem << endl;


    vector<Range> rangeVector{Range(2,5), Range(2,6)};
    shared_ptr<PassiveItem<string,string>> pItem = make_shared<PassiveItem<string,string>>("A", rangeVector);
    activeItem.addRecord(pItem);
    activeItem.scanVariable();
    clog << activeItem << endl;
    clog << activeItem.isArgumentCompleted() << endl;
    activeItem.completeArgument();
    clog << activeItem.isFinished() << endl;
    clog << activeItem.isAtWildcardPosition() << endl;
    PassiveItem<string,string> convItem{activeItem.convert()};
    clog << convItem << endl;
}