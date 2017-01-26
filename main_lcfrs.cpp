//
// Created by Markus on 19.01.17.
//
#include <iostream>

#include "LCFRS.h"
#include "LCFRS_Parser.h"
#include "LCFRS_util.h"

using namespace std;
using namespace LCFR;

void manual_parse(const LCFRS<string, string> &grammar, const vector<string> &word);


int main(){
    LCFRS<string, string> grammar("S","Test");

//    grammar.add_rule(construct_rule("S", vector<string>{"a a x{0,0}"}, "S"));
//    grammar.add_rule(construct_rule("S", vector<string>{"a"}, ""));
//    vector<string> word{"a", "a", "a", "a"};
//    manual_parse(grammar, word);

//    grammar.add_rule(construct_rule("S", vector<string>{"x{0,0} x{0,1}"}, "A"));
//    grammar.add_rule(construct_rule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A"));
//    grammar.add_rule(construct_rule("A", vector<string>{"a", "b"}, "A"));
//    vector<string> word;
//    tokenize<vector<string>>("a a a b b b", word);


//    grammar.add_rule(construct_rule("S", vector<string>{"x{0,0} x{0,1}"}, "A"));
//    grammar.add_rule(construct_rule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A"));
//    grammar.add_rule(construct_rule("A", vector<string>{"a", "b"}, ""));
//    vector<string> word;
//    tokenize<vector<string>>("a a a a a a a a a b b b b b b b b b", word);

//    grammar.add_rule(construct_rule("S", vector<string>{"x{0,0} x{1,0} x{0,1} x{1,1}"}, "A B"));
//    grammar.add_rule(construct_rule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A"));
//    grammar.add_rule(construct_rule("A", vector<string>{"a", "b"}, ""));
//    grammar.add_rule(construct_rule("B", vector<string>{"a", "b"}, ""));
//    grammar.add_rule(construct_rule("B", vector<string>{"x{0,0} a", "x{0,1} b"}, "B"));
//    vector<string> word;
//    tokenize<vector<string>>("a a a a b b b b", word);

    grammar.add_rule(construct_rule("S", vector<string>{"x{0,1} x{1,1} x{0,0} x{1,0}"}, "A B"));
    grammar.add_rule(construct_rule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A"));
    grammar.add_rule(construct_rule("A", vector<string>{"a", "b"}, ""));
    grammar.add_rule(construct_rule("B", vector<string>{"", ""}, "")); // ε-rule
    grammar.add_rule(construct_rule("B", vector<string>{"x{0,0} a", "x{0,1} b"}, "B"));
    vector<string> word;
    tokenize<vector<string>>("b b b b b a a a a a", word);

//    grammar.add_rule(construct_rule("S", vector<string>{"x{0,0} x{1,0}"}, "A B")); // deleting rule
//    grammar.add_rule(construct_rule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A")); // as many a's as b's
//    grammar.add_rule(construct_rule("A", vector<string>{"a", "b"}, ""));
//    grammar.add_rule(construct_rule("B", vector<string>{"", ""}, "")); // ε-rule
//    grammar.add_rule(construct_rule("B", vector<string>{"x{0,0} b b"}, "B")); // an even number of b's
//    vector<string> word;
//    tokenize<vector<string>>("a a a a a a b b b b b b", word);




    clog << grammar << endl;


    LCFRS_Parser<string, string> parser(grammar, word);
    parser.do_parse();


    map<PassiveItem<string>, TraceItem<string,string>> trace = parser.get_trace();
    clog << "Parses:" << endl;


    for (auto const& parse : trace[PassiveItem<string>(grammar.get_initial_nont(), std::vector<Range>{Range(0,word.size())})].parses){
        clog << "    " << *(parse.first) << ": " ;
        for(auto const& ppitem : parse.second){
            clog << *ppitem << ", ";
        }
        clog << endl;
    }

    return 0;
}



void manual_parse(const LCFRS<string, string> &grammar, const vector<string> &word) {
    ActiveItem<string, string> activeItem{grammar.get_rules().at("S").front(), Range{0, 0}};
    clog << activeItem.is_finished() << endl;
    clog << activeItem << endl;
    activeItem.scan_terminal(word);
    clog << activeItem << endl;
    activeItem.scan_terminal(word);
    clog << activeItem << endl;


    vector<Range> rangeVector{Range(2,5), Range(2,6)};
    shared_ptr<PassiveItem<string>> pItem = make_shared<PassiveItem<string>>("A", rangeVector);
    activeItem.add_record(pItem);
    activeItem.scan_variable();
    clog << activeItem << endl;
    clog << activeItem.is_argument_completed() << endl;
    activeItem.complete_argument();
    clog << activeItem.is_finished() << endl;
    clog << activeItem.is_at_wildcard_position() << endl;
    PassiveItem<string> convItem{activeItem.convert()};
    clog << convItem << endl;
}