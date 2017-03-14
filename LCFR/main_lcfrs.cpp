//
// Created by Markus on 19.01.17.
//
#include <iostream>
#include <memory>
#include <vector>
#include <boost/variant.hpp>
#include <map>

#include "LCFRS.h"
#include "LCFRS_Parser.h"
#include "LCFRS_util.h"
#include "../Names.h"

using namespace std;
namespace LCFR {

    void manual_parse(const LCFRS<string, string> &grammar, const vector<string> &word);


    int main() {
        LCFRS<string, string> grammar("S", "Test");

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

        grammar.add_rule(construct_rule("S", vector<string>{"x{0,1} x{1,1} x{0,0} x{1,0}"}, "A B", 0));
        grammar.add_rule(construct_rule("A", vector<string>{"x{0,0} a", "x{0,1} b"}, "A", 1));
        grammar.add_rule(construct_rule("A", vector<string>{"a", "b"}, "", 2));
        grammar.add_rule(construct_rule("B", vector<string>{"", ""}, "", 3)); // ε-rule
        grammar.add_rule(construct_rule("B", vector<string>{"x{0,0} a", "x{0,1} b"}, "B", 4));
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


        map<PassiveItem<string>, TraceItem<string, string>> trace = parser.get_trace();
        clog << "Parses:" << endl;

//        print_top_trace(grammar, trace, word);

        std::vector<std::string> nodeLabels {"S", "A", "B", "C"};
        auto const nodeLabelsPtr = std::make_shared<const std::vector<string>>(nodeLabels);
        std::vector<EdgeLabelT> edgeLabels {0, 1, 2, 3, 4};
        auto const edgeLabelsPtr = std::make_shared<const std::vector<EdgeLabelT>>(edgeLabels);

        parser.prune_trace();
        auto hg{
                parser.convert_trace_to_hypergraph(
                          nodeLabelsPtr
                        , edgeLabelsPtr
                )};

        std::clog << std::endl << "Nodes in the pruned trace: " << std::endl;
        for (auto const &elmenent : *hg) {
            std::clog << elmenent << " ";
        }


        return 0;
    }
}

int main(){
        return LCFR::main();
}