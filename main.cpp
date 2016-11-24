#include <iostream>

#include "HybridTree.h"
#include "SDCP.h"
#include "SDCP_Parser.h"
#include <vector>


std::shared_ptr<HybridTree<std::string, int>> build_hybrid_tree() {
    auto tree = std::make_shared<HybridTree<std::string, int>>(HybridTree<std::string, int>());
    tree->add_node(0, "is", 1);
    tree->add_node(1, ".", 2);
    tree->add_node(3, "hearing", 4);
    tree->add_node(4, "scheduled", 5);
    tree->add_node(6, "A", 7);
    tree->add_node(7, "on", 8);
    tree->add_node(9, "today", 10);
    tree->add_node(11, "issue", 12);
    tree->add_node(13, "the", 14);

    tree->set_entry(0);
    tree->set_exit(2);

    tree->add_child(1, 3);
    tree->add_child(1, 4);
    tree->add_child(1, 5);

    tree->add_child(4, 6);
    tree->add_child(4, 7);
    tree->add_child(4, 8);

    tree->add_child(8, 11);
    tree->add_child(8, 12);

    tree->add_child(12, 13);
    tree->add_child(12, 14);

    tree->add_child(5, 9);
    tree->add_child(5, 10);
    return tree;
};




int main() {
    // std::cout << "Hello, World!" << std::endl;

    HybridTree<std::string, int> tree = *build_hybrid_tree();

    tree.output();

    for (auto p : tree.terminals()) {
        std::cerr << p.first << " " << p.second << std::endl;
    }


    SDCP<std::string, std::string> sDCP;
    sDCP.set_initial("S");

    // Rule 1:
    // std::cout << "rule 1" << std::endl;
    Rule<std::string, std::string> rule1;
    rule1.lhn = "S";
    rule1.rhs.push_back("A");
    rule1.rhs.push_back("B");
    // build lhs
    auto term1 = Term<std::string>("is");
    term1.children.emplace_back(Variable(1, 1));
    term1.children.emplace_back(Variable(2, 1));
    STerm<std::string> arg1;
    arg1.push_back(term1);
    arg1.emplace_back(Term<std::string>("."));
    std::vector<STerm<std::string>> arg1v;
    arg1v.push_back(arg1);
    rule1.outside_attributes.push_back(arg1v);

    // build rhs
    STerm <std::string> arg2;
    arg2.emplace_back(Variable(2, 2));
    std::vector<STerm<std::string>> arg2v;
    arg2v.push_back(arg2);
    rule1.outside_attributes.push_back(arg2v);

    std::vector<STerm<std::string>> arg3v;
    rule1.outside_attributes.push_back(arg3v);
    // std::cout << rule1.outside_attributes.size() << std::endl;
    assert (sDCP.add_rule(rule1));
    // Rule 1 end

    // Rule 2:
    // std::cout << "rule 2" << std::endl;
    Rule<std::string, std::string> rule2;
    rule2.lhn = "A";
    // build lhs
    std::vector<STerm<std::string>> r2_arg1v;
    STerm<std::string> r2_arg_0_1;
    auto r2_term = Term<std::string>("hearing");
    r2_term.children.push_back(Term<std::string>("A"));
    r2_term.children.emplace_back(Variable(0, 1));
    r2_arg_0_1.push_back(r2_term);
    r2_arg1v.push_back(r2_arg_0_1);
    rule2.outside_attributes.push_back(r2_arg1v);
    assert (sDCP.add_rule(rule2));
    // Rule 2 end

    // Rule 3:
    // std::cout << "rule 3" << std::endl;
    Rule<std::string, std::string> rule3;
    rule3.lhn = "B";
    rule3.rhs.push_back("C");
    rule3.rhs.push_back("D");
    // build lhs
    std::vector<STerm<std::string>> r3_arg1v;
    STerm<std::string> r3_arg_0_1;
    r3_arg_0_1.push_back(Variable(1,1));
    STerm<std::string> r3_arg_0_2;
    r3_arg_0_2.push_back(Variable(2,1));
    r3_arg1v.push_back(r3_arg_0_1);
    r3_arg1v.push_back(r3_arg_0_2);
    rule3.outside_attributes.push_back(r3_arg1v);
    rule3.outside_attributes.emplace_back(std::vector<STerm<std::string>>());
    rule3.outside_attributes.emplace_back(std::vector<STerm<std::string>>());
    assert (sDCP.add_rule(rule3));
    // Rule 3 end

    // Rule 4:
    // std::cout << "rule 4" << std::endl;
    Rule<std::string, std::string> rule4;
    rule4.lhn = "C";
    // build lhs
    std::vector<STerm<std::string>> r4_arg1v;
    STerm<std::string> r4_arg_0_1;
    auto r4_term = Term<std::string>("scheduled");
    r4_term.children.emplace_back(Term<std::string>("today"));
    r4_arg_0_1.push_back(r4_term);
    r4_arg1v.push_back(r4_arg_0_1);
    rule4.outside_attributes.push_back(r4_arg1v);
    assert (sDCP.add_rule(rule4));
    // Rule 4 end

    // Rule 5:
    Rule<std::string, std::string> rule5;
    rule5.lhn = "D";
    // build lhs
    std::vector<STerm<std::string>> r5_arg1v;
    STerm<std::string> r5_arg_0_1;
    auto r5_term = Term<std::string>("on");
    auto r5_term_2 = Term<std::string>("issue");
    r5_term_2.children.emplace_back(Term<std::string>("the"));
    r5_term.children.push_back(r5_term_2);
    r5_arg_0_1.push_back(r5_term);
    r5_arg1v.push_back(r5_arg_0_1);
    rule5.outside_attributes.push_back(r5_arg1v);
    assert (sDCP.add_rule(rule5));
    // Rule 5 end

    Rule<std::string, std::string> rule6;
    rule6.lhn = "D";
    // build lhs
    rule6.outside_attributes.emplace_back(std::vector<STerm<std::string>>(1,STerm<std::string>(1, Variable(1, 1))));
    // build rhs
    rule6.rhs.push_back("F");
    rule6.outside_attributes.push_back(std::vector<STerm<std::string>>(1, STerm<std::string>(1, Variable(2, 1))));
    rule6.rhs.push_back("E");
    rule6.outside_attributes.push_back(std::vector<STerm<std::string>>());
    assert (sDCP.add_rule(rule6));

    Rule<std::string, std::string> rule7;
    rule7.lhn = "E";
    rule7.outside_attributes.emplace_back(std::vector<STerm<std::string>>(1,STerm<std::string>(1, Term<std::string>("the"))));
    assert (sDCP.add_rule(rule7));

    Rule<std::string, std::string> rule8;
    rule8.lhn = "F";
    auto term8 = Term<std::string>("on");
    term8.children.push_back(Variable(1, 1));
    rule8.outside_attributes.emplace_back(std::vector<STerm<std::string>>(1,STerm<std::string>(1, term8)));
    rule8.rhs.push_back("F");
    rule8.outside_attributes.push_back(std::vector<STerm<std::string>>(1, STerm<std::string>(1, Variable(0, 1))));
    assert (sDCP.add_rule(rule8));

    Rule<std::string, std::string> rule9;
    rule9.lhn = "F";
    auto term9 = Term<std::string>("issue");
    term9.children.push_back(Variable(0, 1));
    rule9.outside_attributes.emplace_back(std::vector<STerm<std::string>>(1,STerm<std::string>(1, term9)));
    assert (sDCP.add_rule(rule9));

    std::cerr << sDCP;





    for (auto & rule : {rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9}){
        std::cerr << rule.lhn << " " << rule.irank(0) << " " << rule.srank(0) << std::endl;
    }

    std::cerr << std::endl << std::endl;

    SDCPParser<std::string, std::string, int> parser;
    parser.input = tree;
    parser.sDCP = sDCP;

    parser.do_parse();

    auto builder = STermBuilder<std::string, std::string>();

    return 0;
}