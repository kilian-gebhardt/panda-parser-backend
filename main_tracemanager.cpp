//
// Created by Markus on 22.02.17.
//


#include <iostream>
#include <vector>

#include "DCP/HybridTree.h"
#include "DCP/SDCP.h"
#include "DCP/SDCP_Parser.h"
#include "DCP/util.h"
#include "Legacy/Trace.h"
#include "Trainer/TraceManager.h"
#include "Trainer/TrainingCommon.h"
#include "Trainer/LatentAnnotation.h"
#include "Trainer/TrainerBuilder.h"

using namespace Trainer;

std::shared_ptr<HybridTree<std::string, int>> build_hybrid_tree(bool lcfrs) {
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

    if (lcfrs) {
        std::vector<int> linearization;
        linearization.push_back(7);
        linearization.push_back(4);
        linearization.push_back(1);
        linearization.push_back(5);
        linearization.push_back(8);
        linearization.push_back(14);
        linearization.push_back(12);
        linearization.push_back(10);
        linearization.push_back(2);
        tree->set_linearization(linearization);
    }

    return tree;
};


int main() {
    bool lcfrs = true;
    // std::cout << "Hello, World!" << std::endl;

    HybridTree<std::string, int> tree = *build_hybrid_tree(lcfrs);

    tree.output();

    for (auto p : tree.terminals()) {
        std::cerr << p.first << " " << p.second << std::endl;
    }


    SDCP<std::string, std::string> sDCP;
    sDCP.set_initial("S");

    // Rule 1:
    // std::cout << "rule 1" << std::endl;
    Rule<std::string, std::string> rule1;
    rule1.set_id(0);
    rule1.lhn = "S";
    rule1.rhs.push_back("A");
    rule1.rhs.push_back("B");
    // build lhs
    auto term1 = Term<std::string>("is", 0);
    term1.children.emplace_back(Variable(1, 1));
    term1.children.emplace_back(Variable(2, 1));
    STerm<std::string> arg1;
    arg1.push_back(term1);
    arg1.emplace_back(Term<std::string>(".", 1));
    std::vector<STerm<std::string>> arg1v;
    arg1v.push_back(arg1);
    rule1.inside_attributes.push_back(arg1v);

    // build rhs
    STerm<std::string> arg2;
    arg2.emplace_back(Variable(2, 2));
    std::vector<STerm<std::string>> arg2v;
    arg2v.push_back(arg2);
    rule1.inside_attributes.push_back(arg2v);

    std::vector<STerm<std::string>> arg3v;
    rule1.inside_attributes.push_back(arg3v);
    // std::cout << rule1.inside_attributes.size() << std::endl;

    // constructing LCFRS part
    if (lcfrs) {
        rule1.next_word_function_argument();
        rule1.add_var_to_word_function(1, 1);
        rule1.add_terminal_to_word_function("is");
        rule1.add_var_to_word_function(2, 1);
        rule1.add_terminal_to_word_function(".");
    }
    assert (sDCP.add_rule(rule1));
    // Rule 1 end

    // Rule 2:
    // std::cout << "rule 2" << std::endl;
    Rule<std::string, std::string> rule2;
    rule2.set_id(1);
    rule2.lhn = "A";
    // build lhs
    std::vector<STerm<std::string>> r2_arg1v;
    STerm<std::string> r2_arg_0_1;
    auto r2_term = Term<std::string>("hearing", 1);
    r2_term.children.push_back(Term<std::string>("A", 0));
    r2_term.children.emplace_back(Variable(0, 1));
    r2_arg_0_1.push_back(r2_term);
    r2_arg1v.push_back(r2_arg_0_1);
    rule2.inside_attributes.push_back(r2_arg1v);
    // constructing LCFRS part
    if (lcfrs) {
        rule2.next_word_function_argument();
        rule2.add_terminal_to_word_function("A");
        rule2.add_terminal_to_word_function("hearing");
    }
    assert (sDCP.add_rule(rule2));
    // Rule 2 end

    // Rule 3:
    // std::cout << "rule 3" << std::endl;
    Rule<std::string, std::string> rule3;
    rule3.set_id(2);
    rule3.lhn = "B";
    rule3.rhs.push_back("C");
    rule3.rhs.push_back("D");
    // build lhs
    std::vector<STerm<std::string>> r3_arg1v;
    STerm<std::string> r3_arg_0_1;
    r3_arg_0_1.push_back(Variable(1, 1));
    STerm<std::string> r3_arg_0_2;
    r3_arg_0_2.push_back(Variable(2, 1));
    r3_arg1v.push_back(r3_arg_0_1);
    r3_arg1v.push_back(r3_arg_0_2);
    rule3.inside_attributes.push_back(r3_arg1v);
    rule3.inside_attributes.emplace_back(std::vector<STerm<std::string>>());
    rule3.inside_attributes.emplace_back(std::vector<STerm<std::string>>());
    // constructing LCFRS part
    if (lcfrs) {
        rule3.next_word_function_argument();
        rule3.add_var_to_word_function(1, 1);
        rule3.add_var_to_word_function(2, 1);
        rule3.add_var_to_word_function(1, 2);
    }
    assert (sDCP.add_rule(rule3));
    // Rule 3 end

    // Rule 4:
    // std::cout << "rule 4" << std::endl;
    Rule<std::string, std::string> rule4;
    rule4.set_id(3);
    rule4.lhn = "C";
    // build lhs
    std::vector<STerm<std::string>> r4_arg1v;
    STerm<std::string> r4_arg_0_1;
    auto r4_term = Term<std::string>("scheduled", 0);
    r4_term.children.emplace_back(Term<std::string>("today", 1));
    r4_arg_0_1.push_back(r4_term);
    r4_arg1v.push_back(r4_arg_0_1);
    rule4.inside_attributes.push_back(r4_arg1v);
    if (lcfrs) {
        rule4.next_word_function_argument();
        rule4.add_terminal_to_word_function("scheduled");
        rule4.next_word_function_argument();
        rule4.add_terminal_to_word_function("today");
    }
    assert (sDCP.add_rule(rule4));
    // Rule 4 end

    // Rule 5:
    Rule<std::string, std::string> rule5;
    rule5.set_id(4);
    rule5.lhn = "D";
    // build lhs
    std::vector<STerm<std::string>> r5_arg1v;
    STerm<std::string> r5_arg_0_1;
    auto r5_term = Term<std::string>("on", 0);
    auto r5_term_2 = Term<std::string>("issue", 2);
    r5_term_2.children.emplace_back(Term<std::string>("the", 1));
    r5_term.children.push_back(r5_term_2);
    r5_arg_0_1.push_back(r5_term);
    r5_arg1v.push_back(r5_arg_0_1);
    rule5.inside_attributes.push_back(r5_arg1v);
    if (lcfrs) {
        rule5.next_word_function_argument();
        rule5.add_terminal_to_word_function("on");
        rule5.add_terminal_to_word_function("the");
        rule5.add_terminal_to_word_function("issue");
    }
    assert (sDCP.add_rule(rule5));
    // Rule 5 end

    Rule<std::string, std::string> rule6;
    rule6.lhn = "D";
    rule6.set_id(5);
    // build lhs
    rule6.inside_attributes.emplace_back(std::vector<STerm<std::string>>(1, STerm<std::string>(1, Variable(1, 1))));
    // build rhs
    rule6.rhs.push_back("F");
    rule6.inside_attributes.push_back(std::vector<STerm<std::string>>(1, STerm<std::string>(1, Variable(2, 1))));
    rule6.rhs.push_back("E");
    rule6.inside_attributes.push_back(std::vector<STerm<std::string>>());
    if (lcfrs) {
        rule6.next_word_function_argument();
        rule6.add_var_to_word_function(1, 1);
        rule6.add_var_to_word_function(2, 1);
        rule6.add_var_to_word_function(1, 2);
    }
    assert (sDCP.add_rule(rule6));

    Rule<std::string, std::string> rule7;
    rule7.set_id(6);
    rule7.lhn = "E";
    rule7.inside_attributes.emplace_back(
            std::vector<STerm<std::string>>(
                    1, STerm<std::string>(
                            1, Term<std::string>(
                                    "the"
                                    , 0
                            ))));
    if (lcfrs) {
        rule7.next_word_function_argument();
        rule7.add_terminal_to_word_function("the");
    }
    assert (sDCP.add_rule(rule7));

    Rule<std::string, std::string> rule8;
    rule8.set_id(7);
    rule8.lhn = "F";
    auto term8 = Term<std::string>("on", 0);
    term8.children.push_back(Variable(1, 1));
    rule8.inside_attributes.emplace_back(std::vector<STerm<std::string>>(1, STerm<std::string>(1, term8)));
    rule8.rhs.push_back("G");
    rule8.inside_attributes.push_back(std::vector<STerm<std::string>>(1, STerm<std::string>(1, Variable(0, 1))));
    if (lcfrs) {
        rule8.next_word_function_argument();
        rule8.add_terminal_to_word_function("on");
        //rule8.add_var_to_word_function(1,1);
        rule8.next_word_function_argument();
        rule8.add_var_to_word_function(1, 1);
    }
    assert (sDCP.add_rule(rule8));

    Rule<std::string, std::string> rule9;
    rule9.set_id(8);
    rule9.lhn = "G";
    auto term9 = Term<std::string>("issue", 0);
    term9.children.push_back(Variable(0, 1));
    rule9.inside_attributes.emplace_back(std::vector<STerm<std::string>>(1, STerm<std::string>(1, term9)));
    if (lcfrs) {
        // rule9.next_word_function_argument();
        rule9.next_word_function_argument();
        rule9.add_terminal_to_word_function("issue");
    }
    assert (sDCP.add_rule(rule9));

    std::cerr << sDCP;


    for (auto &rule : {rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9}) {
        std::cerr << rule.lhn << " " << rule.irank(0) << " " << rule.srank(0) << std::endl;
    }

    std::cerr << std::endl << std::endl;

    auto parser = SDCPParser<std::string, std::string, int>(lcfrs, true);
    parser.set_sDCP(sDCP);
    parser.set_input(tree);
    parser.set_goal();

    parser.do_parse();

//    parser.goal = new ParseItem<std::string, int>();
//    parser.goal->nonterminal = "S";
//    parser.goal->spans_syn.emplace_back(std::make_pair(tree.get_entry(), tree.get_exit()));

    parser.reachability_simplification();
    std::cerr << "############ reachability simplification #####" << std::endl;
    parser.print_trace();

    auto builder = STermBuilder<std::string, std::string>();

    for (auto obj : parser.query_trace(*parser.goal)) {
        std::cerr << obj.first << " ";
        for (auto item : obj.second)
            std::cerr << " " << item << " " << std::endl;
    }

    std::cerr << "Trace manager: " << std::endl;

// #############################
// Markus: modifications begin:
// #############################


    std::vector<std::string> nodeLabels{"S", "A", "B", "C", "D", "E", "F", "G"};
    std::vector<EdgeLabelT> edgeLabels{rule1.id, rule2.id, rule3.id, rule4.id, rule5.id, rule6.id, rule7.id, rule8.id,
                                       rule9.id};
    auto const nodeLabelsPtr = std::make_shared<std::vector<std::string>>(nodeLabels);
    auto const edgeLabelsPtr = std::make_shared<std::vector<EdgeLabelT>>(edgeLabels);

    TraceManager<std::string, std::string, int> manager(false);
    TraceManagerPtr<std::string, unsigned long> traceManager{
            std::make_shared<TraceManager2<std::string, unsigned long>>(nodeLabelsPtr, edgeLabelsPtr)};

    manager.add_trace_entry(parser.get_trace(), *parser.goal, 0);

    DCP::add_trace_to_manager(parser, traceManager);
//    std::pair<HypergraphPtr<std::string>, Element<Node<std::string>>> transformedTrace{
//            DCP::transform_trace_to_hypergraph<std::string>(parser, nodeLabels, edgeLabels)};
//    traceManager->create(0L, transformedTrace.first, transformedTrace.second);

    std::cerr << "There are " << (*traceManager)[0].get_hypergraph()->size() << " nodes in the first trace\n";

    std::cerr << "Creating IO-weights...";
    std::vector<Double> my_rule_weights{1, 1, 1, 1, 1, 1, 1, 1, 1};
//    std::vector<Double> my_rule_weights{.2, .4, .6, .8, .5, .3, .1, .9, .7};
    auto pair = manager.io_weights(my_rule_weights, 0);
    auto pair2 = (*traceManager)[0].io_weights(my_rule_weights);
    std::cerr << "Done\n Checking Inside/Outside weights";
    {
        auto i1 = pair.first.cbegin();
        auto i2 = pair.first.cbegin();
        while (i1 != pair.first.cend()) {
            assert((*i1).second == (*i2).second);
            ++i1;
            ++i2;
            std::cerr << ".";
        }
        auto o1 = pair.second.cbegin();
        auto o2 = pair.second.cbegin();
        while (o1 != pair.second.cend()) {
            assert((*o1).second == (*o2).second);
            ++o1;
            ++o2;
            std::cerr << ".";
        }
        std::cerr << " Correct!\n\n";
    }


//    for (const auto & item : manager.get_order(0)) {
//        std::cerr << "T: " << item << " " << pair.first[item] << " " << pair.second[item] << std::endl;
//    }

    const std::vector<std::vector<unsigned>> my_rule_groups{{0},
                                                          {1},
                                                          {2},
                                                          {3},
                                                          {4, 5},
                                                          {6},
                                                          {7},
                                                          {8}};

    std::vector<double> my_rule_weights2{1, 1, 1, 1, 0.5, 0.5, 1, 1, 1};

    auto vec_new = manager.do_em_training<Double>(my_rule_weights2, my_rule_groups, 10);

    auto emTrainerBuilder = Trainer::EMTrainerBuilder();
    Trainer::EMTrainer<std::string, unsigned long> emTrainer = emTrainerBuilder.build_em_trainer(traceManager);
    auto vec_new2 = emTrainer.do_em_training<Double>(my_rule_weights2, my_rule_groups, 10);
//    auto vec_new2 = traceManager->do_em_training<Double>(my_rule_weights2, my_rule_groups, 10);


    for (unsigned i = 0; i < vec_new.size(); ++i) {
        assert(vec_new[i] == vec_new2[i]);
//        std::cerr << vec_new[i] << "/" << vec_new2[i] << "  ";
    }
    std::cerr << "EM training: checked!\n";


    // ############################################
    // ############################################
    // ############################################


    const std::map<std::string, unsigned> mymap{
            {"S", 0},
            {"A", 1},
            {"B", 2},
            {"C", 3},
            {"D", 4},
            {"E", 5},
            {"F", 6},
            {"G", 7}
    };

    auto nont_idx2 = [&](const std::string &nont) -> unsigned {
        return mymap.at(nont);
    };

    auto rule_to_nont_idx = std::vector<std::vector<unsigned>>(9);
    for (auto p : sDCP.lhn_to_rule) {
        for (auto rule : p.second) {
            rule_to_nont_idx[rule->id].push_back(nont_idx2(rule->lhn));
            for (auto nont : rule->rhs) {
                rule_to_nont_idx[rule->id].push_back(nont_idx2(nont));
            }
        }
    }
    std::cerr << std::endl;

    for (unsigned nont = 0; nont < my_rule_groups.size(); ++nont) {
        const auto &group = my_rule_groups[nont];
        for (auto rule_id : group) {
            std::cerr << rule_to_nont_idx[rule_id][0] << " " << nont << std::endl;
            assert(rule_to_nont_idx[rule_id][0] == nont);
        }
    }

    manager.split_merge<Double>(2, 1, vec_new, rule_to_nont_idx, 10, mymap, 4, 0.5, 50.0);


    // traceManager->split_merge<Double>(2, 1, vec_new, rule_to_nont_idx, 10, mymap, 4, 0.5, 50.0);

    std::vector<std::vector<size_t>> rule_to_nont_idx_size_t(9);
    for (auto p : sDCP.lhn_to_rule) {
        for (auto rule : p.second) {
            rule_to_nont_idx_size_t[rule->id].push_back(nont_idx2(rule->lhn));
            for (auto nont : rule->rhs) {
                rule_to_nont_idx_size_t[rule->id].push_back(nont_idx2(nont));
            }
        }
    }

    auto grammarInfo = std::make_shared<const GrammarInfo2>(rule_to_nont_idx_size_t, 0);
    auto splitMergeTrainerBuilder = Trainer::SplitMergeTrainerBuilder<std::string, unsigned long>(traceManager, grammarInfo);
    auto splitMergeTrainer = splitMergeTrainerBuilder
            .set_discriminative_expector(traceManager)
            .set_percent_merger()
            .set_simple_validator(traceManager).build();

    std::vector<size_t> nonterminal_splits(8, 1);
//    convert_format()
    std::vector<std::vector<double>> latentified(vec_new2.size());
    for (size_t idx = 0; idx < vec_new2.size(); ++idx) {
        latentified[idx].push_back(vec_new2[idx]);
    }
    std::vector<double> root_weights(1, 1.0);
    StorageManager storageManager;
    Trainer::LatentAnnotation la(nonterminal_splits, root_weights, latentified, *grammarInfo, storageManager);
    auto la_1 = splitMergeTrainer.split_merge_cycle(la);
    auto la_2 = splitMergeTrainer.split_merge_cycle(la_1);

    return 0;
}
