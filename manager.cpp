//
// Created by Markus on 03.02.17.
//

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <map>
#include "Manage/Hypergraph.h"
#include "Trainer/TraceManager.h"
#include "util.h"
#include "Trainer/LatentAnnotation.h"
#include "Trainer/AnnotationProjection.h"


void test_main();
void test_subgraph();
void test_fp_io();


template<typename T1, typename T2>
using MAPTYPE = typename std::unordered_map<T1, T2>;


int main() {

    test_main();

    test_subgraph();

    test_fp_io();

}


void test_main() {
    std::vector<std::string> nLabels{"hello", "world", "this"};
    auto const nLabelsPtr = std::make_shared<const std::vector<std::string>>(nLabels);
    std::vector<EdgeLabelT> eLabels{42};
    auto const eLabelsPtr = std::make_shared<const std::vector<EdgeLabelT>>(eLabels);


    HypergraphPtr<std::string> hg = std::make_shared<Hypergraph<std::string>>(
            nLabelsPtr
            , eLabelsPtr
    );

    auto node1 = hg->create("hello");
    auto node2 = hg->create("world");
    auto node3 = hg->create("this");


    std::clog << node1->get_label();

    std::map<Element<Node<std::string>>, std::string> mymap;
    mymap[node1] = "hallo";
    mymap[node2] = "welt";
    mymap[node3] = "Dies";

    std::clog << mymap[node2] << std::endl;


    std::vector<Element<Node<std::string>>> sources{node1, node2};
    hg->add_hyperedge(42, node3, sources);


    for (auto e : hg->get_outgoing_edges(node2))
        std::clog << "(" << e.first << "," << e.second << ")";
    std::clog << std::endl;


    std::clog << "Fun with iterators:" << std::endl;
    for (auto t : *hg) {
        std::clog << t << " ";
    }

    std::clog << std::endl;

    for (auto const_it = hg->cbegin(); const_it != hg->cend(); ++const_it) {
        std::clog << *const_it;
    }

    std::clog << std::endl;

    std::clog << typeid(std::iterator_traits<Manage::ManagerIterator<Node<std::string>>>::difference_type).name();
    std::clog << std::endl;

    auto it = hg->begin();
    auto it2 = it + 1;
    it += 2;
    std::clog << *it;
    std::clog << *it2;
    it -= 1;
    std::clog << *it;
    std::clog << (it[2]);
    std::clog << (hg->end() - it);
    std::clog << std::endl;


    Trainer::TraceManagerPtr<std::string, std::string> traceManager
            = std::make_shared<Trainer::TraceManager2<std::string, std::string>>(
                    nLabelsPtr
                    , eLabelsPtr
            );
    traceManager->create("The best HG!", hg, node1);


    std::filebuf fb;
    if (fb.open("/tmp/manager.txt", std::ios::out)) {
        std::ostream out(&fb);
        traceManager->serialize(out);
        fb.close();
    }

    if (fb.open("/tmp/manager.txt", std::ios::in)) {
        std::istream in(&fb);
        Trainer::TraceManagerPtr<std::string, std::string> tm2 = Trainer::TraceManager2<std::string
                                                                                        , std::string
        >::deserialize(in);
        std::clog << "Read in TraceManager with Trace#: " << tm2->size() << "and Trace 1 contrains "
                  << (*tm2)[0].get_hypergraph()->size();

        tm2->serialize(std::cout);
    }
}





void test_subgraph() {
    std::clog << "######################## \n ################## \n ##################### \n Testing Subgraph: \n";

    std::vector<int> graphNLabels{1, 2, 3, 4, 5, 6, 7, 8};
    auto graphNLabelsPtr = std::make_shared<std::vector<int>>(graphNLabels);
    std::vector<EdgeLabelT> graphELabels{42, 43, 44, 45};
    auto graphELabelsPtr = std::make_shared<std::vector<EdgeLabelT>>(graphELabels);
    HypergraphPtr<int> graph = std::make_shared<Hypergraph<int>>(graphNLabelsPtr, graphELabelsPtr);

    std::vector<Element<Node<int>>> graphNodes;
    for (const auto l : graphNLabels)
        graphNodes.push_back(graph->create(l));

    graph->add_hyperedge(
            42, graphNodes[0], std::vector<Element<Node<int>>>{graphNodes[2], graphNodes[1], graphNodes[3]}
    );
    graph->add_hyperedge(43, graphNodes[1], std::vector<Element<Node<int>>>{graphNodes[4], graphNodes[5]});
    graph->add_hyperedge(44, graphNodes[3], std::vector<Element<Node<int>>>{graphNodes[6], graphNodes[5]});
    graph->add_hyperedge(45, graphNodes[3], std::vector<Element<Node<int>>>{graphNodes[7]});


    std::vector<int> subNLabels{1, 2, 3, 4, 5, 6, 7, 8};
    auto subNLabelsPtr = std::make_shared<std::vector<int>>(subNLabels);
    std::vector<EdgeLabelT> subELabels{42, 43, 44, 45};
    auto subELabelsPtr = std::make_shared<std::vector<EdgeLabelT>>(subELabels);
    HypergraphPtr<int> sub = std::make_shared<Hypergraph<int>>(subNLabelsPtr, subELabelsPtr);


    std::vector<Element<Node<int>>> subNodes;
    for (const auto l : subNLabels)
        subNodes.push_back(sub->create(l));

    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be true (empty)"
              << std::endl;


    sub->add_hyperedge(42, subNodes[0], std::vector<Element<Node<int>>>{subNodes[3], subNodes[1], subNodes[2]});
    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be true (42)"
              << std::endl;

    sub->add_hyperedge(43, subNodes[1], std::vector<Element<Node<int>>>{subNodes[5], subNodes[4]});
    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be true (43)"
              << std::endl;


    sub->add_hyperedge(44, subNodes[3], std::vector<Element<Node<int>>>{subNodes[5], subNodes[6]});
    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be true (44)"
              << std::endl;

    sub->add_hyperedge(43, subNodes[3], std::vector<Element<Node<int>>>{subNodes[7]});

    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be false (wrong edge)"
              << std::endl;

}








void test_fp_io() {
    std::clog
            << "\n######################## \n ################## \n ##################### \n Testing Fixpoint IO: \n";

    std::vector<int> graphNLabels{0, 1, 2, 3, 4, 5, 6, 7, 8};
    auto graphNLabelsPtr = std::make_shared<std::vector<int>>(graphNLabels);
    std::vector<EdgeLabelT> graphELabels{41, 42, 43, 44, 45, 46, 47};
    auto graphELabelsPtr = std::make_shared<std::vector<EdgeLabelT>>(graphELabels);
    HypergraphPtr<int> graph = std::make_shared<Hypergraph<int>>(graphNLabelsPtr, graphELabelsPtr);

    std::vector<Element<Node<int>>> graphNodes;
    for (const auto l : graphNLabels)
        graphNodes.push_back(graph->create(l));

    graph->add_hyperedge(41, graphNodes[0], std::vector<Element<Node<int>>>{graphNodes[1], graphNodes[2]});
    graph->add_hyperedge(42, graphNodes[1], std::vector<Element<Node<int>>>{graphNodes[3]});
    graph->add_hyperedge(43, graphNodes[2], std::vector<Element<Node<int>>>{graphNodes[3]});
    graph->add_hyperedge(44, graphNodes[2], std::vector<Element<Node<int>>>{graphNodes[1], graphNodes[2]});
    graph->add_hyperedge(45, graphNodes[3], std::vector<Element<Node<int>>>());
    graph->add_hyperedge(46, graphNodes[0], std::vector<Element<Node<int>>>{graphNodes[4]});
    graph->add_hyperedge(47, graphNodes[4], std::vector<Element<Node<int>>>());

    std::vector<Double> ruleWeights {0.8, 1, 0.7, 0.3, 1, 0.2, 1};

    auto tMPtr = std::make_shared<Trainer::TraceManager2<int, int>>(graphNLabelsPtr, graphELabelsPtr);

    tMPtr->create(1, graph, graphNodes[0]);

    auto w = (*tMPtr)[0].io_weights(ruleWeights);
    std::cerr << "Inside Weights:\n";
    for (auto i : w.first)
        std::cerr << i.first << ": " << i.second << "     ";
    std::cerr << std::endl;

    std::cerr << "Outside Weights:\n";
    for (auto i : w.second)
        std::cerr << i.first << ": " << i.second << "     ";
    std::cerr << std::endl;


    auto top = (*tMPtr)[0].get_topological_order();
    for(const auto &i : top){
        std::cerr << i << " ";
    }


    // Latent annotations
    std::cerr << "IO with Latent Annotations\n";

    std::vector<Trainer::RuleTensor<double>> ruleWs {};
    ruleWs.emplace_back(Trainer::RuleTensorRaw<double, 3>{1,1,1});
    boost::get<Trainer::RuleTensorRaw<double, 3>>(ruleWs[0]).setValues({{{0.8}}});
    ruleWs.emplace_back(Trainer::RuleTensorRaw<double, 2>{1,1});
    boost::get<Trainer::RuleTensorRaw<double, 2>>(ruleWs[1]).setValues({{1}});
    ruleWs.emplace_back(Trainer::RuleTensorRaw<double, 2>{1,1});
    boost::get<Trainer::RuleTensorRaw<double, 2>>(ruleWs[2]).setValues({{0.7}});
    ruleWs.emplace_back(Trainer::RuleTensorRaw<double, 3>{1,1,1});
    boost::get<Trainer::RuleTensorRaw<double, 3>>(ruleWs[3]).setValues({{{0.3}}});
    ruleWs.emplace_back(Trainer::RuleTensorRaw<double, 1>{1});
    boost::get<Trainer::RuleTensorRaw<double, 1>>(ruleWs[4]).setValues({1.0});
    ruleWs.emplace_back(Trainer::RuleTensorRaw<double, 2>{1,1});
    boost::get<Trainer::RuleTensorRaw<double, 2>>(ruleWs[5]).setValues({{0.2}});
    ruleWs.emplace_back(Trainer::RuleTensorRaw<double, 1>{1});
    boost::get<Trainer::RuleTensorRaw<double, 1>>(ruleWs[6]).setValues({1.0});

    std::cerr << "RuleWs done \n";


    Eigen::Tensor<double, 1> rootWs(1);
    rootWs.setValues({1.0});

    Eigen::TensorRef<Eigen::Tensor<double,1>> root;
    root = rootWs;

    MAPTYPE<Element<Node<int>>, Trainer::WeightVector> insideWeights;
    MAPTYPE<Element<Node<int>>, Trainer::WeightVector> outsideWeights;
    for(auto n : *graph){
        insideWeights[n] = Trainer::WeightVector{1};
        outsideWeights[n] = Trainer::WeightVector{1};
    }

    std::cerr << "Starting IO by fixpoint_la\n";

    (*tMPtr)[0].io_weights_fixpoint_la(ruleWs, root, insideWeights, outsideWeights);

    std::cerr << "Inside: ";
    for(auto n : *graph){
        std::cerr << insideWeights[n](0) << "  ";
    }
    std::cerr << std::endl;


    std::cerr << "Outside: ";
    for(auto n : *graph){
        std::cerr << outsideWeights[n](0) << "  ";
    }
    std::cerr << std::endl;





    std::cerr << "#### Testing Projection ####\n";

    std::vector<std::vector<size_t>> rules_to_nont {{0,1,2}, {1,3},{2,3},{2,1,2},{3},{0,4},{4}};
    const std::vector<size_t> splits{1,1,1,1,1};


    Trainer::LatentAnnotation lat(splits, std::move(rootWs), std::move(std::make_unique<std::vector<Trainer::RuleTensor<double>>>(ruleWs)));
    Trainer::GrammarInfo2 gramInf(rules_to_nont, 0);

    std::cerr << "before:\n";
    for(const auto tensor : *lat.ruleWeights){
        std::cerr << tensor << "\n";
    }

    Trainer::LatentAnnotation projection = Trainer::project_annotation<size_t>(std::move(lat), gramInf);

    std::cerr << "after:\n";
    for(const auto tensor : *projection.ruleWeights){
        std::cerr << tensor << "\n";
    }


}