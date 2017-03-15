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


int main() {
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
        std::clog << "Read in TraceManager with Trace#: " << tm2->size() << "and Trace 1 contrains " << (*tm2)[0].get_hypergraph()->size();

        tm2->serialize(std::cout);
    }




    std::clog << "######################## \n ################## \n ##################### \n Testing Subgraph: \n";

    std::vector<int> graphNLabels {1, 2, 3, 4, 5, 6, 7, 8};
    auto graphNLabelsPtr = std::make_shared<std::vector<int>>(graphNLabels);
    std::vector<EdgeLabelT> graphELabels {42, 43, 44, 45};
    auto graphELabelsPtr = std::make_shared<std::vector<EdgeLabelT>>(graphELabels);
    HypergraphPtr<int> graph = std::make_shared<Hypergraph<int>>(graphNLabelsPtr, graphELabelsPtr);

    std::vector<Element<Node<int>>> graphNodes;
    for(const auto l : graphNLabels)
        graphNodes.push_back(graph->create(l));

    graph->add_hyperedge(42, graphNodes[0], std::vector<Element<Node<int>>>{graphNodes[2], graphNodes[1], graphNodes[3]});
    graph->add_hyperedge(43, graphNodes[1], std::vector<Element<Node<int>>>{graphNodes[4], graphNodes[5]});
    graph->add_hyperedge(44, graphNodes[3], std::vector<Element<Node<int>>>{graphNodes[6], graphNodes[5]});
    graph->add_hyperedge(45, graphNodes[3], std::vector<Element<Node<int>>>{graphNodes[7]});


    std::vector<int> subNLabels {1, 2, 3, 4, 5, 6, 7, 8};
    auto subNLabelsPtr = std::make_shared<std::vector<int>>(subNLabels);
    std::vector<EdgeLabelT> subELabels {42, 43, 44, 45};
    auto subELabelsPtr = std::make_shared<std::vector<EdgeLabelT>>(subELabels);
    HypergraphPtr<int> sub = std::make_shared<Hypergraph<int>>(subNLabelsPtr, subELabelsPtr);


    std::vector<Element<Node<int>>> subNodes;
    for(const auto l : subNLabels)
        subNodes.push_back(sub->create(l));

    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be true (empty)" << std::endl;


    sub->add_hyperedge(42, subNodes[0], std::vector<Element<Node<int>>>{subNodes[3], subNodes[1], subNodes[2]});
    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be true (42)" << std::endl;

    sub->add_hyperedge(43, subNodes[1], std::vector<Element<Node<int>>>{subNodes[5], subNodes[4]});
    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be true (43)" << std::endl;


    sub->add_hyperedge(44, subNodes[3], std::vector<Element<Node<int>>>{subNodes[5], subNodes[6]});
    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be true (44)" << std::endl;

    sub->add_hyperedge(43, subNodes[3], std::vector<Element<Node<int>>>{subNodes[7]});
    std::clog << Manage::is_sub_hypergraph(graph, sub, graphNodes[0], subNodes[0]) << " should be false (wrong edge)" << std::endl;
}