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
    std::vector<EdgeLabelT> eLabels{42};
    HypergraphPtr<std::string> hg = std::make_shared<Hypergraph<std::string>>(nLabels, eLabels);

    auto node1 = hg->create("hello\te\te\naa");
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
            = std::make_shared<Trainer::TraceManager2<std::string, std::string>>();
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
        std::clog << "Read in TraceManager with Trace#: " << tm2->size();
    }

}