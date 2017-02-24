//
// Created by Markus on 03.02.17.
//

#include <iostream>
#include <memory>
#include <string>
#include <map>
#include "Hypergraph.h"

using namespace Manage;

int main(){
    std::shared_ptr<Hypergraph<Node<std::string>, std::string>> hg {std::make_shared<Hypergraph<Node<std::string>, std::string>>() };

    Node<std::string> node1 = hg->create("hello");
    Element<Node<std::string>> e1 = node1.get_element();
    Node<std::string> node2 = hg->create("world");
    Element<Node<std::string>> e2 = node2.get_element();
    Node<std::string> node3 = hg->create("this");


    std::clog << e1->get_original_id();

    std::map<Element<Node<std::string>>, std::string> mymap;
    mymap[e1] = "hallo";
    mymap[e2] = "welt";
    mymap[node3.get_element()] = "Dies";

    std::clog << mymap[e2] << std::endl;


    std::vector<Element<Node<std::string>>> sources {e1, e2};
    Element<Node<std::string>> e3 = node3.get_element();
    hg->add_hyperedge(e3, sources, "edge 1");



    for (auto e : hg->get_outgoing_edges(e2))
        std::clog << "(" << e.first << "," << e.second << ")";
    std::clog << std::endl;




    std::clog << "Fun with iterators:" << std::endl;
    for (auto t : *hg) {
        std::clog << t << " ";
    }

    std::clog << std::endl;

    for (auto const_it = hg->cbegin(); const_it != hg->cend(); ++const_it){
        std::clog << *const_it;
    }

    std::clog << std::endl;

    std::clog << typeid(std::iterator_traits<ManagerIterator<Node<std::string>>>::difference_type).name();
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




}