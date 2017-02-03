//
// Created by Markus on 03.02.17.
//

#include <iostream>
#include <memory>
#include <string>
#include "Manager.h"

using namespace Manage;

int main(){
    std::shared_ptr<Hypergraph<std::string>> hg = std::make_shared<Hypergraph<std::string>>();

    const Node<std::string> node = hg->create_element("hello World");

    ID id = node.get_id();

    std::clog << id;

}