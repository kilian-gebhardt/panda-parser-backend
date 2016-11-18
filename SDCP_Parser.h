//
// Created by kilian on 18/11/16.
//

#ifndef STERMPARSER_SDCP_PARSER_H
#define STERMPARSER_SDCP_PARSER_H

#include <memory>
#include <vector>
#include <tuple>
#include <queue>
#include "HybridTree.h"
#include "SDCP.h"

template <typename Nonterminal, typename Position>
class ParseItem {
    Nonterminal nonterminal;
    std::vector<std::pair<Position, Position>> spans_inh, spans_syn;
};

template <typename Nonterminal, typename Terminal, typename Position>
class SDCPParser{
public:
    std::queue<std::shared_ptr<ParseItem<Nonterminal, Position>>> agenda;
    void do_parse() {

    }

    bool addToChart(std::shared_ptr<ParseItem<Nonterminal, Position>>) {
        return false;
    }

    ParseItem<Nonterminal, Position> * goal = nullptr;
    HybridTree<Terminal, Position> input;

    SDCP<Nonterminal, Terminal> sDCP;
    // std::vector
};

#endif //STERMPARSER_SDCP_PARSER_H
