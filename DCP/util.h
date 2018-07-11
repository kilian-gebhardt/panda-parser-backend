//
// Created by kilian on 08/03/17.
//

#ifndef STERMPARSER_DCP_UTIL_H
#define STERMPARSER_DCP_UTIL_H
#include "../Names.h"
#include "SDCP.h"
#include "SDCP_Parser.h"
#include <tuple>
#include "../Trainer/TraceManager.h"

namespace DCP {
    template<typename Nonterminal, typename Terminal, typename Position>
    std::pair<HypergraphPtr<Nonterminal>, Element<Node<Nonterminal>>>
    transform_trace_to_hypergraph(
            const SDCPParser<Nonterminal, Terminal, Position> &parser
            , const std::shared_ptr<const std::vector<Nonterminal>>& nodeLabels
            , const std::shared_ptr<const std::vector<EdgeLabelT>>& edgeLabels
    ) {
        const MAPTYPE<ParseItem<Nonterminal, Position>, std::vector<std::pair<std::shared_ptr<Rule<Nonterminal
                                                                                                   , Terminal>>
                                                                              , std::vector<std::shared_ptr<ParseItem<
                        Nonterminal
                        , Position>> >>>> &trace(parser.get_trace());

        HypergraphPtr<Nonterminal> hg{std::make_shared<Hypergraph<Nonterminal>>(nodeLabels, edgeLabels)};

        // construct all nodes
        auto nodelist = std::map<ParseItem<Nonterminal, Position>, Element<Node<Nonterminal>>>();
        for (auto const &item : trace)
            nodelist.emplace(item.first, hg->create(item.first.nonterminal));

        // construct hyperedges
        for (auto const &item : trace) {
            Element<Node<Nonterminal >> outgoing = nodelist.at(item.first);
            std::vector<Element<Node<Nonterminal>>> incoming;
            for (auto const &parse : item.second) {
                incoming.clear();
                incoming.reserve(parse.second.size());
                for (auto const &pItem : parse.second)
                    incoming.push_back(nodelist.at(*pItem));

                /*Element<HyperEdge<Nonterminal>>& edge = */
                hg->add_hyperedge((size_t) parse.first->id, outgoing, incoming);
                // set optional infos on edge here
            }

        }

        return std::make_pair(hg, nodelist.at(*parser.goal));
    };

    template<typename Nonterminal, typename Terminal, typename Position, typename TraceID>
    void add_trace_to_manager(const SDCPParser<Nonterminal, Terminal, Position> & parser
                              , Trainer::TraceManagerPtr<Nonterminal, TraceID> traceManager
                              , double frequency = 1.0
    ) {
        std::pair<HypergraphPtr<Nonterminal>, Element<Node<Nonterminal>>> transformedTrace{
                DCP::transform_trace_to_hypergraph<Nonterminal>(
                        parser
                        , traceManager->get_node_labels()
                        , traceManager->get_edge_labels()
                )};
        traceManager->create(0L, transformedTrace.first, transformedTrace.second, frequency);
    }
}
#endif //STERMPARSER_DCP_UTIL_H
