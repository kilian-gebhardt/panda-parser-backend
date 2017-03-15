//
// Created by kilian on 14/03/17.
//

#ifndef STERMPARSER_LCFRS_MANAGER_UTIL_H
#define STERMPARSER_LCFRS_MANAGER_UTIL_H
#include "LCFRS_Parser.h"
#include "../Trainer/TraceManager.h"

//namespace LCFR {
    template<typename Nonterminal, typename Terminal, typename TraceID>
    void add_trace_to_manager(const LCFR::LCFRS_Parser<Nonterminal, Terminal> & parser
                              , Trainer::TraceManagerPtr<Nonterminal, TraceID> traceManager
    ) {
        auto hg = parser.convert_trace_to_hypergraph(traceManager->get_node_labels()
                                                     , traceManager->get_edge_labels());
        traceManager->create(0L, hg.first, hg.second);
    }
//}


#endif //STERMPARSER_LCFRS_MANAGER_UTIL_H
