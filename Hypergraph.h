//
// Created by Markus on 21.02.17.
//

#ifndef STERM_PARSER_HYPERGRAPH_H
#define STERM_PARSER_HYPERGRAPH_H

#include "Manager.h"

namespace Manage {


    template <typename oID>
    class Node; // forward declaration

    template <typename oID>
    class HyperEdge : public Info<oID> {
    private:
        const ManagerPtr<HyperEdge,oID> manager;
        const Element<Node,oID> outgoing;
        const std::vector<Element<Node,oID>> incoming;

    public:
        HyperEdge(ID aId
                , oID anOriginalId
                , ManagerPtr<HyperEdge,oID> aManager
                , Element<Node, oID> anOutg
                , std::vector<Element<Node, oID>> anInc)
                : Info<oID>(std::move(aId), std::move(anOriginalId))
                , manager(std::move(aManager))
                , outgoing(std::move(anOutg))
                , incoming(std::move(anInc)) { }

        Element<HyperEdge, oID> get_element() const noexcept {
            return Element<HyperEdge, oID>(Info<oID>::get_id(), manager);
        };

    };

    template <typename oID>
    class Node : public Info<oID>{
    private:
        std::vector<Element<HyperEdge,oID>> incoming {std::vector<Element<HyperEdge,oID>>() };
        std::vector<std::pair<Element<HyperEdge,oID>, ID>> outgoing {std::vector<std::pair<Element<HyperEdge,oID>,ID>>() };
        ManagerPtr<Node,oID> manager;
    public:
        Node(const ID aId
                , const oID& anOriginalId
                , const ManagerPtr<Node,oID> aManager)
                : Info<oID>(std::move(aId)
                , std::move(anOriginalId))
                , manager(std::move(aManager)) { }

        const Element<Node, oID> get_element() const noexcept {
            return Element<Node, oID>(Info<oID>::get_id(), manager);
        };

        void add_incoming(Element<HyperEdge,oID> inc){
            incoming.push_back(std::move(inc));
        }

        void add_outgoing(std::pair<Element<HyperEdge,oID>, ID> out){
            outgoing.push_back(std::move(out));
        }

        const std::vector<Element<HyperEdge,oID>>& get_incoming() const noexcept { return incoming; };
        const std::vector<std::pair<Element<HyperEdge,oID>, ID>>& get_outgoing() const noexcept { return outgoing; };
    };

    template <typename oID>
    class Hypergraph : public Manager<Node,oID> {
    private:
        ManagerPtr<HyperEdge,oID> edges{ std::make_shared<Manager<HyperEdge,oID>>() };
    public:

        HyperEdge<oID>& add_hyperedge(const Element<Node,oID>& outgoing
                , const std::vector<Element<Node, oID>>& incoming
                , const oID oId){
            HyperEdge<oID>& edge = edges->create(oId, outgoing, incoming);
            Element<HyperEdge,oID> edgeelement = edge.get_element();

            outgoing->add_incoming(edgeelement);
            for (unsigned long i=0; i<incoming.size(); ++i ){
                incoming[i]->add_outgoing(std::pair<Element<HyperEdge,oID>,unsigned long>(edgeelement, i));
            }

            return edge;
        }

    };

    template <typename oID>
    using HypergraphPtr = std::shared_ptr<Hypergraph<oID>>;





}

#endif //STERM_PARSER_HYPERGRAPH_H
