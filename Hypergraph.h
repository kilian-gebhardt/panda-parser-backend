//
// Created by Markus on 21.02.17.
//

#ifndef STERM_PARSER_HYPERGRAPH_H
#define STERM_PARSER_HYPERGRAPH_H

#include "Manager.h"

namespace Manage {

    template <typename NodeT, typename oID>
    class HyperEdge : public Info<oID> {
    private:
        const ManagerPtr<HyperEdge<NodeT, oID>> manager;
        const Element<NodeT> target;
        const std::vector<Element<NodeT>> sources;

    public:
        HyperEdge(ID aId
                , ManagerPtr<HyperEdge<NodeT, oID>> aManager
                , oID anOriginalId
                , Element<NodeT> aTarget
                , std::vector<Element<NodeT>> someSources)
                : Info<oID>(std::move(aId), std::move(anOriginalId))
                , manager(std::move(aManager))
                , target(std::move(aTarget))
                , sources(std::move(someSources)) { }

        Element<HyperEdge<NodeT, oID>> get_element() const noexcept {
            return Element<HyperEdge<NodeT, oID>>(Info<oID>::get_id(), manager);
        };

        const Element<NodeT>& get_target() const {
            return target;
        }

        const std::vector<Element<NodeT>> get_sources(){
            return sources;
        };

    };


    template <typename oID>
    class Node : public Info<oID>{
    private:
//        std::vector<Element<HyperEdge<NodeT>,oID>> incoming {std::vector<Element<HyperEdge,oID>>() };
//        std::vector<std::pair<Element<HyperEdge,oID>, unsigned int>> outgoing;
        ManagerPtr<Node<oID>> manager;
    public:
        Node(const ID aId
                , const ManagerPtr<Node<oID>> aManager
                , const oID& anOriginalId)
                : Info<oID>(std::move(aId)
                , std::move(anOriginalId))
                , manager(std::move(aManager)) { }

        const Element<Node<oID>> get_element() const noexcept {
            return Element<Node<oID>>(Info<oID>::get_id(), manager);
        }

//        void add_incoming(Element<HyperEdge,oID> inc){
//            incoming.push_back(std::move(inc));
//        }

//        void add_outgoing(std::pair<Element<HyperEdge,oID>, ID> out){
//            outgoing.push_back(std::move(out));
//        }

//        const std::vector<Element<HyperEdge,oID>>& get_incoming() const noexcept { return incoming; };
//
//        const std::vector<std::pair<Element<HyperEdge,oID>, unsigned int>>&
//        get_outgoing() const noexcept {
//            return outgoing;
//        }
    };

    template <typename NodeT, typename HEoriginalID>
    class Hypergraph : public Manager<NodeT> {
    private:
        ManagerPtr<HyperEdge<NodeT, HEoriginalID>> edges{ std::make_shared<Manager<HyperEdge<NodeT,HEoriginalID>>>() };
        std::map<Element<NodeT>, std::vector<Element<HyperEdge<NodeT, HEoriginalID>>>> incoming_edges;
        std::map<Element<NodeT>, std::vector<std::pair<Element<HyperEdge<NodeT, HEoriginalID>>, unsigned int>>> outgoing_edges;
    public:

        HyperEdge<NodeT, HEoriginalID>& add_hyperedge(const Element<NodeT>& target
                , const std::vector<Element<NodeT>>& sources
                , const HEoriginalID oId){
            HyperEdge<NodeT, HEoriginalID>& edge = edges->create(oId, target, sources);
            Element<HyperEdge<NodeT, HEoriginalID>> edgeelement = edge.get_element();

            incoming_edges[target].push_back(edgeelement);
            for (unsigned long i=0; i<sources.size(); ++i ){
                outgoing_edges[sources[i]].push_back(std::make_pair(edgeelement, i));
            }

            return edge;
        }


        const std::vector<Element<HyperEdge<NodeT, HEoriginalID>>>& get_incoming_edges(Element<NodeT> e) const {
            return incoming_edges.at(e);
        }


        const std::vector<std::pair<Element<HyperEdge<NodeT, HEoriginalID>>, unsigned int>> get_outgoing_edges(Element<NodeT> e) const {
            if(outgoing_edges.count(e))
                return outgoing_edges.at(e);
            else
                return {};
        }

    };

    template <typename NodeT, typename HEoriginalID>
    using HypergraphPtr = std::shared_ptr<Hypergraph<NodeT, HEoriginalID>>;


}

#endif //STERM_PARSER_HYPERGRAPH_H
