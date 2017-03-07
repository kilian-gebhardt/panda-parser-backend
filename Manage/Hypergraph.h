//
// Created by Markus on 21.02.17.
//

#ifndef STERM_PARSER_HYPERGRAPH_H
#define STERM_PARSER_HYPERGRAPH_H

#include "Manager.h"
#include <algorithm>

namespace Manage {

    template <typename NodeT, typename LabelT>
    class HyperEdge {
    private:
        const ID id;
        const ManagerPtr<HyperEdge<NodeT, LabelT>> manager;
        const LabelT label;
        const size_t labelID;
        const Element<NodeT> target;
        const std::vector<Element<NodeT>> sources;

    protected:
        ID get_id() const noexcept {return id; }

    public:
        HyperEdge(ID aId
                , ManagerPtr<HyperEdge<NodeT, LabelT>> aManager
                , LabelT aLabel
                , size_t aLabelId
                , Element<NodeT> aTarget
                , std::vector<Element<NodeT>> someSources
        )
                : id(aId)
                , manager(std::move(aManager))
                , label(std::move(aLabel))
                , labelID(aLabelId)
                , target(std::move(aTarget))
                , sources(std::move(someSources)) { }

        Element<HyperEdge<NodeT, LabelT>> get_element() const noexcept {
            return Element<HyperEdge<NodeT, LabelT>>(get_id(), manager);
        };


        const LabelT get_label() const noexcept {return label; }
        size_t get_label_id() const noexcept {return labelID; }

        const Element<NodeT>& get_target() const {
            return target;
        }

        const std::vector<Element<NodeT>> get_sources(){
            return sources;
        };

    };


    template <typename LabelT>
    class Node {
    private:
        ID id;
        ManagerPtr<Node<LabelT>> manager;
        LabelT label;
        size_t labelID;

    protected:
        ID get_id() const noexcept {return id; }

    public:
        Node(const ID aId
                , const ManagerPtr<Node<LabelT>> aManager
                , const LabelT& aLabel
                , size_t aLabelId
        )
                : id(aId)
                , manager(std::move(aManager))
                , label(aLabel)
                , labelID(aLabelId)
        { }

        const Element<Node<LabelT>> get_element() const noexcept {
            return Element<Node<LabelT>>(get_id(), manager);
        }

        const LabelT get_label() const noexcept {return label; }
        size_t get_label_id() const noexcept {return labelID; }
    };



    template <typename NodeLabelT, typename EdgeLabelT>
    class Hypergraph : public Manager<Node<NodeLabelT>> {
    private:
        // todo: shared pointer!
        std::vector<NodeLabelT> nodeLabels;
        std::vector<EdgeLabelT> edgeLabels;

        ManagerPtr<HyperEdge<Node<NodeLabelT>, EdgeLabelT>> edges {std::make_shared<Manager<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>>() };
        std::map<Element<Node<NodeLabelT>>, std::vector<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>>> incoming_edges;
        std::map<Element<Node<NodeLabelT>>, std::vector<std::pair<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>, size_t>>> outgoing_edges;

    public:
        Hypergraph(
                std::vector<NodeLabelT> nlabels
                , std::vector<EdgeLabelT> elabels
        )
                : nodeLabels(nlabels)
                , edgeLabels(elabels)
        {}


        Element<Node<NodeLabelT>> create(
                NodeLabelT nLabel
        ){
            size_t nLabelId = std::distance(nodeLabels.cbegin(), std::find(nodeLabels.cbegin(), nodeLabels.cend(), nLabel));
            return Manager<Node<NodeLabelT>>::create(nLabel, nLabelId);
        }


        Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>
        add_hyperedge(
                const EdgeLabelT edgeLabel
                , const Element<Node<NodeLabelT>>& target
                , const std::vector<Element<Node<NodeLabelT>>>& sources
        ){
            size_t edgeLabelId = std::distance(edgeLabels.cbegin(), std::find(edgeLabels.cbegin(), edgeLabels.cend(), edgeLabel));

            Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>> edge = edges->create(edgeLabel, edgeLabelId, target, sources);

            incoming_edges[target].push_back(edge);
            for (size_t i=0; i<sources.size(); ++i ){
                outgoing_edges[sources[i]].push_back(std::make_pair(edge, i));
            }

            return edge;
        }


        const std::vector<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>>&
        get_incoming_edges(Element<Node<NodeLabelT>> e)
        const {
            return incoming_edges.at(e);
        }


        const std::vector<std::pair<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>, size_t>>
        get_outgoing_edges(Element<Node<NodeLabelT>> e)
        const {
            if(outgoing_edges.count(e))
                return outgoing_edges.at(e);
            else
                return {};
        }

    };
    template <typename NodeLabelT, typename EdgeLabelT>
    using HypergraphPtr = std::shared_ptr<Hypergraph<NodeLabelT, EdgeLabelT>>;


}

#endif //STERM_PARSER_HYPERGRAPH_H
