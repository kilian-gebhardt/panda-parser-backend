//
// Created by Markus on 21.02.17.
//

#ifndef STERM_PARSER_HYPERGRAPH_H
#define STERM_PARSER_HYPERGRAPH_H

#include <algorithm>
#include "Manager.h"
#include "Manager_util.h"

namespace Manage {


    template<typename NodeLabelT, typename EdgeLabelT>
    class Hypergraph;

    template<typename NodeLabelT, typename EdgeLabelT>
    using HypergraphPtr = std::shared_ptr<Hypergraph<NodeLabelT, EdgeLabelT>>;


    template<typename NodeT, typename LabelT>
    class HyperEdge {
    private:
        const ID id;
        const ManagerPtr<HyperEdge<NodeT, LabelT>> manager;
        const LabelT label;
        const size_t labelID;
        const Element<NodeT> target;
        const std::vector<Element<NodeT>> sources;

    protected:
        ID get_id() const noexcept { return id; }

    public:
        HyperEdge(ID aId, ManagerPtr<HyperEdge<NodeT, LabelT>> aManager, LabelT aLabel, size_t aLabelId,
                  Element<NodeT> aTarget, std::vector<Element<NodeT>> someSources
        )
                : id(aId), manager(std::move(aManager)), label(std::move(aLabel)), labelID(aLabelId),
                  target(std::move(aTarget)), sources(std::move(someSources)) {}

        Element<HyperEdge<NodeT, LabelT>> get_element() const noexcept {
            return Element<HyperEdge<NodeT, LabelT>>(get_id(), manager);
        };


        const LabelT get_label() const noexcept { return label; }

        size_t get_label_id() const noexcept { return labelID; }

        const Element<NodeT> &get_target() const {
            return target;
        }

        const std::vector<Element<NodeT>> get_sources() {
            return sources;
        };




        // Serialization works for LabelT of std::string and size_t


        template<typename T = LabelT>
        typename std::enable_if_t<
                std::is_same<T, std::string>::value || std::is_same<T, size_t>::value
                , void
        >
        serialize(std::ostream &out) const {
            out << labelID << ';';
            serialize_string_or_size_t(out, label);
            out << ';' << target << ';' << sources.size() << ';';
            for (auto const &source : sources)
                out << source << ';';
            out << std::endl;

        }

        template<typename NodeLabelT, typename T = LabelT>
        static
        typename std::enable_if_t<
                std::is_same<T, std::string>::value || std::is_same<T, size_t>::value
                , HyperEdge<NodeT, LabelT>
        >
        deserialize(
                std::istream &in
                , ID id
                , ManagerPtr<HyperEdge<NodeT, LabelT>> man
                , HypergraphPtr<NodeLabelT, LabelT> hg
        ) {
            LabelT l;
            size_t lID;
            int noSources = 0;
            char sep;
            ID nodeID;
            in >> lID;
            in >> sep; //read in the separator
            deserialize_string_or_size_t(in, l);

            in >> sep; //read in the separator
            in >> nodeID;
            Element<NodeT> target = Element<NodeT>(nodeID, hg);
            in >> sep; // read in the separator
            in >> noSources;
            in >> sep;
            std::vector<Element<NodeT>> sources;
            for(int i = 0; i < noSources; ++i){
                in >> nodeID;
                sources.emplace_back(nodeID, hg);
                in >> sep; // read in the saparator
            }

            return HyperEdge<NodeT, LabelT>(id, man, l, lID, target, sources);
        }
    };


    template<typename LabelT>
    class Node {
    private:
        ID id;
        ManagerPtr<Node<LabelT>> manager;
        LabelT label;
        size_t labelID;

    protected:
        ID get_id() const noexcept { return id; }

    public:
        Node(const ID aId, const ManagerPtr<Node<LabelT>> aManager, const LabelT &aLabel, size_t aLabelId
        )
                : id(aId), manager(std::move(aManager)), label(aLabel), labelID(aLabelId) {}

        const Element<Node<LabelT>> get_element() const noexcept {
            return Element<Node<LabelT>>(get_id(), manager);
        }

        const LabelT get_label() const noexcept { return label; }

        size_t get_label_id() const noexcept { return labelID; }


        // Serialization works for LabelT of std::string and size_t
        template<typename T = LabelT>
        typename std::enable_if_t<
                std::is_same<T, std::string>::value || std::is_same<T, size_t>::value
                , void
        >
        serialize(std::ostream &out) const {
            out << labelID << ';';
            serialize_string_or_size_t(out, label);
            out << std::endl;
        }


        template<typename T = LabelT>
        static
        typename std::enable_if_t<
                std::is_same<T, std::string>::value || std::is_same<T, size_t>::value
                , Node<LabelT>
        >
        deserialize(std::istream &in, ID id, ManagerPtr<Node<LabelT>> man) {
            LabelT l;
            size_t lID;
            char sep;
            in >> lID;
            in >> sep;
            deserialize_string_or_size_t(in, l);
            return Node<LabelT>(id, man, l, lID);
        }

    };


    template<typename NodeLabelT, typename EdgeLabelT>
    class Hypergraph : public Manager<Node<NodeLabelT>> {
    private:
        std::shared_ptr<const std::vector<NodeLabelT>> nodeLabels;
        std::shared_ptr<const std::vector<EdgeLabelT>> edgeLabels;

        ManagerPtr<HyperEdge<Node<NodeLabelT>, EdgeLabelT>> edges{
                std::make_shared<Manager<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>>()};
        std::map<Element<Node<NodeLabelT>>, std::vector<Element<HyperEdge<Node<NodeLabelT>
                                                                          , EdgeLabelT>>>> incoming_edges;
        std::map<Element<Node<NodeLabelT>>, std::vector<std::pair<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>
                                                                  , size_t>>
        > outgoing_edges;

    public:
        Hypergraph(
                const std::shared_ptr<const std::vector<NodeLabelT>>& nlabels
                , const std::shared_ptr<const std::vector<EdgeLabelT>>& elabels
        )
                : nodeLabels(nlabels), edgeLabels(elabels) {}


        Element<Node<NodeLabelT>> create(
                NodeLabelT nLabel
        ) {

            typename std::vector<NodeLabelT>::const_iterator pos = std::find(
                    nodeLabels->cbegin()
                    , nodeLabels->cend()
                    , nLabel
            );
            if(pos == nodeLabels->cend()){
                std::cerr << "Could not find node label '" << nLabel << "'" << std::endl;
                exit(-1);
            }


            size_t nLabelId = std::distance(
                    nodeLabels->cbegin(), pos);
            return Manager<Node<NodeLabelT>>::create(nLabel, nLabelId);
        }


        Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>
        add_hyperedge(
                const EdgeLabelT edgeLabel
                , const Element<Node<NodeLabelT>> &target
                , const std::vector<Element<Node<NodeLabelT>>> &sources
        ) {

            typename std::vector<EdgeLabelT>::const_iterator pos = std::find(
                    edgeLabels->cbegin()
                    , edgeLabels->cend()
                    , edgeLabel
            );
            if(pos == edgeLabels->cend()) {
                std::cerr << "Could not find edge label '" << edgeLabel << "'" << std::endl;
                exit(-1);
            }

            size_t edgeLabelId = std::distance(
                    edgeLabels->cbegin(), pos);
            // todo: abort if label not valid
            Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>> edge = edges->create(
                    edgeLabel
                    , edgeLabelId
                    , target
                    , sources
            );

            incoming_edges[target].push_back(edge);
            for (size_t i = 0; i < sources.size(); ++i) {
                outgoing_edges[sources[i]].push_back(std::make_pair(edge, i));
            }

            return edge;
        }


        const std::vector<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>>&
        get_incoming_edges(Element<Node<NodeLabelT>> e)
        {
            return incoming_edges[e];
        }


        const std::vector<std::pair<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>, size_t>>&
        get_outgoing_edges(Element<Node<NodeLabelT>> e)
        {
            return outgoing_edges[e];
        }


        // Serialization only possible for NodeLabelT/EdgeLabelT as std::string or size_t


        template<typename T1 = NodeLabelT, typename T2 = EdgeLabelT>
        typename std::enable_if_t<
                (std::is_same<T1, std::string>::value || std::is_same<T1, size_t>::value)
                &&
                (std::is_same<T2, std::string>::value || std::is_same<T2, size_t>::value)
                , void
        >
        serialize(std::ostream& o) {
            o << "Hypergraph Version 1" << std::endl;

            o << Manager<Node<NodeLabelT>>::size() << " Nodes:" << std::endl;
            for(const auto& node : *this) {
                node->serialize(o);
            }

            o << edges->size() << " Edges:" << std::endl;
            for(const auto& edge : *edges) {
                edge->serialize(o);
            }

        }


        template<typename T1 = NodeLabelT, typename T2 = EdgeLabelT>
        static
        typename std::enable_if_t<
                            (std::is_same<T1, std::string>::value || std::is_same<T1, size_t>::value)
                            &&
                            (std::is_same<T2, std::string>::value || std::is_same<T2, size_t>::value)
                , HypergraphPtr<NodeLabelT, EdgeLabelT>
                >
        deserialize(
                std::istream& in
                , std::shared_ptr<const std::vector<NodeLabelT>> nLabels
                , std::shared_ptr<const std::vector<EdgeLabelT>> eLabels
        ){
            std::string line;
            std::getline(in, line); // read the rest of the line (only linebreak)
            std::getline(in, line);
            if(line != "Hypergraph Version 1")
                throw std::string("Version Mismatch for Hypergraph!");

            size_t noItems;
            in >> noItems;
            std::getline(in, line);
            if(line != " Nodes:")
                throw std::string("Unexpected line '" + line + "' expected ' Nodes:'");
            HypergraphPtr<NodeLabelT, EdgeLabelT> hypergraph
                    = std::make_shared<Hypergraph<NodeLabelT, EdgeLabelT>>(nLabels, eLabels);

            for(size_t i = 0; i < noItems; ++i){
                Node<NodeLabelT> node = Node<NodeLabelT>::deserialize(in, i, hypergraph);
                hypergraph->create(node.get_label()); // todo: add it directly?
            }

            noItems = 0;
            in >> noItems;
            std::getline(in, line);
            if(line != " Edges:")
                throw std::string("Unexpected line '" + line + "' expected ' Edges:'");
            for(size_t i = 0; i < noItems; ++i){
                HyperEdge<Node<NodeLabelT>, EdgeLabelT> edge =
                        HyperEdge<Node<NodeLabelT>, EdgeLabelT>::deserialize(in, i, hypergraph->edges, hypergraph);
                hypergraph->add_hyperedge(edge.get_label(), edge.get_target(), edge.get_sources());
            }

            return hypergraph;
        }




    };


}

#endif //STERM_PARSER_HYPERGRAPH_H
