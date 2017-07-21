//
// Created by Markus on 21.02.17.
//

#ifndef STERM_PARSER_HYPERGRAPH_H
#define STERM_PARSER_HYPERGRAPH_H

#include <algorithm>
#include <numeric>
#include "Manager.h"
#include "Manager_util.h"

namespace Manage {


    template<typename NodeLabelT, typename EdgeLabelT>
    class Hypergraph;

    template<typename NodeLabelT, typename EdgeLabelT>
    using HypergraphPtr = std::shared_ptr<Hypergraph<NodeLabelT, EdgeLabelT>>;

    template<typename NodeLabelT, typename EdgeLabelT>
    using HypergraphWeakPtr = std::weak_ptr<Hypergraph<NodeLabelT, EdgeLabelT>>;


    template<typename NodeT, typename LabelT>
    class HyperEdge {
    private:
        const ID id;
        const ManagerWeakPtr<HyperEdge<NodeT, LabelT>> manager;
        const LabelT label;
        const size_t labelID;
        const Element<NodeT> target;
        const std::vector<Element<NodeT>> sources;

    protected:
        ID get_id() const noexcept { return id; }

    public:
        HyperEdge(ID aId, ManagerWeakPtr<HyperEdge<NodeT, LabelT>> aManager, LabelT aLabel, size_t aLabelId,
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
        ManagerWeakPtr<Node<LabelT>> manager;
        LabelT label;
        size_t labelID;

    protected:
        ID get_id() const noexcept { return id; }

    public:
        Node(const ID aId, const ManagerWeakPtr<Node<LabelT>> aManager, const LabelT &aLabel, size_t aLabelId
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
        deserialize(std::istream &in, ID id, ManagerWeakPtr<Node<LabelT>> man) {
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
                const std::shared_ptr<const std::vector<NodeLabelT>> nlabels
                , const std::shared_ptr<const std::vector<EdgeLabelT>> elabels
        )
                : nodeLabels(std::move(nlabels)), edgeLabels(std::move(elabels)) {}


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


            size_t nLabelId = size_t(std::distance(
                    nodeLabels->cbegin(), pos));
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

            size_t edgeLabelId = size_t(std::distance(
                    edgeLabels->cbegin(), pos));
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


        const ManagerWeakPtr<Manage::HyperEdge<Manage::Node<NodeLabelT>, EdgeLabelT>> get_edges() const {
            return edges;
        }

        const std::shared_ptr<const std::vector<NodeLabelT>> get_node_labels(){
            return nodeLabels;
        }

        const std::shared_ptr<const std::vector<EdgeLabelT>> get_edge_labels(){
            return edgeLabels;
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

        const Element<Node<NodeLabelT>> get_node_by_label(NodeLabelT label){
            size_t id = size_t(std::distance(nodeLabels->cbegin(), std::find(nodeLabels->cbegin(),nodeLabels->cend(), label)));
            assert(id < nodeLabels->size());
            return Manager<Node<NodeLabelT>>::infos[id].get_element();
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

    template <typename NodeLabelT, typename EdgeLabelT>
    bool is_sub_hypergraph(
            const HypergraphPtr<NodeLabelT, EdgeLabelT>& graph
            , const HypergraphPtr<NodeLabelT, EdgeLabelT>& sub
            , const Element<Node<NodeLabelT>>& gNode
            , const Element<Node<NodeLabelT>>& sNode
    ){
        if(gNode->get_label() != sNode->get_label())
            return false;


        const std::vector<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>>& sEdges = sub->get_incoming_edges(sNode);
        std::vector<Element<HyperEdge<Node<NodeLabelT>, EdgeLabelT>>> gEdges = graph->get_incoming_edges(gNode);

        if(sEdges.size() > gEdges.size())
            return false;

        bool result = false;
        std::vector<size_t> perm(gEdges.size());
        std::iota(perm.begin(), perm.end(), 0);
        do {
            bool foundSubgraph = true;
            for(size_t i = 0; i < sEdges.size() && foundSubgraph; ++i){
                if(gEdges[perm[i]]->get_label() != sEdges[i]->get_label()) {
                    foundSubgraph = false;
                    break;
                }

                const auto& gSources = gEdges[perm[i]]->get_sources();
                const auto& sSources = sEdges[i]->get_sources();
                if(gSources.size() != sSources.size()){
                    foundSubgraph = false;
                    break;
                }
                std::vector<size_t> perm2(gSources.size());
                std::iota(perm2.begin(), perm2.end(), 0);
                bool foundSubEdge = false;
                do {
                    bool allSourcesFound = true;
                    for(size_t j = 0; j < gSources.size() && allSourcesFound; ++j){
                        allSourcesFound = allSourcesFound
                                        && is_sub_hypergraph(graph, sub, gSources[perm2[j]], sSources[j]);
                    }
                    foundSubEdge = foundSubEdge || allSourcesFound;
                } while(!foundSubEdge && std::next_permutation(perm2.begin(), perm2.end()));
                foundSubgraph = foundSubgraph && foundSubEdge;
            }
            result = result || foundSubgraph;
        } while(!result && std::next_permutation(perm.begin(), perm.end()));

        return result;
    }


}

#endif //STERM_PARSER_HYPERGRAPH_H
