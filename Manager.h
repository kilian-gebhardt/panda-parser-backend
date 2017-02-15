//
// Created by jamb on 03.02.17.
//

#ifndef STERM_PARSER_MANAGER_H
#define STERM_PARSER_MANAGER_H

#include <map>
#include <vector>
#include <memory>

template <typename T1, typename T2>
using MAP = typename std::map<T1, T2>;
using ID = unsigned long;


namespace Manage{

    template <template <typename oID> typename InfoT, typename oID>
    class Manager;


    template <template <typename oID> typename InfoT, typename oID>
    class Element {
    private:
        ID id;
        std::shared_ptr<Manager<InfoT, oID>> manager;

    public:
        Element(){}; // TODO: This constructor is needed for Managing of Hyperedges. Can it be avoided?
        Element(ID aId, std::shared_ptr<Manager<InfoT, oID>> aManager): id(aId), manager(aManager) {};
        Element(const Element<InfoT, oID>& e): id(e.id), manager(e.manager) {};
//        Element(Element<InfoT, oID>&& e): id(std::move(e.id)), manager(std::move(e.manager)) {};

        InfoT<oID>* operator->() {return &((*manager)[id]); }
        inline bool operator==(const Element<InfoT, oID> r) const {return id == r.id; }
        inline bool operator!=(const Element<InfoT, oID> r) const {return id != r.id; }
        inline bool operator<(const Element<InfoT, oID> r) const {return id < r.id; }
        inline bool operator<=(const Element<InfoT, oID> r) const {return id <= r.id; }
        inline bool operator>(const Element<InfoT, oID> r) const {return id <= r.id; }
        inline bool operator>=(const Element<InfoT, oID> r) const {return id <= r.id; }


        friend std::ostream& operator <<(std::ostream& o, const Element<InfoT, oID>& item){
            o << item.id;
            return o;
        }

    };

    template <typename oID>
    class Info {
    private:
        ID id;
        oID originalId;

    public:

        Info(ID aId, oID anOriginalId)
        : id(aId), originalId(anOriginalId) {};


        ID& get_id() {return id; }

        void set_original_id(const oID& anOId){originalId = anOId; }

        const oID& get_original_id() const {return originalId; }
    };




    template <template <typename oID> typename InfoT, typename oID>
    class Manager : public std::enable_shared_from_this<Manager<InfoT,oID>> {
    private:
        std::vector<InfoT<oID>> infos {std::vector<InfoT<oID>>() };

    public:

              InfoT<oID>& operator[](ID id)       {return infos[id]; }
        const InfoT<oID>& operator[](ID id) const {return infos[id]; }

        InfoT<oID>& create_element(const oID anOId){
            const ID id = infos.size();
            infos.push_back(InfoT<oID>(id, this->shared_from_this(), anOId));
            return infos[id];
        }

    };






    //
    // Instantiations
    //




    template <typename oID>
    class Node; // forward declaration

    template <typename oID>
    class HyperEdge : public Info<oID> {
    private:
        Element<Node,oID> outgoing;
        std::vector<Element<Node,oID>> incoming;
        std::shared_ptr<Manager<HyperEdge,oID>> manager;

    public:
        HyperEdge(ID aId
                , std::shared_ptr<Manager<HyperEdge,oID>> aManager
                , Element<Node, oID> anOutg
                , std::vector<Element<Node, oID>> anInc
                , oID anOriginalId)
                : Info<oID>(aId, anOriginalId)
                , outgoing(anOutg)
                , incoming(anInc)
                , manager(aManager){ }

        HyperEdge(ID aId
                , std::shared_ptr<Manager<HyperEdge,oID>> aManager
                , oID anOriginalId)
                : Info<oID>(aId, anOriginalId)
                , manager(aManager){ }

        Element<HyperEdge, oID> get_element() {
            return Element<HyperEdge, oID>(Info<oID>::get_id(), manager);
        };

        void set_outgoing(const Element<Node, oID>& anOutg){
            outgoing = anOutg;
        }

        void set_incoming(const std::vector<Element<Node, oID>>& anInc){
            incoming = anInc;
        }
    };

    template <typename oID>
    class Node : public Info<oID>{
    private:
        std::vector<Element<HyperEdge,oID>> incoming {std::vector<Element<HyperEdge,oID>>() };
        std::vector<std::pair<Element<HyperEdge,oID>, ID>> outgoing {std::vector<std::pair<Element<HyperEdge,oID>,ID>>() };
        std::shared_ptr<Manager<Node,oID>> manager;
    public:

        Node(const ID aId
                , const std::shared_ptr<Manager<Node,oID>> aManager
                , const oID& anOriginalId)
                : Info<oID>(aId, anOriginalId) {
            manager = aManager;
        }

        Element<Node, oID> get_element() {
            return Element<Node, oID>(Info<oID>::get_id(), manager);
        };

        void add_incoming(Element<HyperEdge,oID> inc){
            incoming.push_back(inc);
        }
        void add_incoming(Element<HyperEdge,oID>&& inc){
            incoming.emplace_back(std::move(inc));
        }

        void add_outgoing(const std::pair<Element<HyperEdge,oID>, ID>& out){
            outgoing.push_back(out);
        }

        std::vector<Element<HyperEdge,oID>> get_incoming() const { return incoming; };
        std::vector<std::pair<Element<HyperEdge,oID>, ID>> get_outgoing() const { return outgoing; };

//        const Element<Vertex<oID>> get_element() const;
    };

    template <typename oID>
    class Hypergraph : public Manager<Node,oID> {
    private:
        std::shared_ptr<Manager<HyperEdge,oID>> edges{ std::make_shared<Manager<HyperEdge,oID>>(Manager<HyperEdge,oID>()) };
    public:

        void add_hyperedge(Element<Node,oID> outgoing
                , std::vector<Element<Node, oID>> incoming
                , const oID oId){
            HyperEdge<oID> edge = edges->create_element(oId);
            edge.set_outgoing(outgoing);
            edge.set_incoming(incoming);
            outgoing->add_incoming(edge.get_element());
            for (unsigned long i=0; i<incoming.size(); ++i ){
                incoming[i]->add_outgoing(std::pair<Element<HyperEdge,oID>,unsigned long>(edge.get_element(), i));
            }
        }

    };



}






#endif //STERM_PARSER_MANAGER_H
