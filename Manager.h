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
        std::weak_ptr<Manager<InfoT, oID>> manager;

    public:
        Element(ID aId, std::weak_ptr<Manager<InfoT, oID>> aManager): id(aId), manager(aManager) {};

        InfoT<oID>& operator->() {return (*manager)[id]; }
        inline bool operator==(const Element<InfoT, oID> r) {return id == r.id; }
        inline bool operator!=(const Element<InfoT, oID> r) {return id != r.id; }

    };

    template <typename oID>
    class Info {
    private:
        ID id;
//        std::weak_ptr<Manager<Info,oID>> manager;
        oID originalId;

    public:
//        Info(ID aId
//             , std::weak_ptr<Manager<Info, oID>> aManager)
//        : id(aId), manager(aManager) {};

        Info(const ID aId
//             , const std::weak_ptr<Manager<Info,oID>> aManager
             , const oID& anOriginalId)
        : id(aId)
//                , manager(aManager)
                , originalId(anOriginalId) {};


        ID get_id() const {return id; }

        Element<Info, oID> get_element();//{
//            return Element<Info, oID>(id, manager);
//        }

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
        std::weak_ptr<Manager<HyperEdge,oID>> manager;

    public:
        HyperEdge(ID aId
                , std::weak_ptr<Manager<HyperEdge,oID>> aManager
                , oID anOriginalId)
                : Info<oID>(aId, anOriginalId) {

            manager = aManager;
        }

    };

    template <typename oID>
    class Node : public Info<oID>{
    private:
        std::vector<HyperEdge<oID>> incoming;
        std::vector<std::pair<HyperEdge<oID>, ID>> outgoing;
        std::weak_ptr<Manager<Node,oID>> manager;
    public:

        Node(const ID aId
                , const std::weak_ptr<Manager<Node,oID>> aManager
                , const oID& anOriginalId)
                : Info<oID>(aId, anOriginalId) {
            manager = aManager;
        }

        void add_incoming(const Element<HyperEdge,oID>& inc){
            incoming.push_back(inc);
        }

        void add_outgoing(const std::pair<Element<HyperEdge,oID>, ID> out){
            outgoing.push_back(out);
        }

//        const Element<Vertex<oID>> get_element() const;
    };

    template <typename oID>
    class Hypergraph : public Manager<Node,oID> {
    private:
        Manager<HyperEdge,oID> edges;
    public:
        void add_hyperedge(Element<Node,oID>& target, std::vector<Element<Node, oID>>& sources, oID eId){
            HyperEdge<oID> edge = edges.create_element(eId);
            target->add_incoming(edge.get_element());
        }


        void add_hyperedge(ID target, std::vector<ID> sources){

        }
    };



}






#endif //STERM_PARSER_MANAGER_H
