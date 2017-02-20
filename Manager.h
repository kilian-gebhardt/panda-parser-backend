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
    using ManagerPtr = std::shared_ptr<Manager<InfoT, oID>>;

    template <template <typename oID> typename InfoT, typename oID>
    class Element {
    private:
        ID id;
        ManagerPtr<InfoT, oID> manager;

    public:
        Element(ID aId, ManagerPtr<InfoT, oID> aManager): id(aId), manager(aManager) {};
        Element(const Element<InfoT, oID>& e): id(e.id), manager(e.manager) {};
        Element(Element<InfoT, oID>&& e): id(std::move(e.id)), manager(std::move(e.manager)) {};

        Element<InfoT, oID> operator=(const Element<InfoT, oID>& other){
            if(this != &other) {
                id = other.id;
                manager = other.manager;
            }
            return *this;
        };
        Element<InfoT, oID> operator=(Element<InfoT, oID>&& other){
            if(this != &other) {
                id = std::exchange(other.id, 0);
                manager = std::move(other.manager);
            }
            return *this;
        };

        InfoT<oID>* operator->() const {return &((*manager)[id]); }
        inline bool operator==(const Element<InfoT, oID>& r) const noexcept {return id == r.id; }
        inline bool operator!=(const Element<InfoT, oID>& r) const noexcept {return id != r.id; }
        inline bool operator< (const Element<InfoT, oID>& r) const noexcept {return id < r.id; }
        inline bool operator<=(const Element<InfoT, oID>& r) const noexcept {return id <= r.id; }
        inline bool operator> (const Element<InfoT, oID>& r) const noexcept {return id <= r.id; }
        inline bool operator>=(const Element<InfoT, oID>& r) const noexcept {return id <= r.id; }


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
        : id(std::move(aId)), originalId(std::move(anOriginalId)) {};


        ID get_id() const noexcept {return id; }

        const oID& get_original_id() const noexcept {return originalId; }

        friend std::ostream& operator <<(std::ostream& o, const Info<oID>& item){
            o << "<" << item.id << ">";
            return o;
        }
    };




    template <template <typename oID> typename InfoT, typename oID, bool isconst = false>
    class ManagerIterator; // forward reference


    template <template <typename oID> typename InfoT, typename oID>
    class Manager : public std::enable_shared_from_this<Manager<InfoT,oID>> {
    private:
        std::vector<InfoT<oID>> infos {std::vector<InfoT<oID>>() };

    public:
        using value_type = Element<InfoT, oID>;
        using pointer = Element<InfoT, oID>*;
        using reference = Element<InfoT, oID>&;
        using iterator = ManagerIterator<InfoT, oID>;
        using const_iterator = ManagerIterator<InfoT, oID, true>;



              InfoT<oID>& operator[](ID id)       {return infos[id]; }
        const InfoT<oID>& operator[](ID id) const {return infos[id]; }

        template <typename... Cargs>
        InfoT<oID>& create_element(const oID anOId, Cargs... args){
            const ID id = infos.size();
            infos.emplace_back(id, anOId, this->shared_from_this(), std::forward<Cargs>(args)...);
            return infos[id];
        }

        ManagerIterator<InfoT, oID> begin() {return ManagerIterator<InfoT, oID>(0, this->shared_from_this()); }
        ManagerIterator<InfoT, oID> end() {
            return ManagerIterator<InfoT, oID>(infos.size(), this->shared_from_this());
        }

        const ManagerIterator<InfoT, oID, true> cbegin() {
            return ManagerIterator<InfoT, oID, true>(0, this->shared_from_this());
        }
        const ManagerIterator<InfoT, oID, true> cend() {
            return ManagerIterator<InfoT, oID, true>(infos.size(), this->shared_from_this());
        }

    };

    template <template <typename oID> typename InfoT, typename oID, bool isconst>
    class ManagerIterator {
    private:
        unsigned long index;
        ManagerPtr<InfoT, oID> manager;
    public:
        using difference_type = long;
        using value_type = Element<InfoT, oID>;
        using pointer = typename std::conditional<isconst, const Element<InfoT, oID>*, Element<InfoT, oID>* >::type;
        using reference = typename std::conditional<isconst, const Element<InfoT, oID>, Element<InfoT, oID>>::type;
        using iterator_category = std::random_access_iterator_tag;

        ManagerIterator(): index(0) {}
        ManagerIterator(unsigned long i, const ManagerPtr<InfoT, oID>& m): index(i), manager(m) {}
        ManagerIterator(const ManagerIterator<InfoT, oID, isconst>& mit): index(mit.index), manager(mit.manager) {}
        // todo: enable implicit conversion of non-const iterator to const iterator

        bool operator==(const ManagerIterator<InfoT, oID, isconst>& mit) const noexcept {return mit.index == index; }
        bool operator!=(const ManagerIterator<InfoT, oID, isconst>& mit) const noexcept {return !(*this == mit); }
        bool operator<(const ManagerIterator<InfoT, oID, isconst>& mit) const noexcept {return index < mit.index; }
        bool operator<=(const ManagerIterator<InfoT, oID, isconst>& mit) const noexcept {return index <= mit.index; }
        bool operator>(const ManagerIterator<InfoT, oID, isconst>& mit) const noexcept {return !(*this <= mit); }
        bool operator>=(const ManagerIterator<InfoT, oID, isconst>& mit) const noexcept {return !(*this < mit); }

        value_type operator*() const {return Element<InfoT, oID>(index, manager); }
        value_type operator->() const {return Element<InfoT, oID>(index, manager); }

        ManagerIterator<InfoT, oID, isconst> operator++() { // ++i
            return ManagerIterator<InfoT, oID, isconst>(++index, manager);
        }
        ManagerIterator<InfoT, oID, isconst>& operator++(int) { // i++
            auto result = ManagerIterator<InfoT, oID, isconst>(index, manager);
            ++index;
            return result;
        }
        ManagerIterator<InfoT, oID, isconst> operator--() { // ++i
            return ManagerIterator<InfoT, oID, isconst>(--index, manager);
        }
        ManagerIterator<InfoT, oID, isconst>& operator--(int) { // i++
            auto result = ManagerIterator<InfoT, oID, isconst>(index, manager);
            --index;
            return result;
        }

        ManagerIterator<InfoT, oID, isconst>& operator+=(difference_type shift) {
            index += shift;
            return *this;
        }
        ManagerIterator<InfoT, oID, isconst>& operator-=(difference_type shift) {
            index -= shift;
            return *this;
        }

        ManagerIterator<InfoT, oID, isconst> operator+(difference_type shift) const {
            return ManagerIterator<InfoT, oID, isconst>(index+shift, manager);
        }
        difference_type operator-(const ManagerIterator<InfoT, oID, isconst>& other) const {
            return index - other.index;
        }

        reference operator[](const difference_type i) const {
            return Element<InfoT, oID>(index + i, manager);
        }


        friend class ManagerIterator<InfoT, oID, true>;
        friend class ManagerIterator<InfoT, oID, false>;

    };



    //
    // Instantiations
    //




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

        void add_hyperedge(const Element<Node,oID>& outgoing
                , const std::vector<Element<Node, oID>>& incoming
                , const oID oId){
            HyperEdge<oID> edge = edges->create_element(oId, outgoing, incoming);
            Element<HyperEdge,oID> edgeelement = edge.get_element();

            outgoing->add_incoming(edgeelement);
            for (unsigned long i=0; i<incoming.size(); ++i ){
                incoming[i]->add_outgoing(std::pair<Element<HyperEdge,oID>,unsigned long>(edgeelement, i));
            }
        }

    };



}






#endif //STERM_PARSER_MANAGER_H
