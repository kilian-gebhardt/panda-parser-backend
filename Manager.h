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

//        ID get_id() const noexcept {return id; }

        friend std::ostream& operator <<(std::ostream& o, const Element<InfoT, oID>& item){
            o << item.id;
            return o;
        }
        friend std::hash<Element<InfoT, oID>>;

    };


    template <typename oID>
    class Info {
    private:
        ID id;
        oID originalId;

    protected:

        ID get_id() const noexcept {return id; }

    public:
        Info(ID aId, oID anOriginalId)
        : id(std::move(aId)), originalId(std::move(anOriginalId)) {};

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
        std::vector<InfoT<oID>> infos {};

    public:
        using value_type = Element<InfoT, oID>;
        using pointer = Element<InfoT, oID>*;
        using reference = Element<InfoT, oID>&;
        using iterator = ManagerIterator<InfoT, oID>;
        using const_iterator = ManagerIterator<InfoT, oID, true>;



              InfoT<oID>& operator[](ID id)       {assert(id<infos.size()); return infos[id]; }
        const InfoT<oID>& operator[](ID id) const {assert(id<infos.size()); return infos[id]; }

        template <typename... Cargs>
        InfoT<oID>& create(const oID anOId, Cargs... args){
            const ID id = infos.size();
            infos.emplace_back(id, anOId, this->shared_from_this(), std::forward<Cargs>(args)...);
            return infos[id];
        }

        ManagerIterator<InfoT, oID> begin() {return ManagerIterator<InfoT, oID>(0, this->shared_from_this()); }
        ManagerIterator<InfoT, oID> end() {
            return ManagerIterator<InfoT, oID>(infos.size(), this->shared_from_this());
        }

        const ManagerIterator<InfoT, oID, true> cbegin() const {
            return ManagerIterator<InfoT, oID, true>(0, this->shared_from_this());
        }
        const ManagerIterator<InfoT, oID, true> cend() const {
            return ManagerIterator<InfoT, oID, true>(infos.size(), this->shared_from_this());
        }

        unsigned long size() const noexcept {
            return infos.size();
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



}

namespace std {
    template <template <typename oID> typename InfoT, typename oID>
    struct hash<Manage::Element<InfoT, oID>> {
        std::size_t operator()(const Manage::Element<InfoT, oID>& element) const {
            return std::hash<ID>()(element.id);
        }
    };
}




#endif //STERM_PARSER_MANAGER_H
