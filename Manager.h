//
// Created by jamb on 03.02.17.
//

#ifndef STERM_PARSER_MANAGER_H
#define STERM_PARSER_MANAGER_H

#include <map>
#include <vector>
#include <memory>
#include <cassert>

template <typename T1, typename T2>
using MAP = typename std::map<T1, T2>;


namespace Manage{

    using ID = unsigned long;

    template <typename InfoT>
    class Manager;

    template <typename InfoT>
    using ManagerPtr = std::shared_ptr<Manager<InfoT>>;
    template <typename InfoT>
    using ConstManagerPtr = std::shared_ptr<const Manager<InfoT>>;


    template <typename InfoT, bool isConst = false>
    class Element {
    private:
        ID id;
//        ManagerPtr<InfoT> manager;
        using ManagerType = typename std::conditional<isConst, ConstManagerPtr<InfoT>, ManagerPtr<InfoT>>::type;
        using PointerType = typename std::conditional<isConst, const InfoT*, InfoT*>::type;
        ManagerType manager;
    public:
        Element(ID aId, ManagerType aManager): id(aId), manager(aManager) {};

        PointerType operator->() const {return &((*manager)[id]); }
        inline bool operator==(const Element<InfoT, isConst>& r) const noexcept {return id == r.id; }
        inline bool operator!=(const Element<InfoT, isConst>& r) const noexcept {return id != r.id; }
        inline bool operator< (const Element<InfoT, isConst>& r) const noexcept {return id < r.id; }
        inline bool operator<=(const Element<InfoT, isConst>& r) const noexcept {return id <= r.id; }
        inline bool operator> (const Element<InfoT, isConst>& r) const noexcept {return id <= r.id; }
        inline bool operator>=(const Element<InfoT, isConst>& r) const noexcept {return id <= r.id; }

//        ID get_id() const noexcept {return id; }

        friend std::ostream& operator <<(std::ostream& o, const Element<InfoT, isConst>& item){
            o << item.id;
            return o;
        }

        friend std::hash<Element<InfoT>>;
        friend std::hash<Element<InfoT, true>>;
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




    template <typename InfoT, bool isconst = false>
    class ManagerIterator; // forward reference


    template <typename InfoT>
    class Manager : public std::enable_shared_from_this<Manager<InfoT>> {
    private:
        std::vector<InfoT> infos {};

    public:
        using value_type = Element<InfoT>;
        using pointer = Element<InfoT>*;
        using reference = Element<InfoT>&;
        using iterator = ManagerIterator<InfoT>;
        using const_iterator = ManagerIterator<InfoT, true>;



              InfoT& operator[](ID id)       {assert(id<infos.size()); return infos[id]; }
        const InfoT& operator[](ID id) const {assert(id<infos.size()); return infos[id]; }

        template <typename... Cargs>
        InfoT& create( Cargs... args){
            const ID id = infos.size();
            infos.emplace_back(id, this->shared_from_this(), std::forward<Cargs>(args)...);
            return infos[id];
        }

        ManagerIterator<InfoT> begin() {return ManagerIterator<InfoT>(0, this->shared_from_this()); }
        ManagerIterator<InfoT> end() {
            return ManagerIterator<InfoT>(infos.size(), this->shared_from_this());
        }

        ManagerIterator<InfoT, true> cbegin() const {
            return ManagerIterator<InfoT, true>(0, this->shared_from_this());
        }
        ManagerIterator<InfoT, true> cend() const {
            return ManagerIterator<InfoT, true>(infos.size(), this->shared_from_this());
        }

        unsigned long size() const noexcept {
            return infos.size();
        }

    };


    template <typename InfoT, bool isConst>
    class ManagerIterator {
    private:
        unsigned long index;
//        ManagerPtr<InfoT> manager;
        using ManagerType = typename std::conditional<isConst, ConstManagerPtr<InfoT>, ManagerPtr<InfoT>>::type;
         ManagerType manager;
    public:
        using difference_type = long;
        using value_type = typename std::conditional<isConst, const Element<InfoT, true>, Element<InfoT> >::type;
        using pointer = typename std::conditional<isConst, const Element<InfoT, true>*, Element<InfoT>* >::type;
        using reference = typename std::conditional<isConst, const Element<InfoT, true>, Element<InfoT>>::type;
        using iterator_category = std::random_access_iterator_tag;

        ManagerIterator(): index(0) {}
        ManagerIterator(unsigned long i, typename std::conditional<isConst, ConstManagerPtr<InfoT>, ManagerPtr<InfoT>>::type m): index(i), manager(m) {}
//        ManagerIterator(unsigned long i, ManagerPtr<InfoT> m): index(i), manager(m) {}
        ManagerIterator(const ManagerIterator<InfoT, isConst>& mit): index(mit.index), manager(mit.manager) {}
        // todo: enable implicit conversion of non-const iterator to const iterator


        bool operator==(const ManagerIterator<InfoT, isConst>& mit) const noexcept {return mit.index == index; }
        bool operator!=(const ManagerIterator<InfoT, isConst>& mit) const noexcept {return !(*this == mit); }
        bool operator<(const ManagerIterator<InfoT, isConst>& mit) const noexcept {return index < mit.index; }
        bool operator<=(const ManagerIterator<InfoT, isConst>& mit) const noexcept {return index <= mit.index; }
        bool operator>(const ManagerIterator<InfoT, isConst>& mit) const noexcept {return !(*this <= mit); }
        bool operator>=(const ManagerIterator<InfoT, isConst>& mit) const noexcept {return !(*this < mit); }

//        value_type operator*() const {return Element<InfoT>(index, manager); }
        value_type operator*() {return typename std::conditional<isConst, Element<InfoT, true>, Element<InfoT>>::type(index, manager); }
//        value_type operator->() const {return Element<InfoT>(index, manager); }
        value_type operator->() {return typename std::conditional<isConst, Element<InfoT, true>, Element<InfoT>>::type(index, manager); }

        ManagerIterator<InfoT, isConst>& operator++() { // ++i
            ++index;
            return *this;
        }
        ManagerIterator<InfoT, isConst>& operator++(int) { // i++
            auto result = ManagerIterator<InfoT, isConst>(index, manager);
            ++index;
            return result;
        }
        ManagerIterator<InfoT, isConst> operator--() { // --i
            --index;
            return *this;
        }
        ManagerIterator<InfoT, isConst>& operator--(int) { // i++
            auto result = ManagerIterator<InfoT, isConst>(index, manager);
            --index;
            return result;
        }

        ManagerIterator<InfoT, isConst>& operator+=(difference_type shift) {
            index += shift;
            return *this;
        }
        ManagerIterator<InfoT, isConst>& operator-=(difference_type shift) {
            index -= shift;
            return *this;
        }

        ManagerIterator<InfoT, isConst> operator+(difference_type shift) const {
            return ManagerIterator<InfoT, isConst>(index+shift, manager);
        }
        difference_type operator-(const ManagerIterator<InfoT, isConst>& other) const {
            return index - other.index;
        }

        reference operator[](const difference_type i) const {
//            return Element<InfoT>(index + i, manager);
            return typename std::conditional<isConst, Element<InfoT, true>, Element<InfoT>>::type(index + i, manager);
        }


        friend class ManagerIterator<InfoT, true>;
        friend class ManagerIterator<InfoT, false>;

    };



}

namespace std {
    template <typename InfoT, bool isConst>
    struct hash<Manage::Element<InfoT, isConst>> {
        std::size_t operator()(const Manage::Element<InfoT, isConst>& element) const {
            return std::hash<Manage::ID>()(element.id);
        }
    };
}




#endif //STERM_PARSER_MANAGER_H
