//
// Created by jamb on 03.02.17.
//

#ifndef STERM_PARSER_MANAGER_H
#define STERM_PARSER_MANAGER_H

#include <map>
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>

template <typename T1, typename T2>
using MAP = typename std::map<T1, T2>;


namespace Manage{

    using ID = size_t;

    template <typename InfoT>
    class Manager; // forward reference

    template <typename InfoT>
    using ManagerPtr = std::shared_ptr<Manager<InfoT>>; // forward reference
    template <typename InfoT>
    using ConstManagerPtr = std::shared_ptr<const Manager<InfoT>>; // forward reference
    template <typename InfoT>
    using ManagerWeakPtr = std::weak_ptr<Manager<InfoT>>; // forward reference
    template <typename InfoT>
    using ConstManagerWeakPtr = std::weak_ptr<const Manager<InfoT>>; // forward reference


    template <typename InfoT, bool isConst = false>
    class Element {
        using ManagerType = typename std::conditional<isConst, ConstManagerWeakPtr<InfoT>, ManagerWeakPtr<InfoT>>::type;
        using PointerType = typename std::conditional<isConst, const InfoT*, InfoT*>::type;

    private:
        ID id;
        ManagerType manager;
    public:
        Element(ID aId, ManagerType aManager): id(aId), manager(aManager) {};

        PointerType operator->() const {return &((*manager.lock())[id]); }
        inline bool operator==(const Element<InfoT, isConst>& r) const noexcept {return id == r.id; }
        inline bool operator!=(const Element<InfoT, isConst>& r) const noexcept {return id != r.id; }
        inline bool operator< (const Element<InfoT, isConst>& r) const noexcept {return id < r.id; }
        inline bool operator<=(const Element<InfoT, isConst>& r) const noexcept {return id <= r.id; }
        inline bool operator> (const Element<InfoT, isConst>& r) const noexcept {return id > r.id; }
        inline bool operator>=(const Element<InfoT, isConst>& r) const noexcept {return id >= r.id; }

        friend std::ostream& operator <<(std::ostream& o, const Element<InfoT, isConst>& item){
            o << item.id;
            return o;
        }

        friend std::hash<Element<InfoT>>;
        friend std::hash<Element<InfoT, true>>;

        size_t hash() const {
            return std::hash<Element<InfoT, isConst>>()(*this);
        }
    };


    template <typename InfoT, bool isconst = false>
    class ManagerIterator; // forward reference


    template <typename InfoT>
    class Manager : public std::enable_shared_from_this<Manager<InfoT>> {
    protected:
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
        value_type create( Cargs... args){
            const ID id = infos.size();
            infos.emplace_back(id, this->shared_from_this(), std::forward<Cargs>(args)...);
            return infos[id].get_element();
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



        void serialize(std::ostream& o) const {
            o << "Manager Version 1" << std::endl;
            o << infos.size() << " Items" << std::endl;
            for(const InfoT& info : infos) {
                info.serialize(o);
            }
        }

        static ManagerPtr<InfoT> deserialize(std::istream& in){
            std::string line;
            std::getline(in, line);
            if(line != "Manager Version 1")
                throw std::string("Version Mismatch for Manager");

            int noItems;
            in >> noItems;
            std::getline(in, line);
            if(line != " Items")
                throw std::string("Unexpected line '" + line + "' expected ' Items'");

            ManagerPtr<InfoT> manager = std::make_shared<Manager<InfoT>>();

            for(int i = 0; i < noItems; ++i){
                InfoT info = InfoT::deserialize(in, i, manager);
                manager->infos.push_back(std::move(info));
            }

            return manager;
        }


    };


    template <typename InfoT, bool isConst>
    class ManagerIterator {
    private:
        unsigned long index;
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

        value_type operator*() {return typename std::conditional<isConst, Element<InfoT, true>, Element<InfoT>>::type(index, manager); }
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
