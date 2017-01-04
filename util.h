//
// Created by kilian on 02/12/16.
//

#include <tuple>
#include <iostream>
#include <string>
#include <type_traits>

#ifndef STERMPARSER_UTIL_H
#define STERMPARSER_UTIL_H


// taken from https://stackoverflow.com/a/33683299 for printing tuples
template<size_t N>
struct print_tuple{
    template<typename... T>static typename std::enable_if<(N<sizeof...(T))>::type
    print(std::ostream& os, const std::tuple<T...>& t) {
        char quote = (std::is_convertible<decltype(std::get<N>(t)), std::string>::value) ? '"' : 0;
        os << ", " << quote << std::get<N>(t) << quote;
        print_tuple<N+1>::print(os,t);
    }
    template<typename... T>static typename std::enable_if<!(N<sizeof...(T))>::type
    print(std::ostream&, const std::tuple<T...>&) {
    }
};
std::ostream& operator<< (std::ostream& os, const std::tuple<>&) {
    return os << "()";
}
template<typename T0, typename ...T> std::ostream&
operator<<(std::ostream& os, const std::tuple<T0, T...>& t){
    char quote = (std::is_convertible<T0, std::string>::value) ? '"' : 0;
    os << '(' << quote << std::get<0>(t) << quote;
    print_tuple<1>::print(os,t);
    return os << ')';
}


template <typename T>
bool pairwise_different(const std::vector<T> & vec){
    for (auto s1 = vec.begin(); s1 != vec.end(); ++s1) {
        for(auto s2 = s1 + 1; s2 != vec.end(); ++s2) {
            if (*s1 == *s2) {
                return false;
            }
        }
    }
    return true;
}

template<typename Accum1, typename Val>
std::vector<Val> dot_product(const Accum1 point_product, const std::vector<Val> & arg1, const std::vector<Val> &arg2) {
    assert(arg1.size() == arg2.size());
    std::vector<Val> result;
    for (unsigned i = 0; i < arg1.size(); ++i) {
        result.push_back(point_product(arg1[i], arg2[i]));
    }
    return result;
};

template<typename Accum1, typename Val>
std::vector<Val> scalar_product(const Accum1 point_scalar, const std::vector<Val> &goal, const Val scalar) {
    std::vector<Val> result;
    for (auto i = goal.begin(); i != goal.end(); ++i) {
        result.push_back(point_scalar(*i, scalar));
    }
    return result;
};

template<typename Accum, typename Val>
Val reduce(const Accum accum, const std::vector<Val> & vec, Val init, const unsigned start = 0) {
    for (auto i = vec.begin() + start; i != vec.end(); ++i)
        init = accum(init, *i);
    return init;
};


#endif //STERMPARSER_UTIL_H
