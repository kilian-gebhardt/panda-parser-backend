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
    auto i = 0;
    for (auto s1 : vec) {
        for(auto j = ++i; j < vec.size(); ++j) {
            if (s1 == vec[j]) {
                return false;
            }
        }
    }
    return true;
}




#endif //STERMPARSER_UTIL_H
