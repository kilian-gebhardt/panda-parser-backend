//
// Created by jamb on 09.03.17.
//

#ifndef STERMPARSER_MANAGER_UTIL_H
#define STERMPARSER_MANAGER_UTIL_H

#include <iostream>

namespace Manage {

    void serialize_string_or_size_t(std::ostream &out, const size_t s) {
        out << s;
    }


    void serialize_string_or_size_t(std::ostream& out, const std::string str){
        out << str.size() << ';' << str;
    }


    void deserialize_string_or_size_t(std::istream& in, std::string& str) {
        size_t len;
        char sep;
        in >> len;  //deserialize size of string
        in >> sep; //read in the seperator
        if (in && len) {
            std::vector<char> tmp(len);
            in.read(tmp.data(), len); //deserialize characters of string
            str.assign(tmp.data(), len);
        }
    }


    void deserialize_string_or_size_t(std::istream& in, size_t& s){
        in >> s;
    }


    template<typename T1>
    typename std::enable_if_t<
            (std::is_same<T1, std::string>::value || std::is_same<T1, size_t>::value)
            , void
            >
    serialize_labels(std::ostream& out, const std::vector<T1>& labels) {
        out << labels.size() << ';';
        for(auto const& label : labels) {
            serialize_string_or_size_t(out, label);
            out << ";";
        }
    }

    template<typename T1>
    typename std::enable_if_t<
            (std::is_same<T1, std::string>::value || std::is_same<T1, size_t>::value)
            , std::vector<T1>
    >
    deserialize_labels(std::istream& in) {
        size_t noOfLabels;
        T1 label;
        std::string str;
        std::vector<T1> result;
        char sep;

        in >> noOfLabels;
        in >> sep;
        for(size_t i = 0; i < noOfLabels; ++i) {
            deserialize_string_or_size_t(in, label);
            in >> sep;
            result.push_back(label);
        }
        return result;
    }

}


#endif //STERMPARSER_MANAGER_UTIL_H
