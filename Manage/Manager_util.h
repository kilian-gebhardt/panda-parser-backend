//
// Created by jamb on 09.03.17.
//

#ifndef STERMPARSER_MANAGER_UTIL_H
#define STERMPARSER_MANAGER_UTIL_H

#include <iostream>

namespace Manage {


    void serialize_string(std::ostream &out, const std::string& str) {
        out << str.size() << ';' << str << std::endl;
    }


    std::string deserialize_string(std::istream &in) {
        size_t len;
        std::string str;
        char sep;
        in >> len;  //deserialize size of string
        in >> sep; //read in the seperator
        if (in && len) {
            std::vector<char> tmp(len);
            in.read(tmp.data(), len); //deserialize characters of string
            str.assign(tmp.data(), len);
        }
        return str;
    }


    template<typename T1>
    typename std::enable_if_t<
            (std::is_same<T1, std::string>::value || std::is_same<T1, size_t>::value)
            , void
            >
    serialize_labels(std::ostream &out, const std::vector<T1>& labels) {
        out << labels.size() << ';';
        for(auto const& label : labels) {
            if (std::is_same<T1, std::string>::value)
                serialize_string(out, label);
            if (std::is_same<T1, size_t>::value)
                out << label;
        }
    }

    template<typename T1>
    typename std::enable_if_t<
            (std::is_same<T1, std::string>::value || std::is_same<T1, size_t>::value)
            , std::vector<T1>
    >
    deserialize_labels(std::istream& in) {
        size_t noOfLabels;
        size_t len;
        T1 label;
        std::string str;
        std::vector<T1> result;
        char sep;
        in >> noOfLabels;
        in >> sep;
        for(int i=0; i < noOfLabels; ++i) {
            if(std::is_same<T1, std::string>::value)
                result.emplace_back(deserialize_string(in));
            if(std::is_same<T1, size_t>::value) {
                in >> label;
                result.push_back(std::move(label));
            }
        }
        return result;
    }

}


#endif //STERMPARSER_MANAGER_UTIL_H
