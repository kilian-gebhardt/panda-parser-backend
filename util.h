//
// Created by kilian on 02/12/16.
//

#ifndef STERMPARSER_UTIL_H
#define STERMPARSER_UTIL_H

#include <tuple>
#include <iostream>
#include <string>
#include <type_traits>
#include <boost/operators.hpp>
#include <cmath>


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

constexpr double minus_infinity = -std::numeric_limits<double>::infinity();

class LogDouble : boost::operators<LogDouble> {
private:
    double x;
public:
    const double & get_Value() const {
        return x;
    };

    LogDouble() : x(minus_infinity) {} ;

//    LogDouble(LogDouble&& o) : x(std::move(o.x)) {}
//    LogDouble(LogDouble & o) : x(o.get_Value()) {}
    LogDouble(const double x) : x(x) {};
    bool operator<(const LogDouble& y) const {
        return x < y.get_Value();
    }

//    LogDouble& operator= (const LogDouble & y) {
//        x = y.get_Value();
//        return *this;
//    }

//    LogDouble& operator= (LogDouble && y) {
//        x = std::move(y.x);
//        return *this;
//    }
    bool operator==(const LogDouble & y) const {
        return x == y.get_Value();
    }

    LogDouble& operator+=(const LogDouble& y_){
        const double y = y_.get_Value();

        if (x == minus_infinity)
            x = y;
        else if (y == minus_infinity)
            ;
            // return log(exp(x) + exp(y));
            // cf. wiki, better accuracy with very small probabilites
        else if (x >= y)
            x = x + log1p(exp(y - x));
        else
            x = y + log1p(exp(x - y));
        return *this;
    }

    LogDouble& operator-= (const LogDouble & y_) {
        // const double minus_infinity = std::numeric_limits<double>::infinity();
        if (x >= y_.get_Value())
            x += log(1 - exp(y_.get_Value() - x));
        else
            x = log(exp(x) - exp(y_.get_Value()));
        return *this;
    };

    LogDouble& operator*=(const LogDouble& y_){
        x = x + y_.get_Value();
        return *this;
    }

    LogDouble& operator/=(const LogDouble& y_) {
        x = x - y_.get_Value();
        return *this;
    }

    static const LogDouble one()  {
        return LogDouble(0);
    }

    static const LogDouble zero() {
        return LogDouble(-std::numeric_limits<double>::infinity());
    }

    static const LogDouble to(const double x) {
        return LogDouble(log(x));
    }

    double from() const {
        return exp(x);
    }

    static const LogDouble add_subtract2_divide(const LogDouble base, const LogDouble add, const LogDouble sub1, const LogDouble sub2, const LogDouble div) {
        return LogDouble(log(exp(base.get_Value())
                             + exp(add.get_Value())
                             - exp(sub1.get_Value())
                             - exp(sub2.get_Value()))
                         - div.get_Value());
    }

};


class Double : boost::operators<Double> {
private:
    double x;
    const double minus_infinity = -std::numeric_limits<double>::infinity();
public:
    const double & get_Value() const {
        return x;
    };
    Double(const double x) : x(x) {};
    bool operator<(const Double& y) const {
        return x < y.get_Value();
    }

    Double() : x(0) {};

    Double& operator=(const Double& y) {
        x = y.get_Value();
        return *this;
    }

    bool operator==(const Double & y) const {
        return x == y.get_Value();
    }

    Double& operator+=(const Double& y_){
        x += y_.get_Value();
        return *this;
    }

    Double& operator-=(const Double& y_){
        x -= y_.get_Value();
        return *this;
    }

    Double operator-() const {
        return Double(-x);
    }

    Double operator*=(const Double& y_){
        x *= y_.get_Value();
        return *this;
    }

    Double operator/=(const Double& y_) {
        x = x / y_.get_Value();
        return *this;
    }

    static const Double one()  {
        return Double(1.0);
    }

    static const Double zero() {
        return Double(0.0);
    }

    static const Double to(const double x) {
        return Double(x);
    }

    double from() const {
        return x;
    }

    static Double add_subtract2_divide(const Double base, const Double add, const Double sub1, const Double sub2, const Double div) {
        return Double(((base + add - sub1) - sub2) / div);
    }

};



std::ostream &operator<<(std::ostream &os, const LogDouble &log_double){
    os << " L" << log_double.get_Value();
    return os;
}

std::ostream &operator<<(std::ostream &os, const Double &x){
    os << x.get_Value();
    return os;
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

template<typename T3, typename T1, typename T2, typename Accum>
std::vector<T3> zipWith(const Accum op, const std::vector<T1> &arg1, const std::vector<T2> &arg2) {
    assert(arg1.size() == arg2.size());
    std::vector<T3> result;
    for (unsigned i = 0; i < arg1.size(); ++i) {
        result.push_back(op(arg1[i], arg2[i]));
    }
    return result;
};

template<typename T3, typename T1, typename T2, typename Accum>
std::vector<T3> zipWithConstant(const Accum op, const std::vector<T1> &arg1, const T2 c) {
    std::vector<T3> result;
    for (auto i = arg1.begin(); i != arg1.end(); ++i) {
        result.push_back(op(*i, c));
    }
    return result;
};

template<typename Accum, typename Val>
Val reduce(const Accum accum, const std::vector<Val> & vec, Val init, const unsigned start = 0) {
    for (auto i = vec.begin() + start; i != vec.end(); ++i)
        init = accum(init, *i);
    return init;
};

void output_helper(std::string s) {
    std::cerr << s << std::endl;
}


#endif //STERMPARSER_UTIL_H
