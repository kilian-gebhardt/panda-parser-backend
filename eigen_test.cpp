//
// Created by kilian on 23/01/17.
//

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <vector>
#include <stdio.h>



template<typename Val>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 1> & rule_las, const std::vector<Eigen::Tensor<Val, 1>> & rhs_lass) {
    std::cerr << "special 1";
    return rule_las;
};

template<typename Val>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 2> & rule_las, const std::vector<Eigen::Tensor<Val, 1>> & rhs_lass) {
    auto contraction = rule_las.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return contraction;
}

template<typename Val>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 3> & rule_las, const std::vector<Eigen::Tensor<Val, 1>> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c2 = c1.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c2;
}

template<typename Val>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, 4> & rule_las, const std::vector<Eigen::Tensor<Val, 1>> & rhs_lass) {
    auto c1 = rule_las.contract(rhs_lass[2], {Eigen::IndexPair<int>(3, 0)});
    auto c2 = c1.contract(rhs_lass[1], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)}));
    auto c3 = c2.contract(rhs_lass[0], Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(1, 0)}));
    return c3;
}


template <int N, typename Val>
Eigen::Tensor<Val, 1> rule_probs(const Eigen::Tensor<Val, N> & rule_las, const std::vector<Eigen::Tensor<Val, 1>> & rhs_lass) {
    assert((N > 1) && (rhs_lass.size() > N-2));
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(N - 1, 0)};
    Eigen::Tensor<Val, N-1> contraction = rule_las.contract(rhs_lass[N - 2], product_dims);

    return rule_probs(contraction, rhs_lass);
};

typedef double MyVal;

int main() {
// Create 2 matrices using tensors of rank 2
    Eigen::Tensor<MyVal, 3> a(2, 2, 3);
    a.setValues({{{1, 2, 3}, {6, 5.0, 4}},
                 {{7, 8, 9}, {12, 11, 10.5}}
                });
    Eigen::Tensor<MyVal, 1> b(3);
    b.setValues({1, 2, 3});
    MyVal * c_ptr = (MyVal*) malloc(sizeof(MyVal) * 2);
//    MyVal c_ptr[] = {0, 0};
    Eigen::TensorMap<Eigen::Tensor<MyVal, 1>> c(c_ptr, 2);
    c.setValues({1,2});



// Compute the traditional matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(2, 0)};
    auto t = a.contract(b, product_dims);
    std::cerr << t << std::endl;
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {Eigen::IndexPair<int>(1,0)};
    auto t2 = t.contract(c, product_dims2);
    std::cerr << t2 << std::endl;

    std::cerr << "Rule probs: " << std::endl;
    std::cerr << rule_probs(a, {c, b}) << std::endl;

// Compute the product of the transpose of the matrices
//    Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = {Eigen::IndexPair<int>(0, 1)};
//    Eigen::Tensor<int, 2> AtBt = a.contract(b, transposed_product_dims);
//
//    std::cerr << AtBt << std::endl;

    free(c_ptr);

    Eigen::Tensor<MyVal, 2> A(2,4);
    A.setValues({{1,2,3,4},{5,6,7,8}});
    Eigen::Tensor<MyVal, 1> B(2);
    B.setValues({3,2});

    std::cerr << "A " << A << std::endl;
    std::cerr << "B " << B << std::endl;
    std::cerr << "A chipped " << A.chip(0,0) << std::endl;
    std::cerr << "A chipped " << A.chip(1,0) << std::endl;
    std::cerr << "B chipped " << B.chip(0,0) << std::endl;
    std::cerr << "B chipped " << B.chip(1,0) << std::endl;


    A.chip(0,0) = A.chip(0, 0).unaryExpr([&](const double x) -> double {return x / B(0);});
    std::cerr << "A " << A << std::endl;
    A.chip(1, 0) = A.chip(1, 0).unaryExpr([&] (const double x) -> double {return x / B(1);});
    std::cerr << "A " << A << std::endl;




}