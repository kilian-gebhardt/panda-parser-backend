//
// Created by Markus on 18.08.17.
//

#include <iostream>
#include <Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>



// Generic binary operation support.
template <typename Derived, typename CustomBinaryOp, typename OtherDerived> EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const Eigen::TensorCwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>
binaryExpr(const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& a, const OtherDerived& other, const CustomBinaryOp& func) {
    return Eigen::TensorCwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>(*static_cast<const Derived*>(&a), other, func);
}

template<typename Derived, typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
const Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<double>, const Derived, const OtherDerived>
multiply(const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& a, const OtherDerived& other) {
    return binaryExpr(a, static_cast<const Derived>(other), Eigen::internal::scalar_product_op<double>());
}




int main()
{
    Eigen::Tensor<double, 3> start(2,2,2);
    start(0,0,0) = 5;
    std::cout << start << std::endl;

    auto first = multiply(start * start, start * start); // first is an opration

    Eigen::Tensor<double, 3> out = first; // the operation is evaluated

    std::cout << out(0,0,0);
}

