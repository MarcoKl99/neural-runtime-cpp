#pragma once

#include "nrt/tensor.hpp"

namespace nrt {

Tensor relu(const Tensor& x);
Tensor relu_derivative(const Tensor& x);

Tensor sigmoid(const Tensor& x);
Tensor sigmoid_derivative(const Tensor& x);

}  // namespace nrt
