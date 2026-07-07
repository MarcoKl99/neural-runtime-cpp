#pragma once

#include "nrt/tensor.hpp"

namespace nrt {

double mse(const Tensor& y_hat, const Tensor& y);
Tensor mse_derivative(const Tensor& y_hat, const Tensor& y);

}  // namespace nrt
