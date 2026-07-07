#include "nrt/loss.hpp"

#include <stdexcept>

namespace nrt {

double mse(const Tensor& y_hat, const Tensor& y) {
    // Check the shapes
    if (y_hat.shape() != y.shape()) {
        throw std::invalid_argument("mse: shape mismatch");
    }

    // MSE calculation using the hadamard product with itself
    Tensor diff = y_hat - y;
    Tensor squared = diff.hadamard(diff);

    return squared.sum() / static_cast<double>(y_hat.size());
}

Tensor mse_derivative(const Tensor& y_hat, const Tensor& y) {
    // Check the shapes
    if (y_hat.shape() != y.shape()) {
        throw std::invalid_argument("mse_derivative: shape mismatch");
    }

    // Calculate the derivative for both Tensor dimensions (y_hat and y)
    Tensor diff = y_hat - y;
    double scale = 2.0 / static_cast<double>(y_hat.size());

    return diff * scale;
}

}  // namespace nrt
