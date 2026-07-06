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

}  // namespace nrt
