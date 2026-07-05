#include "nrt/linear.hpp"

#include <random>
#include <stdexcept>

namespace nrt {

Linear::Linear(size_t in_features, size_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weights_({out_features, in_features}),
      bias_({out_features, 1}) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);  // mu=0, sigma=0.1

    for (size_t i = 0; i < out_features_; ++i) {
        for (size_t j = 0; j < in_features_; ++j) {
            weights_(i, j) = dist(gen);
        }
    }
    // bias_ remains zero
}

void Linear::set_weights(const Tensor& w, const Tensor& b) {
    // Check the shapes
    if (w.shape() != std::vector<size_t>{out_features_, in_features_}) {
        throw std::invalid_argument("Linear::set_weights: weight shape mismatch");
    }
    if (b.shape() != std::vector<size_t>{out_features_, 1}) {
        throw std::invalid_argument("Linear::set_weights: bias shape mismatch");
    }

    // Set the values
    weights_ = w;
    bias_ = b;
}

Tensor Linear::forward(const Tensor& x) const {
    // Check the shapes
    if (x.shape() != std::vector<size_t>{in_features_, 1}) {
        throw std::invalid_argument("Linear::forward: input shape mismatch");
    }

    // Affine transformation
    return weights_.matmul(x) + bias_;
}

// Getter
size_t Linear::in_features() const {
    return in_features_;
}

size_t Linear::out_features() const {
    return out_features_;
}

const Tensor& Linear::weights() const {
    return weights_;
}

const Tensor& Linear::bias() const {
    return bias_;
}

}  // namespace nrt
