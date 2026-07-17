#include "nrt/flatten.hpp"

#include <iostream>

#include "nrt/module.hpp"
#include "nrt/operations.hpp"
#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> x) {
    // Input x comes in with batch dimension first
    // -> {batch, dim1, dim2, ..., dim<n>}

    // Calculate the new shape
    std::size_t n_flat = 1;
    for (std::size_t i = 1; i < x->rank(); ++i) {
        n_flat *= x->shape()[i];
    }

    // Get the batch dimension
    std::size_t batch_size = x->shape()[0];

    // Use reshape_autodiff to accomplish the goal
    std::shared_ptr<Tensor> result = reshape_autodiff(x, {batch_size, n_flat});

    return result;
}

}  // namespace nrt
