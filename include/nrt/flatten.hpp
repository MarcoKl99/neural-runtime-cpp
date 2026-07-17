#include "nrt/module.hpp"
#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

class Flatten : public Module {
public:
    Flatten() = default;

    // Forward: {batch, d1, d2, ...} -> {batch, d1*d2*...}
    // Uses reshape_autodiff internally to preserve gradients
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;

    // No learnable parameters
    std::vector<Parameter> parameters() override {
        return {};
    }

private:
    std::vector<size_t> input_shape_;  // Store input shape for backward pass
};

}  // namespace nrt
