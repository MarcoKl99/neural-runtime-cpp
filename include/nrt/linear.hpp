#pragma once

#include "nrt/tensor.hpp"

namespace nrt {

class Linear {
public:
    Linear(size_t in_features, size_t out_features);

    /*
    Computes an affine transformation y = Wx + b with weights W.shape = {out_features, in_features}
    and b.shape = {out_feautes, 1},
    -> y.shape = {out_features, 1}
    */

    // Overwrite randomly init weights for testing
    void set_weights(const Tensor& w, const Tensor& b);

    // x of shape {in_features, 1} - returns shape {out_features, 1}.
    Tensor forward(const Tensor& x) const;

    size_t in_features() const;
    size_t out_features() const;

    // Return a const Tensor reference to avoid unneccesary copies at every call
    // (Tensor instances can be large)
    const Tensor& weights() const;
    const Tensor& bias() const;

private:
    size_t in_features_;
    size_t out_features_;
    Tensor weights_;
    Tensor bias_;
};

}  // namespace nrt
