#include "nrt/operations.hpp"

#include <iostream>
#include <memory>

#include "nrt/computation_node.hpp"
#include "nrt/tensor.hpp"

namespace {

// Helper: Find position of max element in 2×2 pool patch
std::tuple<double, size_t, size_t> find_max_pool_position(const nrt::Tensor& tensor, size_t batch,
                                                          size_t channel, size_t h_in,
                                                          size_t w_in) {
    // TODO: This should not be re-defined here
    const size_t pool_size = 2;

    double max_val = tensor(batch, channel, h_in, w_in);
    size_t max_i = 0, max_j = 0;

    for (size_t i = 0; i < pool_size; ++i) {
        for (size_t j = 0; j < pool_size; ++j) {
            if (tensor(batch, channel, h_in + i, w_in + j) > max_val) {
                max_val = tensor(batch, channel, h_in + i, w_in + j);
                max_i = i;
                max_j = j;
            }
        }
    }

    return {max_val, max_i, max_j};
}

}  // namespace

namespace nrt {

std::shared_ptr<Tensor> matmul_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // Forward: heap-allocate the result so the graph can co-own it
    auto result = std::make_shared<Tensor>(a->matmul(*b));

    result->creator_node_ =
        ComputationNode{.inputs = {a, b},  // <-- node now OWNS a and b (refcount++)
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];
                            auto& b = inputs[1];

                            Tensor grad_a = grad_result.matmul(b->transpose());
                            Tensor grad_b = a->transpose().matmul(grad_result);

                            a->accumulate_gradient(grad_a);
                            b->accumulate_gradient(grad_b);

                            if (a->creator_node_) a->backward_impl(grad_a);
                            if (b->creator_node_) b->backward_impl(grad_b);
                        }};

    return result;
}

std::shared_ptr<Tensor> scalar_mult_autodiff(std::shared_ptr<Tensor> a, double scalar) {
    // Forward: z = a * scalar
    auto result = std::make_shared<Tensor>(*a * scalar);

    // Attach the computational node
    result->creator_node_ = ComputationNode{
        .inputs = {a},
        .backward_fn = [scalar](Tensor& result_output, const Tensor& grad_result,
                                const std::vector<std::shared_ptr<Tensor>>& inputs) {
            auto& a = inputs[0];

            // Chain rule: dL/da = dL/dz * scalar
            Tensor grad_a = grad_result * scalar;

            // Accumulate the gradients
            a->accumulate_gradient(grad_a);

            // Recursion
            if (a->creator_node_) a->backward_impl(grad_a);
        }};

    return result;
}

std::shared_ptr<Tensor> add_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // Forward: z = a + b
    auto result = std::make_shared<Tensor>(*a + *b);

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {a, b},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];
                            auto& b = inputs[1];

                            // Chain rule: both inputs get the same gradient
                            // dL/da = dL/dz
                            // dL/db = dL/dz

                            // Un-broadcast gradients to match original input shapes
                            Tensor grad_a = Tensor::unbroadcast_gradient(grad_result, a->shape());
                            Tensor grad_b = Tensor::unbroadcast_gradient(grad_result, b->shape());

                            a->accumulate_gradient(grad_a);
                            b->accumulate_gradient(grad_b);

                            // Recurse on inputs
                            if (a->creator_node_) a->backward_impl(grad_a);
                            if (b->creator_node_) b->backward_impl(grad_b);
                        }};

    return result;
}

std::shared_ptr<Tensor> subtract_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // Forward: z = a - b
    auto result = std::make_shared<Tensor>(*a - *b);

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {a, b},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];
                            auto& b = inputs[1];

                            // Chain rule:
                            // dL/da = dL/dz
                            // dL/db = -dL/dz (negated)

                            // Un-broadcast gradients to match original input shapes
                            Tensor grad_a = Tensor::unbroadcast_gradient(grad_result, a->shape());
                            Tensor grad_b = Tensor::unbroadcast_gradient(grad_result, b->shape());

                            a->accumulate_gradient(grad_a);
                            b->accumulate_gradient(grad_b * -1.0);  // Negate for b

                            // Recurse on inputs
                            if (a->creator_node_) a->backward_impl(grad_a);
                            if (b->creator_node_) b->backward_impl(grad_b * -1.0);
                        }};

    return result;
}

std::shared_ptr<Tensor> reshape_autodiff(std::shared_ptr<Tensor> a,
                                         const std::vector<size_t>& new_shape) {
    // Forward: z = a.reshape(new_shape)
    auto result = std::make_shared<Tensor>(a->reshape(new_shape));

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {a},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];

                            // Reshape's local derivative is the identity - it's a pure relabeling
                            // of the same values, so the gradient just needs to be reshaped back
                            // to a's original shape before it flows further upstream.
                            Tensor grad_a = grad_result.reshape(a->shape());

                            a->accumulate_gradient(grad_a);
                            if (a->creator_node_) a->backward_impl(grad_a);
                        }};

    return result;
}

std::shared_ptr<Tensor> transpose_autodiff(std::shared_ptr<Tensor> a) {
    // Forward: transpose the tensor
    auto result = std::make_shared<Tensor>(a->transpose());

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {a},
                        .backward_fn = [](Tensor& output, const Tensor& grad_output,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];

                            // Transpose is its own inverse: transposing the gradient back gives us
                            // grad_a
                            Tensor grad_a = grad_output.transpose();

                            a->accumulate_gradient(grad_a);
                            if (a->creator_node_) a->backward_impl(grad_a);
                        }};

    return result;
}

static std::shared_ptr<Tensor> conv_matmul_autodiff(std::shared_ptr<Tensor> weights,
                                                    std::shared_ptr<Tensor> patches, size_t batch,
                                                    size_t out_channels, size_t out_height,
                                                    size_t out_width) {
    size_t K = weights->size() / out_channels;  // in_channels * kernel_size * kernel_size
    size_t S = out_height * out_width;

    auto output =
        std::make_shared<Tensor>(std::vector<size_t>{batch, out_channels, out_height, out_width});
    Tensor weights_2d = weights->reshape({out_channels, K});

    for (size_t b = 0; b < batch; ++b) {
        Tensor patches_b({K, S});
        for (size_t k = 0; k < K; ++k) {
            for (size_t s = 0; s < S; ++s) {
                patches_b(k, s) = (*patches)(k, b * S + s);
            }
        }

        Tensor output_b = weights_2d.matmul(patches_b);  // {out_channels, S}

        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t s = 0; s < S; ++s) {
                (*output)(b, oc, s / out_width, s % out_width) = output_b(oc, s);
            }
        }
    }

    auto backward_fn = [batch, out_channels, out_height, out_width, K, S](
                           Tensor& result_output, const Tensor& grad_output,
                           const std::vector<std::shared_ptr<Tensor>>& inputs) {
        auto& weights = inputs[0];
        auto& patches = inputs[1];

        Tensor weights_2d = weights->reshape({out_channels, K});
        Tensor grad_weights_2d({out_channels, K});
        Tensor grad_patches(patches->shape());

        for (size_t b = 0; b < batch; ++b) {
            Tensor patches_b({K, S});
            for (size_t k = 0; k < K; ++k) {
                for (size_t s = 0; s < S; ++s) {
                    patches_b(k, s) = (*patches)(k, b * S + s);
                }
            }

            Tensor grad_output_b({out_channels, S});
            for (size_t oc = 0; oc < out_channels; ++oc) {
                for (size_t s = 0; s < S; ++s) {
                    grad_output_b(oc, s) = grad_output(b, oc, s / out_width, s % out_width);
                }
            }

            // Weights are shared across the whole batch, so their gradient accumulates.
            grad_weights_2d = grad_weights_2d + grad_output_b.matmul(patches_b.transpose());

            // Each batch sample writes to a disjoint column range, so plain assignment is fine
            Tensor grad_patches_b = weights_2d.transpose().matmul(grad_output_b);  // {K, S}
            for (size_t k = 0; k < K; ++k) {
                for (size_t s = 0; s < S; ++s) {
                    grad_patches(k, b * S + s) = grad_patches_b(k, s);
                }
            }
        }

        Tensor grad_weights = grad_weights_2d.reshape(weights->shape());
        weights->accumulate_gradient(grad_weights);
        // weights is a leaf parameter tensor — no creator_node_ to recurse into.

        patches->accumulate_gradient(grad_patches);
        if (patches->creator_node_) patches->backward_impl(grad_patches);
    };

    output->creator_node_ = ComputationNode{{weights, patches}, backward_fn};
    return output;
}

std::shared_ptr<Tensor> conv2d_autodiff(std::shared_ptr<Tensor> input,
                                        std::shared_ptr<Tensor> weights,
                                        std::shared_ptr<Tensor> bias, size_t kernel_size) {
    // Extract dimensions
    size_t batch = input->shape()[0];
    size_t height = input->shape()[2];
    size_t width = input->shape()[3];
    size_t out_channels = weights->shape()[0];
    size_t out_height = height - kernel_size + 1;
    size_t out_width = width - kernel_size + 1;

    auto patches = im2col_autodiff(input, kernel_size, 1, 0);

    auto output_4d =
        conv_matmul_autodiff(weights, patches, batch, out_channels, out_height, out_width);

    auto bias_reshaped = reshape_autodiff(bias, {1, out_channels, 1, 1});

    auto output = add_autodiff(output_4d, bias_reshaped);

    return output;
}

std::shared_ptr<Tensor> maxpool2d_autodiff(std::shared_ptr<Tensor> input) {
    const size_t pool_size = 2;
    const size_t stride = 2;

    // Extract dimensions
    size_t batch = input->shape()[0];
    size_t channels = input->shape()[1];
    size_t height = input->shape()[2];
    size_t width = input->shape()[3];

    size_t out_height = height / stride;
    size_t out_width = width / stride;

    // Forward pass: compute output and track max positions
    auto output =
        std::make_shared<Tensor>(std::vector<size_t>{batch, channels, out_height, out_width});

    // Compute output and store max positions
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h_pool = 0; h_pool < out_height; ++h_pool) {
                for (size_t w_pool = 0; w_pool < out_width; ++w_pool) {
                    size_t h_in = h_pool * stride;
                    size_t w_in = w_pool * stride;

                    auto [max_val, max_i, max_j] = find_max_pool_position(*input, b, c, h_in, w_in);

                    (*output)(b, c, h_pool, w_pool) = max_val;
                }
            }
        }
    }

    // Create backward function
    auto backward_fn = [batch, channels, out_height, out_width](
                           const Tensor& output, const Tensor& grad_output,
                           const std::vector<std::shared_ptr<Tensor>>& inputs) {
        Tensor grad_input(inputs[0]->shape());

        // Route gradients: only the max position gets gradient, others get 0
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t h_pool = 0; h_pool < out_height; ++h_pool) {
                    for (size_t w_pool = 0; w_pool < out_width; ++w_pool) {
                        size_t h_in = h_pool * stride;
                        size_t w_in = w_pool * stride;

                        // Note that that is not the most efficient way to do it!
                        // To future me: Here you can gain performance by storing the positions
                        // during forward! ;)
                        // Had reference issues in the first attempt though
                        auto [max_val, max_i, max_j] =
                            find_max_pool_position(*inputs[0], b, c, h_in, w_in);

                        // Route gradient to max position only
                        grad_input(b, c, h_in + max_i, w_in + max_j) =
                            grad_output(b, c, h_pool, w_pool);
                    }
                }
            }
        }

        // Accumulate gradient into input
        inputs[0]->accumulate_gradient(grad_input);

        // Recursively backpropagate
        inputs[0]->backward_impl(grad_input);
    };

    // Attach computation node
    output->creator_node_ = ComputationNode{{input}, backward_fn};

    return output;
}

std::shared_ptr<Tensor> im2col(std::shared_ptr<Tensor> input, size_t kernel_size, size_t stride,
                               size_t padding) {
    // Extract dimensions
    size_t batch = input->shape()[0];
    size_t in_channels = input->shape()[1];
    size_t height = input->shape()[2];
    size_t width = input->shape()[3];

    // Calculate output spatial dimensions
    size_t out_height = (height + 2 * padding - kernel_size) / stride + 1;
    size_t out_width = (width + 2 * padding - kernel_size) / stride + 1;

    // Create output matrix: {in_channels * kernel_size^2, batch * out_height * out_width}
    size_t col_size = in_channels * kernel_size * kernel_size;
    size_t num_patches = batch * out_height * out_width;
    auto output = std::make_shared<Tensor>(std::vector<size_t>{col_size, num_patches});

    // Iterate through: batch -> output positions -> extract patches
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h_out = 0; h_out < out_height; ++h_out) {
            for (size_t w_out = 0; w_out < out_width; ++w_out) {
                // Calculate the flattened column index
                // We skip i times the product of all remaining dimensions > i
                // ...pretty cool hmm? ;)
                size_t col_index = b * out_height * out_width + h_out * out_width + w_out;

                // Get the starting values for the patch
                size_t h_start = h_out * stride;
                size_t w_start = w_out * stride;

                // For each channel and kernel value, set the output value
                for (size_t c = 0; c < in_channels; ++c) {
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            (*output)(c * kernel_size * kernel_size + kh * kernel_size + kw,
                                      col_index) = (*input)(b, c, h_start + kh, w_start + kw);
                        }
                    }
                }
            }
        }
    }

    return output;
}

std::shared_ptr<Tensor> im2col_autodiff(std::shared_ptr<Tensor> input, size_t kernel_size,
                                        size_t stride, size_t padding) {
    // Forward: call plain im2col
    auto patches = im2col(input, kernel_size, stride, padding);

    // Extract dimensions for backward
    size_t batch = input->shape()[0];
    size_t in_channels = input->shape()[1];
    size_t height = input->shape()[2];
    size_t width = input->shape()[3];
    size_t out_height = (height + 2 * padding - kernel_size) / stride + 1;
    size_t out_width = (width + 2 * padding - kernel_size) / stride + 1;

    // Create backward function (col2im: convert patch gradients back to input gradients)
    auto backward_fn = [batch, in_channels, height, width, kernel_size, stride, padding, out_height,
                        out_width](const Tensor& patches_output, const Tensor& grad_patches,
                                   const std::vector<std::shared_ptr<Tensor>>& inputs) {
        // grad_patches: {in_channels * kernel_size^2, batch * out_h * out_w}
        // Convert back to: {batch, in_channels, height, width}

        Tensor grad_input(inputs[0]->shape());  // {batch, in_channels, height, width}

        // col2im: accumulate patch gradients back to input gradients
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h_out = 0; h_out < out_height; ++h_out) {
                for (size_t w_out = 0; w_out < out_width; ++w_out) {
                    // Column index in grad_patches
                    size_t col_index = b * out_height * out_width + h_out * out_width + w_out;

                    // Starting position in input for this patch
                    size_t h_start = h_out * stride;
                    size_t w_start = w_out * stride;

                    // For each channel and kernel position
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                // Input position this patch element came from
                                size_t input_h = h_start + kh;
                                size_t input_w = w_start + kw;

                                // Row index in grad_patches for this channel and kernel position
                                size_t row_in_patches =
                                    ic * kernel_size * kernel_size + kh * kernel_size + kw;

                                // Accumulate: multiple patches contribute to same input position
                                grad_input(b, ic, input_h, input_w) +=
                                    grad_patches(row_in_patches, col_index);
                            }
                        }
                    }
                }
            }
        }

        // Accumulate gradient into input
        inputs[0]->accumulate_gradient(grad_input);

        // Recursively backpropagate
        if (inputs[0]->creator_node_) inputs[0]->backward_impl(grad_input);
    };

    patches->creator_node_ = ComputationNode{.inputs = {input}, .backward_fn = backward_fn};
    return patches;
}

}  // namespace nrt
