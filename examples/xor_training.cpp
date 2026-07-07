#include <iostream>
#include <vector>

#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/tensor.hpp"

/*
Manual MLP training on the XOR problem: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0.
Architecture: Linear -> ReLU -> Linear -> Sigmoid -> MSE.
Full-batch gradient descent: forward + backward accumulate gradients across all 4 samples,
then weights/biases are updated manually (no Optimizer class yet, see roadmap v0.2).

Note: depending on random weight initialization, this can occasionally get stuck due to the
"Dying ReLU" problem - see notes/dying_relu_observation.md for details.
*/

namespace {

nrt::Tensor make_input(double a, double b) {
    nrt::Tensor x({2, 1});
    x(0, 0) = a;
    x(1, 0) = b;
    return x;
}

nrt::Tensor make_target(double t) {
    nrt::Tensor y({1, 1});
    y(0, 0) = t;
    return y;
}

// Execute the full forward-backward iteration for one single step
// and returns the loss for this sample
double forward_backward_step(nrt::Linear& layer1, nrt::Linear& layer2, const nrt::Tensor& x,
                             const nrt::Tensor& target) {
    // Forward
    nrt::Tensor z1 = layer1.forward(x);
    nrt::Tensor a1 = nrt::relu(z1);
    nrt::Tensor z2 = layer2.forward(a1);
    nrt::Tensor y_hat = nrt::sigmoid(z2);

    double loss = nrt::mse(y_hat, target);

    // Backward
    nrt::Tensor grad_y_hat = nrt::mse_derivative(y_hat, target);
    nrt::Tensor grad_z2 = nrt::sigmoid_backward(grad_y_hat, z2);
    nrt::Tensor grad_a1 = layer2.backward(grad_z2);
    nrt::Tensor grad_z1 = nrt::relu_backward(grad_a1, z1);
    layer1.backward(grad_z1);

    return loss;
}

double evaluate_average_loss(nrt::Linear& layer1, nrt::Linear& layer2,
                             const std::vector<nrt::Tensor>& inputs,
                             const std::vector<nrt::Tensor>& targets) {
    double total_loss = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        // Forward pass
        nrt::Tensor a1 = nrt::relu(layer1.forward(inputs[i]));
        nrt::Tensor y_hat = nrt::sigmoid(layer2.forward(a1));

        // Loss calculation
        total_loss += nrt::mse(y_hat, targets[i]);
    }

    return total_loss / static_cast<double>(inputs.size());
}

}  // namespace

int main() {
    // Define the neural network: 2 -> Linear -> 4 -> ReLU -> Linear -> 1 -> Sigmoid
    nrt::Linear layer1(2, 4);
    nrt::Linear layer2(4, 1);

    // Define all 4 possible training samples
    // (0, 0) = 0, (0, 1) = 1, (1, 0) = 1, (1, 1) = 0
    std::vector<nrt::Tensor> inputs = {make_input(0.0, 0.0), make_input(0.0, 1.0),
                                       make_input(1.0, 0.0), make_input(1.0, 1.0)};
    std::vector<nrt::Tensor> targets = {make_target(0.0), make_target(1.0), make_target(1.0),
                                        make_target(0.0)};

    // Define the training parameters
    const double learning_rate = 0.1;
    const int epochs = 5000;

    // Calculate the loss before training
    double loss_before = evaluate_average_loss(layer1, layer2, inputs, targets);
    std::cout << "Average loss BEFORE update: " << loss_before << '\n';

    for (int i = 0; i < epochs; ++i) {
        // One complete full-batch gradient descent
        layer1.zero_grad();
        layer2.zero_grad();

        // Gradients get accumulated at every iteration
        for (size_t i = 0; i < inputs.size(); ++i) {
            forward_backward_step(layer1, layer2, inputs[i], targets[i]);
        }

        // Apply the averaged gradients manually
        nrt::Tensor new_w1 = layer1.weights() - layer1.average_grad_weights() * learning_rate;
        nrt::Tensor new_b1 = layer1.bias() - layer1.average_grad_bias() * learning_rate;
        layer1.set_weights(new_w1, new_b1);

        nrt::Tensor new_w2 = layer2.weights() - layer2.average_grad_weights() * learning_rate;
        nrt::Tensor new_b2 = layer2.bias() - layer2.average_grad_bias() * learning_rate;
        layer2.set_weights(new_w2, new_b2);

        // Check if the loss has improved
        if ((i % 1000) == 0) {
            double loss_after = evaluate_average_loss(layer1, layer2, inputs, targets);
            std::cout << "Loss (" << i << "/" << epochs << "):  " << loss_after << '\n';
        }
    }

    return 0;
}
