#include <iostream>
#include <memory>
#include <vector>

#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/module.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

/*
Toy 3-class classification: three well-separated 2D clusters, classified via
Sequential(Linear(2,8,He) -> ReLU -> Linear(8,3,Xavier)) feeding raw logits into
CrossEntropyLoss (softmax applied internally - see include/nrt/loss.hpp).

This is the step between XOR and MNIST: the first genuinely multi-class (not
binary) problem, and the first end-to-end use of Softmax + CrossEntropyLoss.
*/

namespace {

std::shared_ptr<nrt::Tensor> make_input(double x0, double x1) {
    auto x = std::make_shared<nrt::Tensor>(std::vector<size_t>{2, 1});
    (*x)(0, 0) = x0;
    (*x)(1, 0) = x1;
    return x;
}

// Predicted class = index of the largest logit (softmax is monotonic, so this
// is equivalent to picking the largest probability without computing it).
size_t argmax(const nrt::Tensor& logits) {
    size_t best = 0;
    for (size_t i = 1; i < logits.shape()[0]; ++i) {
        if (logits(i, 0) > logits(best, 0)) best = i;
    }
    return best;
}

double evaluate_accuracy(nrt::Sequential& model,
                         const std::vector<std::shared_ptr<nrt::Tensor>>& inputs,
                         const std::vector<size_t>& targets) {
    size_t correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto logits = model.forward(inputs[i]);
        if (argmax(*logits) == targets[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(inputs.size());
}

double evaluate_average_loss(nrt::Sequential& model, nrt::CrossEntropyLoss& loss_fn,
                             const std::vector<std::shared_ptr<nrt::Tensor>>& inputs,
                             const std::vector<size_t>& targets) {
    double total_loss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto logits = model.forward(inputs[i]);
        total_loss += nrt::cross_entropy(*logits, targets[i]);
    }
    return total_loss / static_cast<double>(inputs.size());
}

}  // namespace

int main() {
    // Three well-separated clusters in 2D, 4 points each.
    std::vector<std::shared_ptr<nrt::Tensor>> inputs = {
        // Class 0, centered near (-2, -2)
        make_input(-2.0, -2.0),
        make_input(-2.5, -1.5),
        make_input(-1.5, -2.5),
        make_input(-2.0, -1.0),
        // Class 1, centered near (2, -2)
        make_input(2.0, -2.0),
        make_input(2.5, -1.5),
        make_input(1.5, -2.5),
        make_input(2.0, -1.0),
        // Class 2, centered near (0, 2)
        make_input(0.0, 2.0),
        make_input(0.5, 2.5),
        make_input(-0.5, 2.5),
        make_input(0.0, 1.0),
    };
    std::vector<size_t> targets = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};

    // Model: 2 -> 8 (ReLU, He) -> 3 (raw logits, Xavier). No final activation -
    // CrossEntropyLoss consumes logits directly and applies softmax internally.
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(2, 8, nrt::WeightInit::He));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(8, 3, nrt::WeightInit::Xavier));
    nrt::Sequential model(std::move(modules));

    nrt::CrossEntropyLoss loss_fn;
    const int epochs = 2000;
    const double learning_rate = 0.1;
    const size_t batch_size = inputs.size();
    nrt::SGD optimizer(model.parameters(), learning_rate / batch_size);

    std::cout << "Initial loss:     " << evaluate_average_loss(model, loss_fn, inputs, targets)
              << '\n';
    std::cout << "Initial accuracy: " << evaluate_accuracy(model, inputs, targets) << '\n';

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto logits = model.forward(inputs[i]);
            auto loss = loss_fn.forward(logits, targets[i]);
            loss->backward();
        }
        optimizer.step();

        if ((epoch % 200) == 0) {
            std::cout << "Epoch " << epoch << "/" << epochs
                      << " - loss: " << evaluate_average_loss(model, loss_fn, inputs, targets)
                      << " - accuracy: " << evaluate_accuracy(model, inputs, targets) << '\n';
        }
    }

    std::cout << "Final loss:       " << evaluate_average_loss(model, loss_fn, inputs, targets)
              << '\n';
    std::cout << "Final accuracy:   " << evaluate_accuracy(model, inputs, targets) << '\n';

    return 0;
}
