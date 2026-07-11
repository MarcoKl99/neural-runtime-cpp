#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "mnist_loader.hpp"
#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/module.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

/*
MNIST with a raw MLP: Linear(784,256,He) -> ReLU -> Linear(256,128,He) -> ReLU ->
Linear(128,10,Xavier) -> CrossEntropyLoss, trained with mini-batch SGD.

Runs on a SUBSET of the full 60k/10k MNIST split by default (kTrainSubsetSize /
kTestSubsetSize below) - Tensor::matmul is a plain unvectorized triple loop with
no batch dimension, so a full-dataset run is genuinely slow. Once this converges
on the subset, dial the sizes up (they clamp to the dataset's real size, so a
large value like 60000 safely means "all of it").
*/

namespace {

constexpr size_t kTrainSubsetSize = 5000;
constexpr size_t kTestSubsetSize = 1000;
constexpr size_t kBatchSize = 32;
constexpr int kEpochs = 15;
constexpr double kLearningRate = 0.1;
constexpr unsigned int kSeed = 42;

// Truncate a loaded dataset down to its first `size` samples (subset for fast iteration).
mnist::Dataset take_subset(mnist::Dataset dataset, size_t size) {
    size = std::min(size, dataset.images.size());
    dataset.images.resize(size);
    dataset.labels.resize(size);
    return dataset;
}

size_t argmax(const nrt::Tensor& logits) {
    size_t best = 0;
    for (size_t i = 1; i < logits.shape()[0]; ++i) {
        if (logits(i, 0) > logits(best, 0)) best = i;
    }
    return best;
}

double evaluate_accuracy(nrt::Sequential& model, const mnist::Dataset& dataset) {
    size_t correct = 0;
    for (size_t i = 0; i < dataset.images.size(); ++i) {
        auto logits = model.forward(dataset.images[i]);
        if (argmax(*logits) == dataset.labels[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.images.size());
}

double evaluate_average_loss(nrt::Sequential& model, nrt::CrossEntropyLoss& loss_fn,
                             const mnist::Dataset& dataset) {
    double total_loss = 0.0;
    for (size_t i = 0; i < dataset.images.size(); ++i) {
        auto logits = model.forward(dataset.images[i]);
        total_loss += nrt::cross_entropy(*logits, dataset.labels[i]);
    }
    return total_loss / static_cast<double>(dataset.images.size());
}

}  // namespace

int main() {
    std::cout << "Loading MNIST...\n";
    auto train_data = take_subset(mnist::load("data/MNIST/raw/train-images-idx3-ubyte",
                                              "data/MNIST/raw/train-labels-idx1-ubyte"),
                                  kTrainSubsetSize);
    auto test_data = take_subset(mnist::load("data/MNIST/raw/t10k-images-idx3-ubyte",
                                             "data/MNIST/raw/t10k-labels-idx1-ubyte"),
                                 kTestSubsetSize);

    std::cout << "Train subset: " << train_data.images.size()
              << " images, test subset: " << test_data.images.size() << " images\n";

    // Model: 784 -> 256 (ReLU, He) -> 128 (ReLU, He) -> 10 (raw logits, Xavier)
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(784, 256, nrt::WeightInit::He, kSeed));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(256, 128, nrt::WeightInit::He, kSeed + 1));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(128, 10, nrt::WeightInit::Xavier, kSeed + 2));
    nrt::Sequential model(std::move(modules));

    nrt::CrossEntropyLoss loss_fn;
    nrt::SGD optimizer(model.parameters(), kLearningRate / static_cast<double>(kBatchSize));

    std::cout << "\nInitial test accuracy: " << evaluate_accuracy(model, test_data) << "\n\n";

    std::vector<size_t> indices(train_data.images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(kSeed);

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t batch_start = 0; batch_start < indices.size(); batch_start += kBatchSize) {
            size_t batch_end = std::min(batch_start + kBatchSize, indices.size());

            optimizer.zero_grad();
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = indices[i];
                auto logits = model.forward(train_data.images[idx]);
                auto loss = loss_fn.forward(logits, train_data.labels[idx]);
                loss->backward();
            }
            optimizer.step();
        }

        double train_loss = evaluate_average_loss(model, loss_fn, train_data);
        double test_accuracy = evaluate_accuracy(model, test_data);
        std::cout << "Epoch " << (epoch + 1) << "/" << kEpochs << " - train loss: " << train_loss
                  << " - test accuracy: " << test_accuracy << '\n';
    }

    double final_train_loss = evaluate_average_loss(model, loss_fn, train_data);
    double final_test_accuracy = evaluate_accuracy(model, test_data);
    std::cout << "Final train loss: " << final_train_loss << '\n';
    std::cout << "Final test accuracy: " << final_test_accuracy << '\n';

    return 0;
}
