#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nrt/tensor.hpp"

namespace mnist {

struct Dataset {
    std::vector<std::shared_ptr<nrt::Tensor>> images;  // each {784, 1}, pixels normalized to [0, 1]
    std::vector<size_t> labels;                        // class index in [0, 9]
};

// Reads one MNIST split (train or test) from its IDX image + label files.
Dataset load(const std::string& images_path, const std::string& labels_path);

}  // namespace mnist
