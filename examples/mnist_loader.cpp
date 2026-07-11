#include "mnist_loader.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace mnist {

namespace {

// IDX header fields are big-endian regardless of host byte order - read byte-by-byte
// and reassemble manually rather than reinterpret_cast'ing a uint32_t directly.
uint32_t read_big_endian_u32(std::ifstream& stream) {
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    if (!stream) throw std::runtime_error("mnist: unexpected end of file while reading header");
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

std::vector<uint8_t> read_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("mnist: could not open " + path);

    uint32_t magic = read_big_endian_u32(file);
    if (magic != 0x00000801) {
        throw std::runtime_error("mnist: bad magic number in label file " + path);
    }
    uint32_t count = read_big_endian_u32(file);

    std::vector<uint8_t> labels(count);
    file.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(count));
    if (!file) throw std::runtime_error("mnist: unexpected end of file while reading labels");

    return labels;
}

struct RawImages {
    std::vector<uint8_t> pixels;  // count * rows * cols bytes, row-major, image-major
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
};

RawImages read_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("mnist: could not open " + path);

    uint32_t magic = read_big_endian_u32(file);
    if (magic != 0x00000803) {
        throw std::runtime_error("mnist: bad magic number in image file " + path);
    }

    RawImages raw;
    raw.count = read_big_endian_u32(file);
    raw.rows = read_big_endian_u32(file);
    raw.cols = read_big_endian_u32(file);

    size_t total = static_cast<size_t>(raw.count) * raw.rows * raw.cols;
    raw.pixels.resize(total);
    file.read(reinterpret_cast<char*>(raw.pixels.data()), static_cast<std::streamsize>(total));
    if (!file) throw std::runtime_error("mnist: unexpected end of file while reading images");

    return raw;
}

}  // namespace

Dataset load(const std::string& images_path, const std::string& labels_path) {
    RawImages raw = read_images(images_path);
    std::vector<uint8_t> raw_labels = read_labels(labels_path);

    if (raw.count != raw_labels.size()) {
        throw std::runtime_error("mnist: image count does not match label count");
    }

    size_t image_size = static_cast<size_t>(raw.rows) * raw.cols;  // 784 for MNIST's 28x28

    Dataset dataset;
    dataset.images.reserve(raw.count);
    dataset.labels.reserve(raw.count);

    for (uint32_t i = 0; i < raw.count; ++i) {
        auto image = std::make_shared<nrt::Tensor>(std::vector<size_t>{image_size, 1});
        for (size_t p = 0; p < image_size; ++p) {
            // Normalize 0-255 grayscale intensity to [0, 1].
            (*image)(p, 0) = static_cast<double>(raw.pixels[i * image_size + p]) / 255.0;
        }
        dataset.images.push_back(image);
        dataset.labels.push_back(static_cast<size_t>(raw_labels[i]));
    }

    return dataset;
}

}  // namespace mnist
