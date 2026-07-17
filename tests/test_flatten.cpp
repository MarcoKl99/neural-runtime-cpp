#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "nrt/flatten.hpp"
#include "nrt/tensor.hpp"

TEST_CASE("Flatten basic shape transformation", "[flatten][forward]") {
    nrt::Flatten flatten;

    // {3, 4, 5} -> {3, 20}
    auto x = std::make_shared<nrt::Tensor>(std::vector<size_t>{3, 4, 5});
    auto y = flatten.forward(x);

    REQUIRE(y->shape() == std::vector<size_t>{3, 20});
}

TEST_CASE("Flatten rank-4 (batch, channels, height, width)", "[flatten][forward]") {
    nrt::Flatten flatten;

    // {2, 3, 4, 5} -> {2, 60}  (batch=2, 3*4*5=60)
    auto x = std::make_shared<nrt::Tensor>(std::vector<size_t>{2, 3, 4, 5});
    auto y = flatten.forward(x);

    REQUIRE(y->shape() == std::vector<size_t>{2, 60});
}

TEST_CASE("Flatten with single element dimensions", "[flatten][forward]") {
    nrt::Flatten flatten;

    // {4, 1, 1, 1} -> {4, 1}
    auto x = std::make_shared<nrt::Tensor>(std::vector<size_t>{4, 1, 1, 1});
    auto y = flatten.forward(x);

    REQUIRE(y->shape() == std::vector<size_t>{4, 1});
}

TEST_CASE("Flatten with 2D input (edge case)", "[flatten][forward]") {
    nrt::Flatten flatten;

    // {5, 7} -> {5, 7}  (already 2D, should be unchanged)
    auto x = std::make_shared<nrt::Tensor>(std::vector<size_t>{5, 7});
    auto y = flatten.forward(x);

    REQUIRE(y->shape() == std::vector<size_t>{5, 7});
}

TEST_CASE("Flatten has no learnable parameters", "[flatten][parameters]") {
    nrt::Flatten flatten;

    auto params = flatten.parameters();
    REQUIRE(params.empty());
}

TEST_CASE("Flatten gradient flow with concrete values {1,2,2} → {1,4}",
          "[flatten][backward][gradcheck]") {
    nrt::Flatten flatten;

    // Create {1, 2, 2} tensor with known values
    auto x = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 2, 2});
    (*x)(0, 0, 0) = 1.0;
    (*x)(0, 0, 1) = 2.0;
    (*x)(0, 1, 0) = 3.0;
    (*x)(0, 1, 1) = 4.0;

    // Forward flatten: {1, 2, 2} → {1, 4}
    auto y = flatten.forward(x);
    REQUIRE(y->shape() == std::vector<size_t>{1, 4});

    // Verify the flattened values are correct
    REQUIRE((*y)(0, 0) == 1.0);  // x(0,0,0)
    REQUIRE((*y)(0, 1) == 2.0);  // x(0,0,1)
    REQUIRE((*y)(0, 2) == 3.0);  // x(0,1,0)
    REQUIRE((*y)(0, 3) == 4.0);  // x(0,1,1)

    // Create gradient tensor {1, 4} simulating backprop from next layer
    nrt::Tensor grad_output({1, 4});
    grad_output(0, 0) = 0.1;
    grad_output(0, 1) = 0.2;
    grad_output(0, 2) = 0.3;
    grad_output(0, 3) = 0.4;

    // Backward pass: reshape gradients back to {1, 2, 2}
    y->backward_impl(grad_output);

    // Verify gradients propagated correctly back to x
    auto grad_x = x->gradient();
    REQUIRE(grad_x(0, 0, 0) == 0.1);  // from grad_output(0, 0)
    REQUIRE(grad_x(0, 0, 1) == 0.2);  // from grad_output(0, 1)
    REQUIRE(grad_x(0, 1, 0) == 0.3);  // from grad_output(0, 2)
    REQUIRE(grad_x(0, 1, 1) == 0.4);  // from grad_output(0, 3)
}
