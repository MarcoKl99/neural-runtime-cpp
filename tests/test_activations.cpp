#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "nrt/activations.hpp"
#include "nrt/tensor.hpp"

TEST_CASE("relu", "[activations][relu]") {
    SECTION("1D: positive, negative and zero values") {
        nrt::Tensor x({4});
        x(0) = 3.0;
        x(1) = -2.0;
        x(2) = 0.0;
        x(3) = -0.001;

        nrt::Tensor y = nrt::relu(x);
        REQUIRE(y(0) == 3.0);
        REQUIRE(y(1) == 0.0);
        REQUIRE(y(2) == 0.0);
        REQUIRE(y(3) == 0.0);
    }

    SECTION("2D: shape is preserved") {
        nrt::Tensor x({2, 2});
        x(0, 0) = -1.0;
        x(0, 1) = 2.0;
        x(1, 0) = -3.0;
        x(1, 1) = 4.0;

        nrt::Tensor y = nrt::relu(x);
        REQUIRE(y.shape() == std::vector<size_t>{2, 2});
        REQUIRE(y(0, 0) == 0.0);
        REQUIRE(y(0, 1) == 2.0);
        REQUIRE(y(1, 0) == 0.0);
        REQUIRE(y(1, 1) == 4.0);
    }

    SECTION("Original tensor remains unchanged") {
        nrt::Tensor x({2});
        x(0) = -5.0;
        x(1) = 5.0;
        nrt::Tensor y = nrt::relu(x);
        REQUIRE(x(0) == -5.0);
        REQUIRE(x(1) == 5.0);
    }
}

TEST_CASE("relu_derivative", "[activations][relu]") {
    SECTION("Positive, negative and zero values") {
        nrt::Tensor x({3});
        x(0) = 2.0;
        x(1) = -2.0;
        x(2) = 0.0;

        nrt::Tensor d = nrt::relu_derivative(x);
        REQUIRE(d(0) == 1.0);
        REQUIRE(d(1) == 0.0);
        // Convention used here: Derivative at 0 is 0 (x <= 0 -> 0)
        REQUIRE(d(2) == 0.0);
    }
}

TEST_CASE("sigmoid", "[activations][sigmoid]") {
    SECTION("sigmoid(0) == 0.5") {
        nrt::Tensor x({1});
        x(0) = 0.0;
        nrt::Tensor y = nrt::sigmoid(x);
        REQUIRE(y(0) == 0.5);
    }

    SECTION("Output is always between 0 and 1") {
        nrt::Tensor x({5});
        x(0) = -100.0;
        x(1) = -50.0;
        x(2) = 0.0;
        x(3) = 50.0;
        x(4) = 100.0;

        nrt::Tensor y = nrt::sigmoid(x);
        for (size_t i = 0; i < 5; ++i) {
            REQUIRE(y(i) >= 0.0);  // >= due to rounding scenatios to 0.0
            REQUIRE(y(i) <= 1.0);  // <= due to rounding scenraios to 1.0
        }
    }

    SECTION("Known value: sigmoid(2) approx 0.8808") {
        nrt::Tensor x({1});
        x(0) = 2.0;
        nrt::Tensor y = nrt::sigmoid(x);
        REQUIRE(std::abs(y(0) - 0.8807970779778824) < 1e-9);
    }

    SECTION("2D: shape is preserved") {
        nrt::Tensor x({2, 2});
        nrt::Tensor y = nrt::sigmoid(x);
        REQUIRE(y.shape() == std::vector<size_t>{2, 2});
    }
}

TEST_CASE("sigmoid_derivative", "[activations][sigmoid]") {
    SECTION("At x=0: sigmoid(0)*(1-sigmoid(0)) = 0.25") {
        nrt::Tensor x({1});
        x(0) = 0.0;
        nrt::Tensor d = nrt::sigmoid_derivative(x);
        REQUIRE(std::abs(d(0) - 0.25) < 1e-12);
    }

    SECTION("Consistent with sigmoid(x) * (1 - sigmoid(x))") {
        nrt::Tensor x({5});
        x(0) = -50.5;
        x(1) = -1.32155;
        x(2) = 0.0;
        x(3) = 3.123456;
        x(4) = 72.4;

        nrt::Tensor s = nrt::sigmoid(x);
        nrt::Tensor d = nrt::sigmoid_derivative(x);

        for (size_t i = 0; i < 3; ++i) {
            double expected = s(i) * (1.0 - s(i));
            REQUIRE(std::abs(d(i) - expected) < 1e-12);
        }
    }
}
