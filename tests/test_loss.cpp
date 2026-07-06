#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "nrt/loss.hpp"
#include "nrt/tensor.hpp"

TEST_CASE("mse", "[loss][mse]") {
    SECTION("Identical tensors give zero loss") {
        nrt::Tensor y_hat({3});
        nrt::Tensor y({3});
        y_hat(0) = 1.0;
        y_hat(1) = 2.0;
        y_hat(2) = 3.0;
        y(0) = 1.0;
        y(1) = 2.0;
        y(2) = 3.0;

        REQUIRE(nrt::mse(y_hat, y) == 0.0);
    }

    SECTION("Known value, 1D") {
        // y_hat = [1, 2], y = [3, 4]
        // diff = [-2, -2], squared = [4, 4], sum = 8, mse = 8/2 = 4
        nrt::Tensor y_hat({2});
        nrt::Tensor y({2});
        y_hat(0) = 1.0;
        y_hat(1) = 2.0;
        y(0) = 3.0;
        y(1) = 4.0;

        REQUIRE(nrt::mse(y_hat, y) == 4.0);
    }

    SECTION("Known value, 2D") {
        // y_hat = [[0, 0], [0, 0]], y = [[1, 1], [1, 1]]
        // diff = -1 everywhere, squared = 1 everywhere, sum = 4, mse = 4/4 = 1
        nrt::Tensor y_hat({2, 2});
        nrt::Tensor y({2, 2});
        y(0, 0) = 1.0;
        y(0, 1) = 1.0;
        y(1, 0) = 1.0;
        y(1, 1) = 1.0;

        REQUIRE(nrt::mse(y_hat, y) == 1.0);
    }

    SECTION("Loss is symmetric in sign: mse(a, b) == mse(b, a)") {
        nrt::Tensor a({2});
        nrt::Tensor b({2});
        a(0) = 1.0;
        a(1) = 5.0;
        b(0) = 3.0;
        b(1) = 2.0;

        REQUIRE(nrt::mse(a, b) == nrt::mse(b, a));
    }

    SECTION("Shape mismatch throws") {
        nrt::Tensor y_hat({2});
        nrt::Tensor y({3});
        REQUIRE_THROWS_AS(nrt::mse(y_hat, y), std::invalid_argument);
    }

    SECTION("Rank mismatch throws") {
        nrt::Tensor y_hat({2, 2});
        nrt::Tensor y({4});
        REQUIRE_THROWS_AS(nrt::mse(y_hat, y), std::invalid_argument);
    }
}
