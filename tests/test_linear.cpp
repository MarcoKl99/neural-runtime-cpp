#include <catch2/catch_test_macros.hpp>

#include "nrt/linear.hpp"
#include "nrt/tensor.hpp"

TEST_CASE("Linear construction", "[linear][construction]") {
    nrt::Linear layer(3, 2);

    SECTION("Feature counts are stored correctly") {
        REQUIRE(layer.in_features() == 3);
        REQUIRE(layer.out_features() == 2);
    }

    SECTION("Weights have correct shape") {
        REQUIRE(layer.weights().shape() == std::vector<size_t>{2, 3});
    }

    SECTION("Bias has correct shape") {
        REQUIRE(layer.bias().shape() == std::vector<size_t>{2, 1});
    }

    SECTION("Bias is zero-initialized") {
        REQUIRE(layer.bias()(0, 0) == 0.0);
        REQUIRE(layer.bias()(1, 0) == 0.0);
    }

    SECTION("Weights are randomly initialized (not all zero)") {
        // Loser Sanity-Check: bei zufaelliger Normalverteilung (sigma=0.1)
        // ist es praktisch ausgeschlossen, dass alle 6 Werte exakt 0.0 sind.
        bool all_zero = true;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (layer.weights()(i, j) != 0.0) {
                    all_zero = false;
                }
            }
        }
        REQUIRE_FALSE(all_zero);
    }
}

TEST_CASE("Linear set_weights", "[linear][set_weights]") {
    nrt::Linear layer(2, 2);

    nrt::Tensor w({2, 2});
    w(0, 0) = 1.0;
    w(0, 1) = 2.0;
    w(1, 0) = 3.0;
    w(1, 1) = 4.0;

    nrt::Tensor b({2, 1});
    b(0, 0) = 0.5;
    b(1, 0) = -0.5;

    SECTION("Weights and bias are correctly overwritten") {
        layer.set_weights(w, b);
        REQUIRE(layer.weights()(0, 0) == 1.0);
        REQUIRE(layer.weights()(1, 1) == 4.0);
        REQUIRE(layer.bias()(0, 0) == 0.5);
        REQUIRE(layer.bias()(1, 0) == -0.5);
    }

    SECTION("Wrong weight shape throws") {
        nrt::Tensor wrong_w({3, 2});
        REQUIRE_THROWS_AS(layer.set_weights(wrong_w, b), std::invalid_argument);
    }

    SECTION("Wrong bias shape throws") {
        nrt::Tensor wrong_b({3, 1});
        REQUIRE_THROWS_AS(layer.set_weights(w, wrong_b), std::invalid_argument);
    }
}

TEST_CASE("Linear forward", "[linear][forward]") {
    nrt::Linear layer(3, 2);

    // W = [[1, 2, 3], [4, 5, 6]], b = [1, 1]
    nrt::Tensor w({2, 3});
    w(0, 0) = 1.0;
    w(0, 1) = 2.0;
    w(0, 2) = 3.0;
    w(1, 0) = 4.0;
    w(1, 1) = 5.0;
    w(1, 2) = 6.0;

    nrt::Tensor b({2, 1});
    b(0, 0) = 1.0;
    b(1, 0) = 1.0;

    layer.set_weights(w, b);

    // x = [1, 1, 1]^T
    nrt::Tensor x({3, 1});
    x(0, 0) = 1.0;
    x(1, 0) = 1.0;
    x(2, 0) = 1.0;

    SECTION("Output has correct shape") {
        nrt::Tensor y = layer.forward(x);
        REQUIRE(y.shape() == std::vector<size_t>{2, 1});
    }

    SECTION("Output is correctly computed (W*x + b)") {
        // y[0] = 1*1 + 2*1 + 3*1 + 1 = 7
        // y[1] = 4*1 + 5*1 + 6*1 + 1 = 16
        nrt::Tensor y = layer.forward(x);
        REQUIRE(y(0, 0) == 7.0);
        REQUIRE(y(1, 0) == 16.0);
    }

    SECTION("Wrong input shape throws") {
        nrt::Tensor wrong_x({4, 1});
        REQUIRE_THROWS_AS(layer.forward(wrong_x), std::invalid_argument);
    }

    SECTION("1D input throws (must be {in_features, 1})") {
        nrt::Tensor wrong_x({3});
        REQUIRE_THROWS_AS(layer.forward(wrong_x), std::invalid_argument);
    }
}

TEST_CASE("Linear backward", "[linear][backward]") {
    nrt::Linear layer(2, 2);

    // W = [[1, 2], [3, 4]], b = [0, 0]
    nrt::Tensor w({2, 2});
    w(0, 0) = 1.0;
    w(0, 1) = 2.0;
    w(1, 0) = 3.0;
    w(1, 1) = 4.0;
    nrt::Tensor b({2, 1});
    layer.set_weights(w, b);

    // x = [1, 2]^T
    nrt::Tensor x({2, 1});
    x(0, 0) = 1.0;
    x(1, 0) = 2.0;

    // grad_output = [1, 1]^T
    nrt::Tensor grad_output({2, 1});
    grad_output(0, 0) = 1.0;
    grad_output(1, 0) = 1.0;

    SECTION("backward without prior forward throws") {
        REQUIRE_THROWS_AS(layer.backward(grad_output), std::logic_error);
    }

    SECTION("grad_x is correctly computed (W^T * grad_output)") {
        layer.forward(x);
        nrt::Tensor grad_x = layer.backward(grad_output);
        // Expected: [1*1+3*1, 2*1+4*1] = [4, 6]
        REQUIRE(grad_x.shape() == std::vector<size_t>{2, 1});
        REQUIRE(grad_x(0, 0) == 4.0);
        REQUIRE(grad_x(1, 0) == 6.0);
    }

    SECTION("grad_output shape mismatch throws") {
        layer.forward(x);
        nrt::Tensor wrong_grad({3, 1});
        REQUIRE_THROWS_AS(layer.backward(wrong_grad), std::invalid_argument);
    }

    SECTION("average_grad_weights before any backward call throws") {
        REQUIRE_THROWS_AS(layer.average_grad_weights(), std::logic_error);
    }

    SECTION("average_grad_bias before any backward call throws") {
        REQUIRE_THROWS_AS(layer.average_grad_bias(), std::logic_error);
    }

    SECTION("Single backward call: average equals raw gradient") {
        layer.forward(x);
        layer.backward(grad_output);

        // dL/dW = grad_output * x^T = [[1,2],[1,2]]
        nrt::Tensor gw = layer.average_grad_weights();
        REQUIRE(gw(0, 0) == 1.0);
        REQUIRE(gw(0, 1) == 2.0);
        REQUIRE(gw(1, 0) == 1.0);
        REQUIRE(gw(1, 1) == 2.0);

        // dL/db = grad_output = [1, 1]
        nrt::Tensor gb = layer.average_grad_bias();
        REQUIRE(gb(0, 0) == 1.0);
        REQUIRE(gb(1, 0) == 1.0);
    }

    SECTION("Two identical backward calls: average equals single-call gradient") {
        layer.forward(x);
        layer.backward(grad_output);
        layer.forward(x);
        layer.backward(grad_output);

        // Gradients are equal -> average stays the same
        // Not the sum of the gradients - chosen to use average (would be [2,4]/[2,4] instead of
        // [1,2]/[1,2])
        nrt::Tensor gw = layer.average_grad_weights();
        REQUIRE(gw(0, 0) == 1.0);
        REQUIRE(gw(0, 1) == 2.0);
        REQUIRE(gw(1, 0) == 1.0);
        REQUIRE(gw(1, 1) == 2.0);

        nrt::Tensor gb = layer.average_grad_bias();
        REQUIRE(gb(0, 0) == 1.0);
        REQUIRE(gb(1, 0) == 1.0);
    }

    SECTION("zero_grad resets accumulated gradients and count") {
        layer.forward(x);
        layer.backward(grad_output);

        layer.zero_grad();

        // Nach zero_grad muss average_grad_* wieder werfen (count == 0)
        REQUIRE_THROWS_AS(layer.average_grad_weights(), std::logic_error);
        REQUIRE_THROWS_AS(layer.average_grad_bias(), std::logic_error);
    }

    SECTION("zero_grad followed by a fresh backward gives correct fresh gradient") {
        layer.forward(x);
        layer.backward(grad_output);
        layer.zero_grad();

        // Anderer grad_output, um sicherzustellen, dass nichts vom ersten Aufruf uebrig ist
        nrt::Tensor grad_output_2({2, 1});
        grad_output_2(0, 0) = 2.0;
        grad_output_2(1, 0) = 2.0;

        layer.forward(x);
        layer.backward(grad_output_2);

        // dL/dW = [2,2]^T * [1,2] = [[2,4],[2,4]]
        nrt::Tensor gw = layer.average_grad_weights();
        REQUIRE(gw(0, 0) == 2.0);
        REQUIRE(gw(0, 1) == 4.0);
        REQUIRE(gw(1, 0) == 2.0);
        REQUIRE(gw(1, 1) == 4.0);
    }
}
