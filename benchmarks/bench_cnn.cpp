#include <benchmark/benchmark.h>

#include "nrt/conv2d.hpp"
#include "nrt/flatten.hpp"
#include "nrt/linear.hpp"
#include "nrt/maxpool2d.hpp"
#include "nrt/operations.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

/**************/
/**  Conv2D  **/
/**************/
static void BM_Conv2D_Forward(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 1, 28, 28});
    input->randomize(42);

    nrt::Conv2D conv(1, 16, 3, nrt::WeightInit::He);

    for (auto _ : state) {
        auto output = conv.forward(input);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Conv2D_Forward);

static void BM_Conv2D_BackwardPass(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 1, 28, 28});
    input->randomize(42);

    nrt::Conv2D conv(1, 16, 3, nrt::WeightInit::He);

    for (auto _ : state) {
        state.PauseTiming();
        auto output = conv.forward(input);
        auto grad_output = std::make_shared<nrt::Tensor>(output->shape());
        grad_output->randomize(42);
        output->accumulate_gradient(*grad_output);
        state.ResumeTiming();

        output->backward();

        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Conv2D_BackwardPass);

/*****************/
/**  MaxPool2D  **/
/*****************/
static void BM_MaxPool2D_Forward(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 16, 26, 26});
    input->randomize(42);

    nrt::MaxPool2D pool;

    for (auto _ : state) {
        auto output = pool.forward(input);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_MaxPool2D_Forward);

static void BM_MaxPool2D_BackwardPass(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 16, 26, 26});
    input->randomize(42);

    nrt::MaxPool2D pool;

    for (auto _ : state) {
        state.PauseTiming();
        auto output = pool.forward(input);
        auto grad_output = std::make_shared<nrt::Tensor>(output->shape());
        grad_output->randomize(43);
        output->accumulate_gradient(*grad_output);
        state.ResumeTiming();

        output->backward();

        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_MaxPool2D_BackwardPass);

/***************/
/**  Flatten  **/
/***************/
static void BM_Flatten_Forward(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 16, 13, 13});
    input->randomize(42);

    nrt::Flatten flatten;

    for (auto _ : state) {
        auto output = flatten.forward(input);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Flatten_Forward);

static void BM_Flatten_BackwardPass(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 16, 13, 13});
    input->randomize(42);

    nrt::Flatten flatten;

    for (auto _ : state) {
        state.PauseTiming();
        auto output = flatten.forward(input);
        auto grad_output = std::make_shared<nrt::Tensor>(output->shape());
        grad_output->randomize(43);
        output->accumulate_gradient(*grad_output);
        state.ResumeTiming();

        output->backward();

        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Flatten_BackwardPass);

/***************/
/**  Linear1  **/
/***************/
static void BM_Linear1_Forward(benchmark::State& state) {
    size_t flattened_out_dim = 16 * 13 * 13;
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, flattened_out_dim});
    input->randomize(42);

    nrt::Linear linear(flattened_out_dim, 128, nrt::WeightInit::He, 42);

    for (auto _ : state) {
        auto output = linear.forward(input);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Linear1_Forward);

static void BM_Linear1_BackwardPass(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 16 * 13 * 13});
    input->randomize(42);

    nrt::Linear linear(16 * 13 * 13, 128, nrt::WeightInit::He, 42);

    for (auto _ : state) {
        state.PauseTiming();
        auto output = linear.forward(input);
        auto grad_output = std::make_shared<nrt::Tensor>(output->shape());
        grad_output->randomize(43);
        output->accumulate_gradient(*grad_output);
        state.ResumeTiming();

        output->backward();

        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Linear1_BackwardPass);

/***************/
/**  Linear2  **/
/***************/
static void BM_Linear2_Forward(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 128});
    input->randomize(42);

    nrt::Linear linear(128, 10, nrt::WeightInit::Xavier, 42);

    for (auto _ : state) {
        auto output = linear.forward(input);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Linear2_Forward);

static void BM_Linear2_BackwardPass(benchmark::State& state) {
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 128});
    input->randomize(42);

    nrt::Linear linear(128, 10, nrt::WeightInit::Xavier, 42);

    for (auto _ : state) {
        state.PauseTiming();
        auto output = linear.forward(input);
        auto grad_output = std::make_shared<nrt::Tensor>(output->shape());
        grad_output->randomize(43);
        output->accumulate_gradient(*grad_output);
        state.ResumeTiming();

        output->backward();

        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Linear2_BackwardPass);

/************/
/**  Full  **/
/************/
static void BM_CNN_FullForwardPass(benchmark::State& state) {
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Conv2D>(1, 16, 3, nrt::WeightInit::He));
    modules.push_back(std::make_unique<nrt::MaxPool2D>());
    modules.push_back(std::make_unique<nrt::Flatten>());
    modules.push_back(std::make_unique<nrt::Linear>(16 * 13 * 13, 128, nrt::WeightInit::He));
    modules.push_back(std::make_unique<nrt::Linear>(128, 10, nrt::WeightInit::Xavier));
    nrt::Sequential model(std::move(modules));

    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 1, 28, 28});
    input->randomize(42);

    for (auto _ : state) {
        auto output = model.forward(input);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_CNN_FullForwardPass);

static void BM_CNN_FullBackwardPass(benchmark::State& state) {
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Conv2D>(1, 16, 3, nrt::WeightInit::He));
    modules.push_back(std::make_unique<nrt::MaxPool2D>());
    modules.push_back(std::make_unique<nrt::Flatten>());
    modules.push_back(std::make_unique<nrt::Linear>(16 * 13 * 13, 128, nrt::WeightInit::He));
    modules.push_back(std::make_unique<nrt::Linear>(128, 10, nrt::WeightInit::Xavier));
    nrt::Sequential model(std::move(modules));

    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{32, 1, 28, 28});
    input->randomize(42);

    for (auto _ : state) {
        state.PauseTiming();
        auto output = model.forward(input);
        auto grad_output = std::make_shared<nrt::Tensor>(output->shape());
        grad_output->randomize(43);
        output->accumulate_gradient(*grad_output);
        state.ResumeTiming();

        output->backward();

        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_CNN_FullBackwardPass);

BENCHMARK_MAIN();
