# Neural Runtime C++ 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)<br>
[![CI](https://github.com/MarcoKl99/neural-runtime-cpp/actions/workflows/ci.yaml/badge.svg)](https://github.com/MarcoKl99/neural-runtime-cpp/actions/workflows/ci.yaml)

A C++20 neural runtime built from first principles to explore the engineering behind deep learning systems.

## Components 🧱

- Reverse-mode automatic differentiation
- Tensor library with numerical operations
- Neural network modules, optimizers and loss functions
- Gradient checking and PyTorch validation
- End-to-end MNIST training pipeline

## Motivation 🏃‍➡️

The goal of this project is to develop a deeper understanding of the architecture and implementation behind deep learning frameworks.

By implementing every component from scratch in C++, this project explores topics like

- Tensor computation
- Automatic differentiation
- Numerical stability
- Architecture

The usage of the functionalities follows (to some extent 😉) the patterns known from e.g. PyTorch as shown in the example below.

```cpp
// A 2 -> 4 -> 1 MLP, trained with autograd + SGD
std::vector<std::unique_ptr<nrt::Module>> layers;
layers.push_back(std::make_unique<nrt::Linear>(2, 4));
layers.push_back(std::make_unique<nrt::ReLU>());
layers.push_back(std::make_unique<nrt::Linear>(4, 1));
layers.push_back(std::make_unique<nrt::Sigmoid>());
nrt::Sequential model(std::move(layers));

nrt::MSELoss loss_fn;
nrt::SGD     optimizer(model.parameters(), /*lr=*/0.05);

optimizer.zero_grad();
auto y_hat = model.forward(x);            // builds the computation graph
auto loss  = loss_fn.forward(y_hat, target);
loss->backward();                         // reverse-mode autograd
optimizer.step();                         // gradient-descent update
```

---

## Features ✨

**Tensor** (`nrt/tensor.hpp`)
- 1D / 2D, row-major, double precision, bounds-checked element access
- Arithmetic: `+`, `-`, `+=`, `-=`, scalar `*`, Hadamard product, `matmul`, `transpose`, `sum`

**Autograd engine** (`nrt/computation_node.hpp`, `nrt/operations.hpp`)
- Define-by-run **computational graph** built during the forward pass
- Reverse-mode automatic differentiation via `Tensor::backward()`
- Graph edges use shared ownership (`std::shared_ptr`), so intermediate values stay alive for the backward pass
- Differentiable ops: `matmul_autodiff`, `add_autodiff`, `subtract_autodiff`, `scalar_mult_autodiff`
- Per-tensor gradient storage: `gradient()`, `accumulate_gradient()`, `zero_grad()`

**Modules** (`nrt/module.hpp`)
- `Module` base class (`forward` / `parameters`), mirroring `torch.nn.Module`
- `Linear` — affine transform `y = W·x + b` (weights `{out, in}`, bias `{out, 1}`), with **He / Xavier weight initialization** (`WeightInit::He` / `WeightInit::Xavier`) and an optional seed for reproducible init
- `ReLU`, `Sigmoid`, `Softmax` — activations as graph nodes
- `Sequential` — chains modules and aggregates their parameters

**Losses** (`nrt/loss.hpp`)
- `MSELoss` — mean-squared error as a differentiable graph node (`forward(y_hat, target)` → scalar tensor)
- `CrossEntropyLoss` — softmax + negative-log-likelihood fused into one graph node (`forward(logits, target_class)` → scalar tensor); takes **raw logits**, softmax is applied internally
- Free functions `mse` / `mse_derivative`, `cross_entropy` / `cross_entropy_grad` for graph-free use

**Optimizer** (`nrt/optimizer.hpp`)
- `SGD` — reads gradients straight from the parameter tensors (`step()`, `zero_grad()`)

**Tested & validated**
- Catch2 unit tests for every component
- **Integration tests** that exercise the whole system:
  - a **finite-difference gradient check** (analytic `backward()` vs. numerical gradients)
  - a training loop that provably minimizes a convex regression
  - a full `Sequential` MLP learning XOR
- Forward and backward numerically validated against a PyTorch reference

---

## Project layout 🗂️

| Path | Purpose |
|------|---------|
| `include/nrt/` | Public headers (tensor, autograd, modules, loss, optimizer) |
| `src/` | Implementations |
| `tests/` | Catch2 unit + integration tests (`test_*.cpp`) |
| `examples/` | Runnable demos (`xor_forward`, `xor_training`, `xor_deep`, `three_class`) |
| `notes/` | Write-ups, incl. the Dying-ReLU observation |
| `CMakeLists.txt` | Build configuration |

---

## Build & run 🛠️

Requires a C++20 compiler and CMake ≥ 3.20. Catch2 is fetched automatically.

```bash
# Configure & build
cmake -B build && cmake --build build

# Run the full test suite
cd build && ctest --output-on-failure
#   (or run the test binary directly: ./build/nrt_tests)

# Run the examples
./build/xor_training   # full training loop (autograd + SGD)
./build/xor_forward    # forward pass only
./build/xor_deep       # deeper XOR MLP (He/Xavier init)
./build/three_class    # 3-class classification (Softmax + CrossEntropyLoss)
./build/mnist_mlp      # 10-class classification on the MNIST dataset of handwritten digits
```

---

## How the autograd works 🔬

Each differentiable operation returns a new tensor that stores a **computation node**: the inputs it was built from and a closure describing how to push a gradient back to them. Calling `backward()` on the final (scalar) output seeds a gradient of `1.0` and walks the graph in reverse, applying the chain rule at each node and accumulating gradients into the leaf tensors (your parameters). Because nodes co-own their inputs via `shared_ptr`, the whole graph stays alive from output back to the leaves for the duration of the backward pass.

```
x ──▶ Linear ──▶ ReLU ──▶ Linear ──▶ Sigmoid ──▶ MSELoss ──▶ loss (scalar)
                                                               │
                        gradients flow back through the graph  ▼
        dL/dW, dL/db accumulate into each Linear's parameters via backward()
```

---

## Roadmap 🚀

- More activations (Leaky ReLU, Tanh)
- General-DAG autograd (topological backward, gradient de-duplication)
- Batch dimension, more optimizers (momentum, Adam), model save/load

---

> **⚠️ Work in progress.** This is a learning-focused project under active development. APIs may change between versions, and the feature set is intentionally growing step by step. Feedback and curiosity welcome! 🚧
