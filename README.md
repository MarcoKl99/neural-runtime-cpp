# Neural Runtime C++ 🧠

A small Deep Learning runtime in modern C++, implemented from first principles - Tensor,
Matrixoperations, Backprop, Autograd and more! Step by step without ML dependencies in the core.

## Status 🔎

In active development.

First building block:

- Tensor-class (1D/2D, row-major, double precision)
- Initialization
- Element access
- Tests

Implemented operations:

- Elementwise addition (`operator+`, `operator+=`)
- Elementwise subtraction (`operator-`, `operator-=`)
- Hadamard product (`hadamard(...)`)
- Matrix multiplication (`matmul(...)`, rank 2 only)
- Transpose (`transpose()`, rank 2 only)
- Scalar multiplication (`operator*`, `operator*=`)

Second building block:

- `Linear` layer (affine transformation `y = W*x + b`)
- Randomly initialized weights (normal distribution, mu=0, sigma=0.1)
- `set_weights(...)` for deterministic testing / reference comparison
- Tests

Third building block:

- Activation functions (free functions, `nrt/activations.hpp`): `relu`, `relu_derivative`,
  `sigmoid`, `sigmoid_derivative`
- Tests

Fourth building block:

- `Tensor::sum()`
- MSE loss (`nrt/loss.hpp`, free function `mse(y_hat, y) -> double`)
- Tests

Fifth building block:

- `Linear::backward(...)`: gradients w.r.t. weights, bias and input (accumulating)
- `zero_grad()`, `average_grad_weights()`, `average_grad_bias()`
- Tests

Sixth building block:

- `relu_backward`, `sigmoid_backward` (chain rule: grad_output * local derivative)
- `mse_derivative`
- Tests

## Examples 🧪

- `examples/xor_forward.cpp` — forward pass only, random weights, no training.
- `examples/xor_training.cpp` — full manual training loop (forward + backward + SGD optimizer step) over all 4 XOR samples. See
  `notes/xor_training.md` for an interesting observation on the Dying ReLU problem.

More to come! 🚀
