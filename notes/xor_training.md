# XOR Training - Observations 🔎

## Summary ⏩

- Manually implemented Neural Network trains successfully on the XOR problem
- Loss (MSE) decreases from 0.250 (epoch 0) to 0.001 (epoch 5000)

**C++ Output (examples/xor_training.cpp):**

```text
Loss (0/5000):      0.249599
Loss (1000/5000):   0.0926499
Loss (2000/5000):   0.00656239
Loss (3000/5000):   0.00258529
Loss (4000/5000):   0.00150208
Loss (5000/5000):   0.00102746
```

**Comparison to PyTorch (references/xor_training_pytorch.py):**

```text
Loss (0/5000):      0.24960961
Loss (1000/5000):   0.09308553
Loss (2000/5000):   0.00656716
Loss (3000/5000):   0.00258774
Loss (4000/5000):   0.00150301
Loss (5000/5000):   0.00102766
```

## Context 📖

The file `examples/xor_training.cpp` trains a small MLP (`Linear(2,4) -> ReLU -> Linear(4,1) -> Sigmoid`)
on the XOR problem using full-batch gradient descent.

The program implements the neural network, creates the full dataset consisting of all 4 possible input-output
pairs, and performs 5'000 epochs of training, tracking the MSE loss during the process.

**Strategy:**

- Pass each sample through the network
- Perform the backward pass on the layers and activations
- Update the weights and biases manually (`W -= lr * average_grad`) - No optimizer yet... we did not have these things back in the day! 👴

## Summary 📝

The training works, the loss decreases over the epochs... most of the time! The other runs, where the network does
not really train or not train at all, are a great opportunity to see the **Dying ReLU** issue!

For this, have a look at the following 2 runs. Same epochs (`5'000`), same learning rate (`0.1`), new
random initialization of the weigts and biases.

**Run 1:**

```text
Average loss BEFORE update: 0.250043
Loss (0/5000):    0.250041
Loss (1000/5000): 0.181879
Loss (2000/5000): 0.127690
Loss (3000/5000): 0.125990
Loss (4000/5000): 0.125548
```

-> The network trains as expected.

**Run 2:**

```text
Average loss BEFORE update: 0.250264
Loss (0/5000):    0.250244
Loss (1000/5000): 0.250003
Loss (2000/5000): 0.250002
Loss (3000/5000): 0.250002
Loss (4000/5000): 0.250002
```

...and nothing happens. Imagine the following:

The loss of `0.25` is not random! This value occurs, if the network predicts `y_hat = 0.5`, regardless of the
input `x`.

```text
MSE = 1/4 * [(0.5-0)^2 + (0.5-1)^2 + (0.5-1)^2 + (0.5-0)^2] = 0.25
```

Tracing this back through the network:

1. If, for the given random weights, `z1 = layer1.forward(x) <= 0` for **every** hidden neuron and **every** one
    of the 4 training inputs, then `ReLU(z1) = 0` everywhere — the hidden activation `a1` is the zero vector for every sample.
2. `z2 = W2 * a1 + b2 = W2 * 0 + b2 = b2` — constant, independent of `x`. This explains the constant `0.5` output.
3. During the backward pass, `relu_backward` multiplies the incoming gradient by `relu_derivative(z1)`, which is `0`
    everywhere `z1 <= 0` (by design — see `include/nrt/activations.hpp`).
    So `grad_z1 = 0` everywhere, and `layer1` receives **no gradient at all**.
4. Without a gradient, `layer1`'s weights never change, so `z1` stays `<= 0` forever — the hidden layer is
    permanently "dead", and the network can only ever adjust `layer2`'s bias, which is not enough to solve XOR.

This is known as the **Dying ReLU** problem: if enough (or in this case all) ReLU units are pushed into their zero-gradient
region right from initialization, and no path exists for them to receive a nonzero gradient again, they can never recover
under plain gradient descent.

Run 1 simply got a "luckier" random initialization, where at least one hidden neuron stayed positive for at least one input — enough for a
gradient signal to flow back into `layer1`, allowing it to learn.

## It's a useful Result, not a Bug❗️

This behavior is actually a good sanity check on the implementation: the fact that a fully-dead hidden layer produces an *exact* gradient
of zero in `layer1` is precisely what `relu_backward` is supposed to do (see its tests: "gradient [...] is blocked where x <= 0").
Run 2's stuck loss is the correct, expected consequence of that design applied to an unlucky initialization.

## Future mitigations (not implemented yet - state v0.1.0) 🔧

- **Weight initialization** - E.g. He initialization, which scales the initial variance based on `in_features` — designed
    specifically to reduce the chance of ReLU units starting fully in the dead zone
- **Leaky ReLU** or similar activations that keep a small, nonzero gradient for `x <= 0`, so a dead unit can still recover
- **Multiple random restarts**, picking the run with the lowest initial loss before committing to a full training run.

These strategies are left to the motivated developer of this repository (me 🤓) to implement in future versions.
The goal of this initial version v0.1 was to verify that the manual backprop through the network is
mathematically correct. Both runs demonstrate this behavior, with Run 2 showing the Dying ReLU phenomenon.
