# MNIST with a Raw MLP 🔢

## Summary ⏩

- Extended the framework with everything needed for multi-class classification: `Softmax`,
  `CrossEntropyLoss` (fused softmax + negative-log-likelihood backward), He / Xavier weight
  initialization (with an optional seed) on `Linear`, and an IDX file parser for the MNIST dataset.
- Validated the new pieces incrementally before touching real data: a toy 3-class 2D clustering
  problem (`examples/three_class.cpp`) first, then the actual MNIST MLP (`examples/mnist_mlp.cpp`).
- Trained a `784 -> 256 -> 128 -> 10` MLP directly with our own `nrt::` autograd, losses, and `SGD`
  optimizer - no ML dependencies in the core, same as the rest of the project.

## Network 🧠

```text
Linear(784, 256, He) -> ReLU -> Linear(256, 128, He) -> ReLU -> Linear(128, 10, Xavier) -> CrossEntropyLoss
```

- Input: a flattened `28x28` grayscale image as a `{784, 1}` column tensor, pixel values normalized to `[0, 1]`.
- Output: 10 raw logits (one per digit class) - `CrossEntropyLoss` applies softmax internally, so the
  model itself has no final activation.
- **Total scalar parameters: 235,146**

## Training setup ⚙️

- Mini-batch SGD (batch size 32), gradients accumulated per batch and applied with one `optimizer.step()`
  per batch
- Trained on a shuffled subset of the full 60k/10k MNIST train/test split (subset sizes are a command-line
  argument to `examples/mnist_mlp.cpp`, so re-running with a different size doesn't need a recompile).
- Weights seeded for reproducibility (`He`/`Xavier` init seed + the per-epoch shuffle RNG).
- Progress bar per epoch (batches/sec + ETA), plus predictions on a handful of fixed test samples printed
  as ASCII art before and after training, to see the model's answers on those exact images flip from
  mostly-wrong to mostly-right.

## Results 📊

The following results were obtained by training on a random subset of 1000 training- and 250 test-samples
for 10 epochs.

```text
Initial train loss: 2.40211
Initial test accuracy: 0.08

Epoch 1/10 [##############################] 32/32 (100%) 1.13729 batch/s, ETA 0s
Epoch 1/10 - train loss: 0.72951 - test accuracy: 0.80800

Epoch 2/10 [##############################] 32/32 (100%) 1.14581 batch/s, ETA 0s
Epoch 2/10 - train loss: 0.44695 - test accuracy: 0.85600

Epoch 3/10 [##############################] 32/32 (100%) 1.14752 batch/s, ETA 0s
Epoch 3/10 - train loss: 0.31221 - test accuracy: 0.89600

Epoch 4/10 [##############################] 32/32 (100%) 1.14759 batch/s, ETA 0s
Epoch 4/10 - train loss: 0.23676 - test accuracy: 0.91200

Epoch 5/10 [##############################] 32/32 (100%) 1.14613 batch/s, ETA 0s
Epoch 5/10 - train loss: 0.17425 - test accuracy: 0.91600

Epoch 6/10 [##############################] 32/32 (100%) 1.14494 batch/s, ETA 0s
Epoch 6/10 - train loss: 0.13542 - test accuracy: 0.90000

Epoch 7/10 [##############################] 32/32 (100%) 1.14451 batch/s, ETA 0s
Epoch 7/10 - train loss: 0.11163 - test accuracy: 0.91200

Epoch 8/10 [##############################] 32/32 (100%) 1.14705 batch/s, ETA 0s
Epoch 8/10 - train loss: 0.07724 - test accuracy: 0.90800

Epoch 9/10 [##############################] 32/32 (100%) 1.14427 batch/s, ETA 0s
Epoch 9/10 - train loss: 0.05909 - test accuracy: 0.92000

Epoch 10/10 [##############################] 32/32 (100%) 1.14762 batch/s, ETA 0s
Epoch 10/10 - train loss: 0.05339 - test accuracy: 0.93200

Final train loss: 0.05339
Final test accuracy: 0.93200
```

An example of the printed ASCII art samples can be seen below.

```text
Sample 1 - true label: 6, predicted: 6  (correct)

                            
                            
              ##            
             ##.            
            .##             
            ##              
           ##.              
          .##               
          ##.               
         .##                
         ##                 
        .#.                 
        ##                  
        ##                  
        ##      .###        
        #.     ######       
        ##     #.  ##.      
        ##    ..    ##      
        ###         ##      
         ###.      .#.      
          .#####.####       
            .######.        
```

## Context 📖

This is the payoff of the roadmap laid out after finishing the XOR examples: Softmax + CrossEntropyLoss
were built and unit/gradient-checked first, proven on a tiny synthetic 3-class problem second, and only
then pointed at real MNIST data - each step validated before trusting the next. Next up per the roadmap:
more activations (Leaky ReLU, Tanh), general-DAG autograd, a real batch dimension (the current per-sample
training loop is the main speed bottleneck), and additional optimizers (momentum, Adam).
