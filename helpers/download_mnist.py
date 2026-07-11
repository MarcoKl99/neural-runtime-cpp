from torchvision.datasets import MNIST

"""
Use this to load the MNIST dataset onto your disk in the data directory.
The loaded data is used for the examples/mnist_mlp.cpp program.
"""

# Note: Execute from the project root to correctly save them in the data folder
MNIST(root="data", train=True, download=True)
MNIST(root="data", train=False, download=True)

print("Done. Raw IDX files are in data/MNIST/raw/")
