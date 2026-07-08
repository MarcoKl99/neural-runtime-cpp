import torch
import torch.nn as nn


# Define the same network as in the C++ reference
class XORNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 1, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z1 = self.layer1(x)
        a1 = self.relu(z1)
        z2 = self.layer2(a1)
        y_hat = self.sigmoid(z2)
        return y_hat


# Set the fixed reference weights (from C++ run)
def set_reference_weights(model):
    # Layer 1 weights and bias
    with torch.no_grad():
        model.layer1.weight.copy_(
            torch.tensor(
                [
                    [0.127675, -0.0800557],
                    [-0.0511347, -0.0129438],
                    [0.209888, 0.0099226],
                    [-0.0643893, 0.102362],
                ],
                dtype=torch.float64,
            )
        )
        model.layer1.bias.copy_(torch.zeros(4, dtype=torch.float64))

        # Layer 2 weights and bias
        model.layer2.weight.copy_(
            torch.tensor([[0.119094, 0.103336, 0.101908, -0.0430681]], dtype=torch.float64)
        )
        model.layer2.bias.copy_(torch.zeros(1, dtype=torch.float64))


# Training data
inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float64)

# Create the model and set the weights
model = XORNetwork()
set_reference_weights(model)

# Training setup (same like in the C++ reference)
learning_rate = 0.1
epochs = 5000
loss_fn = nn.MSELoss()

# Calculate initial loss
with torch.no_grad():
    initial_loss = loss_fn(model(inputs), targets).item()
    print(f"Average loss BEFORE update: {initial_loss:.8f}")

# Training loop
for epoch in range(epochs):
    # Zero gradients
    model.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    # Backward pass
    loss.backward()

    # Manual weight update
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    # Print loss at intervals
    if epoch % 1000 == 0:
        print(f"Loss ({epoch}/{epochs}): {loss.item():.8f}")

# Print the final value of the loss
print(f"Loss ({epochs}/{epochs}): {loss.item():.8f}")
