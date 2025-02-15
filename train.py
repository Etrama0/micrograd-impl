import math
import random
import numpy as np
import matplotlib.pyplot as plt
from visualize import draw_dot    # Importing the draw_dot function
from nn import MLP      # Importing the model

# Define input and expected outputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# Initialize the model
n = MLP(3, [4, 4, 1])  # MLP with 3 inputs, 2 hidden layers, and 1 output

# Training loop
losses = []
for k in range(100):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))  # MSE loss

    # Zero gradients
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()     # Backward pass
    
    # Update parameters with gradient descent
    for p in n.parameters():
        p.data += -0.2 * p.grad  # Learning rate: 0.2

    # Store loss for visualization
    losses.append(loss.data)
    print(f"Epoch {k}: Loss = {loss.data}")

# Visualization (optional)
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()

dot = draw_dot(loss)  # Visualize loss computation graph
dot.render(view=False)  # Open the visualization

print(ypred) # Print final predictions