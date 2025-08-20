"""                     Import libraries.                       """
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


"""                     Simple linear regression.                       """
# Create data points.
X = np.array(object=[x for x in range(100)], dtype=np.float32)
X = X.reshape((-1, 1))
print(f"'X' values (flattened) -->\n{X.flatten()}")
print(f"'X' shape --> {X.shape}")
print(f"'X' dtype --> {X.dtype}\n")

y = 15 + 2 * X.flatten()
print(f"'y' values (flattened) -->\n{y}")
print(f"'y' shape --> {y.shape}")
print(f"'y' dtype --> {y.dtype}\n")

# Plot the data points.
plt.scatter(X, y, label='Data points')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter plot of X vs y')
plt.legend()
plt.show()
plt.close()

# Normalize the data.
X_normalized = (X - X.mean()) / X.std()
y_normalized = (y - y.mean()) / y.std()

print(f"'X_normalized' values (flattened) -->\n{X_normalized.flatten()}\n")
print(f"'y_normalized' values (flattened) -->\n{y_normalized.flatten()}\n")

# Convert data to PyTorch tensors.
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y_normalized, dtype=torch.float32)

print(f"'X_tensor' values (flattened) -->\n{X_tensor.flatten()}")
print(f"'X_tensor' shape --> {X_tensor.shape}")
print(f"'X_tensor' dtype --> {X_tensor.dtype}\n")
print(f"'y_tensor' values (flattened) -->\n{y_tensor.flatten()}")
print(f"'y_tensor' shape --> {y_tensor.shape}")
print(f"'y_tensor' dtype --> {y_tensor.dtype}\n")

# Define the model.
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.linear(x).squeeze()

# Define model parameters.
in_features = 1
out_features = 1

# Initialize the model and other components.
model = LinearRegressionModel(in_features=in_features, out_features=out_features)
print(f"Model structure -->\n{model}\n")

criterion = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.01)

"""
weights = weights - learning_rate * gradients
"""

# Train the model.
num_epochs = 1000
for epoch in range (num_epochs):
    # Forward pass.
    y_pred = model(X_tensor)

    # Compute the loss.
    loss = criterion(y_pred, y_tensor)

    # Backward pass.
    optimizer.zero_grad()
    loss.backward()

    # Update the weights.
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the model.
x_test = 114
x_test_normalized = (x_test - X.mean()) / X.std()
x_test_tensor = torch.tensor([x_test_normalized], dtype=torch.float32)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_test_tensor = model(x_test_tensor)

y_test = y_test_tensor.item() * y.std() + y.mean()
print(f"Predicted value for x = {x_test}: {y_test:.4f}")
print(f"Actual value for x = {x_test}: {15 + 2 * x_test:.4f}\n")

# Visualize the results.
plt.scatter(X, y, label='Data points')
plt.scatter(x_test, y_test, label='Predicted point', color='green')
plt.plot(X, model(X_tensor).detach().numpy() * y.std() + y.mean(), label='Regression line', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter plot of X vs y')
plt.legend()
plt.show()
plt.close()