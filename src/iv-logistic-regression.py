"""                     Import libraries.                       """
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


"""                     User-defined variables.                       """
# DATASET_PATH = os.path.join("artifacts", "breast_cancer.csv")

# Model artifacts.
FOLDER_ARTIFACTS = 'artifacts'
MODEL_PATH = os.path.join(FOLDER_ARTIFACTS, 'logistic_regression_model.pth')

# Miscellaneous.
random_seed = 18
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: '{device}'\n")


"""                     Prepare the dataset.                       """
# Generate a synthetic dataset.
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=random_seed
)
print(f"X.shape: {X.shape}\ny.shape: {y.shape}\n")

n_samples, n_features = X.shape

# Split the dataset into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Scale the features.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"X_train.shape: {X_train.shape}\nX_test.shape: {X_test.shape}\n")

# Convert to PyTorch tensors.
X_train_tensor = torch.tensor(data=X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(data=y_train, dtype=torch.float32, device=device).view(-1, 1)
X_test_tensor = torch.tensor(data=X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(data=y_test, dtype=torch.float32, device=device).view(-1, 1)


"""                     Build the logistic regression model.                       """
# Define the logistic regression model.
class LogisticRegressionModel(nn.Module):
    def __init__(self, in_features, h1_units=16, h2_units=8, device='cpu'):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1_units)
        self.fc2 = nn.Linear(h1_units, h2_units)
        self.fc3 = nn.Linear(h2_units, 1)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Instantiate the model.
model = LogisticRegressionModel(in_features=n_features, device=device)
print(model)

# Define the loss function and optimizer.
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


"""                     Train the model.                       """
# Set training and evaluation parameters.
torch.manual_seed(seed=random_seed)
num_epochs = 10000
n = 1000
threshold = 0.5
losses = []


for epoch in range(num_epochs):
    # Set the model to training mode.
    model.train()

    # Forward pass.
    y_pred = model.forward(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.item())

    # Backward pass and optimization.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 'n' epochs.
    if (epoch + 1) % n == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Plot the loss curve.
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(1, num_epochs + 1), losses, label='Loss', color='blue')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Curve')
ax.legend()
plt.show()
plt.close(fig)

"""
Question:
    Which is better to use? `torch.no_grad()` or `torch.inference_mode()`?

Answer:
    `torch.inference_mode()` is generally better for inference than `torch.no_grad()`.
        - `torch.no_grad()` disables gradient calculation, saving memory and computation, and is commonly used during evaluation or inference.
        - `torch.inference_mode()` does everything `torch.no_grad()` does, but also further optimizes memory usage and performance by disabling version counter updates and some autograd internals.

    If you are only evaluating or making predictions (not training), prefer `torch.inference_mode()` for maximum efficiency. Use `torch.no_grad()` if you need to do inference but still want to modify tensors that require gradients. For most inference cases, use `torch.inference_mode()`.

https://docs.pytorch.org/docs/1.9.0/generated/torch.inference_mode.html?highlight=inference%20mode
"""

# Evaluate the model on the test set.
model.eval()
with torch.inference_mode():
    # Predict and calculate loss on the test set.
    y_test_pred = model.forward(X_test_tensor)
    test_loss = criterion(y_test_pred, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

    # Apply threshold to get predicted class (0 or 1)
    predicted = (y_test_pred >= threshold).float()

    # Calculate accuracy
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}\n")

"""
pd.DataFrame(y_test_pred.cpu().numpy()).describe()
pd.Series(y_test_pred.cpu().squeeze().numpy()).describe()

pd.Series((y_test_pred >= 0.5).float().cpu().squeeze().numpy()).value_counts()
"""

# Create and plot confusion matrix.
cm = confusion_matrix(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap="Blues", ax=ax)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix")
plt.show()
plt.close(fig)
