"""                     Import libraries.                       """
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


"""                     Datasets and DataLoaders.                       """
X, y = load_digits(return_X_y=True)
print(f"'X.shape': {X.shape}")
print(f"'y.shape': {y.shape}\n")

print(f"'X[0, :]' value: {X[0, :]}")
print(f"'y[0]' value: {y[0]}\n")

# Set device and random seed.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n")

random_seed = 18
torch.manual_seed(seed=random_seed)

# Split the dataset into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Scale the features.
scaler = StandardScaler()
X_train = scaler.fit_transform(X=X_train)
X_test = scaler.transform(X=X_test)
print(f"'X_train.shape': {X_train.shape}\n'X_test.shape': {X_test.shape}\n")

# Create custom 'Dataset' class.
class CustomDataset(Dataset):
    def __init__(self, features, labels, device='cpu'):
        # features  --> X
        # labels    --> y
        self.features = torch.tensor(features, dtype=torch.float32, device=device)
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create instances of the custom dataset.
train_dataset = CustomDataset(features=X_train, labels=y_train, device=device)
test_dataset = CustomDataset(features=X_test, labels=y_test, device=device)
print(f"Length of 'train_dataset': {len(train_dataset)}")
print(f"Length of 'test_dataset': {len(test_dataset)}\n")

print(f"First 'train_dataset' value: {train_dataset[0]}\n")

print(f"First 'feature' value: {train_dataset[0][0]}")
print(f"First 'label' value: {train_dataset[0][1]}\n")

# Define batch size.
batch_size = 32

# Create DataLoader instances.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create a simple feedforward neural network.
class SimpleNN(nn.Module):
    def __init__(self, in_features=64, out_features=10, device='cpu'):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=out_features)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define model parameters.
in_features = X_train.shape[1]
out_features = len(set(y_train))
learning_rate = 0.001
num_epochs = 100
losses = []
n_print_epochs = 10

# Create an instance of the model.
model = SimpleNN(in_features=in_features, out_features=out_features, device=device)
print(f"Model architecture:\n{model}\n")

# Define loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Training loop.
print("Starting training...\n")
for epoch in range(num_epochs):
    # Set model to training mode.
    model.train()

    # Batch loss tracking.
    batch_losses = []

    for batch in train_dataloader:
        # Extract features and labels from the batch dataset.
        features, labels = batch

        # Forward pass and loss computation.
        outputs = model.forward(x=features)
        loss = criterion(outputs, labels)
        batch_losses.append(loss.item())

        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute average loss for the epoch.
    epoch_loss = sum(batch_losses) / len(batch_losses)
    losses.append(epoch_loss)

    # Print loss every 'n' epochs.
    if (epoch + 1) % n_print_epochs == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
print("\nTraining complete.\n")

# Plot the loss curve.
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(1, num_epochs + 1), losses, label='Loss', color='blue')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Curve')
ax.legend()
plt.show()
plt.close(fig)

# Evaluate the model on the test set.
model.eval()
test_losses = []
# test_accuracies = []
test_pred = []
test_true = []

with torch.inference_mode():
    for batch in test_dataloader:
        features, labels = batch

        # Predict and compute loss.
        outputs = model.forward(x=features)
        loss = criterion(outputs, labels)
        test_losses.append(loss.item())

        # Compute accuracy.
        predicted = torch.argmax(input=outputs, dim=1)
        test_pred.extend(predicted.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

        # accuracy = (predicted == labels).float().mean()
        # test_accuracies.append(accuracy.item())

# Compute average test loss.
average_test_loss = sum(test_losses) / len(test_losses)
print(f"Average 'test' loss: {average_test_loss:.4f}")

# Compute average test accuracy.
"""
average_test_accuracy = sum(test_accuracies) / len(test_accuracies)
print(f"Average 'test' accuracy: {average_test_accuracy:.4f}")
"""
accuracy = accuracy_score(y_true=test_true, y_pred=test_pred) * 100
print(f"Overall 'test' accuracy: {accuracy:.4f}%\n")