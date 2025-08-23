"""                     Import libraries.                       """
import numpy as np
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


"""
https://cli.github.com
https://formulae.brew.sh/formula/gh#default
https://www.youtube.com/watch?v=5rTwOt9Qgik  # uv link
https://www.youtube.com/watch?v=QPCFnbonpNQ&t=1772s  # pytorch link
https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1  # pytorch link

https://docs.pytorch.org/docs/stable/tensor_attributes.html  # PyTorch tensor attributes/data-types

gh auth login
gh repo create pytorch-playground --public --source=. --remote=origin --push
"""


"""                     Tensor basics.                       """
# Lists.
random_list = [1, 2, 3, 4, 5]
print(f"'random_list' values --> {random_list}\n")

# Numpy arrays.
random_np_array = np.random.rand(3, 4)
print(f"'random_np_array' values -->\n{random_np_array}")
print(f"'random_np_array' shape --> {random_np_array.shape}")
print(f"'random_np_array' dtype --> {random_np_array.dtype}\n")

# PyTorch tensors.
random_tensor = torch.randn(3, 4)
print(f"'random_tensor' values -->\n{random_tensor}")
print(f"'random_tensor' shape --> {random_tensor.shape}")
print(f"'random_tensor' dtype --> {random_tensor.dtype}\n")

random_tensor_3d = torch.randn(2, 3, 4)
print(f"'random_tensor_3d' values -->\n{random_tensor_3d}")
print(f"'random_tensor_3d' shape --> {random_tensor_3d.shape}")
print(f"'random_tensor_3d' dtype --> {random_tensor_3d.dtype}\n")

# PyTorch tensor from numpy array.
random_tensor_from_np = torch.tensor(random_np_array)
print(f"'random_tensor_from_np' values -->\n{random_tensor_from_np}")
print(f"'random_tensor_from_np' shape --> {random_tensor_from_np.shape}")
print(f"'random_tensor_from_np' dtype --> {random_tensor_from_np.dtype}\n")


"""                     Tensor operations.                       """
# Sample tensor.
sample_tensor = torch.arange(start=0, end=10, step=1)
print(f"'sample_tensor' values -->\n{sample_tensor}")
print(f"'sample_tensor' shape --> {sample_tensor.shape}")
print(f"'sample_tensor' dtype --> {sample_tensor.dtype}\n")

"""
# Difference between torch.reshape and torch.view
https://discuss.pytorch.org/t/whats-the-difference-between-torch-reshape-vs-torch-view/159172
https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
"""
# Reshape tensor.
reshaped_tensor = sample_tensor.reshape(2, 5)
print(f"'reshaped_tensor' values -->\n{reshaped_tensor}")
print(f"'reshaped_tensor' shape --> {reshaped_tensor.shape}\n")

reshaped_tensor = sample_tensor.reshape(-1, 5)
print(f"'reshaped_tensor' values (using -1) -->\n{reshaped_tensor}")
print(f"'reshaped_tensor' shape --> {reshaped_tensor.shape}\n")

reshaped_tensor = sample_tensor.reshape(5, -1)
print(f"'reshaped_tensor' values (using -1) -->\n{reshaped_tensor}")
print(f"'reshaped_tensor' shape --> {reshaped_tensor.shape}\n")

# View tensor.
viewed_tensor = sample_tensor.view(2, 5)
print(f"'viewed_tensor' values -->\n{viewed_tensor}")
print(f"'viewed_tensor' shape --> {viewed_tensor.shape}\n")

viewed_tensor = sample_tensor.view(-1, 5)
print(f"'viewed_tensor' values (using -1) -->\n{viewed_tensor}")
print(f"'viewed_tensor' shape --> {viewed_tensor.shape}\n")

viewed_tensor = sample_tensor.view(5, -1)
print(f"'viewed_tensor' values (using -1) -->\n{viewed_tensor}")
print(f"'viewed_tensor' shape --> {viewed_tensor.shape}\n")

# Effect of updating original tensor on reshaped and viewed tensor.
sample_tensor = torch.arange(start=0, end=10, step=1)
reshaped_tensor = sample_tensor.reshape(2, 5)
viewed_tensor = sample_tensor.view(2, 5)
print(f"'sample_tensor' values -->\n{sample_tensor}")
print(f"'reshaped_tensor' values -->\n{reshaped_tensor}\n")
print(f"'viewed_tensor' values -->\n{viewed_tensor}\n")

sample_tensor[0] = 100
print(f"'sample_tensor' values (after modification) -->\n{sample_tensor}")
print(f"'reshaped_tensor' values (after modification) -->\n{reshaped_tensor}")
print(f"'viewed_tensor' values (after modification) -->\n{viewed_tensor}")

# Slice a tensor.
tensor_2d = sample_tensor.reshape(2, 5)
print(f"'tensor_2d' values -->\n{tensor_2d}\n")
print(f"'tensor_2d[:, 1]' sliced -->\n{tensor_2d[:, 1]}\n")
print(f"'tensor_2d[:, 1:2]' sliced -->\n{tensor_2d[:, 1:2]}\n")
print(f"'tensor_2d[:, 1:]' sliced -->\n{tensor_2d[:, 1:]}\n")


"""                     Tensor math operations.                       """
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])
print(f"'tensor_a' values -->\n{tensor_a}\n")
print(f"'tensor_b' values -->\n{tensor_b}\n")

# Add two tensors.
print(f"'tensor_a + tensor_b' (regular addition) values -->\n{tensor_a + tensor_b}\n")
print(f"'torch.add(tensor_a, tensor_b)' values -->\n{torch.add(tensor_a, tensor_b)}\n")
print(f"'tensor_a.add(tensor_b)' values -->\n{tensor_a.add(tensor_b)}\n")

# Subtract two tensors.
print(f"'tensor_a - tensor_b' (regular subtraction) values -->\n{tensor_a - tensor_b}\n")
print(f"'torch.sub(tensor_a, tensor_b)' values -->\n{torch.sub(tensor_a, tensor_b)}\n")
print(f"'tensor_a.sub(tensor_b)' values -->\n{tensor_a.sub(tensor_b)}\n")

# Multiply two tensors.
print(f"'tensor_a * tensor_b' (regular multiplication) values -->\n{tensor_a * tensor_b}\n")
print(f"'torch.mul(tensor_a, tensor_b)' values -->\n{torch.mul(tensor_a, tensor_b)}\n")
print(f"'tensor_a.mul(tensor_b)' values -->\n{tensor_a.mul(tensor_b)}\n")

# Divide two tensors.
print(f"'tensor_a / tensor_b' (regular division) values -->\n{tensor_a / tensor_b}\n")
print(f"'torch.div(tensor_a, tensor_b)' values -->\n{torch.div(tensor_a, tensor_b)}\n")
print(f"'tensor_a.div(tensor_b)' values -->\n{tensor_a.div(tensor_b)}\n")

# Remainder of division between two tensors.
print(f"'tensor_b % tensor_a' (regular remainder) values -->\n{tensor_b % tensor_a}\n")
print(f"'torch.remainder(tensor_b, tensor_a)' values -->\n{torch.remainder(tensor_b, tensor_a)}\n")

# Exponential of a tensor.
print(f"'tensor_a ** 2' (regular exponentiation) values -->\n{tensor_a ** 2}\n")
print(f"'torch.pow(tensor_a, 2)' values -->\n{torch.pow(tensor_a, 2)}\n")

print(f"'tensor_a ** tensor_b' (regular exponentiation) values -->\n{tensor_a ** tensor_b}\n")
print(f"'torch.pow(tensor_a, tensor_b)' values -->\n{torch.pow(tensor_a, tensor_b)}\n")

# Reassignment of tensors.
print(f"'tensor_a' values (before reassignment) -->\n{tensor_a}\n")
print(f"'tensor_b' values (before reassignment) -->\n{tensor_b}\n")

tensor_add = tensor_a.add(tensor_b)
print(f"'tensor_add' values -->\n{tensor_add}\n")

tensor_a.add_(tensor_b)  # In-place addition
print(f"'tensor_a' values (after reassignment) -->\n{tensor_a}\n")


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
    def __init__(self, features, labels):
        # features  --> X
        # labels    --> y
        self.features = torch.tensor(features, dtype=torch.float32, device=device)
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create instances of the custom dataset.
train_dataset = CustomDataset(features=X_train, labels=y_train)
test_dataset = CustomDataset(features=X_test, labels=y_test)
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