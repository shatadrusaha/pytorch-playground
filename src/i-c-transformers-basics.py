"""                     Import libraries.                       """
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Question-
    What is the difference between standard scalar and batch normalization?

Answer-
    The difference between using a StandardScaler and batch normalization is:

    StandardScaler:
        This is a preprocessing step (often from scikit-learn) that fits on the entire training set and transforms data so each feature has zero mean and unit variance. It uses the statistics (mean, std) of the whole training set, ensuring consistent scaling for all samples.

    Batch normalization:
        This means normalizing each batch (or sample) on-the-fly, often by subtracting the batch mean and dividing by the batch std. The statistics are computed per batch, so the scaling can change from batch to batch.

    Summary:
        - StandardScaler = consistent, dataset-wide normalization (recommended for tabular data).
        - Batch normalization = per-batch, can introduce noise, used in neural nets for regularization and faster convergence (not the same as sklearn's StandardScaler).
        - Batch normalization layer (nn.BatchNorm) = a learnable layer in neural nets, not the same as just normalizing a batch.

        For most ML tasks, use StandardScaler for preprocessing. Use batch normalization layers in deep nets if needed.
"""


"""                     Transforms.                       """
# Create a synthetic dataset.
n_rows = 1000
n_features = 64
n_classes = 10
data_tuple = (np.random.rand(n_rows, n_features), np.random.randint(0, n_classes, size=n_rows))

# Set device and random seed.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n")

random_seed = 18
torch.manual_seed(seed=random_seed)

# Model parameters.
batch_size = 32
learning_rate = 0.001
num_epochs = 1000
losses = []
n_print_epochs = 100

# Create custom 'Dataset' class.
class CustomDataset(Dataset):
    def __init__(self, data, transform=None, device='cpu'):
        # self.data = data
        self.features, self.labels = data
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = (self.features[idx], self.labels[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample

# Transform to tensor class.
class ToTensor:
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, sample):
        features, labels = sample
        features = torch.tensor(data=features, dtype=torch.float32, device=self.device)
        labels = torch.tensor(data=labels, dtype=torch.long, device=self.device)
        return features, labels

# Normalize class.
class Normalize:
    def __call__(self, sample):
        features, labels = sample
        features = (features - features.mean()) / features.std()
        return features, labels

# Compose the transforms.
transform = transforms.Compose(transforms=[ToTensor(device=device), Normalize()])

# Create the dataset.
dataset = CustomDataset(data=data_tuple, transform=transform, device=device)

# Create the dataloader.
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

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

# Create an instance of the model.
model = SimpleNN(in_features=n_features, out_features=n_classes, device=device)
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

    for batch in dataloader:
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
