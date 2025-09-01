"""                     Import libraries.                       """
import kagglehub

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error, explained_variance_score

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchinfo import summary


"""
https://www.youtube.com/watch?v=mGJ6OZmR-Xk&t=201s
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

https://docs.pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
"""

"""                     User defined settings.                       """
RANDOM_STATE = 18
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PRINT_EVERY = 10

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


"""                     Load and preprocess data.                       """
# Download latest version of dataset from Kaggle.
path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
print(f"Path to dataset files: {path}\n")

"""
https://docs.pytorch.org/vision/stable/transforms.html#conversion

https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToImage.html#torchvision.transforms.v2.ToImage
https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html#torchvision.transforms.v2.ToDtype
https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Normalize.html#torchvision.transforms.v2.Normalize

https://docs.pytorch.org/vision/stable/generated/torchvision.tv_tensors.Image.html#torchvision.tv_tensors.Image
"""

# Define image transformations.
tf = v2.Compose(
    transforms=[
        v2.Resize(size=(256, 256)),
        # v2.Resize(size=(4, 4)),
        v2.ToImage(),  # Convert tensor, ndarray, or PIL Image to Image type.
        v2.ToDtype(
            dtype=torch.float32, scale=True
        ),  # Convert image to float32 and scale to [0, 1].
        v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet stats.
    ]
)

# Create datasets.
ds_train = datasets.ImageFolder(root=os.path.join(path, "Training"), transform=tf)
ds_test = datasets.ImageFolder(root=os.path.join(path, "Testing"), transform=tf)

"""
print(f"'len(ds_train)': {len(ds_train)}")
print(f"'len(ds_test)': {len(ds_test)}")

# Get the first image and label from the dataset
img, _ = ds_train[0]
print(img.shape)  # For torch.Tensor: (C, H, W)
# torch.Size([3, 256, 256])
"""

# Create dataloaders.
dl_train = DataLoader(
    dataset=ds_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # pin_memory=True # Not supported on MPS now.
)
dl_test = DataLoader(
    dataset=ds_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    # pin_memory=True # Not supported on MPS now.
)

"""
test_image, test_label = next(iter(dl_test))

print(f"Test image shape: {test_image.shape}")
print(f"Test label shape: {test_label.shape}")

"""


"""                     Build the model.                       """

"""
https://www.youtube.com/watch?v=pDdP0TFzsoQ

Given your input image shape of [3, 256, 256] and your model's convolution and pooling layers, let's calculate the output size step by step:

Layer-by-layer output shape calculation
- Input:
`[batch, 3, 256, 256]`

- Conv1: `nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)`
    Output:
        Channels: 32
        Height/Width: (256 - 3 + 1*0)/1 + 1 = 254
        So: `[batch, 32, 254, 254]`
- MaxPool1: `nn.MaxPool2d(2, 2)`
    Output:
        Channels: 32
        Height/Width: 254 // 2 = 127
        So: `[batch, 32, 127, 127]`

- Conv2: `nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)`
    Output:
        Channels: 64
        Height/Width: (127 - 3 + 1*0)/1 + 1 = 125
        So: `[batch, 64, 125, 125]`
- MaxPool2: `nn.MaxPool2d(2, 2)`
    Output:
        Channels: 64
        Height/Width: 125 // 2 = 62
        So: `[batch, 64, 62, 62]`
- Conv3: `nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)`
    Output:
        Channels: 128
        Height/Width: (62 - 3 + 1*0)/1 + 1 = 60
        So: `[batch, 128, 60, 60]`
- MaxPool3: `nn.MaxPool2d(2, 2)`
    Output:
        Channels: 128
        Height/Width: 60 // 2 = 30
        So: `[batch, 128, 30, 30]`

- Flatten
    Output:
        [batch, 128 * 30 * 30] = [batch, 115200]

So, the correct in_features for the first fully connected layer is:
    `in_features = 128 * 30 * 30  # = 115200`

Summary:
    - Your current code uses 128 * 28 * 28, which is incorrect for your architecture and input size.
    - You should use 128 * 30 * 30 for in_features in the first linear layer.
"""


# Define the model.
class BrainTumorClassifier(nn.Module):
    def __init__(self, out_features, img_shape):
        super().__init__()
        self.out_features = out_features
        self.img_shape = img_shape

        # Convolutional feature extractor.
        self.conv_features = nn.Sequential(
            # Conv layer #1
            nn.Conv2d(
                in_channels=self.img_shape[0],
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv layer #2
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv layer #3
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dynamically compute the flattened size.
        # The dummy tensor simulates a single input image and passes it through the `conv_features` extractor to determine the correct flattened size.
        with torch.no_grad():
            dummy = torch.zeros(1, *self.img_shape)
            n_flatten = self.conv_features(dummy).view(1, -1).shape[1]

        # Classifier, consisting of fully connected (FC) layers.
        self.classifier = nn.Sequential(
            # Flatten
            nn.Flatten(),
            # FC layer #1
            nn.Linear(in_features=n_flatten, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # FC layer #2
            nn.Linear(in_features=256, out_features=self.out_features),
        )

    def forward(self, x):
        x = self.conv_features(x)
        x = self.classifier(x)
        return x

# Create a training loop.
def train_model(
    model,
    dataloader_train,
    criterion,
    optimizer,
    num_epochs,
    device=torch.device("cpu"),
    print_every=10,
):
    # Move model to the specified device.
    model.to(device)

    # Set the model to training mode.
    model.train()

    # Track training loss.
    loss_train = []

    # Run training epochs.
    for epoch in range(num_epochs):
        # Track loss for each epoch
        loss_epoch = []

        # Iterate over batches.
        for features, labels in dataloader_train:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass.
            preds = model(features)

            # Compute loss.
            loss = criterion(preds, labels)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss for each batch
            loss_epoch.append(loss.item())

        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {np.mean(loss_epoch):.4f}")

        # Update training loss.
        loss_train.append(np.mean(loss_epoch))

    return loss_train

# Create an evaluation loop.
def evaluate_model(model, dataloader_test, criterion, device=torch.device("cpu")):
    # Move model to the specified device.
    model.to(device)

    # Set the model to evaluation mode.
    model.eval()

    # Track loss, true and predicted values for each batch.
    loss_epoch = []
    test_true = []
    test_pred = []

    # Use inference mode for evaluation.
    with torch.inference_mode():
        for features, labels in dataloader_test:
            features = features.to(device)
            labels = labels.to(device)

            # Predict and compute loss.
            preds = model(features)
            loss = criterion(preds, labels)
            loss_epoch.append(loss.item())

            # Store true and predicted values.
            predicted = torch.argmax(input=preds, dim=1)
            test_true.extend(labels.cpu().numpy())
            test_pred.extend(predicted.cpu().numpy())

    # Get the results in a df.
    df_results = pd.DataFrame(data={"y_true": test_true, "y_pred": test_pred})
    df_results["pred_flag"] = (df_results["y_true"] == df_results["y_pred"]).astype(int)

    # Calculate and print metrics for test dataset.
    accuracy = (df_results["pred_flag"].sum() / len(df_results)) * 100

    print("Test set evaluation metrics:")
    print(f"\tTest Loss: {np.mean(loss_epoch):.4f}")
    print(f"\tAccuracy: {accuracy:.2f}%")

    return df_results

# Define model parameters.
out_features = len(ds_train.classes)
img_shape = ds_train[0][0].shape  # (C, H, W)

# Create the model instance.
model_cnn = BrainTumorClassifier(out_features=out_features, img_shape=img_shape)

# View the model architecture/parameters.
print(f"Model architecture:\n{model_cnn}\n")
print(
    f"Model summary:\n{summary(model=model_cnn, input_size=(BATCH_SIZE, *img_shape))}\n"
)
# print(f"Model parameters:\n{list(model_cnn.parameters())}\n")

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=LEARNING_RATE)

# Train the model.
print("Training the model...\n")
loss_train = train_model(
    model=model_cnn,
    dataloader_train=dl_train,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=DEVICE,
    print_every=PRINT_EVERY,
)
print("Training complete.\n")

# Plot training loss.
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(1, NUM_EPOCHS + 1), loss_train, label="Training Loss", color="blue")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Loss Curve")
ax.legend()
plt.show()
plt.close(fig)

# Evaluate the model.
print("Evaluating the model...")
df_results = evaluate_model(
    model=model_cnn, dataloader_test=dl_test, criterion=criterion, device=DEVICE
)
print("Evaluation complete.")
