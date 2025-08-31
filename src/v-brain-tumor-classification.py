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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
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
NUM_EPOCHS = 500
PRINT_EVERY = 50
NUM_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

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
        v2.Resize(size=(224, 224)),
        # v2.Resize(size=(4, 4)),
        v2.ToImage(), # Convert tensor, ndarray, or PIL Image to Image type.
        v2.ToDtype(dtype=torch.float32, scale=True), # Convert image to float32 and scale to [0, 1].
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats.
    ]
)

# Create datasets.
ds_train = datasets.ImageFolder(
    root=os.path.join(path, 'Training'),
    transform=tf
)
ds_test = datasets.ImageFolder(
    root=os.path.join(path, 'Testing'),
    transform=tf
)

"""
print(f"'len(ds_train)': {len(ds_train)}")
print(f"'len(ds_test)': {len(ds_test)}")
"""

# Create dataloaders.
dl_train = DataLoader(
    dataset=ds_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    # pin_memory=True # Not supported on MPS now.
)
dl_test = DataLoader(
    dataset=ds_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    # pin_memory=True # Not supported on MPS now.
)
# next(iter(dl_train))


"""                     Build the model.                       """
# Define the model.
class BrainTumorClassifier(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.model = nn.Sequential(
            # Conv layer #1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv layer #2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv layer #3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Flatten
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(in_features=128 * 28 * 28, out_features=out_features)
        )