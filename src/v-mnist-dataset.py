"""                     Import libraries.                       """
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchinfo import summary

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
https://www.youtube.com/watch?v=6EJaHBJhwDs&t=1s
https://www.youtube.com/watch?v=2w0pRriQG3A

https://www.youtube.com/watch?v=CAgWNxlmYsc

https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
"""

BATCH_SIZE = 64
random_seed = 18

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(seed=random_seed)

ds_train = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

ds_test = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


dl_train = DataLoader(
    dataset=ds_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # pin_memory=True # 'mps' not supported yet.
)

dl_test = DataLoader(
    dataset=ds_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    # pin_memory=True # 'mps' not supported yet.
)

n_images = 1

for i, (image, label) in enumerate(dl_train):
    if i >= n_images:
        break
    print(image.shape, label.shape)
    # print(f"Image tensor:\n{image}\nLabel tensor:\n{label}\nDtype: {image.dtype}\n")
    plt.subplot(1, n_images, i + 1)
    plt.imshow(image[1].squeeze(), cmap='gray')
    plt.title(f"Label: {label[0].item()}")
    plt.axis('off')
plt.show()