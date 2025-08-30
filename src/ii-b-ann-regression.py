"""                     Import libraries.                       """
from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchinfo import summary


"""                     User defined settings.                       """
RANDOM_STATE = 18
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 500
PRINT_EVERY = 50

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


"""                     Load and preprocess data.                       """
# https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

# Load the dataset.
bike_sharing = fetch_ucirepo(id=275)

"""
bike_sharing.keys()
# dict_keys(['data', 'metadata', 'variables'])

bike_sharing.data.keys()
# dict_keys(['ids', 'features', 'targets', 'original', 'headers'])

bike_sharing.data.original.head(10)
# 'casual' + 'registered' = 'cnt'
# Hence, 'casual' and 'registered' are removed in 'bike_sharing.data.features'.

# Variable information.
bike_sharing.variables
bike_sharing.variables[['name', 'type', 'description']]

# Metadata.
# print(bike_sharing.metadata)
for key, value in bike_sharing.metadata.items():
    print(f"{key}: {value}")
print(bike_sharing.metadata.additional_info.variable_info)
# Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
# 	- instant: record index
# 	- dteday : date
# 	- season : season (1:winter, 2:spring, 3:summer, 4:fall)
# 	- yr : year (0: 2011, 1:2012)
# 	- mnth : month ( 1 to 12)
# 	- hr : hour (0 to 23)
# 	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
# 	- weekday : day of the week
# 	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# 	+ weathersit : 
# 		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# 	- temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# 	- atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# 	- hum: Normalized humidity. The values are divided to 100 (max)
# 	- windspeed: Normalized wind speed. The values are divided to 67 (max)
# 	- casual: count of casual users
# 	- registered: count of registered users
# 	- cnt: count of total rental bikes including both casual and registered

"""

# Split the data into features and targets.
X = bike_sharing.data.features
y = bike_sharing.data.targets

"""
X.dtypes
X.describe()
"""

# Get the date column in datetime format.
X.loc[:, 'dteday'] = pd.to_datetime(arg=X['dteday'], format='%Y-%m-%d')

"""
df_test = X.copy()
df_test['yyyy_mm'] = df_test['dteday'].dt.to_period('M')
df_test['yyyy_mm'] = df_test['yyyy_mm'].astype(str).str.replace('-', '_')

df_test.groupby(by='yyyy_mm', as_index=False).size()
df_test.groupby(by='yyyy_mm', as_index=True).size().plot(kind='bar')

df_test.groupby(by=['yyyy_mm', 'season'], as_index=False).size()
df_test.groupby(by=['yyyy_mm', 'yr'], as_index=False).size()
df_test.groupby(by=['yyyy_mm', 'mnth'], as_index=False).size()
df_test.groupby(by=['yyyy_mm', 'weathersit'], as_index=False).size()

df_test['workingday'].value_counts().plot(kind='bar')
df_test['weathersit'].value_counts().plot(kind='bar')
df_test['weathersit'].value_counts(dropna=False)

df_test['temp'].hist(bins=30)
df_test['atemp'].hist(bins=30)
df_test['hum'].hist(bins=30)
df_test['windspeed'].hist(bins=30)

df_test['casual'].hist(bins=30)
df_test['registered'].hist(bins=30)

"""

# Map categorical variables to their string representations.
dict_season = {
    1: 'winter',
    2: 'spring',
    3: 'summer',
    4: 'fall'
}
dict_yr = {
    0: 2011,
    1: 2012
}
dict_mnth = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}

X_model = X.copy()
X_model['season'] = X_model['season'].map(dict_season)
X_model['yr'] = X_model['yr'].map(dict_yr)
X_model['mnth'] = X_model['mnth'].map(dict_mnth)
X_model.drop(columns=['dteday'], inplace=True) # Drop the original date column.

# Define categorical and numerical columns.
cols_cat = ['season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit']
cols_num = ['holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed']

# Convert categorical columns to 'category' dtype.
for col in cols_cat:
    X_model[col] = X_model[col].astype('category')

# Convert numerical columns to 'float32' dtype.
for col in cols_num:
    X_model[col] = X_model[col].astype('float32')

"""
X_model.head(10)
X_model.describe()
X_model.info()
"""

# Get dummies for categorical variables.
X_dummies = pd.get_dummies(data=X_model, columns=cols_cat, dtype=float)

# Get the final feature and target arrays.
X_final = X_dummies.to_numpy(dtype=np.float32)
y_final = y.to_numpy(dtype=np.float32)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=RANDOM_STATE)
print(f"X_train shape: {X_train.shape}\ny_train shape: {y_train.shape}\n")
print(f"X_test shape: {X_test.shape}\ny_test shape: {y_test.shape}\n")

# No scaling is required. Data has dummy and/or normalized and/or binary features.


"""                     ANN model building.                       """
# Create a PyTorch dataset and dataloader.
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(data=features, dtype=torch.float32)
        self.targets = torch.tensor(data=targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = CustomDataset(features=X_train, targets=y_train)
test_dataset = CustomDataset(features=X_test, targets=y_test)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""
print(f"'len(X_train)': {len(X_train)}\n'len(train_dataset)': {len(train_dataset)}\n'len(train_dataloader)': {len(train_dataloader)}\n")
print(f"'len(test_dataset)': {len(test_dataset)}\n'len(X_test)': {len(X_test)}\n'len(test_dataloader)': {len(test_dataloader)}\n")
"""

# Create an ANN model class.
class ANNModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.seq_model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        x = self.seq_model(x)
        return x

# Create a training loop.
def train_model(
    model, 
    train_dataloader, 
    criterion, 
    optimizer, 
    num_epochs, 
    device=torch.device('cpu'), 
    print_every=10
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
        for batch in train_dataloader:
            # Unpack the batch.
            features, targets = batch
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass.
            preds = model(features)

            # Compute loss.
            loss = criterion(preds, targets)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss for each batch
            loss_epoch.append(loss.item())

        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {np.mean(loss_epoch):.4f}")

        # Update training loss.
        loss_train.append(np.mean(loss_epoch))

    return loss_train

# Create an evaluation loop.
def evaluate_model(model, test_dataloader, criterion, device=torch.device('cpu')):
    # Move model to the specified device.
    model.to(device)

    # Set the model to evaluation mode.
    model.eval()

    # Track loss for each batch.
    loss_epoch = []

    # Use inference mode for evaluation.
    with torch.inference_mode():
        for batch in test_dataloader:
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)

            preds = model(features)
            loss = criterion(preds, targets)
            loss_epoch.append(loss.item())

    print(f"Test Loss: {np.mean(loss_epoch):.4f}\n")

# Create a model instance.
model_nn = ANNModel(in_features=X_train.shape[1], out_features=y_train.shape[1])

# View the model architecture/parameters.
print(f"Model architecture:\n{model_nn}\n")
print(f"Model summary:\n{summary(model_nn, input_size=(BATCH_SIZE, X_train.shape[1]))}\n")
print(f"Model parameters:\n{list(model_nn.parameters())}\n")

# Create a loss function and optimizer.
criterion = nn.MSELoss()
optimizer = optim.Adam(params=model_nn.parameters(), lr=LEARNING_RATE)

# Train the model.
print("Training the model...\n")
loss_train = train_model(
    model=model_nn,
    train_dataloader=train_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=DEVICE,
    print_every=PRINT_EVERY
)
print("Training complete.\n")

# Plot training loss.
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(1, NUM_EPOCHS + 1), loss_train, label='Training Loss', color='blue')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Curve')
ax.legend()
plt.show()
plt.close(fig)

# Evaluate the model.
print("Evaluating the model...\n")
evaluate_model(
    model=model_nn,
    test_dataloader=test_dataloader,
    criterion=criterion,
    device=DEVICE
)