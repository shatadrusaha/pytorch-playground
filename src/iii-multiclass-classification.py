"""                     Import libraries.                       """
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


"""                     Prepare the dataset.                       """
# Load the dataset.
url_data = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
df = pd.read_csv(url_data)
df.head()

# Label the target variable
dict_target = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['target'] = df['species'].map(dict_target)

# # Rename the columns, replace dots with underscores.
# df.columns = df.columns.str.replace('.', '_')

# Split the dataset into features and target variable
X = df.drop(columns=['species', 'target'])
y = df['target']

print(f"X shape (df) --> {X.shape}\ny shape (df) --> {y.shape}\n")

# Convert the DataFrames to PyTorch tensors.
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

"""
X_tensor = torch.FloatTensor(X.values)
y_tensor = torch.LongTensor(y.values)
"""

print(f"X shape (tensor) --> {X_tensor.shape}\ny shape (tensor) --> {y_tensor.shape}\n")

# Set random seed for reproducibility.
random_seed = 18
torch.manual_seed(random_seed)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=random_seed)
print(f"X_train shape --> {X_train.shape}\ny_train shape --> {y_train.shape}\n")
print(f"X_test shape --> {X_test.shape}\ny_test shape --> {y_test.shape}\n")


"""                     Build a simple neural-net model.                       """
# Create a class for the neural network and define the architecture.
class SimpleNN(nn.Module):
    """
    Input layer (4 features of Iris flower) -->
    H1 - Hidden layer #1 (# of neurons) -->
    H2 - Hidden layer #2 (# of neurons) -->
    Output layer (3 classes of Iris flowers)
    """
    def __init__(self, in_features=4, h1_neurons=6, h2_neurons=8, out_features=3):
        # Call the parent class (nn.Module) constructor.
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1_neurons)
        self.fc2 = nn.Linear(h1_neurons, h2_neurons)
        self.out = nn.Linear(h2_neurons, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

"""
Question:
    What is the difference between `super(SimpleNN, self).__init__()` and `super().__init__()`?

Answer:
    The only difference is in how the parent class is initialized:

    `super(SimpleNN, self).__init__()` is the old-style way, explicit in Python 2 and early Python 3.
    `super().__init__()` is the modern, preferred way in Python 3+.

    Both work the same in this context. The second version (`super().__init__()`) is cleaner and recommended for new Python code. The rest of the class is identical.
"""

# Instantiate the model.
model = SimpleNN()

# Define the loss function.
criterion = nn.CrossEntropyLoss()

# Define the optimizer.
optimizer = torch.optim.Adam(
    params=model.parameters(), 
    lr=0.005
)

"""
print(f"'model' -->\n{model}\n")
print(f"'model.parameters' -->\n{model.parameters}\n")
print(f"Model parameters -->\n{list(model.parameters())}\n")
"""

"""                     Train the model.                       """
# Number of epochs.
num_epochs = 500
n = 50
losses = []

for epoch in range(num_epochs):
    # Set the model to training mode.
    model.train()

    # Forward pass.
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
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

# Evaluate the model on the test set.
model.eval()
with torch.no_grad():
    # Predict and calculate loss on the test set.
    y_test_pred = model.forward(X_test)
    test_loss = criterion(y_test_pred, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

    # Find the max across columns (classes) for each row (sample).
    _, predicted = torch.max(y_test_pred, 1)

    # Calculate accuracy.
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}\n")

# Make predictions on new data.
new_data = torch.tensor([[5.1, 3.5, 1.4, 0.2]], dtype=torch.float32)
model.eval()
with torch.no_grad():
    y_new_pred = model.forward(new_data)
    _, predicted = torch.max(y_new_pred, 1)
    print(f"Predicted class for new data: {predicted.item()}\n")

# Parameters to save the model.
name_model = "multi_class_nn.pth"
folder_artifacts = "artifacts"
os.makedirs(folder_artifacts, exist_ok=True)

# Save the model.
torch.save(obj=model.state_dict(), f=os.path.join(folder_artifacts, name_model))
print(f"Model saved to: '{os.path.join(folder_artifacts, name_model)}'\n")

# Load the model.
loaded_model = SimpleNN()
loaded_model.load_state_dict(torch.load(os.path.join(folder_artifacts, name_model)))
print(f"Model loaded successfully from: '{os.path.join(folder_artifacts, name_model)}'\n")

# Make predictions with the loaded model.
loaded_model.eval()
with torch.no_grad():
    y_new_pred = loaded_model.forward(new_data)
    _, predicted = torch.max(y_new_pred, 1)
    print(f"Predicted class for new data: {predicted.item()}")
