"""                     Import libraries.                       """
import torch
import numpy as np


"""
https://cli.github.com
https://formulae.brew.sh/formula/gh#default
https://www.youtube.com/watch?v=5rTwOt9Qgik  # uv link
https://www.youtube.com/watch?v=QPCFnbonpNQ&t=1772s  # pytorch link
https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1  # pytorch link

gh auth login
gh repo create pytorch-playground --public --source=. --remote=origin --push
"""


"""                     Tensor introduction.                       """
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

# Add two tensors.
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])
tensor_c = tensor_a + tensor_b
tensor_d = torch.add(tensor_a, tensor_b)
print(f"'tensor_a' values -->\n{tensor_a}\n")
print(f"'tensor_b' values -->\n{tensor_b}\n")
print(f"'tensor_c' (regular addition) values -->\n{tensor_c}\n")
print(f"'tensor_d' (torch.add) values -->\n{tensor_d}\n")

# Subtract two tensors.
tensor_e = tensor_a - tensor_b
tensor_f = torch.sub(tensor_a, tensor_b)
print(f"'tensor_e' (regular subtraction) values -->\n{tensor_e}\n")
print(f"'tensor_f' (torch.sub) values -->\n{tensor_f}\n")

# Multiply two tensors.
tensor_g = tensor_a * tensor_b
tensor_h = torch.mul(tensor_a, tensor_b)
print(f"'tensor_g' (regular multiplication) values -->\n{tensor_g}\n")
print(f"'tensor_h' (torch.mul) values -->\n{tensor_h}\n")

# Divide two tensors.
tensor_i = tensor_a / tensor_b
tensor_j = torch.div(tensor_a, tensor_b)
print(f"'tensor_i' (regular division) values -->\n{tensor_i}\n")
print(f"'tensor_j' (torch.div) values -->\n{tensor_j}\n")