
# A simple example of a diffusion model is the PointNet++ model, which is a neural network architecture 
# for processing point clouds. The model consists of a series of set abstraction layers that aggregate 
# information from local neighborhoods of points. The model is trained to predict the noise added
# to the input point cloud at each timestep of a diffusion process. 
# The diffusion process is simulated by adding noise to the input point cloud and passing it through 
# the model to predict the noise at the next timestep. 
# The model is trained to minimize the difference between the predicted noise and the ground truth 
# noise added to the input point cloud.


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 1
sample_size = 64
d = 2  # Dimensionality of points
N = 10  # Number of points per sample
dataset_path = './SP/sample_packings.pt'  # Path to training tensors
num_epochs = 10
learning_rate = 1e-4
num_train_timesteps = 100

# Dataset
class SpherePackingDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)  # Assuming .pt file with tensor of shape (num_samples, d, N)
        # print the shape and type of the dataset
        print(self.data.shape, self.data.dtype)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# DataLoader
data = SpherePackingDataset(dataset_path)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# PointNet++ Components
class PointNetSetAbstraction(nn.Module):
    def __init__(self, num_points, in_channels, out_channels):
        super(PointNetSetAbstraction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[0], 1),
            nn.ReLU(),
            nn.Conv1d(out_channels[0], out_channels[1], 1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, in_channels, num_points)
        x = x.float()  # Ensure input is float32
        return self.mlp(x)

class PointNetPlusPlus(nn.Module):
    def __init__(self, d, N):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstraction(N, d, [64, 128])
        self.sa2 = PointNetSetAbstraction(N, 128, [256, 512])
        self.mlp = nn.Sequential(
            nn.Conv1d(512, d, 1),
        )

    def forward(self, x):
        # x: (batch_size, d, N)
        x = x.float()  # Ensure input is float32
        x = self.sa1(x)  # (batch_size, 128, N)
        x = self.sa2(x)  # (batch_size, 512, N)
        x = self.mlp(x)  # (batch_size, d, N)
        return x

# Model
model = PointNetPlusPlus(d, N)
scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Loss Function
criterion = nn.MSELoss()

def distance_penalty(output):
    # output: (batch_size, d, N)
    distances = torch.cdist(output.permute(0, 2, 1), output.permute(0, 2, 1))  # (batch_size, N, N)
    mask = (distances < 2) & (distances > 0)  # Ignore self-distances
    penalty = torch.sum(mask * (2 - distances) ** 2)
    return penalty / output.shape[0]  # Normalize by batch size

# Training Loop
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    for batch in data_loader:
        batch = batch.to(device)  # (batch_size, d, N)
        noise = torch.randn_like(batch).to(device)
        timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device).long()
        noisy_data = scheduler.add_noise(batch, noise, timesteps)

        predicted_noise = model(noisy_data)  # (batch_size, d, N)

        # Compute losses
        mse_loss = criterion(predicted_noise, noise)
        penalty_loss = distance_penalty(predicted_noise)
        loss = mse_loss + penalty_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, Penalty Loss: {penalty_loss.item():.4f}")

# Sampling Function: Generate samples from the model and return (input, output) pairs

import torch

def sample(model, scheduler, num_samples, packings):
    model.eval()
    with torch.no_grad():
        # Form samples by adding small gaussian noise to input samples
        samples = packings + 0.1 * torch.randn_like(packings)
        samples = samples.to(device)
        input_sample = samples.clone()
        for t in reversed(range(scheduler.num_train_timesteps)):
            timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)  # Ensure batch consistency
            predicted_noise = model(samples)
            samples = scheduler.step(predicted_noise, timestep[0], samples).prev_sample  # Use scalar for `timestep[0]`
    return (input_sample.cpu(), samples.cpu())

# Example usage
num_samples = 4
# Start with random packings
packings = torch.rand(num_samples, d, N)
input,output = sample(model, scheduler, num_samples, packings)
# Print type and shape of input_output_pairs
print(input.shape, output.shape)

# Post-process and Save
#for i, (input_sample, output_sample) in enumerate(input_output_pairs):
#    np.save(f"input_sample_{i}.npy", input_sample.cpu().numpy())
#    np.save(f"output_sample_{i}.npy", output_sample.cpu().numpy())


# Plot generated pairs: input points in black output points in red in separate plots for each pair

def plot_sample(input, output):
    for i in range(input.shape[0]):
        plt.scatter(input[i][0], input[i][1], c="black")
        plt.scatter(output[i][0], output[i][1], c="red")
        plt.title(f"Sample {i}")
        plt.show()

plot_sample(input, output)
