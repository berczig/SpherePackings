import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import radius_graph
import numpy as np

# Hyperparameters
batch_size = 16
d = 2  # Dimensionality of points
N = 10  # Number of points
num_epochs = 10
learning_rate = 1e-4
num_train_timesteps = 1000
data_path = "./sample_packings.pt"  # Path to training tensors

# Dataset
class SpherePackingDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)  # Assuming data is saved as a tensor (num_samples x d x N)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# DataLoader
data = SpherePackingDataset(data_path)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Graph Neural Network
def create_graph(tensor):
    edge_index = radius_graph(tensor.T, r=1.0)  # Create edges based on distance < 1.0
    return Data(x=tensor.T, edge_index=edge_index)

class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__(aggr="mean")
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.linear(x_j)

    def update(self, aggr_out):
        return aggr_out

class DiffusionModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DiffusionModel, self).__init__()
        self.gnn1 = GNN(in_channels, hidden_channels)
        self.gnn2 = GNN(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gnn1(x, edge_index)
        x = torch.relu(x)
        x = self.gnn2(x, edge_index)
        return x

# Scheduler
class CustomScheduler(DDPMScheduler):
    def add_noise(self, x, noise, timesteps):
        alpha_t = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        return alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise

# Model & Scheduler
model = DiffusionModel(d, 64, d)
scheduler = CustomScheduler(num_train_timesteps=num_train_timesteps)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Loss Function
criterion = nn.MSELoss()

# Training Loop
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    for batch in data_loader:
        batch = batch.to(device)
        noise = torch.randn_like(batch).to(device)
        timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device).long()
        noisy_data = scheduler.add_noise(batch, noise, timesteps)

        graphs = [create_graph(data) for data in noisy_data]
        batched_graphs = Data.from_data_list(graphs).to(device)
        predicted_noise = model(batched_graphs)

        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Sampling Function
def sample(model, scheduler, num_samples):
    model.eval()
    with torch.no_grad():
        samples = torch.randn((num_samples, d, N), device=device)
        for t in reversed(range(scheduler.num_train_timesteps)):
            timestep = torch.tensor([t] * num_samples, device=device).long()
            graphs = [create_graph(sample) for sample in samples]
            batched_graphs = Data.from_data_list(graphs).to(device)
            predicted_noise = model(batched_graphs)
            samples = scheduler.step(predicted_noise, timestep, samples).prev_sample
    return samples

# Generate samples
num_samples = 4
samples = sample(model, scheduler, num_samples)

# Post-process and Save
for i, sample in enumerate(samples):
    np.save(f"sample_{i}.npy", sample.cpu().numpy())
