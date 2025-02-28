
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
from SP.data_generation import get_data_loader, split_data_loader


# PointNet++ Components
class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels):
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
    def __init__(self, d):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstraction(d, [64, 128])
        self.sa2 = PointNetSetAbstraction(128, [256, 512])
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


def distance_penalty(output, radius):
    # output: (batch_size, d, N)
    distances = torch.cdist(output.permute(0, 2, 1), output.permute(0, 2, 1))  # (batch_size, N, N)
    mask = (distances < 2*radius) & (distances > 0)  # Ignore self-distances
    penalty = torch.sum(mask * (2*radius - distances) ** 2)
    return penalty / output.shape[0]  # Normalize by batch size

# Training Loop
def train_diffusion_model(train_data_loader, num_epochs, learning_rate, num_train_timesteps, dimension, batch_size, sphere_radius):
    # Model
    model = PointNetPlusPlus(dimension)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss Function
    criterion = nn.MSELoss()

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        for batch in train_data_loader:
            batch = batch.to(device)  # (batch_size, d, N)
            noise = torch.randn_like(batch).to(device)
            timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device).long()
            noisy_data = scheduler.add_noise(batch, noise, timesteps)

            predicted_noise = model(noisy_data)  # (batch_size, d, N)

            # Compute losses
            mse_loss = criterion(predicted_noise, noise)
            penalty_loss = distance_penalty(predicted_noise, sphere_radius)
            loss = mse_loss + penalty_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, Penalty Loss: {penalty_loss.item():.4f}")
    return model

# Sampling Function: Generate samples from the model and return (input, output) pairs

def sample_diffusion_model(model, data_loader, num_train_timesteps):
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs_batch = []
    samples_batch = []
    with torch.no_grad():
        # Form samples by adding small gaussian noise to input samples
        for packings in data_loader:
            samples = packings + 0.1 * torch.randn_like(packings)
            samples = samples.to(device)
            input_sample = samples.clone()
            inputs_batch.append(np.array(input_sample.cpu()))
            samples_history = [np.array(samples.cpu())]
            for t in reversed(range(scheduler.config.num_train_timesteps)):
                predicted_noise = model(samples)
                samples = scheduler.step(predicted_noise, t, samples).prev_sample  # Use scalar for `timestep[0]`
                samples_history.append(np.array(samples.cpu()))
            samples_history = np.transpose(np.array(samples_history), (1,0,2,3)) # to make the first axis the batch not the history
            samples_batch.append(samples_history)
    return (np.concat(inputs_batch, 0), 
    np.concat(samples_batch, 0))

# Example usage
if __name__ == "__main__":
    num_samples = 4
    # Start with random packings
    packings = torch.rand(num_samples, d, N)
    print(packings.shape)
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
