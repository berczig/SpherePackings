
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
from torch.utils.data import DataLoader, random_split
from diffusers import DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import configparser
from spheres_in_Rd.data_generation import SpherePackingDataset
from spheres_in_Rd import cfg2

# --- PointNet++ Components ---
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
        x = x.float()
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
        x = x.float()
        x = self.sa1(x)
        x = self.sa2(x)
        x = self.mlp(x)
        return x

# --- SetTransformer and dependencies ---
class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value):
        B, Nq, D = query.shape
        Nk = key.shape[1]
        q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(key).view(B, Nk, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim).transpose(1,2)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(B, Nq, D)
        out = self.out_proj(out)
        return out, attn

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_heads, num_inds):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inds, dim_hidden))
        self.mab1 = MultiheadAttention(dim_hidden, num_heads)
        self.mab2 = MultiheadAttention(dim_hidden, num_heads)
        self.fc0 = nn.Linear(dim_in, dim_hidden)
    def forward(self, X):
        B, N, _ = X.shape
        H = self.fc0(X)
        inducing = self.inducing.expand(B, -1, -1)
        H1, _ = self.mab1(inducing, H, H)
        H2, _ = self.mab2(H, H1, H1)
        return H2

class SetTransformer(nn.Module):
    def __init__(self, dim_in, dim_hidden=128, num_heads=4, num_inds=16, num_isab=2, dim_out=None):
        super().__init__()
        self.isabs = nn.ModuleList([
            ISAB(dim_in if i==0 else dim_hidden, dim_hidden, num_heads, num_inds)
            for i in range(num_isab)
        ])
        self.fc_out = nn.Linear(dim_hidden, dim_out if dim_out is not None else dim_in)
    def forward(self, X):
        # X: (batch, d, N)
        X = X.permute(0, 2, 1)  # (batch, N, d)
        for isab in self.isabs:
            X = isab(X)
        X = self.fc_out(X)      # (batch, N, d_out)
        X = X.permute(0, 2, 1)  # (batch, d_out, N)
        return X
    
def distance_penalty(output, radius):
    # output: (batch, d, N)
    B, d, N = output.shape
    coords = output.permute(0, 2, 1)  # (B, N, d)
    distances = torch.cdist(coords, coords)  # (B, N, N)
    violation = torch.relu(2*radius - distances)      # (B,N,N)
    num_pairs = B * N * (N - 1) / 2
    penalty = (violation**2).sum() / num_pairs
    return penalty / num_pairs

def train_diffusion_model(
    model,
    train_data_loader,
    num_epochs,
    learning_rate,
    num_train_timesteps,
    sphere_radius,
    beta_start,
    beta_end,
    clip_sample,
    clip_sample_range,
    device
):
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=clip_sample,
        clip_sample_range=clip_sample_range
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train().to(device)

    loss_history = []
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        for batch in train_data_loader:
            # x0: clean point set
            batch = batch.to(device)                   # (B, d, N)
            noise = torch.randn_like(batch)            # (B, d, N)
            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (batch.shape[0],),
                device=device
            ).long()                                   # (B,)

            # Add noise according to schedule
            noisy = scheduler.add_noise(batch, noise, timesteps)

            # Predict the added noise
            predicted_noise = model(noisy)

            # Manually reconstruct x0_pred from noisy + predicted_noise
            # ᾱ_t = cumulative product of alphas at each timestep
            alphas_cumprod = scheduler.alphas_cumprod.to(device)           # (T,)
            a_bar_t = alphas_cumprod[timesteps].view(-1, 1, 1)             # (B,1,1)
            sqrt_a_bar_t = torch.sqrt(a_bar_t)                             # (B,1,1)
            sqrt_1m_a_bar_t = torch.sqrt(1 - a_bar_t)                      # (B,1,1)
            x0_pred = (noisy - sqrt_1m_a_bar_t * predicted_noise) / sqrt_a_bar_t
            x0_clamped = x0_pred.clamp(min=0, max=clip_sample_range)  # Clamping to avoid out of bounds

            # Apply geometric penalty to reconstructed positions
            penalty_loss = distance_penalty(x0_clamped, sphere_radius)

            # Standard MSE between predicted and true noise
            mse_loss = criterion(predicted_noise, noise)
            x0_true = batch
            recon_loss = nn.MSELoss()(x0_pred, x0_true)

            l_max = 500
            l = l_max * (epoch/num_epochs)  # Exponential decay
            loss = mse_loss + 0.0*recon_loss + l*penalty_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        # Print loss every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    print("Training complete.")
            
    return model, loss_history

def sample_diffusion_model(
    model,
    data_loader,
    num_train_timesteps,
    num_inference_timesteps,
    beta_start,
    beta_end,
    clip_sample,
    clip_sample_range,
    device,
    num_new_from_one
):
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=clip_sample,
        clip_sample_range=clip_sample_range
    )
    scheduler.set_timesteps(num_inference_timesteps)
    model.eval()

    inputs_batch = []
    noised_batch = []
    samples_batch = []

    with torch.no_grad():
        for packings in data_loader:
            packings = packings.to(device)  # Original input sample (B, d, N)

            for _ in range(num_new_from_one):
                # Add noise to the input packings
                noise = torch.randn_like(packings)  # (B, d, N)
                timesteps = torch.randint(
                    0, num_train_timesteps, (packings.shape[0],), device=device
                ).long()  # (B,)
                noised = scheduler.add_noise(packings, noise, timesteps)

                inputs_batch.append(np.array(packings.cpu()))
                noised_batch.append(np.array(noised.cpu()))

                # Run the diffusion process
                samples = noised.clone()
                samples_history = [np.array(samples.cpu())]
                for t in scheduler.timesteps:
                    predicted_noise = model(samples)
                    samples = scheduler.step(predicted_noise, t, samples).prev_sample
                    samples_history.append(np.array(samples.cpu()))

                # Collect the full denoising trajectory
                samples_history = np.transpose(np.array(samples_history), (1, 0, 2, 3))
                samples_batch.append(samples_history)

    return (
        np.concatenate(inputs_batch, 0),    # Original inputs
        np.concatenate(noised_batch, 0),   # Noised inputs
        np.concatenate(samples_batch, 0)   # Full denoising trajectories
    )

import matplotlib.animation as animation

def animate_sample(input, output, sample_idx=0):
    num_steps = output.shape[1]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    scat = ax.scatter([], [], c="red", label="Denoised")
    ax.scatter(input[sample_idx][0], input[sample_idx][1], c="black", label="Input", alpha=0.3)
    ax.legend()

    def update(frame):
        ax.set_title(f"Sample {sample_idx}, Step {frame+1}/{num_steps}")
        data = output[sample_idx][frame]  # (d, N)
        scat.set_offsets(np.column_stack([data[0], data[1]]))  # (N, d)
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, interval=1, blit=True, repeat=False
    )
    plt.show()

def plot_packings_noised_denoised(inputs, noised, output, sample_idx=0):
    plt.figure(figsize=(6, 6))
    # Original packing
    plt.scatter(inputs[sample_idx][0], inputs[sample_idx][1], c="black", label="Original", alpha=0.5)
    # Noised sample
    plt.scatter(noised[sample_idx][0], noised[sample_idx][1], c="blue", label="Noised", alpha=0.5)
    # Final denoised
    final = output[sample_idx]
    for i in range(min(5,final.shape[0])):
        plt.scatter(final[i][0], final[i][1], c="yellow", alpha=0.7, s=10)
    # Plot the final denoised packing
    final = final[-1]  # Take the last step of denoising
    plt.scatter(final[0], final[1], c="red", label="Denoised", alpha=0.7)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Sample {sample_idx}: Original, Noised, Denoised")
    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.show()

if __name__ == "__main__":
    section = cfg2["diffusion_model"]
    # --- Extract parameters ---

    batch_size = int(section["batch_size"])
    d = int(section["dimension"])
    num_train_timesteps = int(section["num_train_timesteps"])
    num_inference_timesteps = int(section["num_inference_timesteps"])
    num_new_from_one = int(section.get("num_new_from_one", 1))  # Default to 1 if not specified
    beta_start = float(section["beta_start"])
    beta_end = float(section["beta_end"])
    clip_sample = section.getboolean("clip_sample")
    clip_sample_range = float(section["clip_sample_range"])
    learning_rate = float(section["learning_rate"])
    num_epochs = int(section["num_epochs"])
    sphere_radius = float(section["sphere_radius"])
    dataset_path = section["dataset_path"]
    model_type = section.get("model_type", "pointnet").lower()
    output_save_path = section["output_save_path"]

    # SetTransformer-specific parameters (with defaults)
    st_dim_hidden = int(section.get("st_dim_hidden", 128))
    st_num_heads = int(section.get("st_num_heads", 4))
    st_num_inds = int(section.get("st_num_inds", 16))
    st_num_isab = int(section.get("st_num_isab", 2))

    # --- Load and split dataset ---
    dataset = SpherePackingDataset(dataset_path)
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train DataLoader: {len(train_data_loader.dataset)} samples")
    print(f"Validation DataLoader: {len(val_data_loader.dataset)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model selection ---
    if model_type == "pointnet":
        model = PointNetPlusPlus(d)
    elif model_type == "settransformer":
        model = SetTransformer(dim_in=d, dim_hidden=st_dim_hidden, num_heads=st_num_heads, num_inds=st_num_inds, num_isab=st_num_isab, dim_out=d)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SetTransformer: total params = {total_params:,}, trainable = {trainable_params:,}")
    # Print the shape of the train dataset
    print(f"Train dataset shape: {train_dataset.dataset.data.shape}")
    print("Training the diffusion model...")
    model, loss_history = train_diffusion_model(
        model, train_data_loader, num_epochs, learning_rate, num_train_timesteps, sphere_radius,
        beta_start, beta_end, clip_sample, clip_sample_range, device
    )
    

    from datetime import datetime
    s_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the model
    model_name = f"diffusion_model_{model_type}_{s_now}.pth"
    model_save_path = f"output/saved_models/{model_name}"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    
    
    # 2) Epoch-average loss
    batches_per_epoch = len(train_data_loader)
    epoch_avgs = [
        np.mean(loss_history[i*batches_per_epoch:(i+1)*batches_per_epoch])
        for i in range(num_epochs)
    ]
    plt.plot(
        np.arange(batches_per_epoch/2, batches_per_epoch*num_epochs, batches_per_epoch),
        epoch_avgs, color="red", marker="o", label="Epoch avg"
    )

    plt.xlabel("Batch index")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.yscale("log")        # optional: log scale often helps
    plt.show()

    inputs, noised, output = sample_diffusion_model(
    model, val_data_loader, num_train_timesteps, num_inference_timesteps,
    beta_start, beta_end, clip_sample, clip_sample_range, device, num_new_from_one
    )

    print("Input shape:", inputs.shape, "Output shape:", output.shape)
    
    output_tensor = torch.tensor(output, dtype=torch.float32)
    # Save the output tensor to the given output_save_path directory, by adding the date and time to the filename
    output_save_path = f"{output_save_path}/diffusion_output_{s_now}.pt"
    torch.save(output_tensor, output_save_path)
    print(f"Generated output saved to {output_save_path}") 

    # Plot the first 5 samples
    for i in range(min(5, inputs.shape[0])):
        plot_packings_noised_denoised(
            inputs, noised, output, sample_idx=i
        )

        
    #animate_sample(inputs, output, sample_idx=0)