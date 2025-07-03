import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import configparser

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
        return self.mlp(x.float())

class PointNetPlusPlus(nn.Module):
    def __init__(self, d):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstraction(d, [64, 128])
        self.sa2 = PointNetSetAbstraction(128, [256, 512])
        self.mlp = nn.Conv1d(512, d, 1)
    def forward(self, x):
        x = x.float()
        x = self.sa1(x)
        x = self.sa2(x)
        return self.mlp(x)

# --- SetTransformer Components ---
class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
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
        out = (attn @ v).transpose(1,2).contiguous().view(B, Nq, D)
        return self.out_proj(out), attn

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
        I = self.inducing.expand(B, -1, -1)
        H1, _ = self.mab1(I, H, H)
        H2, _ = self.mab2(H, H1, H1)
        return H2

class SetTransformer(nn.Module):
    def __init__(self, dim_in, dim_hidden=128, num_heads=4, num_inds=16, num_isab=2, dim_out=None):
        super().__init__()
        self.isabs = nn.ModuleList([
            ISAB(dim_in if i==0 else dim_hidden, dim_hidden, num_heads, num_inds)
            for i in range(num_isab)
        ])
        self.fc_out = nn.Linear(dim_hidden, dim_out or dim_in)
    def forward(self, X):
        X = X.permute(0, 2, 1)  # (B, N, d)
        for isab in self.isabs:
            X = isab(X)
        X = self.fc_out(X)      # (B, N, d_out)
        return X.permute(0, 2, 1)  # (B, d_out, N)

# --- Distance penalty ---
def distance_penalty(output, radius):
    B, d, N = output.shape
    coords = output.permute(0, 2, 1)
    distances = torch.cdist(coords, coords)
    violation = torch.relu(2 * radius - distances)
    num_pairs = B * N * (N - 1) / 2
    return (violation**2).sum() / (num_pairs**2)

# --- Training routine ---
def train_diffusion_model(
    model, train_data_loader, num_epochs, learning_rate,
    num_train_timesteps, sphere_radius,
    beta_start, beta_end, clip_sample, clip_sample_range, device
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
    for epoch in tqdm(range(num_epochs), desc="Training"):
        for batch in train_data_loader:
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, num_train_timesteps, (batch.shape[0],), device=device)
            noisy = scheduler.add_noise(batch, noise, timesteps)
            predicted_noise = model(noisy)
            mse_loss = criterion(predicted_noise, noise)
            a_bar = scheduler.alphas_cumprod[timesteps].view(-1,1,1).to(device)
            x0_pred = (noisy - torch.sqrt(1 - a_bar) * predicted_noise) / torch.sqrt(a_bar)
            penalty_loss = distance_penalty(x0_pred, sphere_radius)
            l_max = 500
            l = l_max * (epoch / num_epochs)
            loss = mse_loss + l * penalty_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model, loss_history

# --- Revised sampling from Gaussian prior ---
def sample_diffusion_model(
    model, num_samples, num_points,
    num_train_timesteps, num_inference_timesteps,
    beta_start, beta_end,
    clip_sample, clip_sample_range,
    sphere_radius, device,
    d  # dimension read from config
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
    x = torch.randn(num_samples, d, num_points, device=device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            eps = model(x)
            x = scheduler.step(eps, t, x).prev_sample
            x= x.clamp(sphere_radius, clip_sample_range-sphere_radius)
    return x.cpu().numpy()

# --- Dataset Definition ---
class SpherePackingDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path)
        print(f"Loaded dataset {path}, shape {self.data.shape}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int):
        return self.data[idx]

# --- Main script ---
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.cfg")
    sec = config["diffusion_model"]
    # Read dimension from config
    d = int(sec["dimension"])
    batch_size = int(sec["batch_size"])
    dataset_path = sec["dataset_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    from SP.data_generation import SpherePackingDataset
    dataset = SpherePackingDataset(dataset_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model init
    if sec.get("model_type", "pointnet").lower() == "pointnet":
        model = PointNetPlusPlus(d)
    else:
        model = SetTransformer(
            dim_in=d,
            dim_hidden=int(sec.get("st_dim_hidden", 128)),
            num_heads=int(sec.get("st_num_heads", 4)),
            num_inds=int(sec.get("st_num_inds", 16)),
            num_isab=int(sec.get("st_num_isab", 2)),
            dim_out=d
        )
    # report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SetTransformer: total params = {total_params:,}, trainable = {trainable_params:,}")
    # Print the shape of the train dataset
    print(f"Train dataset shape: {train_loader.dataset.data.shape}")
    print("Training the diffusion model...")
    # Train
    model, loss_history = train_diffusion_model(
        model,
        train_loader,
        int(sec["num_epochs"]),
        float(sec["learning_rate"]),
        int(sec["num_train_timesteps"]),
        float(sec["sphere_radius"]),
        float(sec["beta_start"]),
        float(sec["beta_end"]),
        sec.getboolean("clip_sample"),
        float(sec["clip_sample_range"]),
        device
    )

    # Save the trained model
    from datetime import datetime
    s_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"diffusion_model_{s_now}.pth"
    # `save_model_path` in config should be the **directory** where to save
    save_dir = sec["save_model_path"]
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 2) Epoch-average loss
    batches_per_epoch = len(train_loader)
    epoch_avgs = [
        np.mean(loss_history[i*batches_per_epoch:(i+1)*batches_per_epoch])
        for i in range(int(sec["num_epochs"]))
    ]
    plt.plot(
        np.arange(batches_per_epoch/2, batches_per_epoch*int(sec["num_epochs"]), batches_per_epoch),
        epoch_avgs, color="red", marker="o", label="Epoch avg"
    )

    plt.xlabel("Batch index")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.yscale("log")        # optional: log scale often helps
    plt.show()

    # Sample new packings
    num_generate = int(sec.get("num_generate", 10))
    num_points = int(sec["num_spheres"])
    print(f"Starting diffusion-based generation of {num_generate} samples...")
    samples = sample_diffusion_model(
        model,
        num_generate,
        num_points,
        int(sec["num_train_timesteps"]),
        int(sec["num_inference_timesteps"]),
        float(sec["beta_start"]),
        float(sec["beta_end"]),
        sec.getboolean("clip_sample"),
        float(sec["clip_sample_range"]),
        float(sec["sphere_radius"]),
        device,
        d
    )

    # Save outputs

    s_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Generated {num_generate} samples, saving to {sec['save_generated_path']}")
    output_save_path = os.path.join(sec["save_generated_path"], f"generated_{s_now}.pt")
    os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
    torch.save(torch.from_numpy(samples), output_save_path)
    print(f"Samples saved to {output_save_path}")