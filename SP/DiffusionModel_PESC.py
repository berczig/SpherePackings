import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import configparser
from SP import data_load_save
from datetime import datetime
from plot_data_points import plot_3d

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
def distance_penalty(output:np.array, radius):
    B, d, N = output.shape
    coords = output.permute(0, 2, 1) # B, N, d
    distances = torch.cdist(coords, coords) # B, N, N
    violation = torch.relu(2 * radius - distances) # B, N, N
    upper_triangle_mask = torch.triu(
        torch.ones(B, N, N, dtype=torch.bool, device=distances.device), 
        diagonal=1 # Excludes the diagonal elements (self-distances)
    )
    violation_masked = violation*upper_triangle_mask
    num_pairs = N * (N - 1) / 2
    return (violation_masked**2).sum() / (num_pairs*B)

# --- Save Model with Loss plot ---
def save_with_plot(model, optimizer, loss_history, current_epoch, model_parameters, sec):
    save_dir = sec["save_model_path"]
    # Save the trained model
    current_loss = loss_history[-1][2]
    s_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"diffusion_model_loss={current_loss:.4f}_{s_now}.pth"
    # `save_model_path` in config should be the **directory** where to save
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, model_name)
    #torch.save(model.state_dict(), model_save_path)
    data_load_save.save_model(model_save_path, model, optimizer, current_epoch, model_parameters)
    print(f"Model saved to {model_save_path}")

    # 2) Epoch-average loss
    #batches_per_epoch = len(train_loader)
    print("shape: ", loss_history[:, 1].shape)
    distance_penality_strength = sec["distance_penality_strength"]
    mse_strength = sec["mse_strength"]
    plt.plot(range(len(loss_history)),
        loss_history[:, 0], color="red", marker="o", label=f"MSE Loss [{mse_strength}]")
    plt.plot(range(len(loss_history)),
        loss_history[:, 1], color="blue", marker="o", label=f"Distance Loss [{distance_penality_strength}]")
    plt.plot(range(len(loss_history)),
        loss_history[:, 2], color="yellow", marker="o", label="Loss")
    """plt.plot(
        np.arange(batches_per_epoch/2, batches_per_epoch*int(sec["num_epochs"]), batches_per_epoch),
        epoch_avgs, color="red", marker="o", label="Epoch avg"
    )"""

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend(loc='best')
    plt.yscale("log")        # optional: log scale often helps
    #plt.show()
    plt.rcParams["figure.figsize"] = (12,6)
    plot_filename = f"diffusion_model_loss={current_loss:.6f}_{s_now}_loss.png"
    plt.savefig(os.path.join(save_dir, plot_filename))

# --- Training routine ---
def train_diffusion_model(
    model, optimizer, train_data_loader, num_epochs, learning_rate,
    num_train_timesteps, sphere_radius,mse_strength, distance_penality_strength,
    beta_start, beta_end, clip_sample, clip_sample_range, device, model_parameters, save_model_path
):
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=clip_sample,
        clip_sample_range=clip_sample_range
    )
    criterion = nn.MSELoss()
    model.train().to(device)
    loss_history = []
    best_loss = np.inf
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_losses = []
        epoch_ratio = epoch / num_epochs
        for batch in train_data_loader:
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, num_train_timesteps, (batch.shape[0],), device=device)
            noisy = scheduler.add_noise(batch, noise, timesteps)
            predicted_noise = model(noisy)
            mse_loss = criterion(predicted_noise, noise)
            a_bar = scheduler.alphas_cumprod[timesteps].view(-1,1,1).to(device)
            x0_pred = (noisy - torch.sqrt(1 - a_bar) * predicted_noise) / torch.sqrt(a_bar)
            #x0_pred = x0_pred.clamp(sphere_radius, clip_sample_range-sphere_radius)
            penalty_loss = distance_penalty(x0_pred, sphere_radius)
            #plot_3d(x0_pred.detach().cpu().numpy())
            #print("penalty_loss: ", penalty_loss, "mse: ", mse_loss, "l: ", l, "l_max: ", distance_penality_strength)
            loss = mse_strength*mse_loss + distance_penality_strength*epoch_ratio*penalty_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(np.array([mse_loss.item(), penalty_loss.item(), loss.item()]))
            #print("current loss: ", loss.item())
        loss_avg = np.mean(epoch_losses, axis=0)
        loss_history.append(loss_avg)
        print(f"Epoch [{epoch+1}/{num_epochs}], MSE Loss: {loss_avg[0]:.9f}, Penalty Loss: {loss_avg[1]:.9f}, Loss: {loss_avg[2]:.9f}")

        if loss_avg[2] < best_loss:
            best_loss = loss_avg[2]
            save_with_plot(model, optimizer, np.array(loss_history), epoch, model_parameters, sec)
    return model, np.array(loss_history)

# --- Revised sampling from Gaussian prior ---
def sample_diffusion_model(
    model, num_samples:int, num_samples_batch_size:int, num_points,
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
    print(f"sample on device {device}")

    
    batches = np.empty(int(np.ceil(num_samples/num_samples_batch_size)))
    batches.fill(int(num_samples_batch_size))
    if num_samples%num_samples_batch_size != 0:
        batches[-1] = num_samples%num_samples_batch_size

    samples_batched = []
    for batch_size in tqdm(batches.astype(int), desc="Sampling"):
        x = torch.randn(batch_size, d, num_points, device=device)
        with torch.no_grad():
            for t in scheduler.timesteps:
                eps = model(x)
                x = scheduler.step(eps, t, x).prev_sample
                #plot_3d(x.cpu().numpy(), "pre clamp")
                x= x.clamp(sphere_radius, clip_sample_range-sphere_radius)
                #plot_3d(x.cpu().numpy(), "after clamp")
        samples_batched.append(x.cpu().numpy())
    return np.concatenate(samples_batched)

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
    config.read("config.cfg", encoding="utf-8")
    sec = config["diffusion_model"]
    # Read dimension from config
    d = int(sec["dimension"])
    batch_size = int(sec["batch_size"])
    dataset_path = sec["dataset_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)

    # Load data
    from SP.data_generation import SpherePackingDataset
    dataset = SpherePackingDataset(dataset_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get Model
    start_epoch = 0
    if sec["load_model"]=="True":
        # Load Model
        model_class = PointNetPlusPlus if sec.get("model_type", "pointnet").lower() == "pointnet" else SetTransformer
        model, optimizer, checkpoint = data_load_save.load_model(sec["load_model_path"], model_class, torch.optim.AdamW, learning_rate=float(sec["learning_rate"]), device=device)
        start_epoch = int(checkpoint["epoch"])
        model_parameters = checkpoint["model_parameters"]
    else:
        # Model init
        if sec.get("model_type", "pointnet").lower() == "pointnet":
            model_parameters = {"d":d}
            model = PointNetPlusPlus(**model_parameters)
        else:
            model_parameters = {"dim_in":d,
                "dim_hidden":int(sec.get("st_dim_hidden", 128)),
                "num_heads":int(sec.get("st_num_heads", 4)),
                "num_inds":int(sec.get("st_num_inds", 16)),
                "num_isab":int(sec.get("st_num_isab", 2)),
                "dim_out":d}
            model = SetTransformer(**model_parameters)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(sec["learning_rate"]))
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
        optimizer,
        train_loader,
        int(sec["num_epochs"]),
        float(sec["learning_rate"]),
        int(sec["num_train_timesteps"]),
        float(sec["sphere_radius"]),
        float(sec["mse_strength"]),
        float(sec["distance_penality_strength"]),
        float(sec["beta_start"]),
        float(sec["beta_end"]),
        sec.getboolean("clip_sample"),
        float(sec["clip_sample_range"]),
        device,
        model_parameters,
        sec["save_model_path"]
    )

    save_with_plot(model, optimizer, loss_history, start_epoch+int(sec["num_epochs"]), model_parameters, sec)

    # Sample new packings
    sample_new_points = int(sec.get("sample_new_points", 10))
    sample_new_points_batch_size = int(sec.get("sample_new_points_batch_size", 10))
    num_points = int(sec["num_spheres"])
    print(f"Starting diffusion-based generation of {sample_new_points} samples...")
    samples = sample_diffusion_model(
        model,
        sample_new_points,
        sample_new_points_batch_size,
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

    s_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Generated {sample_new_points} samples, saving to {sec['save_generated_path']}")
    output_save_path = os.path.join(sec["save_generated_path"], f"generated_{s_now}.pt")
    os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
    torch.save(torch.from_numpy(samples), output_save_path)
    print(f"Samples saved to {output_save_path}")
