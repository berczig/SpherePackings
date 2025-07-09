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
        X = X.permute(0, 2, 1)
        for isab in self.isabs:
            X = isab(X)
        X = self.fc_out(X)
        return X.permute(0, 2, 1)

# --- Distance penalty ---
def distance_penalty(output: torch.Tensor, radius):
    B, d, N = output.shape
    coords = output.permute(0, 2, 1)
    distances = torch.cdist(coords, coords)
    violation = torch.relu(2 * radius - distances)
    mask = torch.triu(torch.ones(B, N, N, device=distances.device, dtype=torch.bool), diagonal=1)
    violation = violation * mask
    num_pairs = N * (N - 1) / 2
    return (violation**2).sum() / (num_pairs * B)

# --- Reflection projection ---
def reflect(x: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    x = torch.where(x < bmin, 2*bmin - x, x)
    x = torch.where(x > bmax, 2*bmax - x, x)
    return x

# --- Save Model with Loss plot ---
def save_with_plot(model, optimizer, loss_history, current_epoch, model_parameters, sec):
    save_dir = sec["save_model_path"]
    current_loss = loss_history[-1][2]
    s_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    model_name = f"diffusion_model_loss={current_loss:.4f}_{s_now}.pth"
    path = os.path.join(save_dir, model_name)
    data_load_save.save_model(path, model, optimizer, current_epoch, model_parameters)
    print(f"Model saved to {path}")

    plt.plot(range(len(loss_history)), loss_history[:, 0], marker='o', label=f"MSE [{sec['mse_strength']}] ")
    plt.plot(range(len(loss_history)), loss_history[:, 1], marker='o', label=f"Distance [{sec['distance_penality_strength']}] ")
    plt.plot(range(len(loss_history)), loss_history[:, 2], marker='o', label="Total")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.legend(); plt.yscale("log")
    plt.rcParams['figure.figsize'] = (12, 6)
    fname = f"diffusion_loss={current_loss:.6f}_{s_now}.png"
    plt.savefig(os.path.join(save_dir, fname))

# --- Training routine ---
def train_diffusion_model(
    model, optimizer, loader, num_epochs,
    num_train_timesteps, sphere_radius, mse_strength,
    distance_penality_strength, beta_start, beta_end,
    clip_sample, clip_sample_range, device,
    model_parameters, save_model_path
):
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start, beta_end=beta_end,
        clip_sample=clip_sample, clip_sample_range=clip_sample_range
    )
    criterion = nn.MSELoss()
    model.train().to(device)

    # box bounds as tensors
    bmin_t = torch.tensor(sphere_radius, device=device)
    bmax_t = torch.tensor(clip_sample_range - sphere_radius, device=device)

    best_loss = np.inf
    loss_history = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_losses = []
        epoch_ratio = epoch / num_epochs
        for batch in loader:
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            t = torch.randint(0, num_train_timesteps, (batch.size(0),), device=device)
            noisy = scheduler.add_noise(batch, noise, t)

            eps_pred = model(noisy)
            mse = criterion(eps_pred, noise)

            a_bar = scheduler.alphas_cumprod[t].view(-1,1,1).to(device)
            raw_x0 = (noisy - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
            x0_pred = reflect(raw_x0, bmin_t, bmax_t)

            pen = distance_penalty(x0_pred, sphere_radius)
            loss = mse_strength*mse + distance_penality_strength*epoch_ratio*pen

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_losses.append([mse.item(), pen.item(), loss.item()])

        avg = np.mean(epoch_losses, axis=0)
        loss_history.append(avg)
        print(f"Epoch {epoch+1}/{num_epochs}: MSE={avg[0]:.6f}, Pen={avg[1]:.6f}, Tot={avg[2]:.6f}")

        if avg[2] < best_loss:
            best_loss = avg[2]
            save_with_plot(model, optimizer, np.array(loss_history), epoch, model_parameters, {'save_model_path': save_model_path, 'mse_strength': mse_strength, 'distance_penality_strength': distance_penality_strength})

    return model, np.array(loss_history)

# --- Sampling routine ---
def sample_diffusion_model(
    model, num_samples, batch_size, num_points,
    num_train_timesteps, num_inference_timesteps,
    beta_start, beta_end, clip_sample, clip_sample_range,
    sphere_radius, device, d
):
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start, beta_end=beta_end,
        clip_sample=clip_sample, clip_sample_range=clip_sample_range
    )
    scheduler.set_timesteps(num_inference_timesteps)
    model.eval()

    bmin_t = torch.tensor(sphere_radius, device=device)
    bmax_t = torch.tensor(clip_sample_range - sphere_radius, device=device)

    sizes = np.full(int(np.ceil(num_samples/batch_size)), batch_size, dtype=int)
    if num_samples % batch_size != 0:
        sizes[-1] = num_samples % batch_size

    all_samples = []
    for bs in tqdm(sizes, desc="Sampling"):
        x = torch.randn(bs, d, num_points, device=device)
        with torch.no_grad():
            for t in scheduler.timesteps:
                eps = model(x)
                out = scheduler.step(eps, t, x)
                x = out.prev_sample
                x = reflect(x, bmin_t, bmax_t)
        all_samples.append(x.cpu().numpy())

    return np.concatenate(all_samples)

# --- Dataset ---
class SpherePackingDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path)
        print(f"Loaded {path}, shape={self.data.shape}")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- Main ---
if __name__ == "__main__":
    cfg = configparser.ConfigParser()
    cfg.read("config.cfg")
    sec = cfg["diffusion_model"]
    d = int(sec["dimension"])
    bs = int(sec["batch_size"])
    path = sec["dataset_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpherePackingDataset(path)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    if sec.getboolean("load_model"):
        cls = PointNetPlusPlus if sec.get("model_type","pointnet").lower()=="pointnet" else SetTransformer
        model, opt, ckpt = data_load_save.load_model(
            sec["load_model_path"], cls, torch.optim.AdamW,
            learning_rate=float(sec["learning_rate"]), device=device)
        start_epoch = int(ckpt["epoch"])
        params = ckpt["model_parameters"]
    else:
        if sec.get("model_type","pointnet").lower()=="pointnet":
            params = {"d": d}
            model = PointNetPlusPlus(**params)
        else:
            params = {"dim_in":d, "dim_hidden":int(sec.get("st_dim_hidden",128)),
                      "num_heads":int(sec.get("st_num_heads",4)), "num_inds":int(sec.get("st_num_inds",16)),
                      "num_isab":int(sec.get("st_num_isab",2)), "dim_out":d}
            model = SetTransformer(**params)
        opt = torch.optim.AdamW(model.parameters(), lr=float(sec["learning_rate"]))
        start_epoch = 0

    print(f"Params: total={sum(p.numel() for p in model.parameters())}, trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model, lh = train_diffusion_model(
        model, opt, loader, int(sec["num_epochs"]),
        int(sec["num_train_timesteps"]), float(sec["sphere_radius"]),
        float(sec["mse_strength"]), float(sec["distance_penality_strength"]),
        float(sec["beta_start"]), float(sec["beta_end"]),
        sec.getboolean("clip_sample"), float(sec["clip_sample_range"]),
        device, params, sec["save_model_path"]
    )

    save_with_plot(model, opt, lh, start_epoch+int(sec["num_epochs"]), params, sec)

    nnew = int(sec.get("sample_new_points",10))
    bnew = int(sec.get("sample_new_points_batch_size",10))
    npts = int(sec["num_spheres"])
    print(f"Generating {nnew} samples...")
    gen = sample_diffusion_model(
        model, nnew, bnew, npts,
        int(sec["num_train_timesteps"]), int(sec["num_inference_timesteps"]),
        float(sec["beta_start"]), float(sec["beta_end"]),
        sec.getboolean("clip_sample"), float(sec["clip_sample_range"]),
        float(sec["sphere_radius"]), device, d
    )
    out = os.path.join(sec["save_generated_path"], f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(torch.from_numpy(gen), out)
    print(f"Saved to {out}")
