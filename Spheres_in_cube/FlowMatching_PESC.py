import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import configparser
from spheres_in_cube import data_load_save
from datetime import datetime
from plot_data_points import plot_3d
from torchdiffeq import odeint
from torch.optim.lr_scheduler import CosineAnnealingLR
from spheres_in_cube import cfg

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
        self.fc0  = nn.Linear(dim_in, dim_hidden)
    def forward(self, X):
        B, N, _ = X.shape
        H = self.fc0(X)
        I = self.inducing.expand(B, -1, -1)
        H1,_ = self.mab1(I, H, H)
        H2,_ = self.mab2(H, H1, H1)
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

# --- Time Embedding for Flow Matching ---
class TimeEmbedding(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, t):
        # expects t shape (B,1), returns (B,dim)
        return self.mlp(t)

# --- Flow Matching Model ---
class FlowSetTransformer(nn.Module):
    def __init__(self, d, **st_kwargs):
        super().__init__()
        self.time_emb = TimeEmbedding(d)
        self.set_tf   = SetTransformer(dim_in=d*2, **st_kwargs)

    def forward(self, t, x):
        B, d, N = x.shape
        xt = x.permute(0,2,1)            # (B, N, d)
        t_vec = t.view(-1,1).to(x.device)  # (B,1)
        t_emb = self.time_emb(t_vec)    # (B, d)
        t_rep = t_emb.unsqueeze(1).expand(B, N, d)  # (B, N, d)
        inp   = torch.cat([xt, t_rep], dim=-1)  # (B, N, 2d)
        inp   = inp.permute(0,2,1)             # (B, 2d, N)
        out   = self.set_tf(inp)               # (B, d, N)
        return out

# --- Distance penalty ---
def distance_penalty(output: torch.Tensor, radius):
    B, d, N = output.shape
    coords   = output.permute(0,2,1)
    distances= torch.cdist(coords, coords)
    violation= torch.relu(2*radius - distances)
    mask     = torch.triu(torch.ones(B,N,N,device=distances.device), diagonal=1).bool()
    vio_mask = violation * mask
    num_pairs= N*(N-1)/2
    return (vio_mask**2).sum()/(num_pairs*B)

# --- Save Model & Loss Plot ---
def save_with_plot(model, optimizer, history, epoch, params, sec):
    save_dir = sec
    os.makedirs(save_dir, exist_ok=True)
    loss = history[-1,2]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"flow_model_loss={loss:.4f}_{ts}.pth"
    path = os.path.join(save_dir, name)
    data_load_save.save_model(path, model, optimizer, epoch, params)

    plt.figure(figsize=(12,6))
    plt.plot(history[:,0], label="Flow MSE Loss")
    plt.plot(history[:,1], label="Distance Penalty")
    plt.plot(history[:,2], label="Total Loss")
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(save_dir, f"flow_loss_{ts}.png"))

# --- Training routine (Flow Matching) ---
def train_flow_model(
    model, optimizer, scheduler, loader, num_epochs,
    sphere_radius, mse_strength, dist_strength,
    clip_min, clip_max, device, params, save_path
):
    model.train().to(device)
    mse = nn.MSELoss()
    history, best = [], 1e9

    bmin   = torch.tensor(sphere_radius, device=device)
    bmax   = torch.tensor(clip_max - sphere_radius, device=device)
    center = (bmin + bmax)*0.5
    half   = (bmax - bmin)*0.5

    for epoch in range(num_epochs):
        ep_losses=[]
        ratio=epoch/num_epochs
        for x0 in loader:
            x0  = x0.to(device)
            eps = torch.randn_like(x0)
            t   = torch.rand(x0.size(0), device=device)
            xt  = (1-t).view(-1,1,1)*x0 + t.view(-1,1,1)*eps

            u_star = eps - x0
            u_pred = model(t, xt)
            loss_fm= mse(u_pred, u_star)

            x0_pred= xt - t.view(-1,1,1)*u_pred
            x0_proj= x0_pred #center + half*torch.tanh((x0_pred-center)/half)
            pen    = distance_penalty(x0_proj, sphere_radius)

            loss   = mse_strength*loss_fm + dist_strength*ratio*pen

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_losses.append([loss_fm.item(), pen.item(), loss.item()])

        avg = np.mean(ep_losses, axis=0)
        history.append(avg)
        print(f"Epoch {epoch+1}/{num_epochs} | FM={avg[0]:.4f} Pen={avg[1]:.4f} Tot={avg[2]:.4f}")

    save_with_plot(model, optimizer, np.array(history), num_epochs-1, params, save_path)
    return model, np.array(history)

# --- Sampling via ODE Integration ---
def sample_flow_model(model, num_samples, batch_size, num_points,
                      device, sphere_radius, clip_min, clip_max, dim):
    model.eval()
    samples=[]

    bmin   = torch.tensor(sphere_radius, device=device)
    bmax   = torch.tensor(clip_max - sphere_radius, device=device)
    center = (bmin + bmax)*0.5
    half   = (bmax - bmin)*0.5

    for _ in range(int(np.ceil(num_samples/batch_size))):
        bs = min(batch_size, num_samples)
        xT = torch.randn(bs, dim, num_points, device=device)
        t_span = torch.tensor([1.0, 0.0], device=device)
        with torch.no_grad():
            def vf(t, x): return model(torch.tensor(t, device=device), x)
            out = odeint(vf, xT, t_span, atol=1e-5, rtol=1e-5)[-1]
            proj= out #center + half*torch.tanh((out-center)/half)
            samples.append(proj.cpu().numpy())

    return np.concatenate(samples, axis=0)

# --- Dataset and Main ---
class SpherePackingDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
        print(f"Loaded {path}, shape {self.data.shape}")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

if __name__ == '__main__':
    sec = cfg['flow_matching']

    d      = int(sec['dimension'])
    bs     = int(sec['batch_size'])
    path   = sec['dataset_path']
    lr     = float(sec['learning_rate'])
    epochs = int(sec['num_epochs'])
    radius = float(sec['sphere_radius'])
    mse_s  = float(sec['mse_strength'])
    pen_s  = float(sec['distance_penality_strength'])
    clip_r = float(sec['clip_sample_range'])
    save_m = sec['save_model_path']
    num_new= int(sec.get('sample_new_points', 10))
    batch_n= int(sec.get('sample_new_points_batch_size', 10))
    pts    = int(sec['num_spheres'])
    save_g = sec['save_generated_path']

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SpherePackingDataset(path)
    max_samples = min(len(dataset), 5000)
    dataset = Subset(dataset, list(range(max_samples)))
    loader  = DataLoader(dataset, batch_size=bs, shuffle=True)

    st_kwargs = {
        'dim_hidden': int(sec.get('st_dim_hidden', 128)),
        'num_heads':  int(sec.get('st_num_heads', 4)),
        'num_inds':   int(sec.get('st_num_inds', 16)),
        'num_isab':   int(sec.get('st_num_isab', 2)),
        'dim_out':    d
    }
    model = FlowSetTransformer(d, **st_kwargs).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    model, hist = train_flow_model(
        model, opt, scheduler, loader, epochs,
        radius, mse_s, pen_s,
        clip_r, clip_r, dev,
        {'dim_in': d}, save_m
    )

    samples = sample_flow_model(
        model, num_new, batch_n, pts,
        dev, radius, clip_r, clip_r, d
    )
    os.makedirs(save_g, exist_ok=True)
    out_path = os.path.join(save_g, f"flow_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(torch.from_numpy(samples), out_path)
    print(f"Saved {num_new} samples to {out_path}")
