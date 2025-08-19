# Conditional Flow Matching for sphere packings with lattice sources

#The Conditional Flow Model builds directly on your original flow-matching code but adds a 
# key idea: instead of always flowing from Gaussian noise to good packings, it learns flows 
# conditioned on a structured lattice source arrangement (SC, BCC, FCC, HCP, with 
# jitter/rotation). To handle the fact that packings are unordered sets, it uses Sinkhorn 
# optimal transport matching (or cheaper alternatives) to align lattice points with target 
# packings before computing displacements. The model itself is extended with an EGNN-based 
# context encoder for the source set, a metadata encoder for lattice type/parameters, and 
# optional local geometry features. Training includes new regularizers like a boundary 
# velocity penalty (to keep spheres inside the cube) and EMA for stability. 
# Overall, it’s a more specialized, geometry-aware variant of your original flow model, 
# designed to leverage lattice priors and better respect physical constraints.

# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os, platform
# If you need the quick unblock, leave this enabled on macOS.
# Disable by exporting SPHEREPACK_DISABLE_KMP_HACK=1 in your shell.
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    # Intel notes this is an unsafe workaround; prefer the environment fix below.
    # It must be set BEFORE any library initializes OpenMP.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time, math, random
import warnings
import torch                 
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from datetime import datetime
from torchdiffeq import odeint
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as nn_utils

from diffuse_boost.spheres_in_cube import data_load_save
from diffuse_boost import cfg
from diffuse_boost.spheres_in_cube.DiffusionModel_PESC import SetTransformer
from plot_data_points import plot_3d

# Simple debug flag to print sanity checks occasionally
DEBUG_SANITY = True

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def lerp(a, b, w): return a*(1.0 - w) + b*w

def schedule_sigmoid(epoch, total_epochs, center=0.5, sharpness=8.0):
    # smooth 0->1 ramp
    if total_epochs <= 1: return 1.0
    x = (epoch / (total_epochs - 1) - center) * sharpness
    return float(1.0 / (1.0 + math.exp(-x)))

# -----------------------------------------------------------
# Geometry & constraints
# -----------------------------------------------------------

def pairwise_dists(x):  # (B,d,N) -> (B,N,N)
    return torch.cdist(x.permute(0,2,1).contiguous(), x.permute(0,2,1).contiguous())

def collision_energy(x, r):
    # x: (B,d,N)
    B, d, N = x.shape
    coords = x.permute(0,2,1)               # (B,N,d)
    D = torch.cdist(coords, coords)         # (B,N,N)
    viol = torch.relu(2*r - D)
    mask = torch.triu(torch.ones_like(viol), diagonal=1)
    e = (viol**4 * mask).sum(dim=(1,2)) / (N*(N-1)/2 + 1e-8)
    return e  # (B,)

def distance_penalty(x, r):
    return collision_energy(x, r).mean()

def grad_collision(x, r):
    x = x.requires_grad_(True)
    e = collision_energy(x, r).sum()
    g = torch.autograd.grad(e, x, create_graph=False, retain_graph=False)[0]
    return g

# ---------------- Boundary velocity penalty ----------------
def boundary_velocity_penalty(x_t, v_pred, r, cube_min, cube_max, margin=0.03, power=2.0):
    """
    Penalize outward velocity components for points near cube faces.
    For axis a:
      lower face @ x = cube_min + r: outward is negative v_a
      upper face @ x = cube_max - r: outward is positive v_a
    We weight the hinge by proximity to the face within `margin`.
    x_t, v_pred: (B,d,N)
    """
    B, d, N = x_t.shape
    xmin = cube_min + r
    xmax = cube_max - r
    pen = 0.0
    for a in range(d):
        xa = x_t[:, a, :]  # (B,N)
        va = v_pred[:, a, :]
        # gaps to the faces
        gap_lower = xa - xmin
        gap_upper = xmax - xa
        # proximity weights [0,1]
        w_lower = (margin - gap_lower).clamp(min=0.0) / max(margin, 1e-8)
        w_upper = (margin - gap_upper).clamp(min=0.0) / max(margin, 1e-8)
        # outward components
        out_lower = torch.relu(-va)  # pointing further out of domain on lower face
        out_upper = torch.relu( va)  # pointing further out of domain on upper face
        term = (out_lower * w_lower)**power + (out_upper * w_upper)**power
        pen = pen + term.mean()
    return pen / d

# -----------------------------------------------------------
# Lattice samplers (SC/BCC/FCC/HCP) with rotation + jitter
# -----------------------------------------------------------

def random_rotation(device):
    M = torch.randn(3,3, device=device)
    Q,_ = torch.linalg.qr(M)
    if torch.det(Q) < 0: Q[:,0] = -Q[:,0]
    trace = torch.clamp(Q.trace(), -1.0, 3.0)
    angle = torch.acos(torch.clamp((trace - 1)/2, -1.0, 1.0)).item()
    return Q, angle

def make_lattice_points(kind, k, a, clip_min, clip_max, device):
    offsets = {
        'sc' : [(0.0,0.0,0.0)],
        'bcc': [(0.0,0.0,0.0), (0.5,0.5,0.5)],
        'fcc': [(0.0,0.0,0.0), (0.5,0.5,0.0), (0.5,0.0,0.5), (0.0,0.5,0.5)],
        'hcp': [(0.0,0.0,0.0), (2/3,1/3,0.5)],  # approximate
    }[kind]
    pts = []
    for i in range(k):
        for j in range(k):
            for l in range(k):
                for ox,oy,oz in offsets:
                    x = (i + ox) * a + clip_min
                    y = (j + oy) * a + clip_min
                    z = (l + oz) * a + clip_min
                    if (clip_min <= x <= clip_max) and (clip_min <= y <= clip_max) and (clip_min <= z <= clip_max):
                        pts.append((x,y,z))
    return torch.tensor(pts, dtype=torch.float32, device=device)

def choose_k_for_kind(kind, N):
    if kind == 'sc':  k = math.ceil(N ** (1/3))
    elif kind == 'bcc': k = math.ceil((N/2) ** (1/3))
    elif kind == 'fcc': k = math.ceil((N/4) ** (1/3))
    elif kind == 'hcp': k = math.ceil((N/2) ** (1/3))
    else: raise ValueError(kind)
    return k

def sample_lattice_batch(
    batch_size, d, N, radius, cube_min, cube_max, device,
    kind='fcc', jitter=0.02, rotate=True, mix_kinds=False
):
    assert d == 3, "This sampler assumes 3D."
    kinds_all = ['sc','bcc','fcc','hcp']
    out = []
    metas = []
    side = cube_max - cube_min
    center = torch.tensor([(cube_min+cube_max)/2]*3, device=device)

    for _ in range(batch_size):
        knd = (random.choice(kinds_all) if mix_kinds else kind)
        k = choose_k_for_kind(knd, N)
        a = side / max(k,1)

        base = make_lattice_points(knd, k, a, cube_min, cube_max, device)  # (M,3)
        # pick N closest to center (dense core)
        dists = ((base - center)**2).sum(dim=1)
        idx = torch.argsort(dists)[:N]
        pts = base[idx]  # (N,3)

        Q, angle = (random_rotation(device) if rotate else (torch.eye(3, device=device), 0.0))
        pts = (Q @ (pts - center).T).T + center

        if jitter > 0:
            pts = pts + (jitter * a) * torch.randn_like(pts)

        pts = pts.clamp(min=cube_min, max=cube_max)
        pts = pts[torch.randperm(pts.shape[0], device=device)]
        out.append(pts.T.contiguous())  # (3,N)
        metas.append({'type': knd, 'a': a, 'jitter': float(jitter), 'rot_angle': float(angle)})

    return torch.stack(out, dim=0), metas  # (B,3,N), list(dict)

# -----------------------------------------------------------
# Soft Sinkhorn matching (OT) to align sets
# -----------------------------------------------------------

def sinkhorn_soft_matching(x_src, x_tgt, epsilon=0.05, iters=80):
    """
    x_src, x_tgt: (B,d,N)
    returns P: (B,N,N) **row-stochastic** aligning src rows to tgt cols
    """
    B, d, N = x_src.shape
    xs = x_src.permute(0,2,1)  # (B,N,d)
    xt = x_tgt.permute(0,2,1)
    C = torch.cdist(xs, xt).pow(2)  # (B,N,N)
    K = torch.exp(-C / max(epsilon, 1e-6)).clamp_min(1e-9)

    # target row/col marginals ≈ 1 (we'll normalize rows at the end)
    u = torch.ones(B, N, device=x_src.device)
    v = torch.ones(B, N, device=x_src.device)
    for _ in range(iters):
        Kv = (K @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
        u = 1.0 / Kv
        KTu = (K.transpose(1,2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
        v = 1.0 / KTu

    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # (B,N,N)
    # Ensure row-stochastic (sums to 1 along last dim)
    P = P / (P.sum(dim=-1, keepdim=True) + 1e-9)

    if DEBUG_SANITY:
        with torch.no_grad():
            rs = P.sum(-1)  # (B,N)
            cs = P.sum(-2)  # (B,N)
            if not torch.isfinite(P).all():
                warnings.warn("Non-finite values in Sinkhorn P")
            # Print once per call (mean ± std)
            print(f"[Sinkhorn] rows {rs.mean():.3f}±{rs.std():.3f} | cols {cs.mean():.3f}±{cs.std():.3f}")

    return P

# -----------------------------------------------------------
# Local geometry features (kNN)
# -----------------------------------------------------------

def knn_indices(x, k):
    # x: (B,d,N) -> (B,N,k) neighbor indices (excluding self)
    B, d, N = x.shape
    k = max(1, min(k, N - 1))  # avoid k >= N
    with torch.no_grad():
        D = pairwise_dists(x)  # (B,N,N)
        # push self-distance to +inf so it's never selected
        D = D + torch.eye(N, device=x.device).unsqueeze(0) * 1e9
        idx = torch.topk(D, k, dim=-1, largest=False).indices  # (B,N,k)
    return idx

def local_geom_features(x, k=8):
    """
    x: (B,d,N)
    returns feats: (B, 2d+2, N)
      - centered coords
      - vec to kNN barycenter
      - mean kNN distance (excluding self)
      - min  kNN distance (excluding self)
    """
    B, d, N = x.shape
    k = max(1, min(k, N - 1))
    xc = x - x.mean(dim=2, keepdim=True)  # (B,d,N)

    # kNN indices per point
    idx = knn_indices(x, k=k)  # (B,N,k)

    # neighbors via advanced indexing (robust and simple)
    xp = x.permute(0, 2, 1)  # (B,N,d)
    batch = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)  # (B,N,k)
    nb = xp[batch, idx, :]  # (B,N,k,d)

    bary = nb.mean(dim=2)                  # (B,N,d)
    vec_to_bary = (bary - xp).permute(0, 2, 1)  # (B,d,N)

    # distances for stats (exclude self)
    dmat = pairwise_dists(x)                                # (B,N,N)
    dmat = dmat + torch.eye(N, device=x.device).unsqueeze(0) * 1e9  # mask self
    d_sorted, _ = torch.sort(dmat, dim=-1)
    mean_k = d_sorted[:, :, :k].mean(dim=-1)  # (B,N)
    min_k  = d_sorted[:, :, 0]                # (B,N) nearest neighbor distance

    feats = torch.cat([
        xc,                              # (B,d,N)
        vec_to_bary,                     # (B,d,N)
        mean_k.unsqueeze(1),             # (B,1,N)
        min_k.unsqueeze(1)               # (B,1,N)
    ], dim=1)  # -> (B, 2d+2, N)

    return feats

# -----------------------------------------------------------
# Time embedding
# -----------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, dim))
    def forward(self, t):  # (B,) -> (B,dim)
        return self.mlp(t.view(-1,1))

# -----------------------------------------------------------
# Minimal EGNN context encoder (pose-robust global context)
# -----------------------------------------------------------

class EGNNLayer(nn.Module):
    def __init__(self, feat_in, feat_out, edge_hidden=64):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(feat_in*2 + 1, edge_hidden), nn.SiLU(),
            nn.Linear(edge_hidden, edge_hidden), nn.SiLU()
        )
        self.phi_x = nn.Sequential(nn.Linear(edge_hidden, 1), nn.Tanh())
        self.phi_h = nn.Sequential(nn.Linear(feat_in + edge_hidden, feat_out), nn.SiLU())

    def forward(self, x, h):
        # x: (B,N,3), h: (B,N,F)
        B,N,_ = x.shape
        xi = x.unsqueeze(2).expand(-1,-1,N,-1)
        xj = x.unsqueeze(1).expand(-1,N,-1,-1)
        rij = torch.norm(xi - xj, dim=-1, keepdim=True)
        hi = h.unsqueeze(2).expand(-1,-1,N,-1)
        hj = h.unsqueeze(1).expand(-1,N,-1,-1)
        e_ij = torch.cat([hi, hj, rij], dim=-1)      # (B,N,N,2F+1)
        e = self.phi_e(e_ij)                         # (B,N,N,H)
        x_update = ((xi - xj) * self.phi_x(e)).sum(dim=2)  # (B,N,3)
        x_new = x + x_update
        e_agg = e.sum(dim=2)                         # (B,N,H)
        h_new = self.phi_h(torch.cat([h, e_agg], dim=-1))
        return x_new, h_new

class EGNNContext(nn.Module):
    def __init__(self, d=3, feat=32, layers=2, out_dim=64):
        super().__init__()
        self.input = nn.Linear(d, feat)
        self.layers = nn.ModuleList([EGNNLayer(feat, feat) for _ in range(layers)])
        self.head   = nn.Sequential(nn.Linear(feat, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x_src):  # (B,d,N)
        B,d,N = x_src.shape
        x = x_src.permute(0,2,1)
        h = self.input(x)
        for layer in self.layers:
            x, h = layer(x, h)
        h = self.head(h)                 # (B,N,out_dim)
        g = self.pool(h.permute(0,2,1)).squeeze(-1)  # (B,out_dim)
        return g

# -----------------------------------------------------------
# Metadata encoder (+ optional context dropout)
# -----------------------------------------------------------

class LatticeMetadataEncoder(nn.Module):
    def __init__(self, type_list=('sc','bcc','fcc','hcp'), out_dim=32):
        super().__init__()
        self.types = list(type_list)
        self.type2idx = {t:i for i,t in enumerate(self.types)}
        self.emb = nn.Embedding(len(self.types), 16)
        self.mlp = nn.Sequential(
            nn.Linear(16 + 3, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim), nn.SiLU()
        )
    def forward(self, metas):
        B = len(metas)
        idx = torch.tensor([self.type2idx[m['type']] for m in metas], dtype=torch.long, device=self.emb.weight.device)
        emb = self.emb(idx)  # (B,16)
        cont = torch.tensor([[m['a'], m['jitter'], m['rot_angle']] for m in metas],
                            dtype=torch.float32, device=self.emb.weight.device)  # (B,3)
        return self.mlp(torch.cat([emb, cont], dim=-1))  # (B,out_dim)

# -----------------------------------------------------------
# Conditional Flow Model (SetTransformer head)
# -----------------------------------------------------------

class FlowSetTransformerCond(nn.Module):
    def __init__(self, d=3, st_kwargs=None,
                 t_dim=32, ctx_dim=64, meta_dim=32, geom_dim=None, k_local=8):
        super().__init__()
        self.d = d
        self.k_local = k_local
        self.time_emb = TimeEmbedding(t_dim)
        self.src_ctx  = EGNNContext(d=d, out_dim=ctx_dim)
        self.meta_enc = LatticeMetadataEncoder(out_dim=meta_dim)
        gdim = (2*d + 2) if geom_dim is None else geom_dim
        C_in = d + t_dim + ctx_dim + meta_dim + gdim
        st_kwargs = dict(st_kwargs or {})
        st_kwargs["dim_in"] = C_in
        st_kwargs["dim_out"] = d
        self.set_tf = SetTransformer(**st_kwargs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try: nn_utils.spectral_norm(m)
                except Exception: pass

    def forward(self, t, x_t, x_src, metas, context_dropout_p=0.15):
        B,d,N = x_t.shape
        x_t = x_t.float()
        t_emb  = self.time_emb(t).unsqueeze(-1).repeat(1,1,N)  # (B,t_dim,N)
        src_ctx= self.src_ctx(x_src).unsqueeze(-1).repeat(1,1,N)
        meta   = self.meta_enc(metas).unsqueeze(-1).repeat(1,1,N)
        if self.training and context_dropout_p > 0:
            mask = (torch.rand(B,1,1, device=x_t.device) > context_dropout_p).float()
            src_ctx = src_ctx * mask; meta = meta * mask
        geom = local_geom_features(x_t, k=self.k_local)
        x_centered = x_t - x_t.mean(dim=2, keepdim=True)
        inp = torch.cat([x_centered, t_emb, src_ctx, meta, geom], dim=1)
        return self.set_tf(inp)  # (B,d,N) velocity

# -----------------------------------------------------------
# EMA helper
# -----------------------------------------------------------

class EMAHelper:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

# -----------------------------------------------------------
# Dataset
# -----------------------------------------------------------

class SpherePackingDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)  # (num_samples, d, N)
        print(f"Loaded {path}, shape {self.data.shape}")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx].float()

# -----------------------------------------------------------
# Save utils
# -----------------------------------------------------------

def save_with_plot(model, optimizer, history, epoch, params, sec_dir):
    os.makedirs(sec_dir, exist_ok=True)
    loss = history[-1,2]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"flow_model_loss={loss:.6f}_{ts}.pth"
    path = os.path.join(sec_dir, name)
    data_load_save.save_model(path, model, optimizer, epoch, params)

    plt.figure(figsize=(12,6))
    plt.plot(history[:,0], label="FM (MSE)")
    plt.plot(history[:,1], label="Collision Penalty")
    plt.plot(history[:,2], label="Total Loss")
    plt.plot(history[:,3], label="GradNorm")
    plt.plot(history[:,4], label="Boundary Penalty")
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(sec_dir, f"flow_loss_{ts}.png"))
    plt.close()

# -----------------------------------------------------------
# Training (Conditional Flow Matching, lattice->good)
# -----------------------------------------------------------

def train_flow_model_optionB(
    model, optimizer, loader, num_epochs,
    sphere_radius_target,
    mse_strength, dist_strength,
    cube_min, cube_max,
    device, params, save_path,
    lattice_kind='mix', lattice_jitter=0.03, lattice_rotate=True,
    t_beta=(0.5,0.5), t_importance_weight=False,
    grad_clip=2.0,
    ema_decay=0.999,
    epsilon_sinkhorn=0.05, iters_sinkhorn=80,
    curriculum_ratio=0.8,
    # --- boundary penalty controls ---
    boundary_enable=True,
    boundary_margin=0.05, boundary_power=2.0,
    boundary_strength=1.0,
    boundary_warmup=True, boundary_schedule_center=0.5, boundary_schedule_sharpness=8.0,
    # --- schedules for collision penalty ---
    pen_schedule_center=0.5, pen_schedule_sharpness=8.0,
    # --- scheduler (optional) ---
    scheduler=None
):
    model.train().to(device)
    mse = nn.MSELoss(reduction='none')
    history = []
    ema = EMAHelper(model, decay=ema_decay)

    for epoch in tqdm(range(num_epochs), desc="Training (Option B + Boundary)"):
        ep_losses, gn_list, bnd_list = [], [], []
        # curriculum on radius
        r_curr = lerp(sphere_radius_target * curriculum_ratio, sphere_radius_target, epoch / max(1, num_epochs-1))
        # weights
        pen_w  = schedule_sigmoid(epoch, num_epochs, center=pen_schedule_center, sharpness=pen_schedule_sharpness)
        bnd_w  = schedule_sigmoid(epoch, num_epochs, center=boundary_schedule_center, sharpness=boundary_schedule_sharpness) if boundary_warmup else 1.0

        for x0 in loader:  # x0: (B,d,N)
            x0 = x0.to(device)
            B,d,N = x0.shape

            # Source (lattice) batch
            x_src, metas = sample_lattice_batch(
                batch_size=B, d=d, N=N, radius=r_curr,
                cube_min=cube_min, cube_max=cube_max, device=device,
                kind=lattice_kind if lattice_kind!='mix' else 'fcc',
                jitter=lattice_jitter, rotate=lattice_rotate, mix_kinds=(lattice_kind=='mix')
            )

            # Soft OT matching (align good→source); expect row-stochastic P
            P = sinkhorn_soft_matching(x_src, x0, epsilon=epsilon_sinkhorn, iters=iters_sinkhorn)  # (B,N,N)
            if not torch.isfinite(P).all():
                raise RuntimeError("NaNs/Infs in Sinkhorn matrix P")
            x0_matched = torch.bmm(x0, P.transpose(1,2))  # (B,d,N)

            # t sampling
            if t_beta is None:
                t = torch.rand(B, device=device)
                weight_t = torch.ones(B, device=device)
            else:
                a,b = t_beta
                beta_dist = torch.distributions.Beta(a, b)
                t = beta_dist.sample((B,)).to(device)
                # Clamp away from edges to stabilize
                t = t.clamp(1e-3, 1-1e-3)
                weight_t = (1.0 / torch.exp(beta_dist.log_prob(t)).clamp_min(1e-6)) if t_importance_weight else torch.ones(B, device=device)

            # straight path lattice->good (t=1 is lattice)
            x_t   = (1 - t).view(-1,1,1) * x0_matched + t.view(-1,1,1) * x_src
            u_star= (x_src - x0_matched)

            # predict velocity
            u_pred = model(t, x_t, x_src, metas)  # (B,d,N)

            # quick NaN guards
            for tens, name in [(x_t, "x_t"), (u_pred, "u_pred"), (x_src, "x_src"), (x0_matched, "x0_matched")]:
                if not torch.isfinite(tens).all():
                    raise RuntimeError(f"Non-finite values detected in {name}")

            # FM loss
            fm_per  = ((u_pred - u_star)**2).mean(dim=(1,2))
            fm_loss = (fm_per * weight_t).mean()

            # collision penalty on projected x0 estimate
            x0_pred = x_t - t.view(-1,1,1) * u_pred
            x0_proj = x0_pred.clamp(min=cube_min + r_curr, max=cube_max - r_curr)
            pen_col = distance_penalty(x0_proj, r_curr)

            # boundary velocity penalty near faces
            if boundary_enable:
                pen_bnd = boundary_velocity_penalty(
                    x_t, u_pred, r_curr, cube_min, cube_max,
                    margin=boundary_margin, power=boundary_power
                )
            else:
                pen_bnd = torch.tensor(0.0, device=device)

            loss = mse_strength * fm_loss \
                 + dist_strength * pen_w * pen_col \
                 + boundary_strength * bnd_w * pen_bnd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn_utils.clip_grad_norm_(model.parameters(), grad_clip) if grad_clip and grad_clip > 0 else torch.tensor(0.0, device=device)
            optimizer.step()
            ema.update(model)

            ep_losses.append([fm_loss.item(), pen_col.item(), loss.item()])
            gn_list.append(float(gnorm))
            bnd_list.append(float(pen_bnd))

        # Step LR scheduler per-epoch (if provided)
        if scheduler is not None:
            scheduler.step()

        avg = np.mean(ep_losses, axis=0)
        gavg= np.mean(gn_list)
        bavg= np.mean(bnd_list)
        history.append([avg[0], avg[1], avg[2], gavg, bavg])
        print(f"Epoch {epoch+1}/{num_epochs} | FM={avg[0]:.5f} ColPen={avg[1]:.5f} BndPen={bavg:.5f} Tot={avg[2]:.5f} | r={r_curr:.5f} pen_w={pen_w:.3f} bnd_w={bnd_w:.3f} grad={gavg:.3f}")

    hist = np.array(history)
    ema.apply_to(model)  # use EMA for sampling
    save_with_plot(model, optimizer, hist, num_epochs-1, params, save_path)
    return model, hist

# -----------------------------------------------------------
# Sampling (start from lattice; optional collision guidance)
# -----------------------------------------------------------

@torch.no_grad()
def sample_flow_model_optionB(
    model, num_samples, batch_size, num_points, device,
    sphere_radius, cube_min, cube_max, dim=3,
    lattice_kind='mix', lattice_jitter=0.02, lattice_rotate=True,
    guided_gamma=0.0, ode_atol=1e-6, ode_rtol=1e-6, k_local=8,
    ode_method='rk4', ode_n_steps=128
):
    model.eval()
    samples, meta_log = [], []

    def _sanitize_x(x):
        # Pull any NaNs/Infs back into the box interior
        x = torch.nan_to_num(
            x,
            nan=cube_min + sphere_radius,
            posinf=cube_max - sphere_radius,
            neginf=cube_min + sphere_radius,
        )
        return x.clamp(min=cube_min + sphere_radius, max=cube_max - sphere_radius)

    def _vf_core(t, x, x_src, metas, guided: bool):
        # Make sure time has same dtype as state
        t_scalar = torch.as_tensor(t, device=device, dtype=x.dtype)
        B = x.shape[0]
        t_batch = t_scalar.expand(B)  # (B,)

        # Evaluate model on a clamped (safe) state to avoid exploding cdist etc.
        x_safe = _sanitize_x(x)

        v = model(t_batch, x_safe, x_src, metas, context_dropout_p=0.0)
        if guided and guided_gamma > 0.0:
            g = grad_collision(x_safe, sphere_radius)
            v = v - guided_gamma * g

        # Last-ditch safety: scrub any non-finite velocity
        v = torch.nan_to_num(v)
        return v

    steps = int(math.ceil(num_samples / batch_size))
    for _ in tqdm(range(steps), desc="Sampling"):
        bs = min(batch_size, num_samples - len(samples))
        xT, metas = sample_lattice_batch(
            batch_size=bs, d=dim, N=num_points, radius=sphere_radius,
            cube_min=cube_min, cube_max=cube_max, device=device,
            kind=lattice_kind if lattice_kind!='mix' else 'fcc',
            jitter=lattice_jitter, rotate=lattice_rotate, mix_kinds=(lattice_kind=='mix')
        )

        # Start from a sanitized state just in case
        xT = _sanitize_x(xT)

        t_span = torch.tensor([1.0, 0.0], device=device, dtype=xT.dtype)
        guided = (guided_gamma > 0.0)

        def vf(tt, xx):  # closure that captures xT/metas
            return _vf_core(tt, xx, xT, metas, guided)

        if ode_method.lower() == 'rk4':
            step_size = 1.0 / float(max(1, ode_n_steps))
            out = odeint(vf, xT, t_span, method='rk4', options={'step_size': step_size})[-1]
        else:
            out = odeint(vf, xT, t_span, atol=ode_atol, rtol=ode_rtol)[-1]

        proj = _sanitize_x(out)

        # Keep going even if one batch had issues; log and repair instead of crashing
        if not torch.isfinite(proj).all():
            print("[WARN] Non-finite values after integration; sanitizing batch.")
            proj = _sanitize_x(proj)

        samples.append(proj.cpu().numpy())
        meta_log.extend(metas)

    X = np.concatenate(samples, axis=0)
    return X, meta_log

# -----------------------------------------------------------
# Main: read ALL params from cfg['Conditional Flow Matching']
# -----------------------------------------------------------

if __name__ == '__main__':
    set_seed(123)

    sec = cfg['Conditional Flow Matching']  # <--- NEW config section

    d      = int(sec['dimension'])
    bs     = int(sec['batch_size'])
    path   = sec['dataset_path']
    lr     = float(sec['learning_rate'])
    eta_min= float(sec.get('eta_min', 1e-5))
    epochs = int(sec['num_epochs'])
    radius = float(sec['sphere_radius'])
    mse_s  = float(sec['mse_strength'])
    pen_s  = float(sec['distance_penality_strength'])
    cube_min = float(sec.get('cube_min', 0.0))
    cube_max = float(sec.get('cube_max', 1.0))
    save_m = sec['save_model_path']
    save_g = sec['save_generated_path']
    pts    = int(sec['num_spheres'])
    num_new= int(sec.get('sample_new_points', 10))
    batch_n= int(sec.get('sample_new_points_batch_size', 10))
    weight_decay = float(sec.get('weight_decay', 0.0))
    max_train_samples = int(sec.get('max_train_samples', 5000))

    # Option-B extras
    lattice_kind     = sec.get('lattice_kind', 'mix')          # 'sc'|'bcc'|'fcc'|'hcp'|'mix'
    lattice_jitter   = float(sec.get('lattice_jitter', 0.03))
    lattice_rotate   = bool(int(sec.get('lattice_rotate', 1)))
    t_alpha          = float(sec.get('t_beta_alpha', 0.5))
    t_beta_param     = float(sec.get('t_beta_beta', 0.5))
    t_imp_weight     = bool(int(sec.get('t_importance_weight', 0)))
    grad_clip        = float(sec.get('grad_clip', 2.0))
    ema_decay        = float(sec.get('ema_decay', 0.999))
    sink_eps         = float(sec.get('sinkhorn_epsilon', 0.05))
    sink_iters       = int(sec.get('sinkhorn_iters', 80))
    curriculum_ratio = float(sec.get('curriculum_ratio', 0.8))
    k_local          = int(sec.get('k_local', 8))
    # schedules
    pen_center       = float(sec.get('pen_schedule_center', 0.5))
    pen_sharp        = float(sec.get('pen_schedule_sharpness', 8.0))

    # Boundary penalty params
    boundary_enable  = bool(int(sec.get('boundary_enable', 1)))
    boundary_margin  = float(sec.get('boundary_margin', 0.05))
    boundary_power   = float(sec.get('boundary_power', 2.0))
    boundary_strength= float(sec.get('boundary_penalty_strength', 1.0))
    boundary_warmup  = bool(int(sec.get('boundary_warmup', 1)))
    boundary_center  = float(sec.get('boundary_schedule_center', 0.5))
    boundary_sharp   = float(sec.get('boundary_schedule_sharpness', 8.0))

    # Sampling guidance
    guided_gamma     = float(sec.get('guided_gamma', 0.0))
    ode_atol         = float(sec.get('ode_atol', 1e-5))
    ode_rtol         = float(sec.get('ode_rtol', 1e-5))
    ode_method       = sec.get('ode_method', 'rk45')   # default old behavior
    ode_n_steps      = int(sec.get('ode_n_steps', 64))

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SpherePackingDataset(path)
    max_samples = min(len(dataset), max_train_samples)
    dataset = Subset(dataset, list(range(max_samples)))
    loader  = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    st_kwargs = {
        'dim_hidden': int(sec.get('st_dim_hidden', 128)),
        'num_heads':  int(sec.get('st_num_heads', 4)),
        'num_inds':   int(sec.get('st_num_inds', 16)),
        'num_isab':   int(sec.get('st_num_isab', 2)),
        'dim_out':    d
    }
    model = FlowSetTransformerCond(
        d=d, st_kwargs=st_kwargs,
        t_dim=int(sec.get('t_dim', 32)),
        ctx_dim=int(sec.get('ctx_dim', 64)),
        meta_dim=int(sec.get('meta_dim', 32)),
        k_local=k_local
    ).to(dev)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)

    print("param groups:", len(opt.param_groups))

    model, hist = train_flow_model_optionB(
        model, opt, loader, epochs,
        radius, mse_s, pen_s,
        cube_min, cube_max, dev,
        st_kwargs, save_m,
        lattice_kind=lattice_kind, lattice_jitter=lattice_jitter, lattice_rotate=lattice_rotate,
        t_beta=(t_alpha, t_beta_param), t_importance_weight=t_imp_weight,
        grad_clip=grad_clip, ema_decay=ema_decay,
        epsilon_sinkhorn=sink_eps, iters_sinkhorn=sink_iters,
        curriculum_ratio=curriculum_ratio,
        # boundary
        boundary_enable=boundary_enable,
        boundary_margin=boundary_margin, boundary_power=boundary_power,
        boundary_strength=boundary_strength,
        boundary_warmup=boundary_warmup,
        boundary_schedule_center=boundary_center, boundary_schedule_sharpness=boundary_sharp,
        # collision schedule
        pen_schedule_center=pen_center, pen_schedule_sharpness=pen_sharp,
        # scheduler
        scheduler=scheduler
    )

    X, metas = sample_flow_model_optionB(
        model, num_new, batch_n, pts, dev,
        radius, cube_min, cube_max, dim=d,
        lattice_kind=lattice_kind, lattice_jitter=lattice_jitter, lattice_rotate=lattice_rotate,
        guided_gamma=guided_gamma, ode_atol=ode_atol, ode_rtol=ode_rtol, k_local=k_local,
        ode_method=ode_method, ode_n_steps=ode_n_steps
    )
    os.makedirs(save_g, exist_ok=True)
    out_path = os.path.join(save_g, f"flow_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(torch.from_numpy(X), out_path)
    print(f"Saved {X.shape[0]} samples to {out_path}")
