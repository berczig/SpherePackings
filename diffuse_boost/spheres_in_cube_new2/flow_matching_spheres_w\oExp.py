import os
import math
import time
import copy
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt
from tqdm import tqdm

import schedulefree
from x_transformers import Encoder

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

from diffuse_boost.spheres_in_cube_new2.pipeline import PipelineState

from diffuse_boost.spheres_in_cube import data_load_save
from diffuse_boost import cfg  # assumes diffuse_boost.cfg is a ConfigParser

# ============================================================================
# FM path
# ============================================================================
FM_PATH = AffineProbPath(scheduler=CondOTScheduler())

# ============================================================================
# Model (unchanged)
# ============================================================================
class FlowSetTransformer(nn.Module):
    class TimeEmbedFourier(nn.Module):
        def __init__(
            self,
            out_dim: int,
            fourier_dim: int,
            hidden: int,
            sigma: float = 1.0,
            include_poly: bool = True,
        ):
            super().__init__()
            fourier_dim = max(16, fourier_dim)
            # fixed (non-trainable) Fourier frequencies
            self.register_buffer("freqs", torch.randn(fourier_dim) * sigma, persistent=False)
            self.include_poly = include_poly
            in_dim = 2 * fourier_dim
            if include_poly:
                in_dim += 4  # [t, t^2, t^3, log(1+t)]
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            # t: (B,) or (B,1) -> (B,)
            if t.ndim == 2:
                assert t.shape[1] == 1, f"t must be (B,) or (B,1); got {t.shape}"
                t = t[:, 0]
            t = t.contiguous()
            # Fourier features
            angles = (t[:, None] * self.freqs[None, :]) * (2.0 * math.pi)
            sin_feat = torch.sin(angles)
            cos_feat = torch.cos(angles)
            feats = [sin_feat, cos_feat]
            if self.include_poly:
                t1 = t
                t2 = t1 * t1
                t3 = t2 * t1
                tlog = torch.log1p(t1.clamp_min(0))  # stable near 0
                feats += [t1[:, None], t2[:, None], t3[:, None], tlog[:, None]]

            h = torch.cat(feats, dim=-1)
            return self.net(h)

    def __init__(self, d: int = 3, **st_kwargs):
        super().__init__()
        self.d = d
        model_dim = int(st_kwargs.get("dim_hidden", 512))
        heads = int(st_kwargs.get("num_heads", 8))
        depth = int(st_kwargs.get("num_isab", st_kwargs.get("depth", 6)))
        ff_mult = float(st_kwargs.get("ff_mult", 4.0))
        attn_do = float(st_kwargs.get("attn_dropout", 0.1))
        ff_do = float(st_kwargs.get("ff_dropout", 0.1))
        dim_time = int(st_kwargs.get("dim_time", max(64, 4 * d)))
        time_F = int(st_kwargs.get("time_fourier_dim", max(16, dim_time // 2)))
        time_hid = int(st_kwargs.get("time_hidden", 2 * dim_time))
        sigma = float(st_kwargs.get("time_fourier_sigma", 1.0))

        # conditioning setup
        cond_dim_in  = int(st_kwargs.get("cond_dim_in", 4))   # [r/L, N_scaled, p_face, minsep/L]
        cond_hidden  = int(st_kwargs.get("cond_hidden", max(64, dim_time)))
        self.uses_cond = cond_dim_in > 0

        # Time embedding (B,) / (B,1) -> (B, dim_time)
        self.time_emb = self.TimeEmbedFourier(
            out_dim=dim_time, fourier_dim=time_F, hidden=time_hid,
            sigma=sigma, include_poly=True
        )

        if self.uses_cond:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim_in, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, cond_hidden)
            )
            dim_cond = cond_hidden
        else:
            self.cond_mlp = None
            dim_cond = 0

        # per-point coords + time + (optional) cond
        self.token_in = nn.Linear(d + dim_time + dim_cond, model_dim)

        # FiLM generator takes [time ⊕ cond]
        self.film_in_dim = dim_time + dim_cond
        self.timecond_to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.film_in_dim, 2 * model_dim)
        )

        self.encoder = Encoder(
            dim = model_dim,
            depth=depth,
            heads=heads,
            layer_dropout= 0.1,
            attn_dropout = attn_do,
            ff_dropout = ff_do,
            use_rmsnorm=True,
            ff_glu=True,
            ff_no_bias=True,
            attn_flash=True
        )

        self.token_out = nn.Linear(model_dim, d)
        # Zero-init output to start near 0 velocity (stabilizes FM training)
        nn.init.zeros_(self.token_out.weight)
        nn.init.zeros_(self.token_out.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        t:    (B,) or (B,1)
        x:    (B, d, N)
        cond: (B, C) or None
        ->    (B, d, N)
        """
        B, d, N = x.shape
        assert d == self.d, f"Expected d={self.d}, got {d}"
        device = x.device
        t = t.to(device=device, dtype=x.dtype)
        if t.ndim == 2:
            assert t.shape[1] == 1, f"t must be (B,) or (B,1); got {t.shape}"
            t_1d = t[:, 0]
        else:
            t_1d = t
        t_embed = self.time_emb(t_1d)  # (B, dim_time)
        if self.uses_cond:
            if cond is None:
                # classifier-free guidance: treat None as zeros
                cond = torch.zeros(B, self.cond_mlp[0].in_features, device=device, dtype=x.dtype)
            else:
                cond = cond.to(device=device, dtype=x.dtype)
            c_embed = self.cond_mlp(cond)        # (B, dim_cond)
        else:
            c_embed = None

        tokens = x.permute(0, 2, 1).contiguous()            # (B, N, d)
        if c_embed is not None:
            tc = torch.cat([t_embed, c_embed], dim=-1)      # (B, dim_time+dim_cond)
            tc_rep = tc[:, None, :].expand(B, N, -1)        # (B, N, ...)
            h = torch.cat([tokens, tc_rep], dim=-1)         # (B, N, d + dim_time + dim_cond)
            film_in = tc                                     # (B, dim_time+dim_cond)
        else:
            t_rep = t_embed[:, None, :].expand(B, N, -1)    # (B, N, dim_time)
            h = torch.cat([tokens, t_rep], dim=-1)          # (B, N, d + dim_time)
            film_in = t_embed                                # (B, dim_time)
        h = self.token_in(h)                                 # (B, N, model_dim)
        # FiLM modulation from joint [time ⊕ cond]
        gamma_beta = self.timecond_to_film(film_in)          # (B, 2*model_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)            # (B, model_dim) each
        gamma = gamma[:, None, :]
        beta  = beta[:, None, :]
        h = h * (1 + gamma) + beta

        h = self.encoder(h)                                  # (B, N, model_dim)
        out = self.token_out(h).permute(0, 2, 1).contiguous()# (B, d, N)
        return out

# ============================================================================
# Geometry / loss utilities
# ============================================================================
_TRI_CACHE = {}

def distance_penalty(output, radius, margin=0.0, beta=10.0, p=2, q=0.05, eps=1e-12):
    """
    Penalty on near / actual overlaps.
    output: (B,d,N) predicted *centers*
    radius: sphere radius
    margin: extra safety margin (absolute, in same units as coords)
    q: fraction of top pair-violations to aggregate
    """
    B, d, N = output.shape
    coords = output.permute(0, 2, 1)                  # (B, N, d)
    dmat = torch.cdist(coords, coords) + eps
    r = torch.as_tensor(radius, device=output.device, dtype=output.dtype)
    m = torch.as_tensor(margin, device=output.device, dtype=output.dtype)
    gap = (2 * r + m) - dmat                          # >0 means violation / near-violation
    v = F.softplus(beta * gap) / beta                 # smooth hinge

    key = (output.device, N)
    tri = _TRI_CACHE.get(key)
    if tri is None:
        tri = torch.triu_indices(N, N, offset=1, device=output.device)
        _TRI_CACHE[key] = tri
    v = v[:, tri[0], tri[1]]                          # (B, pairs)

    if p != 1:
        v = v.pow(p)

    Pairs = v.size(1)
    k = max(1, int(q * Pairs))
    topk, _ = torch.topk(v, k=k, dim=1, largest=True, sorted=False)
    return topk.mean()


def _box_clamp(x, r, L):
    # Snap to [r, L-r] per coordinate
    return x.clamp(r, L - r)


def sample_t(B, device, small_t_weight=0.5, gamma=2.0):
    s = torch.rand(B, device=device)
    t_small = s**gamma
    use_small = (torch.rand(B, device=device) < small_t_weight).float()
    return use_small * t_small + (1 - use_small) * s


def _sample_x1_box_faces_per_batch(x0, r, L=1.0, p_face_batch=None, jitter=5e-3):
    """
    x0: (B,d,N) (only for shape/device)
    p_face_batch: (B,) per-sample prob a token sits on a face
    """
    B, d, N = x0.shape
    u = torch.rand_like(x0) * (L - 2*r) + r
    if p_face_batch is not None:
        mask = (torch.rand(B, N, device=x0.device) < p_face_batch[:, None])
        axis = torch.randint(0, d, (B, N), device=x0.device)
        side = torch.randint(0, 2, (B, N), device=x0.device)  # 0->r, 1->L-r
        face_val = torch.where(side == 0, torch.full((), r, device=x0.device),
                                         torch.full((), L - r, device=x0.device))
        ar = torch.arange(B, device=x0.device)[:, None].expand(B, N)
        nr = torch.arange(N, device=x0.device)[None, :].expand(B, N)
        u[ar, axis, nr] = torch.where(mask, face_val, u[ar, axis, nr])
    u = (u + jitter * torch.randn_like(u)).clamp(r, L - r)
    return u

# ============================================================================
# Dataset
# ============================================================================
class SpherePackingDataset(Dataset):
    def __init__(self, path, radius, box_len, tol=1e-4, chunk=256, scale_N=128):
        """
        path: torch file with tensor of shape (M, d, N)
        radius: sphere radius r
        box_len: cube side length L
        tol: face snapping tolerance for 'on-face' test
        chunk: chunk size for minsep precompute
        """
        data = torch.load(path)  # (M, d, N)
        assert data.ndim == 3, f"expected (M,d,N), got {tuple(data.shape)}"
        self.data = data.contiguous()
        self.M, self.d, self.N = self.data.shape
        self.r = float(radius)
        self.L = float(box_len)

        # p_face per sample
        near_r  = (self.data - self.r).abs() <= tol
        near_lr = (self.data - (self.L - self.r)).abs() <= tol
        on_face_any = (near_r | near_lr).any(dim=1)        # (M, N)
        p_face = on_face_any.float().mean(dim=1)           # (M,)

        # minsep per sample
        minsep = torch.empty(self.M, dtype=self.data.dtype)
        for s in range(0, self.M, chunk):
            e = min(self.M, s + chunk)
            xb = self.data[s:e]                            # (B, d, N)
            P  = xb.permute(0, 2, 1).contiguous()          # (B, N, d)
            D  = torch.cdist(P, P)                         # (B, N, N)
            Bn, Nn, _ = D.shape
            eye = torch.eye(Nn, dtype=torch.bool)[None].expand(Bn, -1, -1)
            D  = D.masked_fill(eye, float('inf'))
            # per-sample min pair distance = min over all pairs
            minsep[s:e] = D.amin(dim=-1).amin(dim=-1).cpu()
        
        self.minsep = minsep.clone()   # shape (M,)

        # cond vector: [r/L, N/scale_N, p_face, minsep/L]
        r_over_L = torch.full((self.M,), self.r / self.L, dtype=self.data.dtype)
        N_scaled = torch.full((self.M,), float(self.N) / float(scale_N), dtype=self.data.dtype)
        minsep_L = (minsep / self.L).to(self.data.dtype)

        self.cond = torch.stack([r_over_L, N_scaled, p_face.to(self.data.dtype), minsep_L], dim=1)  # (M, 4)

        print(f"[Load] Loaded tensors from '{path}', shape {self.data.shape} | precomputed conds: {tuple(self.cond.shape)}")

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        return self.data[idx], self.cond[idx]

# ============================================================================
# Reward-Guided CFM
# ============================================================================
class RGCFMTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        sphere_radius: float,
        clip_range: float,
        num_points: int,
        dim: int = 3,
        cond_loader=None,
    ):
        self.device = device
        self.dim = dim
        self.sphere_radius = float(sphere_radius)
        self.clip_range = float(clip_range)
        self.num_points = int(num_points)
        self.cond_loader = cond_loader
        self._cond_iter = iter(cond_loader) if cond_loader is not None else None

        # Models
        self.net_model = model.to(device)
        self.ref_model = copy.deepcopy(model).to(device)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        # Optimizer
        lr = float(config.get("learning_rate", 1e-4))
        self.optimizer = schedulefree.RAdamScheduleFree(self.net_model.parameters(), lr=lr)

        # Hyperparameters
        self.grad_clip = float(config.get("grad_clip", 1.0))
        self.temperature = float(config.get("temperature", 1.0))
        self.w2_coefficient = float(config.get("w2_coefficient", 0.1))
        self.batch_size = int(config.get("batch_size", 16))
        self.weight_clip = float(config.get("weight_clip", 1e6))
        self.small_t_weight = float(config.get("small_t_weight", 0.5))
        self.small_t_gamma = float(config.get("small_t_gamma", 2.0))

        # Sampler knobs (reuse PCFM defaults where possible)
        self.pcfm_steps = int(config.get("pcfm_steps", 40))
        self.ode_method = config.get("ode_method", "midpoint")
        self.ode_step_cap = float(config.get("ode_step_cap", 0.05))
        self.jitter = float(config.get("pcfm_jitter", 5e-3))
        self.eps_nrm = float(config.get("pcfm_eps_nrm", 1e-6))
        self.proj_outer_iters = int(config.get("proj_outer_iters", 8))
        self.alpha_proj = float(config.get("alpha_proj", 0.25))
        self.contact_q = float(config.get("contact_q", 1.0))
        self.wall_weight = float(config.get("wall_weight", 1.0))
        self.wall_margin = float(config.get("wall_margin", 0.05))
        self.prox_iters = int(config.get("prox_iters", 10))
        self.prox_step = float(config.get("prox_step", 0.1))
        self.prox_lambda = float(config.get("prox_lambda", 2.0))
        self.final_passes = int(config.get("final_passes", 6))
        self.tol_finish = float(config.get("tol_finish", 1e-8))

        self.history = []
        self.global_step = 0

    def load_pretrained(self, path: str):
        """Load weights into net_model only; ref_model stays frozen."""
        assert path and os.path.isfile(path), f"Reference path '{path}' is invalid."
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if isinstance(sd, dict) and len(sd) and next(iter(sd)).startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}
        self.net_model.load_state_dict(sd, strict=True)
        self.ref_model.load_state_dict(sd)
        self.ref_model.eval()

    @torch.no_grad()
    def sample_batch(self, ep: int):
        """Sample x1 from the current sampling policy and compute rewards."""
        samples_np, cond_batches = sample_flow_model(
            self.net_model,
            self.optimizer,
            self.batch_size,
            self.batch_size,
            self.num_points,
            self.device,
            self.sphere_radius,
            0.0,
            self.clip_range,
            self.dim,
            cond_loader=self.cond_loader,
            n_steps=self.pcfm_steps,
            ode_method=self.ode_method,
            ode_step_cap=self.ode_step_cap,
            jitter=self.jitter,
            eps_nrm=self.eps_nrm,
            proj_outer_iters=self.proj_outer_iters,
            alpha_proj=self.alpha_proj,
            contact_q=self.contact_q,
            wall_weight=self.wall_weight,
            wall_margin=self.wall_margin,
            prox_iters=self.prox_iters,
            prox_step=self.prox_step,
            prox_lambda=self.prox_lambda,
            final_passes=self.final_passes,
            tol_finish=self.tol_finish,
            return_cond=True,
        )
        x1 = torch.from_numpy(samples_np).to(self.device, dtype=torch.float32)
        cond_used = None
        if cond_batches is not None and cond_batches.numel() > 0:
            cond_used = cond_batches.to(self.device)
            if cond_used.size(0) > x1.size(0):
                cond_used = cond_used[:x1.size(0)]

        # Reward: min pairwise distance / L
        P = x1.permute(0, 2, 1).contiguous()  # (B, N, d)
        dmat = torch.cdist(P, P)
        eye = torch.eye(self.num_points, device=self.device, dtype=torch.bool)[None]
        dmat = dmat.masked_fill(eye, float('inf'))
        minsep = dmat.amin(dim=-1).amin(dim=-1)
        rewards = (minsep / self.clip_range).detach()
        return x1, rewards, cond_used

    def compute_loss(self, x_data: torch.Tensor, rewards: torch.Tensor, cond: torch.Tensor = None):
        """
        x_data: policy sample treated as 'data' endpoint (x0 in FM_PATH).
        Prior sample is generated as x1, matching pretraining semantics (data -> prior).
        """
        B, d, N = x_data.shape
        assert d == self.dim and N == self.num_points, "Shape mismatch for x_data"
        if cond is None and self.net_model.uses_cond:
            cond = torch.zeros(B, self.net_model.cond_mlp[0].in_features, device=self.device, dtype=x_data.dtype)
        elif cond is not None:
            cond = cond.to(self.device, dtype=x_data.dtype)

        x_prior = _sample_x1_box_faces_per_batch(
            torch.empty(B, self.dim, self.num_points, device=self.device),
            self.sphere_radius,
            L=self.clip_range,
            p_face_batch=cond[:, 2].clamp(0, 1),
            jitter=self.jitter
        )
        t = sample_t(B, device=self.device, small_t_weight=self.small_t_weight, gamma=self.small_t_gamma)

        # Keep path semantics consistent with supervised FM: x0=data-like, x1=prior
        path_sample = FM_PATH.sample(t=t, x_0=x_data, x_1=x_prior)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        x_t_clamped = _box_clamp(x_t, self.sphere_radius, self.clip_range)

        v_ft = self.net_model(t, x_t_clamped, cond=cond)
        with torch.no_grad():
            v_ref = self.ref_model(t, x_t_clamped, cond=cond)

        fm_loss_per_sample = (v_ft - u_t).pow(2).mean(dim=(1, 2))

        # Reward weights
        r_mean = rewards.mean()
        r_std = rewards.std().clamp_min(1e-8)
        r_norm = (rewards - r_mean) / r_std
        w = torch.exp(self.temperature * r_norm).clamp(max=self.weight_clip)
        w = w / (w.mean().detach() + 1e-8)

        L_fm = (w * fm_loss_per_sample).mean()
        w2_per_sample = (v_ft - v_ref).pow(2).mean(dim=(1, 2))
        L_w2 = w2_per_sample.mean()
        loss = L_fm + self.w2_coefficient * L_w2

        metrics = {
            "fm_loss": fm_loss_per_sample.mean().item(),
            "fm_loss_weighted": L_fm.item(),
            "w2_loss": L_w2.item(),
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "reward_max": rewards.max().item(),
            "weight_mean": w.mean().item(),
        }
        return loss, metrics

    def train(self, num_epochs: int, steps_per_epoch: int):
        hist = []
        for ep in range(num_epochs):
            ep_metrics = []
            for _ in range(steps_per_epoch):
                x1_batch, rewards, cond_used = self.sample_batch(ep)
                # sample_flow_model sets model/optimizer to eval; switch back for training
                self.net_model.train()
                self.optimizer.train()
                loss, metrics = self.compute_loss(x1_batch, rewards, cond=cond_used)

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.net_model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.global_step += 1

                ep_metrics.append(metrics)

            # Epoch averages
            if ep_metrics:
                avg = {k: float(np.mean([m[k] for m in ep_metrics])) for k in ep_metrics[0].keys()}
                hist.append([avg["fm_loss"], avg["fm_loss_weighted"], avg["w2_loss"], avg["loss"], avg["reward_mean"]])
                print(f"[RG-CFM] Epoch {ep+1}/{num_epochs} | FM={avg['fm_loss']:.4f} FMw={avg['fm_loss_weighted']:.4f} W2={avg['w2_loss']:.4f} Loss={avg['loss']:.4f} R={avg['reward_mean']:.4f}")
        self.history = np.array(hist, dtype=np.float32)
        self.net_model.eval()
        self.optimizer.eval()
        return self.history

# ============================================================================
# Training
# ============================================================================
def train_flow_model(
    model,
    optimizer,
    loader,
    num_epochs,
    sphere_radius,
    mse_strength,
    dist_strength,
    clip_max,
    device,
    params,
    save_model_dir,
    # training-time knobs for auxiliary penalty
    aux_margin_factor=0.02,
    aux_beta=80.0,
    aux_q=0.20,
    small_t_weight=0.5,
    small_t_gamma=2.0,
):
    model.train().to(device)
    optimizer.train()
    mse = nn.MSELoss()
    history = []

    os.makedirs(save_model_dir, exist_ok=True)

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        ep_losses=[]
        # cosine warmup of penalty
        ratio = 0.5 * (1 - math.cos(math.pi * min(1.0, epoch / max(1, int(0.5 * num_epochs)))))

        for x_0, cond in loader:
            x_0 = x_0.to(device)
            cond = cond.to(device)
            B = x_0.size(0)
            cond_in = cond  # could do CFG here if desired

            # source x_1 uses per-sample p_face from cond
            p_face_batch = cond[:, 2].clamp(0, 1)     # (B,)
            x_1 = _sample_x1_box_faces_per_batch(
                x_0, sphere_radius, L=clip_max,
                p_face_batch=p_face_batch, jitter=5e-3
            )

            t_in = sample_t(B, device=device, small_t_weight=small_t_weight, gamma=small_t_gamma)  # (B,)
            path_sample = FM_PATH.sample(t=t_in, x_0=x_0, x_1=x_1)
            x_t  = path_sample.x_t
            dx_t = path_sample.dx_t         # target velocity along the path

            u_pred = model(t_in, _box_clamp(x_t, sphere_radius, clip_max), cond=cond_in)

            # Flow-matching loss
            diff = u_pred - dx_t
            fm_loss_per_sample = diff.pow(2).mean(dim=(1, 2))
            loss_fm = fm_loss_per_sample.mean()

            # Penalty on projected x0 from velocity
            eps_t = 1e-3
            with torch.no_grad():
                sch0 = FM_PATH.scheduler(t_in)  # t_in: (B,)
                sch1 = FM_PATH.scheduler((t_in + eps_t).clamp_max(1.0))
            alpha_dot = ((sch1.alpha_t - sch0.alpha_t) / eps_t).view(-1, 1, 1).to(x_t.dtype)
            sigma_dot = ((sch1.sigma_t - sch0.sigma_t) / eps_t).view(-1, 1, 1).to(x_t.dtype)
            alpha_dot_safe = alpha_dot.sign() * alpha_dot.abs().clamp_min(1e-6)
            x0_proj = (u_pred - sigma_dot * x_1) / alpha_dot_safe

            pen = distance_penalty(
                x0_proj, sphere_radius,
                margin=aux_margin_factor * sphere_radius,
                beta=aux_beta,
                p=2,
                q=aux_q
            )

            loss = mse_strength * loss_fm + dist_strength * ratio * pen

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_losses.append([
                loss_fm.item(),
                pen.item(),
                loss.item()
            ])

        avg = np.mean(ep_losses, axis=0)
        history.append(avg)
        pbar.set_postfix({
            "FM": f"{avg[0]:.4f}",
            "Pen": f"{avg[1]:.4f}",
            "Tot": f"{avg[2]:.4f}"
        })

    hist = np.array(history, dtype=np.float32)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"spheres_fm_loss={hist[-1,-1]:.6f}_{ts}.pth"
    path = os.path.join(save_model_dir, name)

    # save checkpoint similar to Heilbronn style
    torch.save(
        {
            "state_dict": model.state_dict(),
            "opt": optimizer.state_dict(),
            "epochs": num_epochs,
            "params": params,
            "history": hist,
        },
        path
    )
    print(f"[Train] Saved model to: '{path}'")

    # also save a loss plot
    plt.figure(figsize=(12,6))
    hist_np = hist
    labels = ["Flow MSE Loss", "Distance Penalty", "Total Loss"]
    for idx in range(hist_np.shape[1]):
        plt.plot(hist_np[:, idx], label=labels[idx] if idx < len(labels) else f"Loss{idx}")
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_model_dir, f"spheres_fm_loss_{ts}.png"))
    plt.close()

    return model, hist, path

# ============================================================================
# PCFM-style sampler (BEST version), with config-controllable knobs
# ============================================================================
@torch.no_grad()
def sample_flow_model(
    model,
    optimizer,
    num_samples,
    batch_size,
    num_points,
    device,
    sphere_radius,
    clip_min,
    clip_max,
    dim,
    cond_loader=None,
    # PCFM hyperparameters
    n_steps=40,
    ode_method='midpoint',
    ode_step_cap=0.05,
    jitter=5e-3,
    eps_nrm=1e-6,
    proj_outer_iters=8,
    alpha_proj=0.25,
    contact_q=1.0,
    wall_weight=1.0,
    wall_margin=0.05,
    prox_iters=10,
    prox_step=0.1,
    prox_lambda=2.0,
    final_passes=6,
    tol_finish=1e-8,
    return_cond=False,
):
    """
    Physics-Constrained Flow Matching (PCFM) for hard spheres (radius r) in a box [r, L-r]^d.
    Returns:
        np.ndarray of shape (num_samples, d, N)
    """

    L = float(clip_max)
    r = float(sphere_radius)

    def _clamp_box(x):
        return x.clamp(r, L - r)

    # model expects t \in [1->0]
    class _FMVF:
        def __init__(self, mdl, cond):
            self.mdl, self.cond = mdl, cond
        def __call__(self, x, t, **_):
            tb = torch.full((x.size(0),), float(t), device=x.device, dtype=x.dtype)
            return self.mdl(tb, x, cond=self.cond) if self.cond is not None else self.mdl(tb, x)

    def _ode_solve_with_model(x_init, tau_start, tau_end, cond):
        """ODESolve with learned field from tau_start to tau_end, mapped to model time t=1-tau."""
        t0, t1 = 1.0 - float(tau_start), 1.0 - float(tau_end)   # decreasing t as tau increases
        solver = ODESolver(velocity_model=_FMVF(model, cond))
        T = torch.tensor([t0, t1], device=x_init.device, dtype=x_init.dtype)
        dt = abs(t1 - t0)
        step_size = min(ode_step_cap, max(1e-3, dt))
        x_end = solver.sample(time_grid=T, x_init=x_init, method=ode_method,
                              step_size=step_size, return_intermediates=False, enable_grad=False)
        return _clamp_box(x_end)

    def _active_overlap(x, for_stop=False):
        P = x.permute(0, 2, 1).contiguous()         # (B,N,d)
        if not for_stop and eps_nrm > 0.0:
            P = P + (eps_nrm * r) * torch.randn_like(P)

        diff = P[:, :, None, :] - P[:, None, :, :]  # (B,N,N,d)
        dist = diff.norm(dim=-1).clamp_min(1e-12)   # (B,N,N)
        n    = diff / dist.unsqueeze(-1)            # (B,N,N,d)

        overlap = (2.0 * r - dist)                  # positive => penetrating
        B, N, _ = overlap.shape
        eye = torch.eye(N, device=x.device, dtype=torch.bool)[None]
        overlap = overlap.masked_fill(eye, 0.0)
        return overlap, n, eye

    def _select_active(overlap, eye, q=1.0):
        pos = (overlap > 0) & (~eye)
        if q >= 1.0:
            return pos
        flat = overlap.clone()
        flat[~pos] = float('-inf')
        thr = torch.quantile(flat.view(flat.size(0), -1), 1.0 - q, dim=1, keepdim=True)
        return pos & (overlap >= thr.view(-1, 1, 1))

    def _JJt_inv_h_times_Jt(overlap, n, eye, q):
        active = _select_active(overlap, eye, q)
        w = (0.5 * overlap * active.float())                    # (B,N,N)
        term_i = (w.unsqueeze(-1) * n).sum(dim=2)               # (B,N,d)
        term_j = (w.transpose(1, 2).unsqueeze(-1)
                  * n.transpose(1, 2)).sum(dim=2)               # (B,N,d)
        delta = term_i - term_j                                 # (B,N,d)
        deg = active.float().sum(dim=2, keepdim=True).clamp_min(1.0)
        delta = (delta / deg).permute(0, 2, 1)                  # (B,d,N)
        return delta

    def _wall_push(u, margin_frac=0.05, scale=1.0):
        """Soft wall ghost constraints"""
        B, d_, N = u.shape
        delta = torch.zeros_like(u)
        thr = margin_frac * (2.0 * r)
        for ax in range(d_):
            dl = u[:, ax, :] - r              # distance to low face along axis ax
            dh = (L - r) - u[:, ax, :]        # distance to high face
            near_low  = (dl < thr)
            near_high = (dh < thr)
            delta[:, ax, :] += torch.where(near_low,  (thr - dl), 0.0)
            delta[:, ax, :] -= torch.where(near_high, (thr - dh), 0.0)
        return scale * delta

    def _project_terminal(u):
        for _ in range(proj_outer_iters):
            overlap, n, eye = _active_overlap(u, for_stop=False)
            has_pairs = bool((overlap > 0).any())
            if not has_pairs and wall_weight <= 0.0:
                break
            delta_pairs = _JJt_inv_h_times_Jt(overlap, n, eye, contact_q) if has_pairs else 0.0
            delta_walls = _wall_push(u, margin_frac=wall_margin, scale=wall_weight) if wall_weight > 0.0 else 0.0
            u = _clamp_box(u + alpha_proj * (delta_pairs + delta_walls))
        return u

    def _Jt_h_at_state(u_next):
        """
        Fast Approximatation J^T h(u_next): pairwise + wall residual gradients, shape (B,d,N).
        """
        overlap, n, eye = _active_overlap(u_next, for_stop=False)
        active = (overlap > 0) & (~eye)
        # pair contribution (same as GN but without 0.5 and without degree normalization)
        w = overlap * active.float()
        term_i = (-(w.unsqueeze(-1) * n).sum(dim=2))                       # (B,N,d)
        term_j = (+(w.transpose(1, 2).unsqueeze(-1) * n.transpose(1, 2)).sum(dim=2))
        jt_pairs = (term_i + term_j).permute(0, 2, 1)                      # (B,d,N)
        # wall contribution
        jt_walls = _wall_push(u_next, margin_frac=wall_margin, scale=1.0)  # inward normal
        return jt_pairs + wall_weight * jt_walls

    def _prox_relaxed(u, u0, u_proj, tau_prime, cond):
        # gradient steps
        u_hat = (1.0 - tau_prime) * u0 + tau_prime * u_proj
        t_prime = 1.0 - float(tau_prime)
        for _ in range(max(1, prox_iters)):
            # evaluate v_theta at tau'
            tb = torch.full((u.size(0),), t_prime, device=u.device, dtype=u.dtype)
            v = model(tb, u, cond=cond) if cond is not None else model(tb, u)
            u_next = _clamp_box(u + (1.0 - tau_prime) * v)

            jt_h = _Jt_h_at_state(u_next)
            grad = (u - u_hat) + prox_lambda * jt_h
            u = _clamp_box(u - prox_step * grad)
        return u

    def _final_polish(u):
        for _ in range(final_passes):
            overlap, _, _ = _active_overlap(u, for_stop=True)
            if float(overlap.clamp_min(0.0).amax()) <= tol_finish:
                break
            u = _project_terminal(u)
        return u

    model.eval()
    try:
        optimizer.eval()
    except Exception:
        pass

    samples, remaining = [], int(num_samples)
    cond_collected = [] if return_cond else None
    cond_iter = iter(cond_loader) if cond_loader is not None else None

    pbar = tqdm(total=remaining, desc="Sampling")
    while remaining > 0:
        if cond_iter is not None:
            try:
                x_dummy, cond = next(cond_iter)
            except StopIteration:
                cond_iter = iter(cond_loader)
                x_dummy, cond = next(cond_iter)
            cond = cond.to(device)
            # *** NEW: batch size must respect cond.size(0) ***
            bs = min(batch_size, remaining, cond.size(0))
            cond = cond[:bs]
            if return_cond:
                cond_collected.append(cond.detach().cpu())
        else:
            bs = min(batch_size, remaining)
            cond = None

        # Initialize u0 at tau=0 (x1 in standard FM), then clamp inside box.
        p_face_batch = cond[:, 2].clamp(0, 1) if cond is not None else None
        #COMMENT: We can improve the prior by more parametric u0 proposal to improve diversity
        u0 = _sample_x1_box_faces_per_batch(
            torch.empty(bs, dim, num_points, device=device),
            r, L=L, p_face_batch=p_face_batch, jitter=jitter
        )
        u0 = _clamp_box(u0)
        u = u0.clone()

        # tau grid
        for k in range(n_steps):
            tau      = k / n_steps
            tau_next = (k + 1) / n_steps
            # Forward shoot to tau=1 with learned field
            u1 = _ode_solve_with_model(u, tau, 1.0, cond)
            # Terminal projection Π_H
            u_proj = _project_terminal(u1)
            # Reverse OT
            u = _prox_relaxed(u, u0, u_proj, tau_next, cond)

        # final polishing
        u = _final_polish(u)
        samples.append(u.cpu().numpy())
        remaining -= bs

        # Update Progess bar
        pbar.update(bs)

    if return_cond:
        cond_tensor = torch.cat(cond_collected, dim=0) if cond_collected else None
        return np.concatenate(samples, axis=0), cond_tensor

    return np.concatenate(samples, axis=0)

# ============================================================================
# I/O helper
# ============================================================================
def load_model_if_exists(model, opt, path, device):
    if path and os.path.isfile(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if isinstance(sd, dict) and len(sd) and next(iter(sd)).startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        try:
            if "opt" in ckpt:
                opt.load_state_dict(ckpt["opt"])
        except Exception:
            pass
        print(f"[Load] Loaded model from '{path}'")
    else:
        print(f"[Load] No model found at '{path}'")
    return model, opt


def rg_cfm_main(state: PipelineState = None):
    sec = "flow_matching"
    # Geometry / model dims
    d = cfg.getint(sec, "dimension", fallback=3)
    num_spheres = cfg.getint(sec, "num_spheres", fallback=None)
    sphere_radius = cfg.getfloat(sec, "sphere_radius", fallback=0.05)
    clip_range = cfg.getfloat(sec, "clip_sample_range", fallback=1.0)
    dataset_path = cfg.get(sec, "dataset_path", fallback="").strip()
    assert dataset_path, "dataset_path required for RG-CFM (conditioning loader)."

    save_model_dir = cfg.get(sec, "save_model_dir", fallback="./outputs_spheres_models")
    stamp = datetime.now().strftime("%Y-%m-%d")
    save_model_dir = os.path.join(save_model_dir, stamp)
    os.makedirs(save_model_dir, exist_ok=True)

    # RG-CFM hyperparameters
    rg_lr = cfg.getfloat(sec, "rg_learning_rate", fallback=1e-4)
    rg_w2 = cfg.getfloat(sec, "rg_w2_coefficient", fallback=1.0)
    rg_tau = cfg.getfloat(sec, "rg_temperature", fallback=0.5)
    rg_grad_clip = cfg.getfloat(sec, "rg_grad_clip", fallback=1.0)
    rg_batch = cfg.getint(sec, "rg_batch_size", fallback=32)
    rg_steps_per_epoch = cfg.getint(sec, "rg_steps_per_epoch", fallback=100)
    rg_epochs = cfg.getint(sec, "rg_epochs", fallback=1)
    rg_weight_clip = cfg.getfloat(sec, "rg_weight_clip", fallback=100.0)
    rg_ref_path = cfg.get(sec, "rg_ref_path", fallback="").strip()
    if not rg_ref_path:
        rg_ref_path = cfg.get(sec, "resume_model_path", fallback="").strip()
    assert rg_ref_path, "rg_ref_path (or resume_model_path) must be set for RG-CFM."

    # Time sampling knobs
    small_t_weight = cfg.getfloat(sec, "small_t_weight", fallback=0.5)
    small_t_gamma = cfg.getfloat(sec, "small_t_gamma", fallback=2.0)

    # Sampler knobs (reuse PCFM settings)
    pcfm_steps = cfg.getint(sec, "pcfm_steps", fallback=40)
    ode_method = cfg.get(sec, "ode_method", fallback="midpoint")
    ode_step_cap = cfg.getfloat(sec, "ode_step_cap", fallback=0.05)
    jitter = cfg.getfloat(sec, "pcfm_jitter", fallback=5e-3)
    eps_nrm = cfg.getfloat(sec, "pcfm_eps_nrm", fallback=1e-6)
    proj_outer_iters = cfg.getint(sec, "proj_outer_iters", fallback=8)
    alpha_proj = cfg.getfloat(sec, "alpha_proj", fallback=0.25)
    contact_q = cfg.getfloat(sec, "contact_q", fallback=1.0)
    wall_weight = cfg.getfloat(sec, "wall_weight", fallback=1.0)
    wall_margin = cfg.getfloat(sec, "wall_margin", fallback=0.05)
    prox_iters = cfg.getint(sec, "prox_iters", fallback=10)
    prox_step = cfg.getfloat(sec, "prox_step", fallback=0.1)
    prox_lambda = cfg.getfloat(sec, "prox_lambda", fallback=2.0)
    final_passes = cfg.getint(sec, "final_passes", fallback=6)
    tol_finish = cfg.getfloat(sec, "tol_finish", fallback=1e-8)
    face_tol   = cfg.getfloat(sec, "face_tolerance", fallback=1e-4)
    minsep_chunk = cfg.getint(sec, "minsep_chunk", fallback=256)
    scale_N    = cfg.getint(sec, "scale_N", fallback=128)

    # Architecture
    st_kwargs = {
        'dim_hidden':     cfg.getint(sec, "st_dim_hidden", fallback=512),
        'num_heads':      cfg.getint(sec, "st_num_heads", fallback=8),
        'depth':          cfg.getint(sec, "st_depth", fallback=6),
        'attn_dropout':   cfg.getfloat(sec, "st_attn_dropout", fallback=0.1),
        'ff_dropout':     cfg.getfloat(sec, "st_ff_dropout", fallback=0.1),
        'dim_time':       cfg.getint(sec, "st_dim_time", fallback=max(64, 4*d)),
        'time_fourier_dim': cfg.getint(sec, "time_fourier_dim", fallback=max(16, (max(64,4*d))//2)),
        'time_hidden':    cfg.getint(sec, "time_hidden", fallback=2*max(64,4*d)),
        'time_fourier_sigma': cfg.getfloat(sec, "time_fourier_sigma", fallback=1.0),
        'cond_dim_in':    cfg.getint(sec, "cond_dim_in", fallback=4),
        'cond_hidden':    cfg.getint(sec, "cond_hidden", fallback=max(64, 4*d)),
    }
    st_kwargs["dim_out"] = d

    if num_spheres is None:
        raise ValueError("num_spheres must be set in config for RG-CFM.")
    # Conditioning loader from dataset (reuse sphere packing dataset)
    full_ds = SpherePackingDataset(
        dataset_path,
        radius=sphere_radius,
        box_len=clip_range,
        tol=face_tol,
        chunk=minsep_chunk,
        scale_N=scale_N
    )
    assert full_ds.N == num_spheres, f"num_spheres={num_spheres} but dataset N={full_ds.N}"
    cond_loader = DataLoader(full_ds, batch_size=rg_batch, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FlowSetTransformer(d, **st_kwargs).to(device)

    trainer_cfg = {
        "learning_rate": rg_lr,
        "w2_coefficient": rg_w2,
        "temperature": rg_tau,
        "grad_clip": rg_grad_clip,
        "batch_size": rg_batch,
        "weight_clip": rg_weight_clip,
        "small_t_weight": small_t_weight,
        "small_t_gamma": small_t_gamma,
        "pcfm_steps": pcfm_steps,
        "ode_method": ode_method,
        "ode_step_cap": ode_step_cap,
        "pcfm_jitter": jitter,
        "pcfm_eps_nrm": eps_nrm,
        "proj_outer_iters": proj_outer_iters,
        "alpha_proj": alpha_proj,
        "contact_q": contact_q,
        "wall_weight": wall_weight,
        "wall_margin": wall_margin,
        "prox_iters": prox_iters,
        "prox_step": prox_step,
        "prox_lambda": prox_lambda,
        "final_passes": final_passes,
        "tol_finish": tol_finish,
    }

    trainer = RGCFMTrainer(
        model=model,
        config=trainer_cfg,
        device=device,
        sphere_radius=sphere_radius,
        clip_range=clip_range,
        num_points=num_spheres,
        dim=d,
        cond_loader=cond_loader,
    )
    trainer.load_pretrained(rg_ref_path)
    hist = trainer.train(num_epochs=rg_epochs, steps_per_epoch=rg_steps_per_epoch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if hist.size > 0:
        name = f"spheres_rgcfm_loss={hist[-1,3]:.6f}_{ts}.pth"
    else:
        name = f"spheres_rgcfm_{ts}.pth"
    path = os.path.join(save_model_dir, name)

    torch.save(
        {
            "state_dict": trainer.net_model.state_dict(),
            "opt": trainer.optimizer.state_dict(),
            "epochs": rg_epochs,
            "params": st_kwargs,
            "history": hist,
        },
        path
    )
    print(f"[RG-CFM] Saved fine-tuned model to: '{path}'")
    if state:
        state.set_model_path(path)
    return path

# ============================================================================
# Main controlled by INI (flow_matching section)
# ============================================================================
def main(state:PipelineState=None):
    sec = "flow_matching"
    mode = cfg.get(sec, "mode", fallback="training_and_sampling").strip()

    if mode == "rg_cfm":
        rg_cfm_main(state=state)
        return

    # Data / IO
    dataset_path = cfg.get(sec, "dataset_path")
    assert dataset_path, "dataset_path required (torch tensor file of shape (M,d,N))"

    stamp      = datetime.now().strftime("%Y-%m-%d")

    save_model_dir = cfg.get(sec, "save_model_dir", fallback="./outputs_spheres_models")
    save_model_dir = os.path.join(save_model_dir, stamp)
    os.makedirs(save_model_dir, exist_ok=True)

    save_generated_dir = cfg.get(sec, "save_generated_dir", fallback="./outputs_spheres_samples")
    save_generated_dir = os.path.join(save_generated_dir, stamp)
    os.makedirs(save_generated_dir, exist_ok=True)

    # Model + train params
    d = cfg.getint(sec, "dimension", fallback=3)
    bs = cfg.getint(sec, "batch_size", fallback=64)
    lr = cfg.getfloat(sec, "learning_rate", fallback=1e-4)
    epochs = cfg.getint(sec, "num_epochs", fallback=200)
    mse_s  = cfg.getfloat(sec, "mse_strength", fallback=1.0)
    pen_s  = cfg.getfloat(sec, "distance_penalty_strength", fallback=0.2)
    sphere_radius = cfg.getfloat(sec, "sphere_radius", fallback=0.05)
    clip_range = cfg.getfloat(sec, "clip_sample_range", fallback=1.0)
    test_fraction = cfg.getfloat(sec, "test_fraction", fallback=0.1)
    split_seed = cfg.getint(sec, "split_seed", fallback=1234)
    max_samples = cfg.getint(sec, "max_samples", fallback=25000)

    train_top_fraction = cfg.getfloat(sec, "train_top_fraction", fallback=1.0)

    # Training penalty knobs
    aux_margin_factor = cfg.getfloat(sec, "aux_margin_factor", fallback=0.02)
    aux_beta          = cfg.getfloat(sec, "aux_beta", fallback=80.0)
    aux_q             = cfg.getfloat(sec, "aux_q", fallback=0.20)
    small_t_weight    = cfg.getfloat(sec, "small_t_weight", fallback=0.5)
    small_t_gamma     = cfg.getfloat(sec, "small_t_gamma", fallback=2.0)

    # Dataset-specific knobs
    face_tol   = cfg.getfloat(sec, "face_tolerance", fallback=1e-4)
    minsep_chunk = cfg.getint(sec, "minsep_chunk", fallback=256)
    scale_N    = cfg.getint(sec, "scale_N", fallback=128)

    # Sampling params
    num_new = cfg.getint(sec, "num_generated_samples", fallback=1000)
    batch_n = cfg.getint(sec, "generation_batch_size", fallback=50)
    points_N = cfg.getint(sec, "num_spheres", fallback=None)  # if None, use N from dataset

    # PCFM sampler knobs
    pcfm_steps   = cfg.getint(sec, "pcfm_steps", fallback=40)
    ode_method   = cfg.get(sec, "ode_method", fallback="midpoint")
    ode_step_cap = cfg.getfloat(sec, "ode_step_cap", fallback=0.05)
    jitter       = cfg.getfloat(sec, "pcfm_jitter", fallback=5e-3)
    eps_nrm      = cfg.getfloat(sec, "pcfm_eps_nrm", fallback=1e-6)
    proj_outer_iters = cfg.getint(sec, "proj_outer_iters", fallback=8)
    alpha_proj   = cfg.getfloat(sec, "alpha_proj", fallback=0.25)
    contact_q    = cfg.getfloat(sec, "contact_q", fallback=1.0)
    wall_weight  = cfg.getfloat(sec, "wall_weight", fallback=1.0)
    wall_margin  = cfg.getfloat(sec, "wall_margin", fallback=0.05)
    prox_iters   = cfg.getint(sec, "prox_iters", fallback=10)
    prox_step    = cfg.getfloat(sec, "prox_step", fallback=0.1)
    prox_lambda  = cfg.getfloat(sec, "prox_lambda", fallback=2.0)
    final_passes = cfg.getint(sec, "final_passes", fallback=6)
    tol_finish   = cfg.getfloat(sec, "tol_finish", fallback=1e-8)

    # Architecture
    st_kwargs = {
        'dim_hidden':     cfg.getint(sec, "st_dim_hidden", fallback=512),
        'num_heads':      cfg.getint(sec, "st_num_heads", fallback=8),
        'depth':          cfg.getint(sec, "st_depth", fallback=6),
        'attn_dropout':   cfg.getfloat(sec, "st_attn_dropout", fallback=0.1),
        'ff_dropout':     cfg.getfloat(sec, "st_ff_dropout", fallback=0.1),
        'dim_time':       cfg.getint(sec, "st_dim_time", fallback=max(64, 4*d)),
        'time_fourier_dim': cfg.getint(sec, "time_fourier_dim", fallback=max(16, (max(64,4*d))//2)),
        'time_hidden':    cfg.getint(sec, "time_hidden", fallback=2*max(64,4*d)),
        'time_fourier_sigma': cfg.getfloat(sec, "time_fourier_sigma", fallback=1.0),
        'cond_dim_in':    cfg.getint(sec, "cond_dim_in", fallback=4),
        'cond_hidden':    cfg.getint(sec, "cond_hidden", fallback=max(64, 4*d)),
    }
    # also record dim_out in params (for loading)
    st_kwargs["dim_out"] = d

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    full_ds = SpherePackingDataset(
        dataset_path,
        radius=sphere_radius,
        box_len=clip_range,
        tol=face_tol,
        chunk=minsep_chunk,
        scale_N=scale_N
    )

    if points_N is None:
        points_N = full_ds.N
    assert points_N == full_ds.N, f"Config num_spheres={points_N} but dataset N={full_ds.N}"

    # Train/test split with optional cap on max_samples
    max_samples = min(len(full_ds), max_samples)
    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(max_samples, generator=gen)
    test_size = max(1, int(round(max_samples * test_fraction)))
    test_idx = perm[:test_size].tolist()
    train_idx = perm[test_size:].tolist()

    try:
        frac = float(train_top_fraction)
    except Exception:
        frac = 1.0

    if frac < 1.0 and len(train_idx) > 0:
        k = max(1, int(math.ceil(len(train_idx) * frac)))
        if k < len(train_idx):
            # minsep is precomputed and stored in full_ds.minsep
            minsep_all = full_ds.minsep  # (M,)
            idx_tensor = torch.tensor(train_idx, dtype=torch.long)
            minsep_train = minsep_all[idx_tensor]
            _, top_pos = torch.topk(minsep_train, k=k, largest=True)
            filtered_train_idx = idx_tensor[top_pos].tolist()
            print(f"[Spheres] Filtering training to top {100.0*frac:.1f}% by minsep: {len(filtered_train_idx)} of {len(train_idx)}")
            train_idx = filtered_train_idx
        else:
            print(f"[Spheres] train_top_fraction keeps all {len(train_idx)} training samples.")
    else:
        print(f"[Spheres] train_top_fraction={frac} -> using all {len(train_idx)} training samples.")


    train_ds = Subset(full_ds, train_idx)
    test_ds  = Subset(full_ds,  test_idx)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_n, shuffle=False)

    print(f"[Spheres] Train size: {len(train_ds)} | Test size: {len(test_ds)} | N={points_N}")

    model = FlowSetTransformer(d, **st_kwargs).to(device)
    opt = schedulefree.RAdamScheduleFree(model.parameters(), lr=lr)

    resume_path = cfg.get(sec, "resume_model_path", fallback="").strip()
    model_path = resume_path

    if mode == "training_and_sampling":
        # Supervised training
        model, hist, model_path = train_flow_model(
            model, opt, train_loader, epochs,
            sphere_radius, mse_s, pen_s,
            clip_range, device,
            st_kwargs, save_model_dir,
            aux_margin_factor=aux_margin_factor,
            aux_beta=aux_beta,
            aux_q=aux_q,
            small_t_weight=small_t_weight,
            small_t_gamma=small_t_gamma,
        )
        if state:
            state.set_model_path(model_path)

        # Optional RG-CFM immediately after training
        use_rg_after_train = cfg.getboolean("spheres_in_cube_new_pipeline", "use_rg_cfm", fallback=False)
        if use_rg_after_train:
            if state and state.model_path:
                cfg.set(sec, "rg_ref_path", state.model_path)
                cfg.set(sec, "resume_model_path", state.model_path)
            print("[flow_matching] Starting RG-CFM immediately after supervised training.")
            rg_cfm_main(state=state)
            # Reload the fine-tuned model into memory before sampling
            if state and state.model_path:
                model, opt = load_model_if_exists(model, opt, state.model_path, device)
                model_path = state.model_path
            else:
                model, opt = load_model_if_exists(model, opt, cfg.get(sec, "resume_model_path", fallback="").strip(), device)

        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N,
            device, sphere_radius, 0.0, clip_range, d,
            cond_loader=test_loader,
            n_steps=pcfm_steps,
            ode_method=ode_method,
            ode_step_cap=ode_step_cap,
            jitter=jitter,
            eps_nrm=eps_nrm,
            proj_outer_iters=proj_outer_iters,
            alpha_proj=alpha_proj,
            contact_q=contact_q,
            wall_weight=wall_weight,
            wall_margin=wall_margin,
            prox_iters=prox_iters,
            prox_step=prox_step,
            prox_lambda=prox_lambda,
            final_passes=final_passes,
            tol_finish=tol_finish,
        )

    elif mode == "sampling_only":
        assert resume_path and os.path.isfile(resume_path), "resume_model_path must point to a saved model"
        model, opt = load_model_if_exists(model, opt, resume_path, device)
        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N,
            device, sphere_radius, 0.0, clip_range, d,
            cond_loader=test_loader,
            n_steps=pcfm_steps,
            ode_method=ode_method,
            ode_step_cap=ode_step_cap,
            jitter=jitter,
            eps_nrm=eps_nrm,
            proj_outer_iters=proj_outer_iters,
            alpha_proj=alpha_proj,
            contact_q=contact_q,
            wall_weight=wall_weight,
            wall_margin=wall_margin,
            prox_iters=prox_iters,
            prox_step=prox_step,
            prox_lambda=prox_lambda,
            final_passes=final_passes,
            tol_finish=tol_finish,
        )

    elif mode == "retrain_and_sampling":
        assert resume_path and os.path.isfile(resume_path), "resume_model_path must point to a saved model"
        model, opt = load_model_if_exists(model, opt, resume_path, device)
        model, hist, model_path = train_flow_model(
            model, opt, train_loader, epochs,
            sphere_radius, mse_s, pen_s,
            clip_range, device,
            st_kwargs, save_model_dir,
            aux_margin_factor=aux_margin_factor,
            aux_beta=aux_beta,
            aux_q=aux_q,
            small_t_weight=small_t_weight,
            small_t_gamma=small_t_gamma,
        )
        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N,
            device, sphere_radius, 0.0, clip_range, d,
            cond_loader=test_loader,
            n_steps=pcfm_steps,
            ode_method=ode_method,
            ode_step_cap=ode_step_cap,
            jitter=jitter,
            eps_nrm=eps_nrm,
            proj_outer_iters=proj_outer_iters,
            alpha_proj=alpha_proj,
            contact_q=contact_q,
            wall_weight=wall_weight,
            wall_margin=wall_margin,
            prox_iters=prox_iters,
            prox_step=prox_step,
            prox_lambda=prox_lambda,
            final_passes=final_passes,
            tol_finish=tol_finish,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")
    if state: state.set_model_path(model_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_generated_dir, f"spheres_gen_{num_new}x{points_N}_{ts}.pt")
    torch.save(torch.from_numpy(samples), out_path)
    if state: state.set_samples_path(out_path)
    print(f"[Save] Saved {num_new} generated sphere packings to '{out_path}', shape {samples.data.shape}")


if __name__ == "__main__":
    main()
