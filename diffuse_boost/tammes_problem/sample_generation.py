# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os, platform
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

import numba as nb
from numba import njit

# Optional config support
try:
    from diffuse_boost import cfg
    HAS_CFG = True
except Exception:
    HAS_CFG = False

# =============================================================================
# Utilities / small constants
# =============================================================================
EPS = 1e-12

def _get_cfg(section, key, fallback):
    if HAS_CFG:
        try:
            if isinstance(fallback, int):
                return cfg.getint(section, key, fallback=fallback)
            if isinstance(fallback, float):
                return cfg.getfloat(section, key, fallback=fallback)
            if isinstance(fallback, bool):
                return cfg.getboolean(section, key, fallback=fallback)
            return cfg.get(section, key, fallback=fallback)
        except Exception:
            return fallback
    return fallback

# =============================================================================
# Random initialization on the sphere S^{n-1}
# =============================================================================
def sample_uniform_on_sphere(N, dim):
    """
    Draw N points uniformly on S^{dim-1} via Gaussian normalization.
    """
    X = np.random.normal(size=(N, dim)).astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-18
    return (X / norms).astype(np.float64)

# =============================================================================
# Spherical geometry helpers
# =============================================================================
@njit(cache=True, fastmath=True)
def _tangent_projection(g, x):
    # project Euclidean gradient g to tangent space at x \in S^{n-1}
    dot = 0.0
    for k in range(x.size):
        dot += g[k] * x[k]
    out = g.copy()
    for k in range(x.size):
        out[k] -= dot * x[k]
    return out

@njit(cache=True, fastmath=True)
def _expmap_sphere(x, u):
    """
    Exponential map on S^{n-1} at x in direction u (tangent at x).
    Returns point on sphere. If ||u|| ~ 0, just return x + u then renorm.
    """
    un = 0.0
    for k in range(u.size):
        un += u[k]*u[k]
    un = math.sqrt(un)
    if un < 1e-18:
        y = x + u
        yn = 0.0
        for k in range(y.size):
            yn += y[k]*y[k]
        yn = math.sqrt(yn) + 1e-18
        for k in range(y.size):
            y[k] /= yn
        return y
    c = math.cos(un)
    s = math.sin(un) / un
    y = np.empty_like(x)
    for k in range(x.size):
        y[k] = c * x[k] + s * u[k]
    # numerical renormalization
    yn = 0.0
    for k in range(y.size):
        yn += y[k]*y[k]
    yn = math.sqrt(yn) + 1e-18
    for k in range(y.size):
        y[k] /= yn
    return y

@njit(cache=True, fastmath=True)
def _l2_norm_vec(v):
    s = 0.0
    for k in range(v.size):
        s += v[k]*v[k]
    return math.sqrt(s) + 1e-18

# =============================================================================
# Losses and gradients
# mode = 0  -> inverse-power repulsion   L = sum_{i<j} 1 / ||x_i - x_j||^p
# mode = 1  -> log-sum-exp of dot prods  L = tau * log( sum_{i<j} exp( (xi·xj)/tau ) )
# =============================================================================
@njit(cache=True, fastmath=True)
def spherical_loss_and_grad(X, N, dim, mode, p_power, lse_tau):
    """
    Returns (L, g_flat) where g_flat is the Euclidean gradient (not yet tangent-projected).
    We keep it Euclidean so callers can project per-point and step on the sphere.
    """
    P = X.reshape((N, dim))
    G = np.zeros_like(P)
    L = 0.0

    if mode == 0:
        # inverse-power on chord distances
        for i in range(N):
            xi = P[i]
            for j in range(i+1, N):
                xj = P[j]
                s = 0.0
                for k in range(dim):
                    v = xi[k] - xj[k]
                    s += v*v
                d = math.sqrt(s) + EPS
                inv = 1.0 / (d**p_power)
                L += inv
                coeff = -p_power / (d**(p_power + 2.0))
                for k in range(dim):
                    vij = (xi[k] - xj[k])
                    gi = coeff * vij
                    G[i, k] += gi
                    G[j, k] -= gi
        return L, G.ravel()

    # mode == 1: log-sum-exp over dot products
    # First pass: accumulate max z for stability
    maxz = -1e300
    for i in range(N):
        xi = P[i]
        for j in range(i+1, N):
            xj = P[j]
            dot = 0.0
            for k in range(dim):
                dot += xi[k]*xj[k]
            z = dot / lse_tau
            if z > maxz:
                maxz = z

    # Second pass: sum exp(z - maxz), accumulate gradient weights
    S = 0.0
    for i in range(N):
        xi = P[i]
        for j in range(i+1, N):
            xj = P[j]
            dot = 0.0
            for k in range(dim):
                dot += xi[k]*xj[k]
            z = (dot / lse_tau) - maxz
            ez = math.exp(z)
            S += ez
            # d/d xi of dot = xj  (and symmetric)
            for k in range(dim):
                G[i, k] += (ez / lse_tau) * xj[k]
                G[j, k] += (ez / lse_tau) * xi[k]

    if S <= 0.0:
        return 0.0, G.ravel()
    L = lse_tau * (math.log(S) + maxz)  # tau * log sum exp

    return L, G.ravel()

# =============================================================================
# SRP (stochastic relaxed projection) on the sphere with backtracking + seeding
# =============================================================================
@njit(cache=True, fastmath=True)
def srp_sphere(
    X0, N, dim,
    Imax, m, step_pos, beta, backtrack,
    mode, p_power, lse_tau, noise_scale,
    seed
):
    """
    - X0: flat vector (N*dim,) with unit-norm rows
    - step_pos: base geodesic stepsize
    - noise_scale: tangent-space jitter relative to step_pos
    - seed: seeds Numba RNG for this run (per-sample)
    """
    # Seed Numba RNG
    np.random.seed(np.int64(seed))

    Xc = X0.copy()
    eta = 1.0

    for _ in range(Imax):
        # 1) Jitter in tangent spaces
        Xtrial = Xc.copy()
        P = Xtrial.reshape((N, dim))
        for i in range(N):
            z = np.random.normal(0.0, 1.0, size=dim)
            # project to tangent at x_i
            dot = 0.0
            for k in range(dim):
                dot += z[k] * P[i, k]
            for k in range(dim):
                z[k] = z[k] - dot * P[i, k]
            # scale
            zn = _l2_norm_vec(z)
            if zn > 0.0:
                for k in range(dim):
                    z[k] = z[k] * (eta * noise_scale * step_pos / zn)
            # move via exponential map
            P[i, :] = _expmap_sphere(P[i, :], z)

        # 2) m inner gradient steps with backtracking on the sphere
        for __ in range(m):
            L0, g = spherical_loss_and_grad(Xtrial, N, dim, mode, p_power, lse_tau)
            G = g.reshape((N, dim))
            GT = np.zeros_like(G)
            for i in range(N):
                GT[i, :] = _tangent_projection(G[i, :], P[i, :])

            Pprop = P.copy()
            for i in range(N):
                gn = _l2_norm_vec(GT[i, :])
                if gn > 0.0:
                    u = - (eta * step_pos / gn) * GT[i, :]
                else:
                    u = np.zeros(dim)
                Pprop[i, :] = _expmap_sphere(P[i, :], u)

            Xprop = Pprop.ravel()
            Lprop, _ = spherical_loss_and_grad(Xprop, N, dim, mode, p_power, lse_tau)

            bt = 0
            cur_step = step_pos
            while Lprop > L0 and bt < backtrack:
                cur_step *= 0.5
                for i in range(N):
                    gn = _l2_norm_vec(GT[i, :])
                    if gn > 0.0:
                        u = - (eta * cur_step / gn) * GT[i, :]
                    else:
                        u = np.zeros(dim)
                    Pprop[i, :] = _expmap_sphere(P[i, :], u)
                Xprop = Pprop.ravel()
                Lprop, _ = spherical_loss_and_grad(Xprop, N, dim, mode, p_power, lse_tau)
                bt += 1

            P[:, :] = Pprop[:, :]

        Xc = P.ravel()
        eta *= beta

    return Xc

# =============================================================================
# Polishing pass: short Riemannian GD with backtracking
# =============================================================================
@njit(cache=True, fastmath=True)
def polish_sphere(
    X0, N, dim,
    iters, step_pos, backtrack,
    mode, p_power, lse_tau
):
    X = X0.copy()
    P = X.reshape((N, dim))

    for _ in range(iters):
        L0, g = spherical_loss_and_grad(X, N, dim, mode, p_power, lse_tau)
        G = g.reshape((N, dim))
        GT = np.zeros_like(G)
        for i in range(N):
            GT[i, :] = _tangent_projection(G[i, :], P[i, :])

        # propose step
        Pprop = P.copy()
        for i in range(N):
            gn = _l2_norm_vec(GT[i, :])
            if gn > 0.0:
                u = - (step_pos / gn) * GT[i, :]
            else:
                u = np.zeros(dim)
            Pprop[i, :] = _expmap_sphere(P[i, :], u)

        Xprop = Pprop.ravel()
        Lprop, _ = spherical_loss_and_grad(Xprop, N, dim, mode, p_power, lse_tau)

        bt = 0
        cur_step = step_pos
        while Lprop > L0 and bt < backtrack:
            cur_step *= 0.5
            for i in range(N):
                gn = _l2_norm_vec(GT[i, :])
                if gn > 0.0:
                    u = - (cur_step / gn) * GT[i, :]
                else:
                    u = np.zeros(dim)
                Pprop[i, :] = _expmap_sphere(P[i, :], u)
            Xprop = Pprop.ravel()
            Lprop, _ = spherical_loss_and_grad(Xprop, N, dim, mode, p_power, lse_tau)
            bt += 1

        P[:, :] = Pprop[:, :]
        X = P.ravel()

    return X

# =============================================================================
# Metrics
# =============================================================================
@njit(cache=True, fastmath=True)
def min_pairwise_chord(P):
    N, dim = P.shape
    best = 1e9
    for i in range(N):
        for j in range(i+1, N):
            s = 0.0
            for k in range(dim):
                v = P[i,k] - P[j,k]
                s += v*v
            d = math.sqrt(s)
            if d < best:
                best = d
    return best

@njit(cache=True, fastmath=True)
def min_pairwise_angle(P):
    # angle = arccos( dot(x,y) ), x,y unit
    N, dim = P.shape
    best = 1e9
    for i in range(N):
        for j in range(i+1, N):
            dot = 0.0
            for k in range(dim):
                dot += P[i,k] * P[j,k]
            if dot > 1.0: dot = 1.0
            if dot < -1.0: dot = -1.0
            ang = math.acos(dot)
            if ang < best:
                best = ang
    return best

# =============================================================================
# Plotting (only for dim == 3)
# =============================================================================
def plot_first_k_spherical_samples(data_tensor, k, out_dir, filename_prefix="tammes"):
    """
    Draws the first k samples from data tensor of shape (M, dim, N) with rows [coords...].
    For dim==3, makes 3D scatter and sphere wireframe.
    """
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    M, dim, _ = data_tensor.shape
    if dim != 3:
        return

    k = min(k, M)
    for s in range(k):
        X = data_tensor[s, :, :].T  # (N,3)
        N = X.shape[0]
        # chord min
        best = 1e9
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(X[i]-X[j])
                best = min(best, d)
        # angle min
        dots = np.clip(X @ X.T, -1.0, 1.0)
        mask = ~np.eye(N, dtype=bool)
        min_ang = np.min(np.arccos(dots[mask]))
        min_ang_deg = np.degrees(min_ang)

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        # unit sphere wireframe
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.3)

        ax.scatter(X[:,0], X[:,1], X[:,2], s=20)
        ax.set_title(f"Sample #{s}, N={N}, min chord={best:.4f}, min angle={min_ang_deg:.2f}°")
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)

        out_path = os.path.join(out_dir, f"{filename_prefix}_N{N}_minang_{min_ang_deg:.2f}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# Main generator
# =============================================================================
def generate_tammes_dataset():
    sec = "tammes_SRP"

    # ---------- Problem ----------
    N        = _get_cfg(sec, "num_points",   50)     # N
    dim      = _get_cfg(sec, "dim",          3)      # sphere S^{dim-1}
    M        = _get_cfg(sec, "num_samples",  5000)   # number of samples

    # ---------- SRP hyperparameters ----------
    Imax       = _get_cfg(sec, "srp_Imax",        400)
    m          = _get_cfg(sec, "srp_m",           30)
    beta       = _get_cfg(sec, "srp_beta",        0.985)
    backtrack  = _get_cfg(sec, "srp_backtrack",   3)
    step_pos   = _get_cfg(sec, "srp_step_pos",    0.05)
    noise_scale= _get_cfg(sec, "srp_noise_scale", 0.5)

    # ---------- Loss selection ----------
    # mode: 0 = inverse-power chord repulsion; 1 = log-sum-exp of dot products
    loss_mode  = _get_cfg(sec, "loss_mode",       "inv_p")
    mode = 0 if loss_mode.lower() in ["inv_p", "invp", "inverse", "power"] else 1
    p_power    = _get_cfg(sec, "repulsion_p",     12)      # used if mode==0
    lse_tau    = _get_cfg(sec, "lse_tau",         0.03)    # used if mode==1

    # ---------- Polishing ----------
    do_polish  = _get_cfg(sec, "polish_enable",   True)
    polish_it  = _get_cfg(sec, "polish_iters",    120)
    polish_step= _get_cfg(sec, "polish_step",     0.02)
    polish_bt  = _get_cfg(sec, "polish_backtrack",3)

    # ---------- Seeding / rotations ----------
    base_seed  = _get_cfg(sec, "base_seed",       123456)
    rotate_fin = _get_cfg(sec, "random_rotate_final", True)

    # ---------- I/O + plotting ----------
    out_dir    = _get_cfg(sec, "output_dir",      "./outputs_tammes")
    plot_k     = _get_cfg(sec, "plot_k",          0) 
    plot_dir   = os.path.join(out_dir, "plots")

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"tammes_srp_generated_{M}x{N}d{dim}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"tammes_srp_metrics_{M}x{N}d{dim}_{stamp}.csv")

    print(f"Generating {M} Tammes samples on S^{dim-1} (N={N})")
    with open(metrics_fn, "w") as mf:
        # Added cosine_min_degree = cos(min_angle_radians)
        mf.write("sample,min_chord,min_angle_rad,min_angle_deg,cosine_min_degree,loss\n")

    data = np.zeros((M, dim, N), dtype=np.float32)
    # keep per-sample min chord for sorting
    per_sample_min = np.empty(M, dtype=np.float64)

    bar = tqdm(range(M), desc="Generating Tammes packings")
    for s in bar:
        # Random start on the sphere (Python RNG; distinct each sample)
        P0 = sample_uniform_on_sphere(N, dim)
        X0 = P0.ravel()

        # SRP exploration with per-sample seed for Numba RNG
        seed = int(base_seed + s)
        X_fin = srp_sphere(
            X0, N, dim,
            Imax=Imax, m=m, step_pos=step_pos, beta=beta, backtrack=backtrack,
            mode=mode, p_power=p_power, lse_tau=lse_tau, noise_scale=noise_scale,
            seed=np.int64(seed)
        )

        # Optional: polishing pass
        if do_polish:
            X_fin = polish_sphere(
                X_fin, N, dim,
                iters=polish_it, step_pos=polish_step, backtrack=polish_bt,
                mode=mode, p_power=p_power, lse_tau=lse_tau
            )

        # Reshape and ensure exact unit norm (defensive)
        P = X_fin.reshape((N, dim))
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-18)

        # Optional: random rotation for visual diversity (metrics unchanged)
        if rotate_fin and dim == 3:
            A = np.random.normal(size=(dim, dim))
            Q, _ = np.linalg.qr(A)
            # Ensure proper rotation (det +1)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            P = P @ Q

        # Metrics
        mc  = float(min_pairwise_chord(P))
        ma  = float(min_pairwise_angle(P))          # radians
        mad = float(np.degrees(ma))                 # degrees
        cos_min = float(math.cos(ma))               # requested extra metric

        L, _ = spherical_loss_and_grad(P.ravel(), N, dim, mode, p_power, lse_tau)

        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{mc:.8f},{ma:.8f},{mad:.8f},{cos_min:.8f},{L:.8e}\n")

        data[s, :, :] = P.T.astype(np.float32)
        per_sample_min[s] = mc
        bar.set_postfix(min_chord=f"{mc:.4f}", min_angle_deg=f"{mad:.2f}")

    # Sort samples by min chord distance (descending = better)
    sorted_indices = np.argsort(-per_sample_min)
    data = data[sorted_indices]

    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved dataset:  {dataset_fn}")
    print(f"Saved metrics:  {metrics_fn}")

    # Save plots for the first top_k samples (only if dim==3)
    if plot_k > 0 and int(dim) == 3:
        plot_first_k_spherical_samples(data, plot_k, plot_dir, filename_prefix="tammes")
        print(f"Saved plots:   {plot_dir}")

# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    generate_tammes_dataset()
