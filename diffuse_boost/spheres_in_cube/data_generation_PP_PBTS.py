import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import itertools
from scipy.optimize import minimize
from datetime import datetime
from diffuse_boost import cfg
from diffuse_boost.spheres_in_cube.physics_push_PESC import eliminate_overlaps_box

# -----------------------------------------------------------------------------
# Utilities: Penalty gradient, energy evaluation, and clearance maximization
# -----------------------------------------------------------------------------
EPS_SMALL = 1e-6

def compute_EL_grad(X_rel, L, N, r):
    """
    Compute energy and gradient in the 'relative' coordinate frame by mapping
    through to absolute cube coordinates, computing the true overlap penalties
    there, and then chaining the gradient back.
    X_rel: array of shape (3N,), representing coords in [-L/2, L/2].
    """
    # Map to absolute sphere centers in [r, L-r]^3
    coords_rel = X_rel.reshape((N, 3))
    scale = (L - 2*r) / L
    half = L / 2
    coords_abs = (coords_rel + half) * scale + r

    EL = 0.0
    grad_abs = np.zeros_like(coords_abs)

    # Wall penalties in absolute coordinates
    for i, xi in enumerate(coords_abs):
        for d in range(3):
            if xi[d] < r:
                over = r - xi[d]
                EL += over**2
                # dE/dxi = -2 * over
                grad_abs[i, d] += -2 * over
            elif xi[d] > L - r:
                over = xi[d] - (L - r)
                EL += over**2
                # dE/dxi = +2 * over
                grad_abs[i, d] += 2 * over

    # Sphere-sphere overlap penalties in absolute coordinates
    for i in range(N):
        for j in range(i + 1, N):
            diff = coords_abs[i] - coords_abs[j]
            dist = np.linalg.norm(diff)
            over = max(0.0, 2*r - dist)
            if over > 0:
                EL += over**2
                # avoid zero-distance trap
                if dist < EPS_SMALL:
                    direction = np.random.randn(3)
                    direction /= (np.linalg.norm(direction) + EPS_SMALL)
                else:
                    direction = diff / dist
                g_abs = 2 * over * direction
                grad_abs[i] -= g_abs
                grad_abs[j] += g_abs

    # Chain-rule: mapping x_rel -> x_abs is x_abs = scale * (x_rel + half) + r
    # so dE/dx_rel = scale * dE/dx_abs
    grad_rel = grad_abs * scale
    return EL, grad_rel.ravel()


def compute_EL(X_rel, L, N, r):
    """
    Compute just the energy (no gradient) via compute_EL_grad.
    """
    EL, _ = compute_EL_grad(X_rel, L, N, r)
    return EL


def maximize_clearance(X_rel, N, steps=10, step_size=0.01):
    coords = X_rel.reshape((N, 3))
    for _ in range(steps):
        grad = np.zeros_like(coords)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = coords[i] - coords[j]
                dist = np.linalg.norm(diff) + EPS_SMALL
                grad[i] += diff / dist
        coords += step_size * grad
    return coords.ravel()


def SRP(X_rel, L, N, r, Imax, m, sigma, beta):
    eta = sigma
    Xc = X_rel.copy()
    for _ in range(Imax):
        Xc += np.random.uniform(-eta, eta, size=Xc.shape)
        for __ in range(m):
            EL, g = compute_EL_grad(Xc, L, N, r)
            g = g.reshape((N, 3))
            norm = np.linalg.norm(g, ord=np.inf) + EPS_SMALL
            Xc -= (sigma * eta) * (g / norm).ravel()
        eta *= beta
    return Xc


def local_opt(X_rel, L, N, r, tol, maxiter):
    """
    Local optimization in the relative frame with explicit L-BFGS-B bounds.
    """
    def objective(x):
        EL, _ = compute_EL_grad(x, L, N, r)
        return EL
    def gradient(x):
        _, grad = compute_EL_grad(x, L, N, r)
        return grad

    x0 = X_rel.copy()
    half = L / 2
    eps = 1e-6
    bounds = [(-half + eps, half - eps)] * (3 * N)

    res = minimize(
        fun=objective,
        x0=x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'ftol': tol, 'gtol': tol, 'maxiter': maxiter}
    )
    return res.x, res.fun

# -----------------------------------------------------------------------------
# Sampling & symmetries
# -----------------------------------------------------------------------------
def sample_uniform_points(dim, L, r, N):
    low, high = r, L - r
    if high <= low:
        raise ValueError("bounding_box_width must exceed 2*radius.")
    return np.random.uniform(low, high, size=(N, dim))


def get_cube_symmetry_matrices(dim):
    mats = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product([-1, 1], repeat=dim):
            M = np.zeros((dim, dim))
            for i in range(dim):
                M[i, perm[i]] = signs[i]
            mats.append(M)
    return mats


def apply_symmetries_to_data(data, L):
    M, D, N = data.shape
    mats = get_cube_symmetry_matrices(D)
    out = np.zeros((M * len(mats), D, N), dtype=data.dtype)
    center = L / 2
    idx = 0
    for i in range(M):
        coords = data[i].T
        for mat in mats:
            T = (mat @ (coords - center).T).T + center
            out[idx] = T.T
            idx += 1
    return out

# -----------------------------------------------------------------------------
# Main generation: SRP + local_opt + maximin + physics push
# -----------------------------------------------------------------------------
def generate_dataset_push_srp():
    sec = "sample_generation_PP+PBTS"
    D = cfg.getint(sec, "dimension")
    L = cfg.getfloat(sec, "bounding_box_width")
    r = cfg.getfloat(sec, "sphere_radius")
    best_d = cfg.getfloat(sec, "best_known_diameter", fallback=2 * r)
    N = cfg.getint(sec, "num_spheres")
    M = cfg.getint(sec, "num_samples")
    dt = cfg.getfloat(sec, "dt")
    max_iter = cfg.getint(sec, "max_iter")
    tol = cfg.getfloat(sec, "tol")
    mode = cfg.get(sec, "boundary_mode")

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    metrics_fn = cfg.get(sec, "output_filename_metrics", fallback="srp_metrics.csv").replace("{DATE}", timestamp_str)
    metrics_dir = os.path.dirname(metrics_fn)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_fn, 'w') as mf:
        mf.write("sample,srp_restart,EL_before,EL_after,pre_push_min,post_push_min,excess\n")

    Imax = cfg.getint(sec, "srp_Imax", fallback=500)
    m = cfg.getint(sec, "srp_m", fallback=20)
    sigma_frac = cfg.getfloat(sec, "srp_sigma_frac", fallback=0.2)
    sigma = sigma_frac * L
    beta = cfg.getfloat(sec, "srp_beta", fallback=0.95)
    tol_opt = cfg.getfloat(sec, "srp_tol", fallback=1e-8)
    maxiter_opt = cfg.getint(sec, "srp_maxiter", fallback=300)
    restarts = cfg.getint(sec, "srp_restarts", fallback=10)

    data = np.zeros((M, D, N), dtype=np.float32)
    min_dists = []

    for i in range(M):
        print(f"Generating sample {i+1}/{M}...")
        pts = sample_uniform_points(D, L, r, N)
        centers0, _ = eliminate_overlaps_box(
            pts, r, [L] * D,
            max_iter=max_iter, dt=dt, tol=tol,
            boundary_mode=mode, visualize=False
        )
        half = L / 2
        X0 = (centers0 - half).ravel()
        diffs0 = centers0[:, None, :] - centers0[None, :, :]
        init_min = np.min(np.linalg.norm(diffs0, axis=-1)[np.triu_indices(N, k=1)])
        best_centers = centers0.copy()
        best_min = init_min
        excess = best_d - init_min
        print(f"Sample {i+1}/{M}: initial min distance = {init_min:.6f}, excess = {excess:.6f}")
        for k in range(restarts):
            print(f"  SRP restart {k+1}/{restarts} for sample {i+1}/{M}")
            eps = 1e-6
            X_srp = SRP(X0, L, N, r, Imax, m, sigma, beta)
            X_srp_clip = np.clip(X_srp, -L/2 + eps, L/2 - eps)
            EL_before = compute_EL(X_srp_clip, L, N, r)
            print(f"    SRP restart {k+1}/{restarts}: EL before local_opt = {EL_before:.6f}")
            X_lo, EL_after = local_opt(X_srp_clip, L, N, r, tol_opt, maxiter_opt)
            print(f"    SRP restart {k+1}/{restarts}: EL after local_opt  = {EL_after:.6f}")
            # Print min distance and excess after local_opt
            coords = X_lo.reshape((N, 3))
            centers_opt = (coords + half) * ((L - 2 * r) / L) + r
            centers_opt = np.minimum(np.maximum(centers_opt, r + EPS_SMALL), L - r - EPS_SMALL)

            diffs_pre = centers_opt[:, None, :] - centers_opt[None, :, :]
            pre_min = np.min(np.linalg.norm(diffs_pre, axis=-1)[np.triu_indices(N, k=1)])
            excess = best_d - pre_min
            print(f"    min distance after local_opt = {pre_min:.6f}, excess = {excess:.6f}")
            #X_max = maximize_clearance(X_lo, N, steps=5, step_size=0.01)
            #coords = X_max.reshape((N, 3))
            #centers_opt = (coords + half) * ((L - 2 * r) / L) + r
            #centers_opt = np.minimum(np.maximum(centers_opt, r + EPS_SMALL), L - r - EPS_SMALL)

            #diffs_pre = centers_opt[:, None, :] - centers_opt[None, :, :]
            #pre_min = np.min(np.linalg.norm(diffs_pre, axis=-1)[np.triu_indices(N, k=1)])
            #excess = best_d - pre_min
            #print(f"    min_after_maximize_clearance = {pre_min:.6f}, excess = {excess:.6f}")

            centers_k, _ = eliminate_overlaps_box(
                centers_opt, r, [L] * D,
                max_iter=max_iter, dt=dt, tol=tol,
                boundary_mode=mode, visualize=False
            )
            diffs_k = centers_k[:, None, :] - centers_k[None, :, :]
            post_min = np.min(np.linalg.norm(diffs_k, axis=-1)[np.triu_indices(N, k=1)])
            excess = best_d - post_min
            print(f"    min_after_physics_push = {post_min:.6f}, excess = {excess:.6f}")
            with open(metrics_fn, 'a') as mf:
                mf.write(f"{i},{k+1},{EL_before:.6f},{EL_after:.6f},{pre_min:.6f},{post_min:.6f},{excess:.6f}\n")
            if post_min > best_min:
                best_min = post_min
                best_centers = centers_k.copy()

        data[i] = best_centers.T
        min_dists.append(best_min)
        print(f"Finished sample {i+1}/{M}, best_min = {best_min:.6f}\n")

    data_fn = cfg.get(sec, "output_filename").replace("{DATE}", timestamp_str)
    data_dir = os.path.dirname(data_fn)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
    torch.save(torch.from_numpy(data), data_fn)
    print(f"Saved full dataset to {data_fn}")

    try:
        sym_data = apply_symmetries_to_data(data, L)
        sym_fn = cfg.get(sec, "output_filename_sym", fallback=data_fn.replace('.pt', '_sym.pt')).replace("{DATE}", timestamp_str)
        sym_dir = os.path.dirname(sym_fn)
        if sym_dir:
            os.makedirs(sym_dir, exist_ok=True)
        torch.save(torch.from_numpy(sym_data), sym_fn)
        print(f"Saved symmetrized dataset to {sym_fn}")
    except ValueError as e:
        print(f"Skipping symmetry enrichment: {e}")

    k_top = max(1, int(np.ceil(0.25 * M)))
    best_idx = np.argsort(min_dists)[-k_top:]
    top_data = data[best_idx]
    top_fn = cfg.get(sec, "output_filename_top").replace("{DATE}", timestamp_str)
    top_dir = os.path.dirname(top_fn)
    if top_dir:
        os.makedirs(top_dir, exist_ok=True)
    torch.save(torch.from_numpy(top_data), top_fn)
    print(f"Saved top {k_top} samples to {top_fn}")

    try:
        sym_top = apply_symmetries_to_data(top_data, L)
        sym_top_fn = cfg.get(sec, "output_filename_sym_top", fallback=top_fn.replace('.pt', '_sym.pt')).replace("{DATE}", timestamp_str)
        sym_top_dir = os.path.dirname(sym_top_fn)
        if sym_top_dir:
            os.makedirs(sym_top_dir, exist_ok=True)
        torch.save(torch.from_numpy(sym_top), sym_top_fn)
        print(f"Saved symmetrized top dataset to {sym_top_fn}")
    except ValueError as e:
        print(f"Skipping symmetry enrichment for top samples: {e}")

def load_metrics_PP_p_PBTS(filename):
    data_excess = []
    with open(filename) as m_file:
        lines = m_file.readlines()
        for data_text in lines[1:]:
            data_string = data_text.split(",")
            data_excess.append(float(data_string[-1]))
    return data_excess

if __name__ == "__main__":
    generate_dataset_push_srp()
