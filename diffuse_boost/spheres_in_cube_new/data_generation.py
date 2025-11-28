# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os
import platform
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import math
import torch
from torch.utils.data import DataLoader, Dataset
import itertools
from scipy.optimize import minimize
from datetime import datetime
import numba as nb
from numba import njit
from diffuse_boost import cfg
from diffuse_boost.spheres_in_cube.physics_push_PESC import eliminate_overlaps_box
from diffuse_boost.spheres_in_cube.best_results import load_best_results
from diffuse_boost.spheres_in_cube_new.pipeline import PipelineState
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config helper (same style as in Heilbronn script)
# -----------------------------------------------------------------------------
EPS_SMALL = 1e-6

def _get_cfg(section, key, fallback):
    from diffuse_boost import cfg
    try:
        if isinstance(fallback, bool):
            return cfg.getboolean(section, key)
        if isinstance(fallback, int):
            return cfg.getint(section, key)
        if isinstance(fallback, float):
            return cfg.getfloat(section, key)
        return cfg.get(section, key)
    except Exception:
        return fallback
    
# -----------------------------------------------------------------------------
# Utilities: Penalty gradient, energy evaluation, and clearance maximization
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _unit_direction(dx0, dx1, dx2, dist):
    if dist >= EPS_SMALL:
        inv = 1.0 / dist
        return dx0 * inv, dx1 * inv, dx2 * inv
    v0 = np.random.normal()
    v1 = np.random.normal()
    v2 = np.random.normal()
    n = math.sqrt(v0*v0 + v1*v1 + v2*v2) + EPS_SMALL
    return v0/n, v1/n, v2/n

@njit(cache=True, fastmath=True)
def compute_EL_grad(X_rel, L, N, r):
    """
    Compute energy and gradient in the 'relative' coordinate frame by mapping
    through to absolute cube coordinates, computing the true overlap penalties
    there, and then chaining the gradient back.
    X_rel: array of shape (3N,), representing coords in [-L/2, L/2].
    """
    coords_rel = X_rel.reshape((N, 3))
    scale = (L - 2.0 * r) / L
    half = L / 2.0
    coords_abs = (coords_rel + half) * scale + r

    EL = 0.0
    grad_abs = np.zeros((N, 3), dtype=coords_abs.dtype)

    # Walls
    for i in range(N):
        for d in range(3):
            x = coords_abs[i, d]
            if x < r:
                over = r - x
                EL += over * over
                grad_abs[i, d] += -2.0 * over
            elif x > L - r:
                over = x - (L - r)
                EL += over * over
                grad_abs[i, d] += 2.0 * over

    # Sphere-sphere
    for i in range(N):
        xi0 = coords_abs[i, 0]
        xi1 = coords_abs[i, 1]
        xi2 = coords_abs[i, 2]
        for j in range(i + 1, N):
            dx0 = xi0 - coords_abs[j, 0]
            dx1 = xi1 - coords_abs[j, 1]
            dx2 = xi2 - coords_abs[j, 2]
            dist = math.sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2)
            over = 2.0 * r - dist
            if over > 0.0:
                EL += over * over
                u0, u1, u2 = _unit_direction(dx0, dx1, dx2, dist)
                g0 = 2.0 * over * u0
                g1 = 2.0 * over * u1
                g2 = 2.0 * over * u2
                grad_abs[i, 0] -= g0; grad_abs[j, 0] += g0
                grad_abs[i, 1] -= g1; grad_abs[j, 1] += g1
                grad_abs[i, 2] -= g2; grad_abs[j, 2] += g2

    grad_rel = grad_abs * scale
    return EL, grad_rel.ravel()

@njit(cache=True, fastmath=True)
def compute_EL(X_rel, L, N, r):
    EL, _ = compute_EL_grad(X_rel, L, N, r)
    return EL

@njit(cache=True, fastmath=True)
def maximize_clearance(X_rel, N, steps=10, step_size=0.01):
    coords = X_rel.reshape((N, 3)).copy()
    for _ in range(steps):
        grad = np.zeros_like(coords)
        for i in range(N):
            xi0, xi1, xi2 = coords[i, 0], coords[i, 1], coords[i, 2]
            for j in range(N):
                if i == j:
                    continue
                dx0 = xi0 - coords[j, 0]
                dx1 = xi1 - coords[j, 1]
                dx2 = xi2 - coords[j, 2]
                dist = math.sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2) + EPS_SMALL
                inv = 1.0 / dist
                grad[i, 0] += dx0 * inv
                grad[i, 1] += dx1 * inv
                grad[i, 2] += dx2 * inv
        coords += step_size * grad
    return coords.ravel()

@njit(cache=True, fastmath=True)
def _linf_norm_with_eps(g):
    m = 0.0
    for k in range(g.size):
        a = g[k]
        if a < 0: a = -a
        if a > m: m = a
    return m + EPS_SMALL

@njit(cache=True, fastmath=True)
def SRP(X_rel, L, N, r, Imax, m, sigma, beta):
    eta = sigma
    Xc = X_rel.copy()
    for _ in range(Imax):
        Xc += np.random.uniform(-eta, eta, size=Xc.shape[0])
        for __ in range(m):
            EL, g = compute_EL_grad(Xc, L, N, r)
            Xc -= (sigma * eta) * (g / _linf_norm_with_eps(g))
        eta *= beta
    return Xc

@njit(cache=True, fastmath=True)
def min_pairwise_distance(centers):
    N = centers.shape[0]
    best = 1e300
    for i in range(N):
        x0, x1, x2 = centers[i, 0], centers[i, 1], centers[i, 2]
        for j in range(i + 1, N):
            dx0 = x0 - centers[j, 0]
            dx1 = x1 - centers[j, 1]
            dx2 = x2 - centers[j, 2]
            d = math.sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2)
            if d < best:
                best = d
    return best

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
# Main generation: SRP + local_opt + optional physics push (TRAINING MODE)
# -----------------------------------------------------------------------------
def generate_dataset_push_srp(verbose=True):
    sec = "sample_generation_PP+PBTS"

    # Read parameters
    D        = _get_cfg(sec, "dimension",            3)
    L        = _get_cfg(sec, "bounding_box_width",   1.0)
    r        = _get_cfg(sec, "sphere_radius",        0.1)
    best_d   = _get_cfg(sec, "best_known_diameter",  2.0 * r)
    N        = _get_cfg(sec, "num_spheres",          10)
    M        = _get_cfg(sec, "num_samples",          100)
    dt       = _get_cfg(sec, "dt",                   1e-3)
    max_iter = _get_cfg(sec, "max_iter",             10000)
    tol      = _get_cfg(sec, "tol",                  1e-6)
    mode_bnd = _get_cfg(sec, "boundary_mode",        "reflect")

    Imax       = _get_cfg(sec, "srp_Imax",        500)
    m          = _get_cfg(sec, "srp_m",           20)
    sigma_frac = _get_cfg(sec, "srp_sigma_frac",  0.2)
    sigma      = sigma_frac * L
    beta       = _get_cfg(sec, "srp_beta",        0.95)
    tol_opt    = _get_cfg(sec, "srp_tol",         1e-8)
    maxiter_opt= _get_cfg(sec, "srp_maxiter",     300)
    num_srp_restarts = _get_cfg(sec, "num_srp_restarts", 15)

    physics_push_mode = _get_cfg(sec, "physics_push_mode", True)

    print(f"Generating dataset with N={N}, M={M}, SRP restarts={num_srp_restarts}, physics_push_mode={physics_push_mode}")

    # Output filenames (include N in the timestamp token)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stamp_with_N = f"N{N}_{timestamp_str}"
    day_stamp = datetime.now().strftime("%Y-%m-%d")

    base_metrics = _get_cfg(sec, "output_filename_metrics", "srp_metrics_{DATE}.csv")
    metrics_fn = base_metrics.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp) 

    base_data = _get_cfg(sec, "output_filename", "srp_data_{DATE}.pt")
    data_fn = base_data.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp) 

    base_sym = _get_cfg(sec, "output_filename_sym", data_fn.replace('.pt', '_sym.pt'))
    sym_fn = base_sym.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp) 

    base_top = _get_cfg(sec, "output_filename_top", "srp_top_{DATE}.pt")
    top_fn = base_top.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp) 

    base_sym_top = _get_cfg(sec, "output_filename_sym_top", top_fn.replace('.pt', '_sym.pt'))
    sym_top_fn = base_sym_top.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp) 

    metrics_dir = os.path.dirname(metrics_fn)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_fn, 'w') as mf:
        mf.write("sample,srp_restart,EL_before,EL_after,pre_push_min,post_push_min,excess\n")

    data = np.zeros((M * num_srp_restarts, D, N), dtype=np.float32)
    min_dists = []

    # For per-sample best stats (for top10 metrics)
    best_EL_before_list = [0.0] * M
    best_EL_after_list  = [0.0] * M
    best_pre_min_list   = [0.0] * M
    best_post_min_list  = [0.0] * M
    best_excess_list    = [float('inf')] * M
    best_restart_idx_list = [-1] * M

    for i in tqdm(range(M), desc=f"Generating sample - will be used for {num_srp_restarts} restarts"):
        if verbose:
            print(f"Generating sample {i+1}/{M}...")
        pts = sample_uniform_points(D, L, r, N)

        if physics_push_mode:
            centers0, _ = eliminate_overlaps_box(
                pts, r, [L] * D,
                max_iter=max_iter, dt=dt, tol=tol,
                boundary_mode=mode_bnd, visualize=False, verbose=False
            )
        else:
            centers0 = pts.copy()

        half = L / 2
        X0 = (centers0 - half).ravel()
        diffs0 = centers0[:, None, :] - centers0[None, :, :]
        init_min = np.min(np.linalg.norm(diffs0, axis=-1)[np.triu_indices(N, k=1)])
        best_centers = centers0.copy()
        best_min = init_min
        excess_sample = best_d - init_min
        if verbose:
            print(f"Sample {i+1}/{M}: initial min distance = {init_min:.6f}, excess = {excess_sample:.6f}")

        # "best" metrics for this sample
        best_EL_before = math.inf
        best_EL_after  = math.inf
        best_pre_min   = init_min
        best_post_min  = init_min
        best_excess    = best_d - init_min
        best_restart   = -1

        for k in range(num_srp_restarts):
            if verbose:
                print(f"  SRP restart {k+1}/{num_srp_restarts} for sample {i+1}/{M}")
            eps = 1e-6
            X_srp = SRP(X0, L, N, r, Imax, m, sigma, beta)
            X_srp_clip = np.clip(X_srp, -L/2 + eps, L/2 - eps)
            EL_before = compute_EL(X_srp_clip, L, N, r)
            if verbose:
                print(f"    SRP restart {k+1}/{num_srp_restarts}: EL before local_opt = {EL_before:.6f}")
            X_lo, EL_after = local_opt(X_srp_clip, L, N, r, tol_opt, maxiter_opt)
            if verbose:
                print(f"    SRP restart {k+1}/{num_srp_restarts}: EL after local_opt  = {EL_after:.6f}")

            coords = X_lo.reshape((N, 3))
            centers_opt = (coords + half) * ((L - 2 * r) / L) + r
            centers_opt = np.minimum(np.maximum(centers_opt, r + EPS_SMALL), L - r - EPS_SMALL)

            diffs_pre = centers_opt[:, None, :] - centers_opt[None, :, :]
            pre_min = np.min(np.linalg.norm(diffs_pre, axis=-1)[np.triu_indices(N, k=1)])

            if physics_push_mode:
                centers_k, _ = eliminate_overlaps_box(
                    centers_opt, r, [L] * D,
                    max_iter=max_iter, dt=dt, tol=tol,
                    boundary_mode=mode_bnd, visualize=False, verbose=False
                )
            else:
                centers_k = centers_opt

            diffs_k = centers_k[:, None, :] - centers_k[None, :, :]
            post_min = np.min(np.linalg.norm(diffs_k, axis=-1)[np.triu_indices(N, k=1)])
            excess = best_d - post_min
            if verbose:
                print(f"    min_after{'_physics_push' if physics_push_mode else ''} = {post_min:.6f}, excess = {excess:.6f}")

            with open(metrics_fn, 'a') as mf:
                mf.write(f"{i},{k+1},{EL_before:.6f},{EL_after:.6f},{pre_min:.6f},{post_min:.6f},{excess:.6f}\n")

            # Update best over restarts
            if post_min > best_min:
                best_min = post_min
                best_centers = centers_k.copy()
            if excess < best_excess:
                best_excess   = excess
                best_EL_before = EL_before
                best_EL_after  = EL_after
                best_pre_min   = pre_min
                best_post_min  = post_min
                best_restart   = k + 1

            data[i * num_srp_restarts + k] = centers_k.copy().T

        min_dists.append(best_min)
        best_EL_before_list[i] = best_EL_before
        best_EL_after_list[i]  = best_EL_after
        best_pre_min_list[i]   = best_pre_min
        best_post_min_list[i]  = best_post_min
        best_excess_list[i]    = best_excess
        best_restart_idx_list[i] = best_restart

        if verbose:
            print(f"Finished sample {i+1}/{M}, best_min = {best_min:.6f}\n")

    # Save full dataset
    data_dir = os.path.dirname(data_fn)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
    torch.save(torch.from_numpy(data), data_fn)
    if verbose:
        print(f"Saved full dataset to {data_fn}")

    # Symmetrized full dataset
    try:
        sym_data = apply_symmetries_to_data(data, L)
        sym_dir = os.path.dirname(sym_fn)
        if sym_dir:
            os.makedirs(sym_dir, exist_ok=True)
        torch.save(torch.from_numpy(sym_data), sym_fn)
        if verbose:
            print(f"Saved symmetrized dataset to {sym_fn}")
    except ValueError as e:
        print(f"Skipping symmetry enrichment: {e}")

    # Top-k subset (by min distance)
    k_top = max(1, int(np.ceil(0.25 * M)))
    best_idx = np.argsort(min_dists)[-k_top:]
    top_data = data[best_idx]
    top_dir = os.path.dirname(top_fn)
    if top_dir:
        os.makedirs(top_dir, exist_ok=True)
    torch.save(torch.from_numpy(top_data), top_fn)
    if verbose:
        print(f"Saved top {k_top} samples to {top_fn}")

    # Symmetrized top subset
    try:
        sym_top = apply_symmetries_to_data(top_data, L)
        sym_top_dir = os.path.dirname(sym_top_fn)
        if sym_top_dir:
            os.makedirs(sym_top_dir, exist_ok=True)
        torch.save(torch.from_numpy(sym_top), sym_top_fn)
        if verbose:
            print(f"Saved symmetrized top dataset to {sym_top_fn}")
    except ValueError as e:
        print(f"Skipping symmetry enrichment for top samples: {e}")

    # -----------------------------------------------------------------
    # Save metrics of TOP 10 samples (minimal excess) into separate file
    # -----------------------------------------------------------------
    num_top10 = min(10, M)
    order_top10 = np.argsort(best_excess_list)[:num_top10]  # smaller excess is better
    metrics_top10_fn = metrics_fn.replace(".csv", "_top10.csv")
    with open(metrics_top10_fn, "w") as mf:
        mf.write("sample,EL_before,EL_after,pre_push_min,post_push_min,excess,srp_restart\n")
        for idx in order_top10:
            mf.write(
                f"{idx},"
                f"{best_EL_before_list[idx]:.6f},"
                f"{best_EL_after_list[idx]:.6f},"
                f"{best_pre_min_list[idx]:.6f},"
                f"{best_post_min_list[idx]:.6f},"
                f"{best_excess_list[idx]:.6f},"
                f"{best_restart_idx_list[idx]}\n"
            )
    if verbose:
        print(f"Saved top-10 metrics to {metrics_top10_fn}")

# -----------------------------------------------------------------------------
# Multi-sphere-count training generation
# -----------------------------------------------------------------------------
def generate_dataset_push_srp_different_sphere_count():
    secmul = "sample_generation_PP+PBTS_multiple_sphere_num"
    secgen = "sample_generation_PP+PBTS"
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    num_spheres_start = _get_cfg(secmul, "num_spheres_start", 5)
    num_spheres_end   = _get_cfg(secmul, "num_spheres_end",   10)

    base_output_filename         = _get_cfg(secmul, "output_filename",           "srp_mult_{SPHERE_NUM}_{DATE}.pt").replace("{DATE}", timestamp_str)
    base_output_filename_top     = _get_cfg(secmul, "output_filename_top",       "srp_mult_top_{SPHERE_NUM}_{DATE}.pt").replace("{DATE}", timestamp_str)
    base_output_filename_metrics = _get_cfg(secmul, "output_filename_metrics",   "srp_mult_metrics_{SPHERE_NUM}_{DATE}.csv").replace("{DATE}", timestamp_str)
    base_output_filename_metrics_excess = _get_cfg(secmul, "output_filename_metrics_excess", "srp_mult_metrics_excess_{SPHERE_NUM}_{DATE}.csv").replace("{DATE}", timestamp_str)

    print("base_output_filename: ", base_output_filename)
    print(f"generate multiple packing starting from {num_spheres_start} spheres, up to {num_spheres_end} spheres per packing")
    box_sizes = load_best_results()
    bar = tqdm(range(num_spheres_start, num_spheres_end+1))
    for sphere_num in bar:
        bar.set_description(f"Generating packings from {num_spheres_start} to {num_spheres_end}, Currently at {sphere_num} spheres", refresh=True)
        radius = 1 / box_sizes[sphere_num]

        # Parameters (set in cfg so generate_dataset_push_srp reads them)
        cfg.set(secgen, "num_spheres",          str(sphere_num))
        cfg.set(secgen, "bounding_box_width",   "1.0")
        cfg.set(secgen, "sphere_radius",        str(radius))
        cfg.set(secgen, "best_known_diameter",  str(2*radius))

        # File paths for this sphere_num
        cfg.set(secgen, "output_filename",             base_output_filename.replace("{SPHERE_NUM}", str(sphere_num)))
        cfg.set(secgen, "output_filename_top",         base_output_filename_top.replace("{SPHERE_NUM}", str(sphere_num)))
        cfg.set(secgen, "output_filename_metrics",     base_output_filename_metrics.replace("{SPHERE_NUM}", str(sphere_num)))
        cfg.set(secgen, "output_filename_metrics_excess", base_output_filename_metrics_excess.replace("{SPHERE_NUM}", str(sphere_num)))

        generate_dataset_push_srp(verbose=False)

def load_metrics_PP_p_PBTS(filename):
    data_excess = []
    with open(filename) as m_file:
        lines = m_file.readlines()
        for data_text in lines[1:]:
            data_string = data_text.split(",")
            data_excess.append(float(data_string[-1]))
    return data_excess

# -----------------------------------------------------------------------------
# FINAL PUSH MODE: read packings from file, apply SRP + local opt + optional physics push
# -----------------------------------------------------------------------------
def final_push_existing_samples():
    sec = "sample_generation_PP+PBTS"

    # Shared geometric / physical parameters
    D        = _get_cfg(sec, "dimension",            3)
    L        = _get_cfg(sec, "bounding_box_width",   1.0)
    r        = _get_cfg(sec, "sphere_radius",        0.1)
    best_d   = _get_cfg(sec, "best_known_diameter",  2.0 * r)
    N        = _get_cfg(sec, "num_spheres",          10)
    dt       = _get_cfg(sec, "dt",                   1e-3)
    max_iter = _get_cfg(sec, "max_iter",             10000)
    tol      = _get_cfg(sec, "tol",                  1e-6)
    mode_bnd = _get_cfg(sec, "boundary_mode",        "reflect")

    # SRP / local optimization parameters
    Imax       = _get_cfg(sec, "srp_Imax",        500)
    m          = _get_cfg(sec, "srp_m",           20)
    sigma_frac = _get_cfg(sec, "srp_sigma_frac",  0.2)
    sigma      = sigma_frac * L
    beta       = _get_cfg(sec, "srp_beta",        0.95)
    tol_opt    = _get_cfg(sec, "srp_tol",         1e-8)
    maxiter_opt= _get_cfg(sec, "srp_maxiter",     300)
    num_srp_restarts = _get_cfg(sec, "num_srp_restarts", 15)

    physics_push_mode = _get_cfg(sec, "physics_push_mode", True)

    # IO paths for final push
    stamp      = datetime.now().strftime("%Y-%m-%d")
    out_dir    = _get_cfg(sec, "final_push_output", "./outputs_spheres_push")
    out_dir = os.path.join(out_dir, stamp)
    input_path = _get_cfg(sec, "final_push_input",  "")

    assert isinstance(input_path, str) and len(input_path) > 0 and os.path.exists(input_path), \
        "Set sample_generation_PP+PBTS.final_push_input to a valid .pt file"

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"spheres_srp_pushed_N{N}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"spheres_metrics_pushed_N{N}_{stamp}.csv")

    loaded = torch.load(input_path, map_location="cpu")
    arr = loaded.detach().cpu().numpy() if isinstance(loaded, torch.Tensor) else None
    if arr is None or arr.ndim != 3:
        raise ValueError(f"Expected a tensor at final_push_input, got shape {getattr(loaded, 'shape', None)}")

    # Expect either (M, D, N) or (M, N, D)
    if arr.shape[1] == D:
        M_in, d_in, N_in = arr.shape
    elif arr.shape[-1] == D:
        arr = np.transpose(arr, (0, 2, 1))  # to (M, D, N)
        M_in, d_in, N_in = arr.shape
    else:
        raise ValueError(f"Second or last dimension must be {D}; got shape {arr.shape}")

    if N_in != N:
        raise ValueError(f"Config num_spheres={N} but input samples have N={N_in}")

    K = M_in
    print(f"Pushing {K} loaded samples (N={N}, D={D})")

    with open(metrics_fn, "w") as mf:
        mf.write("sample,srp_restart,EL_before,EL_after,pre_push_min,post_push_min,excess\n")

    data_out = np.zeros((K * num_srp_restarts, D, N), dtype=np.float32)
    half = L / 2.0
    scale = (L - 2.0 * r) / L

    min_dists = []

    # Arrays for top-10 stats
    best_EL_before_list = [0.0] * K
    best_EL_after_list  = [0.0] * K
    best_pre_min_list   = [0.0] * K
    best_post_min_list  = [0.0] * K
    best_excess_list    = [float('inf')] * K
    best_restart_idx_list = [-1] * K

    for s in tqdm(range(K), desc="[Pushing Samples]", unit="sample"):
        centers_in = arr[s].T.astype(np.float64)  # (N, D)

        if physics_push_mode:
            centers0, _ = eliminate_overlaps_box(
                centers_in, r, [L] * D,
                max_iter=max_iter, dt=dt, tol=tol,
                boundary_mode=mode_bnd, visualize=False, verbose=False
            )
        else:
            centers0 = centers_in.copy()

        X0 = (centers0 - half).ravel()

        # per-sample best
        best_post_min_sample = -1.0
        best_excess_sample = float('inf')
        best_EL_before = math.inf
        best_EL_after  = math.inf
        best_pre_min   = 0.0
        best_restart   = -1
        best_centers   = centers0.copy()

        for k in range(num_srp_restarts):
            eps = 1e-6
            X_srp = SRP(X0, L, N, r, Imax, m, sigma, beta)
            X_srp_clip = np.clip(X_srp, -L/2 + eps, L/2 - eps)
            EL_before = compute_EL(X_srp_clip, L, N, r)

            X_lo, EL_after = local_opt(X_srp_clip, L, N, r, tol_opt, maxiter_opt)

            coords = X_lo.reshape((N, 3))
            centers_opt = (coords + half) * scale + r
            centers_opt = np.minimum(np.maximum(centers_opt, r + EPS_SMALL), L - r - EPS_SMALL)

            diffs_pre = centers_opt[:, None, :] - centers_opt[None, :, :]
            pre_min = np.min(np.linalg.norm(diffs_pre, axis=-1)[np.triu_indices(N, k=1)])

            if physics_push_mode:
                centers_k, _ = eliminate_overlaps_box(
                    centers_opt, r, [L] * D,
                    max_iter=max_iter, dt=dt, tol=tol,
                    boundary_mode=mode_bnd, visualize=False, verbose=False
                )
            else:
                centers_k = centers_opt

            diffs_k = centers_k[:, None, :] - centers_k[None, :, :]
            post_min = np.min(np.linalg.norm(diffs_k, axis=-1)[np.triu_indices(N, k=1)])
            excess = best_d - post_min

            with open(metrics_fn, "a") as mf:
                mf.write(f"{s},{k+1},{EL_before:.6f},{EL_after:.6f},{pre_min:.6f},{post_min:.6f},{excess:.6f}\n")

            if post_min > best_post_min_sample:
                best_post_min_sample = post_min
                best_centers = centers_k.copy()
            if excess < best_excess_sample:
                best_excess_sample = excess
                best_EL_before = EL_before
                best_EL_after  = EL_after
                best_pre_min   = pre_min
                best_restart   = k + 1

            data_out[s * num_srp_restarts + k] = centers_k.copy().T.astype(np.float32)

        #data_out[s] = best_centers.T.astype(np.float32)
        min_dists.append(best_post_min_sample)
        best_EL_before_list[s] = best_EL_before
        best_EL_after_list[s]  = best_EL_after
        best_pre_min_list[s]   = best_pre_min
        best_post_min_list[s]  = best_post_min_sample
        best_excess_list[s]    = best_excess_sample
        best_restart_idx_list[s] = best_restart

    torch.save(torch.from_numpy(data_out), dataset_fn)
    print(f"\nSaved pushed dataset:  {dataset_fn}")
    print(f"Saved pushed metrics:  {metrics_fn}")

    # Top-10 metrics (minimal excess)
    num_top10 = min(10, K)
    order_top10 = np.argsort(best_excess_list)[:num_top10]
    metrics_top10_fn = metrics_fn.replace(".csv", "_top10.csv")
    with open(metrics_top10_fn, "w") as mf:
        mf.write("sample,EL_before,EL_after,pre_push_min,post_push_min,excess,srp_restart\n")
        for idx in order_top10:
            mf.write(
                f"{idx},"
                f"{best_EL_before_list[idx]:.6f},"
                f"{best_EL_after_list[idx]:.6f},"
                f"{best_pre_min_list[idx]:.6f},"
                f"{best_post_min_list[idx]:.6f},"
                f"{best_excess_list[idx]:.6f},"
                f"{best_restart_idx_list[idx]}\n"
            )
    print(f"Saved top-10 pushed metrics: {metrics_top10_fn}")

# -----------------------------------------------------------------------------
# Main mode switch
# -----------------------------------------------------------------------------
def main(state:PipelineState=None):
    main_sec = "sample_generation_PP+PBTS"
    mode = _get_cfg(main_sec, "mode", "training_set_gen").strip().lower()

    if mode == "training_set_gen":
        # Only look at the multi-sphere section here
        multi_sec = "sample_generation_PP+PBTS_multiple_sphere_num"
        multi_active = _get_cfg(multi_sec, "active", False)

        if multi_active:
            generate_dataset_push_srp_different_sphere_count()
        else:
            generate_dataset_push_srp(verbose=False)

    elif mode == "final_push":
        final_push_existing_samples()

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'training_set_gen' or 'final_push'.")

if __name__ == "__main__":
    main()