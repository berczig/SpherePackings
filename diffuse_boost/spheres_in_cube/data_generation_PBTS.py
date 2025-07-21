import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import itertools
from scipy.optimize import minimize
from diffuse_boost import cfg

# -----------------------------------------------------------------------------
# PBTS core functions
# -----------------------------------------------------------------------------

def compute_EL_grad(X, L, N):
    coords = X.reshape((N, 3))
    EL = 0.0
    grad = np.zeros_like(coords)
    half = L / 2
    # wall overlaps
    for i, xi in enumerate(coords):
        for d in range(3):
            over = max(0, 1 - (half - xi[d]))
            EL += over * over
            if over > 0:
                grad[i, d] -= 2 * over
            over = max(0, 1 - (half + xi[d]))
            EL += over * over
            if over > 0:
                grad[i, d] += 2 * over
    # sphere overlaps
    for i in range(N):
        for j in range(i + 1, N):
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            over = max(0, 2 - dist)
            if over > 0:
                EL += over * over
                g = diff / (dist + 1e-12) * 2 * over
                grad[i] += g
                grad[j] -= g
    return EL, grad.ravel()


def URP(X, eta0=0.8):
    return X + np.random.uniform(-eta0, eta0, size=X.shape)


def SRP(X, L, N, Imax=5, m=3, sigma=0.1, beta=0.9):
    eta = sigma
    Xc = X.copy()
    for _ in range(Imax):
        Xc += np.random.uniform(-eta, eta, size=Xc.shape)
        for __ in range(m):
            EL, g = compute_EL_grad(Xc, L, N)
            g = g.reshape((N, 3))
            norm = np.linalg.norm(g, ord=np.inf) + 1e-12
            Xc -= (sigma * eta) * (g / norm).ravel()
        eta *= beta
    return Xc


def local_opt(X, L, N, tol=1e-13, maxiter=200):
    res = minimize(lambda x: compute_EL_grad(x, L, N), X,
                   method='L-BFGS-B', jac=True,
                   options={'ftol': tol, 'gtol': tol, 'maxiter': maxiter})
    return res.x, res.fun


def threshold_search(X, L, N, flag, max_iter=1000):
    Xb, ELb = X, compute_EL_grad(X, L, N)[0]
    for _ in range(max_iter):
        Xt = URP(Xb) if flag == 0 else SRP(Xb, L, N)
        Xt, ELt = local_opt(Xt, L, N)
        if ELt < ELb:
            Xb, ELb = Xt, ELt
        else:
            break
    return Xb, ELb


def adjust_container(X, L, N):
    lo, hi = L * 0.99, L
    for _ in range(20):
        mid = (lo + hi) / 2
        EL, _ = compute_EL_grad(X, mid, N)
        if EL < 1e-25:
            hi = mid
        else:
            lo = mid
    return X, hi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sample_uniform_points(dimension, bounding_box_width, radius, num_points):
    low, high = radius, bounding_box_width - radius
    if high <= low:
        raise ValueError("bounding_box_width must exceed 2*radius.")
    return np.random.uniform(low, high, size=(num_points, dimension))


def get_cube_symmetry_matrices(dim):
    mats = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product([-1, 1], repeat=dim):
            M = np.zeros((dim, dim))
            for i in range(dim):
                M[i, perm[i]] = signs[i]
            mats.append(M)
    return mats


def apply_symmetries_to_data(data, box_width):
    M, D, N = data.shape
    mats = get_cube_symmetry_matrices(D)
    out = np.zeros((M * len(mats), D, N), dtype=data.dtype)
    center = box_width / 2
    idx = 0
    for i in range(M):
        coords = data[i].T
        for mat in mats:
            T = (mat @ (coords - center).T).T + center
            out[idx] = T.T
            idx += 1
    return out

# -----------------------------------------------------------------------------
# Main PBTS-based data-generation
# -----------------------------------------------------------------------------

def generate_dataset_pbts():
    sec = "sample_generation_PESC"
    D = cfg.getint(sec, "dimension")
    L_box = cfg.getfloat(sec, "bounding_box_width")
    if D != 3:
        raise NotImplementedError("PBTS only supports 3D currently.")
    r0 = cfg.getfloat(sec, "sphere_radius")
    best_known_diam = cfg.getfloat(sec, "best_known_diameter", fallback=2*r0)
    N = cfg.getint(sec, "num_spheres")
    M = cfg.getint(sec, "num_samples")
    tmax = cfg.getfloat(sec, "tmax",fallback=300.0)

    data = np.zeros((M, D, N), dtype=np.float32)
    min_dists, avg_dists = [], []

    # metrics
    metrics_fn = cfg.get(sec, "output_filename_metrics")
    excess_fn = cfg.get(sec, "output_filename_metrics_excess")
    os.makedirs(os.path.dirname(metrics_fn), exist_ok=True)
    with open(metrics_fn, 'a') as mf:
        mf.write(f"# PBTS based packing\n")
        mf.write(f"N={N},r0={r0},best_known_diam={best_known_diam},samples={M},tmax={tmax}\n")
        mf.write("# idx,min_dist,avg_dist,excess\n")
    with open(excess_fn, 'a') as mf:
        mf.write("idx,min_dist,avg_dist,excess\n")

    for idx in range(M):
        pts = sample_uniform_points(D, L_box, r0, N)
        # Phase 1
        p = cfg.getfloat(sec, "p0")
        L = (4 * np.pi * N / (3 * p)) ** (1/3)
        X = (pts - L_box/2) * (L / L_box)
        flag = 0
        X, EL = threshold_search(X, L, N, flag)
        while EL < 1e-25:
            p += 1e-3 * np.random.rand()
            L = (4 * np.pi * N / (3 * p)) ** (1/3)
            X, EL = threshold_search(X, L, N, flag)
        X, Lb = adjust_container(X, L, N)
        X_best, L_best = X.copy(), Lb
        # Phase 2
        import time
        t0 = time.time()
        while time.time() - t0 < tmax:
            X0 = np.random.uniform(-Lb/2, Lb/2, size=(3*N,))
            X1, EL1 = threshold_search(X0, Lb, N, flag)
            if EL1 < 1e-25:
                X1, L1 = adjust_container(X1, Lb, N)
                if L1 < L_best:
                    X_best, L_best = X1.copy(), L1
                else:
                    flag ^= 1
            else:
                flag ^= 1
        # scale to [0,1]
        centers = X_best.reshape((N,3))
        scale = L_box / L_best
        centers_unit = (centers + L_best/2) * scale
        r_final = r0 * scale
        data[idx] = centers_unit.T
        # metrics
        coords = centers_unit
        diffs = coords[:, None, :] - coords[None, :, :]
        dmat = np.linalg.norm(diffs, axis=-1)
        i1, j1 = np.triu_indices(N, k=1)
        pd = dmat[i1, j1]
        mn, av = pd.min(), pd.mean()
        excess = best_known_diam - mn
        min_dists.append(mn); avg_dists.append(av)
        with open(metrics_fn, 'a') as mf:
            mf.write(f"{idx},{mn:.6f},{av:.6f},{excess:.6f}\n")
        if excess < 0:
            with open(excess_fn, 'a') as mf:
                mf.write(f"{idx},{mn:.6f},{av:.6f},{excess:.6f}\n")
        print(f"Sample {idx+1}/{M}: min={mn:.6f}, avg={av:.6f}, excess={excess:.6f}, r={r_final:.6f}")

    # save dataset
    data_fn = cfg.get(sec, "output_filename")
    os.makedirs(os.path.dirname(data_fn), exist_ok=True)
    torch.save(torch.from_numpy(data), data_fn)
    print(f"Saved full dataset to {data_fn}")

    # symmetries
    try:
        sym_data = apply_symmetries_to_data(data, 1.0)
        sym_fn = cfg.get(sec, "output_filename_sym", fallback=data_fn.replace('.pt','_sym.pt'))
        torch.save(torch.from_numpy(sym_data), sym_fn)
        print(f"Saved symmetrized data to {sym_fn}")
    except ValueError:
        pass

    # top 25%
    k = max(1, int(np.ceil(0.25 * M)))
    best_idx = np.argsort(min_dists)[-k:]
    top_data = data[best_idx]
    top_fn = cfg.get(sec, "output_filename_top")
    torch.save(torch.from_numpy(top_data), top_fn)
    print(f"Saved top {k} samples to {top_fn}")

    try:
        sym_top = apply_symmetries_to_data(top_data, 1.0)
        sym_top_fn = cfg.get(sec, "output_filename_sym_top", fallback=top_fn.replace('.pt','_sym.pt'))
        torch.save(torch.from_numpy(sym_top), sym_top_fn)
        print(f"Saved symmetrized top data to {sym_top_fn}")
    except ValueError:
        pass

# -----------------------------------------------------------------------------
# Dataset & DataLoader
# -----------------------------------------------------------------------------

class SpherePackingDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path)
        print(f"Loaded {path}, shape {self.data.shape}")
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]


def get_data_loader(batch_size: int, dataset_path: str):
    ds = SpherePackingDataset(dataset_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    generate_dataset_pbts()
