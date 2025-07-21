import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import itertools
from scipy.optimize import minimize
from diffuse_boost import cfg

# -----------------------------------------------------------------------------
# SSR core functions
# -----------------------------------------------------------------------------

def energy_and_grad(X, r, L, n):
    coords = X.reshape((n,3))
    U = 0.0
    grad = np.zeros_like(coords)
    half = L/2
    # wall overlaps
    for i, xi in enumerate(coords):
        for d in range(3):
            over = max(0, half - r - xi[d])
            U += over*over
            if over>0:
                grad[i,d] -= 2*over
            over = max(0, half - r + xi[d])
            U += over*over
            if over>0:
                grad[i,d] += 2*over
    # sphere overlaps
    for i in range(n):
        for j in range(i+1, n):
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            over = max(0, 2*r - dist)
            if over>0:
                U += over*over
                g = diff/(dist+1e-12) * (2*over)
                grad[i] +=  g
                grad[j] -=  g
    return U, grad.ravel()

def minU(k, energies):
    return np.argsort(energies)[:k]

def maxU(k, energies):
    return np.argsort(energies)[-k:]

def invert_subset(X, subset):
    X2 = X.reshape(-1,3).copy()
    for i in subset:
        X2[i] = -X2[i]
    return X2.ravel()

def ssr_scan(X0, r, L, n):
    # local optimize (A0)
    res = minimize(lambda x: energy_and_grad(x, r, L, n),
                   X0, method='L-BFGS-B', jac=True,
                   options={'ftol':1e-16,'gtol':1e-16,'maxiter':5000})
    X_loc = res.x
    U0, _ = energy_and_grad(X_loc, r, L, n)
    # per-sphere energies
    coords = X_loc.reshape((n,3))
    sphere_U = np.zeros(n)
    for i in range(n):
        Ui = 0.0
        # walls
        for d in range(3):
            Ui += max(0, (L/2 - r) - coords[i,d])**2
            Ui += max(0, (L/2 - r) + coords[i,d])**2
        # other spheres
        for j in range(n):
            if i==j: continue
            dist = np.linalg.norm(coords[i]-coords[j])
            Ui += max(0, 2*r - dist)**2
        sphere_U[i] = Ui

    best_X, best_U = X_loc, U0
    # generate n(n-1)/2 SSR children
    for i in range(1, n+1):
        small = minU(i, sphere_U)
        for j in range(1, i):
            big = maxU(j, sphere_U[small])
            trial = invert_subset(X_loc, small[big])
            res_t = minimize(lambda x: energy_and_grad(x, r, L, n),
                             trial, method='L-BFGS-B', jac=True,
                             options={'ftol':1e-16,'gtol':1e-16,'maxiter':2000})
            U_t, _ = energy_and_grad(res_t.x, r, L, n)
            if U_t < best_U:
                best_U, best_X = U_t, res_t.x
    return best_X, best_U

def serial_sym_reloc(N, r0, L_init, scans=10):
    """
    Returns: centers array of shape (N,3), minimal side-length L_final
    for fixed radius r0.
    """
    X = np.random.uniform(-L_init/2, L_init/2, size=(3*N,))
    L = L_init
    best_U = np.inf
    best_X = X.copy()
    for _ in range(scans):
        # Print current state
        print(f"SSR scan: L={L:.6f}, best_U={best_U:.6f}")
        # Perform SSR scan
        X, U = ssr_scan(X, r0, L, N)
        if U < 1e-16 and U < best_U:
            best_U = U
            best_X = X.copy()
            L *= 0.999     # shrink container
            X = best_X
        else:
            break
    return best_X.reshape(N,3), L

# -----------------------------------------------------------------------------
# Helpers from your original script
# -----------------------------------------------------------------------------

def sample_uniform_points(dimension, bounding_box_width, radius, num_points):
    low, high = radius, bounding_box_width - radius
    if high <= low:
        raise ValueError("bounding_box_width must exceed 2*radius.")
    return np.random.uniform(low, high, size=(num_points, dimension))

def get_cube_symmetry_matrices(dim):
    mats = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product([-1,1], repeat=dim):
            M = np.zeros((dim,dim),float)
            for i in range(dim):
                M[i, perm[i]] = signs[i]
            mats.append(M)
    return mats

def apply_symmetries_to_data(data, box_width):
    M, D, N = data.shape
    if D != 3:
        raise ValueError("Symmetry only implemented for 3D.")
    mats = get_cube_symmetry_matrices(D)
    out = np.zeros((M*len(mats), D, N), dtype=data.dtype)
    center = box_width/2
    idx = 0
    for i in range(M):
        coords = data[i].T  # (N,3)
        for mat in mats:
            T = (mat @ (coords - center).T).T + center
            out[idx] = T.T
            idx += 1
    return out

# -----------------------------------------------------------------------------
# Main SSR‐based data‐generation routine
# -----------------------------------------------------------------------------

def generate_dataset_ssr():
    sec = "sample_generation_PESC"
    D = cfg.getint(sec, "dimension")
    if D != 3:
        raise NotImplementedError("SSR version currently only supports 3D.")
    L_box = cfg.getfloat(sec, "bounding_box_width")
    r0    = cfg.getfloat(sec, "sphere_radius")
    best_known_diam = cfg.getfloat(sec, "best_known_diameter", fallback=2*r0)
    N     = cfg.getint(sec, "num_spheres")
    M     = cfg.getint(sec, "num_samples")
    scans = cfg.getint(sec, "ssr_scans", fallback=10)

    data = np.zeros((M, D, N), dtype=np.float32)
    min_dists, avg_dists = [], []

    # prepare metrics files
    metrics_fn = cfg.get(sec, "output_filename_metrics")
    excess_fn  = cfg.get(sec, "output_filename_metrics_excess")
    os.makedirs(os.path.dirname(metrics_fn), exist_ok=True)
    with open(metrics_fn, 'a') as mf:
        mf.write(f"# SSR based packing\n")
        mf.write(f"N={N},init_r={r0},best_known_diam={best_known_diam},samples={M},ssr_scans={scans}\n")
        mf.write("# idx,min_dist,avg_dist,excess\n")
    with open(excess_fn, 'a') as mf:
        mf.write("idx,min_dist,avg_dist,excess\n")

    for idx in range(M):
        print(f"Generating sample {idx+1}/{M}...")
        # 1) sample
        pts = sample_uniform_points(D, L_box, r0, N)
        # 2) SSR run
        centers_ssr, L_ssr = serial_sym_reloc(N, r0, L_box, scans=scans)
        # 3) scale into [0,1]^3
        print(f"SSR scan {idx+1}/{M} done, L_ssr={L_ssr:.6f}")
        scale = L_box / L_ssr
        centers_unit = (centers_ssr + L_ssr/2) * scale
        r_final = r0 * scale
        # 4) store
        data[idx] = centers_unit.T.astype(np.float32)
        # 5) metrics
        # centers_unit: shape (N,3)
        coords = centers_unit
        diffs  = coords[:, None, :] - coords[None, :, :]
        dmat   = np.linalg.norm(diffs, axis=-1)
        i1, j1 = np.triu_indices(N, k=1)
        pdist  = dmat[i1, j1]

        mn = float(pdist.min())
        av = float(pdist.mean())

        excess = best_known_diam - mn
        min_dists.append(mn)
        avg_dists.append(av)
        with open(metrics_fn, 'a') as mf:
            mf.write(f"{idx},{mn:.6f},{av:.6f},{excess:.6f}\n")
        if excess < 0:
            with open(excess_fn, 'a') as mf:
                mf.write(f"{idx},{mn:.6f},{av:.6f},{excess:.6f}\n")
        print(f"[{idx+1}/{M}] min={mn:.6f}, avg={av:.6f}, excess={excess:.6f}, r={r_final:.6f}")

    # save dataset
    data_fn = cfg.get(sec, "output_filename")
    torch.save(torch.from_numpy(data), data_fn)
    print(f"Saved full dataset to {data_fn}")

    # symmetries
    try:
        sym_data = apply_symmetries_to_data(data, 1.0)
        sym_fn = cfg.get(sec, "output_filename_sym",
                        fallback=data_fn.replace('.pt','_sym.pt'))
        torch.save(torch.from_numpy(sym_data), sym_fn)
        print(f"Saved symmetrized data ({sym_data.shape[0]} samples) to {sym_fn}")
    except ValueError as e:
        print(f"Skipping symmetry: {e}")

    # top-25% samples
    k = max(1, int(np.ceil(0.25*M)))
    best_idx = np.argsort(min_dists)[-k:]
    top_data = data[best_idx]
    top_fn = cfg.get(sec, "output_filename_top")
    torch.save(torch.from_numpy(top_data), top_fn)
    print(f"Saved top {k} samples to {top_fn}")

    try:
        sym_top = apply_symmetries_to_data(top_data, 1.0)
        sym_top_fn = cfg.get(sec, "output_filename_sym_top",
                             fallback=top_fn.replace('.pt','_sym.pt'))
        torch.save(torch.from_numpy(sym_top), sym_top_fn)
        print(f"Saved symmetrized top data to {sym_top_fn}")
    except ValueError as e:
        print(f"Skipping top-symmetry: {e}")

# -----------------------------------------------------------------------------
# Dataset & DataLoader (unchanged)
# -----------------------------------------------------------------------------

class SpherePackingDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path)
        print(f"Loaded {path}, shape {self.data.shape}")
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, idx): return self.data[idx]

def get_data_loader(batch_size: int, dataset_path: str):
    ds = SpherePackingDataset(dataset_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    generate_dataset_ssr()
