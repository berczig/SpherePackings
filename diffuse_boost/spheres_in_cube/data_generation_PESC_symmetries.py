import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import itertools
from diffuse_boost import cfg
from diffuse_boost.spheres_in_cube.physics_push_PESC import eliminate_overlaps_box

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sample_uniform_points(
    dimension: int,
    bounding_box_width: float,
    radius: float,
    num_points: int
) -> np.ndarray:
    """
    Sample exactly num_points uniformly in [radius, bounding_box_width - radius]^d.
    Returns array of shape (num_points, dimension).
    """
    low = radius
    high = bounding_box_width - radius
    if high <= low:
        raise ValueError("bounding_box_width must exceed 2*radius.")
    return np.random.uniform(low, high, size=(num_points, dimension))

def get_cube_symmetry_matrices(dim: int) -> list[np.ndarray]:
    """
    Generate all signed permutation matrices of size dim x dim (hyperoctahedral group B_dim),
    here for dim=3 gives 48 cube symmetries (including reflections).
    """
    mats = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product([-1, 1], repeat=dim):
            mat = np.zeros((dim, dim), dtype=float)
            for i in range(dim):
                mat[i, perm[i]] = signs[i]
            mats.append(mat)
    return mats


def apply_symmetries_to_data(data: np.ndarray, box_width: float) -> np.ndarray:
    """
    Given data array of shape (M, D, N), apply all cube symmetries to each sample.
    Returns enriched data of shape (M * num_sym, D, N).
    """
    M, D, N = data.shape
    if D != 3:
        raise ValueError("Symmetry group of cube implemented only for 3D data.")
    mats = get_cube_symmetry_matrices(D)
    num_sym = len(mats)
    enriched = np.zeros((M * num_sym, D, N), dtype=data.dtype)
    center = box_width / 2.0
    idx_out = 0
    for idx in range(M):
        centers = data[idx]            # shape (D, N)
        coords = centers.T             # shape (N, D)
        for mat in mats:
            transformed = (mat @ (coords - center).T).T + center  # (N, D)
            enriched[idx_out] = transformed.T  # back to (D, N)
            idx_out += 1
    return enriched

# -----------------------------------------------------------------------------
# Main data‐generation routine with streaming metrics
# -----------------------------------------------------------------------------

def generate_dataset_push():
    sec = "sample_generation_PESC"
    D = cfg.getint(sec, "dimension")
    L = cfg.getfloat(sec, "bounding_box_width")
    r = cfg.getfloat(sec, "sphere_radius")
    # fallback best_known_diameter to 2*r if not set
    best_known_diameter = cfg.getfloat(sec, "best_known_diameter", fallback=2*r)
    N = cfg.getint(sec, "num_spheres")
    M = cfg.getint(sec, "num_samples")
    dt = cfg.getfloat(sec, "dt")
    max_iter = cfg.getint(sec, "max_iter")
    tol = cfg.getfloat(sec, "tol")
    mode = cfg.get(sec, "boundary_mode")  # "clamp" or "stophit"
    data_fn = cfg.get(sec, "output_filename")
    top_fn = cfg.get(sec, "output_filename_top")
    metrics_fn = cfg.get(sec, "output_filename_metrics")
    excess_fn = cfg.get(sec, "output_filename_metrics_excess")

    # Prepare container for all centers
    data = np.zeros((M, D, N), dtype=np.float32)
    min_dists = []
    avg_dists = []

    # Initialize metrics files and clean up if they already exist
    for fn in (metrics_fn, excess_fn):
        dirn = os.path.dirname(fn)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
    with open(metrics_fn, 'a') as mf:
        # Write the parameters to the metrics file
        mf.write(f"num_spheres={N},sphere_radius={r},best_known_diameter={best_known_diameter},num_samples={M},dt={dt},max_iter={max_iter},tol={tol},boundary_mode={mode}\n")
        # Write a horizontal line for clarity
        mf.write("#" * 80 + "\n")
        # Write the header
        mf.write("sample_index,min_distance,avg_distance,excess\n")
    with open(excess_fn, 'a') as mf:
        mf.write("sample_index,min_distance,avg_distance,excess\n")

    # Generate samples
    for idx in range(M):
        # 1) sample uniform
        pts = sample_uniform_points(D, L, r, N)
        # 2) remove overlaps
        final_centers, _ = eliminate_overlaps_box(
            pts, r, [L]*D,
            max_iter=max_iter, dt=dt, tol=tol,
            boundary_mode=mode, visualize=False
        )
        # 3) store
        data[idx] = final_centers.T
        # 4) compute pairwise dists
        diffs = final_centers[:, None, :] - final_centers[None, :, :]
        dmat = np.linalg.norm(diffs, axis=-1)
        i1, j1 = np.triu_indices(N, k=1)
        pdist = dmat[i1, j1]
        mn = float(pdist.min())
        av = float(pdist.mean())
        overlap_amt = best_known_diameter - mn
        min_dists.append(mn)
        avg_dists.append(av)
        # 5) log metrics
        with open(metrics_fn, 'a') as mf:
            mf.write(f"{idx},{mn:.6f},{av:.6f},{overlap_amt:.6f}\n")
        if overlap_amt < 0:
            with open(excess_fn, 'a') as mf:
                mf.write(f"{idx},{mn:.6f},{av:.6f},{overlap_amt:.6f}\n")
        print(f"Sample {idx+1}/{M}: min={mn:.6f}, avg={av:.6f}, overlap={overlap_amt:.6f}")

    # Save full dataset
    dirn = os.path.dirname(data_fn)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    torch.save(torch.from_numpy(data), data_fn)
    print(f"Saved full dataset to {data_fn}")

    # Apply symmetries and save enriched dataset
    try:
        sym_data = apply_symmetries_to_data(data, L)
        # Print symmetrized dataset shape
        print(f"Symmetrized dataset shape: {sym_data.shape}")
        # Print first few samples
        print("First few symmetrized samples:")
        for i in range(min(5, sym_data.shape[0])):
            print(f"Sample {i+1}: {sym_data[i].shape} - {sym_data[i].tolist()}")
        # Save symmetrized dataset
        sym_fn = cfg.get(sec, "output_filename_sym", fallback=data_fn.replace('.pt', '_sym.pt'))
        dirn = os.path.dirname(sym_fn)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        torch.save(torch.from_numpy(sym_data), sym_fn)
        print(f"Saved symmetrized dataset with {sym_data.shape[0]} samples to {sym_fn}")
    except ValueError as e:
        print(f"Skipping symmetry enrichment: {e}")

    # Plot metrics
    #plt.figure(figsize=(8,4))
    #plt.plot(min_dists, label="Min distance")
    #plt.plot(avg_dists, label="Avg distance")
    #plt.xlabel("Sample index")
    #plt.ylabel("Distance")
    #plt.title("Pairwise Distance Statistics per Sample")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    # Save top 25% best samples
    k = max(1, int(np.ceil(0.25 * M)))
    best_idx = np.argsort(min_dists)[-k:]
    top_data = data[best_idx]
    dirn = os.path.dirname(top_fn)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    torch.save(torch.from_numpy(top_data), top_fn)
    print(f"Saved top {k} samples (best min distance) to {top_fn}")

    # Apply symmetries to top samples and save
    try:
        sym_top_data = apply_symmetries_to_data(top_data, L)
        # Save symmetrized top dataset
        sym_top_fn = cfg.get(sec, "output_filename_sym_top", fallback=top_fn.replace('.pt', '_sym.pt'))
        dirn = os.path.dirname(sym_top_fn)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        torch.save(torch.from_numpy(sym_top_data), sym_top_fn)
        print(f"Saved symmetrized top dataset with {sym_top_data.shape[0]} samples to {sym_top_fn}") 

    except ValueError as e:
        print(f"Skipping symmetry enrichment for top samples: {e}")  

# -----------------------------------------------------------------------------
# Dataset & DataLoader
# -----------------------------------------------------------------------------

class SpherePackingDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path)
        print(f"Loaded dataset {path}, shape {self.data.shape}")
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx: int):
        return self.data[idx]


def get_data_loader(batch_size: int, dataset_path: str):
    ds = SpherePackingDataset(dataset_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def main():
    sec = "sample_generation_PESC"
    scenario_file = "SP/Results_SpheresInCube.txt"
    # load columns: N, L, ...
    scenarios = np.loadtxt(
        scenario_file,
        skiprows=50,         # skip header lines
        usecols=(0, 1),     # first two columns: N and L
        dtype=float
    )
    for N_i, L_i in scenarios:
        N_i = int(N_i)
        L_i = float(L_i)
        r_i = 1.0 / L_i
        d_i = 2.0 / L_i
        # override config parameters
        cfg.set(sec, "num_spheres", str(N_i))
        cfg.set(sec, "bounding_box_width", "1.0")
        cfg.set(sec, "sphere_radius", str(r_i))
        cfg.set(sec, "best_known_diameter", str(d_i))
        cfg.set(sec, "num_samples", "5")
        # update output filenames per scenario
        cfg.set(sec, "output_filename", f"dataset_N{N_i}.pt")
        cfg.set(sec, "output_filename_top", f"dataset_top_N{N_i}.pt")
        #cfg.set(sec, "output_filename_metrics", f"metrics_N{N_i}.csv")
        #cfg.set(sec, "output_filename_metrics_excess", f"metrics_excess_N{N_i}.csv")
        print(f"\n=== Generating for N={N_i}, r={r_i:.4f} ===")
        generate_dataset_push()

if __name__ == "__main__":
    generate_dataset_push()
