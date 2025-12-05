import os, platform
import sys

# If you need the quick unblock, leave this enabled on macOS.
# Disable by exporting SPHEREPACK_DISABLE_KMP_HACK=1 in your shell.
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# Loading utilities
# ============================================================

def load_raw(file):
    """Load a PyTorch file (.pt) or return tensor directly."""
    if isinstance(file, torch.Tensor):
        return file
    return torch.load(file)


def extract_tensor(obj):
    """
    Try to extract a tensor of shape (M, d, N) or (M, N, d) from various formats:
    - torch.Tensor
    - dict with common keys like 'pushed', 'data', 'samples', etc.
    - numpy array
    """
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)

    if isinstance(obj, dict):
        # Try some common keys
        for key in ["pushed", "data", "samples", "points", "packings"]:
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
        # Fallback: if there's exactly one tensor value, use it
        tensor_vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensor_vals) == 1:
            return tensor_vals[0]
        raise ValueError("Could not infer which tensor to use from dict keys.")

    raise TypeError(f"Unsupported object type for extract_tensor: {type(obj)}")


def load_dataset(file):
    raw = load_raw(file)
    return extract_tensor(raw)


def _to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def normalize_dataset_shape(data_3d):
    """
    Normalize input of shape (M, d, N) or (M, N, d) to (M, d, N).
    Returns (data_d_n, d, N).
    """
    arr = _to_numpy(data_3d)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor/array, got shape {arr.shape}")

    if arr.shape[1] <= arr.shape[2]:
        d, n = arr.shape[1], arr.shape[2]
        return arr, d, n
    d, n = arr.shape[2], arr.shape[1]
    return np.transpose(arr, (0, 2, 1)), d, n


# ============================================================
# Geometry: min pair distance & radii
# ============================================================

def _min_pairwise_distance_sample(points_nd: np.ndarray) -> float:
    """
    points_nd: shape (N, d) numpy array.
    Returns the minimum pairwise Euclidean distance.
    """
    N = points_nd.shape[0]
    if N < 2:
        return 0.0
    best = np.inf
    for i in range(N):
        xi = points_nd[i]
        for j in range(i + 1, N):
            dx = xi - points_nd[j]
            d = float(np.sqrt(np.dot(dx, dx)))
            if d < best:
                best = d
    return float(best if np.isfinite(best) else 0.0)


def _min_pairwise_distance_sample_periodic(points_nd: np.ndarray, L: float) -> float:
    """
    Minimum-image distance for periodic box of side L.
    """
    N, d = points_nd.shape
    if N < 2:
        return 0.0
    best = np.inf
    for i in range(N):
        xi = points_nd[i]
        for j in range(i + 1, N):
            delta = xi - points_nd[j]
            delta = (delta + 0.5 * L) % L - 0.5 * L
            d2 = float(np.dot(delta, delta))
            if d2 < best * best:
                best = math.sqrt(d2)
    return float(best if np.isfinite(best) else 0.0)


def compute_radii(tensor_data, L: float = 1.0, boundary_mode: str = "reflect"):
    """
    tensor_data: expected shapes (M,d,N) or (M,N,d) or torch.Tensor.
    Returns dict with per-sample:
      - 'radii':          effective radius = 0.5 * min pairwise distance
      - 'min_distances':  min pairwise distance itself
    """
    data_d_n, d, N = normalize_dataset_shape(tensor_data)
    M = data_d_n.shape[0]
    min_dists = np.empty(M, dtype=np.float64)
    radii = np.empty(M, dtype=np.float64)
    periodic = str(boundary_mode).lower() == "periodic"

    for idx in range(M):
        # data_d_n[idx]: (d, N) -> (N, d)
        pts = data_d_n[idx].T.astype(np.float64, copy=False)
        if periodic:
            md = _min_pairwise_distance_sample_periodic(pts, L)
        else:
            md = _min_pairwise_distance_sample(pts)
        min_dists[idx] = md
        radii[idx] = 0.5 * md

    return {
        "radii": radii,
        "min_distances": min_dists,
        "dimension": d,
        "num_spheres": N,
    }


# ============================================================
# Plotting utilities
# ============================================================

def plot(Arrays, labels, savepath,
         n_bins=100,
         title="Sphere radius distribution",
         xlabel="Radius",
         ylabel="Frequency",
         plotmode="overlay",
         n_xticks=15,
         show=True,
         filename="radii_hist.png"):
    """
    Arrays: list of 1D numpy arrays (radii per dataset)
    labels: list of labels (same length as Arrays)
    savepath: directory where to save the figure (if not None)
    plotmode: 'overlay' (default) or 'stacked' (hist)
    """
    if len(Arrays) != len(labels):
        raise ValueError("Arrays and labels must have the same length.")

    # Range across all arrays
    max_val = -np.inf
    min_val = np.inf
    for values in Arrays:
        if len(values) == 0:
            continue
        max_val = max(max_val, float(np.max(values)))
        min_val = min(min_val, float(np.min(values)))

    if not np.isfinite(max_val) or not np.isfinite(min_val):
        raise ValueError("No finite values to plot.")

    # Pad range slightly
    pad = 0.05 * (max_val - min_val if max_val > min_val else max_val or 1.0)
    max_val = max_val + pad
    min_val = max(0.0, min_val - pad)

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_widths = np.diff(bin_edges)
    bar_x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10, 6))

    if plotmode == "overlay":
        max_freq = 0.0
        for index, values in enumerate(Arrays):
            if len(values) == 0:
                continue
            hist_counts, _ = np.histogram(values, bins=bin_edges)
            # normalize to probability (frequency)
            hist_data = hist_counts / max(1, len(values))
            color = mpl.colors.hsv_to_rgb((index / max(1, len(Arrays)), 1, 1))
            max_freq = max(max_freq, float(np.max(hist_data)))
            label = f"{labels[index]} [{len(values)} samples]"
            plt.bar(bar_x_positions,
                    hist_data,
                    width=bin_widths,
                    edgecolor='black',
                    alpha=0.4,
                    label=label,
                    color=color)
        plt.ylim(0, max_freq if max_freq > 0 else 1.0)
    else:
        datas = []
        weights = []
        labels_samples = []
        for index, values in enumerate(Arrays):
            datas.append(values)
            if len(values) == 0:
                w = np.array([1.0])
            else:
                w = np.full(len(values), 1.0 / len(values))
            weights.append(w)
            labels_samples.append(f"{labels[index]} [{len(values)} samples]")
        plt.hist(datas, bin_edges, weights=weights, label=labels_samples)

    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlim(min_val, max_val)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # optional: nicer x-ticks
    try:
        xticks = np.linspace(min_val, max_val, n_xticks)
        plt.xticks(xticks)
    except Exception:
        pass

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        out_file = os.path.join(savepath, filename)
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved figure to {out_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_files_combined(files, labels, savepath, boundary_mode="reflect", box_len=1.0, **kwargs):
    """
    files: list of .pt paths (or tensors)
    labels: list of labels (same length)
    savepath: directory to save plot
    kwargs: forwarded to plot(...)
    """
    arrays_radii = []
    for file in files:
        dataset = load_dataset(file)
        metr = compute_radii(dataset, L=box_len, boundary_mode=boundary_mode)
        arrays_radii.append(metr["radii"])

    plot(Arrays=arrays_radii,
         labels=labels,
         savepath=savepath,
         title="Sphere radius distribution",
         xlabel="Effective radius (min distance / 2)",
         ylabel="Frequency",
         **kwargs)


# ============================================================
# 3D visualization (optional)
# ============================================================

def plot_3d(dataset, title="plot"):
    """
    Quick 3D scatter for a few configurations.
    dataset: tensor or path to .pt with shape (M, d, N) or (M, N, d)
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    arr_d_n, d, _ = normalize_dataset_shape(dataset)
    if d != 3:
        raise ValueError(f"plot_3d only supports d=3; got d={d}")
    data_3n = arr_d_n

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    for idx, data in enumerate(data_3n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xdata, ydata, zdata = data
        ax.scatter3D(xdata, ydata, zdata, c=zdata)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{title} (sample {idx})")
        plt.show()


# ============================================================
# Main: interactive multi-file radius plots
# ============================================================

def main():
    # Usage 1: pass files as command-line arguments
    #   python plot_radii.py file1.pt file2.pt ...
    # Usage 2: run with no args, enter paths interactively

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = ["diffuse_boost/output/spheres_in_cube_new/training_sets/srp_data_N191_2025-11-27_00-00-19.pt",
                 "diffuse_boost/output/spheres_in_cube_new/generated_sets/spheres_gen_500x191_20251128_000000.pt",
                 "diffuse_boost/output/spheres_in_cube_new/final_pushed/spheres_srp_pushed_N191_2025-11-28_002657.pt"]

    if not files:
        print("No files provided. Exiting.")
        sys.exit(0)

    labels = []
    for f in files:
        default_label = os.path.splitext(os.path.basename(str(f)))[0]
        lab = input(f"Label for {f} (default: {default_label}): ").strip()
        if not lab:
            lab = default_label
        labels.append(lab)

    out_dir_default = "diffuse_boost/output/spheres_in_cube_new/distribution_plots"
    out_dir = input(f"Output directory for plots (default: {out_dir_default}): ").strip()
    if not out_dir:
        out_dir = out_dir_default
    else:
        out_dir = os.path.join(out_dir_default, out_dir)

    plot_files_combined(
        files,
        labels,
        savepath=out_dir,
        plotmode="overlay",   # or "stacked"
        n_bins=80,
        show=True,
        filename="sphere_radii_hist.png",
    )

    # Optional: print max radius per set
    print("\n=== Summary: max effective radius per dataset ===")
    for f, lab in zip(files, labels):
        td = load_dataset(f)
        metr = compute_radii(td)
        print(f"{lab}: max radius = {np.max(metr['radii']):.6f} (over {len(metr['radii'])} samples)")


if __name__ == "__main__":
    main()
    
