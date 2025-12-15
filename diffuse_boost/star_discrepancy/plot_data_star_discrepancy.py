import os, platform
import sys

# If you need the quick unblock, leave this enabled on macOS.
# Disable by exporting SPHEREPACK_DISABLE_KMP_HACK=1 in your shell.
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
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


# ============================================================
# Geometry: exact 2D star discrepancy (critical grid)
# ============================================================

def exact_star_discrepancy_2d(points_2d: np.ndarray) -> float:
    """
    points_2d: shape (N,2) numpy array in [0,1]^2.
    Exact 2D L_infinity star discrepancy on the critical grid.
    Checks both open [0,a)×[0,b) and closed [0,a]×[0,b].
    Returns D*.
    """
    pts = np.asarray(points_2d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2), got {pts.shape}")
    N = pts.shape[0]
    if N == 0:
        return 0.0

    x = np.clip(pts[:, 0], 0.0, 1.0)
    y = np.clip(pts[:, 1], 0.0, 1.0)

    Ax = np.unique(np.concatenate([x, [1.0]]))
    Ay = np.unique(np.concatenate([y, [1.0]]))
    U, V = Ax.size, Ay.size

    # --- OPEN (<,<) variant
    iu = np.searchsorted(Ax, x, side="left")
    iv = np.searchsorted(Ay, y, side="left")
    M_open = np.zeros((U, V), dtype=np.int64)
    for i, j in zip(iu, iv):
        M_open[i, j] += 1
    C_open = M_open.cumsum(axis=0).cumsum(axis=1)
    frac_open = C_open / float(N)

    Agrid, Bgrid = np.meshgrid(Ax, Ay, indexing="ij")
    D_minus = Agrid * Bgrid - frac_open
    D_minus_max = float(D_minus.max())

    # --- CLOSED (<=,<=) variant
    iu2 = np.searchsorted(Ax, x, side="right") - 1
    iv2 = np.searchsorted(Ay, y, side="right") - 1
    M_closed = np.zeros((U, V), dtype=np.int64)
    for i, j in zip(iu2, iv2):
        M_closed[i, j] += 1
    C_closed = M_closed.cumsum(axis=0).cumsum(axis=1)
    frac_closed = C_closed / float(N)

    D_plus = frac_closed - Agrid * Bgrid
    D_plus_max = float(D_plus.max())

    return max(D_minus_max, D_plus_max)


def compute_star_discrepancies(tensor_data):
    """
    tensor_data: expected shapes (M,2,N) or (M,N,2) or torch.Tensor.
    Returns dict with per-sample:
      - 'star_discrepancy': exact star discrepancy D* (smaller is better)
    """
    if isinstance(tensor_data, torch.Tensor):
        arr = tensor_data.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_data)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {arr.shape}")

    # Normalize to (M, 2, N)
    if arr.shape[1] == 2:
        data_2n = arr  # (M,2,N)
    elif arr.shape[2] == 2:
        data_2n = np.transpose(arr, (0, 2, 1))  # (M,2,N)
    else:
        raise ValueError(f"Second or last dimension must be 2; got shape {arr.shape}")

    M, d, N = data_2n.shape
    star = np.empty(M, dtype=np.float64)

    for idx in range(M):
        pts = data_2n[idx].T.astype(np.float64, copy=False)  # (N,2)
        star[idx] = exact_star_discrepancy_2d(pts)

    return {"star_discrepancy": star}


# ============================================================
# Plotting utilities
# ============================================================

def plot(Arrays, labels, savepath,
         n_bins=100,
         title="Star discrepancy distribution",
         xlabel="Exact star discrepancy (smaller is better)",
         ylabel="Frequency",
         plotmode="overlay",
         n_xticks=15,
         show=True,
         filename="star_discrepancy_hist.png"):
    """
    Arrays: list of 1D numpy arrays (metric per dataset)
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
    pad = 0.05 * (max_val - min_val if max_val > min_val else (max_val or 1.0))
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


def plot_files_combined(files, labels, savepath, **kwargs):
    """
    files: list of .pt paths (or tensors)
    labels: list of labels (same length)
    savepath: directory to save plot
    kwargs: forwarded to plot(...)
    """
    arrays_metric = []
    for file in files:
        dataset = load_dataset(file)
        metr = compute_star_discrepancies(dataset)
        arrays_metric.append(metr["star_discrepancy"])

    plot(Arrays=arrays_metric,
         labels=labels,
         savepath=savepath,
         title="Exact star discrepancy distribution",
         xlabel="Exact star discrepancy D* (smaller is better)",
         ylabel="Frequency",
         **kwargs)


# ============================================================
# 3D visualization (optional)
# (left unchanged; not used for star discrepancy in 2D)
# ============================================================

def plot_3d(dataset, title="plot"):
    """
    Quick 3D scatter for a few configurations.
    dataset: tensor or path to .pt with shape (M, d, N) or (M, N, d)
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    if isinstance(dataset, torch.Tensor):
        arr = dataset.detach().cpu().numpy()
    else:
        arr = np.asarray(dataset)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor for plotting, got shape {arr.shape}")

    # Normalize to (M, 3, N)
    if arr.shape[1] == 3:
        data_3n = arr
    elif arr.shape[2] == 3:
        data_3n = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError(f"Second or last dimension must be 3; got shape {arr.shape}")

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
# Main: interactive multi-file discrepancy plots
# ============================================================

def main():
    # Usage 1: pass files as command-line arguments
    #   python plot_star_discrepancy.py file1.pt file2.pt ...
    # Usage 2: run with no args, edit defaults below

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = [
            "diffuse_boost/output/star_discrepancy/final_pushed/star_srp_pushed_2025-12-14_125652.pt",
            "diffuse_boost/output/star_discrepancy/final_pushed/star_srp_pushed_2025-12-14_125908.pt",
            "diffuse_boost/output/star_discrepancy/final_pushed/star_srp_pushed_2025-12-14_130129.pt",
        ]

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

    out_dir_default = "diffuse_boost/output/star_discrepancy/distribution_plots"
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
        filename="star_discrepancy_hist.png",
    )

    # Optional: print best (min) discrepancy per set
    print("\n=== Summary: best (minimum) exact star discrepancy per dataset ===")
    for f, lab in zip(files, labels):
        td = load_dataset(f)
        metr = compute_star_discrepancies(td)
        vals = metr["star_discrepancy"]
        print(f"{lab}: min D* = {np.min(vals):.8f} | median D* = {np.median(vals):.8f} | max D* = {np.max(vals):.8f} (over {len(vals)} samples)")


if __name__ == "__main__":
    main()
