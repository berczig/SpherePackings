import os, platform

import diffuse_boost
# Quick unblock for macOS OpenMP duplication. Disable via:
#   export SPHEREPACK_DISABLE_KMP_HACK=1
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------------------------------------------------------
# Edit these paths
# -----------------------------------------------------------------------------
training_data = "diffuse_boost/output/circles_in_square/training_sets/circle_srp_generated_10000x32_2025-10-23_161628.pt"  # e.g. "diffuse_boost/output/circles_in_square/training_sets/circle_srp_generated_2000x32_2025-10-22_120000.pt"
gen_samples    = "diffuse_boost/output/circles_in_square/generated_sets/flow_circles_gen_20251103_120323.pt"  # e.g. "diffuse_boost/output/circles_in_square/generated_sets/flow_circles_gen_20251102_143210.pt"
pushed_samples = "diffuse_boost/output/circles_in_square/fixed_gen_sets/circle_final_push_mod_100x32_2025-11-04_120536.pt"  
save_dir      = "diffuse_boost/output/circles_in_square/fixed_gen_sets/distribution_plots"  # where to save the plot PNG

# -----------------------------------------------------------------------------
# Loaders / metrics
# -----------------------------------------------------------------------------
def load_dataset(file):
    if isinstance(file, torch.Tensor):
        return file
    obj = torch.load(file)
    if isinstance(obj, dict):
        # Try common keys; fallback to first tensor-like value
        for key in ("pushed", "samples", "data"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError("Dict did not contain a tensor.")
    return obj

def _to_Mx3xN(arr):
    a = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    if a.ndim == 3 and a.shape[1] == 3:
        return a
    if a.ndim == 3 and a.shape[2] == 3:
        return np.transpose(a, (0, 2, 1))
    if a.ndim == 2 and a.shape[0] == 3:
        return a[None, ...]
    if a.ndim == 2 and a.shape[1] == 3:
        return np.transpose(a, (1, 0))[None, ...]
    raise ValueError(f"Expected (M,3,N) or (M,N,3), got {a.shape}")

def compute_sum_radii(tensor_data):
    data = _to_Mx3xN(tensor_data)
    # sum over radii channel
    sum_r = data[:, 2, :].sum(axis=1)
    return {"sum_r": sum_r}

# -----------------------------------------------------------------------------
# Plot (overlay normalized histograms)
# -----------------------------------------------------------------------------
def plot(Arrays, labels, savepath, n_bins=100,
         title="Sum of Radii Distribution", xlabel="Sum of Radii",
         ylabel="Frequency", plotmode="overlay", n_xticks=15):
    max_val = -np.inf
    min_val = np.inf
    for values in Arrays:
        if len(values) == 0:
            continue
        max_val = max(max_val, np.max(values))
        min_val = min(min_val, np.min(values))

    if not np.isfinite(max_val) or not np.isfinite(min_val):
        raise ValueError("No finite values to plot.")

    # expand a bit for visual margins
    span = max_val - min_val
    if span <= 0:
        span = max(1.0, abs(max_val)) * 0.05
    max_val = max_val + 0.05 * span
    min_val = max(0.0, min_val - 0.05 * span)

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_widths = np.diff(bin_edges)
    bar_x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10, 6))

    if plotmode == "overlay":
        max_freq = 0.0
        for index, values in enumerate(Arrays):
            if len(values) == 0:
                continue
            hist_data = np.histogram(values, bins=bin_edges)[0] / max(1, len(values))
            color = mpl.colors.hsv_to_rgb((index / max(1, len(Arrays)), 0.85, 0.95))
            max_freq = max(max_freq, np.max(hist_data))
            label = f"{labels[index]}[{len(values)}]"
            plt.bar(bar_x_positions, hist_data, width=bin_widths,
                    edgecolor='black', alpha=0.45, label=label, color=color)
        plt.ylim(0, max_freq if max_freq > 0 else 1.0)
    else:
        datas = []
        weights = []
        labels_samples = []
        for index, values in enumerate(Arrays):
            datas.append(values)
            w = np.full(len(values), 1 / max(1, len(values))) if len(values) > 0 else np.array([1.0])
            weights.append(w)
            labels_samples.append(f"{labels[index]}[{len(values)}]")
        plt.hist(datas, bin_edges, weights=weights, label=labels_samples)

    # Ticks
    ticks = np.linspace(min_val, max_val, n_xticks)
    plt.xticks(ticks, rotation=45)

    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlim(min_val, max_val)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    os.makedirs(savepath, exist_ok=True)
    out_png = os.path.join(savepath, "sum_radii_hist.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_png}")

# -----------------------------------------------------------------------------
# Aggregate and plot the three sets
# -----------------------------------------------------------------------------
def plot_files_combined(files, labels, savepath, **kwargs):
    arrays_sum_r = []
    for file in files:
        dataset = load_dataset(file)
        metr = compute_sum_radii(dataset)
        arrays_sum_r.append(metr["sum_r"])
    plot(Arrays=arrays_sum_r, labels=labels, savepath=os.path.join(savepath, "sum_radii"),
         title="Normalized Sum of Radii Distribution",
         xlabel="Sum of Radii", ylabel="Frequency", **kwargs)

if __name__ == "__main__":
    if not training_data or not gen_samples or not pushed_samples:
        print("Please set 'training_data', 'gen_samples', and 'pushed_samples' paths at the top of this script.")
        raise SystemExit(1)

    plot_files_combined(
        [training_data, gen_samples, pushed_samples],
        ["Training data", "Samples (Flow matching)", "Final pushed samples"],
        save_dir,
        n_bins=120, plotmode="overlay", n_xticks=12
    )

    # Optional: print best/mean sums
    for label, path in zip(["Training", "Generated", "Final pushed"],
                           [training_data, gen_samples, pushed_samples]):
        arr = _to_Mx3xN(load_dataset(path))
        sums = arr[:, 2, :].sum(axis=1)
        print(f"{label}: mean={sums.mean():.6f} std={sums.std():.6f} min={sums.min():.6f} max={sums.max():.6f}")

