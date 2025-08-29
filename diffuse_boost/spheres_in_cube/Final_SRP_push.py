# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os, platform
# If you need the quick unblock, leave this enabled on macOS.
# Disable by exporting SPHEREPACK_DISABLE_KMP_HACK=1 in your shell.
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    # Intel notes this is an unsafe workaround; prefer the environment fix below.
    # It must be set BEFORE any library initializes OpenMP.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from datetime import datetime

# --- Project-config + physics push ---
from diffuse_boost import cfg
from diffuse_boost.spheres_in_cube.physics_push_PESC import eliminate_overlaps_box
from diffuse_boost.spheres_in_cube.data_generation_PP_PBTS_jit import SRP, compute_EL, local_opt  


def run_srp_push_on_loaded_dataset():
    # Where to read hyperparameters
    sec  = "sample_generation_PP+PBTS"

    # --- Parameters from SRP section (same ones used by generate_dataset_push_srp) ---
    D        = cfg.getint(sec, "dimension")
    L        = cfg.getfloat(sec, "bounding_box_width")
    r        = cfg.getfloat(sec, "sphere_radius")
    best_d   = cfg.getfloat(sec, "best_known_diameter", fallback=2.0 * r)
    N        = cfg.getint(sec, "num_spheres")
    dt       = cfg.getfloat(sec, "dt")
    max_iter = cfg.getint(sec, "max_iter")
    tol      = cfg.getfloat(sec, "tol")
    mode     = cfg.get(sec, "boundary_mode")

    # SRP hyperparameters (as in generate_dataset_push_srp)
    Imax        = cfg.getint(sec, "srp_Imax", fallback=500)
    m           = cfg.getint(sec, "srp_m", fallback=20)
    sigma_frac  = cfg.getfloat(sec, "srp_sigma_frac", fallback=0.2)
    sigma       = sigma_frac * L
    beta        = cfg.getfloat(sec, "srp_beta", fallback=0.95)
    tol_opt     = cfg.getfloat(sec, "srp_tol", fallback=1e-8)
    maxiter_opt = cfg.getint(sec, "srp_maxiter", fallback=300)
    restarts    = cfg.getint(sec, "srp_restarts", fallback=10)

    # --- I/O paths ---
    dataset_load_path = cfg.get(sec, "filename_generated")  # SRP-generated .pt (shape: M x D x N)
    output_dir        = cfg.get(sec, "output_filename_path")
    os.makedirs(output_dir, exist_ok=True)
    s_now             = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_save_path  = os.path.join(output_dir, f"srp_pushed_{s_now}.pt")
    metrics_fn        = os.path.join(output_dir, f"srp_push_metrics_{s_now}.csv")

    print(f"Loading Flow-model-generated dataset from: {dataset_load_path}")
    if not os.path.isfile(dataset_load_path):
        raise FileNotFoundError(f"No generated dataset found at {dataset_load_path}")

    loaded = torch.load(dataset_load_path)
    if isinstance(loaded, torch.Tensor):
        initial_dataset = loaded.detach().cpu().numpy()  # (M, D, N)
    else:
        initial_dataset = np.asarray(loaded)

    if initial_dataset.ndim != 3:
        raise ValueError(f"Expected dataset of shape (M, D, N); got {initial_dataset.shape}")

    M_in, D_in, N_in = initial_dataset.shape
    if (D_in != D) or (N_in != N):
        print(f"[warning] Dataset shape (D={D_in}, N={N_in}) differs from config (D={D}, N={N}). Using dataset's D,N.")
        D, N = D_in, N_in

    if D != 3:
        raise NotImplementedError("This SRP energy/gradient is 3D-specific (D must be 3).")

    print(f"Processing {M_in} samples with SRP + local_opt + physics push "
          f"(D={D}, N={N}, L={L}, r={r}, restarts={restarts})")

    # Metrics CSV header
    with open(metrics_fn, 'w') as mf:
        mf.write("sample,srp_restart,EL_before,EL_after,pre_push_min,post_push_min,pre_excess,post_excess\n")

    out_data = np.zeros((M_in, D, N), dtype=np.float32)
    half = L / 2.0
    eps  = 1e-6

    for i in range(M_in):
        print(f"Sample {i+1}/{M_in}")

        # Input sample is (D, N); transpose to (N, D) absolute coords
        centers0 = initial_dataset[i].T  # (N, D), typically within [r, L-r]

        # Relative coordinates used by SRP/local_opt
        X0 = (centers0 - half).ravel()

        # Track best post-push configuration across SRP restarts
        diffs0    = centers0[:, None, :] - centers0[None, :, :]
        best_min  = np.min(np.linalg.norm(diffs0, axis=-1)[np.triu_indices(N, 1)])
        best_cent = centers0.copy()
        print(f"  initial min distance = {best_min:.6f}")

        for k in range(restarts):
            # SRP random perturb + normalized gradient steps
            X_srp = SRP(X0, L, N, r, Imax, m, sigma, beta) if k > 0 else X0
            X_srp = np.clip(X_srp, -L/2 + eps, L/2 - eps)
            EL_before = compute_EL(X_srp, L, N, r)

            # Local optimization with L-BFGS-B in relative frame
            X_lo, EL_after = local_opt(X_srp, L, N, r, tol_opt, maxiter_opt)

            # Map back to absolute cube and clip to [r, L-r]
            coords_rel  = X_lo.reshape((N, D))
            centers_opt = (coords_rel + half) * ((L - 2.0 * r) / L) + r
            centers_opt = np.minimum(np.maximum(centers_opt, r + eps), L - r - eps)

            # Pre-physics-push min pairwise distance
            diffs_pre = centers_opt[:, None, :] - centers_opt[None, :, :]
            pre_min   = np.min(np.linalg.norm(diffs_pre, axis=-1)[np.triu_indices(N, 1)])
            pre_excess    = best_d - pre_min

            if pre_min > best_min:
                best_min  = pre_min
                best_cent = centers_opt.copy()

            # Physics push (removes any residual overlaps respecting walls)
            #centers_k, _ = eliminate_overlaps_box(
            #    centers_opt, r, [L] * D,
            #    max_iter=max_iter, dt=dt, tol=tol,
            #    boundary_mode=mode, visualize=False)

            # Post-push stats
            #diffs_post = centers_k[:, None, :] - centers_k[None, :, :]
            #post_min   = np.min(np.linalg.norm(diffs_post, axis=-1)[np.triu_indices(N, 1)])
            #post_excess     = best_d - post_min

            # Log metrics
            post_min = pre_min
            post_excess = best_d - post_min
            with open(metrics_fn, 'a') as mf:
                mf.write(f"{i},{k+1},{EL_before:.6f},{EL_after:.6f},{pre_min:.6f},{post_min:.6f},{pre_excess:.6f},{post_excess:.6f}\n")
            
            if post_min > best_min:
                best_min  = post_min
                best_cent = centers_k.copy()

            print(f"    restart {k+1}/{restarts}: "
                  f"EL {EL_before:.3e} -> {EL_after:.3e}, "
                  f"pre_min {pre_min:.6f}, post_min {post_min:.6f}")
            

        out_data[i] = best_cent.T
        print(f"  best_min after restarts = {best_min:.6f}\n")

        # Save to output_save_path after each 10 samples 
        if (i + 1) % 10 == 0:
            torch.save(torch.from_numpy(out_data[:i+1]), output_save_path)
            print(f"  Saved intermediate dataset to: {output_save_path}")

    # Save pushed dataset + metrics
    torch.save(torch.from_numpy(out_data[:i+1]), output_save_path)
    print(f"\nSaved SRP-pushed dataset to: {output_save_path}")
    print(f"Metrics written to:          {metrics_fn}")


if __name__ == "__main__":
    run_srp_push_on_loaded_dataset()
