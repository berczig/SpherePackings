import os
import numpy as np
import torch
from torch import nn
from SP.physics_push_PESC import eliminate_overlaps_box
from SP.data_evaluation import plot_evaluations
from SP.data_generation import get_data_loader
from SP import cfg
import ast
from datetime import datetime

"""
Runs the physics_push file on generated sphere-packing samples
using settings from [sample_generation_PESC] in config.cfg.
"""

if __name__ == "__main__":
    # Load Parameters from [sample_generation_PESC]
    sec = "sample_generation_PESC"
    n_points    = cfg.getint(sec, "num_spheres")
    dimension   = cfg.getint(sec, "dimension")
    box_size    = cfg.getfloat(sec, "bounding_box_width")
    radius      = cfg.getfloat(sec, "sphere_radius")
    max_iter    = cfg.getint(sec, "max_iter")
    evaluations = cfg.getint(sec, "evaluations")
    simulations = cfg.getint(sec, "num_samples")  # we'll override if loading
    dt          = cfg.getfloat(sec, "dt")
    tol         = cfg.getfloat(sec, "tol")

    # Paths from sample_generation_PESC section
    dataset_load_path = cfg.get(sec, "filename_generated")
    simulations = cfg.getint(sec, "num_samples")  # will override if loading
    # timestamped output subdir
    s_now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    print(f"Running physics_push on dataset {dataset_load_path} with {simulations} simulations.")
    output_dir = cfg.get(sec, "output_filename_path")
    output_save_path = os.path.join(output_dir, f"physics_push_{s_now}.pt")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to {output_save_path}")
    #os.makedirs(output_save_path, exist_ok=True)

    # Prepare simulation counter and load dataset
    if os.path.isfile(dataset_load_path):
        # Generated tensor shape: (M, d, N)
        initial_dataset = torch.load(dataset_load_path)
        simulations = initial_dataset.shape[0]
        print(f"Loaded {simulations} generated packings from {dataset_load_path}")
    else:
        # Empty dataset if none
        initial_dataset = torch.empty((0, dimension, n_points), dtype=torch.float32)
        print(f"No generated dataset found at {dataset_load_path}, will random-init.")

    # Evaluation setup
    lower_bound    = ast.literal_eval(cfg["lower_bounds"]["data"])[dimension]
    evaluation_skip= max_iter // evaluations
    batched_iterations = evaluation_skip * np.ones(evaluations, dtype=int)
    box = box_size * np.ones(dimension)

    # Track best
    best_min_dist = None
    best_min_dist_value = 0.0
    best_ratio = None
    best_ratio_value = 0.0

    s_now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dataset = []
    for sim in range(simulations):
        print(f"Simulation: {sim+1}/{simulations}")

        # Prepare initial centers
        if initial_dataset.shape[0] > 0:
            sample = initial_dataset[sim]       # (d, N)
            initial_centers = sample.cpu().numpy().T  # (N, d)
        else:
            initial_centers = np.random.rand(n_points, dimension) * box

        # Run physics_push with box constraints (now returns (centers, anim))
        final_centers, _ = eliminate_overlaps_box(
             initial_centers,
             radius,
             [box_size]*dimension,
             max_iter=max_iter,
             dt=dt,
             tol=tol,
             boundary_mode=cfg.get(sec, "boundary_mode"),
             visualize=False
        )  # returns (N, d), anim
        
        # Compute metrics (min & avg pairwise distances)
        coords = final_centers  # now a NumPy array of shape (N, d)
        diffs = coords[:, None, :] - coords[None, :, :]
        dmat = np.linalg.norm(diffs, axis=-1)
        i1, j1 = np.triu_indices(n_points, k=1)
        pdist = dmat[i1, j1]
        mn = float(pdist.min())
        av = float(pdist.mean())
        # Log to metrics file
        metrics_file = os.path.join(output_dir, f"metrics_physics_push_{s_now}.txt")
        if sim == 0:
            with open(metrics_file, 'w') as mf:
                mf.write("sim,min_dist,avg_dist,overlap_amount\n")
        overlap_amt = 2*radius - mn
        with open(metrics_file, 'a') as mf:
            mf.write(f"{sim+1},{mn:.6f},{av:.6f},{overlap_amt:.6f}\n")

        # Update best based on min_dist or box_ratio if still needed
        if mn > best_min_dist_value:
            best_min_dist_value = mn
            best_min_dist = (final_centers, mn, av)

        # Save valid packings: criterion on final min distance
        print(f"Final min_dist={mn:.4f}")
        output_dataset.append(final_centers.T)
        # new_tensor = torch.tensor(final_centers.T, dtype=torch.float32).unsqueeze(0)

    # Save updated dataset
    output_dataset = np.array(output_dataset)
    torch.save(output_dataset, output_save_path)
    print(f"Saved physics-pushed dataset to {output_save_path}")

    
    print("Physics push simulation completed.")