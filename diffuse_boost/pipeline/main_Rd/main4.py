import torch
from diffuse_boost.spheres_in_Rd.physics_push import eliminate_overlaps_batched, main
from diffuse_boost.spheres_in_Rd.data_evaluation import SphereDatasetEvaluator, plot_evaluations
from diffuse_boost.spheres_in_Rd.data_generation import get_data_loader
from diffuse_boost import cfg, reffolder
import ast
import os
import numpy as np
from datetime import datetime

"""
Runs the physics_push file on diffuser-generated samples
"""

if __name__ == "__main__":
    # Load Parameters
    n_points    = cfg.getint("physics_push", "n_points")
    dimension   = cfg.getint("physics_push", "dimension")
    radius      = cfg.getfloat("physics_push", "radius")
    box_size    = cfg.getfloat("physics_push", "box_size")
    max_iter    = cfg.getint("physics_push", "max_iter")
    evaluations = cfg.getint("physics_push", "evaluations")
    simulations = cfg.getint("physics_push", "simulations")
    dt          = cfg.getfloat("physics_push", "dt")
    tol         = cfg.getfloat("physics_push", "tol")

    best_min_dist = None
    best_min_dist_value = 0
    best_ratio = None
    best_ratio_value = 0

    s_now = datetime.now().strftime("%Y-%m-%d %H_%M_%S")

    # ——— Load initial packings from .pt file ———
    dataset_load_path = cfg["physics_push"].get("dataset_load_path", "")
    if dataset_load_path:
        initial_dataset = torch.load(dataset_load_path)  # Tensor of shape (M, k, d, N)
        # Extract the last timestep (k=-1) for all samples
        initial_dataset = initial_dataset[:, -1]  # Shape becomes (M, d, N)
        simulations = initial_dataset.shape[0]
        print(f"Loaded {simulations} packings from {dataset_load_path}")
    else:
        # Initialize an empty tensor for storing valid packings
        initial_dataset = torch.empty((0, dimension, n_points), dtype=torch.float32)
    # ——————————————————————————————————————————————

    lower_bound    = ast.literal_eval(cfg["lower_bounds"]["data"])[dimension]
    evaluation_skip= max_iter // evaluations
    batched_iterations = evaluation_skip * np.ones(evaluations)
    box            = box_size * np.ones(dimension)

    for sim in range(simulations):
        print(f"Simulation: {sim+1}/{simulations}")

        # Extract the initial centers for the current simulation
        if initial_dataset.shape[0] > 0:
            sample = initial_dataset[sim]  # Shape: (d, N)
            arr = sample.cpu().numpy()
            # Transpose to (N, d) for compatibility with eliminate_overlaps_batched
            initial_centers = arr.T
        else:
            # Random initialization as before
            initial_centers = np.random.rand(n_points, dimension) * box

        # Run the simulation
        final_centers, data_evaluations = eliminate_overlaps_batched(
            initial_centers,
            radius,
            box,
            batched_iterations=batched_iterations,
            dt=dt,
            tol=tol,
            visualize=False
        )

        # Plot evaluations
        plot_evaluations(
            data_evaluations, lower_bound, n_spheres=n_points, dimension=dimension,
            box_size=box_size, dt=dt, tol=tol, evaluation_skip=evaluation_skip, show=False,
            savepath=os.path.join(reffolder, f"output/fixed_gen_sets/{s_now}/eval{sim+1:05d}.png")
        )

        # Get best results
        for it in data_evaluations:
            eval = data_evaluations[it]

            if eval["box_ratio"] > best_ratio_value:
                best_ratio_value = eval["box_ratio"]
                best_ratio = data_evaluations

            if min(eval["min_distances"]) > best_min_dist_value:
                best_min_dist = data_evaluations
                best_min_dist_value = min(eval["min_distances"])

        # Save only valid packings
        smallest_dist = min(data_evaluations[max_iter]["min_distances"])
        print("biggest_dist: ", smallest_dist)
        if smallest_dist > 1.5:
            # Convert final_centers (N, d) → (d, N) and append to initial_dataset
            final_tensor = torch.tensor(final_centers.T, dtype=torch.float32).unsqueeze(0)  # Shape: (1, d, N)
            initial_dataset = torch.cat((initial_dataset, final_tensor), dim=0)  # Append along the first dimension

    # Save dataset as a tensor
    print("Dataset shape:", tuple(initial_dataset.shape))
    savepath = os.path.join(reffolder, f"output/fixed_gen_sets/{s_now}/dataset.pt")
    print(f"Saving dataset as {savepath}")
    torch.save(initial_dataset, savepath)

    # How to load the data
    # loader = get_data_loader(batch_size=2, dataset_path=savepath)

    # Plot best evaluations
    plot_evaluations(
        best_min_dist, lower_bound, n_spheres=n_points, dimension=dimension,
        box_size=box_size, dt=dt, tol=tol, evaluation_skip=evaluation_skip, show=False,
        savepath=os.path.join(reffolder, f"output/fixed_gen_sets/{s_now}/best_min_dist.png")
    )
    plot_evaluations(
        best_ratio, lower_bound, n_spheres=n_points, dimension=dimension,
        box_size=box_size, dt=dt, tol=tol, evaluation_skip=evaluation_skip, show=False,
        savepath=os.path.join(reffolder, f"output/fixed_gen_sets/{s_now}/best_ratio.png")
    )