from SP.physics_push import eliminate_overlaps_batched, main
from SP.data_evaluation import SphereDatasetEvaluator, plot_evaluations
from SP.data_generation import get_data_loader
from SP import cfg
import ast
import SP
import os
import numpy as np
from datetime import datetime

"""
Runs the physics_push file.
"""


if __name__ == "__main__":
    #main()
    # Load Parameters
    n_points = cfg.getint("physics_push", "n_points")
    dimension = cfg.getint("physics_push", "dimension")
    radius = cfg.getfloat("physics_push", "radius")
    box_size = cfg.getfloat("physics_push", "box_size")
    max_iter = cfg.getint("physics_push", "max_iter")
    evaluations = cfg.getint("physics_push", "evaluations")
    simulations = cfg.getint("physics_push", "simulations")
    dt = cfg.getfloat("physics_push", "dt")
    tol = cfg.getfloat("physics_push", "tol")

    lower_bound = ast.literal_eval(cfg["lower_bounds"]["data"])[dimension]
    evaluation_skip = max_iter//evaluations
    batched_iterations = evaluation_skip * np.ones(evaluations)
    box = box_size*np.ones(dimension)

    print("The current best lower bound ({:.3f}%) for dim={} can be improved using {} spheres in a box of size={}".format(100*lower_bound, dimension,
        SphereDatasetEvaluator.get_num_sphere_to_cover(lower_bound, box_size, dimension, radius)+1, box_size))

    best_min_dist = None
    best_min_dist_value = 0
    best_ratio = None
    best_ratio_value = 0

    s_now = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    datasets = []

    for sim in range(simulations):
        print("Simulation: {}/{}".format(sim+1, simulations))
        # Generate random starting points
        initial_centers = np.random.rand(n_points, dimension) * box

        # Run the simulation
        final_centers, data_evaluations = eliminate_overlaps_batched(
            initial_centers,
            radius,
            box,
            batched_iterations=batched_iterations,
            dt=dt,
            tol=tol,
            visualize=False # Set to False or it will just print a message
        )

        # plot
        plot_evaluations(data_evaluations, lower_bound, n_spheres=n_points, dimension=dimension, box_size=box_size, dt=dt, tol=tol, evaluation_skip=evaluation_skip, show=False, 
                        savepath=os.path.join(SP.reffolder, "output/push_simulation/{}/eval{:05d}.png".format(s_now, sim+1)))
        
        # get best
        for it in data_evaluations:
            eval = data_evaluations[it]

            if eval["box_ratio"] > best_ratio_value:
                best_ratio_value =  eval["box_ratio"]
                best_ratio = data_evaluations

            if min(eval["min_distances"]) > best_min_dist_value:
                best_min_dist = data_evaluations
                best_min_dist_value =  min(eval["min_distances"])

        smallest_dist = min(data_evaluations[max_iter]["min_distances"])
        print("biggest_dist: ", smallest_dist)
        if smallest_dist > 1.96:
            datasets.append(final_centers.T)

    # Save dataset as a tensor
    from torch import tensor, save, float32
    dataset = np.array(datasets)
    print("Dataset shape:", dataset.shape)
    dataset_tensor = tensor(dataset, dtype=float32)

    savepath = os.path.join(SP.reffolder, "output/push_simulation/{}/dataset.pt".format(s_now))
    print("saving data set as {}".format(savepath))
    save(dataset_tensor, savepath)

    # how to load the data
    # loader = get_data_loader(batch_size=2, dataset_path=savepath)

    plot_evaluations(best_min_dist, lower_bound, n_spheres=n_points, dimension=dimension, box_size=box_size, dt=dt, tol=tol, evaluation_skip=evaluation_skip, show=False, 
                        savepath=os.path.join(SP.reffolder, "output/push_simulation/{}/best_min_dist.png".format(s_now)))
    plot_evaluations(best_ratio, lower_bound, n_spheres=n_points, dimension=dimension, box_size=box_size, dt=dt, tol=tol, evaluation_skip=evaluation_skip, show=False, 
                        savepath=os.path.join(SP.reffolder, "output/push_simulation/{}/best_ratio.png".format(s_now)))
