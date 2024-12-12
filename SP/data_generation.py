import numpy as np; #NumPy package for arrays, random number generation, etc
from SP import cfg
import matplotlib.pyplot as plt
from SP.Max_Indep_Sets import MIS_basic, MIS_luby
import torch

def sample_poisson_point_process(dimension, bounding_box_width, intensity):
    areaTotal = bounding_box_width**dimension
    half = bounding_box_width/2

    #Simulate a Poisson point process
    numbPoints = np.random.poisson(intensity*areaTotal);#Poisson number of points
    ppp = np.random.uniform(-half, half, (dimension, numbPoints))
    return ppp

def plot_and_sample_test():
    points = sample_poisson_point_process(dimension = cfg.getint("ppp_sample_generation", "dimension"), 
        bounding_box_width = cfg.getfloat("ppp_sample_generation", "bounding_box_width"), 
        intensity = cfg.getfloat("ppp_sample_generation", "intensity"))
    plt.scatter(points[0], points[1], c="black")
    thinned_points = MIS_luby(points.T, min_distance = cfg.getfloat("ppp_sample_generation", "sphere_radius"))
    plt.scatter(thinned_points[0], thinned_points[1], c="red")
    # print the number of red points as a caption of the plot
    plt.title(str(len(thinned_points[0])) + " points")
    plt.show()

def generate_dataset(num_samples, min_size):
    num_big_samples = 0
    dimension = cfg.getint("ppp_sample_generation", "dimension")
    dataset = np.empty((0, dimension, min_size))

    while num_big_samples < num_samples:
        # Sample points from a Poisson point process
        points = sample_poisson_point_process(
            dimension=dimension,
            bounding_box_width=cfg.getfloat("ppp_sample_generation", "bounding_box_width"),
            intensity=cfg.getfloat("ppp_sample_generation", "intensity")
        )

        # Apply MIS Luby's algorithm to thin points
        thinned_points = MIS_luby(points.T, min_distance=cfg.getfloat("ppp_sample_generation", "sphere_radius"))
        
        if thinned_points.shape[1] < min_size:
            continue
        
        # Drop excess points to match `min_size`
        excess = thinned_points.shape[1] - min_size
        if excess > 0:
            print("Dropping", excess, "excess points")  
            indices = np.random.choice(thinned_points.shape[1], excess, replace=False)
            thinned_points = np.delete(thinned_points, indices, axis=1)
        
        # Ensure the shape matches expected dimensions
        if thinned_points.shape[1] == min_size:
            dataset = np.concatenate((dataset, thinned_points[np.newaxis, :, :]), axis=0)
            num_big_samples += 1

    print("Dataset shape:", dataset.shape)
    # Save dataset as a tensor
    dataset_tensor = torch.tensor(dataset, dtype=torch.float32)
    torch.save(dataset_tensor, "./SP/sample_packings.pt")

    