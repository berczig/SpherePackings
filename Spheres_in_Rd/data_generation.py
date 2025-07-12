import numpy as np; #NumPy package for arrays, random number generation, etc
from spheres_in_Rd import cfg, reffolder
import matplotlib.pyplot as plt
from spheres_in_Rd.Max_Indep_Sets import MIS_basic, MIS_luby
import torch
from torch.utils.data import DataLoader, Dataset
import os


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

def generate_dataset(num_samples, min_size, filename):
    print("generate {} points..".format(num_samples))
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
        print("points:",   len(points.T))

        # Apply MIS Luby's algorithm to thin points
        thinned_points = MIS_luby(points.T, min_distance=2*cfg.getfloat("ppp_sample_generation", "sphere_radius"))
        print("luby:", thinned_points.shape[1])
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
            print("Generated {}/{} samples with {} spheres each".format(num_big_samples, num_samples, min_size))

    print("Dataset shape:", dataset.shape)
    # Save dataset as a tensor
    dataset_tensor = torch.tensor(dataset, dtype=torch.float32)

    savepath = os.path.join(reffolder, filename)
    print("saving data set as {}".format(savepath))
    torch.save(dataset_tensor, savepath)

def generate_dataset_push():
    pass

# Dataset
class SpherePackingDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)  # Assuming .pt file with tensor of shape (num_samples, d, N)
        # print the shape and type of the dataset
        print("loading dataset {} of shape {}".format(path, self.data.shape))
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# DataLoader

def get_data_loader(batch_size, dataset_path):
    data = SpherePackingDataset(dataset_path)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader

# Splitting data_loader into training and test sets

def split_data_loader(data_loader, train_ratio):
    total_size = len(data_loader.dataset)   
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    train_data_loader, test_data_loader = torch.utils.data.random_split(data_loader, [train_size, test_size])
    print("data_loader:", type(data_loader), data_loader)
    print("train_data_loader:", type(train_data_loader), train_data_loader)
    return train_data_loader, test_data_loader

    