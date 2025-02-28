from SP.data_generation import plot_and_sample_test, generate_dataset
import numpy as np
from SP import cfg
import torch
import matplotlib.pyplot as plt
from SP.DiffusionModel import sample_diffusion_model
from SP.DiffusionModel import train_diffusion_model
from SP.data_generation import get_data_loader, split_data_loader
from torch.utils.data import DataLoader
from torchsummary import summary
from SP.animation import animate_scatter


    
if __name__ == "__main__":
    # Load Parameters
    batch_size = cfg.getint("diffusion_model", "batch_size")
    dimension = cfg.getint("ppp_sample_generation", "dimension")
    radius = cfg.getfloat("ppp_sample_generation", "sphere_radius")
    num_spheres = 45

    # Generate Point Cloud Data and MIS
    #plot_and_sample_test()
    #generate_dataset(10,num_spheres, filename=cfg.get("ppp_sample_generation", 'train_dataset_path'))
    #generate_dataset(1,num_spheres, filename=cfg.get("ppp_sample_generation", 'test_dataset_path'))


    # Load and print the dataset
    train_data_loader = get_data_loader(
        batch_size=cfg.getint("diffusion_model", 'batch_size'), 
        dataset_path=cfg.get("ppp_sample_generation", 'train_dataset_path'))
    
    test_data_loader = get_data_loader(
        batch_size=cfg.getint("diffusion_model", 'batch_size'), 
        dataset_path=cfg.get("ppp_sample_generation", 'test_dataset_path'))
    #train_data_loader, test_data_loader= split_data_loader(data_loader, train_ratio=0.8)
    # Train the diffusion model with the dataset
    
    model = train_diffusion_model(train_data_loader, 
        num_epochs=cfg.getint("diffusion_model", "num_epochs"), 
        learning_rate=cfg.getfloat("diffusion_model", "learning_rate"), 
        num_train_timesteps=cfg.getint("diffusion_model", "num_train_timesteps"), 
        dimension = dimension, 
        batch_size = batch_size, 
        sphere_radius=radius)
    summary(model, (dimension, num_spheres))
    # Load the trained model and sample from it


    originals = []
    sampleds = []
    OG = next(iter(test_data_loader))[0]
    print("OG:", OG)
    for t in range(60):
        original, sampled = sample_diffusion_model(model, test_data_loader, num_train_timesteps=t)
        #originals.append(np.array(original[0]).T)
        originals.append(np.array(OG).T)
        sampleds.append(np.array(sampled[0]).T)
    animate_scatter(np.array(originals), np.array(sampleds), interval=60)
    # Plot the samples
    #print("original points shape: ", original.shape)
    #for i in range(1):
        #plt.scatter(original[i, 0, :], original[i, 1, :], c='black')
        #plt.scatter(sampled[i, 0, :], sampled[i, 1, :], c='red')
        #plt.show()
