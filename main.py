from SP.data_generation import plot_and_sample_test, generate_dataset
import numpy as np
from SP import cfg
import torch
import matplotlib.pyplot as plt
from SP.DiffusionModel import sample_diffusion_model
from SP.DiffusionModel import train_diffusion_model
from SP.data_generation import data_loader, split_data_loader
from torch.utils.data import DataLoader


    
if __name__ == "__main__":
    plot_and_sample_test()
    generate_dataset(100,55)
    # Load and pribt the dataset
    data_loader = data_loader(batch_size=cfg.getint('batch_size'), dataset_path=cfg.get('dataset_path'))
    train_data_loader, test_data_loader= split_data_loader(data_loader, train_ratio=0.8)
    # Train the diffusion model with the dataset
    
    model = train_diffusion_model(train_data_loader, num_epochs=10, learning_rate=1e-4, num_train_timesteps=100)

    # Load the trained model and sample from it


    sample_diffusion_model(model, test_data_loader, num_samples=10, num_timesteps=100)

    # Plot the samples
    for i in range(model.shape[0]):
        plt.scatter(model[i, 0, :], model[i, 1, :], c='black')
        plt.show()
