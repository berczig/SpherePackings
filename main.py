from SP.data_generation import plot_and_sample_test, generate_dataset
import numpy as np
from SP import cfg
import torch
    
if __name__ == "__main__":
    plot_and_sample_test()
    generate_dataset(100,55)
    # Load and pribt the dataset
    dataset = torch.load('./SP/sample_packings.pt')
    print(dataset)