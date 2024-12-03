import numpy as np; #NumPy package for arrays, random number generation, etc
from SP import cfg
import matplotlib.pyplot as plt

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
    print(points)
    plt.scatter(points[0], points[1])
    plt.show()