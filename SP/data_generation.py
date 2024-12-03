import numpy as np; #NumPy package for arrays, random number generation, etc

def sample_poisson_point_process(dimension, bounding_radius, intensity):
    #Simulation window parameters
    xMin=0;xMax=1;
    yMin=0;yMax=1;
    xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
    areaTotal=xDelta*yDelta;

    #Point process parameters
    lambda0=100; #intensity (ie mean density) of the Poisson process

    #Simulate a Poisson point process
    numbPoints = np.random.poisson(lambda0*areaTotal);#Poisson number of points
    xx = xDelta*np.random.uniform(0,1,numbPoints)+xMin;#x coordinates of Poisson points
    yy = yDelta*np.random.uniform(0,1,numbPoints)+yMin;#y coordinates of Poisson points

def plot_and_sample_test():
    sample_poisson_point_process
