import math
from scipy.special import betainc
from collections import defaultdict, Counter
import numpy as np
import os
from pathlib import Path
import spheres_in_Rd as SP
import matplotlib.pyplot as plt

# -------------------
# Union-Find Structure
# -------------------
class UnionFind:
    """
    Helper class to compute the intersection graph
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.intersections = 0
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)
        self.intersections+=1
    
    def groups(self):
        comps = defaultdict(list)
        for i in range(len(self.parent)):
            comps[self.find(i)].append(i)
        return list(comps.values())



class SphereDatasetEvaluator:
    """
    Calculates the volume of the union of spheres and computes the number of connected components in the intersection graph. 
    """
    def __init__(self, centers, sphere_radius, box_size, monte_carlo_sim_steps=100_000):
        self.centers = np.array(centers)
        self.sphere_radius = sphere_radius
        self.intsect_dst_sq = (2*sphere_radius)**2
        self.radius_sq = sphere_radius**2
        self.n_spheres = len(centers)
        self.box_size = box_size
        self.monte_carlo_sim_steps = monte_carlo_sim_steps
        self.min_distances = [np.inf for _ in range(self.n_spheres)]
        if len(centers) > 0:
            self.dimension = len(centers[0])

        self.sphere_volume = self.get_sphere_volume(self.dimension, self.sphere_radius)

    @staticmethod
    def get_num_sphere_to_cover(ratio, bounding_box_width, dimension, radius):
        return math.floor((ratio*bounding_box_width**dimension)/SphereDatasetEvaluator.get_sphere_volume(dimension, radius))

    @staticmethod
    def get_sphere_volume(dimension, radius) -> float:
        return math.pi**(dimension/2)/math.gamma(dimension/2 + 1) * (radius**dimension)
    
    def intersection_volume_two_spheres(self, center1,  center2, eps = 10**-6):
        """intersection volume of two d-spheres using spherical caps"""
        dist = np.sum(SP.physics_push.shortest_vector_torus(center1, center2, self.box_size)**2)**0.5

        if dist >= 2*self.sphere_radius:
            return 0  # No overlap

        elif dist <= eps:
            # The centers coincide
            return self.sphere_volume

        # Volume of Spherical cap
        height = (2*self.sphere_radius-dist)/2
        x = (2*self.sphere_radius*height-height**2)/self.sphere_radius**2
        spherical_cap_vol = self.sphere_volume*betainc((self.dimension+1)/2, 1/2, x)/2
        return 2 * spherical_cap_vol
    
    def get_spheres_dist_sq(self, center1, center2):
        vec_ij = SP.physics_push.shortest_vector_torus(center1, center2, self.box_size)
        return np.sum(vec_ij**2)
    
    def get_bouding_box(self, centers):
        lower_corner = np.min(centers - self.sphere_radius, axis=0)
        upper_corner = np.max(centers + self.sphere_radius, axis=0)
        return lower_corner, upper_corner
    
    def monte_carlo_union_volume(self, centers, num_samples=10_000):
        """Monte Carlo estimation of the union volume of multiple d-spheres"""     
        # Get bounding box
        #lower_corner, upper_corner = self.get_bouding_box(centers)
        #volume_box = np.prod(upper_corner - lower_corner)
        lower_corner = -np.ones(self.dimension)*self.box_size/2
        upper_corner = np.ones(self.dimension)*self.box_size/2
        volume_box = self.box_size**self.dimension
        # Sample random points
        samples = np.random.uniform(low=lower_corner, high=upper_corner, size=(num_samples, self.dimension))
        
        # Check if inside any sphere
        inside = np.zeros(num_samples, dtype=bool)
        for c in centers:
            dist_sq = np.sum(SP.physics_push.shortest_vector_torus(samples, c, self.box_size)**2, axis=1)
            inside |= dist_sq <= self.radius_sq
        
        fraction_inside = np.mean(inside)
        return volume_box * fraction_inside
    
    def compute_connected_components(self) -> list:
        """
        The spheres define an intersection graph, having the spheres as vertices and edges between the sphere if they intersect
        """
        uf = UnionFind(self.n_spheres)
        
        # Build intersection graph
        for i in range(self.n_spheres):
            for j in range(i + 1, self.n_spheres):
                dist_sq =  self.get_spheres_dist_sq(self.centers[i], self.centers[j])
                self.min_distances[i] = min(self.min_distances[i], dist_sq**0.5)
                self.min_distances[j] = min(self.min_distances[j], dist_sq**0.5)
                if dist_sq < self.intsect_dst_sq:
                    uf.union(i, j)

        return uf.groups(), uf.intersections

    def evaluate(self):
        print("Evaluating Spheres...")
        connected_components, n_intersections = self.compute_connected_components()
        total_volume = 0
        cc_sizes = Counter()

        for cc in connected_components:
            cc_sizes[len(cc)] += 1
            if len(cc) == 1:
                # Single sphere
                total_volume += self.sphere_volume
            elif len(cc) == 2:
                # Two Spheres
                i, j = cc
                total_volume += 2*self.sphere_volume - self.intersection_volume_two_spheres(self.centers[i], self.centers[j])
            else:
                # Estimate the volume using monte carlo for >= 3 spheres
                total_volume += self.monte_carlo_union_volume(self.centers[cc], num_samples=self.monte_carlo_sim_steps)

        #lower_corner, upper_corner = self.get_bouding_box(self.centers)
        box_volume = self.box_size**self.dimension

        return {"sphere_vol":total_volume,
                "box_vol":box_volume, 
                "box_ratio":total_volume/box_volume,
                "n_intersections":n_intersections, 
                "min_distances":self.min_distances,
                "sphere_intersect_vol":self.n_spheres*self.sphere_volume-total_volume,
                "intersect_ratio":(self.n_spheres*self.sphere_volume-total_volume)/((self.n_spheres-1)*self.sphere_volume),
                "connected_components":cc_sizes}
    
def plot_evaluations(data_evaluations, lowerbound, savepath, n_spheres, dimension, box_size, dt, tol, evaluation_skip, show=True):
    # Using Numpy to create an array X
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    Y5 = []
    for it in data_evaluations:
        X.append(it)
        eval = data_evaluations[it]
        Y1.append(eval["box_ratio"])
        Y2.append(min(eval["min_distances"]))
        Y3.append(eval["n_intersections"])
        Y4.append(eval["intersect_ratio"])
        Y5.append((np.linspace(1, n_spheres, n_spheres), sorted(eval["min_distances"])))


    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(2, 2)
    figure.set_size_inches(12,10)
    title = "spheres={}, dim={}, box={}, dt={}, tol={}, eval_skip={}".format(n_spheres, dimension, box_size, dt, tol, evaluation_skip)
    figure.suptitle(title)

    # For Sine Function
    axis[0, 0].plot(X, Y1)
    axis[0, 0].set_title("density - [{:.4f}]".format(Y1[-1]))
    axis[0, 0].axhline(y=lowerbound, color='r', linestyle='-')

    # For Cosine Function
    axis[0, 1].plot(X, Y2)
    axis[0, 1].set_title("min. dist among spheres - [{:.4f}]".format(Y2[-1]))
    axis[0, 1].axhline(y=2, color='r', linestyle='-')

    # For Tangent Function
    axis[1, 0].plot(X, Y3)
    axis[1, 0].set_title("#intersections - [{}]".format(Y3[-1]))

    axis[1, 1].plot(X, Y4)
    axis[1, 1].set_title("overlap quotient - [{:.4f}]".format(Y4[-1]))

    
    savefolder = os.path.dirname(savepath)
    Path(savefolder).mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath)
    SP.animation.animate_histogram(Y5, savepath[:-4]+".gif", title=title, fps=max(1,len(data_evaluations)//2))

    if show:
        plt.show()
    else:
        plt.close()

    

if __name__ == "__main__":
    #centers = np.array([(0,0,0), (0.01, 0.01,0), (0.003, -0.01,0), (4,5,3)])
    #S = SphereDatasetEvaluator(centers, 1, 12)
    centers = np.array([(2.9,2.9,2.9), (-2.9, -2.9, -2.9)])
    S = SphereDatasetEvaluator(centers, 1, 8, 10**6)
    print("number spheres: ", S.get_num_sphere_to_cover(0.46526, 4, 5, 1))
    print(S.evaluate())