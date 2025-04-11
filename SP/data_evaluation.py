import math
from scipy.special import betainc
from collections import defaultdict, Counter
import numpy as np


# -------------------
# Union-Find Structure
# -------------------
class UnionFind:
    """
    Helper class to compute the intersection graph
    """
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)
    
    def groups(self):
        comps = defaultdict(list)
        for i in range(len(self.parent)):
            comps[self.find(i)].append(i)
        return list(comps.values())



class SphereDatasetEvaluator:
    """
    Calculates the volume of the union of spheres and computes the number of connected components in the intersection graph. 
    """
    def __init__(self, centers, sphere_radius):
        self.centers = np.array(centers)
        self.sphere_radius = sphere_radius
        self.n_spheres = len(centers)
        if len(centers) > 0:
            self.dimension = len(centers[0])

        self.sphere_volume = self.get_sphere_volume(self.dimension, self.sphere_radius)

    def get_sphere_volume(self, dimension, radius) -> float:
        return math.pi**(dimension/2)/math.gamma(dimension/2 + 1) * (radius**dimension)
    
    def intersection_volume_two_spheres(self, center1,  center2, eps = 10**-6):
        """intersection volume of two d-spheres using spherical caps"""
        dist = np.linalg.norm(center1 - center2)

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
    
    def get_bouding_box(self, centers):
        lower_corner = np.min(centers - self.sphere_radius, axis=0)
        upper_corner = np.max(centers + self.sphere_radius, axis=0)
        return lower_corner, upper_corner
    
    def monte_carlo_union_volume(self, centers, num_samples=10_000):
        """Monte Carlo estimation of the union volume of multiple d-spheres"""     
        # Get bounding box
        lower_corner, upper_corner = self.get_bouding_box(centers)
        volume_box = np.prod(upper_corner - lower_corner)

        # Sample random points
        samples = np.random.uniform(low=lower_corner, high=upper_corner, size=(num_samples, self.dimension))
        
        # Check if inside any sphere
        inside = np.zeros(num_samples, dtype=bool)
        for c in centers:
            inside |= np.linalg.norm(samples - c, axis=1) <= self.sphere_radius
        
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
                dist = np.linalg.norm(self.centers[i] - self.centers[j])
                if dist < 2 *self.sphere_radius:
                    uf.union(i, j)

        return uf.groups()

    def evaluate(self):
        connected_components = self.compute_connected_components()
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
                total_volume += self.monte_carlo_union_volume(self.centers[cc])

        lower_corner, upper_corner = self.get_bouding_box(self.centers)
        box_volume = np.prod(upper_corner - lower_corner)

        return {"no_intersections":cc_sizes[1]==self.n_spheres, 
                "sphere_vol":total_volume,
                "sphere_intersect_vol":self.n_spheres*self.sphere_volume-total_volume,
                "box_vol":box_volume, 
                "box_ratio":total_volume/box_volume,
                "intersect_ratio":(self.n_spheres*self.sphere_volume-total_volume)/self.n_spheres*self.sphere_volume,
                "connected_components":cc_sizes,
                "bounding_box":[lower_corner, upper_corner]}
    

if __name__ == "__main__":
    centers = np.array([(0,0,0), (0.01, 0.01,0), (0.003, -0.01,0), (4,5,3)])
    S = SphereDatasetEvaluator(centers, 1)
    print(S.evaluate())