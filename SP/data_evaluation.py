import math

def calc_sphere_packing_density(centers, dimension, bounding_box_width, sphere_radius):
    space_area = bounding_box_width**dimension
    n = len(centers)
    sphere_area = n * math.pi**(dimension/2)/math.gamma(dimension/2 + 1) * (sphere_radius**dimension)
    return sphere_area/space_area