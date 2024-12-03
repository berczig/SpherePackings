import numpy as np

def thin_points(points, min_distance):
    thinned_points = []
    for p in points:
        closest = 9999999
        for q in thinned_points:
            closest = min(np.linalg.norm(p-q), closest)
        print("A:", closest, min_distance)
        if closest >= min_distance:
            thinned_points.append(p)
    return np.array(thinned_points).T