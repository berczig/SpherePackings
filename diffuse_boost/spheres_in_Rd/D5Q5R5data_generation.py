import numpy as np
import itertools
from scipy.linalg import expm

def generate_D5(n):
    m = int(np.floor(n))
    centers = [coords for coords in itertools.product(range(-m, m+1), repeat=5) if sum(coords) % 2 == 0]
    return np.array(centers, dtype=float)

def generate_Q5(n):
    basis = np.array([[ 1, -1,  0,  0,  0],
                      [ 0,  1, -1,  0,  0],
                      [ 0,  0,  1, -1,  0],
                      [ 0,  0,  0,  1, -1],
                      [0.8,0.8,0.8,0.8,0.8]]).T
    translations = [np.zeros(5), np.array([0,0,0,-1,-1], dtype=float)]
    norms = np.linalg.norm(basis, axis=0)
    max_c = int(np.ceil(n / norms.min())) + 1
    centers = []
    for coeffs in itertools.product(range(-max_c, max_c+1), repeat=5):
        vec = basis.dot(coeffs)
        for t in translations:
            pt = vec + t
            if np.all(np.abs(pt) <= n):
                centers.append(pt)
    return np.array(centers)

def generate_R5(n):
    basis = np.array([[  1,   -1,    0,    0,    0],
                      [  0,    1,   -1,    0,    0],
                      [  0,    0,    1,   -1,    0],
                      [-0.5, -0.5, -0.5, -0.5,    2],
                      [ 0.8,  0.8,  0.8,  0.8,  0.8]]).T
    translations = [
        np.zeros(5),
        np.array([0,0,0,1,-1], dtype=float),
        np.array([0,0,-1,-1,0], dtype=float),
        np.array([0,0,0,-1,-1], dtype=float),
    ]
    norms = np.linalg.norm(basis, axis=0)
    max_c = int(np.ceil(n / norms.min())) + 1
    centers = []
    for coeffs in itertools.product(range(-max_c, max_c+1), repeat=5):
        vec = basis.dot(coeffs)
        for t in translations:
            pt = vec + t
            if np.all(np.abs(pt) <= n):
                centers.append(pt)
    return np.array(centers)

def random_small_rotation(dim=5, eps=0.1):
    A = np.random.uniform(-eps, eps, size=(dim, dim))
    A = A - A.T
    return expm(A)

def crop_to_core(points, n, r):
    """
    Keep only points in the core window [-n+r, n-r]^5.
    """
    low, high = -n + r, n - r
    mask = np.all((points >= low) & (points <= high), axis=0)
    return points[:, mask]

def downsample_by_boundary(points, target_count, n, r):
    """
    Downsample to target_count by dropping points closest to the core boundary.
    """
    # compute distance to nearest core face for each point
    low, high = -n + r, n - r
    # for each point, compute min distance to any face
    dists = np.minimum.reduce([
        points - low,      # dist to low face in each dim
        high - points      # dist to high face in each dim
    ]).min(axis=0)
    # sort indices by distance ascending (closest to boundary first)
    idx = np.argsort(dists)
    # indices to keep: drop first (current_count - target_count) points
    keep = idx[(len(idx) - target_count):]
    return points[:, keep]

def generate_equal_count_samples(n=2.5, num_transforms=10, eps=0.1, t_max=0.1):
    """
    1) Generate small-perturbation samples for D5, Q5, R5
    2) Crop to core window [-n+r, n-r]^5
    3) Downsample to equal counts by dropping boundary points
    Returns:
      tensor: (3*num_transforms, 5, target_count)
    """
    gens = [generate_D5, generate_Q5, generate_R5]
    r = np.sqrt(2) / 2
    cropped_samples = []

    # generate and crop
    for gen in gens:
        base = gen(n).T  # shape (5, num_points)
        for _ in range(num_transforms):
            R = random_small_rotation(5, eps)
            t = np.random.uniform(-t_max, t_max, size=5)
            transformed = R @ base + t[:, None]
            cropped = crop_to_core(transformed, n, r)
            cropped_samples.append(cropped)

    # find minimal count across all samples
    counts = [s.shape[1] for s in cropped_samples]
    target = min(counts)

    # downsample all to target
    final_samples = [downsample_by_boundary(s, target, n, r) for s in cropped_samples]
    tensor = np.stack(final_samples, axis=0)
    return tensor

if __name__ == "__main__":
    tensor = generate_equal_count_samples(
        n=100,
        num_transforms=10,
        eps=0.01,
        t_max=0.01
    )
    print("Final tensor shape:", tensor.shape)
