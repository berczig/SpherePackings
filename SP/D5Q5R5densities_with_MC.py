import numpy as np
import itertools
import math
import pandas as pd
from scipy.spatial import cKDTree

def generate_D5(n):
    """
    True D5 enumeration in [-n,n]^5: D5 = {x∈Z^5 : sum(x) even}. 
    """
    m = int(np.floor(n))
    centers = []
    for coords in itertools.product(range(-m, m+1), repeat=5):
        if sum(coords) % 2 == 0:
            centers.append(coords)
    return np.array(centers, dtype=float)

def generate_Q5(n):
    """
    Q5 (2-periodic) per Szöllősi’s Table 4.2 :contentReference[oaicite:1]{index=1}:
      Basis columns v0…v4 and translations x0, x1.
    """
    # basis vectors v0…v4
    basis = np.column_stack([
        [ 1, -1,  0,  0,  0],  # v0
        [ 0,  1, -1,  0,  0],  # v1
        [ 0,  0,  1, -1,  0],  # v2
        [ 0,  0,  0,  1, -1],  # v3
        [0.8,0.8,0.8,0.8,0.8],  # v4
    ]).T  # shape (5,5)

    # two coset translations
    translations = [
        np.zeros(5),            
        np.array([0,0,0,-1,-1], dtype=float)
    ]  # x0, x1 :contentReference[oaicite:2]{index=2}

    # enumeration bound
    norms = np.linalg.norm(basis, axis=0)
    max_c = int(np.ceil(n / norms.min())) + 1

    centers = []
    for coeffs in itertools.product(range(-max_c, max_c+1), repeat=5):
        base_pt = basis.dot(coeffs)
        for t in translations:
            p = base_pt + t
            if np.all(np.abs(p) <= n):
                centers.append(p)
    return np.array(centers)

def generate_R5(n):
    """
    R5 (4-periodic) per Szöllősi’s Table 4.2 :contentReference[oaicite:3]{index=3}:
      Basis v0…v4 and translations x0…x3.
    """
    basis = np.column_stack([
        [ 1,  -1,   0,   0,   0],   # v0
        [ 0,   1,  -1,   0,   0],   # v1
        [ 0,   0,   1,  -1,   0],   # v2
        [-0.5,-0.5,-0.5,-0.5,   2],  # v3
        [ 0.8, 0.8, 0.8, 0.8, 0.8]   # v4
    ]).T  # shape (5,5)

    translations = [
        np.zeros(5),            
        np.array([0,0,0, 1,-1], dtype=float),  # x1
        np.array([0,0,-1,-1, 0], dtype=float), # x2
        np.array([0,0, 0,-1,-1], dtype=float)  # x3
    ]  # x0…x3 :contentReference[oaicite:4]{index=4}

    norms = np.linalg.norm(basis, axis=0)
    max_c = int(np.ceil(n / norms.min())) + 1

    centers = []
    for coeffs in itertools.product(range(-max_c, max_c+1), repeat=5):
        base_pt = basis.dot(coeffs)
        for t in translations:
            p = base_pt + t
            if np.all(np.abs(p) <= n):
                centers.append(p)
    return np.array(centers)

def estimate_density_with_kdtree(centers, n, r, M=100_000):
    """
    Estimate the volume fraction in the inner cube [-n+r,n-r]^5
    by sampling M points and nearest‐neighbor queries via cKDTree :contentReference[oaicite:5]{index=5}.
    """
    tree = cKDTree(centers)  # build spatial index

    low, high = -n + r, n - r
    X = np.random.uniform(low, high, size=(M, 5))

    # `workers=-1` uses all cores 
    dists, _ = tree.query(X, k=1, workers=-1)
    return np.count_nonzero(dists <= r) / M

def compute_refined_densities(n, M=100_000):
    r = math.sqrt(2)/2
    gens = {'D5': generate_D5, 'Q5': generate_Q5, 'R5': generate_R5}

    rows = []
    for name, gen in gens.items():
        centers = gen(n)
        dens = estimate_density_with_kdtree(centers, n, r, M=M)
        rows.append({'Packing': name, 'Density': round(dens, 6)})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    np.random.seed(0)
    df = compute_refined_densities(n=2.5, M=200000)
    print(df.to_string(index=False))
