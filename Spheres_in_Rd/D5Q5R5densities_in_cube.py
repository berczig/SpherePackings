import numpy as np
import itertools
import math
import pandas as pd

def generate_D5(n):
    m = int(np.floor(n))
    total = (2*m + 1)**5
    even_count = (total + (total % 2)) // 2
    return np.zeros((even_count, 5))

def generate_Q5(n):
    # Szöllősi’s basis for Q5 (columns v0..v4) and 2 cosets
    basis = np.column_stack([
        [ 1.0,  0.0,  0.0,  0.0, 0.8],
        [-1.0,  1.0,  0.0,  0.0, 0.8],
        [ 0.0, -1.0,  1.0,  0.0, 0.8],
        [ 0.0,  0.0, -1.0,  1.0, 0.8],
        [ 0.0,  0.0,  0.0, -1.0, 0.8],
    ])
    translations = [np.zeros(5), np.ones(5)*0]  # coset translations are built into k=2
    norms = np.linalg.norm(basis, axis=0)
    max_c = int(np.ceil(n / norms.min())) + 1

    count = 0
    for coeffs in itertools.product(range(-max_c, max_c+1), repeat=5):
        vec = basis.dot(coeffs)
        # two cosets: origin and v4 offset
        for t in ([0]*5, [0.8]*5):
            pt = vec + t
            if np.all(np.abs(pt) <= n):
                count += 1
    return count

def generate_R5(n):
    # Szöllősi’s basis for R5 (columns v0..v4) and 4 cosets
    basis = np.column_stack([
        [ 1.0,  0.0,  0.0, -0.5, 0.8],
        [-1.0,  1.0,  0.0, -0.5, 0.8],
        [ 0.0, -1.0,  1.0, -0.5, 0.8],
        [ 0.0,  0.0, -1.0, -0.5, 0.8],
        [ 0.0,  0.0,  0.0,  2.0, 0.8],
    ])
    # The 4 coset translations are exactly the four distinct offsets in Szöllősi’s construction:
    translations = [
        np.zeros(5),
        np.array([0,0,0,1,-1],dtype=float),
        np.array([0,0,-1,-1,0],dtype=float),
        np.array([0,0,0,-1,-1],dtype=float),
    ]
    norms = np.linalg.norm(basis, axis=0)
    max_c = int(np.ceil(n / norms.min())) + 1

    count = 0
    for coeffs in itertools.product(range(-max_c, max_c+1), repeat=5):
        vec = basis.dot(coeffs)
        for t in translations:
            pt = vec + t
            if np.all(np.abs(pt) <= n):
                count += 1
    return count

# Compute counts & densities for n=10.75
n = 3.75
d5_count = generate_D5(n).shape[0]
q5_count = generate_Q5(n)
r5_count = generate_R5(n)

# Sphere‐volume in 5D for r = √2/2
r_sph = math.sqrt(2) / 2
V5 = math.pi**(2.5) / math.gamma(3.5) * r_sph**5
V_cube = (2*n)**5

density_d5 = d5_count * V5 / V_cube
density_q5 = q5_count * V5 / V_cube
density_r5 = r5_count * V5 / V_cube

df = pd.DataFrame({
    'Packing': ['D5', 'Q5', 'R5'],
    'NumPoints': [d5_count, q5_count, r5_count],
    'Density': [density_d5, density_q5, density_r5]
})

#print the results
print(df)
