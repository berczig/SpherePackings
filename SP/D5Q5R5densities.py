import numpy as np
import math
import pandas as pd

# 1) Sphere-volume in 5D for r = sqrt(2)/2
r = math.sqrt(2) / 2
V5 = math.pi**(2.5) / math.gamma(3.5) * r**5

# 2) Packing definitions: basis B (columns v0…v4) and coset count k
packings = {
    'D5': {
        'B': np.column_stack([
            [1,  1,  0,  0,  0],
            [1, -1,  1,  0,  0],
            [0,  0, -1,  1,  0],
            [0,  0,  0, -1,  1],
            [0,  0,  0,  0, -1],
        ]),  # det = 2
        'k': 1
    },
    'Q5': {
        'B': np.column_stack([
            [ 1.0,  0.0,  0.0,  0.0, 0.8],
            [-1.0,  1.0,  0.0,  0.0, 0.8],
            [ 0.0, -1.0,  1.0,  0.0, 0.8],
            [ 0.0,  0.0, -1.0,  1.0, 0.8],
            [ 0.0,  0.0,  0.0, -1.0, 0.8],
        ]),  # det = 4
        'k': 2
    },
    'R5': {
        'B': np.column_stack([
            [ 1.0,  0.0,  0.0, -0.5, 0.8],
            [-1.0,  1.0,  0.0, -0.5, 0.8],
            [ 0.0, -1.0,  1.0, -0.5, 0.8],
            [ 0.0,  0.0, -1.0, -0.5, 0.8],
            [ 0.0,  0.0,  0.0,  2.0, 0.8],
        ]),  # det = 8
        'k': 4
    }
}

def compute_counts_and_density(n):
    V_cube = (2*n)**5
    df = []
    for name, data in packings.items():
        detB = abs(np.linalg.det(data['B']))
        k    = data['k']
        num_points = int(round(k * V_cube / detB))
        density    = num_points * V5 / V_cube
        df.append((name, num_points, density))
    return pd.DataFrame(df, columns=['Packing','NumPoints','Density'])

if __name__ == "__main__":
    for n in [3.75, 5.0, 10.75]:
        print(f"\nn = {n}")
        print(compute_counts_and_density(n).to_string(index=False))
