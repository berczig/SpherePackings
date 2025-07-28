import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import itertools
from scipy.optimize import minimize
from diffuse_boost import cfg

# -----------------------------------------------------------------------------
# PBTS core functions
# -----------------------------------------------------------------------------

def compute_EL_grad(X, L, N, r):
    coords = X.reshape((N, 3))
    EL = 0.0
    grad = np.zeros_like(coords)
    half = L / 2

    # Wall overlaps: penalty if sphere extends beyond walls at +-half
    for i, xi in enumerate(coords):
        for d in range(3):
            dist_to_wall = half - abs(xi[d])
            over = max(0.0, r - dist_to_wall)
            EL += over * over
            if over > 0:
                sign = 1.0 if xi[d] < 0 else -1.0
                grad[i, d] += 2 * over * sign

    # Sphere–sphere overlaps
    for i in range(N):
        for j in range(i + 1, N):
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            over = max(0.0, 2 * r - dist)
            if over > 0:
                EL += over * over
                direction = diff / (dist + 1e-12)
                g = 2 * over * direction
                grad[i] -= g
                grad[j] += g

    return EL, grad.ravel()


def URP(X, eta0):
    return X + np.random.uniform(-eta0, eta0, size=X.shape)


def SRP(X, L, N, r, Imax, m, sigma, beta):
    eta = sigma
    Xc = X.copy()
    for _ in range(Imax):
        Xc += np.random.uniform(-eta, eta, size=Xc.shape)
        for __ in range(m):
            EL, g = compute_EL_grad(Xc, L, N, r)
            g = g.reshape((N, 3))
            norm = np.linalg.norm(g, ord=np.inf) + 1e-12
            Xc -= (sigma * eta) * (g / norm).ravel()
        eta *= beta
    return Xc


def local_opt(X, L, N, r, tol, maxiter):
    x0 = X.ravel()
    res = minimize(
        lambda x: compute_EL_grad(x, L, N, r),
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'ftol': tol, 'gtol': tol, 'maxiter': maxiter}
    )
    return res.x, res.fun


def threshold_search(X, L, N, r, flag,
                     eta0, Imax, m, sigma, beta,
                     MaxIter, tol, maxiter,
                     tau0, rho_tau):
    # Initialize best and current states
    X_best = X.copy()
    EL_best = compute_EL_grad(X_best, L, N, r)[0]
    X_curr = X_best.copy()
    EL_curr = EL_best
    tau = tau0

    for _ in range(MaxIter):
        # generate candidate
        print(f"Current energy: {EL_curr:.6f}, Best energy: {EL_best:.6f}, Threshold: {tau:.6f}")
        if np.random.rand() < 0.5:
            Xt = URP(X_curr, eta0)
        else:
            Xt = SRP(X_curr, L, N, r, Imax, m, sigma, beta)
        # local optimization
        Xt_opt, ELt = local_opt(Xt, L, N, r, tol, maxiter)

        # threshold acceptance on current
        if ELt < EL_curr + tau:
            X_curr, EL_curr = Xt_opt, ELt

        # update best so far
        if ELt < EL_best:
            X_best, EL_best = Xt_opt, ELt

        # decay threshold
        tau *= rho_tau

    return X_best, EL_best


def adjust_container(X, L, N, r, shrink):
    lo, hi = L * (1 - shrink), L
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        EL, _ = compute_EL_grad(X, mid, N, r)
        if EL < 1e-25:
            hi = mid
        else:
            lo = mid
    return X, hi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sample_uniform_points(dimension, num_points):
    return np.random.rand(num_points, dimension)


def get_cube_symmetry_matrices(dim):
    mats = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product([-1, 1], repeat=dim):
            M = np.zeros((dim, dim))
            for i in range(dim):
                M[i, perm[i]] = signs[i]
            mats.append(M)
    return mats


def apply_symmetries_to_data(data, box_width):
    M, D, N = data.shape
    mats = get_cube_symmetry_matrices(D)
    out = np.zeros((M * len(mats), D, N), dtype=data.dtype)
    center = box_width / 2
    idx = 0
    for i in range(M):
        coords = data[i].T
        for mat in mats:
            T = (mat @ (coords - center).T).T + center
            out[idx] = T.T
            idx += 1
    return out

# -----------------------------------------------------------------------------
# Main PBTS-based data-generation
# -----------------------------------------------------------------------------

def generate_dataset_pbts():
    sec  = "sample_generation_PBTS"
    D    = cfg.getint(sec, "dimension")
    Lbox = cfg.getfloat(sec, "bounding_box_width")

    r0      = cfg.getfloat(sec, "sphere_radius")
    best_d  = cfg.getfloat(sec, "best_known_diameter", fallback=2*r0)
    N       = cfg.getint(sec,   "num_spheres")
    M       = cfg.getint(sec,   "num_samples")
    tmax    = cfg.getfloat(sec, "tmax",               fallback=10.0)
    p0      = cfg.getfloat(sec, "p0",                 fallback=0.476)
    MaxIter = cfg.getint(sec,   "MaxIter",            fallback=5000)
    tol     = cfg.getfloat(sec, "tol",                fallback=1e-8)
    maxiter = cfg.getint(sec,   "max_iter",            fallback=300)
    Imax    = cfg.getint(sec,   "Imax",               fallback=500)
    m       = cfg.getint(sec,   "m",                  fallback=20)
    sigma   = cfg.getfloat(sec, "sigma",              fallback=15.0)
    beta    = cfg.getfloat(sec, "beta",               fallback=0.95)
    eta0    = cfg.getfloat(sec, "eta0",               fallback=3.0)
    tau0    = cfg.getfloat(sec, "tau0",               fallback=2.0)
    rho_tau = cfg.getfloat(sec, "rho_tau",            fallback=0.999)
    shrink  = cfg.getfloat(sec, "shrink",             fallback=0.01)

    data = np.zeros((M, D, N), dtype=np.float32)
    min_dists, avg_dists = [], []

    for idx in range(M):
        pts = sample_uniform_points(D, N)

        # Phase 1
        p = p0
        L = (4 * np.pi * N / (3 * p)) ** (1/3)
        X = (pts - 0.5) * L
        print(f"Initial packing fraction: {p:.6f}, Box size: {L:.6f}")
        X, EL = local_opt(X, L, N, r0, tol, maxiter)
        print(f"Initial energy: {EL:.6f}")

        flag = 0
        L_old = L
        X, EL = threshold_search(
            X, L_old, N, r0, flag,
            eta0, Imax, m, sigma, beta,
            MaxIter, tol, maxiter,
            tau0, rho_tau
        )
        print(f"Post-threshold search energy: {EL:.6f}")
        while EL < 1e-25:
            p += 1e-3 * np.random.rand()
            print(f"Adjusting packing fraction to {p:.6f}")
            L = (4 * np.pi * N / (3 * p)) ** (1/3)
            X *= (L / L_old)
            L_old = L
            X, EL = threshold_search(
                X, L_old, N, r0, flag,
                eta0, Imax, m, sigma, beta,
                MaxIter, tol, maxiter,
                tau0, rho_tau
            )

        print(f"Final energy after threshold search: {EL:.6f}")
        X, Lb = adjust_container(X, L_old, N, r0, shrink)
        X_best, L_best = X.copy(), Lb

        # Phase 2
        t0 = time.time()
        while time.time() - t0 < tmax:
            X0 = np.random.uniform(-Lb/2, Lb/2, size=(3*N,))
            X1, EL1 = threshold_search(
                X0, Lb, N, r0, flag,
                eta0, Imax, m, sigma, beta,
                MaxIter, tol, maxiter,
                tau0, rho_tau
            )
            print(f"New candidate energy: {EL1:.6f}")
            if EL1 < 1e-25:
                X1, L1 = adjust_container(X1, Lb, N, r0, shrink)
                if L1 < L_best:
                    X_best, L_best = X1.copy(), L1
                else:
                    flag ^= 1
            else:
                flag ^= 1

        centers = X_best.reshape((N, 3))
        scale = Lbox / L_best
        centers_unit = (centers + L_best/2) * scale
        data[idx] = centers_unit.T

        diffs = centers_unit[:, None, :] - centers_unit[None, :, :]
        dmat = np.linalg.norm(diffs, axis=-1)
        i1, j1 = np.triu_indices(N, k=1)
        pd = dmat[i1, j1]
        mn, av = pd.min(), pd.mean()
        excess = best_d - mn
        min_dists.append(mn)
        avg_dists.append(av)
        print(f"Sample {idx+1}/{M}: min={mn:.6f}, avg={av:.6f}, excess={excess:.6f}")

    # Saving and augmentation as before...

if __name__ == '__main__':
    generate_dataset_pbts()
