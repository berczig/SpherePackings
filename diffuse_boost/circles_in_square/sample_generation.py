# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os, platform
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from scipy.optimize import minimize

import numba as nb
from numba import njit

# Optional config support
try:
    from diffuse_boost import cfg
    HAS_CFG = True
except Exception:
    HAS_CFG = False

# =============================================================================
# Utilities / small constants
# =============================================================================
EPS = 1e-12

def _get_cfg(section, key, fallback):
    if HAS_CFG:
        try:
            if isinstance(fallback, int):
                return cfg.getint(section, key, fallback=fallback)
            if isinstance(fallback, float):
                return cfg.getfloat(section, key, fallback=fallback)
            if isinstance(fallback, bool):
                return cfg.getboolean(section, key, fallback=fallback)
            return cfg.get(section, key, fallback=fallback)
        except Exception:
            return fallback
    return fallback

# =============================================================================
# Random initialization
# =============================================================================
def sample_uniform_centers(N):
    return np.random.uniform(0.0, 1.0, size=(N, 2)).astype(np.float64)

def init_radii(N, r0=0.01, jitter=0.005):
    r = r0 + jitter * (np.random.rand(N) - 0.5) * 2.0
    r = np.maximum(r, 1e-4)
    return r.astype(np.float64)

# =============================================================================
# Objective: overlap + wall penalties - alpha * sum r
# X = [x0,y0,...,x_{N-1},y_{N-1}, r0,...,r_{N-1}]
# =============================================================================
@njit(cache=True, fastmath=True)
def _unit(dx, dy):
    d = math.sqrt(dx*dx + dy*dy) + EPS
    return dx/d, dy/d, d

@njit(cache=True, fastmath=True)
def loss_and_grad(X, N, w_overlap, w_wall, alpha):
    centers = X[:2*N].reshape((N, 2))
    radii   = X[2*N:2*N+N]

    L = 0.0
    g = np.zeros_like(X)

    # Walls
    for i in range(N):
        ri = radii[i]
        xi = centers[i, 0]
        yi = centers[i, 1]
        # x-left
        v = ri - xi
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 0] += -2.0 * w_wall * v
            g[2*N + i] +=  2.0 * w_wall * v
        # x-right
        v = ri - (1.0 - xi)
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 0] +=  2.0 * w_wall * v
            g[2*N + i] +=  2.0 * w_wall * v
        # y-bottom
        v = ri - yi
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 1] += -2.0 * w_wall * v
            g[2*N + i] +=  2.0 * w_wall * v
        # y-top
        v = ri - (1.0 - yi)
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 1] +=  2.0 * w_wall * v
            g[2*N + i] +=  2.0 * w_wall * v

    # Pair overlaps
    for i in range(N):
        xi0, yi0, ri = centers[i,0], centers[i,1], radii[i]
        for j in range(i+1, N):
            dx = xi0 - centers[j,0]
            dy = yi0 - centers[j,1]
            ux, uy, d = _unit(dx, dy)
            over = (ri + radii[j]) - d
            if over > 0.0:
                L += w_overlap * (over * over)
                c = 2.0 * w_overlap * over
                # i center  (NOTE the minus signs)
                g[2*i + 0] += -c * ux
                g[2*i + 1] += -c * uy
                # j center  (opposite)
                g[2*j + 0] +=  c * ux
                g[2*j + 1] +=  c * uy
                # radii stay the same
                g[2*N + i] +=  c
                g[2*N + j] +=  c


    # Maximize sum radii
    L += -alpha * np.sum(radii)
    for i in range(N):
        g[2*N + i] += -alpha

    return L, g

# =============================================================================
# SRP (adaptive) with backtracking
# =============================================================================
@njit(cache=True, fastmath=True)
def _l2_norm(v):
    s = 0.0
    for k in range(v.size):
        s += v[k]*v[k]
    return math.sqrt(s) + 1e-18

@njit(cache=True, fastmath=True)
def srp_adaptive(X, N, Imax, m, step_center, step_radius, beta, backtrack,
                 w_overlap, w_wall, alpha):
    Xc = X.copy()
    eta = 1.0
    for _ in range(Imax):
        # jitter
        noise = np.zeros_like(Xc)
        for i in range(N):
            noise[2*i+0] = (np.random.rand()*2.0 - 1.0) * (eta * step_center)
            noise[2*i+1] = (np.random.rand()*2.0 - 1.0) * (eta * step_center)
        for i in range(N):
            noise[2*N + i] = (np.random.rand()*2.0 - 1.0) * (eta * step_radius)
        Xtrial = Xc + noise

        # m gradient steps with simple backtracking
        for __ in range(m):
            L0, g = loss_and_grad(Xtrial, N, w_overlap, w_wall, alpha)

            step = np.empty_like(g)
            for i in range(N):
                step[2*i+0] = eta * step_center
                step[2*i+1] = eta * step_center
            for i in range(N):
                step[2*N + i] = eta * step_radius

            gn = _l2_norm(g)
            if gn > 0.0:
                Xprop = Xtrial - step * (g / gn)
                Lprop, _ = loss_and_grad(Xprop, N, w_overlap, w_wall, alpha)

                bt = 0
                while Lprop > L0 and bt < backtrack:
                    for i in range(N):
                        step[2*i+0] *= 0.5
                        step[2*i+1] *= 0.5
                    for i in range(N):
                        step[2*N + i] *= 0.5
                    Xprop = Xtrial - step * (g / gn)
                    Lprop, _ = loss_and_grad(Xprop, N, w_overlap, w_wall, alpha)
                    bt += 1

                Xtrial = Xprop

        # clip to domain for stability
        for i in range(N):
            if Xtrial[2*i+0] < 0.0: Xtrial[2*i+0] = 0.0
            if Xtrial[2*i+0] > 1.0: Xtrial[2*i+0] = 1.0
            if Xtrial[2*i+1] < 0.0: Xtrial[2*i+1] = 0.0
            if Xtrial[2*i+1] > 1.0: Xtrial[2*i+1] = 1.0
        for i in range(N):
            if Xtrial[2*N + i] < 0.0: Xtrial[2*N + i] = 0.0
            if Xtrial[2*N + i] > 0.5: Xtrial[2*N + i] = 0.5

        Xc = Xtrial
        eta *= beta
    return Xc

# =============================================================================
# Local optimization (L-BFGS-B)
# =============================================================================
def local_optimize(X0, N, w_overlap, w_wall, alpha, gtol=1e-8, ftol=1e-12,
                   maxiter=1000, maxcor=20):
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N

    def fun(x):
        L, _ = loss_and_grad(x, N, w_overlap, w_wall, alpha)
        return L

    def jac(x):
        _, g = loss_and_grad(x, N, w_overlap, w_wall, alpha)
        return g

    res = minimize(
        fun=fun, x0=X0, method='L-BFGS-B', jac=jac, bounds=bounds,
        options={'gtol': gtol, 'ftol': ftol, 'maxiter': maxiter, 'maxcor': maxcor}
    )
    return res.x, res.fun

# =============================================================================
# Metrics
# =============================================================================
@njit(cache=True, fastmath=True)
def min_wall_clearance(centers, radii):
    N = centers.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi = centers[i,0], centers[i,1]
        ri = radii[i]
        c1 = xi - ri
        if c1 < best: best = c1
        c2 = 1.0 - xi - ri
        if c2 < best: best = c2
        c3 = yi - ri
        if c3 < best: best = c3
        c4 = 1.0 - yi - ri
        if c4 < best: best = c4
    return best

@njit(cache=True, fastmath=True)
def min_pair_clearance(centers, radii):
    N = centers.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi, ri = centers[i,0], centers[i,1], radii[i]
        for j in range(i+1, N):
            dx = xi - centers[j,0]
            dy = yi - centers[j,1]
            d  = math.sqrt(dx*dx + dy*dy)
            clr = d - (ri + radii[j])
            if clr < best:
                best = clr
    return best

def hard_project_max_sum_radii(centers, radii, safety=1e-9, pair_safety_mul=1.0):
    """
    Given centers (N,2) in [0,1]^2 and current radii (N,),
    compute radii' that:
      maximize sum(r_i)
      subject to:
        0 <= r_i <= wall_i
        r_i + r_j <= d_ij
    and then shrink by a tiny `safety` to avoid visual tangency.
    Returns: radii_proj (N,), info dict with diagnostics.
    """
    import numpy as np
    from scipy.optimize import linprog

    N = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]

    # Wall upper bounds for each circle
    wall_ub = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    wall_ub = np.clip(wall_ub - safety, 0.0, None)  # tiny safety

    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = np.sqrt(dx * dx + dy * dy)

    # Build LP:
    # maximize sum r  <=>  minimize -sum r
    c = -np.ones(N, dtype=float)

    # Bounds: 0 <= r_i <= wall_ub[i]
    bounds = [(0.0, float(wall_ub[i])) for i in range(N)]

    # Inequalities A_ub @ r <= b_ub
    # For each pair (i<j): r_i + r_j <= d_ij - safety
    # We do not need to include wall constraints in A_ub; they’re in `bounds`.
    rows = []
    rhs = []
    for i in range(N):
        for j in range(i + 1, N):
            dij = D[i, j]
            # Optional: a multiplicative slack on pairwise distance if desired
            rhs_ij = max(dij * pair_safety_mul - safety, 0.0)
            row = np.zeros(N, dtype=float)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(rhs_ij)

    if rows:
        A_ub = np.vstack(rows)
        b_ub = np.asarray(rhs, dtype=float)
    else:
        # N < 2: trivial case
        A_ub = None
        b_ub = None

    # Solve LP
    res = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )

    info = {
        "success": res.success,
        "status": res.status,
        "message": res.message,
        "sum_r": None,
        "violations": None,
    }

    if not res.success:
        # Fallback: clamp to wall_ub (feasible w.r.t. walls; pairs may still violate)
        r_proj = np.minimum(radii, wall_ub).copy()
        info["sum_r"] = float(np.sum(r_proj))
        info["message"] = f"LP failed; returned clamped radii. HiGHS status {res.status}: {res.message}"
        return r_proj, info

    r_star = res.x

    # Final tiny shrink to guarantee strict feasibility in plotting
    r_proj = np.maximum(0.0, r_star - safety)

    # Diagnostics: check min clearances
    def _min_wall_clear(cent, rr):
        xi, yi = cent[:, 0], cent[:, 1]
        return float(np.min([xi - rr, 1.0 - xi - rr, yi - rr, 1.0 - yi - rr]))

    def _min_pair_clear(cent, rr):
        dx = cent[:, None, 0] - cent[None, :, 0]
        dy = cent[:, None, 1] - cent[None, :, 1]
        D = np.sqrt(dx * dx + dy * dy)
        # set diagonal to large so it won't affect min
        np.fill_diagonal(D, np.inf)
        S = rr[:, None] + rr[None, :]
        return float(np.min(D - S))

    mwc = _min_wall_clear(centers, r_proj)
    mpc = _min_pair_clear(centers, r_proj)

    info["sum_r"] = float(np.sum(r_proj))
    info["violations"] = {"min_wall_clear": mwc, "min_pair_clear": mpc}
    return r_proj, info



# =============================================================================
# Plotting (NEW)
# =============================================================================
def plot_first_k_samples(data_tensor, k, out_dir, filename_prefix="sample"):
    """
    Draws the first k samples from data tensor of shape (M, 3, N) with rows [x, y, r].
    Creates one figure per sample and saves PNGs to out_dir.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    os.makedirs(out_dir, exist_ok=True)

    M = data_tensor.shape[0]
    k = min(k, M)

    for s in range(k):
        xy = data_tensor[s, :2, :].T  # (N,2)
        rr = data_tensor[s, 2, :]     # (N,)
        fig, ax = plt.subplots(figsize=(5,5))
        # unit square boundary
        ax.plot([0,1,1,0,0], [0,0,1,1,0])
        # draw circles
        for i in range(xy.shape[0]):
            c = Circle((xy[i,0], xy[i,1]), rr[i], fill=False)
            ax.add_patch(c)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Put sum of radii to title
        sum_r = float(np.sum(rr))
        ax.set_title(f"Sample #{s}, num_circles= {xy.shape[0]}, sum_r = {sum_r:.4f})")
        out_path = os.path.join(out_dir, f"{filename_prefix}_{xy.shape[0]}_{sum_r:.4f}_{s}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# Main generator
# =============================================================================
def generate_circle_packing_dataset():
    sec = "circle_packing_SRP"

    # Config / defaults
    N   = _get_cfg(sec, "num_circles",  50)
    M   = _get_cfg(sec, "num_samples",  5000)

    # SRP hyperparams
    Imax       = _get_cfg(sec, "srp_Imax",        400)
    m          = _get_cfg(sec, "srp_m",           30)
    beta       = _get_cfg(sec, "srp_beta",        0.985)
    backtrack  = _get_cfg(sec, "srp_backtrack",   3)
    step_center= _get_cfg(sec, "srp_step_center", 0.05)
    step_radius= _get_cfg(sec, "srp_step_radius", 0.01)

    # Penalty/objective weights
    w_overlap  = _get_cfg(sec, "w_overlap",       1.0)
    w_wall     = _get_cfg(sec, "w_wall",          1.0)
    alpha      = _get_cfg(sec, "alpha_sum_r",     1.0)

    # Local opt
    gtol       = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol       = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter    = _get_cfg(sec, "lbfgs_maxiter",   1000)
    maxcor     = _get_cfg(sec, "lbfgs_maxcor",    20)

    # Initialization
    r0         = _get_cfg(sec, "init_r0",         0.01)
    rj         = _get_cfg(sec, "init_r_jitter",   0.005)

    # I/O + plotting
    out_dir    = _get_cfg(sec, "output_dir",      "./outputs_circle_packing")
    plot_k     = _get_cfg(sec, "plot_k",          0) 
    plot_dir   = os.path.join(out_dir, "plots")

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"circle_srp_generated_{M}x{N}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"circle_srp_metrics_{M}x{N}_{stamp}.csv")

    print(f"Generating {M} circle packings in unit square (N={N})")
    with open(metrics_fn, "w") as mf:
        mf.write("sample,sum_r,min_wall_clear,min_pair_clear,loss_after\n")

    data = np.zeros((M, 3, N), dtype=np.float32)

    bar = tqdm(range(M), desc="Generating circle packings")
    for s in bar:
        # Random start
        C0 = sample_uniform_centers(N)
        R0 = init_radii(N, r0=r0, jitter=rj)
        X0 = np.concatenate([C0.ravel(), R0], axis=0)

        # SRP exploration
        X_srp = srp_adaptive(
            X0, N,
            Imax=Imax, m=m,
            step_center=step_center,
            step_radius=step_radius,
            beta=beta, backtrack=backtrack,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha
        )

        # Local refinement
        X_fin, L_fin = local_optimize(
            X_srp, N,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha,
            gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
        )

        centers = X_fin[:2*N].reshape(N, 2)
        radii   = X_fin[2*N:2*N+N]

        # Project to hard-feasible radii with max sum
        r_proj, info = hard_project_max_sum_radii(centers, radii, safety=1e-9)

        # Replace radii for saving / plotting
        radii = r_proj

        # Optional: log diagnostics
        print(f"[hard-project] success={info['success']} sum_r={info['sum_r']:.6f} "
        f"min_wall={info['violations']['min_wall_clear']:.3e} "
        f"min_pair={info['violations']['min_pair_clear']:.3e}")

        # Metrics
        sum_r   = float(np.sum(radii))
        mwc     = float(min_wall_clearance(centers, radii))
        mpc     = float(min_pair_clearance(centers, radii))

        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{sum_r:.8f},{mwc:.8f},{mpc:.8f},{L_fin:.8e}\n")

        data[s, 0, :] = centers[:,0].astype(np.float32)
        data[s, 1, :] = centers[:,1].astype(np.float32)
        data[s, 2, :] = radii.astype(np.float32)

        bar.set_postfix(sum_r=f"{sum_r:.3f}", clr=f"{min(mwc, mpc):.4f}")

    # Sort the samples by sum of radii
    sum_radii = data[:, 2, :].sum(axis=1)
    sorted_indices = np.argsort(-sum_radii)  # descending
    data = data[sorted_indices]
    
    # Save dataset
    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved dataset:  {dataset_fn}")
    print(f"Saved metrics:  {metrics_fn}")

    # Save plots for the first top_k samples
    if plot_k > 0:
        plot_first_k_samples(data, plot_k, plot_dir, filename_prefix="circle_packing")
        print(f"Saved plots:   {plot_dir}") 
# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    generate_circle_packing_dataset()
