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
def sample_uniform_points(N):
    return np.random.uniform(0.0, 1.0, size=(N, 2)).astype(np.float64)

# =============================================================================
# Geometry helpers
# =============================================================================

def _triangle_area_exact(p1, p2, p3):
    """
    Exact (non-smoothed) triangle area: 0.5 * |(p2-p1) x (p3-p1)|
    p1,p2,p3 are 2D numpy arrays.
    """
    ux, uy = p2[0] - p1[0], p2[1] - p1[1]
    vx, vy = p3[0] - p1[0], p3[1] - p1[1]
    return 0.5 * abs(ux * vy - uy * vx)


def find_min_area_and_triangles(pts, atol=1e-12, rtol=1e-9):
    """
    Given pts: (N,2), return (A_min, triangles) where triangles is a list of (i,j,k)
    for all triples whose area equals the minimum within tolerance:
        |A - A_min| <= max(atol, rtol*A_min)

    We compute A_min first, then collect all ties within tolerance.
    """
    N = pts.shape[0]
    if N < 3:
        return 0.0, []

    # First pass: find minimum area
    A_min = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                A = _triangle_area_exact(pts[i], pts[j], pts[k])
                if A < A_min:
                    A_min = A

    tol = max(atol, rtol * max(A_min, 1.0))  # robust tolerance
    # Second pass: collect all triangles achieving the min area within tol
    tris = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                A = _triangle_area_exact(pts[i], pts[j], pts[k])
                if abs(A - A_min) <= tol:
                    tris.append((i, j, k))

    return A_min, tris

@njit(cache=True, fastmath=True)
def _l2_norm(v):
    s = 0.0
    for k in range(v.size):
        s += v[k]*v[k]
    return math.sqrt(s) + 1e-18

@njit(cache=True, fastmath=True)
def _triangle_area_and_grads(p1x, p1y, p2x, p2y, p3x, p3y, eps_abs=1e-12):
    """
    Smooth absolute triangle area with gradients.
    A = 0.5 * sqrt(cross^2 + eps_abs)
    cross = (p2 - p1) x (p3 - p1) = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)

    Returns: A, dA/dp1x, dA/dp1y, dA/dp2x, dA/dp2y, dA/dp3x, dA/dp3y
    """
    ux = p2x - p1x
    uy = p2y - p1y
    vx = p3x - p1x
    vy = p3y - p1y

    cross = ux*vy - uy*vx
    denom = math.sqrt(cross*cross + eps_abs)
    dA_dcross = 0.5 * cross / denom  # derivative of 0.5*sqrt(c^2+eps)

    # derivatives of cross wrt coordinates
    # cross = ux*vy - uy*vx
    # wrt p1: ux = (x2-x1) -> d/dx1 = -vy *? careful:
    # d cross / d p1x = d(ux*vy - uy*vx)/d p1x = (0)*vy + ux*(0) - (0)*vx - uy*( -1 ) - ( -1 )*vy*? Let's do directly via partials:
    # Better: expand cross in original coords:
    # cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    # d/dx1: = ( -1 )*(y3-y1) + (x2-x1)*( -1 ) - (y2-y1)*( 0 - 1? )? Let's compute systematically:
    # We'll use compact result known for triangle cross:
    # d cross / d p1 = (y2 - y3, x3 - x2)
    # d cross / d p2 = (y3 - y1, x1 - x3)
    # d cross / d p3 = (y1 - y2, x2 - x1)
    dc_dp1x = (p2y - p3y)
    dc_dp1y = (p3x - p2x)
    dc_dp2x = (p3y - p1y)
    dc_dp2y = (p1x - p3x)
    dc_dp3x = (p1y - p2y)
    dc_dp3y = (p2x - p1x)

    dA_dp1x = dA_dcross * dc_dp1x
    dA_dp1y = dA_dcross * dc_dp1y
    dA_dp2x = dA_dcross * dc_dp2x
    dA_dp2y = dA_dcross * dc_dp2y
    dA_dp3x = dA_dcross * dc_dp3x
    dA_dp3y = dA_dcross * dc_dp3y

    A = 0.5 * denom
    return A, dA_dp1x, dA_dp1y, dA_dp2x, dA_dp2y, dA_dp3x, dA_dp3y

# =============================================================================
# Loss: walls + smooth negative soft-min of triangle areas
# X = [x0,y0,...,x_{N-1},y_{N-1}]
# =============================================================================
@njit(cache=True, fastmath=True)
def heilbronn_loss_and_grad(X, N, w_wall, beta_softmin,
                            eps_abs=1e-12, topk_K=0, topk_tol=1e-12):
    """
    L = wall penalties + (-softmin over areas)
    softmin(A) = -(1/beta) * log( sum_t exp(-beta * A_t) )

    If topk_K > 0, we only include triangles whose area is among the K smallest
    (within tolerance topk_tol). This concentrates gradients on the bottlenecks.
    """
    pts = X[:2*N].reshape((N, 2))
    g = np.zeros_like(X)
    L = 0.0

    # ---- Walls ----
    for i in range(N):
        xi = pts[i, 0]
        yi = pts[i, 1]

        v = -xi
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 0] += -2.0 * w_wall * v

        v = xi - 1.0
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 0] +=  2.0 * w_wall * v

        v = -yi
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 1] += -2.0 * w_wall * v

        v = yi - 1.0
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 1] +=  2.0 * w_wall * v

    # ---- Triangles ----
    T = N*(N-1)*(N-2)//6
    if T == 0:
        return L, g

    # (A) If Top-K enabled, find threshold A_thresh = K-th smallest area
    A_thresh = 1e99
    use_topk = (topk_K > 0) and (topk_K < T)
    if use_topk:
        # Keep an unsorted buffer of size K with the current K smallest areas.
        K = topk_K
        buf = np.empty(K, dtype=np.float64)
        for kk in range(K):
            buf[kk] = 1e99

        def _buf_worst_val(arr):
            worst = -1.0
            worst_i = -1
            for ii in range(arr.size):
                if arr[ii] > worst:
                    worst = arr[ii]
                    worst_i = ii
            return worst, worst_i

        worst, worst_i = _buf_worst_val(buf)

        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, d1x, d1y, d2x, d2y, d3x, d3y = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    # Insert into buf if better than current worst
                    if A < worst:
                        buf[worst_i] = A
                        # recompute worst
                        worst, worst_i = _buf_worst_val(buf)
        A_thresh = worst  # K-th smallest

    # (B) Compute Z and gradients; if Top-K, only include A <= A_thresh + tol
    Z = 0.0
    if use_topk:
        # First compute Z with the gate
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, d1x, d1y, d2x, d2y, d3x, d3y = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    if A <= A_thresh + topk_tol:
                        Z += math.exp(-beta_softmin * A)
    else:
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, d1x, d1y, d2x, d2y, d3x, d3y = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    Z += math.exp(-beta_softmin * A)

    if Z <= 0.0:
        # extremely degenerate
        return L, g

    neg_softmin = (math.log(Z) / beta_softmin)  # equals -softmin
    L += neg_softmin

    invZ = 1.0 / Z
    if use_topk:
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, d1x, d1y, d2x, d2y, d3x, d3y = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    if A <= A_thresh + topk_tol:
                        w_t = math.exp(-beta_softmin * A) * invZ
                        coeff = -w_t  # dL/dA
                        g[2*i+0] += coeff * d1x; g[2*i+1] += coeff * d1y
                        g[2*j+0] += coeff * d2x; g[2*j+1] += coeff * d2y
                        g[2*k+0] += coeff * d3x; g[2*k+1] += coeff * d3y
    else:
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, d1x, d1y, d2x, d2y, d3x, d3y = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    w_t = math.exp(-beta_softmin * A) * invZ
                    coeff = -w_t
                    g[2*i+0] += coeff * d1x; g[2*i+1] += coeff * d1y
                    g[2*j+0] += coeff * d2x; g[2*j+1] += coeff * d2y
                    g[2*k+0] += coeff * d3x; g[2*k+1] += coeff * d3y

    return L, g


# =============================================================================
# SRP (adaptive) with backtracking (points only)
# =============================================================================
@njit(cache=True, fastmath=True)
def srp_adaptive_points(X, N, Imax, m, step_center, beta_sched_decay, backtrack,
                        w_wall,
                        beta_softmin_start, beta_softmin_final,
                        eps_abs,
                        topk_K, topk_tol):
    """
    Deterministic annealing: beta(t) grows from start->final over Imax iterations.
    Exponential schedule: beta(t) = beta_start * (beta_final / beta_start)^(t / Imax)
    """
    Xc = X.copy()
    eta = 1.0
    for it in range(Imax):
        # Exponential schedule for beta
        tfrac = (it + 0.0) / max(Imax - 1, 1)
        beta_t = beta_softmin_start * math.pow(beta_softmin_final / beta_softmin_start, tfrac)

        # jitter
        noise = np.zeros_like(Xc)
        for i in range(N):
            noise[2*i+0] = (np.random.rand()*2.0 - 1.0) * (eta * step_center)
            noise[2*i+1] = (np.random.rand()*2.0 - 1.0) * (eta * step_center)
        Xtrial = Xc + noise

        # m gradient steps with backtracking
        for __ in range(m):
            L0, g = heilbronn_loss_and_grad(
                Xtrial, N, w_wall, beta_t, eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol
            )

            step = np.empty_like(g)
            for i in range(N):
                step[2*i+0] = eta * step_center
                step[2*i+1] = eta * step_center

            gn = _l2_norm(g)
            if gn > 0.0:
                Xprop = Xtrial - step * (g / gn)
                Lprop, _ = heilbronn_loss_and_grad(
                    Xprop, N, w_wall, beta_t, eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol
                )

                bt = 0
                while Lprop > L0 and bt < backtrack:
                    for i in range(N):
                        step[2*i+0] *= 0.5
                        step[2*i+1] *= 0.5
                    Xprop = Xtrial - step * (g / gn)
                    Lprop, _ = heilbronn_loss_and_grad(
                        Xprop, N, w_wall, beta_t, eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol
                    )
                    bt += 1

                Xtrial = Xprop

        # clip to domain
        for i in range(N):
            if Xtrial[2*i+0] < 0.0: Xtrial[2*i+0] = 0.0
            if Xtrial[2*i+0] > 1.0: Xtrial[2*i+0] = 1.0
            if Xtrial[2*i+1] < 0.0: Xtrial[2*i+1] = 0.0
            if Xtrial[2*i+1] > 1.0: Xtrial[2*i+1] = 1.0

        Xc = Xtrial
        eta *= beta_sched_decay
    return Xc

# =============================================================================
# Local optimization (L-BFGS-B) on [0,1]^{2N}
# =============================================================================
def local_optimize_points(X0, N, w_wall,
                          beta_softmin_final, eps_abs,
                          topk_K, topk_tol,
                          gtol=1e-8, ftol=1e-12, maxiter=1000, maxcor=20):
    bounds = [(0.0, 1.0)] * (2*N)

    def fun(x):
        L, _ = heilbronn_loss_and_grad(x, N, w_wall, beta_softmin_final,
                                       eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol)
        return L

    def jac(x):
        _, g = heilbronn_loss_and_grad(x, N, w_wall, beta_softmin_final,
                                       eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol)
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
def min_wall_clearance_points(pts):
    N = pts.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi = pts[i,0], pts[i,1]
        c1 = xi
        if c1 < best: best = c1
        c2 = 1.0 - xi
        if c2 < best: best = c2
        c3 = yi
        if c3 < best: best = c3
        c4 = 1.0 - yi
        if c4 < best: best = c4
    return best

@njit(cache=True, fastmath=True)
def min_pair_distance(pts):
    N = pts.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi = pts[i,0], pts[i,1]
        for j in range(i+1, N):
            dx = xi - pts[j,0]
            dy = yi - pts[j,1]
            d  = math.sqrt(dx*dx + dy*dy)
            if d < best:
                best = d
    return best

@njit(cache=True, fastmath=True)
def min_triangle_area(pts, eps_abs=1e-12):
    N = pts.shape[0]
    best = 1e18
    for i in range(N):
        x1, y1 = pts[i,0], pts[i,1]
        for j in range(i+1, N):
            x2, y2 = pts[j,0], pts[j,1]
            for k in range(j+1, N):
                x3, y3 = pts[k,0], pts[k,1]
                A, d1x, d1y, d2x, d2y, d3x, d3y = _triangle_area_and_grads(x1, y1, x2, y2, x3, y3, eps_abs)
                if A < best:
                    best = A

    return best

# =============================================================================
# Plotting (points + one smallest triangle)
# =============================================================================
def plot_top_k_minarea_samples(data_tensor, k, out_dir, filename_prefix="heilbronn_top"):
    """
    data_tensor: (M, 2, N) with rows [x, y], presumably already sorted by descending min-area.
    If not sorted, this function will compute the order by min-area anyway.

    For each of the top-k samples, draws points and overlays all triangles whose
    area equals the sample's minimum (within tolerance).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    M = data_tensor.shape[0]
    k = min(k, M)

    # Compute per-sample min area (in case upstream didn’t pre-sort correctly)
    min_areas = np.zeros(M, dtype=np.float64)
    for s in range(M):
        pts = data_tensor[s].T.astype(np.float64)  # (N,2)
        A_min, _ = find_min_area_and_triangles(pts)
        min_areas[s] = A_min

    # Sort by descending min-area
    order = np.argsort(-min_areas)

    for rank in range(k):
        s = int(order[rank])
        pts = data_tensor[s].T.astype(np.float64)
        A_min, tris = find_min_area_and_triangles(pts)

        fig, ax = plt.subplots(figsize=(5, 5))
        # unit square
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])

        # points
        ax.scatter(pts[:, 0], pts[:, 1], s=12)

        # overlay ALL minimal triangles
        # (thin lines to keep multiple overlays readable)
        for (i, j, ktri) in tris:
            poly = np.array([pts[i], pts[j], pts[ktri], pts[i]])
            ax.plot(poly[:, 0], poly[:, 1], linewidth=1.0)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Top #{rank+1} (sample {s}), N={pts.shape[0]}, "
                     f"min area = {A_min:.10f}, triangles = {len(tris)}")
        out_path = os.path.join(out_dir, f"{filename_prefix}_n={pts.shape[0]}_Amin_{A_min:.10f}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# Main generator
# =============================================================================
def generate_heilbronn_dataset():
    sec = "heilbronn_SRP"

    # Config / defaults
    N   = _get_cfg(sec, "num_points",   50)
    M   = _get_cfg(sec, "num_samples",  5000)

    # SRP hyperparams
    Imax        = _get_cfg(sec, "srp_Imax",        400)
    m           = _get_cfg(sec, "srp_m",           30)
    beta_sched  = _get_cfg(sec, "srp_beta",        0.985)
    backtrack   = _get_cfg(sec, "srp_backtrack",   3)
    step_center = _get_cfg(sec, "srp_step_center", 0.05)

    # Loss params
    w_wall      = _get_cfg(sec, "w_wall",          1.0)
    beta_softmin= _get_cfg(sec, "beta_softmin",    50.0)   # higher β → sharper min
    eps_abs     = _get_cfg(sec, "area_eps_abs",    1e-12)

    # Local opt
    gtol        = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol        = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter     = _get_cfg(sec, "lbfgs_maxiter",   1000)
    maxcor      = _get_cfg(sec, "lbfgs_maxcor",    20)

        # Loss/annealing
    w_wall           = _get_cfg(sec, "w_wall",             1.0)
    beta_softmin0    = _get_cfg(sec, "beta_softmin_start", 40.0)
    beta_softminF    = _get_cfg(sec, "beta_softmin_final", 300.0)
    eps_abs          = _get_cfg(sec, "area_eps_abs",       1e-12)

    # Top-K
    topk_K           = _get_cfg(sec, "topk_K",             0)
    topk_tol         = _get_cfg(sec, "topk_tol",           1e-12)


    # I/O + plotting
    out_dir     = _get_cfg(sec, "output_dir",      "./outputs_heilbronn")
    plot_k      = _get_cfg(sec, "plot_k",          0)
    plot_dir    = os.path.join(out_dir, "plots")

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"heilbronn_srp_{M}x{N}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"heilbronn_metrics_{M}x{N}_{stamp}.csv")

    print(f"Generating {M} Heilbronn point sets in unit square (N={N})")
    with open(metrics_fn, "w") as mf:
        mf.write("sample,min_triangle_area,min_wall_clear,min_pair_dist,loss_after\n")

    data = np.zeros((M, 2, N), dtype=np.float32)

    bar = tqdm(range(M), desc="Generating point sets")
    for s in bar:
        # Random start
        P0 = sample_uniform_points(N)
        X0 = P0.ravel()

        # SRP exploration
        X_srp = srp_adaptive_points(
            X0, N,
            Imax=Imax, m=m,
            step_center=step_center,
            beta_sched_decay=beta_sched, backtrack=backtrack,
            w_wall=w_wall,
            beta_softmin_start=beta_softmin0, beta_softmin_final=beta_softminF,
            eps_abs=eps_abs,
            topk_K=topk_K, topk_tol=topk_tol
        )

        X_fin, L_fin = local_optimize_points(
            X_srp, N,
            w_wall=w_wall,
            beta_softmin_final=beta_softminF, eps_abs=eps_abs,
            topk_K=topk_K, topk_tol=topk_tol,
            gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
        )


        pts = X_fin.reshape(N, 2)

        # Metrics
        A_min = float(min_triangle_area(pts, eps_abs))
        mwc   = float(min_wall_clearance_points(pts))
        mpd   = float(min_pair_distance(pts))

        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{A_min:.10f},{mwc:.10f},{mpd:.10f},{L_fin:.8e}\n")

        data[s, 0, :] = pts[:,0].astype(np.float32)
        data[s, 1, :] = pts[:,1].astype(np.float32)

        bar.set_postfix(min_area=f"{A_min:.6f}", clr=f"{min(mwc, mpd):.4f}")

    # Sort samples by min triangle area (descending)
    # compute per-sample min area again (cheap for ordering)
    scores = np.zeros(M, dtype=np.float64)
    for s in range(M):
        pts = data[s].T.astype(np.float64)
        scores[s] = min_triangle_area(pts, 1e-12)
    sorted_indices = np.argsort(-scores)
    data = data[sorted_indices]

    # Save dataset
    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved dataset:  {dataset_fn}")
    print(f"Saved metrics:  {metrics_fn}")

    # Save plots for the top_k samples (by minimum triangle area)
    if plot_k > 0:
        plot_top_k_minarea_samples(data, plot_k, plot_dir, filename_prefix="heilbronn_mintriangles")
        print(f"Saved plots:   {plot_dir}")


# =============================================================================
# Final-push mode: apply SRP to an existing [M,2,N] sample tensor
# =============================================================================
def final_push_existing_samples():
    sec = "heilbronn_SRP"

    # Config / defaults
    N   = _get_cfg(sec, "num_points",   50)
    M   = _get_cfg(sec, "num_samples",  5000)

    # SRP hyperparams
    Imax        = _get_cfg(sec, "srp_Imax",        400)
    m           = _get_cfg(sec, "srp_m",           30)
    beta_sched  = _get_cfg(sec, "srp_beta",        0.985)
    backtrack   = _get_cfg(sec, "srp_backtrack",   3)
    step_center = _get_cfg(sec, "srp_step_center", 0.05)

    # Loss params
    w_wall      = _get_cfg(sec, "w_wall",          1.0)
    beta_softmin= _get_cfg(sec, "beta_softmin",    50.0)   # higher β → sharper min
    eps_abs     = _get_cfg(sec, "area_eps_abs",    1e-12)

    # Local opt
    gtol        = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol        = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter     = _get_cfg(sec, "lbfgs_maxiter",   1000)
    maxcor      = _get_cfg(sec, "lbfgs_maxcor",    20)

    # Loss/annealing
    w_wall           = _get_cfg(sec, "w_wall",             1.0)
    beta_softmin0    = _get_cfg(sec, "beta_softmin_start", 40.0)
    beta_softminF    = _get_cfg(sec, "beta_softmin_final", 300.0)
    eps_abs          = _get_cfg(sec, "area_eps_abs",       1e-12)

    # Top-K
    topk_K           = _get_cfg(sec, "topk_K",             0)
    topk_tol         = _get_cfg(sec, "topk_tol",           1e-12)

    # I/O + plotting
    out_dir     = _get_cfg(sec, "final_push_output",      "./outputs_heilbronn")
    plot_k      = _get_cfg(sec, "plot_k",          0)
    plot_dir    = os.path.join(out_dir, "plots")

    # NEW: where to read existing samples (torch .pt tensor, shape (M,2,N) or (M,N,2))
    input_path  = _get_cfg(sec, "final_push_input", "")
    assert isinstance(input_path, str) and len(input_path) > 0 and os.path.exists(input_path), \
        "Set heilbronn_SRP.final_push_input to a valid .pt file"

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"heilbronn_srp_pushed_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"heilbronn_metrics_{stamp}.csv")

    # NEW: load the input tensor (Torch), normalize shape to (M,2,N)
    loaded = torch.load(input_path, map_location="cpu")  # supports tensors saved via torch.save
    if isinstance(loaded, torch.Tensor):
        arr = loaded.detach().cpu().numpy()
    else:
        # if it was saved as a dict or list, try to extract the tensor-like payload
        raise ValueError("Expected a tensor at final_push_input")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {arr.shape}")

    # Accept either (M,2,N) or (M,N,2); enforce (M,2,N)
    if arr.shape[1] == 2:
        M_in, d_in, N_in = arr.shape
    elif arr.shape[-1] == 2:
        # transpose to (M,2,N)
        arr = np.transpose(arr, (0, 2, 1))
        M_in, d_in, N_in = arr.shape
    else:
        raise ValueError(f"Second or last dimension must be 2; got shape {arr.shape}")
    if d_in != 2:
        raise ValueError("Point dimension must be 2")

    # If config provided M/N, cap/verify against input
    if M is None or M <= 0 or M > M_in:
        M = M_in
    if N is None or N <= 0:
        N = N_in
    if N != N_in:
        raise ValueError(f"Config N={N} but input samples have N={N_in} points")

    print(f"Pushing {M} loaded Heilbronn point sets from file (N={N})")
    with open(metrics_fn, "w") as mf:
        mf.write("sample,min_triangle_area,min_wall_clear,min_pair_dist,loss_after\n")

    # allocate output (same shape as input subset)
    data = np.zeros((M, 2, N), dtype=np.float32)

    bar = tqdm(range(M), desc="Pushing point sets")
    for s in bar:
        # NEW: start from loaded sample instead of random
        # arr[s] is (2,N); convert to (N,2) float64 for optimizer
        P0 = arr[s].astype(np.float64).T
        X0 = P0.ravel()

        # SRP exploration (unchanged)
        X_srp = srp_adaptive_points(
            X0, N,
            Imax=Imax, m=m,
            step_center=step_center,
            beta_sched_decay=beta_sched, backtrack=backtrack,
            w_wall=w_wall,
            beta_softmin_start=beta_softmin0, beta_softmin_final=beta_softminF,
            eps_abs=eps_abs,
            topk_K=topk_K, topk_tol=topk_tol
        )

        # Local refinement (unchanged)
        X_fin, L_fin = local_optimize_points(
            X_srp, N,
            w_wall=w_wall,
            beta_softmin_final=beta_softminF, eps_abs=eps_abs,
            topk_K=topk_K, topk_tol=topk_tol,
            gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
        )

        pts = X_fin.reshape(N, 2)

        # Metrics (unchanged)
        A_min = float(min_triangle_area(pts, eps_abs))
        mwc   = float(min_wall_clearance_points(pts))
        mpd   = float(min_pair_distance(pts))

        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{A_min:.10f},{mwc:.10f},{mpd:.10f},{L_fin:.8e}\n")

        data[s, 0, :] = pts[:, 0].astype(np.float32)
        data[s, 1, :] = pts[:, 1].astype(np.float32)

        bar.set_postfix(min_area=f"{A_min:.6f}", clr=f"{min(mwc, mpd):.4f}")

    # Sort by min triangle area (descending) — unchanged
    scores = np.zeros(M, dtype=np.float64)
    for s in range(M):
        pts = data[s].T.astype(np.float64)
        scores[s] = min_triangle_area(pts, 1e-12)
    sorted_indices = np.argsort(-scores)
    data = data[sorted_indices]

    # Save dataset
    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved pushed dataset:  {dataset_fn}")
    print(f"Saved metrics:         {metrics_fn}")

    # Optional plots (unchanged)
    if plot_k > 0:
        plot_top_k_minarea_samples(data, plot_k, plot_dir, filename_prefix="heilbronn_mintriangles")
        print(f"Saved plots:   {plot_dir}")

# =============================================================================
# Entrypoint
# =============================================================================

# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    sec  = "heilbronn_SRP"
    mode = _get_cfg(sec, "mode", "training_set_gen").strip().lower()

    if mode == "training_set_gen":
        # Unchanged behavior
        generate_heilbronn_dataset()
    elif mode == "final_push":
        # New behavior: SRP applied to an existing [M,2,N] tensor loaded from disk
        final_push_existing_samples()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'training_set_gen' or 'final_push'.")