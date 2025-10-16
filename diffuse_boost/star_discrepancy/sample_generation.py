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
# Smooth star discrepancy loss
# =============================================================================

@njit(cache=True, fastmath=True)
def _sigmoid(z):
    # stable-ish sigmoid for our ranges
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

@njit(cache=True, fastmath=True)
def _smooth_abs(u, eps_abs=1e-12):
    return math.sqrt(u*u + eps_abs)

@njit(cache=True, fastmath=True)
def _l2_norm(v):
    s = 0.0
    for k in range(v.size):
        s += v[k]*v[k]
    return math.sqrt(s) + 1e-18

import numpy as np
import math

def _sigmoid_np(z):
    # stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def _smooth_abs_np(u, eps=1e-12):
    return np.sqrt(u*u + eps)

def critical_grid_from_points(pts, pad_eps=0.0, include_one=True):
    """
    pts: (N,2) in [0,1].
    Build an anchored grid from unique point coordinates.
    Return Ax, Ay (1D arrays) of length ~N+1 each.
    pad_eps>0 optionally adds tiny offsets to avoid ties in the smooth gate.
    """
    x = np.clip(pts[:,0], 0.0, 1.0)
    y = np.clip(pts[:,1], 0.0, 1.0)
    xs = np.unique(x)
    ys = np.unique(y)
    if include_one:
        xs = np.unique(np.concatenate([xs, np.array([1.0])]))
        ys = np.unique(np.concatenate([ys, np.array([1.0])]))
    if pad_eps > 0.0:
        xs = np.clip(xs + pad_eps, 0.0, 1.0)
        ys = np.clip(ys + pad_eps, 0.0, 1.0)
    return xs.astype(np.float64), ys.astype(np.float64)


def star_discrepancy_loss_and_grad(
    X, N, Ax, Ay,
    beta_softmax,
    tau_sigmoid,
    w_wall=1e-8,
    eps_abs=1e-12,
    topk_boxes=0,
):
    """
    Vectorized smooth 2D star discrepancy surrogate with optional Top-K boxes.

    Parameters
    ----------
    X : (2N,) flat array of point coords in [0,1]^2
    N : int, number of points
    Ax, Ay : 1D arrays defining the anchored-box grid along x and y
    beta_softmax : float, softmax sharpness (larger => closer to max)
    tau_sigmoid  : float, smoothing temperature for the indicator gates
    w_wall       : float, small quadratic wall penalty
    eps_abs      : float, epsilon for smooth |.| = sqrt(u^2 + eps)
    topk_boxes   : int, if >0 only the Top-K boxes (largest |delta|) are used
                   in the softmax and gradient (speed optimization)

    Returns
    -------
    L : float
        The loss value (smooth sup over anchored boxes + wall penalties)
    g : (2N,) ndarray
        Gradient w.r.t. X
    """

    # ---- small helpers -------------------------------------------------------
    def _sigmoid_np(z):
        # stable sigmoid
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def _smooth_abs_np(u, eps=1e-12):
        return np.sqrt(u * u + eps)

    # ---- unpack coordinates & init grad -------------------------------------
    pts = X[: 2 * N].reshape(N, 2)
    x = pts[:, 0]  # (N,)
    y = pts[:, 1]  # (N,)
    g = np.zeros_like(X, dtype=np.float64)

    # ---- tiny wall penalty to keep domain honest ----------------------------
    # (kept simple and vectorized)
    v = -x
    mask = v > 0
    if np.any(mask):
        g[0::2][mask] += -2.0 * w_wall * v[mask]
    L = w_wall * float(np.sum(v[mask] ** 2)) if np.any(mask) else 0.0

    v = x - 1.0
    mask = v > 0
    if np.any(mask):
        g[0::2][mask] += 2.0 * w_wall * v[mask]
        L += w_wall * float(np.sum(v[mask] ** 2))

    v = -y
    mask = v > 0
    if np.any(mask):
        g[1::2][mask] += -2.0 * w_wall * v[mask]
        L += w_wall * float(np.sum(v[mask] ** 2))

    v = y - 1.0
    mask = v > 0
    if np.any(mask):
        g[1::2][mask] += 2.0 * w_wall * v[mask]
        L += w_wall * float(np.sum(v[mask] ** 2))

    # ---- vectorized smooth counts over grid ---------------------------------
    U = Ax.size
    V = Ay.size

    # Sx[u,i] = sigmoid((a_u - x_i)/tau), Sy[v,i] = sigmoid((b_v - y_i)/tau)
    Sx = _sigmoid_np((Ax[:, None] - x[None, :]) / tau_sigmoid)  # (U,N)
    Sy = _sigmoid_np((Ay[:, None] - y[None, :]) / tau_sigmoid)  # (V,N)

    # c[u,v] = (1/N) * sum_i Sx[u,i] * Sy[v,i] = (1/N) * (Sx @ Sy^T)
    C = (Sx @ Sy.T) / N  # (U,V)

    # delta and smooth |delta|
    Agrid, Bgrid = np.meshgrid(Ax, Ay, indexing="ij")  # (U,V)
    Delta = C - (Agrid * Bgrid)                        # (U,V)
    Dabs = _smooth_abs_np(Delta, eps_abs)              # (U,V)

    # ---- Top-K selection (optional) -----------------------------------------
    if topk_boxes and topk_boxes < Dabs.size:
        K = int(topk_boxes)

        # Indices of the K largest |delta| using partial sort (argpartition)
        # (O(n) selection; much faster than argsort on the whole array)
        # Docs: numpy.argpartition.  :contentReference[oaicite:1]{index=1}
        flat_idx = np.argpartition(Dabs.ravel(), -K)[-K:]
        u_idx, v_idx = np.unravel_index(flat_idx, Dabs.shape)

        # Gather selected values
        Dsel = Dabs[u_idx, v_idx]        # (K,)
        Delta_sel = Delta[u_idx, v_idx]  # (K,)

        # Numerically stable log-sum-exp over beta * |delta|
        Xlog = beta_softmax * Dsel
        m = float(np.max(Xlog))
        Z = float(np.exp(Xlog - m).sum())
        L += (m + np.log(Z)) / beta_softmax  # LSE trick.  :contentReference[oaicite:2]{index=2}

        # Softmax weights on the K entries
        W = np.exp(Xlog - m) / Z  # (K,)

        # d|delta|/d delta = delta / sqrt(delta^2 + eps)
        dabs_ddelta = Delta_sel / (Dsel + 1e-18)  # (K,)
        Coeff_flat = (W * dabs_ddelta) / N        # (K,)

        # Derivatives of Sx, Sy wrt x_i, y_i
        dSx_dx = -(Sx * (1.0 - Sx)) / tau_sigmoid  # (U,N)
        dSy_dy = -(Sy * (1.0 - Sy)) / tau_sigmoid  # (V,N)

        # Accumulate gradient over selected (u,v)
        # Using a small Python loop over K indices is fine (K ~ 1e3)
        grad_x = np.zeros(N, dtype=np.float64)
        grad_y = np.zeros(N, dtype=np.float64)
        for k, (u, v) in enumerate(zip(u_idx, v_idx)):
            c = Coeff_flat[k]
            grad_x += c * dSx_dx[u, :] * Sy[v, :]   # (N,)
            grad_y += c * dSy_dy[v, :] * Sx[u, :]   # (N,)

        g[0::2] += grad_x
        g[1::2] += grad_y
        return float(L), g

    # ---- Full (all boxes) path ----------------------------------------------
    # Numerically stable log-sum-exp over all U*V boxes
    Xlog = beta_softmax * Dabs
    m = float(np.max(Xlog))
    Z = float(np.exp(Xlog - m).sum())
    L += (m + np.log(Z)) / beta_softmax  # LSE trick.  :contentReference[oaicite:3]{index=3}

    # Softmax weights W[u,v]
    W = np.exp(Xlog - m) / Z  # (U,V)

    # d|delta|/d delta and common coefficient
    Dabs_ddelta = Delta / (Dabs + 1e-18)  # (U,V)
    Coeff = (W * Dabs_ddelta) / N         # (U,V)

    # Derivatives of Sx, Sy wrt x_i, y_i
    dSx_dx = -(Sx * (1.0 - Sx)) / tau_sigmoid  # (U,N)
    dSy_dy = -(Sy * (1.0 - Sy)) / tau_sigmoid  # (V,N)

    # For x: grad_x[i] = sum_u dSx_dx[u,i] * sum_v Coeff[u,v] * Sy[v,i]
    T_x = Coeff @ Sy            # (U,N)
    grad_x = np.sum(dSx_dx * T_x, axis=0)  # (N,)

    # For y: grad_y[i] = sum_v dSy_dy[v,i] * sum_u Coeff[u,v] * Sx[u,i]
    T_y = (Coeff.T @ Sx)        # (V,N)
    grad_y = np.sum(dSy_dy * T_y, axis=0)  # (N,)

    g[0::2] += grad_x
    g[1::2] += grad_y
    return float(L), g


def exact_star_discrepancy_2d(pts):
    """
    Exact 2D L_infinity star discrepancy on the critical grid.
    Checks both open [0,a)×[0,b) and closed [0,a]×[0,b] boxes.

    Parameters
    ----------
    pts : (N,2) ndarray in [0,1]

    Returns
    -------
    D : float
        Exact star discrepancy.
    details : dict
        Extra info: separate maxima for the two variants and
        the maximizing boxes + counts.
    """
    pts = np.asarray(pts, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 2
    N = pts.shape[0]
    if N == 0:
        return 0.0, {"open_max": 0.0, "closed_max": 0.0}

    # clamp to [0,1]
    x = np.clip(pts[:, 0], 0.0, 1.0)
    y = np.clip(pts[:, 1], 0.0, 1.0)

    # Critical coordinates: unique point coords plus boundary 1
    Ax = np.unique(np.concatenate([x, [1.0]]))
    Ay = np.unique(np.concatenate([y, [1.0]]))
    U, V = Ax.size, Ay.size

    # --- OPEN (<,<) variant: counts of x<a, y<b via 2D prefix sum
    iu = np.searchsorted(Ax, x, side="left")
    iv = np.searchsorted(Ay, y, side="left")
    M_open = np.zeros((U, V), dtype=np.int64)
    for i, j in zip(iu, iv):
        # i in [0..U-1], j in [0..V-1]
        M_open[i, j] += 1
    C_open = M_open.cumsum(axis=0).cumsum(axis=1)        # counts(<,<) at (a_u,b_v)
    frac_open = C_open / float(N)

    Agrid, Bgrid = np.meshgrid(Ax, Ay, indexing="ij")
    D_minus = Agrid * Bgrid - frac_open                   # >= 0 at maximizer
    D_minus_max = float(D_minus.max())
    u_minus, v_minus = np.unravel_index(np.argmax(D_minus), D_minus.shape)

    # --- CLOSED (<=,<=) variant: counts of x<=a, y<=b
    iu2 = np.searchsorted(Ax, x, side="right") - 1
    iv2 = np.searchsorted(Ay, y, side="right") - 1
    M_closed = np.zeros((U, V), dtype=np.int64)
    for i, j in zip(iu2, iv2):
        M_closed[i, j] += 1
    C_closed = M_closed.cumsum(axis=0).cumsum(axis=1)     # counts(<=,<=)
    frac_closed = C_closed / float(N)

    D_plus = frac_closed - Agrid * Bgrid                  # >= 0 at maximizer
    D_plus_max = float(D_plus.max())
    u_plus, v_plus = np.unravel_index(np.argmax(D_plus), D_plus.shape)

    D = max(D_minus_max, D_plus_max)
    details = {
        "open_max": D_minus_max,
        "open_box": (float(Ax[u_minus]), float(Ay[v_minus])),     # [0,a)×[0,b)
        "open_count": int(C_open[u_minus, v_minus]),
        "closed_max": D_plus_max,
        "closed_box": (float(Ax[u_plus]), float(Ay[v_plus])),      # [0,a]×[0,b]
        "closed_count": int(C_closed[u_plus, v_plus]),
        "grid_sizes": (int(U), int(V)),
    }
    return D, details


# =============================================================================
# SRP (adaptive) reusing your loop, now calling the star discrepancy loss
# =============================================================================
#@njit(cache=True, fastmath=True)
def srp_adaptive_points_star(
    X, N,
    Imax, m, step_center, beta_sched_decay, backtrack,
    Ax, Ay,
    beta_softmax_start, beta_softmax_final,
    tau_sigmoid,
    w_wall, eps_abs
):
    Xc = X.copy()
    eta = 1.0
    pts = Xc[:2*N].reshape(N,2)
    Ax, Ay = critical_grid_from_points(pts)   # start with critical grid
    refresh_every = max(1, Imax // 10)        # e.g., 10 refreshes total

    for it in range(Imax):
        tfrac = (it + 0.0) / max(Imax - 1, 1)
        beta_t = beta_softmax_start * math.pow(beta_softmax_final / beta_softmax_start, tfrac)

        # (re)build critical grid occasionally to track moving points
        if (it % refresh_every) == 0:
            pts = Xc[:2*N].reshape(N,2)
            Ax, Ay = critical_grid_from_points(pts)

        # SRP noise step as you had it...
        noise = (np.random.rand(Xc.size) * 2.0 - 1.0) * (eta * step_center)
        Xtrial = Xc + noise

        # m gradient steps
        for _ in range(m):
            L0, g = star_discrepancy_loss_and_grad(
                Xtrial, N, Ax, Ay, beta_t, tau_sigmoid, w_wall=w_wall, eps_abs=eps_abs
            )
            step = np.full_like(g, eta * step_center)
            gn = float(np.linalg.norm(g) + 1e-18)
            Xprop = Xtrial - step * (g / gn)

            Lprop, _ = star_discrepancy_loss_and_grad(
                Xprop, N, Ax, Ay, beta_t, tau_sigmoid, w_wall=w_wall, eps_abs=eps_abs
            )

            bt = 0
            while Lprop > L0 and bt < backtrack:
                step *= 0.5
                Xprop = Xtrial - step * (g / gn)
                Lprop, _ = star_discrepancy_loss_and_grad(
                    Xprop, N, Ax, Ay, beta_t, tau_sigmoid, w_wall=w_wall, eps_abs=eps_abs
                )
                bt += 1
            Xtrial = Xprop

        # clip + decay
        Xtrial[0::2] = np.clip(Xtrial[0::2], 0.0, 1.0)
        Xtrial[1::2] = np.clip(Xtrial[1::2], 0.0, 1.0)
        Xc = Xtrial
        eta *= beta_sched_decay

    return Xc

# =============================================================================
# Local optimization (L-BFGS-B) on [0,1]^{2N} for the star discrepancy surrogate
# =============================================================================
def local_optimize_points_star(
    X0, N, Ax, Ay,
    beta_softmax_final, tau_sigmoid,
    w_wall, eps_abs,
    gtol=1e-8, ftol=1e-12, maxiter=500, maxcor=20
):
    bounds = [(0.0, 1.0)] * (2*N)

    def fun(x):
        L, _ = star_discrepancy_loss_and_grad(
            x, N, Ax, Ay, beta_softmax_final, tau_sigmoid, w_wall=w_wall, eps_abs=eps_abs
        )
        return L

    def jac(x):
        _, g = star_discrepancy_loss_and_grad(
            x, N, Ax, Ay, beta_softmax_final, tau_sigmoid, w_wall=w_wall, eps_abs=eps_abs
        )
        return g

    res = minimize(fun=fun, x0=X0, method='L-BFGS-B', jac=jac, bounds=bounds,
                   options={'gtol': gtol, 'ftol': ftol, 'maxiter': maxiter, 'maxcor': maxcor})
    return res.x, res.fun

# =============================================================================
# Metrics: estimate star discrepancy (same smooth surrogate at sharp params)
# =============================================================================
@njit(cache=True, fastmath=True)
def star_disc_surrogate_value(pts, Ax, Ay, beta_softmax, tau_sigmoid, eps_abs=1e-12):
    N = pts.shape[0]
    Z = 0.0
    for ui in range(Ax.size):
        a = Ax[ui]
        for vi in range(Ay.size):
            b = Ay[vi]
            c = 0.0
            for i in range(N):
                sx = _sigmoid((a - pts[i,0]) / tau_sigmoid)
                sy = _sigmoid((b - pts[i,1]) / tau_sigmoid)
                c += sx * sy
            c /= N
            delta = c - a*b
            dabs  = _smooth_abs(delta, eps_abs)
            Z += math.exp(beta_softmax * dabs)
    return (1.0 / beta_softmax) * math.log(Z)

# =============================================================================
# Plotting (unchanged: just points + unit square)
# =============================================================================
def plot_point_sets(data_tensor, k, out_dir, filename_prefix="stardisc"):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    M = data_tensor.shape[0]
    # Compute per-sample exact discrepancy
    min_dis = np.zeros(M, dtype=np.float64)
    for s in range(M):
        pts = data_tensor[s].T.astype(np.float64)  # (N,2)
        A_min, _ = exact_star_discrepancy_2d(pts)
        min_dis[s] = A_min
    k = min(k, M)
    for s in range(k):
        pts = data_tensor[s].T.astype(np.float64)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot([0,1,1,0,0],[0,0,1,1,0])
        ax.scatter(pts[:,0], pts[:,1], s=12)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f"Sample {s}, N={pts.shape[0]}, star disc={min_dis[s]:.6f}")
        out_path = os.path.join(out_dir, f"{filename_prefix}_n={pts.shape[0]}_{s}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# Main generator
# =============================================================================
def generate_star_discrepancy_dataset():
    sec = "star_SRP"

    # Config / defaults
    N   = _get_cfg(sec, "num_points",   50)
    M   = _get_cfg(sec, "num_samples",  2000)

    # SRP hyperparams
    Imax        = _get_cfg(sec, "srp_Imax",        400)
    m           = _get_cfg(sec, "srp_m",           30)
    beta_sched  = _get_cfg(sec, "srp_beta",        0.985)
    backtrack   = _get_cfg(sec, "srp_backtrack",   3)
    step_center = _get_cfg(sec, "srp_step_center", 0.05)

    # Grid for anchored boxes (uniform; can increase for sharper estimates)
    Gx          = _get_cfg(sec, "grid_x",          64)
    Gy          = _get_cfg(sec, "grid_y",          64)
    Ax = np.linspace(1.0/Gx, 1.0, Gx).astype(np.float64)  # (avoid 0 which gives trivial boxes)
    Ay = np.linspace(1.0/Gy, 1.0, Gy).astype(np.float64)

    # Loss / smoothing params
    beta0       = _get_cfg(sec, "beta_softmax_start", 10.0)
    betaF       = _get_cfg(sec, "beta_softmax_final", 200.0)
    tau_sigmoid = _get_cfg(sec, "tau_sigmoid",        0.01)
    w_wall      = _get_cfg(sec, "w_wall",             1e-8)
    eps_abs     = _get_cfg(sec, "abs_eps",            1e-12)

    # Local opt
    gtol        = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol        = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter     = _get_cfg(sec, "lbfgs_maxiter",   500)
    maxcor      = _get_cfg(sec, "lbfgs_maxcor",    20)

    # I/O + plotting
    out_dir     = _get_cfg(sec, "output_dir",      "./outputs_star")
    plot_k      = _get_cfg(sec, "plot_k_dis",          1)
    plot_dir    = os.path.join(out_dir, "plots")

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"star_srp_{M}x{N}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"star_metrics_{M}x{N}_{stamp}.csv")

    print(f"Generating {M} point sets for star discrepancy (N={N})")
    with open(metrics_fn, "w") as mf:
        mf.write("sample,exact_discr,star_surr,beta_eval,tau_eval,loss_after\n")

    data = np.zeros((M, 2, N), dtype=np.float32)

    bar = tqdm(range(M), desc="Generating point sets")
    for s in bar:
        # Random start
        P0 = sample_uniform_points(N)
        X0 = P0.ravel()

        # SRP exploration
        X_srp = srp_adaptive_points_star(
            X0, N,
            Imax=Imax, m=m,
            step_center=step_center,
            beta_sched_decay=beta_sched, backtrack=backtrack,
            Ax=Ax, Ay=Ay,
            beta_softmax_start=beta0, beta_softmax_final=betaF,
            tau_sigmoid=tau_sigmoid,
            w_wall=w_wall, eps_abs=eps_abs
        )

        # Local refine
        X_fin, L_fin = local_optimize_points_star(
            X_srp, N, Ax, Ay,
            beta_softmax_final=betaF, tau_sigmoid=tau_sigmoid,
            w_wall=w_wall, eps_abs=eps_abs,
            gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
        )

        pts = X_fin.reshape(N, 2)

        # Metric: evaluate surrogate at sharp-ish params (you can re-use betaF)
        beta_eval = float(betaF)
        tau_eval  = float(tau_sigmoid)
        star_val  = float(star_disc_surrogate_value(pts, Ax, Ay, beta_eval, tau_eval, eps_abs))

        D_exact, info = exact_star_discrepancy_2d(pts)  # pts shape (N,2)
        print(f"Exact star discrepancy: {D_exact:.6f}  |  open={info['open_max']:.6f}, closed={info['closed_max']:.6f}")


        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{D_exact:.10f},{star_val:.10f},{beta_eval:.1f},{tau_eval:.5f},{L_fin:.8e}\n")

        data[s, 0, :] = pts[:,0].astype(np.float32)
        data[s, 1, :] = pts[:,1].astype(np.float32)

        bar.set_postfix(star=f"{star_val:.5f}")

    # Sort samples by (ascending) D_exact
    metrics = np.loadtxt(metrics_fn, delimiter=",", skiprows=1)
    sorted_idx = np.argsort(metrics[:,1])
    data = data[sorted_idx] 

    # Save dataset
    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved dataset:  {dataset_fn}")
    print(f"Saved metrics:  {metrics_fn}")

    # Optional plots
    if plot_k > 0:
        plot_point_sets(data, plot_k, plot_dir, filename_prefix="star_srp")
        print(f"Saved plots:   {plot_dir}")

# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    generate_star_discrepancy_dataset()
