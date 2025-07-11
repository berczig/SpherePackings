import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import types

import SP
import SP.data_evaluation

# ----------------------------------------
# 1) TORUS‐WRAPPED OVERLAP‐ELIMINATION
# ----------------------------------------

def shortest_vector_torus(p1, p2, box_size):
    delta = p1 - p2
    box = np.array(box_size, float)
    return delta - box * np.round(delta / box)

# Monkey‐patch so evaluator can call it
SP.physics_push = types.SimpleNamespace(
    shortest_vector_torus=shortest_vector_torus
)

def eliminate_overlaps_torus(
    initial_centers, radius, box_size,
    max_iter=100, dt=0.2, tol=1e-5, visualize=False
):
    centers = np.array(initial_centers, float)
    n, d = centers.shape
    box = np.array(box_size, float)
    target_sq = (2*radius)**2

    history = []
    bar = tqdm(range(max_iter), desc="Torus Simulation")
    for it in bar:
        if visualize and d == 2:
            history.append(centers.copy())

        moves = np.zeros_like(centers)
        max_ov = 0.0
        any_overlap = False

        # compute pairwise forces
        for i in range(n):
            for j in range(i+1, n):
                vec = shortest_vector_torus(centers[i], centers[j], box)
                dist_sq = np.dot(vec, vec)
                if dist_sq < target_sq and dist_sq > 1e-12:
                    any_overlap = True
                    dist = np.sqrt(dist_sq)
                    overlap = 2*radius - dist
                    max_ov = max(max_ov, overlap)
                    direction = vec / dist
                    mag = ((1+overlap)**2 - 1)/2.0
                    moves[i] +=  direction * mag
                    moves[j] += -direction * mag

        # convergence check
        if max_ov < tol:
            print(f"\nTorus: converged in {it+1} iterations.")
            if visualize and d == 2:
                history.append(centers.copy())
            break

        # apply & wrap
        centers += dt * moves
        centers %= box
        bar.set_postfix({"max_overlap": f"{max_ov:.2e}"})

    # build animation if needed
    anim = None
    if visualize and d == 2:
        fig, ax = plt.subplots()
        ax.set_xlim(0, box[0])
        ax.set_ylim(0, box[1])
        ax.set_aspect('equal')
        circles = [
            plt.Circle((0, 0), radius, alpha=0.6)
            for _ in range(n)
        ]
        cols = plt.cm.viridis(np.linspace(0, 1, n))
        for c, col in zip(circles, cols):
            c.set_facecolor(col)
            c.set_edgecolor('black')
            ax.add_patch(c)

        def update(frame):
            for c, ctr in zip(circles, history[frame]):
                c.center = ctr
            return circles

        frames = len(history)
        if frames > 200:
            step = int(np.ceil(frames/200))
            history[:] = history[::step]
            frames = len(history)

        anim = FuncAnimation(
            fig, update, frames=frames,
            interval=50, blit=True, repeat=False
        )
        print("Torus animation ready.")
    return (centers, anim) if (visualize and d==2) else centers


# ----------------------------------------
# 2) BOX‐CONSTRAINED OVERLAP‐ELIMINATION
# ----------------------------------------

def eliminate_overlaps_box(
    initial_centers, radius, box_size,
    max_iter=100, dt=0.2, tol=1e-5,
    boundary_mode="clamp", visualize=False
):
    centers = np.array(initial_centers, float)
    n, d = centers.shape

    full_box = np.array(box_size, float)
    box_min = radius
    box_max = full_box - radius
    target_sq = (2*radius)**2

    history = []
    bar = tqdm(range(max_iter), desc=f"Box({boundary_mode})")
    for it in bar:
        if visualize and d == 2:
            history.append(centers.copy())

        moves = np.zeros_like(centers)
        max_ov = 0.0

        # compute pairwise forces
        for i in range(n):
            for j in range(i+1, n):
                vec = centers[i] - centers[j]
                dist_sq = np.dot(vec, vec)
                if dist_sq < target_sq and dist_sq > 1e-12:
                    dist = np.sqrt(dist_sq)
                    overlap = 2*radius - dist
                    max_ov = max(max_ov, overlap)
                    direction = vec / dist
                    mag = ((1+overlap)**2 - 1)/2.0
                    moves[i] +=  direction * mag
                    moves[j] += -direction * mag

        # convergence check
        if max_ov < tol:
            print(f"\nBox({boundary_mode}): converged in {it+1} iterations.")
            if visualize and d == 2:
                history.append(centers.copy())
            break

        # apply moves + boundary handling
        for idx in range(n):
            mv = dt * moves[idx]
            orig = centers[idx]
            tentative = orig + mv

            if boundary_mode == "clamp":
                # keep center in [r, L-r]
                centers[idx] = np.minimum(
                    np.maximum(tentative, box_min),
                    box_max
                )

            else:  # stophit
                # if fully inside, take it
                if np.all(tentative >= box_min) and np.all(tentative <= box_max):
                    centers[idx] = tentative
                else:
                    length = np.linalg.norm(mv)
                    if length < 1e-12:
                        continue
                    dir_unit = mv / length
                    ts = []
                    for dim in range(d):
                        if dir_unit[dim] > 0:
                            ts.append((box_max[dim] - orig[dim]) / dir_unit[dim])
                        elif dir_unit[dim] < 0:
                            ts.append((box_min - orig[dim]) / dir_unit[dim])
                    # move at most full step
                    t_hit = min([t for t in ts if t>0] + [length])
                    new_pos = orig + dir_unit * t_hit
                    # clamp any tiny overshoot
                    centers[idx] = np.minimum(
                        np.maximum(new_pos, box_min),
                        box_max
                    )

        bar.set_postfix({"max_overlap": f"{max_ov:.2e}"})

    # animation
    anim = None
    if visualize and d == 2:
        fig, ax = plt.subplots()
        ax.set_xlim(0, full_box[0])
        ax.set_ylim(0, full_box[1])
        ax.set_aspect('equal')
        circles = [
            plt.Circle((0, 0), radius, alpha=0.6)
            for _ in range(n)
        ]
        cols = plt.cm.viridis(np.linspace(0, 1, n))
        for c, col in zip(circles, cols):
            c.set_facecolor(col)
            c.set_edgecolor('black')
            ax.add_patch(c)

        def update(frame):
            for c, ctr in zip(circles, history[frame]):
                c.center = ctr
            return circles

        frames = len(history)
        if frames > 500:
            step = int(np.ceil(frames/200))
            history[:] = history[::step]
            frames = len(history)

        anim = FuncAnimation(
            fig, update, frames=frames,
            interval=50, blit=True, repeat=False
        )
        print(f"Box({boundary_mode}) animation ready.")
    # always return (centers, anim); anim will be None if not visualizing 2D
    return centers, anim

# ----------------------------------------
# EXAMPLE USAGE
# ----------------------------------------

def main():
    # Parameters
    n2, dims2, r2 = 51, 2, 0.07104313811
    box2 = [1.0, 1.0]  # 2D box

    # Initialize centers in [r2, box2-r2]^d
    rng = np.random.default_rng(42)
    min_corner = r2
    max_corner = np.array(box2) - r2
    init2 = rng.random((n2, dims2)) * (max_corner - min_corner) + min_corner

    # Torus version
    #final_torus, anim_t = eliminate_overlaps_torus(
    #    init2, r2, box2,
    #    max_iter=300, dt=0.3, tol=1e-4,
    #    visualize=True
    #)
    #anim = anim_t  # keep alive
    #anim.save("./output/push_tests/torus_elim2d.mp4",
    #          writer='ffmpeg', fps=30)
    #print("Torus final centers (first 5):\n", final_torus[:5])

    # Box-clamp version
    final_clamp, anim_c = eliminate_overlaps_box(
        init2, r2, box2,
        max_iter=3000, dt=0.04, tol=1e-8,
        boundary_mode="clamp", visualize=True
    )
    anim = anim_c
    if anim is not None:
        anim.save("./output/push_tests/box_clamp_elim2d_51spheres.mp4",
                  writer='ffmpeg', fps=30)
    print("Box-clamp final centers (first 5):\n", final_clamp[:5])

    # Box-stophit version
    final_hit, anim_h = eliminate_overlaps_box(
        init2, r2, box2,
        max_iter=3000, dt=0.04, tol=1e-8,
        boundary_mode="stophit", visualize=True
    )
    anim = anim_h
    if anim is not None:
        anim.save("./output/push_tests/box_stophit_elim2d_51spheres.mp4",
                  writer='ffmpeg', fps=30)
    print("Box-stophit final centers (first 5):\n", final_hit[:5])

    # Evaluate
    S = SP.data_evaluation.SphereDatasetEvaluator(
        final_hit, r2, box2[0]
    )
    print("Final overlap-free score:", S.evaluate())

if __name__ == "__main__":
    main()
