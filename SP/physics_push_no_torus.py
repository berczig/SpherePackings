import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import SP
import SP.data_evaluation

def eliminate_overlaps(
    initial_centers,
    radius,
    box_size,
    max_iter=100,
    dt=0.2,
    tol=1e-5,
    visualize=False
):
    """
    Iteratively eliminates overlaps between spheres inside a fixed box [0, box_size]^d.

    Args:
        initial_centers (np.ndarray): Array of shape (n, d) with initial sphere centers.
        radius (float): Radius of each sphere.
        box_size (list or np.ndarray): Size of the box in each dimension (d elements).
        max_iter (int): Maximum number of simulation iterations.
        dt (float): Step size (learning rate) for moving spheres.
        tol (float): Tolerance for maximum overlap to consider converged.
        visualize (bool): If True and d=2, creates an animation.

    Returns:
        np.ndarray: Final sphere centers (shape (n,d)).
        matplotlib.animation.FuncAnimation (optional): Animation if visualize=True and d=2.
    """
    centers = np.array(initial_centers, dtype=float)
    n, d = centers.shape
    box_size = np.array(box_size, dtype=float)

    if box_size.shape[0] != d:
        raise ValueError("Dimension mismatch between centers and box_size.")
    if radius <= 0:
        raise ValueError("Radius must be positive.")

    target_dist_sq = (2 * radius)**2
    history = []
    progress_bar = tqdm(range(max_iter), desc="Simulation Progress")
    converged = False

    for iteration in progress_bar:
        if visualize and d == 2:
            history.append(centers.copy())

        movements = np.zeros_like(centers)
        max_overlap = 0.0
        overlaps_found = False

        # --- Collision detection & resolve forces ---
        for i in range(n):
            for j in range(i + 1, n):
                vec = centers[i] - centers[j]
                dist_sq = np.dot(vec, vec)
                if dist_sq < target_dist_sq and dist_sq > 1e-12:
                    overlaps_found = True
                    dist = np.sqrt(dist_sq)
                    overlap = 2*radius - dist
                    if overlap > max_overlap:
                        max_overlap = overlap

                    # direction and magnitude
                    direction = vec / dist
                    move_mag = ((1 + overlap)**2 - 1) / 2.0
                    movements[i] +=  direction * move_mag
                    movements[j] += -direction * move_mag

        # check convergence
        if not overlaps_found and max_overlap < tol:
            print(f"\nConverged in {iteration+1} iterations.")
            converged = True
            if visualize and d == 2:
                history.append(centers.copy())
            break

        # apply movements
        centers += dt * movements

        # clamp to box boundaries
        centers = np.clip(centers, 0.0, box_size)

        progress_bar.set_postfix({"max_overlap": f"{max_overlap:.2e}"})

        if overlaps_found and max_overlap < tol:
            print(f"\nConverged in {iteration+1} iterations (tolerance reached).")
            converged = True
            if visualize and d == 2:
                history.append(centers.copy())
            break

    if not converged and visualize and d == 2:
        # ensure final state is in history
        if len(history) == 0 or not np.allclose(history[-1], centers):
            history.append(centers.copy())

    # --- Visualization (2D only) ---
    anim = None
    if visualize and d == 2:
        fig, ax = plt.subplots()
        ax.set_xlim(0, box_size[0])
        ax.set_ylim(0, box_size[1])
        ax.set_aspect('equal')
        ax.set_title(f'Sphere De-overlapping in Box (n={n}, r={radius:.2f})')
        circles = [plt.Circle((0,0), radius, alpha=0.6) for _ in range(n)]
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        for c, col in zip(circles, colors):
            c.set_facecolor(col)
            c.set_edgecolor('black')
            ax.add_patch(c)

        def update(frame):
            centers_fr = history[frame]
            for circle, ctr in zip(circles, centers_fr):
                circle.center = ctr
            return circles

        # down-sample frames if too many
        num_frames = len(history)
        max_disp = 200
        if num_frames > max_disp:
            step = int(np.ceil(num_frames/max_disp))
            history[:] = history[::step]
            num_frames = len(history)

        anim = animation.FuncAnimation(
            fig, update, frames=num_frames, interval=50, blit=True, repeat=False
        )
        print("Animation ready; use `display(anim)` or `anim.save(...)` to view/save.")

    return (centers, anim) if visualize and d==2 else centers


def eliminate_overlaps_batched(
    initial_centers,
    radius,
    box_size,
    batched_iterations,
    dt=0.2,
    tol=1e-5,
    visualize=False
):
    """
    Runs eliminate_overlaps in batches and evaluates intermediate results.
    """
    centers = np.array(initial_centers, dtype=float)
    data_evaluations = {}
    data_evaluations[0] = SP.data_evaluation.SphereDatasetEvaluator(
        centers, radius, box_size[0]
    ).evaluate()

    total = 0
    for iters in batched_iterations:
        total += iters
        centers = eliminate_overlaps(
            centers, radius, box_size,
            max_iter=int(iters),
            dt=dt, tol=tol,
            visualize=visualize
        )
        S = SP.data_evaluation.SphereDatasetEvaluator(
            centers, radius, box_size[0]
        )
        data_evaluations[total] = S.evaluate()

    return centers, data_evaluations


def eliminate_overlaps(
    initial_centers,
    radius,
    box_size,
    max_iter=100,
    dt=0.2,
    tol=1e-5,
    visualize=False
):
    centers = np.array(initial_centers, dtype=float)
    n, d = centers.shape
    box_size = np.array(box_size, dtype=float)
    target_dist_sq = (2 * radius)**2
    history = []
    progress = tqdm(range(max_iter), desc="Simulation")

    for it in progress:
        if visualize and d==2:
            history.append(centers.copy())

        # compute all pairwise overlap-resolving moves
        movements = np.zeros_like(centers)
        max_ov = 0.0
        found = False
        for i in range(n):
            for j in range(i+1, n):
                vec = centers[i] - centers[j]
                dist_sq = np.dot(vec, vec)
                if dist_sq < target_dist_sq and dist_sq > 1e-12:
                    found = True
                    dist = np.sqrt(dist_sq)
                    overlap = 2*radius - dist
                    max_ov = max(max_ov, overlap)
                    # direction & magnitude
                    dir_ij = vec / dist
                    mag = ((1+overlap)**2 - 1) / 2.0
                    movements[i] +=  dir_ij * mag
                    movements[j] += -dir_ij * mag

        # check convergence
        if (not found and max_ov<tol) or (found and max_ov<tol):
            print(f"\nConverged in {it+1} iterations.")
            if visualize and d==2:
                history.append(centers.copy())
            break

        # now apply each point’s move, but “stop at boundary” instead of clamping
        for idx in range(n):
            orig = centers[idx]
            mv   = dt * movements[idx]
            tentative = orig + mv

            # if fully inside, accept it
            if np.all(tentative>=0) and np.all(tentative<=box_size):
                centers[idx] = tentative
                continue

            # otherwise, walk along mv’s direction until you hit the box
            norm = np.linalg.norm(mv)
            if norm < 1e-12:
                # no movement
                continue
            direction = mv / norm

            # find t > 0 s.t. orig + t*direction hits any face
            ts = []
            for dim in range(d):
                if direction[dim] > 0:
                    # candidate to hit upper face
                    ts.append((box_size[dim] - orig[dim]) / direction[dim])
                elif direction[dim] < 0:
                    # candidate to hit lower face
                    ts.append(-orig[dim] / direction[dim])
            # pick smallest positive t (can't exceed full mv length)
            t_hit = min([t for t in ts if t>0] + [0.0])
            t_hit = min(t_hit, norm)
            centers[idx] = orig + direction * t_hit

        progress.set_postfix({"max_overlap":f"{max_ov:.2e}"})

    # visualization (2D) — unchanged except that centers history used above
    anim = None
    if visualize and d==2:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(0, box_size[0]); ax.set_ylim(0, box_size[1])
        circles = [plt.Circle((0,0), radius, alpha=0.6) for _ in range(n)]
        colors = plt.cm.viridis(np.linspace(0,1,n))
        for c,col in zip(circles, colors):
            c.set_facecolor(col); c.set_edgecolor('black')
            ax.add_patch(c)

        def update(frame):
            for circle, ctr in zip(circles, history[frame]):
                circle.center = ctr
            return circles

        # down-sample if too many frames
        num_frames = len(history)
        if num_frames > 200:
            step = int(np.ceil(num_frames/200))
            history[:] = history[::step]
            num_frames = len(history)

        anim = animation.FuncAnimation(
            fig, update, frames=num_frames, interval=50, blit=True, repeat=False
        )
        print("Animation ready — use `display(anim)` or `anim.save(...)`.")

    return (centers, anim) if (visualize and d==2) else centers

# --- Example usage ---
if __name__ == "__main__":
    np.random.seed(0)  # for reproducibility
    initial_centers = np.random.uniform(0, 2, size=(100, 2))  # 100 points in 5D
    radius = 0.1
    box_size = [2.1] * 2

    # Run the elimination process with visualization
    final_centers, animation = eliminate_overlaps(
        initial_centers, radius, box_size,
        max_iter=200, dt=0.1, tol=1e-6, visualize=True
    )

    # If you want to save the animation:
    animation.save('sphere_elimination.mp4', writer='ffmpeg')


