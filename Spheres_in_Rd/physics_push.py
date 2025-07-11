import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import SP
import SP.data_evaluation

def shortest_vector_torus(p1, p2, box_size):
    """
    Calculates the shortest vector from p2 to p1 on a torus.
    Handles multiple dimensions.
    """
    delta = p1 - p2
    # Ensure box_size(total width from left to right) is a numpy array for broadcasting
    box_size_arr = np.array(box_size)
    # Correct wrapping calculation
    wrapped_delta = delta - box_size_arr * np.round(delta / box_size_arr)
    return wrapped_delta


def eliminate_overlaps(initial_centers, radius, box_size,
                       max_iter=100, dt=0.2, tol=1e-5, visualize=False):
    """
    Iteratively eliminates overlaps between spheres on a torus.

    Args:
        initial_centers (np.ndarray): Array of shape (n, d) with initial sphere centers.
        radius (float): Radius of each sphere.
        box_size (list or np.ndarray): Size of the torus box in each dimension (d elements).
        max_iter (int): Maximum number of simulation iterations.
        dt (float): Step size (learning rate) for moving spheres. Controls stability.
        tol (float): Tolerance for maximum overlap to consider the simulation converged.
        visualize (bool): If True and d=2, creates an animation.

    Returns:
        np.ndarray: Array of shape (n, d) with final sphere centers.
        matplotlib.animation.FuncAnimation (optional): Animation object if visualize=True and d=2.
    """
    centers = np.array(initial_centers, dtype=float)
    n, d = centers.shape
    box_size = np.array(box_size, dtype=float)

    if len(box_size) != d:
        raise ValueError("Dimension mismatch between centers and box_size.")
    if radius <= 0:
        raise ValueError("Radius must be positive.")

    target_dist_sq = (2 * radius) ** 2
    history = [] # Store history for visualization

    #print(f"Starting simulation: {n} spheres, d={d}, radius={radius}, box={box_size}")
    #print(f"Parameters: max_iter={max_iter}, dt={dt}, tol={tol}")

    progress_bar = tqdm(range(max_iter), desc="Simulation Progress")
    converged = False

    for iteration in progress_bar:
        if visualize and d == 2:
            history.append(centers.copy())

        movements = np.zeros_like(centers)
        max_overlap_this_iter = 0.0
        overlaps_found = False

        # --- Collision Detection and Movement Calculation ---
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate shortest vector and squared distance on torus
                vec_ij = shortest_vector_torus(centers[i], centers[j], box_size)
                dist_sq = np.sum(vec_ij**2)

                # Check for overlap (avoiding sqrt for efficiency)
                # Add small epsilon to avoid division by zero if centers are identical
                if dist_sq < target_dist_sq and dist_sq > 1e-12:
                    overlaps_found = True
                    dist = np.sqrt(dist_sq)
                    overlap = (2 * radius) - dist
                    max_overlap_this_iter = max(max_overlap_this_iter, overlap)

                    # Calculate movement direction (unit vector) and magnitude
                    # Move each sphere by overlap / 2 along the connecting vector
                    move_magnitude = ((1+overlap)**2 - 1) / 2.0
                    #move_magnitude = overlap / 2.0
                    direction = vec_ij / dist

                    # Accumulate movements (apply dt later)
                    movements[i] += direction * move_magnitude
                    movements[j] -= direction * move_magnitude

        # --- Apply Movements and Boundary Conditions ---
        if not overlaps_found:
             # If no overlaps found in this iteration, check against tolerance
            if max_overlap_this_iter < tol:
                print(f"\nConverged in {iteration+1} iterations (no overlaps found).")
                converged = True
                # Add final state to history for visualization
                if visualize and d == 2:
                     history.append(centers.copy())
                break
            # If no overlaps found, but max_overlap is still slightly > tol from previous steps
            # This shouldn't happen often with the current logic, but as a fallback:
            # centers += movements * dt # Apply any residual tiny movements? No, better stop.

        # Apply scaled movements
        centers += movements * dt

        # Apply periodic boundary conditions (wrap coordinates)
        centers %= box_size # Modulo operator handles wrapping efficiently

        progress_bar.set_postfix({"Max Overlap": f"{max_overlap_this_iter:.2e}"})

        # --- Check Convergence (based on max overlap this iteration) ---
        if max_overlap_this_iter < tol and overlaps_found: # Need overlaps_found check in case dt is too small
             print(f"\nConverged in {iteration+1} iterations (max overlap < tolerance).")
             converged = True
             # Add final state to history for visualization
             if visualize and d == 2:
                  history.append(centers.copy())
             break

    if not converged:
        #print(f"\nReached maximum iterations ({max_iter}) without full convergence.")
        #print(f"Final maximum overlap: {max_overlap_this_iter:.2e}")
        # Add final state to history for visualization
        if visualize and d == 2 and len(history) == 0: # Ensure final state is added if loop finished
             history.append(centers.copy())
        elif visualize and d == 2 and len(history) > 0 and not np.array_equal(history[-1], centers):
             history.append(centers.copy())


    # --- Visualization (if requested and d=2) ---
    anim = None
    if visualize and d == 2:
        print("Creating animation...")
        fig, ax = plt.subplots()
        ax.set_xlim(0, box_size[0])
        ax.set_ylim(0, box_size[1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Sphere De-overlapping (n={n}, r={radius:.2f})')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Create patch collection for efficiency
        circles = [plt.Circle((0, 0), radius, fill=True, alpha=0.6) for _ in range(n)]
        # Use different colors for visual clarity
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        for circle, color in zip(circles, colors):
            ax.add_patch(circle)
            circle.set_facecolor(color)
            circle.set_edgecolor('black')

        # Trajectory lines (optional, can be messy)
        # lines = [ax.plot([], [], '-', lw=0.5, color=colors[i], alpha=0.5)[0] for i in range(n)]

        def update(frame):
            current_centers = history[frame]
            ax.set_title(f'Iteration {frame * (frame_interval if frame_interval else 1)}') # Show approximate iteration
            for i, circle in enumerate(circles):
                circle.center = current_centers[i]
            # Update trajectories (optional)
            # for i, line in enumerate(lines):
            #     traj_x = [h[i, 0] for h in history[:frame+1]]
            #     traj_y = [h[i, 1] for h in history[:frame+1]]
            #     line.set_data(traj_x, traj_y)
            # Need to return the modified artists
            return circles # + lines

        # Adjust interval and frame skipping for smoother/faster animation
        num_frames = len(history)
        max_display_frames = 200 # Limit frames in animation object for performance
        frame_interval = 1
        if num_frames > max_display_frames:
            frame_interval = max(1, num_frames // max_display_frames)
            history_sampled = history[::frame_interval]
            num_frames = len(history_sampled)
            print(f"Sampling history for animation (displaying {num_frames} frames).")
            history = history_sampled # Use the sampled history for animation


        interval_ms = 50 # milliseconds per frame
        anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                       interval=interval_ms, blit=True, repeat=False)
        #plt.close(fig) # Prevent duplicate static plot display in notebooks
        print("Animation created. Call `display(anim)` or `anim.save()`.")

    elif visualize and d != 2:
        print("Visualization is only supported for d=2.")

    if visualize:
        return centers, anim    
    return centers

def eliminate_overlaps_batched(initial_centers, radius, box_size,
                       batched_iterations, dt=0.2, tol=1e-5, visualize=False):
    # Generate random batched_iterations points
    centers = initial_centers
    data_evaluations = {}
    data_evaluations[0] = SP.data_evaluation.SphereDatasetEvaluator(initial_centers, radius, box_size[0]).evaluate()
    it = 0
    print("running simulations using this iteration schedule: ", batched_iterations)
    for iterations in batched_iterations:
        it += iterations
        centers = eliminate_overlaps(centers, radius, box_size, int(iterations), dt, tol, visualize)

        # Evaluate resulting dataset
        S = SP.data_evaluation.SphereDatasetEvaluator(centers, radius, box_size[0])
        data_evaluations[it] = S.evaluate()

    return centers, data_evaluations

# --- Example Usage ---

# --- 2D Example ---
def main():
    print("--- Running 2D Example ---")
    n_points_2d = 29
    dims_2d = 2
    radius_2d = 1
    box_2d = [10.0, 10.0]

    # Generate random starting points
    #np.random.seed(42)
    initial_centers_2d = np.random.rand(n_points_2d, dims_2d) * box_2d

    # Run the simulation with visualization
    final_centers_2d, anim_2d = eliminate_overlaps(
        initial_centers_2d,
        radius_2d,
        box_2d,
        max_iter=300,
        dt=0.3, # May need tuning
        tol=1e-4,
        visualize=True
    )

    print("\nInitial Centers (2D Head):\n", initial_centers_2d[:5])
    print("Final Centers (2D Head):\n", final_centers_2d[:5])
    S = SP.data_evaluation.SphereDatasetEvaluator(final_centers_2d, radius_2d, box_2d[0])
    print("evaluate: ", S.evaluate())

    # To display the animation in a Jupyter Notebook:
    # from IPython.display import display, HTML
    # display(HTML(anim_2d.to_jshtml()))
    # Or save it:
    # anim_2d.save('sphere_overlap_2d.gif', writer='pillow', fps=15)
    anim_2d.save('sphere_overlap_2d.mp4', writer='ffmpeg', fps=15)

    # If running outside Jupyter, you might need to show it directly (less common for animations)
    plt.show() # This usually shows the static plot after FuncAnimation

    # --- 3D Example ---
    print("\n--- Running 3D Example ---")
    n_points_3d = 90
    dims_3d = 5
    radius_3d = 1
    box_3d = 4*np.ones(dims_3d)

    # Generate random starting points
    initial_centers_3d = np.random.rand(n_points_3d, dims_3d) * box_3d

    # Run the simulation (no visualization for 3D)
    final_centers_3d = eliminate_overlaps(
        initial_centers_3d,
        radius_3d,
        box_3d,
        max_iter=100,
        dt=0.2,
        tol=1e-4,
        visualize=False # Set to False or it will just print a message
    )

    print("\nInitial Centers (3D Head):\n", initial_centers_3d[:5])
    print("Final Centers (3D Head):\n", final_centers_3d[:5])

    S = SP.data_evaluation.SphereDatasetEvaluator(final_centers_3d, radius_3d, box_3d[0])
    print("evaluate 3D: ", S.evaluate())

    # Verify final overlaps (optional check)
    final_overlaps = 0
    min_dist_sq = float('inf')
    target_dist_sq_3d = (2 * radius_3d) ** 2
    for i in range(n_points_3d):
        for j in range(i + 1, n_points_3d):
            vec = shortest_vector_torus(final_centers_3d[i], final_centers_3d[j], box_3d)
            d_sq = np.sum(vec**2)
            min_dist_sq = min(min_dist_sq, d_sq)
            if d_sq < target_dist_sq_3d - 1e-6: # Allow slightly smaller due to tolerance
                final_overlaps += 1
                # print(f"Overlap detected between {i} and {j}, dist_sq={d_sq:.4f}")

    print(f"\nVerification (3D): Found {final_overlaps} overlaps after simulation.")
    print(f"Minimum squared distance: {min_dist_sq:.4f} (Target non-overlap > {target_dist_sq_3d:.4f})")