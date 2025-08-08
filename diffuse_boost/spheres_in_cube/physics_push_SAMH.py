import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from numba import jit
from numba import objmode, types
import itertools

import diffuse_boost.spheres_in_Rd
import diffuse_boost.spheres_in_Rd.data_evaluation
import diffuse_boost.spheres_in_Rd.data_evaluation as data_evaluation
import diffuse_boost



def eliminate_overlaps_box_metropolis_sa(
    initial_centers, radius, box_size,
    stage1_iter=3000,      
    stage2_iter=7000,
    hardness_soft=5.0,
    hardness_hard=500.0,
    T_initial=0.1,
    pressure=0.01,
    max_displacement=0.5, #around the radius size
    visualize=False,
    show_progress=False,
    propose_move=None
):
    """
    Two-stage simulated annealing process
    Stage 1: Uses low 'hardness' to resolve major overlaps. (balls can go trhough each other)
    Stage 2: Uses high 'hardness' to refine the packing to zero overlap.
    """
    centers = np.array(initial_centers, float)
    n, d = centers.shape
    
    if d not in [2, 3]:
        raise ValueError(f"Code is JIT-optimized for d=2 or d=3. You have d={d}.")
        
    @jit(nopython=True, cache=True)
    def _arr_to_tuple_jit(arr):
        if d == 2: return (arr[0], arr[1])
        elif d == 3: return (arr[0], arr[1], arr[2])
        return ()
    @jit(nopython=True, cache=True, fastmath=True)
    def _calculate_particle_energy_jit(current_centers, idx, grid, cell_dim, neighbor_offsets, radius, target_sq, pressure, box_center, hardness):
        energy = 0.0
        center_i = current_centers[idx]
        if pressure > 0:
            vec_to_center = box_center - center_i
            energy += 0.5 * pressure * np.dot(vec_to_center, vec_to_center)
        cell_coords_arr = (center_i / cell_dim).astype(np.int64)
        for i in range(neighbor_offsets.shape[0]):
            offset = neighbor_offsets[i]
            neighbor_cell_coords_arr = cell_coords_arr + offset
            neighbor_cell_coords = _arr_to_tuple_jit(neighbor_cell_coords_arr)
            if neighbor_cell_coords in grid:
                for j in grid[neighbor_cell_coords]:
                    if idx == j: continue
                    vec = center_i - current_centers[j]
                    dist_sq = np.dot(vec, vec)
                    if dist_sq < target_sq:
                        overlap = 2 * radius - np.sqrt(dist_sq)
                        energy += 0.5 * hardness * overlap**2
        return energy

    @jit(nopython=True, cache=True, fastmath=True)
    def _repulsive_dir_jit(current_centers, idx, grid, cell_dim, neighbor_offsets, radius, target_sq):
        center_i = current_centers[idx]
        cell_coords_arr = (center_i / cell_dim).astype(np.int64)
        dir_acc = np.zeros(center_i.shape)
        for oi in range(neighbor_offsets.shape[0]):
            neighbor_cell_coords = _arr_to_tuple_jit(cell_coords_arr + neighbor_offsets[oi])
            if neighbor_cell_coords in grid:
                for j in grid[neighbor_cell_coords]:
                    if idx == j: continue
                    vec = center_i - current_centers[j]
                    dist_sq = np.dot(vec, vec)
                    if dist_sq < target_sq and dist_sq > 1e-18:
                        dist = np.sqrt(dist_sq)
                        overlap = 2*radius - dist
                        dir_acc += (overlap / dist) * vec  # push away
        # normalize
        nrm = np.sqrt((dir_acc*dir_acc).sum())
        return dir_acc / (nrm + 1e-12)


    def _run_sweep_py(current_centers, grid, T, move_dist, neighbor_offsets,
                  cell_dim, radius, target_sq, pressure_eff, box_center,
                  hardness, box_min, box_max, propose_move):
        n, d = current_centers.shape
        accepted_moves_in_sweep = 0
        for _ in range(n):
            idx = np.random.randint(n)
            old_pos = current_centers[idx].copy()
            E_old_particle = _calculate_particle_energy_jit(current_centers, idx, grid, cell_dim, 
                                                            neighbor_offsets, radius, target_sq, 
                                                            pressure_eff, box_center, hardness)    
            step = propose_move(idx, current_centers, cell_dim, radius, T)
            # limit step length
            norm = np.linalg.norm(step)
            if norm > move_dist:
                step *= move_dist / (norm + 1e-12)
    
            new_pos = old_pos + step
            for dim_idx in range(d):
                if new_pos[dim_idx] < box_min[dim_idx]:
                    new_pos[dim_idx] = 2*box_min[dim_idx] - new_pos[dim_idx]
                elif new_pos[dim_idx] > box_max[dim_idx]:
                    new_pos[dim_idx] = 2*box_max[dim_idx] - new_pos[dim_idx]
            #valid_move = True
            #for dim_idx in range(d):
            #    if not (box_min[dim_idx] <= new_pos[dim_idx] <= box_max[dim_idx]):
            #        valid_move = False
            #        break
            #if not valid_move: continue
            current_centers[idx] = new_pos
            old_cell_coords = _arr_to_tuple_jit((old_pos / cell_dim).astype(np.int64))
            new_cell_coords = _arr_to_tuple_jit((new_pos / cell_dim).astype(np.int64))
            if new_cell_coords != old_cell_coords:
                grid[old_cell_coords].remove(idx)
                if new_cell_coords not in grid:
                    grid[new_cell_coords] = List.empty_list(types.int64)
                grid[new_cell_coords].append(idx)
            E_new_particle = _calculate_particle_energy_jit(current_centers, idx, grid, cell_dim, neighbor_offsets, radius, target_sq, pressure_eff, box_center, hardness)
            delta_E = E_new_particle - E_old_particle
            if delta_E < 0 or (T > 1e-9 and np.random.rand() < np.exp(-delta_E / T)):
                accepted_moves_in_sweep += 1
            else: 
                current_centers[idx] = old_pos
                if new_cell_coords != old_cell_coords:
                    grid[new_cell_coords].remove(idx)
                    grid[old_cell_coords].append(idx)
        return accepted_moves_in_sweep
            
    @jit(nopython=True, cache=True)
    def _run_sweep_jit(current_centers, grid, T, move_dist, neighbor_offsets, cell_dim, radius, target_sq, pressure_eff, box_center, hardness, box_min, box_max, propose_move):
        accepted_moves_in_sweep = 0
        for _ in range(n):
            idx = np.random.randint(n)
            old_pos = current_centers[idx].copy()
            E_old_particle = _calculate_particle_energy_jit(current_centers, idx, grid, cell_dim, neighbor_offsets, 
                                                            radius, target_sq, pressure_eff, box_center, hardness)
            rep = _repulsive_dir_jit(current_centers, idx, grid, cell_dim, neighbor_offsets, radius, target_sq)
            # normal
            #rand_vec = np.random.random(d) - 0.5
            #new_pos = old_pos + rand_vec * 2 * move_dist
            # isotropic step
            rand_vec = np.random.normal(0.0, 1.0, d)
            norm = np.sqrt((rand_vec*rand_vec).sum())
            rand_vec = rand_vec / (norm + 1e-12)
            prop_dir = 0.7*rep + 0.3*rand_vec
            nrm = np.sqrt((prop_dir*prop_dir).sum())
            prop_dir /= (nrm + 1e-12)

            new_pos = old_pos + rand_vec * move_dist
            #propose_move? with neural SA
            valid_move = True
            for dim_idx in range(d):
                if not (box_min[dim_idx] <= new_pos[dim_idx] <= box_max[dim_idx]):
                    valid_move = False
                    break
            if not valid_move: continue
            current_centers[idx] = new_pos
            old_cell_coords = _arr_to_tuple_jit((old_pos / cell_dim).astype(np.int64))
            new_cell_coords = _arr_to_tuple_jit((new_pos / cell_dim).astype(np.int64))
            if new_cell_coords != old_cell_coords:
                grid[old_cell_coords].remove(idx)
                if new_cell_coords not in grid:
                    grid[new_cell_coords] = List.empty_list(types.int64)
                grid[new_cell_coords].append(idx)
            E_new_particle = _calculate_particle_energy_jit(current_centers, idx, grid, cell_dim, neighbor_offsets, radius, target_sq, pressure_eff, box_center, hardness)
            delta_E = E_new_particle - E_old_particle
            if delta_E < 0 or (T > 1e-9 and np.random.rand() < np.exp(-delta_E / T)):
                accepted_moves_in_sweep += 1
            else: 
                current_centers[idx] = old_pos
                if new_cell_coords != old_cell_coords:
                    grid[new_cell_coords].remove(idx)
                    grid[old_cell_coords].append(idx)
        return accepted_moves_in_sweep
    @jit(nopython=True, cache=True)
    def _calculate_max_overlap_jit(current_centers, grid, cell_dim, neighbor_offsets, radius, target_sq):
        current_max_ov = 0.0
        for i in range(n):
            center_i = current_centers[i]
            cell_coords_arr = (center_i / cell_dim).astype(np.int64)
            for offset_idx in range(neighbor_offsets.shape[0]):
                offset = neighbor_offsets[offset_idx]
                neighbor_cell_coords_arr = cell_coords_arr + offset
                neighbor_cell_coords = _arr_to_tuple_jit(neighbor_cell_coords_arr)
                if neighbor_cell_coords in grid:
                    for j in grid[neighbor_cell_coords]:
                        if i >= j: continue
                        vec = center_i - current_centers[j]
                        dist_sq = np.dot(vec, vec)
                        if dist_sq < target_sq:
                            overlap = 2 * radius - np.sqrt(dist_sq)
                            if overlap > current_max_ov:
                                current_max_ov = overlap
        return current_max_ov


    int_tuple_type = types.UniTuple(types.int64, d)
    list_type = types.ListType(types.int64)
    full_box = np.array(box_size, float)
    box_center = full_box / 2.0
    box_min = np.full(d, radius)
    box_max = full_box - radius
    target_sq = (2 * radius)**2
    initial_move_dist = radius * max_displacement
    history = []
    cell_dim = 2 * radius * (1 + 1e-9)
    neighbor_offsets = np.array(list(itertools.product(range(-1, 2), repeat=d)), dtype=np.int64)
    
    T = float(T_initial)
    T_start = T
    T_final = 1e-5
    
    #Grid Initialization
    grid_jit = Dict.empty(key_type=int_tuple_type, value_type=list_type)
    for i in range(n):
        cell_coords_arr = (centers[i] / cell_dim).astype(np.int64)
        cell_coords_tuple = tuple(cell_coords_arr)
        if cell_coords_tuple not in grid_jit:
            grid_jit[cell_coords_tuple] = List.empty_list(types.int64)
        grid_jit[cell_coords_tuple].append(i)

    # Main Loop SA Logic 
    total_iter = stage1_iter + stage2_iter
    #the cooling factor alpha for an exponential schedule
    if total_iter > 0 and T_start > T_final:
        alpha = (T_final / T_start)**(1.0 / total_iter)
    else:
        alpha = 1.0  # No cooling if no iterations are planned

    if show_progress:
        bar = tqdm(range(total_iter), desc=f"SA Staged Annealing")
    else:
        bar = tqdm(range(total_iter), desc="SA Staged Annealing", leave=False)
    
    accepted_moves = 0
    pressure_eff = pressure
    max_ov = 0.0
    
    # Combined loop for both stages
    for it in bar:
        # Determine current stage and set hardness
        if it < stage1_iter:
            hardness = hardness_soft
            stage_desc = f"Stage 1 (Soft): h={hardness_soft}"
        else:
            hardness = hardness_hard
            stage_desc = f"Stage 2 (Hard): h={hardness_hard}"
            
        # Cooling schedule applied across both stages continuously
        # The temperature drop is very slow, which is good.
        #T = T_start * (1 - it / total_iter) # I am using a linear cooling schedule (can be improved)
        T = T_start * (alpha**it)  # Exponential cooling schedule
        if T < 1e-6: T = 1e-6

        #move_dist = initial_move_dist * (T / T_start)
        move_dist = max(radius*1e-3, initial_move_dist * (T/T_start))
        
        sweep_fn = _run_sweep_jit if propose_move is None else _run_sweep_py
        accepted_in_sweep = sweep_fn(
            centers, grid_jit, T, move_dist, neighbor_offsets, cell_dim,
            radius, target_sq, pressure_eff, box_center, hardness, box_min, box_max, propose_move
        )
        accepted_moves += accepted_in_sweep
        acceptance_rate = accepted_moves / ((it + 1) * n)

        if it % 200 == 0:
            max_ov = _calculate_max_overlap_jit(centers, grid_jit, cell_dim, neighbor_offsets, radius, target_sq)
            if visualize:
                history.append(centers.copy())
        
        bar.set_description(stage_desc)
        bar.set_postfix({"T": f"{T:.3f}", "accept": f"{acceptance_rate:.2f}", "max_ov": f"{max_ov:.4f}"})
        
        # Convergence check is very important in the second stage
        if it > stage1_iter and max_ov < 1e-8:
            print(f"\nSystem converged to zero overlap at iteration {it}.")
            break
        
        if T < 1e-6 and move_dist < radius * 1e-7:
            print(f"\nSystem frozen at iteration {it}.")
            break
            
    # Final check on overlap
    final_ov = _calculate_max_overlap_jit(centers, grid_jit, cell_dim, neighbor_offsets, radius, target_sq)
    print(f"Final maximum overlap: {final_ov:.2e}")


    #viz stuff
    anim = None
    if visualize and len(history) > 0:
        if d == 2:
            fig, ax = plt.subplots()
            ax.set_xlim(0, full_box[0]); ax.set_ylim(0, full_box[1]); ax.set_aspect('equal')
            circles = [plt.Circle((0, 0), radius, alpha=0.6) for _ in range(n)]
            cols = plt.cm.viridis(np.linspace(0, 1, n))
            for c, col in zip(circles, cols):
                c.set_facecolor(col); c.set_edgecolor('black'); ax.add_patch(c)
            def update(frame):
                for c, ctr in zip(circles, history[frame]):
                    c.center = ctr
                return circles
            anim = FuncAnimation(fig, update, frames=len(history), interval=50, blit=True, repeat=False)
            print(f"\nMetropolis SA 2D animation ready.")
        elif d == 3:
            # Add this import at the top of your file: from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(0, full_box[0]); ax.set_ylim(0, full_box[1]); ax.set_zlim(0, full_box[2])
            ax.set_box_aspect((full_box[0], full_box[1], full_box[2])) # Aspect ratio

            # Scatter plot is more efficient for 3D animation than drawing surfaces
            cols = plt.cm.viridis(np.linspace(0, 1, n))
            # The 's' parameter for scatter is area, so we use radius^2
            scatter_plot = ax.scatter(history[0][:,0], history[0][:,1], history[0][:,2], c=cols, s=(np.pi * (radius*72)**2), alpha=0.6)
            
            def update(frame):
                scatter_plot._offsets3d = (history[frame][:,0], history[frame][:,1], history[frame][:,2])
                return scatter_plot,
            anim = FuncAnimation(fig, update, frames=len(history), interval=50, blit=False, repeat=False)
            print(f"\nMetropolis SA 3D animation ready.")
        else:
            print(f"\nVisualization is only supported for d=2 or d=3. Cannot generate animation for d={d}.")

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


    # Simulated Annealing with Metropolis-Hastings version
    final_annealed, anim_c = eliminate_overlaps_box_metropolis_sa(
        init2, r2, box2,
        stage1_iter=2000,
        stage2_iter=10000,
        hardness_soft=5.0,
        hardness_hard=2000.0,
        T_initial=0.1,
        pressure=0.001,
        max_displacement=0.05,
        visualize=True,         
    )

    anim = anim_c
    if anim is not None:
        anim.save("./output/push_tests/box_clamp_elim2d_51spheres.mp4",
                  writer='ffmpeg', fps=30)
    print("Box-clamp final centers (first 5):\n", final_annealed[:5])


    # Evaluate
    S = diffuse_boost.spheres_in_Rd.data_evaluation.SphereDatasetEvaluator(
        final_annealed, r2, box2[0]
    )
    print("Final overlap-free score:", S.evaluate())

if __name__ == "__main__":
    main()
