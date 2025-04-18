import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dense_to_sparse, add_self_loops
import matplotlib.pyplot as plt
import numpy as np
import time
import math

# --- Configuration ---
DIM = 2  # Dimension of the space (e.g., 2 for 2D, 3 for 3D)
NUM_SPHERES = 20
RADIUS = 1.0
INTERACTION_THRESHOLD_MULTIPLIER = 3.0 # Multiplier for radius to build graph edges
# New: Toroidal space configuration
BOX_SIZE = 30 # Adjust multiplier as needed
# New: Stability configuration
MAX_DISPLACEMENT_PER_STEP = RADIUS/3 # Max distance a sphere can move in one step
GRADIENT_CLIP_NORM = 0.02 # Max norm for gradients

# --- Helper Functions ---

def generate_spheres(num_spheres, dim, radius, box_size):
    """Generates sphere positions within the box [0, box_size)."""
    # Ensure initial positions are within the box
    positions = torch.rand(num_spheres, dim) * box_size
    # Small check for initial overlap (can be made more robust if needed)
    # overlap_count = count_overlaps(positions, radius, box_size) # Need count_overlaps defined first
    # print(f"Generated initial state with estimated {overlap_count} overlaps.")
    return positions

def pairwise_toroidal_distance(pos1, pos2, box_size):
    """Calculates the pairwise toroidal distance matrix between two sets of points."""
    delta = torch.abs(pos1.unsqueeze(1) - pos2.unsqueeze(0)) # Diffs for all pairs [N, M, dim]
    # For each dimension, consider the wrap-around distance
    delta = torch.min(delta, box_size - delta) # Shortest distance along each axis
    # Calculate Euclidean distance based on the shortest axis distances
    dist_sq = torch.sum(delta**2, dim=-1)
    return torch.sqrt(dist_sq) # [N, M] distance matrix

def create_graph_toroidal(positions, threshold, box_size):
    """
    Creates graph connectivity (edge_index) based on TOROIDAL distance threshold.
    """
    num_nodes = positions.shape[0]
    # Calculate pairwise TOROIDAL distances efficiently
    dist_matrix = pairwise_toroidal_distance(positions, positions, box_size)
    # Create adjacency matrix based on threshold
    adj = (dist_matrix > 0) & (dist_matrix < threshold) # Exclude self (dist=0) explicitly
    # Convert dense adjacency to sparse edge_index format
    edge_index = dense_to_sparse(adj)[0]
    # Add self-loops (important for GNNs)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index

def calculate_overlap_loss_toroidal(positions, radius, box_size):
    """
    Calculates overlap loss based on TOROIDAL distances. (Corrected version)
    """
    num_nodes = positions.shape[0]
    if num_nodes < 2:
        return torch.tensor(0.0, device=positions.device, requires_grad=True) # No pairs, no loss

    dist_matrix = pairwise_toroidal_distance(positions, positions, box_size)

    # Calculate overlap amount: max(0, 2*radius - distance)
    epsilon = 1e-7 # Slightly increased epsilon for stability
    overlap = torch.relu(2 * radius - dist_matrix + epsilon)

    # Create a mask that is True for off-diagonal elements (ignore self-distance)
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=positions.device)

    # Apply the mask: compute sum of squared overlaps only over off-diagonal elements
    # Divide by 2 because mask counts both (i,j) and (j,i) for the squared sum
    loss = torch.sum((overlap * mask)**2) / 2.0

    # Normalize by number of potential pairs
    num_pairs = num_nodes * (num_nodes - 1) / 2
    if num_pairs > 0:
        loss = loss / num_pairs
    return loss

def count_overlaps_toroidal(positions, radius, box_size):
    """Counts overlapping sphere pairs using TOROIDAL distance."""
    num_nodes = positions.shape[0]
    if num_nodes < 2:
        return 0

    dist_matrix = pairwise_toroidal_distance(positions, positions, box_size)
    # Check overlap condition (strictly less than 2*r)
    overlaps = dist_matrix < (2 * radius - 1e-6) # Use epsilon for floating point safety
    overlaps.fill_diagonal_(False) # Ignore self - inplace OK here as no grads needed
    # Count pairs (divide by 2 since matrix is symmetric)
    num_overlapping_pairs = torch.sum(overlaps).item() / 2
    return int(num_overlapping_pairs)

# --- GNN Model (No changes needed here) ---
class SphereRepulsionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim) # Output is displacement vector
        self.relu = nn.ReLU()
        # Optional: Add LayerNorm for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        print("x: ", x)
        h = self.relu(self.conv1(x, edge_index))
        print("h: ", h)
        h = self.ln1(h) # Optional LayerNorm
        print("h: ", h)
        h = self.relu(self.conv2(h, edge_index))
        print("h: ", h)
        h = self.ln2(h) # Optional LayerNorm
        print("h: ", h)
        displacement = self.conv3(h, edge_index)
        return displacement

# --- Training Function (Updated) ---

def train_model(model, initial_positions, radius, box_size, interaction_threshold,
                epochs=100, lr=0.005, max_displacement=MAX_DISPLACEMENT_PER_STEP, clip_norm=GRADIENT_CLIP_NORM):
    """Trains the GNN model with toroidal space and stability controls."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    positions_over_time = [initial_positions.clone().detach()] # Store initial state
    initial_overlap_count = count_overlaps_toroidal(initial_positions, radius, box_size)
    initial_loss = calculate_overlap_loss_toroidal(initial_positions, radius, box_size).item()
    print(f"Box Size: {box_size:.2f}")
    print(f"Initial state: {initial_overlap_count} overlaps, Loss: {initial_loss:.4f}")

    current_positions = initial_positions.clone().detach().requires_grad_(False)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Ensure positions stay within the box bounds *before* graph/prediction
        # This is defensive programming, the modulo later should handle it.
        # current_positions = current_positions % box_size

        # --- Forward Pass ---
        # 1. Create graph based on *current* TOROIDAL positions
        edge_index = create_graph_toroidal(current_positions, interaction_threshold, box_size)

        # 2. Get predicted displacements from the GNN
        pred_displacements_raw = model(current_positions, edge_index)
        print("pred_displacements_raw: ", pred_displacements_raw)

        # --- Displacement Scaling/Clamping ---
        with torch.no_grad(): # Don't track gradients for the clamping itself
            disp_norm = torch.norm(pred_displacements_raw, dim=1, keepdim=True)
            # Scale down displacements that are too large
            scale = torch.clamp(max_displacement / (disp_norm + 1e-9), max=1.0) # Add epsilon for stability
            pred_displacements = pred_displacements_raw * scale
            # Detach here as clamping shouldn't influence GNN gradients directly
            pred_displacements = pred_displacements.detach()

        # Create a version attached to the graph for loss calculation
        # We want gradients to flow back through the *magnitude* of the raw displacement
        # but use the *clamped* displacement for calculating the next position.
        # Let's rethink: Calculate loss based on applying the SCALED displacement.
        # The scaling factor itself depends on the raw output, allowing gradients.
        disp_norm_raw = torch.norm(pred_displacements_raw, dim=1, keepdim=True)
        scale_factor = torch.clamp(max_displacement / (disp_norm_raw + 1e-9), max=1.0)
        # This scaled displacement still has gradient history from pred_displacements_raw
        pred_displacements_for_loss = pred_displacements_raw * scale_factor

        # 3. Calculate *predicted* next positions (using scaled displacements)
        predicted_next_positions = current_positions + pred_displacements_for_loss
        # Apply TOROIDAL boundary condition to predicted positions
        predicted_next_positions = predicted_next_positions % box_size

        # --- Loss Calculation ---
        # Calculate overlap loss based on the PREDICTED TOROIDAL positions
        print("predicted_next_positions: ", predicted_next_positions)
        loss = calculate_overlap_loss_toroidal(predicted_next_positions, radius, box_size)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at epoch {epoch+1}! Stopping training.")
            print("Consider lowering learning rate, increasing epsilon, or checking model stability.")
            # Optional: Print intermediate values
            # print("Current Positions:", current_positions)
            # print("Raw Displacements:", pred_displacements_raw)
            # print("Scaled Displacements:", pred_displacements_for_loss)
            # print("Predicted Next Positions:", predicted_next_positions)
            break

        # --- Backward Pass & Optimization ---
        loss.backward()

        # --- Gradient Clipping ---
        if clip_norm is not None:
             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()

        for param in model.parameters():
            print("param: ", param)

        # --- Update Positions (for the *next* iteration's graph) ---
        # Use the clamped/scaled displacements (detached) for the actual update
        with torch.no_grad():
            current_positions = current_positions + pred_displacements # Use scaled, detached version
            # Apply TOROIDAL boundary condition AFTER update
            current_positions = current_positions % box_size

        # --- Logging ---
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_overlaps = count_overlaps_toroidal(current_positions, radius, box_size)
            avg_disp_magnitude = torch.mean(torch.norm(pred_displacements, dim=1)).item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Overlaps: {current_overlaps}, Avg Disp: {avg_disp_magnitude:.4f}")

        # Store positions every few epochs for visualization
        if (epoch + 1) % (epochs // 10) == 0 or epoch == (epochs - 1) :
             positions_over_time.append(current_positions.clone().detach())


    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds")
    final_overlaps = count_overlaps_toroidal(current_positions, radius, box_size)
    final_loss = calculate_overlap_loss_toroidal(current_positions, radius, box_size).item()
    print(f"Final state: {final_overlaps} overlaps, Loss: {final_loss:.4f}")

    return losses, current_positions.detach(), positions_over_time

# --- Plotting Function (Updated for Torus) ---

def plot_loss(losses):
    """Plots the training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Overlap Loss")
    plt.grid(True)
    plt.yscale('log') # Log scale often helpful for loss
    plt.show()

def plot_spheres_toroidal(positions, radius, box_size, title="Sphere Positions", ax=None):
    """Plots spheres in TOROIDAL space (assumes 2D for visualization)."""
    if positions.shape[1] != 2:
        print("Plotting only supported for 2D spheres.")
        if ax:
             ax.text(0.5, 0.5, '3D+ data (cannot plot circles)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_title(title)
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.cla() # Clear previous plot
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    # Set limits strictly to the box size
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    # Draw box boundary
    ax.plot([0, box_size, box_size, 0, 0], [0, 0, box_size, box_size, 0], 'k--', alpha=0.7)
    # ax.grid(True) # Grid might be distracting with torus

    overlaps = count_overlaps_toroidal(positions, radius, box_size) # Use toroidal count

    # Check for overlaps and plot spheres
    dist_matrix = pairwise_toroidal_distance(positions, positions, box_size)

    for i in range(positions.shape[0]):
        is_overlapping = False
        for j in range(i + 1, positions.shape[0]):
            if dist_matrix[i, j] < 2 * radius - 1e-6:
                 is_overlapping = True
                 # Optionally draw line between overlapping centers (tricky in torus)
                 # Simple line within the box:
                 # ax.plot([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]], 'r--', alpha=0.3)
                 # TODO: Draw shortest toroidal line if needed (more complex)

        color = 'red' if is_overlapping else 'blue'
        # Ensure position is plotted within main box [0, box_size)
        plot_pos = positions[i] % box_size
        circle = plt.Circle((plot_pos[0], plot_pos[1]), radius, color=color, fill=True, alpha=0.6)
        ax.add_patch(circle)
        # Add sphere index number
        ax.text(plot_pos[0], plot_pos[1], str(i), ha='center', va='center', fontsize=8, color='white')

    ax.set_title(f"{title} ({overlaps} overlaps)")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")


# --- Inference Function (Updated) ---

def inference(model, initial_positions, radius, box_size, interaction_threshold,
              steps=10, max_displacement=MAX_DISPLACEMENT_PER_STEP):
    """Applies the trained model iteratively in TOROIDAL space."""
    model.eval()  # Set model to evaluation mode
    current_positions = initial_positions.clone().detach() % box_size # Ensure start in box
    positions_history = [current_positions.clone()]
    print(f"\n--- Inference Start (Toroidal, Max Disp: {max_displacement:.3f}) ---")
    initial_overlaps = count_overlaps_toroidal(current_positions, radius, box_size)
    print(f"Step 0 (Initial): {initial_overlaps} overlaps")

    with torch.no_grad(): # No need for gradients during inference
        for step in range(steps):
            # Create graph for current TOROIDAL state
            edge_index = create_graph_toroidal(current_positions, interaction_threshold, box_size)
            # Predict displacements
            displacements_raw = model(current_positions, edge_index)

            # --- Displacement Scaling/Clamping ---
            disp_norm = torch.norm(displacements_raw, dim=1, keepdim=True)
            scale = torch.clamp(max_displacement / (disp_norm + 1e-9), max=1.0)
            displacements = displacements_raw * scale

            # Update positions
            current_positions = current_positions + displacements
            # Apply TOROIDAL boundary condition
            current_positions = current_positions % box_size

            positions_history.append(current_positions.clone())
            overlaps = count_overlaps_toroidal(current_positions, radius, box_size)
            avg_disp = torch.mean(torch.norm(displacements, dim=1)).item()
            print(f"Step {step+1}: {overlaps} overlaps, Avg Disp: {avg_disp:.4f}")

    print(f"--- Inference End ---")
    return current_positions, positions_history


# --- Main Execution (Updated) ---

if __name__ == "__main__":
    # --- Setup ---
    torch.manual_seed(43) # Changed seed slightly
    np.random.seed(43)

    hidden_dim = 64
    output_dim = DIM
    learning_rate = 0.00002 # Potentially lower LR for more stability
    training_epochs = 300 # Might need more epochs with smaller steps
    interaction_threshold = RADIUS * INTERACTION_THRESHOLD_MULTIPLIER

    # --- Data Generation ---
    initial_sphere_positions = generate_spheres(NUM_SPHERES, DIM, RADIUS, BOX_SIZE)

    # --- Model Initialization ---
    model = SphereRepulsionGNN(input_dim=DIM, hidden_dim=hidden_dim, output_dim=output_dim)
    print("Model Architecture:")
    print(model)
    print(f"\nConfiguration: DIM={DIM}, N_Spheres={NUM_SPHERES}, Radius={RADIUS:.2f}, BoxSize={BOX_SIZE:.2f}")
    print(f"Interaction Threshold: {interaction_threshold:.2f}, Max Disp: {MAX_DISPLACEMENT_PER_STEP:.3f}, Grad Clip: {GRADIENT_CLIP_NORM}")
    print(f"LR: {learning_rate}, Epochs: {training_epochs}\n")

    # --- Plot Initial State ---
    if DIM == 2:
         fig_initial, ax_initial = plt.subplots(figsize=(8, 8))
         plot_spheres_toroidal(initial_sphere_positions, RADIUS, BOX_SIZE, title="Initial Sphere Positions (Toroidal)", ax=ax_initial)
         plt.tight_layout()
         plt.show()
    initial_overlaps_main = count_overlaps_toroidal(initial_sphere_positions, radius=RADIUS, box_size=BOX_SIZE)
    print(f"Initial Overlaps (Main): {initial_overlaps_main}")


    # --- Training ---
    print("\n--- Starting Training ---")
    losses, final_positions_train, train_pos_history = train_model(
        model,
        initial_sphere_positions,
        RADIUS,
        BOX_SIZE,
        interaction_threshold,
        epochs=training_epochs,
        lr=learning_rate,
        max_displacement=MAX_DISPLACEMENT_PER_STEP,
        clip_norm=GRADIENT_CLIP_NORM
    )

    # --- Plotting Results ---
    plot_loss(losses)

    if DIM == 2:
        fig_final_train, ax_final_train = plt.subplots(figsize=(8, 8))
        plot_spheres_toroidal(final_positions_train, RADIUS, BOX_SIZE, title="Final Positions (After Training, Toroidal)", ax=ax_final_train)
        plt.tight_layout()
        plt.show()

        # Optional: Animate training process (requires celluloid or similar)
        # ... (animation code would need plot_spheres_toroidal)


    # --- Inference ---
    # Test on the original configuration or a new one
    test_positions = initial_sphere_positions
    # test_positions = generate_spheres(NUM_SPHERES, DIM, RADIUS, BOX_SIZE) # Optional: new test set

    final_positions_inference, inference_history = inference(
        model,
        test_positions,
        RADIUS,
        BOX_SIZE,
        interaction_threshold,
        steps=25 # Increase inference steps maybe
    )

    # --- Plot Inference Results ---
    if DIM == 2:
        fig_inf_initial, ax_inf_initial = plt.subplots(figsize=(8, 8))
        plot_spheres_toroidal(test_positions, RADIUS, BOX_SIZE, title="Inference - Initial State (Toroidal)", ax=ax_inf_initial)
        plt.tight_layout()
        plt.show()

        fig_inf_final, ax_inf_final = plt.subplots(figsize=(8, 8))
        plot_spheres_toroidal(final_positions_inference, RADIUS, BOX_SIZE, title=f"Inference - Final State ({len(inference_history)-1} steps, Toroidal)", ax=ax_inf_final)
        plt.tight_layout()
        plt.show()

        # Optional: Animate inference process
        # ... (animation code would need plot_spheres_toroidal)