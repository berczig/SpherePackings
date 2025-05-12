import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_generation import get_data_loader  # <-- imported here
from SP import cfg  # <-- imported here
from tqdm import tqdm

# --- Strong Repulsion Force ---
def repulsion_force_strong(x, r=1.0, power=4):
    d = torch.cdist(x, x) + 1e-6
    mask = ~torch.eye(x.shape[1], dtype=bool, device=x.device)[None, :, :]  # [1, N, N]
    force_magnitude = ((r / d) ** power) * mask
    direction = (x.unsqueeze(2) - x.unsqueeze(1)) / d.unsqueeze(-1)  # [B, N, N, D]
    force = (force_magnitude.unsqueeze(-1) * direction).sum(dim=2)  # [B, N, D]
    return force

# --- Score Network ---
class ScoreNetwork(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):  # x: [B, N, D]
        B, N, D = x.shape
        x = x.view(B * N, D)  # Flatten to [B * N, D]
        x = self.net(x)       # Pass through the network
        x = x.view(B, N, D)   # Reshape back to [B, N, D]
        return x
# --- EDM Model with External DataLoader ---
class EDMModel:
    def __init__(self, n_points, dim, r=1.0, epsilon=0.01, noise=0.1, lambda_energy=10.0, device='cpu'):
        self.n_points = n_points
        self.dim = dim
        self.r = r
        self.epsilon = epsilon
        self.noise = noise
        self.lambda_energy = lambda_energy
        self.device = device

        self.score_network = ScoreNetwork(dim).to(device)
        self.optimizer = torch.optim.Adam(self.score_network.parameters(), lr=1e-3)

    def langevin_step(self, x):
        with torch.no_grad():
            score = self.score_network(x)
            repel = repulsion_force_strong(x, self.r, power=4)
            noise = torch.randn_like(x) * self.noise
            x = x + self.epsilon * (score + self.lambda_energy * repel) + noise
            return x

    def score_matching_loss(self, x, noise_level=0.1):
        noise = torch.randn_like(x) * noise_level
        x_noisy = x + noise
        score_pred = self.score_network(x_noisy)
        score_true = -noise / (noise_level ** 2)
        score_loss = torch.mean((score_pred - score_true) ** 2)
        return score_loss

    def train(self, dataloader, epochs=100):
        self.score_network.train()
        for epoch in range(epochs):
            for i, batch in enumerate(dataloader):
                x = batch.to(self.device)
                self.optimizer.zero_grad()
                loss = self.score_matching_loss(x, noise_level=self.noise)
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    print(f"Epoch {epoch} Batch {i} Loss: {loss.item():.6f}")

    def diffuse(self, steps=50, batch_size=1, box_size=2):
        x = torch.rand(batch_size, self.n_points, self.dim, device=self.device) * box_size - 1
        history = []
        for t in range(steps):
            x = self.langevin_step(x)
            if t % (steps // 100) == 0:
                history.append(x[0].clone().detach().cpu().numpy())  # Save first sample
            self.noise *= 0.995
        return x, history

# --- 2D Plotting ---
def plot_packings(history, r=1.0, box_size=0.1):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-box_size/2, box_size/2)
    ax.set_ylim(-box_size/2, box_size/2)
    ax.set_aspect('equal')
    ax.set_title("EDM Sphere Packings")

    scat = ax.scatter([], [], s=100, alpha=0.7)
    circles = []

    def init():
        scat.set_offsets(np.empty((0, 2)))
        for circle in circles:
            circle.remove()
        circles.clear()
        return scat,

    def update(frame):
        x = history[frame]
        scat.set_offsets(x)
        for circle in circles:
            circle.remove()
        circles.clear()
        for j in range(len(x)):
            circle = plt.Circle((x[j, 0], x[j, 1]), r, color='b', alpha=0.3)
            ax.add_artist(circle)
            circles.append(circle)

    ani = animation.FuncAnimation(fig, update, frames=len(history), init_func=init, interval=50, blit=False)
    plt.show()

# --- Main ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = cfg.getint("diffusion_model", "batch_size")
    dimension = cfg.getint("ppp_sample_generation", "dimension")
    radius = cfg.getfloat("ppp_sample_generation", "sphere_radius")
    num_spheres = cfg.getint("ppp_sample_generation", "num_sphere_per_sample")
    box_size = cfg.getfloat("ppp_sample_generation", "box_size")
    


    model = EDMModel(n_points=num_spheres, dim=dimension, r=radius, device=device)

    dataloader = get_data_loader(
        batch_size=batch_size,
        dataset_path=cfg.get("ppp_sample_generation", 'train_dataset_path'),
        )
    # Rearrange data from [B, D, N] to [B, N, D]
    dataloader.dataset.data = dataloader.dataset.data.permute(0, 2, 1)
    print(f"Data shape: {dataloader.dataset.data.shape}")

    # Train model
    print("Training Score Network...")
    model.train(dataloader, epochs=10)

    # Sample new configuration
    print("Sampling with Langevin dynamics...")
    final, history = model.diffuse(steps=100, batch_size=1, box_size=box_size)
    plot_packings(history, r=radius, box_size=box_size)
