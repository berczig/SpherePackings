import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Overlap Energy (Soft Repulsion Potential)
def overlap_energy(x, r=1.0):
    d = torch.cdist(x, x, p=2)
    mask = torch.triu(torch.ones(d.shape), diagonal=1).bool()
    d = d[mask]
    loss = torch.relu(r - d) ** 2
    return torch.sum(loss)

# Neural Network to Learn the Score Function
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

    def forward(self, x):
        return self.net(x)

# Diffusion Model with EDM architecture
class EDMModel:
    def __init__(self, n_points, dim, r=1.0, epsilon=0.01, noise=0.1, lambda_energy=0.1):
        self.n_points = n_points
        self.dim = dim
        self.r = r
        self.epsilon = epsilon
        self.noise = noise
        self.lambda_energy = lambda_energy
        self.score_network = ScoreNetwork(dim)
        self.optimizer = torch.optim.Adam(self.score_network.parameters(), lr=1e-3)

    def langevin_step(self, x):
        with torch.no_grad():
            score = self.score_network(x)
            noise = torch.randn_like(x) * self.noise
            x = x + self.epsilon * score + noise
            return x

    def score_matching_loss(self, x, noise_level=0.1):
        noise = torch.randn_like(x) * noise_level
        x_noisy = x + noise
        score_pred = self.score_network(x_noisy)
        score_true = -noise / (noise_level ** 2)
        score_loss = torch.mean((score_pred - score_true) ** 2)
        energy_loss = overlap_energy(x_noisy, self.r)
        return score_loss + self.lambda_energy * energy_loss

    def train(self, steps=1000):
        x = torch.rand(self.n_points, self.dim) * 2 - 1
        for t in range(steps):
            self.optimizer.zero_grad()
            loss = self.score_matching_loss(x, noise_level=self.noise)
            loss.backward()
            self.optimizer.step()
            if t % 100 == 0:
                print(f"Step {t}, Loss: {loss.item()}")

    def diffuse(self, steps=1000):
        x = torch.rand(self.n_points, self.dim) * 2 - 1
        history = []
        for t in range(steps):
            x = self.langevin_step(x)
            if t % (steps // 100) == 0:
                history.append(x.clone().detach().numpy())
            self.noise *= 0.99  # Annealing
        return x, history

# Plotting Function
def plot_packings(history, r=1.0):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal', adjustable='box')
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
            circle = plt.Circle((x[j, 0], x[j, 1]), r, color='b', alpha=0.2)
            ax.add_artist(circle)
            circles.append(circle)

    ani = animation.FuncAnimation(fig, update, frames=len(history), init_func=init, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    model = EDMModel(n_points=50, dim=2, r=0.2, epsilon=0.01, noise=0.1, lambda_energy=0.1)
    print("Training Score Network...")
    model.train(steps=5000)
    print("Diffusing...")
    final, history = model.diffuse(steps=5000)
    plot_packings(history, r=0.2)
