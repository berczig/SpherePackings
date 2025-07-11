import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import SP

from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import make_interp_spline

def animate_scatter(black_data, red_data, interval=200, filename="anim", show=True):
    """
    Animate scatter plots with black and red points.
    :param black_data: NumPy array of shape (timesteps, num_points, 2) for black scatter points.
    :param red_data: NumPy array of shape (timesteps, num_points, 2) for red scatter points.
    :param interval: Time interval between frames in milliseconds.
    """
    timesteps, _, _ = black_data.shape
    
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(black_data[..., 0]) - 1, np.max(black_data[..., 0]) + 1)
    ax.set_ylim(np.min(black_data[..., 1]) - 1, np.max(black_data[..., 1]) + 1)
    
    black_scatter = ax.scatter([], [], color='black')
    red_scatter = ax.scatter([], [], color='red')
    
    def update(frame):
        black_scatter.set_offsets(black_data[frame])
        red_scatter.set_offsets(red_data[frame])
        return black_scatter, red_scatter
    
    ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=interval, blit=True)
    if show:
        plt.show()
    ani.save(os.path.join(SP.reffolder, "output/{}.mp4".format(filename)))

def animate_histogram(frames, filename='scatter_line_animation.gif', fps=10, title = "Animation"):
    """
    Saves an animation of (x, y) data points for each frame. Each frame shows:
    - Scatter plot of the data points
    - Line connecting the points
    - Fixed axes to fit all data
    
    Parameters:
    - frames: List of (x, y) pairs, where x and y are arrays or lists of the same length
    - filename: Name of the output animation file (supports .gif or .mp4)
    - fps: Frames per second
    """
    # Flatten all x and y values to find global min/max
    all_x = np.concatenate([np.array(f[0]) for f in frames])
    all_y = np.concatenate([np.array(f[1]) for f in frames])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    # Add padding
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    fig, ax = plt.subplots()
    fig.suptitle(title)
    scatter = ax.plot([], [], 'bo')[0]
    line = ax.plot([], [], 'b-', alpha=0.5)[0]
    
    def init():
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        return scatter, line

    def update(frame):
        x, y = np.array(frame[0]), np.array(frame[1])
        scatter.set_data(x, y)
        line.set_data(x, y)
        return scatter, line

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Animation saved to {filename}")

# Example usage:
def main():
    # Create example frames with noisy sine wave
    dates = np.linspace(0, 10, 100)
    values = 3*dates**2
    frames = [(dates, 3*dates**2), 
              (dates, 4*dates**2), 
              (dates, 3*dates**2.6)]
    animate_histogram(frames)
    """x = np.linspace(0, 2*np.pi, 100)
    frames = []
    for i in range(30):
        y = np.sin(x + i * 0.1) + np.random.normal(scale=0.1, size=len(x))
        frames.append((x, y))

    save_envelope_animation(frames, filename='sine_envelope.gif', fps=10)"""
