import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_scatter(black_data, red_data, interval=200):
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
    plt.show()
