import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import load_parameters

params = load_parameters()
forcingfile = params["forcing_file"]
config = params["name"]

# Load the dataset
ds = xr.open_dataset(forcingfile)

# Extract variables
x = ds.x.values/1e3
y = ds.y.values/1e3
time = ds.time.values
tau_x = ds.forcing_x.values
tau_y = ds.forcing_y.values

# Subsampling for clarity
step = 4  # Plot every 4th vector
x_sub = x[::step]
y_sub = y[::step]
X, Y = np.meshgrid(x_sub, y_sub, indexing="ij")

# Compute global max magnitude for consistent scaling
magnitude = np.sqrt(tau_x**2 + tau_y**2)
global_max_magnitude = np.max(magnitude)
scale_factor = 0.1 / global_max_magnitude if global_max_magnitude > 0 else 1  # Prevent division by zero

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim([x.min(), x.max()])
ax.set_ylim([y.min(), y.max()])
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Forcing Field Animation")

# Initialize quiver plot
quiver = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), scale=1)

# Update function for animation
def update(frame):
    U = tau_x[frame, ::step, ::step] * scale_factor
    V = tau_y[frame, ::step, ::step] * scale_factor
    quiver.set_UVC(U, V)
    ax.set_title(f"Forcing Field - Time: {time[frame]/(60*60*24):.0f} days")
    return quiver,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time), interval=200, blit=False)

# Save or show animation
ani.save(f"slope/animations/forcings/{config}_forcing_animation.mp4", writer="ffmpeg", dpi=150)  # Save as MP4
plt.show()  # Show the animation