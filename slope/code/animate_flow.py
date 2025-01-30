import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from utils import load_parameters, load_dataset, truncate_time_series, get_h

# Parameters for arrow density and scaling
arrow_density = 5  # Number of bins for averaging
arrow_scale =   1  # Scaling factor for arrow length

# Load the NetCDF file using xarray
params = load_parameters()
data = load_dataset(params["filepath"], params["name"])
data = truncate_time_series(data)

# Extract variables
x = data['xC'].values
y = data['yC'].values
time = data['time'].values
u = data['u'].values  # Shape: (time, y, x)
v = data['v'].values  # Shape: (time, y, x)
omega = data['omega'].values  # Shape: (time, y, x)
h = data['h'].values
eta = h - get_h(params)

print(np.max(np.abs(u)))
print(np.max(np.abs(v)))

# Interpolate velocities to cell centers
# u: average over y-axis, ignoring the extra row
u_centered = 0.5 * (u[:, :, :-1] + u[:, :, 1:])

# v: average over x-axis, ignoring the extra column
v_centered = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
v_centered = np.concatenate((v_centered, 0.5 * (v[:, -1:, :] + v[:, :1, :])), axis=1)

# Calculate binned averages
def bin_average(data, bins_y, bins_x):
    # Ensure the data can be evenly reshaped
    ny, nx = data.shape[1], data.shape[2]  # Original spatial dimensions
    data = data[:, :bins_y * arrow_density, :bins_x * arrow_density]  # Trim to fit bins
    binned = data.reshape(data.shape[0], bins_y, arrow_density, bins_x, arrow_density)
    binned = binned.mean(axis=2).mean(axis=3)  # Average over bins
    return binned

bins_y = len(y) // arrow_density
bins_x = len(x) // arrow_density
x_binned = x[:bins_x * arrow_density].reshape(bins_x, arrow_density).mean(axis=1)
y_binned = y[:bins_y * arrow_density].reshape(bins_y, arrow_density).mean(axis=1)

u_binned = bin_average(u_centered, bins_y, bins_x)
v_binned = bin_average(v_centered, bins_y, bins_x)

# Calculate vmin and vmax using percentiles to exclude outliers
omega_vmin, omega_vmax = np.percentile(omega, [1, 99])
omega_norm = TwoSlopeNorm(vmin=omega_vmin, vcenter=0, vmax=omega_vmax)

h_vmin, h_vmax = np.min(eta[1:]), np.max(eta[1:])


# Set up the figure and axes
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(2, 3, height_ratios=[30,1])  # Adjust ratios as needed
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
cax2 = fig.add_subplot(gs[1, 1])
cax3 = fig.add_subplot(gs[1, 2])

# Velocity field plot
quiver = ax1.quiver(x_binned, y_binned, u_binned[0], v_binned[0], scale=1 / arrow_scale)
ax1.set_title("Velocity Field")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_aspect('equal') 

# Omega plot
omega_plot = ax2.imshow(omega[0], extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', aspect='equal', cmap='coolwarm', norm=omega_norm)
ax2.set_title("Omega")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
fig.colorbar(omega_plot, cax=cax2, orientation='horizontal', pad=0.2, label='Omega Value', aspect=20)  # Smaller colorbar


# SSH plot
ssh_plot = ax3.imshow(eta[0], extent=[x.min(), x.max(), y.min(), y.max()],
                      origin='lower', aspect='equal', cmap='cividis', vmin=h_vmin, vmax=h_vmax)
ax3.set_title("Sea Surface Height (SSH)")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
fig.colorbar(ssh_plot, cax=cax3, orientation='horizontal', pad=0.2, label='SSH Value', aspect=20)


fig.tight_layout()

# Update function for animation
def update(frame):
    days = time[frame].astype('timedelta64[D]').astype(int) 
    quiver.set_UVC(u_binned[frame], v_binned[frame])
    omega_plot.set_data(omega[frame])
    ssh_plot.set_data(eta[frame])  # Update SSH frame
    ax1.set_title(f"Velocity Field (time={days} days)")
    ax2.set_title(f"Omega (time={days} days)")
    ax3.set_title(f"Sea Surface Height (time={days} days)")
    return quiver, omega_plot

# Create animation
anim = FuncAnimation(fig, update, frames=len(time), interval=200)

# Save or show the animation
anim.save(f"slope/animations/{params["name"]}.mp4", writer="ffmpeg", fps=10)
plt.show()