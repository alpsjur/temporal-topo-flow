import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from utils import load_parameters, load_dataset, truncate_time_series, calculate_bathymetry

# Parameters for arrow density and scaling
arrow_density = 5  # Number of bins for averaging

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
X, Y = np.meshgrid(x, y)
bath = calculate_bathymetry(X, Y, params)
eta = h - bath

# Interpolate velocities to cell centers
# u: average over y-axis, ignoring the extra row
u_centered = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
#u_centered = np.concatenate((u_centered, 0.5 * (u[:, :, -1:] + u[:, :, :1])), axis=2)


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
#omega_norm = TwoSlopeNorm(vmin=omega_vmin, vcenter=0, vmax=omega_vmax)

h_vmin, h_vmax = np.min(eta[1:]), np.max(eta[1:])

# Compute speed for the full grid (not binned)
speed_full = np.sqrt(u_centered**2 + v_centered**2)

# Normalize the arrows to have equal length
u_norm = u_binned / (np.sqrt(u_binned**2 + v_binned**2) + 1e-10)
v_norm = v_binned / (np.sqrt(u_binned**2 + v_binned**2) + 1e-10)

# Compute vmin and vmax for speed colormap
speed_vmin, speed_vmax = np.percentile(speed_full, [1, 99])

# Set up the figure and axes
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(2, 3, height_ratios=[30, 1])  # Adjust ratios for colorbars
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
cax1 = fig.add_subplot(gs[1, 0])  # Colorbar for speed
cax2 = fig.add_subplot(gs[1, 1])  # Colorbar for omega
cax3 = fig.add_subplot(gs[1, 2])  # Colorbar for SSH

# Velocity field plot with full-resolution speed colormap
speed_plot = ax1.pcolormesh(x, y, speed_full[0], cmap='plasma', 
                            norm=Normalize(vmin=speed_vmin, vmax=speed_vmax))
quiver = ax1.quiver(x_binned, y_binned, u_norm[0], v_norm[0], scale=30)  
ax1.set_title("Velocity Field")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_aspect('equal')

# Omega plot
omega_plot = ax2.imshow(omega[0], extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', aspect='equal', cmap='coolwarm', vmin=omega_vmin, vmax=omega_vmax)
ax2.set_title("Relative vorticity")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
fig.colorbar(omega_plot, cax=cax2, orientation='horizontal', pad=0.2, label='[s-1]', aspect=20)

# SSH plot
ssh_plot = ax3.imshow(eta[0], extent=[x.min(), x.max(), y.min(), y.max()],
                      origin='lower', aspect='equal', cmap='cividis', vmin=h_vmin, vmax=h_vmax)
ax3.set_title("Sea Surface Height (SSH)")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
fig.colorbar(ssh_plot, cax=cax3, orientation='horizontal', pad=0.2, label='[m]', aspect=20)

# Add colorbar for speed
fig.colorbar(speed_plot, cax=cax1, orientation='horizontal', pad=0.2, label='[ms-1]', aspect=20)

fig.tight_layout()

# Update function for animation
def update(frame):
    days = time[frame].astype('timedelta64[D]').astype(int)
    speed_plot.set_array(speed_full[frame].ravel())  # Update full-resolution speed colormesh
    quiver.set_UVC(u_norm[frame], v_norm[frame])  # Update quiver arrows
    omega_plot.set_data(omega[frame])
    ssh_plot.set_data(eta[frame])  # Update SSH frame
    ax1.set_title(f"Velocity Field (time={days} days)")
    ax2.set_title(f"Relative vorticity (time={days} days)")
    ax3.set_title(f"Sea Surface Height (time={days} days)")
    return speed_plot, quiver, omega_plot, ssh_plot

# Create animation
anim = FuncAnimation(fig, update, frames=len(time), interval=200)

# Save or show the animation
anim.save(f"slope/animations/{params['name']}.mp4", writer="ffmpeg", fps=10)
plt.show()
