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

#data = data.isel(time=slice(-25*8, None))
#data = data.isel(time=slice(None, 25*8))

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
eta = h - h[1]
eta[0] = eta[1]
etap = eta - np.mean(eta, axis=1)[:,np.newaxis,:]

x /= 1e3
y /= 1e3

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

h_vmin, h_vmax = np.min(etap[1:]), np.max(etap[1:])

# Compute speed for the full grid (not binned)
speed_full = np.sqrt(u_centered**2 + v_centered**2)

# Normalize the arrows to have equal length
u_norm = u_binned / (np.sqrt(u_binned**2 + v_binned**2) + 1e-10)
v_norm = v_binned / (np.sqrt(u_binned**2 + v_binned**2) + 1e-10)

# Compute vmin and vmax for speed colormap
speed_vmin, speed_vmax = np.percentile(speed_full, [1, 99])

# Set up the figure and axes (2x2 grid + space for colorbars)
fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

ax1, ax2 = axs[0]
ax3, ax4 = axs[1]

# Plot 1: Velocity magnitude with quiver
speed_plot = ax1.pcolormesh(x, y, speed_full[0], cmap='plasma', 
                            norm=Normalize(vmin=speed_vmin, vmax=speed_vmax))
quiver = ax1.quiver(x_binned, y_binned, u_norm[0], v_norm[0], scale=30)
ax1.contour(x, y, bath, colors='k', linewidths=0.5)
ax1.set_title("Velocity Field")
ax1.set_xlabel("x [km]")
ax1.set_ylabel("y [km]")
ax1.set_aspect('equal')
cbar1 = fig.colorbar(speed_plot, ax=ax1, orientation='vertical', label='Speed [m/s]')

# Plot 2: Vorticity
omega_plot = ax2.imshow(omega[0], extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', aspect='equal', cmap='coolwarm', 
                        vmin=omega_vmin, vmax=omega_vmax)
ax2.contour(x, y, bath, colors='k', linewidths=0.5)
ax2.set_title("Relative Vorticity")
ax2.set_xlabel("x [km]")
ax2.set_ylabel("y [km]")
cbar2 = fig.colorbar(omega_plot, ax=ax2, orientation='vertical', label='[s$^{-1}$]')

# Plot 3: Eta
eta_vmin, eta_vmax = np.percentile(eta[1:], [1, 99])
eta_plot = ax3.imshow(eta[0], extent=[x.min(), x.max(), y.min(), y.max()],
                      origin='lower', aspect='equal', cmap='viridis',
                      vmin=eta_vmin, vmax=eta_vmax)
ax3.contour(x, y, bath, colors='k', linewidths=0.5)
ax3.set_title("Sea Surface Height η")
ax3.set_xlabel("x [km]")
ax3.set_ylabel("y [km]")
cbar3 = fig.colorbar(eta_plot, ax=ax3, orientation='vertical', label='[m]')

# Plot 4: Eta perturbation
etap_plot = ax4.imshow(etap[0], extent=[x.min(), x.max(), y.min(), y.max()],
                       origin='lower', aspect='equal', cmap='cividis',
                       vmin=h_vmin, vmax=h_vmax)
ax4.contour(x, y, bath, colors='k', linewidths=0.5)
ax4.set_title("SSH Perturbation η'")
ax4.set_xlabel("x [km]")
ax4.set_ylabel("y [km]")
cbar4 = fig.colorbar(etap_plot, ax=ax4, orientation='vertical', label='[m]')


# Update function
def update(frame):
    days = time[frame].astype('timedelta64[D]').astype(int)

    speed_plot.set_array(speed_full[frame].ravel())
    quiver.set_UVC(u_norm[frame], v_norm[frame])
    omega_plot.set_data(omega[frame])
    eta_plot.set_data(eta[frame])
    etap_plot.set_data(etap[frame])

    ax1.set_title(f"Velocity Field (day {days})")
    ax2.set_title(f"Relative Vorticity (day {days})")
    ax3.set_title(f"Sea Surface Height η (day {days})")
    ax4.set_title(f"SSH Perturbation η' (day {days})")

    return speed_plot, quiver, omega_plot, eta_plot, etap_plot

# Create animation
anim = FuncAnimation(fig, update, frames=len(time), interval=200)

# Save or show
anim.save(f"slope/animations/flow/{params['name']}.mp4", writer="ffmpeg", fps=10)
plt.show()
